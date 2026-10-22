#include "common/runtime/tp4_nvls_workspace.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

#include "attn/kernel/include/m3_decode_output_projection_handoff.hpp"
#include "common/kernel/include/tp4_mailbox_layout.hpp"
#include "common/kernel/include/tp4_nvls_layout.hpp"

namespace mega_atten::m3_decode {
namespace {

static_assert(tp4_nvls::kTokenCount == mega_atten::m3_decode::output_projection::kProblemN);
static_assert(tp4_nvls::kHiddenSize == mega_atten::m3_decode::output_projection::kProblemM);

void check_driver(CUresult status, char const *operation) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  char const *name = nullptr;
  char const *text = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &text);
  TORCH_CHECK(false, operation, " failed: ", name ? name : "unknown", " (", text ? text : "unknown",
              ")");
}

void check_runtime(cudaError_t status, char const *operation) {
  TORCH_CHECK(status == cudaSuccess, operation, " failed: ", cudaGetErrorString(status));
}

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

} // namespace

Tp4NvlsWorkspace::Tp4NvlsWorkspace(int device, size_t requested_size, int world_size,
                                   bool create_multicast)
    : device_(device), world_size_(world_size) {
  TORCH_CHECK(world_size_ == 4, "TP4 NVLS workspace requires world_size=4");
  TORCH_CHECK(requested_size >= tp4_nvls::kMinimumWorkspaceBytes,
              "TP4 NVLS workspace is too small: requested ", requested_size, ", minimum ",
              tp4_nvls::kMinimumWorkspaceBytes);

  try {
    c10::cuda::CUDAGuard guard(device_);
    check_driver(cuInit(0), "cuInit");
    CUcontext context = nullptr;
    check_driver(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
    TORCH_CHECK(context != nullptr, "CUDA context is not initialized");
    check_driver(cuDeviceGet(&cu_device_, device_), "cuDeviceGet");

    int multicast_supported = 0;
    check_driver(cuDeviceGetAttribute(&multicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
                                      cu_device_),
                 "cuDeviceGetAttribute(MULTICAST_SUPPORTED)");
    TORCH_CHECK(multicast_supported != 0, "CUDA multicast is not supported on device ", device_);

    CUmemAllocationProp allocation_prop{};
    allocation_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocation_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocation_prop.location.id = device_;
    allocation_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    allocation_prop.allocFlags.gpuDirectRDMACapable = 1;

    size_t unicast_granularity = 0;
    check_driver(cuMemGetAllocationGranularity(&unicast_granularity, &allocation_prop,
                                               CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                 "cuMemGetAllocationGranularity");

    multicast_prop_.numDevices = world_size_;
    multicast_prop_.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    multicast_prop_.flags = 0;
    multicast_prop_.size =
        align_up(std::max(requested_size, tp4_mailbox::kWorkspaceBytes), unicast_granularity);
    size_t multicast_granularity = 0;
    check_driver(cuMulticastGetGranularity(&multicast_granularity, &multicast_prop_,
                                           CU_MULTICAST_GRANULARITY_MINIMUM),
                 "cuMulticastGetGranularity");

    size_ = align_up(multicast_prop_.size, multicast_granularity);
    size_ = align_up(size_, unicast_granularity);
    multicast_prop_.size = size_;

    check_driver(cuMemCreate(&unicast_handle_, size_, &allocation_prop, 0), "cuMemCreate");
    check_driver(cuMemAddressReserve(&unicast_va_, size_, unicast_granularity, 0, 0),
                 "cuMemAddressReserve(unicast)");
    check_driver(cuMemMap(unicast_va_, size_, 0, unicast_handle_, 0), "cuMemMap(unicast)");
    set_access(unicast_va_);

    if (create_multicast) {
      check_driver(cuMulticastCreate(&multicast_handle_, &multicast_prop_), "cuMulticastCreate");
      owns_created_multicast_ = true;
    }
  } catch (...) {
    release_resources();
    throw;
  }
}

Tp4NvlsWorkspace::~Tp4NvlsWorkspace() { release_resources(); }

int Tp4NvlsWorkspace::export_multicast_fd() const {
  TORCH_CHECK(!closed_ && multicast_handle_ != 0 && owns_created_multicast_,
              "only the live multicast creator can export its handle");
  int fd = -1;
  check_driver(cuMemExportToShareableHandle(&fd, multicast_handle_,
                                            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0),
               "cuMemExportToShareableHandle(multicast)");
  return fd;
}

void Tp4NvlsWorkspace::import_multicast_fd(int fd) {
  TORCH_CHECK(!closed_ && multicast_handle_ == 0, "multicast handle has already been initialized");
  TORCH_CHECK(fd >= 0, "invalid multicast file descriptor");
  check_driver(cuMemImportFromShareableHandle(&multicast_handle_,
                                              reinterpret_cast<void *>(static_cast<uintptr_t>(fd)),
                                              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
               "cuMemImportFromShareableHandle(multicast)");
}

void Tp4NvlsWorkspace::add_device() {
  TORCH_CHECK(!closed_ && multicast_handle_ != 0, "multicast handle is not initialized");
  TORCH_CHECK(!device_added_, "device has already been added");
  check_driver(cuMulticastAddDevice(multicast_handle_, cu_device_), "cuMulticastAddDevice");
  device_added_ = true;
}

void Tp4NvlsWorkspace::map_and_bind() {
  TORCH_CHECK(!closed_ && device_added_, "add_device must run before map_and_bind");
  TORCH_CHECK(!bound_, "workspace has already been mapped and bound");
  c10::cuda::CUDAGuard guard(device_);
  check_driver(cuMemAddressReserve(&multicast_va_, size_, 0, 0, 0),
               "cuMemAddressReserve(multicast)");
  try {
    check_driver(cuMemMap(multicast_va_, size_, 0, multicast_handle_, 0), "cuMemMap(multicast)");
    set_access(multicast_va_);
    check_driver(cuMulticastBindMem(multicast_handle_, 0, unicast_handle_, 0, size_, 0),
                 "cuMulticastBindMem");
    bound_ = true;
  } catch (...) {
    if (multicast_va_ != 0) {
      cuMemUnmap(multicast_va_, size_);
      cuMemAddressFree(multicast_va_, size_);
      multicast_va_ = 0;
    }
    throw;
  }
}

void Tp4NvlsWorkspace::zero() {
  TORCH_CHECK(ready(), "workspace must be bound before zero initialization");
  c10::cuda::CUDAGuard guard(device_);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_).stream();
  check_runtime(cudaMemsetAsync(reinterpret_cast<void *>(unicast_va_), 0, size_, stream),
                "cudaMemsetAsync");
  check_driver(cuMemsetD16Async(unicast_va_ + tp4_mailbox::kPayloadOffset, tp4_mailbox::kEmpty,
                                tp4_mailbox::kPayloadBytes / sizeof(unsigned short), stream),
               "cuMemsetD16Async");
}

at::Tensor Tp4NvlsWorkspace::next_hidden() const {
  TORCH_CHECK(ready(), "workspace must be bound before creating next_hidden");
  auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, device_);
  void *data = reinterpret_cast<void *>(unicast_va_ + tp4_nvls::kNextHiddenOffset);
  return torch::from_blob(
      data, {tp4_nvls::kTokenCount, tp4_nvls::kHiddenSize}, [](void *) {}, options);
}

void Tp4NvlsWorkspace::close() {
  TORCH_CHECK(!closed_, "TP4 NVLS workspace is already closed");
  release_resources();
}

void Tp4NvlsWorkspace::set_access(CUdeviceptr address) {
  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device_;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  check_driver(cuMemSetAccess(address, size_, &access, 1), "cuMemSetAccess");
}

void Tp4NvlsWorkspace::release_resources() noexcept {
  if (closed_) {
    return;
  }
  closed_ = true;
  int previous_device = -1;
  cudaGetDevice(&previous_device);
  cudaSetDevice(device_);
  if (bound_) {
    cuMulticastUnbind(multicast_handle_, cu_device_, 0, size_);
    bound_ = false;
  }
  if (multicast_va_ != 0) {
    cuMemUnmap(multicast_va_, size_);
    cuMemAddressFree(multicast_va_, size_);
    multicast_va_ = 0;
  }
  if (multicast_handle_ != 0) {
    cuMemRelease(multicast_handle_);
    multicast_handle_ = 0;
  }
  if (unicast_va_ != 0) {
    cuMemUnmap(unicast_va_, size_);
    cuMemAddressFree(unicast_va_, size_);
    unicast_va_ = 0;
  }
  if (unicast_handle_ != 0) {
    cuMemRelease(unicast_handle_);
    unicast_handle_ = 0;
  }
  if (previous_device >= 0 && previous_device != device_) {
    cudaSetDevice(previous_device);
  }
}

} // namespace mega_atten::m3_decode
