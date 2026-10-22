#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "src/moe_token_parallel.hpp"

namespace py = pybind11;

namespace mega_moe::token_parallel {
namespace {

void check_cuda(cudaError_t status, char const *operation) {
  TORCH_CHECK(status == cudaSuccess, operation, ": ", cudaGetErrorString(status));
}

void check_not_capturing(int device) {
  cudaStreamCaptureStatus status;
  check_cuda(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(device).stream(), &status),
             "query stream capture");
  TORCH_CHECK(status == cudaStreamCaptureStatusNone,
              "token-parallel MoE resources must be prepared before CUDA Graph capture");
}

void check_current_device(int device) {
  int current = -1;
  check_cuda(cudaGetDevice(&current), "query current CUDA device");
  TORCH_CHECK(current == device, "select the workspace CUDA device first");
}

struct TensorContract {
  char const *name;
  at::ScalarType dtype;
  std::vector<int64_t> shape;
};

// Order of the Plan constructor's tensor arguments after the workspace.
std::vector<TensorContract> const kTensors = {
    {"hidden", at::kBFloat16, {T, H}},
    {"residual", at::kBFloat16, {T, H}},
    {"output", at::kBFloat16, {T, H}},
    {"router_weight", at::kFloat, {E, H}},
    {"router_correction_bias", at::kFloat, {E}},
    {"shared_fc1_weight", at::kFloat8_e4m3fn, {2 * I, H}},
    {"shared_fc1_weight_sf", at::kByte, {2 * I, H / 32}},
    {"shared_fc2_weight", at::kFloat8_e4m3fn, {H, I}},
    {"shared_fc2_weight_sf", at::kByte, {H, I / 32}},
    {"routed_fc1_weight", at::kByte, {E, 2 * I, H / 2}},
    {"routed_fc1_weight_sf", at::kFloat8_e4m3fn, {E, 2 * I, H / 16}},
    {"routed_fc2_weight", at::kByte, {E, H, I / 2}},
    {"routed_fc2_weight_sf", at::kFloat8_e4m3fn, {E, H, I / 16}},
    {"routed_fc1_alpha", at::kFloat, {E}},
    {"routed_fc2_alpha", at::kFloat, {E}},
};
constexpr size_t kOutput = 2;

void validate_tensor(at::Tensor const &tensor, TensorContract const &contract, int device) {
  TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.get_device() == device &&
                  tensor.scalar_type() == contract.dtype &&
                  tensor.sizes() == at::IntArrayRef(contract.shape) && tensor.is_contiguous() &&
                  !tensor.requires_grad() &&
                  reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
              contract.name, " must be a contiguous, 16-byte aligned ",
              c10::toString(contract.dtype), " tensor on the workspace device, without gradients");
}

bool overlaps(at::Tensor const &left, at::Tensor const &right) {
  uintptr_t const left_begin = reinterpret_cast<uintptr_t>(left.data_ptr());
  uintptr_t const right_begin = reinterpret_cast<uintptr_t>(right.data_ptr());
  return left_begin < right_begin + right.nbytes() && right_begin < left_begin + left.nbytes();
}

} // namespace

// One allocation keeps IPC ownership independent of the caching allocator. The
// arena is zeroed once; the kernel advances its generations on every call.
class Workspace : public std::enable_shared_from_this<Workspace> {
public:
  Workspace(int device, int rank) : device_(device), rank_(rank) {
    TORCH_CHECK(rank >= 0 && rank < WORLD, "token-parallel MoE requires a TP4 rank");
    c10::cuda::CUDAGuard guard(device_);
    check_not_capturing(device_);
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device_), "query CUDA device");
    TORCH_CHECK(properties.major == 10 && (properties.minor == 0 || properties.minor == 3),
                "token-parallel MoE requires SM100 or SM103");
    device_uuid_ = properties.uuid;
    int resident = 0;
    check_cuda(configure_kernel(device_, &resident), "configure token-parallel MoE kernel");
    TORCH_CHECK(resident >= kGrid, "the device cannot co-schedule the ", kGrid,
                " cooperative token-parallel MoE CTAs");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&allocation_), kArenaBytes),
               "allocate token-parallel MoE workspace");
    try {
      auto stream = at::cuda::getCurrentCUDAStream(device_).stream();
      check_cuda(cudaMemsetAsync(allocation_, 0, kArenaBytes, stream),
                 "initialize token-parallel MoE workspace");
      check_cuda(cudaStreamSynchronize(stream), "finish token-parallel MoE initialization");
      check_cuda(cudaIpcGetMemHandle(&handle_, allocation_), "export token-parallel MoE workspace");
      peers_[rank_] = allocation_;
    } catch (...) {
      cudaFree(allocation_);
      allocation_ = nullptr;
      throw;
    }
  }

  Workspace(Workspace const &) = delete;
  Workspace &operator=(Workspace const &) = delete;

  ~Workspace() {
    if (!allocation_)
      return;
    // Explicit collective close is required after graphs are destroyed.
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(device_);
    for (int rank = 0; rank < WORLD; ++rank)
      if (rank != rank_ && peers_[rank])
        cudaIpcCloseMemHandle(peers_[rank]);
    cudaFree(allocation_);
    if (previous >= 0)
      cudaSetDevice(previous);
  }

  int device() const { return device_; }
  int rank() const { return rank_; }
  bool ready() const { return allocation_ && connected_; }

  py::bytes device_uuid() const {
    return py::bytes(device_uuid_.bytes, sizeof(device_uuid_.bytes));
  }

  py::bytes ipc_handle() const {
    TORCH_CHECK(allocation_, "token-parallel MoE workspace is closed");
    return py::bytes(reinterpret_cast<char const *>(&handle_), sizeof(handle_));
  }

  void connect(std::vector<std::string> const &handles) {
    c10::cuda::CUDAGuard guard(device_);
    check_not_capturing(device_);
    TORCH_CHECK(allocation_ && !connected_, "workspace is closed or already connected");
    TORCH_CHECK(handles.size() == WORLD, "four peer IPC handles are required");
    for (int rank = 0; rank < WORLD; ++rank) {
      TORCH_CHECK(handles[rank].size() == sizeof(cudaIpcMemHandle_t), "invalid IPC handle");
      if (rank == rank_)
        TORCH_CHECK(std::memcmp(handles[rank].data(), &handle_, sizeof(handle_)) == 0,
                    "local IPC handle does not match the TP rank");
      for (int prior = 0; prior < rank; ++prior)
        TORCH_CHECK(handles[rank] != handles[prior], "TP ranks require distinct workspaces");
    }
    try {
      for (int rank = 0; rank < WORLD; ++rank) {
        if (rank == rank_)
          continue;
        cudaIpcMemHandle_t handle;
        std::memcpy(&handle, handles[rank].data(), sizeof(handle));
        // Lazy peer enablement applies only to the imported allocation.
        check_cuda(cudaIpcOpenMemHandle(reinterpret_cast<void **>(&peers_[rank]), handle,
                                        cudaIpcMemLazyEnablePeerAccess),
                   "import token-parallel MoE workspace");
      }
      connected_ = true;
    } catch (...) {
      for (int rank = 0; rank < WORLD; ++rank) {
        if (rank != rank_ && peers_[rank]) {
          cudaIpcCloseMemHandle(peers_[rank]);
          peers_[rank] = nullptr;
        }
      }
      throw;
    }
  }

  void bind(Args &args) const {
    TORCH_CHECK(ready(), "token-parallel MoE workspace is not connected");
    for (int rank = 0; rank < WORLD; ++rank)
      args.ready[rank] = reinterpret_cast<uint32_t *>(peers_[rank]);
    args.rank = rank_;
    args.shared_input = allocation_ + kSharedInput;
    args.shared_input_sf = allocation_ + kSharedInputScale;
    args.routed_input = allocation_ + kRoutedInput;
    args.routed_input_sf = allocation_ + kRoutedInputScale;
    args.shared_mid = allocation_ + kSharedMid;
    args.shared_mid_sf = allocation_ + kSharedMidScale;
    args.routed_mid = allocation_ + kRoutedMid;
    args.routed_mid_sf = allocation_ + kRoutedMidScale;
  }

  void close_imports() {
    if (!allocation_)
      return;
    c10::cuda::CUDAGuard guard(device_);
    check_not_capturing(device_);
    for (int rank = 0; rank < WORLD; ++rank) {
      if (rank == rank_ || !peers_[rank])
        continue;
      check_cuda(cudaIpcCloseMemHandle(peers_[rank]), "close token-parallel MoE IPC mapping");
      peers_[rank] = nullptr;
    }
    connected_ = false;
  }

  void close() {
    if (!allocation_)
      return;
    c10::cuda::CUDAGuard guard(device_);
    check_not_capturing(device_);
    for (int rank = 0; rank < WORLD; ++rank)
      TORCH_CHECK(rank == rank_ || !peers_[rank], "close peer IPC mappings before freeing owners");
    check_cuda(cudaFree(allocation_), "free token-parallel MoE workspace");
    allocation_ = nullptr;
    peers_[rank_] = nullptr;
    connected_ = false;
  }

private:
  int device_, rank_;
  uint8_t *allocation_ = nullptr;
  std::array<uint8_t *, WORLD> peers_{};
  cudaIpcMemHandle_t handle_{};
  cudaUUID_t device_uuid_{};
  bool connected_ = false;
};

// Binds one fixed set of tensors; tensor maps are encoded once here.
class Plan {
public:
  Plan(std::shared_ptr<Workspace> workspace, std::vector<at::Tensor> tensors)
      : workspace_(std::move(workspace)), tensors_(std::move(tensors)) {
    TORCH_CHECK(workspace_ && workspace_->ready(), "an open TP4 workspace is required");
    int const device = workspace_->device();
    check_current_device(device);
    check_not_capturing(device);
    TORCH_CHECK(tensors_.size() == kTensors.size(), "token-parallel MoE plan requires ",
                kTensors.size(), " tensors");
    for (size_t index = 0; index < tensors_.size(); ++index)
      validate_tensor(tensors_[index], kTensors[index], device);
    for (size_t index = 0; index < tensors_.size(); ++index)
      if (index != kOutput)
        TORCH_CHECK(!overlaps(tensors_[kOutput], tensors_[index]),
                    "output must be disjoint from every input");
    TORCH_CHECK(!overlaps(tensors_[0], tensors_[1]), "hidden and residual must be disjoint");
    Args &a = arguments_.a;
    a.post_attention_norm = pointer<uint16_t>(0);
    a.attention_residual = pointer<uint16_t>(1);
    a.out = pointer<uint16_t>(2);
    a.router_weight = pointer<float>(3);
    a.router_bias = pointer<float>(4);
    a.shared_fc1 = pointer<uint8_t>(5);
    a.shared_fc1_sf = pointer<uint8_t>(6);
    a.shared_fc2 = pointer<uint8_t>(7);
    a.shared_fc2_sf = pointer<uint8_t>(8);
    a.routed_fc1 = pointer<uint8_t>(9);
    a.routed_fc1_sf = pointer<uint8_t>(10);
    a.routed_fc2 = pointer<uint8_t>(11);
    a.routed_fc2_sf = pointer<uint8_t>(12);
    a.fc1_alpha = pointer<float>(13);
    a.fc2_alpha = pointer<float>(14);
    workspace_->bind(a);
    check_cuda(encode_maps(a, &arguments_.maps), "encode token-parallel MoE tensor maps");
  }

  void run() {
    TORCH_CHECK(workspace_->ready(), "token-parallel MoE workspace is closed");
    int const device = workspace_->device();
    check_current_device(device);
    check_cuda(launch(arguments_, at::cuda::getCurrentCUDAStream(device).stream()),
               "launch token-parallel MoE");
  }

private:
  template <class Value> Value *pointer(size_t index) const {
    return reinterpret_cast<Value *>(tensors_[index].data_ptr());
  }

  std::shared_ptr<Workspace> workspace_;
  std::vector<at::Tensor> tensors_;
  KernelArgs arguments_{};
};

} // namespace mega_moe::token_parallel

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  using mega_moe::token_parallel::Plan;
  using mega_moe::token_parallel::Workspace;
  py::class_<Workspace, std::shared_ptr<Workspace>>(module, "Workspace")
      .def(py::init<int, int>())
      .def_property_readonly("ready", &Workspace::ready)
      .def("device_uuid", &Workspace::device_uuid)
      .def("ipc_handle", &Workspace::ipc_handle)
      .def("connect", &Workspace::connect)
      .def("close_imports", &Workspace::close_imports)
      .def("close", &Workspace::close);
  py::class_<Plan>(module, "Plan")
      .def(py::init([](std::shared_ptr<Workspace> workspace, py::args tensors) {
        std::vector<at::Tensor> values;
        for (auto const &tensor : tensors)
          values.push_back(tensor.cast<at::Tensor>());
        return std::make_unique<Plan>(std::move(workspace), std::move(values));
      }))
      .def("run", &Plan::run);
}
