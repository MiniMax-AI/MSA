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

#include "include/draft_args.hpp"

namespace py = pybind11;

namespace m31_draft {
namespace {

void check_cuda(cudaError_t status, char const *operation) {
  TORCH_CHECK(status == cudaSuccess, operation, ": ", cudaGetErrorString(status));
}

void check_not_capturing(int device) {
  cudaStreamCaptureStatus status;
  check_cuda(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(device).stream(), &status),
             "query stream capture");
  TORCH_CHECK(status == cudaStreamCaptureStatusNone,
              "draft backbone resources must be prepared before CUDA Graph capture");
}

void check_current_device(int device) {
  int current = -1;
  check_cuda(cudaGetDevice(&current), "query current CUDA device");
  TORCH_CHECK(current == device, "select the workspace CUDA device first");
}

struct TensorContract {
  char const *name;
  at::ScalarType dtype;
  std::vector<int64_t> shape; // -1: any extent
};

// Order of one layer's tensors.
std::vector<TensorContract> const kLayerTensors = {
    {"input_norm", at::kBFloat16, {kHidden}},
    {"post_norm", at::kBFloat16, {kHidden}},
    {"q_norm", at::kBFloat16, {kHeadDim}},
    {"k_norm", at::kBFloat16, {kHeadDim}},
    {"qkv", at::kFloat8_e4m3fn, {kQkvRows, kHidden}},
    {"qkv_sf", at::kByte, {kQkvRows, kHidden / kScaleBlock}},
    {"o", at::kFloat8_e4m3fn, {kHidden, kQSize}},
    {"o_sf", at::kByte, {kHidden, kQSize / kScaleBlock}},
    {"gate_up", at::kFloat8_e4m3fn, {kGateUpRows, kHidden}},
    {"gate_up_sf", at::kByte, {kGateUpRows, kHidden / kScaleBlock}},
    {"down", at::kFloat8_e4m3fn, {kHidden, kInter}},
    {"down_sf", at::kByte, {kHidden, kInter / kScaleBlock}},
    {"k_cache", at::kBFloat16, {-1, kHeadDim}},
    {"v_cache", at::kBFloat16, {-1, kHeadDim}},
};
// Order of the step tensors after the layers.
std::vector<TensorContract> const kStepTensors = {
    {"input_ids", at::kLong, {kRows}},
    {"embed", at::kBFloat16, {-1, kHidden}},
    {"out", at::kBFloat16, {kRows, kHidden}},
    {"final_norm", at::kBFloat16, {kHidden}},
    {"positions", at::kLong, {kRows}},
    {"slot_mapping", at::kLong, {kRows}},
    {"page_table", at::kInt, {-1}},
    {"kv_len", at::kInt, {1}},
    {"cos_sin", at::kFloat, {-1, kRotaryDim}},
};

void validate_tensor(at::Tensor const &tensor, TensorContract const &contract, int device) {
  bool shape_ok = tensor.defined() && tensor.dim() == int64_t(contract.shape.size());
  for (size_t i = 0; shape_ok && i < contract.shape.size(); ++i)
    shape_ok = contract.shape[i] < 0 ? tensor.size(i) > 0 : tensor.size(i) == contract.shape[i];
  TORCH_CHECK(
      shape_ok && tensor.is_cuda() && tensor.get_device() == device &&
          tensor.scalar_type() == contract.dtype && tensor.is_contiguous() &&
          !tensor.requires_grad() && reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
      contract.name, " must be a contiguous, 16-byte aligned ", c10::toString(contract.dtype),
      " tensor of the contract shape on the workspace device");
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
    TORCH_CHECK(rank >= 0 && rank < kTp, "the draft backbone requires a TP4 rank");
    c10::cuda::CUDAGuard guard(device_);
    check_not_capturing(device_);
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device_), "query CUDA device");
    TORCH_CHECK(properties.major == 10 && (properties.minor == 0 || properties.minor == 3),
                "the draft backbone requires SM100 or SM103");
    device_uuid_ = properties.uuid;
    int needed = 0, resident = 0;
    check_cuda(configure_kernel(device_, &needed, &resident), "configure draft backbone kernel");
    TORCH_CHECK(resident >= needed, "the device cannot co-schedule the ", needed,
                " cooperative draft backbone CTAs");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&allocation_), kArenaBytes),
               "allocate draft backbone workspace");
    try {
      auto stream = at::cuda::getCurrentCUDAStream(device_).stream();
      check_cuda(cudaMemsetAsync(allocation_, 0, kArenaBytes, stream),
                 "initialize draft backbone workspace");
      check_cuda(cudaStreamSynchronize(stream), "finish draft backbone initialization");
      check_cuda(cudaIpcGetMemHandle(&handle_, allocation_), "export draft backbone workspace");
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
    for (int rank = 0; rank < kTp; ++rank)
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
    TORCH_CHECK(allocation_, "draft backbone workspace is closed");
    return py::bytes(reinterpret_cast<char const *>(&handle_), sizeof(handle_));
  }

  void connect(std::vector<std::string> const &handles) {
    c10::cuda::CUDAGuard guard(device_);
    check_not_capturing(device_);
    TORCH_CHECK(allocation_ && !connected_, "workspace is closed or already connected");
    TORCH_CHECK(handles.size() == kTp, "four peer IPC handles are required");
    for (int rank = 0; rank < kTp; ++rank) {
      TORCH_CHECK(handles[rank].size() == sizeof(cudaIpcMemHandle_t), "invalid IPC handle");
      if (rank == rank_)
        TORCH_CHECK(std::memcmp(handles[rank].data(), &handle_, sizeof(handle_)) == 0,
                    "local IPC handle does not match the TP rank");
      for (int prior = 0; prior < rank; ++prior)
        TORCH_CHECK(handles[rank] != handles[prior], "TP ranks require distinct workspaces");
    }
    try {
      for (int rank = 0; rank < kTp; ++rank) {
        if (rank == rank_)
          continue;
        cudaIpcMemHandle_t handle;
        std::memcpy(&handle, handles[rank].data(), sizeof(handle));
        check_cuda(cudaIpcOpenMemHandle(reinterpret_cast<void **>(&peers_[rank]), handle,
                                        cudaIpcMemLazyEnablePeerAccess),
                   "import draft backbone workspace");
      }
      connected_ = true;
    } catch (...) {
      for (int rank = 0; rank < kTp; ++rank) {
        if (rank != rank_ && peers_[rank]) {
          cudaIpcCloseMemHandle(peers_[rank]);
          peers_[rank] = nullptr;
        }
      }
      throw;
    }
  }

  void bind(Args &args) const {
    TORCH_CHECK(ready(), "draft backbone workspace is not connected");
    for (int rank = 0; rank < kTp; ++rank)
      args.arena[rank] = peers_[rank];
    args.rank = rank_;
  }

  void close_imports() {
    if (!allocation_)
      return;
    c10::cuda::CUDAGuard guard(device_);
    check_not_capturing(device_);
    for (int rank = 0; rank < kTp; ++rank) {
      if (rank == rank_ || !peers_[rank])
        continue;
      check_cuda(cudaIpcCloseMemHandle(peers_[rank]), "close draft backbone IPC mapping");
      peers_[rank] = nullptr;
    }
    connected_ = false;
  }

  void close() {
    if (!allocation_)
      return;
    c10::cuda::CUDAGuard guard(device_);
    check_not_capturing(device_);
    for (int rank = 0; rank < kTp; ++rank)
      TORCH_CHECK(rank == rank_ || !peers_[rank], "close peer IPC mappings before freeing owners");
    check_cuda(cudaFree(allocation_), "free draft backbone workspace");
    allocation_ = nullptr;
    peers_[rank_] = nullptr;
    connected_ = false;
  }

private:
  int device_, rank_;
  uint8_t *allocation_ = nullptr;
  std::array<uint8_t *, kTp> peers_{};
  cudaIpcMemHandle_t handle_{};
  cudaUUID_t device_uuid_{};
  bool connected_ = false;
};

// Binds one fixed set of tensors. Plans sharing a workspace must run in stream order.
class Plan {
public:
  Plan(std::shared_ptr<Workspace> workspace, std::vector<std::vector<at::Tensor>> layers,
       std::vector<at::Tensor> step, int64_t vocab_start, double eps)
      : workspace_(std::move(workspace)), layers_(std::move(layers)), step_(std::move(step)) {
    TORCH_CHECK(workspace_ && workspace_->ready(), "an open TP4 workspace is required");
    int const device = workspace_->device();
    check_current_device(device);
    check_not_capturing(device);
    TORCH_CHECK(layers_.size() == kLayers, "the draft backbone plan requires ", kLayers, " layers");
    TORCH_CHECK(step_.size() == kStepTensors.size(), "the draft backbone plan requires ",
                kStepTensors.size(), " step tensors");
    for (auto const &layer : layers_) {
      TORCH_CHECK(layer.size() == kLayerTensors.size(), "every layer requires ",
                  kLayerTensors.size(), " tensors");
      for (size_t i = 0; i < layer.size(); ++i)
        validate_tensor(layer[i], kLayerTensors[i], device);
      TORCH_CHECK(layer[12].size(0) == layer[13].size(0), "K and V caches need the same slots");
    }
    for (size_t i = 0; i < step_.size(); ++i)
      validate_tensor(step_[i], kStepTensors[i], device);
    at::Tensor const &out = step_[2];
    for (auto const &layer : layers_)
      for (auto const &tensor : layer)
        TORCH_CHECK(!overlaps(out, tensor), "out must be disjoint from every operand");
    for (size_t i = 0; i < step_.size(); ++i)
      if (i != 2)
        TORCH_CHECK(!overlaps(out, step_[i]), "out must be disjoint from every operand");
    TORCH_CHECK(eps > 0, "eps must be positive");
    Args &a = args_;
    for (int l = 0; l < kLayers; ++l) {
      auto const &t = layers_[l];
      LayerArgs &layer = a.layer[l];
      layer.input_norm = pointer<uint16_t>(t[0]);
      layer.post_norm = pointer<uint16_t>(t[1]);
      layer.q_norm = pointer<uint16_t>(t[2]);
      layer.k_norm = pointer<uint16_t>(t[3]);
      layer.qkv = pointer<uint8_t>(t[4]);
      layer.qkv_sf = pointer<uint8_t>(t[5]);
      layer.o = pointer<uint8_t>(t[6]);
      layer.o_sf = pointer<uint8_t>(t[7]);
      layer.gate_up = pointer<uint8_t>(t[8]);
      layer.gate_up_sf = pointer<uint8_t>(t[9]);
      layer.down = pointer<uint8_t>(t[10]);
      layer.down_sf = pointer<uint8_t>(t[11]);
      layer.k_cache = pointer<uint16_t>(t[12]);
      layer.v_cache = pointer<uint16_t>(t[13]);
    }
    a.input_ids = pointer<int64_t>(step_[0]);
    a.embed = pointer<uint16_t>(step_[1]);
    a.out = pointer<uint16_t>(step_[2]);
    a.final_norm = pointer<uint16_t>(step_[3]);
    a.positions = pointer<int64_t>(step_[4]);
    a.slot_mapping = pointer<int64_t>(step_[5]);
    a.page_table = pointer<int32_t>(step_[6]);
    a.kv_len = pointer<int32_t>(step_[7]);
    a.cos_sin = pointer<float>(step_[8]);
    a.vocab_start = vocab_start;
    a.vocab_rows = step_[1].size(0);
    a.max_pages = int(step_[6].size(0));
    a.eps = float(eps);
    workspace_->bind(a);
    check_cuda(prepare_plan(a, state_), "prepare draft backbone plan");
  }

  void run() {
    TORCH_CHECK(workspace_->ready(), "draft backbone workspace is closed");
    int const device = workspace_->device();
    check_current_device(device);
    check_cuda(launch(args_, state_, at::cuda::getCurrentCUDAStream(device).stream()),
               "launch draft backbone");
  }

private:
  template <class Value> static Value *pointer(at::Tensor const &tensor) {
    return reinterpret_cast<Value *>(tensor.data_ptr());
  }

  std::shared_ptr<Workspace> workspace_;
  std::vector<std::vector<at::Tensor>> layers_;
  std::vector<at::Tensor> step_;
  Args args_{};
  alignas(64) uint8_t state_[kPlanStateBytes] = {};
};

} // namespace m31_draft

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  using m31_draft::Plan;
  using m31_draft::Workspace;
  // Module-local types: several builds of this extension may be loaded into one process.
  py::class_<Workspace, std::shared_ptr<Workspace>>(module, "Workspace", py::module_local())
      .def(py::init<int, int>())
      .def_property_readonly("ready", &Workspace::ready)
      .def("device_uuid", &Workspace::device_uuid)
      .def("ipc_handle", &Workspace::ipc_handle)
      .def("connect", &Workspace::connect)
      .def("close_imports", &Workspace::close_imports)
      .def("close", &Workspace::close);
  py::class_<Plan>(module, "Plan", py::module_local())
      .def(py::init<std::shared_ptr<Workspace>, std::vector<std::vector<at::Tensor>>,
                    std::vector<at::Tensor>, int64_t, double>())
      .def("run", &Plan::run);
}
