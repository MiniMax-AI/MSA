// m3_decode_api.cpp - C++ implementation of the M3 decode kernel API.
//
// Kernel calls go through tvm_ffi; tensor management uses ATen (libtorch).

#include "common/api/support.hpp"

#include <ATen/DLConvertor.h>
#include <ATen/cuda/CUDAContext.h>
#include <Python.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <optional>
#include <utility>

#include "attn/kernel/include/m3_decode_output_projection_handoff.hpp"
#include "common/kernel/include/tp4_nvls_layout.hpp"
#include "common/runtime/m3_decode_variant_manager.hpp"

namespace mega_atten::m3_decode {
static std::once_flag g_init_flag;

void initialize() {
  std::call_once(g_init_flag, []() {
    if (!Py_IsInitialized()) {
      Py_Initialize();
    }
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject *modules = PySys_GetObject("modules");
    if (!modules || !PyDict_GetItemString(modules, "inference.high_tps.msa_v1.common.backend")) {
      PyRun_SimpleString("import inference.high_tps.msa_v1.common.backend");
    }
    PyGILState_Release(gstate);
  });
}

void ensure_initialized() { initialize(); }

static void fix_size_one_strides(DLManagedTensor *dl) {
  if (!dl || !dl->dl_tensor.strides)
    return;
  for (int i = 0; i < dl->dl_tensor.ndim; ++i) {
    if (dl->dl_tensor.shape[i] == 1) {
      int64_t compact = 1;
      for (int j = i + 1; j < dl->dl_tensor.ndim; ++j)
        compact *= dl->dl_tensor.shape[j];
      dl->dl_tensor.strides[i] = compact;
    }
  }
}

tvm::ffi::Tensor torch_to_tvm(at::Tensor t) {
  bool native_fp8 = (t.scalar_type() == at::kFloat8_e4m3fn);
  if (native_fp8)
    t = t.view(at::kByte);
  auto *dl = at::toDLPack(t);
  fix_size_one_strides(dl);
  auto tvm_t = tvm::ffi::Tensor::FromDLPack(dl);
  if (native_fp8) {
    const_cast<DLTensor *>(tvm_t.GetDLTensorPtr())->dtype = {10, 8, 1};
  }
  return tvm_t;
}

tvm::ffi::Tensor tensor_or_null(const at::Tensor &t) {
  if (!t.defined())
    return tvm::ffi::Tensor(nullptr);
  return torch_to_tvm(t);
}

static std::vector<int32_t> tensor_to_vec_i32(const at::Tensor &t) {
  auto cpu = t.cpu().contiguous().to(torch::kInt32);
  auto ptr = cpu.data_ptr<int32_t>();
  return std::vector<int32_t>(ptr, ptr + cpu.numel());
}

void validate_hidden_state(at::Tensor const &tensor, at::Device const &device, char const *name,
                           int64_t tokens) {
  TORCH_CHECK(tensor.defined() && tensor.device() == device &&
                  tensor.scalar_type() == at::kBFloat16 && tensor.dim() == 2 &&
                  tensor.size(0) == tokens && tensor.size(1) == 6144 && tensor.is_contiguous(),
              name, " must be contiguous BF16 [", tokens, ", 6144] on the target device");
}

PlanInfo m3_decode_plan(at::Tensor qo_segment_lens, at::Tensor kv_segment_lens, int64_t topk,
                        std::optional<int> device_opt) {
  ensure_initialized();
  TORCH_CHECK(topk == 16,
              "m3_decode_megakernel plan requires sparse attention topK=16; got topK=", topk);
  TORCH_CHECK(qo_segment_lens.dim() == 1 && qo_segment_lens.numel() == 1 &&
                  kv_segment_lens.dim() == 1 && kv_segment_lens.numel() == 1,
              "m3_decode_megakernel requires one batch entry");
  std::vector<int32_t> const qo_lens = tensor_to_vec_i32(qo_segment_lens);
  std::vector<int32_t> const kv_lens = tensor_to_vec_i32(kv_segment_lens);
  TORCH_CHECK(qo_lens[0] == 8, "m3_decode_megakernel requires q_len=8; got q_len=", qo_lens[0]);
  TORCH_CHECK(kv_lens[0] >= 8, "m3_decode_megakernel requires kv_len>=8; got kv_len=", kv_lens[0]);

  int const device = device_opt.value_or(at::cuda::current_device());
  PlanInfo info;
  info.kv_segment_lens = kv_segment_lens.to(at::Device(at::kCUDA, device), at::kInt,
                                            /*non_blocking=*/false, /*copy=*/true);
  info.output_projection_barrier =
      torch::zeros({mega_atten::m3_decode::output_projection::kHandoffStateWords32},
                   torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device));
  info.kv_block_num = static_cast<int>(topk);
  info.kv_lens_cpu = kv_lens;
  return info;
}

// ============================================================================
// Full megakernel wrapper
// ============================================================================

} // namespace mega_atten::m3_decode
