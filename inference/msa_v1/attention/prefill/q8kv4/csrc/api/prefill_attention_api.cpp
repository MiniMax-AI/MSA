#include "prefill_attention_api.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "sm100/device/prefill_attention.hpp"

namespace minimax::msa_v1::attention::prefill::q8kv4 {
namespace {

constexpr int kQHeadsPerKv = 16;
constexpr int kHeadDim = 128;
constexpr int kPageSize = 128;
constexpr int kTopK = 16;

void check_cuda_contiguous(torch::Tensor const &tensor, char const *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(torch::Tensor const &reference, torch::Tensor const &tensor,
                       char const *name) {
  TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same device as q");
}

int checked_int_dimension(int64_t value, char const *name) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(), name, " must fit int32");
  return static_cast<int>(value);
}

} // namespace

void prefill_run(torch::Tensor q, torch::Tensor packed_k, torch::Tensor packed_v,
                 torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor page_table,
                 torch::Tensor cu_seqlens_q, torch::Tensor cu_seqlens_k, torch::Tensor k2q_row_ptr,
                 torch::Tensor qsplit_indices, torch::Tensor scheduler_metadata,
                 torch::Tensor work_count, torch::Tensor o_partial, torch::Tensor lse_partial,
                 double softmax_scale) {
  torch::Tensor tensors[] = {
      q,           packed_k,       packed_v,           k_scale,
      v_scale,     page_table,     cu_seqlens_q,       cu_seqlens_k,
      k2q_row_ptr, qsplit_indices, scheduler_metadata, work_count,
      o_partial,   lse_partial,
  };
  char const *names[] = {
      "q",           "packed_k",       "packed_v",           "k_scale",
      "v_scale",     "page_table",     "cu_seqlens_q",       "cu_seqlens_k",
      "k2q_row_ptr", "qsplit_indices", "scheduler_metadata", "work_count",
      "o_partial",   "lse_partial",
  };
  constexpr std::size_t kTensorCount = sizeof(tensors) / sizeof(tensors[0]);
  static_assert(kTensorCount == sizeof(names) / sizeof(names[0]));
  for (std::size_t i = 0; i < kTensorCount; ++i) {
    check_cuda_contiguous(tensors[i], names[i]);
    check_same_device(q, tensors[i], names[i]);
  }

  TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn, "q must have dtype torch.float8_e4m3fn");
  TORCH_CHECK(packed_k.scalar_type() == at::kByte && packed_v.scalar_type() == at::kByte,
              "packed K/V must have dtype torch.uint8");
  TORCH_CHECK(k_scale.scalar_type() == at::kFloat8_e4m3fn &&
                  v_scale.scalar_type() == at::kFloat8_e4m3fn,
              "K/V scale must have dtype torch.float8_e4m3fn");
  for (auto const &tensor : {page_table, cu_seqlens_q, cu_seqlens_k, k2q_row_ptr, qsplit_indices,
                             scheduler_metadata, work_count}) {
    TORCH_CHECK(tensor.scalar_type() == at::kInt,
                "scheduler and page metadata must have dtype torch.int32");
  }
  TORCH_CHECK(o_partial.scalar_type() == at::kBFloat16, "o_partial must have dtype torch.bfloat16");
  TORCH_CHECK(lse_partial.scalar_type() == at::kFloat, "lse_partial must have dtype torch.float32");

  TORCH_CHECK(q.dim() == 3 && q.size(1) > 0 && q.size(2) == kHeadDim,
              "q must have shape [total_q, num_q_heads, 128]");
  int const num_q_heads = checked_int_dimension(q.size(1), "num_q_heads");
  int64_t const total_q_64 = q.size(0);
  int const total_q = checked_int_dimension(total_q_64, "total_q");
  TORCH_CHECK(packed_k.dim() == 4 && packed_k.size(1) > 0 && packed_k.size(2) == kPageSize &&
                  packed_k.size(3) == kHeadDim / 2,
              "packed_k must have shape [pages, num_kv_heads, 128, 64]");
  int const num_kv_heads = checked_int_dimension(packed_k.size(1), "num_kv_heads");
  TORCH_CHECK(num_q_heads == static_cast<int64_t>(num_kv_heads) * kQHeadsPerKv,
              "Q8KV4 prefill requires exactly 16 Q heads per KV head");
  TORCH_CHECK(packed_v.sizes() == packed_k.sizes(),
              "packed_v must have the same shape as packed_k");
  TORCH_CHECK(k_scale.dim() == 4 && k_scale.size(0) == packed_k.size(0) &&
                  k_scale.size(1) == num_kv_heads && k_scale.size(2) == kPageSize &&
                  k_scale.size(3) == kHeadDim / 16,
              "k_scale must have shape [pages, num_kv_heads, 128, 8]");
  TORCH_CHECK(v_scale.sizes() == k_scale.sizes(), "v_scale must have the same shape as k_scale");
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.numel() >= 2,
              "cu_seqlens_q must have shape [batch + 1]");
  TORCH_CHECK(cu_seqlens_k.sizes() == cu_seqlens_q.sizes(), "cu_seqlens_k must match cu_seqlens_q");
  int const batch = static_cast<int>(cu_seqlens_q.numel() - 1);
  TORCH_CHECK(page_table.dim() == 2 && page_table.size(0) == batch && page_table.size(1) > 0,
              "page_table must have shape [batch, max_pages]");
  TORCH_CHECK(k2q_row_ptr.dim() == 2 && k2q_row_ptr.size(0) == num_kv_heads &&
                  k2q_row_ptr.size(1) >= 1,
              "k2q_row_ptr must have shape [num_kv_heads, total_rows + 1]");
  TORCH_CHECK(qsplit_indices.dim() == 2 && qsplit_indices.size(0) == num_kv_heads,
              "qsplit_indices must have shape [num_kv_heads, nnz_capacity]");
  TORCH_CHECK(scheduler_metadata.dim() == 2 && scheduler_metadata.size(1) == 6,
              "scheduler_metadata must have shape [work_capacity, 6]");
  TORCH_CHECK(work_count.dim() == 1 && work_count.numel() == 1, "work_count must have shape [1]");
  TORCH_CHECK(o_partial.dim() == 4 && o_partial.size(0) == kTopK &&
                  o_partial.size(1) == total_q_64 && o_partial.size(2) == num_q_heads &&
                  o_partial.size(3) == kHeadDim,
              "o_partial must have shape [16, total_q, num_q_heads, 128]");
  TORCH_CHECK(lse_partial.dim() == 3 && lse_partial.size(0) == kTopK &&
                  lse_partial.size(1) == total_q_64 && lse_partial.size(2) == num_q_heads,
              "lse_partial must have shape [16, total_q, num_q_heads]");
  TORCH_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0.0,
              "softmax_scale must be finite and positive");

  c10::cuda::CUDAGuard const device_guard(q.device());
  cudaDeviceProp const *properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 10 && (properties->minor == 0 || properties->minor == 3),
              "Q8KV4 sparse prefill requires an SM100-family GPU");

  PrefillArguments arguments{};
  arguments.q_ptr = reinterpret_cast<uint8_t const *>(q.data_ptr());
  arguments.packed_k_ptr = packed_k.data_ptr<uint8_t>();
  arguments.packed_v_ptr = packed_v.data_ptr<uint8_t>();
  arguments.k_scale_ptr = reinterpret_cast<uint8_t const *>(k_scale.data_ptr());
  arguments.v_scale_ptr = reinterpret_cast<uint8_t const *>(v_scale.data_ptr());
  arguments.page_table_ptr = page_table.data_ptr<int32_t>();
  arguments.cu_seqlens_q_ptr = cu_seqlens_q.data_ptr<int32_t>();
  arguments.cu_seqlens_k_ptr = cu_seqlens_k.data_ptr<int32_t>();
  arguments.k2q_row_ptr = k2q_row_ptr.data_ptr<int32_t>();
  arguments.qsplit_indices_ptr = qsplit_indices.data_ptr<int32_t>();
  arguments.scheduler_metadata_ptr = scheduler_metadata.data_ptr<int32_t>();
  arguments.work_count_ptr = work_count.data_ptr<int32_t>();
  arguments.o_partial_ptr =
      reinterpret_cast<cutlass::bfloat16_t *>(o_partial.data_ptr<at::BFloat16>());
  arguments.lse_partial_ptr = lse_partial.data_ptr<float>();
  arguments.total_q = total_q;
  arguments.num_q_heads = num_q_heads;
  arguments.num_kv_heads = num_kv_heads;
  arguments.physical_pages = checked_int_dimension(packed_k.size(0), "physical_pages");
  arguments.max_pages = checked_int_dimension(page_table.size(1), "max_pages");
  arguments.total_rows = checked_int_dimension(k2q_row_ptr.size(1) - 1, "total_rows");
  arguments.qsplit_stride = checked_int_dimension(qsplit_indices.size(1), "qsplit_stride");
  arguments.work_capacity = checked_int_dimension(scheduler_metadata.size(0), "work_capacity");
  arguments.softmax_scale_log2 = static_cast<float>(softmax_scale * 1.4426950408889634074);

  cudaStream_t const stream = at::cuda::getCurrentCUDAStream().stream();
  cudaError_t const status = launch_prefill_attention(arguments, stream);
  TORCH_CHECK(status == cudaSuccess,
              "Q8KV4 sparse prefill launch failed: ", cudaGetErrorString(status));
}

} // namespace minimax::msa_v1::attention::prefill::q8kv4
