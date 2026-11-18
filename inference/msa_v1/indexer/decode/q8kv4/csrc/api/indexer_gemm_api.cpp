#include "indexer_gemm_api.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "sm100/common/indexer_gemm_arguments.hpp"
#include "sm100/common/indexer_gemm_traits.hpp"

namespace minimax::msa_v1::indexer::decode::q8kv4 {
namespace {

void check_cuda_contiguous(torch::Tensor const &tensor, char const *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(torch::Tensor const &reference, torch::Tensor const &tensor,
                       char const *name) {
  TORCH_CHECK(tensor.device() == reference.device(), name,
              " must be on the same device as the planned metadata");
}

void check_cache_layout(torch::Tensor const &tensor, char const *name, int alignment) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.dim() >= 2, name, " must have a page dimension");
  TORCH_CHECK(tensor.stride(-1) == 1, name, " last dimension must have unit stride");
  for (int64_t d = 0; d < tensor.dim() - 1; ++d) {
    TORCH_CHECK(tensor.size(d) == 1 || (tensor.stride(d) > 0 &&
                                        tensor.stride(d) * tensor.element_size() % alignment == 0),
                name, " axis ", d, " must have positive byte stride aligned to ", alignment);
  }
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0, name,
              " data pointer must be aligned to ", alignment);
}

void check_page_strided(torch::Tensor const &tensor, char const *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.dim() >= 2, name, " must have a page dimension");
  int64_t packed = 1;
  for (int64_t d = tensor.dim() - 1; d >= 1; --d) {
    TORCH_CHECK(tensor.size(d) == 1 || tensor.stride(d) == packed, name,
                " must be packed below the page dimension");
    packed *= tensor.size(d);
  }
  int64_t const page_stride_bytes = tensor.stride(0) * tensor.element_size();
  TORCH_CHECK(tensor.stride(0) >= packed && page_stride_bytes % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
              name, " page stride must cover one page and keep 16-byte alignment");
}
int checked_batch(int64_t batch_size) {
  TORCH_CHECK(batch_size > 0 && batch_size <= std::numeric_limits<int>::max(),
              "batch_size must be positive and fit in int32");
  return static_cast<int>(batch_size);
}

size_t scheduler_workspace_bytes(int batch, int sm_count) {
  int64_t const element_count =
      static_cast<int64_t>(batch) + (1 + kWorkerStartFields) * sm_count + 2;
  return static_cast<size_t>(element_count) * sizeof(int32_t);
}

void check_metadata(torch::Tensor const &page_table, torch::Tensor const &seq_lens) {
  using Traits = IndexerGemmTraits;
  check_cuda_contiguous(page_table, "page_table");
  check_cuda_contiguous(seq_lens, "seq_lens");
  check_same_device(page_table, seq_lens, "seq_lens");
  TORCH_CHECK(page_table.scalar_type() == at::kInt, "page_table must have dtype torch.int32");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "seq_lens must have dtype torch.int32");
  TORCH_CHECK(page_table.dim() == 2 && page_table.size(0) > 0,
              "page_table must have shape [batch, max_pages]");
  TORCH_CHECK(page_table.size(0) <= std::numeric_limits<int>::max(), "batch must fit in int32");
  TORCH_CHECK(page_table.size(1) > 0 && page_table.size(1) <= Traits::kMaximumPages,
              "max_pages must be in [1, ", Traits::kMaximumPages, "]");
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.size(0) == page_table.size(0),
              "seq_lens must have shape [batch]");
}

void check_workspace(torch::Tensor const &reference, torch::Tensor const &workspace,
                     size_t required_bytes) {
  check_cuda_contiguous(workspace, "workspace");
  check_same_device(reference, workspace, "workspace");
  TORCH_CHECK(workspace.scalar_type() == at::kByte, "workspace must have dtype torch.uint8");
  TORCH_CHECK(workspace.dim() == 1, "workspace must be a one-dimensional opaque buffer");
  TORCH_CHECK(static_cast<uint64_t>(workspace.numel()) >= required_bytes,
              "workspace is too small: need ", required_bytes, " bytes, got ", workspace.numel());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % alignof(int32_t) == 0,
              "workspace address must be aligned to int32");
}

} // namespace

torch::Tensor indexer_gemm_run(torch::Tensor q, torch::Tensor packed_k_cache, torch::Tensor k_scale,
                               torch::Tensor page_table, torch::Tensor seq_lens,
                               torch::Tensor workspace, int64_t sm_count, torch::Tensor output,
                               bool pdl) {
  using Traits = IndexerGemmTraits;
  check_metadata(page_table, seq_lens);
  check_cuda_contiguous(q, "q");
  check_cache_layout(packed_k_cache, "packed_k_cache", 16);
  check_page_strided(k_scale, "k_scale");
  check_cuda_contiguous(output, "out");
  check_same_device(page_table, q, "q");
  check_same_device(page_table, packed_k_cache, "packed_k_cache");
  check_same_device(page_table, k_scale, "k_scale");
  check_same_device(page_table, output, "out");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0, "q must be 16-byte aligned");

  TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn, "q must have dtype torch.float8_e4m3fn");
  TORCH_CHECK(packed_k_cache.scalar_type() == at::kByte,
              "packed_k_cache must have dtype torch.uint8");
  TORCH_CHECK(k_scale.scalar_type() == at::kFloat8_e4m3fn,
              "k_scale must have dtype torch.float8_e4m3fn");
  TORCH_CHECK(output.scalar_type() == at::kFloat, "out must have dtype torch.float32");

  int const batch = checked_batch(page_table.size(0));
  int const max_pages = static_cast<int>(page_table.size(1));
  TORCH_CHECK(sm_count > 0 && sm_count <= std::numeric_limits<int>::max(),
              "planned sm_count must be positive and fit in int32");
  TORCH_CHECK(q.dim() == 4 && q.size(0) == batch && q.size(1) >= 1 &&
                  q.size(1) <= Traits::kMaxQueryLength && q.size(2) == Traits::kNumIndexHeads &&
                  q.size(3) == Traits::kHeadDim,
              "q must have shape [batch, Q, H, 128], Q in [1, 16]");
  TORCH_CHECK(q.size(1) * Traits::kNumIndexHeads <= Traits::kQueryColumns,
              "query columns exceed the compiled MMA capacity");
  TORCH_CHECK(packed_k_cache.dim() == 3 && packed_k_cache.size(0) > 0 &&
                  packed_k_cache.size(0) <= std::numeric_limits<int>::max() &&
                  packed_k_cache.size(1) == Traits::kPageTokens &&
                  packed_k_cache.size(2) == Traits::kHeadDim / 2,
              "packed_k_cache must have shape [physical_pages, 128, 64]");
  TORCH_CHECK(k_scale.dim() == 3 && k_scale.size(0) == packed_k_cache.size(0) &&
                  k_scale.size(1) == Traits::kPageTokens && k_scale.size(2) == Traits::kScaleGroups,
              "k_scale must have shape [physical_pages, 128, 8]");
  TORCH_CHECK(output.dim() == 3 && output.size(0) == Traits::kNumIndexHeads &&
                  output.size(1) == batch * q.size(1) && output.size(2) == max_pages,
              "out must have shape [H, batch * Q, max_pages]");
  check_workspace(page_table, workspace,
                  scheduler_workspace_bytes(batch, static_cast<int>(sm_count)));

  c10::cuda::CUDAGuard const device_guard(q.device());
  IndexerGemmArguments arguments{};
  arguments.q_ptr = q.data_ptr();
  arguments.query_length = static_cast<int>(q.size(1));
  arguments.packed_k_ptr = packed_k_cache.data_ptr();
  arguments.k_scale_ptr = k_scale.data_ptr();
  arguments.page_table_ptr = page_table.data_ptr<int32_t>();
  arguments.kv_lengths_ptr = seq_lens.data_ptr<int32_t>();
  arguments.scheduler_workspace_ptr = reinterpret_cast<int32_t *>(workspace.data_ptr<uint8_t>());
  arguments.output_ptr = output.data_ptr<float>();
  arguments.batch = batch;
  arguments.max_pages = max_pages;
  arguments.physical_pages = static_cast<int>(packed_k_cache.size(0));
  // Singleton axes do not address another element, but TMA still validates their strides.
  arguments.packed_k_page_stride =
      arguments.physical_pages == 1 ? Traits::kPackedKBytes : packed_k_cache.stride(0);
  arguments.packed_k_token_stride = packed_k_cache.stride(1);
  arguments.k_scale_page_stride = k_scale.stride(0) * k_scale.element_size();
  arguments.sm_count = static_cast<int>(sm_count);
  arguments.pdl = pdl;

  cudaStream_t const stream = at::cuda::getCurrentCUDAStream().stream();
  cudaError_t const status = launch_indexer_gemm(arguments, stream);
  TORCH_CHECK(status == cudaSuccess, "indexer GEMM run failed: ", cudaGetErrorString(status));
  return output;
}

} // namespace minimax::msa_v1::indexer::decode::q8kv4
