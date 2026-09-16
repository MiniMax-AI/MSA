#include "indexer_gemm_api.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "sm100/common/indexer_gemm_arguments.hpp"
#include "sm100/common/indexer_gemm_traits.hpp"

namespace minimax::msa_v1::indexer::decode::tp4_q8kv4 {
namespace {

constexpr size_t kWorkspaceAlignment = 256;

struct WorkspaceLayout {
  size_t scheduler_bytes = 0;
  size_t temp_storage_offset = 0;
  size_t temp_storage_bytes = 0;
  size_t total_bytes = 0;
};

void check_cuda_contiguous(torch::Tensor const& tensor, char const* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(torch::Tensor const& reference,
                       torch::Tensor const& tensor, char const* name) {
  TORCH_CHECK(tensor.device() == reference.device(), name,
              " must be on the same device as the planned metadata");
}

int checked_batch(int64_t batch_size) {
  TORCH_CHECK(batch_size > 0 &&
                  batch_size <= std::numeric_limits<int>::max(),
              "batch_size must be positive and fit in int32");
  return static_cast<int>(batch_size);
}

size_t scheduler_workspace_bytes(int batch) {
  int64_t const element_count = std::max<int64_t>(
      static_cast<int64_t>(batch) + 2,
      static_cast<int64_t>(IndexerGemmTraits::kPrepareThreads) + 2);
  return static_cast<size_t>(element_count) * sizeof(int32_t);
}

WorkspaceLayout get_workspace_layout(int batch) {
  WorkspaceLayout layout{};
  layout.scheduler_bytes = scheduler_workspace_bytes(batch);
  cudaError_t const status = get_indexer_gemm_scheduler_temp_storage_bytes(
      batch, layout.temp_storage_bytes);
  TORCH_CHECK(status == cudaSuccess,
              "indexer GEMM scheduler workspace query failed: ",
              cudaGetErrorString(status));
  if (layout.temp_storage_bytes == 0) {
    layout.temp_storage_offset = layout.scheduler_bytes;
    layout.total_bytes = layout.scheduler_bytes;
    return layout;
  }
  layout.temp_storage_offset =
      (layout.scheduler_bytes + kWorkspaceAlignment - 1) &
      ~(kWorkspaceAlignment - 1);
  TORCH_CHECK(
      layout.temp_storage_bytes <=
          std::numeric_limits<size_t>::max() - layout.temp_storage_offset,
      "workspace size overflow");
  layout.total_bytes =
      layout.temp_storage_offset + layout.temp_storage_bytes;
  return layout;
}

void check_metadata(torch::Tensor const& page_table,
                    torch::Tensor const& seq_lens) {
  using Traits = IndexerGemmTraits;
  check_cuda_contiguous(page_table, "page_table");
  check_cuda_contiguous(seq_lens, "seq_lens");
  check_same_device(page_table, seq_lens, "seq_lens");
  TORCH_CHECK(page_table.scalar_type() == at::kInt,
              "page_table must have dtype torch.int32");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt,
              "seq_lens must have dtype torch.int32");
  TORCH_CHECK(page_table.dim() == 2 && page_table.size(0) > 0,
              "page_table must have shape [batch, max_pages]");
  TORCH_CHECK(page_table.size(0) <= std::numeric_limits<int>::max(),
              "batch must fit in int32");
  TORCH_CHECK(page_table.size(1) > 0 &&
                  page_table.size(1) <= Traits::kMaximumPages,
              "max_pages must be in [1, ", Traits::kMaximumPages, "]");
  TORCH_CHECK(seq_lens.dim() == 1 &&
                  seq_lens.size(0) == page_table.size(0),
              "seq_lens must have shape [batch]");
}

void check_workspace(torch::Tensor const& reference,
                     torch::Tensor const& workspace,
                     size_t required_bytes) {
  check_cuda_contiguous(workspace, "workspace");
  check_same_device(reference, workspace, "workspace");
  TORCH_CHECK(workspace.scalar_type() == at::kByte,
              "workspace must have dtype torch.uint8");
  TORCH_CHECK(workspace.dim() == 1,
              "workspace must be a one-dimensional opaque buffer");
  TORCH_CHECK(static_cast<uint64_t>(workspace.numel()) >= required_bytes,
              "workspace is too small: need ", required_bytes,
              " bytes, got ", workspace.numel());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) %
                      alignof(int32_t) ==
                  0,
              "workspace address must be aligned to int32");
}

int check_sm100_family(torch::Tensor const& reference) {
  c10::cuda::CUDAGuard const device_guard(reference.device());
  cudaDeviceProp const* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 10 &&
                  (properties->minor == 0 || properties->minor == 3),
              "MSA v1 decode indexer GEMM requires an SM100-family GPU");
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  TORCH_CHECK(status == cudaSuccess, "failed to query CUDA device: ",
              cudaGetErrorString(status));
  int sm_count = 0;
  status = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                  device);
  TORCH_CHECK(status == cudaSuccess, "failed to query CUDA SM count: ",
              cudaGetErrorString(status));
  return sm_count;
}

void check_not_capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  cudaError_t const status = cudaStreamIsCapturing(stream, &capture_status);
  TORCH_CHECK(status == cudaSuccess,
              "failed to query CUDA stream capture state: ",
              cudaGetErrorString(status));
  TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
              "plan() must be called outside CUDA Graph capture");
}

}  // namespace

int64_t indexer_gemm_workspace_size(int64_t batch_size) {
  int const batch = checked_batch(batch_size);
  WorkspaceLayout const layout = get_workspace_layout(batch);
  TORCH_CHECK(layout.total_bytes <=
                  static_cast<size_t>(std::numeric_limits<int64_t>::max()),
              "workspace size does not fit in int64");
  return static_cast<int64_t>(layout.total_bytes);
}

int64_t indexer_gemm_plan(torch::Tensor page_table, torch::Tensor seq_lens,
                          torch::Tensor workspace) {
  check_metadata(page_table, seq_lens);
  int const batch = checked_batch(page_table.size(0));
  WorkspaceLayout const layout = get_workspace_layout(batch);
  check_workspace(page_table, workspace, layout.total_bytes);
  int const sm_count = check_sm100_family(page_table);

  c10::cuda::CUDAGuard const device_guard(page_table.device());
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream().stream();
  check_not_capturing(stream);

  auto* workspace_ptr = workspace.data_ptr<uint8_t>();
  IndexerGemmArguments arguments{};
  arguments.page_table_ptr = page_table.data_ptr<int32_t>();
  arguments.kv_lengths_ptr = seq_lens.data_ptr<int32_t>();
  arguments.scheduler_workspace_ptr =
      reinterpret_cast<int32_t*>(workspace_ptr);
  arguments.scheduler_temp_storage_ptr =
      layout.temp_storage_bytes > 0
          ? workspace_ptr + layout.temp_storage_offset
          : nullptr;
  arguments.scheduler_temp_storage_bytes = layout.temp_storage_bytes;
  arguments.batch = batch;
  arguments.max_pages = static_cast<int>(page_table.size(1));

  cudaError_t const status = prepare_indexer_gemm(arguments, stream);
  TORCH_CHECK(status == cudaSuccess, "indexer GEMM plan failed: ",
              cudaGetErrorString(status));
  return sm_count;
}

torch::Tensor indexer_gemm_run(
    torch::Tensor q, torch::Tensor packed_k_cache, torch::Tensor k_scale,
    torch::Tensor page_table, torch::Tensor seq_lens,
    torch::Tensor workspace, int64_t sm_count, torch::Tensor output) {
  using Traits = IndexerGemmTraits;
  check_metadata(page_table, seq_lens);
  check_cuda_contiguous(q, "q");
  check_cuda_contiguous(packed_k_cache, "packed_k_cache");
  check_cuda_contiguous(k_scale, "k_scale");
  check_cuda_contiguous(output, "out");
  check_same_device(page_table, q, "q");
  check_same_device(page_table, packed_k_cache, "packed_k_cache");
  check_same_device(page_table, k_scale, "k_scale");
  check_same_device(page_table, output, "out");

  TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn,
              "q must have dtype torch.float8_e4m3fn");
  TORCH_CHECK(packed_k_cache.scalar_type() == at::kByte,
              "packed_k_cache must have dtype torch.uint8");
  TORCH_CHECK(k_scale.scalar_type() == at::kFloat8_e4m3fn,
              "k_scale must have dtype torch.float8_e4m3fn");
  TORCH_CHECK(output.scalar_type() == at::kFloat,
              "out must have dtype torch.float32");

  int const batch = checked_batch(page_table.size(0));
  int const max_pages = static_cast<int>(page_table.size(1));
  TORCH_CHECK(sm_count > 0 && sm_count <= std::numeric_limits<int>::max(),
              "planned sm_count must be positive and fit in int32");
  TORCH_CHECK(q.dim() == 3 && q.size(0) == batch &&
                  q.size(1) == Traits::kQueryLength &&
                  q.size(2) == Traits::kHeadDim,
              "q must have shape [batch, 8, 128]");
  TORCH_CHECK(packed_k_cache.dim() == 3 &&
                  packed_k_cache.size(0) > 0 &&
                  packed_k_cache.size(0) <=
                      std::numeric_limits<int>::max() &&
                  packed_k_cache.size(1) == Traits::kPageTokens &&
                  packed_k_cache.size(2) == Traits::kHeadDim / 2,
              "packed_k_cache must have shape [physical_pages, 128, 64]");
  TORCH_CHECK(k_scale.dim() == 3 &&
                  k_scale.size(0) == packed_k_cache.size(0) &&
                  k_scale.size(1) == Traits::kPageTokens &&
                  k_scale.size(2) == Traits::kScaleGroups,
              "k_scale must have shape [physical_pages, 128, 8]");
  TORCH_CHECK(output.dim() == 3 && output.size(0) == batch &&
                  output.size(1) == Traits::kQueryLength &&
                  output.size(2) == max_pages,
              "out must have shape [batch, 8, max_pages]");
  check_workspace(page_table, workspace,
                  scheduler_workspace_bytes(batch));

  c10::cuda::CUDAGuard const device_guard(q.device());
  IndexerGemmArguments arguments{};
  arguments.q_ptr = q.data_ptr();
  arguments.packed_k_ptr = packed_k_cache.data_ptr();
  arguments.k_scale_ptr = k_scale.data_ptr();
  arguments.page_table_ptr = page_table.data_ptr<int32_t>();
  arguments.kv_lengths_ptr = seq_lens.data_ptr<int32_t>();
  arguments.scheduler_workspace_ptr =
      reinterpret_cast<int32_t*>(workspace.data_ptr<uint8_t>());
  arguments.output_ptr = output.data_ptr<float>();
  arguments.batch = batch;
  arguments.max_pages = max_pages;
  arguments.physical_pages = static_cast<int>(packed_k_cache.size(0));
  arguments.sm_count = static_cast<int>(sm_count);

  cudaStream_t const stream = at::cuda::getCurrentCUDAStream().stream();
  cudaError_t const status = launch_indexer_gemm(arguments, stream);
  TORCH_CHECK(status == cudaSuccess, "indexer GEMM run failed: ",
              cudaGetErrorString(status));
  return output;
}

}  // namespace minimax::msa_v1::indexer::decode::tp4_q8kv4
