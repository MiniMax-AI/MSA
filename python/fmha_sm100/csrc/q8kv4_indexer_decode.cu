// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

// Q8KV4 paged decode indexer proxy scores over the vLLM packed NVFP4 page.

#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "q8kv4_indexer/runner.hpp"
#include "tvm_ffi_utils.h"

namespace q8kv4_indexer {

cudaError_t get_indexer_gemm_scheduler_temp_storage_bytes(int batch, size_t &temp_storage_bytes) {
  temp_storage_bytes = 0;
  if (batch <= IndexerGemmTraits::kPrepareThreads) {
    return cudaSuccess;
  }
  return cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, static_cast<int32_t *>(nullptr),
                                       static_cast<int32_t *>(nullptr), batch);
}

cudaError_t run_indexer_gemm_scheduler_scan(int32_t *prefix, int batch, void *temp_storage,
                                            size_t temp_storage_bytes, cudaStream_t stream) {
  return cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes, prefix, prefix, batch,
                                       stream);
}

cudaError_t prepare_indexer_gemm(IndexerGemmArguments const &arguments, cudaStream_t stream) {
  using Runner = IndexerGemmRunner<IndexerGemmTraits>;
  if (!Runner::can_plan(arguments)) {
    return cudaErrorNotSupported;
  }
  IndexerGemmParams params{};
  cudaError_t status = Runner::to_plan_params(arguments, params);
  if (status != cudaSuccess) {
    return status;
  }
  return Runner::plan(params, arguments.scheduler_temp_storage_ptr,
                      arguments.scheduler_temp_storage_bytes, stream);
}

cudaError_t launch_indexer_gemm(IndexerGemmArguments const &arguments, cudaStream_t stream) {
  using Runner = IndexerGemmRunner<IndexerGemmTraits>;
  if (!Runner::can_run(arguments)) {
    return cudaErrorNotSupported;
  }
  IndexerGemmParams params{};
  cudaError_t status = Runner::to_underlying_arguments(arguments, params);
  if (status != cudaSuccess) {
    return status;
  }
  return Runner::run(params, stream);
}

} // namespace q8kv4_indexer

namespace {

using q8kv4_indexer::IndexerGemmArguments;
using q8kv4_indexer::IndexerGemmTraits;

constexpr size_t kWorkspaceAlignment = 256;

struct WorkspaceLayout {
  size_t scheduler_bytes = 0;
  size_t temp_storage_offset = 0;
  size_t temp_storage_bytes = 0;
  size_t total_bytes = 0;
};

template <typename T> T *tensor_data(TensorView tensor) {
  return reinterpret_cast<T *>(static_cast<char *>(tensor.data_ptr()) + tensor.byte_offset());
}

int checked_batch(int64_t batch_size) {
  TVM_FFI_ICHECK(batch_size > 0 && batch_size <= std::numeric_limits<int>::max())
      << "batch_size must be positive and fit in int32";
  return static_cast<int>(batch_size);
}

size_t scheduler_workspace_bytes(int batch) {
  int64_t const element_count =
      std::max<int64_t>(static_cast<int64_t>(batch) + 2,
                        static_cast<int64_t>(IndexerGemmTraits::kPrepareThreads) + 2);
  return static_cast<size_t>(element_count) * sizeof(int32_t);
}

WorkspaceLayout get_workspace_layout(int batch) {
  WorkspaceLayout layout{};
  layout.scheduler_bytes = scheduler_workspace_bytes(batch);
  cudaError_t const status = q8kv4_indexer::get_indexer_gemm_scheduler_temp_storage_bytes(
      batch, layout.temp_storage_bytes);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Q8KV4 indexer scheduler workspace query failed: " << cudaGetErrorString(status);
  if (layout.temp_storage_bytes == 0) {
    layout.temp_storage_offset = layout.scheduler_bytes;
    layout.total_bytes = layout.scheduler_bytes;
    return layout;
  }
  layout.temp_storage_offset =
      (layout.scheduler_bytes + kWorkspaceAlignment - 1) & ~(kWorkspaceAlignment - 1);
  TVM_FFI_ICHECK(layout.temp_storage_bytes <=
                 std::numeric_limits<size_t>::max() - layout.temp_storage_offset)
      << "workspace size overflow";
  layout.total_bytes = layout.temp_storage_offset + layout.temp_storage_bytes;
  return layout;
}

void check_metadata(TensorView page_table, TensorView seq_lens) {
  CHECK_INPUT_AND_TYPE(page_table, dl_int32);
  CHECK_INPUT_AND_TYPE(seq_lens, dl_int32);
  CHECK_DEVICE(seq_lens, page_table);
  CHECK_DIM(2, page_table);
  CHECK_DIM(1, seq_lens);
  TVM_FFI_ICHECK(page_table.size(0) > 0 && page_table.size(0) <= std::numeric_limits<int>::max())
      << "block_table batch must be positive and fit in int32";
  TVM_FFI_ICHECK(page_table.size(1) > 0 && page_table.size(1) <= IndexerGemmTraits::kMaximumPages)
      << "max_pages must be in [1, " << IndexerGemmTraits::kMaximumPages << "]";
  TVM_FFI_ICHECK(seq_lens.size(0) == page_table.size(0)) << "seq_lens must have shape [batch]";
}

void check_workspace(TensorView reference, TensorView workspace, size_t required_bytes) {
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_DIM(1, workspace);
  CHECK_DEVICE(workspace, reference);
  TVM_FFI_ICHECK(static_cast<uint64_t>(workspace.size(0)) >= required_bytes)
      << "workspace is too small: need " << required_bytes << " bytes, got " << workspace.size(0);
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(tensor_data<uint8_t>(workspace)) % alignof(int32_t) ==
                 0)
      << "workspace address must be aligned to int32";
}

} // namespace

int64_t q8kv4_indexer_workspace_size(int64_t batch_size) {
  WorkspaceLayout const layout = get_workspace_layout(checked_batch(batch_size));
  TVM_FFI_ICHECK(layout.total_bytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()))
      << "workspace size does not fit in int64";
  return static_cast<int64_t>(layout.total_bytes);
}

void q8kv4_indexer_plan(TensorView page_table, TensorView seq_lens, TensorView workspace,
                        int64_t stream_ptr) {
  check_metadata(page_table, seq_lens);
  int const batch = checked_batch(page_table.size(0));
  WorkspaceLayout const layout = get_workspace_layout(batch);
  check_workspace(page_table, workspace, layout.total_bytes);

  ffi::CUDADeviceGuard device_guard(page_table.device().device_id);
  cudaStream_t const stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  uint8_t *workspace_ptr = tensor_data<uint8_t>(workspace);
  IndexerGemmArguments arguments{};
  arguments.page_table_ptr = tensor_data<int32_t const>(page_table);
  arguments.kv_lengths_ptr = tensor_data<int32_t const>(seq_lens);
  arguments.scheduler_workspace_ptr = reinterpret_cast<int32_t *>(workspace_ptr);
  arguments.scheduler_temp_storage_ptr =
      layout.temp_storage_bytes > 0 ? workspace_ptr + layout.temp_storage_offset : nullptr;
  arguments.scheduler_temp_storage_bytes = layout.temp_storage_bytes;
  arguments.batch = batch;
  arguments.max_pages = static_cast<int>(page_table.size(1));

  cudaError_t const status = q8kv4_indexer::prepare_indexer_gemm(arguments, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Q8KV4 indexer plan failed: " << cudaGetErrorString(status);
}

void q8kv4_indexer_run(TensorView q, TensorView k_cache, TensorView page_table, TensorView seq_lens,
                       TensorView workspace, int64_t sm_count, TensorView output,
                       int64_t stream_ptr) {
  using Traits = IndexerGemmTraits;
  check_metadata(page_table, seq_lens);
  CHECK_INPUT_AND_TYPE(q, dl_float8_e4m3fn);
  CHECK_CUDA(k_cache);
  CHECK_INPUT_TYPE(k_cache, dl_uint8);
  CHECK_INPUT_AND_TYPE(output, dl_float32);
  CHECK_DEVICE(q, page_table);
  CHECK_DEVICE(k_cache, page_table);
  CHECK_DEVICE(output, page_table);

  int const batch = checked_batch(page_table.size(0));
  int const max_pages = static_cast<int>(page_table.size(1));
  TVM_FFI_ICHECK(sm_count > 0 && sm_count <= std::numeric_limits<int>::max())
      << "sm_count must be positive and fit in int32";
  TVM_FFI_ICHECK(q.ndim() == 3 && q.size(0) == static_cast<int64_t>(batch) * Traits::kQueryLength &&
                 q.size(1) == 1 && q.size(2) == Traits::kHeadDim)
      << "q must have shape [batch * 8, 1, 128]";
  TVM_FFI_ICHECK(k_cache.ndim() == 3 && k_cache.size(0) > 0 &&
                 k_cache.size(0) <= std::numeric_limits<int>::max() &&
                 k_cache.size(1) == Traits::kPageTokens &&
                 k_cache.size(2) * Traits::kPageTokens == Traits::kPageBytes)
      << "k_cache must have shape [num_blocks, 128, 72]";
  TVM_FFI_ICHECK(k_cache.stride(2) == 1 && k_cache.stride(1) == k_cache.size(2))
      << "k_cache pages must be contiguous";
  TVM_FFI_ICHECK(k_cache.stride(0) >= Traits::kPageBytes && k_cache.stride(0) % 16 == 0)
      << "k_cache page stride must be at least 9216 bytes and 16-byte aligned";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(tensor_data<uint8_t>(k_cache)) % 16 == 0)
      << "k_cache data pointer must be 16-byte aligned";
  TVM_FFI_ICHECK(output.ndim() == 3 && output.size(0) == batch &&
                 output.size(1) == Traits::kQueryLength && output.size(2) == max_pages)
      << "scores must have shape [batch, 8, max_pages]";
  check_workspace(page_table, workspace, scheduler_workspace_bytes(batch));

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  IndexerGemmArguments arguments{};
  arguments.q_ptr = tensor_data<void const>(q);
  arguments.k_cache_ptr = tensor_data<void const>(k_cache);
  arguments.page_table_ptr = tensor_data<int32_t const>(page_table);
  arguments.kv_lengths_ptr = tensor_data<int32_t const>(seq_lens);
  arguments.scheduler_workspace_ptr = reinterpret_cast<int32_t *>(tensor_data<uint8_t>(workspace));
  arguments.output_ptr = tensor_data<float>(output);
  arguments.batch = batch;
  arguments.max_pages = max_pages;
  arguments.physical_pages = static_cast<int>(k_cache.size(0));
  arguments.page_stride_bytes = k_cache.stride(0);
  arguments.sm_count = static_cast<int>(sm_count);

  cudaError_t const status =
      q8kv4_indexer::launch_indexer_gemm(arguments, reinterpret_cast<cudaStream_t>(stream_ptr));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Q8KV4 indexer run failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(q8kv4_indexer_workspace_size, q8kv4_indexer_workspace_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(q8kv4_indexer_plan, q8kv4_indexer_plan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(q8kv4_indexer_run, q8kv4_indexer_run);
