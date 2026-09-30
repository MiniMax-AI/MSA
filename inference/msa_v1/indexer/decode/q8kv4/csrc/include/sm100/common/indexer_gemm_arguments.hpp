#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace minimax::msa_v1::indexer::decode::q8kv4 {

// Structure-of-arrays fields: request index, first page, and first segment count.
inline constexpr int kWorkerStartFields = 3;

struct IndexerGemmArguments {
  void const *q_ptr = nullptr;
  void const *packed_k_ptr = nullptr;
  void const *k_scale_ptr = nullptr;
  int32_t const *page_table_ptr = nullptr;
  int32_t const *kv_lengths_ptr = nullptr;
  int32_t *scheduler_workspace_ptr = nullptr;
  float *output_ptr = nullptr;
  int batch = 0;
  int query_length = 0;
  int max_pages = 0;
  int physical_pages = 0;
  int sm_count = 0;
};

cudaError_t launch_indexer_gemm(IndexerGemmArguments const &arguments, cudaStream_t stream);

} // namespace minimax::msa_v1::indexer::decode::q8kv4
