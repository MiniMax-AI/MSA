#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace minimax::msa_v1::indexer::decode::tp4_q8kv4 {

struct IndexerGemmArguments {
  void const* q_ptr = nullptr;
  void const* packed_k_ptr = nullptr;
  void const* k_scale_ptr = nullptr;
  int32_t const* page_table_ptr = nullptr;
  int32_t const* kv_lengths_ptr = nullptr;
  int32_t* scheduler_workspace_ptr = nullptr;
  void* scheduler_temp_storage_ptr = nullptr;
  float* output_ptr = nullptr;
  size_t scheduler_temp_storage_bytes = 0;
  int batch = 0;
  int max_pages = 0;
  int physical_pages = 0;
  int sm_count = 0;
};

cudaError_t prepare_indexer_gemm(IndexerGemmArguments const& arguments,
                                 cudaStream_t stream);

cudaError_t launch_indexer_gemm(IndexerGemmArguments const& arguments,
                                cudaStream_t stream);

cudaError_t get_indexer_gemm_scheduler_temp_storage_bytes(
    int batch, size_t& temp_storage_bytes);

cudaError_t run_indexer_gemm_scheduler_scan(
    int32_t* prefix, int batch, void* temp_storage,
    size_t temp_storage_bytes, cudaStream_t stream);

}  // namespace minimax::msa_v1::indexer::decode::tp4_q8kv4
