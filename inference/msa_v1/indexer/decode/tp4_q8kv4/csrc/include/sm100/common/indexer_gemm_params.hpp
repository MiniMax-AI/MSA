#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "sm100/common/indexer_gemm_arguments.hpp"

namespace minimax::msa_v1::indexer::decode::tp4_q8kv4 {

struct alignas(128) IndexerGemmParams {
  CUtensorMap packed_k{};
  CUtensorMap k_scale{};
  uint8_t const* q_ptr = nullptr;
  int32_t const* page_table_ptr = nullptr;
  int32_t const* kv_lengths_ptr = nullptr;
  int32_t* scheduler_workspace_ptr = nullptr;
  float* output_ptr = nullptr;
  int batch = 0;
  int max_pages = 0;
  int sm_count = 0;
};

}  // namespace minimax::msa_v1::indexer::decode::tp4_q8kv4
