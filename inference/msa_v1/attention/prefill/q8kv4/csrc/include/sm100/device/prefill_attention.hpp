#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "cutlass/bfloat16.h"

namespace minimax::msa_v1::attention::prefill::q8kv4 {

struct PrefillArguments {
  uint8_t const *q_ptr = nullptr;
  uint8_t const *packed_k_ptr = nullptr;
  uint8_t const *packed_v_ptr = nullptr;
  uint8_t const *k_scale_ptr = nullptr;
  uint8_t const *v_scale_ptr = nullptr;
  int32_t const *page_table_ptr = nullptr;
  int32_t const *cu_seqlens_q_ptr = nullptr;
  int32_t const *cu_seqlens_k_ptr = nullptr;
  int32_t const *k2q_row_ptr = nullptr;
  int32_t const *qsplit_indices_ptr = nullptr;
  int32_t const *scheduler_metadata_ptr = nullptr;
  int32_t const *work_count_ptr = nullptr;
  cutlass::bfloat16_t *o_partial_ptr = nullptr;
  float *lse_partial_ptr = nullptr;

  int total_q = 0;
  int num_q_heads = 0;
  int num_kv_heads = 0;
  int physical_pages = 0;
  int max_pages = 0;
  int total_rows = 0;
  int qsplit_stride = 0;
  int work_capacity = 0;
  float softmax_scale_log2 = 0.0f;
};

cudaError_t launch_prefill_attention(PrefillArguments const &arguments, cudaStream_t stream);

} // namespace minimax::msa_v1::attention::prefill::q8kv4
