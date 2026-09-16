#pragma once

#include <torch/extension.h>

namespace minimax::msa_v1::attention::prefill::q8kv4 {

void prefill_run(torch::Tensor q,
                 torch::Tensor packed_k,
                 torch::Tensor packed_v,
                 torch::Tensor k_scale,
                 torch::Tensor v_scale,
                 torch::Tensor page_table,
                 torch::Tensor cu_seqlens_q,
                 torch::Tensor cu_seqlens_k,
                 torch::Tensor k2q_row_ptr,
                 torch::Tensor qsplit_indices,
                 torch::Tensor scheduler_metadata,
                 torch::Tensor work_count,
                 torch::Tensor o_partial,
                 torch::Tensor lse_partial,
                 double softmax_scale);

}  // namespace minimax::msa_v1::attention::prefill::q8kv4
