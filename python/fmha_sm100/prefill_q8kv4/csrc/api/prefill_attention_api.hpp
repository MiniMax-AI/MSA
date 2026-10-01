// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <torch/extension.h>

namespace fmha_sm100::prefill_q8kv4 {

void prefill_run(torch::Tensor q,
                 torch::Tensor packed_k,
                 torch::Tensor packed_v,
                 torch::Tensor k_scale,
                 torch::Tensor v_scale,
                 torch::Tensor kv_indices,
                 c10::optional<torch::Tensor> kv_indptr,
                 torch::Tensor cu_seqlens_q,
                 torch::Tensor cu_seqlens_k,
                 c10::optional<torch::Tensor> seqused_k,
                 torch::Tensor k2q_row_ptr,
                 torch::Tensor qsplit_indices,
                 torch::Tensor scheduler_metadata,
                 torch::Tensor work_count,
                 torch::Tensor o_partial,
                 torch::Tensor lse_partial,
                 c10::optional<torch::Tensor> k_global_scale,
                 c10::optional<torch::Tensor> v_global_scale,
                 double softmax_scale,
                 double output_scale);

// The block-scale shift this extension was built with.
int64_t block_scale_shift();

}  // namespace fmha_sm100::prefill_q8kv4
