#pragma once

#include <torch/extension.h>

#include <cstdint>

namespace minimax::msa_v1::indexer::decode::tp4_q8kv4 {

int64_t indexer_gemm_workspace_size(int64_t batch_size);

int64_t indexer_gemm_plan(torch::Tensor page_table, torch::Tensor seq_lens,
                          torch::Tensor workspace);

torch::Tensor indexer_gemm_run(
    torch::Tensor q, torch::Tensor packed_k_cache, torch::Tensor k_scale,
    torch::Tensor page_table, torch::Tensor seq_lens,
    torch::Tensor workspace, int64_t sm_count, torch::Tensor output);

}  // namespace minimax::msa_v1::indexer::decode::tp4_q8kv4
