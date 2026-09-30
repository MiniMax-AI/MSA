#pragma once

#include <torch/extension.h>

#include <cstdint>

namespace minimax::msa_v1::indexer::decode::q8kv4 {

torch::Tensor indexer_gemm_run(torch::Tensor q, torch::Tensor packed_k_cache, torch::Tensor k_scale,
                               torch::Tensor page_table, torch::Tensor seq_lens,
                               torch::Tensor workspace, int64_t sm_count, torch::Tensor output);

} // namespace minimax::msa_v1::indexer::decode::q8kv4
