#pragma once

#include <torch/extension.h>

namespace minimax::msa_v1::indexer::topk {

torch::Tensor indexer_topk_run(torch::Tensor scores, torch::Tensor lengths,
                               torch::Tensor output);

}  // namespace minimax::msa_v1::indexer::topk
