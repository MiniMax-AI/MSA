// Inspect the private plan in tests without extending the production Python ABI.
#include "decode_attention_api.hpp"

#include <torch/extension.h>

using minimax::msa_v1::attention::decode::q8kv4::PlanInfo;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("split_counts", [](PlanInfo const &plan) { return plan.num_kv_splits_per_row; });
}
