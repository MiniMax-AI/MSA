#include <torch/extension.h>

#include "prefill_attention_api.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "run", &minimax::msa_v1::attention::prefill::q8kv4::prefill_run,
      "SM100 Q8KV4 paged sparse-prefill K1");
}
