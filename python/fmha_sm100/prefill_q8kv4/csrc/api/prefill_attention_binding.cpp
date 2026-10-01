// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#include <torch/extension.h>

#include "prefill_attention_api.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "run", &fmha_sm100::prefill_q8kv4::prefill_run,
      "SM100 Q8KV4 paged sparse-prefill K1");
  module.def("block_scale_shift", &fmha_sm100::prefill_q8kv4::block_scale_shift,
             "Block-scale shift this extension was built with");
}
