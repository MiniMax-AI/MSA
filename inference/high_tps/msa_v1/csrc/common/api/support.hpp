#pragma once

#include <tvm/ffi/container/tensor.h>

#include "common/api/types.hpp"

namespace mega_atten::m3_decode {
void initialize();
void ensure_initialized();
tvm::ffi::Tensor torch_to_tvm(at::Tensor tensor);
tvm::ffi::Tensor tensor_or_null(at::Tensor const &tensor);
void validate_hidden_state(at::Tensor const &tensor, at::Device const &device, char const *name,
                           int64_t tokens = 8);
PlanInfo m3_decode_plan(at::Tensor qo_segment_lens, at::Tensor kv_segment_lens, int64_t topk = 16,
                        std::optional<int> device = std::nullopt);

} // namespace mega_atten::m3_decode
