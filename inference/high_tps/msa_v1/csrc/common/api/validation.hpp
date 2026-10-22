#pragma once

#include "common/api/types.hpp"

namespace mega_atten::m3_decode {
int64_t compute_mxfp8_sf_size(int64_t m, int64_t k);

bool is_fp8_byte_storage_tensor(at::Tensor const &tensor);

bool tensors_overlap(at::Tensor const &lhs, at::Tensor const &rhs);

} // namespace mega_atten::m3_decode
