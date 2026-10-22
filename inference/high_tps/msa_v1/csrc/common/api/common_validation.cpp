#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <utility>

#include "attn/kernel/include/m3_decode_output_projection_handoff.hpp"
#include "common/api/validation.hpp"
#include "common/kernel/include/tp4_nvls_layout.hpp"

namespace mega_atten::m3_decode {

int64_t compute_mxfp8_sf_size(int64_t m, int64_t k) {
  TORCH_CHECK(k % 32 == 0, "MXFP8 O quantization requires head_dim divisible by 32, got ", k);
  int64_t sf_cols = k / 32;
  int64_t padded_m = ((m + 127) / 128) * 128;
  int64_t padded_sf_cols = ((sf_cols + 3) / 4) * 4;
  return padded_m * padded_sf_cols;
}

bool is_fp8_byte_storage_tensor(const at::Tensor &t) {
  return t.scalar_type() == at::kFloat8_e4m3fn || t.scalar_type() == at::kByte;
}

// Byte extent [begin, end) spanned by a tensor; strided views (e.g. page-strided caches) span
// more than numel() * element_size() bytes.
static std::pair<uintptr_t, uintptr_t> tensor_byte_extent(at::Tensor const &tensor) {
  uintptr_t const begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  if (tensor.numel() == 0) {
    return {begin, begin};
  }
  int64_t last = 0;
  for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
    last += (tensor.size(dim) - 1) * tensor.stride(dim);
  }
  return {begin, begin + static_cast<uintptr_t>((last + 1) * tensor.element_size())};
}

bool tensors_overlap(at::Tensor const &lhs, at::Tensor const &rhs) {
  if (!lhs.defined() || !rhs.defined()) {
    return false;
  }
  auto const [lhs_begin, lhs_end] = tensor_byte_extent(lhs);
  auto const [rhs_begin, rhs_end] = tensor_byte_extent(rhs);
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

} // namespace mega_atten::m3_decode
