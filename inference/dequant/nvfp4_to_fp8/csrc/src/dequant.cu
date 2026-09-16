#include "sm100/device/dequant.hpp"

#include <algorithm>
#include <cstdint>

#include "sm100/common/dequant_config.hpp"
#include "sm100/kernel/nvfp4_to_fp8_kernel.cuh"

namespace minimax::inference::dequant::sm100 {

cudaError_t launch_dense_nvfp4_to_fp8(uint8_t const *packed_nvfp4, uint8_t const *scale,
                                      uint8_t *output, int64_t rows, int sm_count,
                                      cudaStream_t stream) {
  int64_t const required_blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
  int64_t const resident_blocks = static_cast<int64_t>(sm_count) * 8;
  int const grid = static_cast<int>(std::min(required_blocks, resident_blocks));
  dense_nvfp4_to_fp8_kernel<<<grid, kThreads, 0, stream>>>(packed_nvfp4, scale, output, rows);
  return cudaGetLastError();
}

cudaError_t launch_sparse_paged_nvfp4_to_fp8(uint8_t const *packed_k, uint8_t const *packed_v,
                                             uint8_t const *k_scale, uint8_t const *v_scale,
                                             int64_t const *pair_keys, uint8_t *output_k,
                                             uint8_t *output_v, int64_t pair_count,
                                             int num_kv_heads, int page_size, cudaStream_t stream) {
  sparse_paged_nvfp4_to_fp8_kernel<<<static_cast<unsigned int>(pair_count), kThreads, 0, stream>>>(
      packed_k, packed_v, k_scale, v_scale, pair_keys, output_k, output_v, pair_count, num_kv_heads,
      page_size);
  return cudaGetLastError();
}

char const *nvfp4_to_fp8_backend() {
#if MINIMAX_DEQUANT_HAS_QMUL4
  return "qmul4";
#else
  return "fp16_fallback";
#endif
}

} // namespace minimax::inference::dequant::sm100
