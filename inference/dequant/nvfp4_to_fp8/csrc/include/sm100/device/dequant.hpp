#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace minimax::inference::dequant::sm100 {

cudaError_t launch_dense_nvfp4_to_fp8(uint8_t const *packed_nvfp4, uint8_t const *scale,
                                      uint8_t *output, int64_t rows, int sm_count,
                                      cudaStream_t stream);

cudaError_t launch_sparse_paged_nvfp4_to_fp8(uint8_t const *packed_k, uint8_t const *packed_v,
                                             uint8_t const *k_scale, uint8_t const *v_scale,
                                             int64_t const *pair_keys, uint8_t *output_k,
                                             uint8_t *output_v, int64_t pair_count,
                                             int num_kv_heads, int page_size, cudaStream_t stream);

char const *nvfp4_to_fp8_backend();

} // namespace minimax::inference::dequant::sm100
