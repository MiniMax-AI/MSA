#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "sm100/collective/nvfp4_to_fp8.hpp"
#include "sm100/common/dequant_config.hpp"

namespace minimax::inference::dequant::sm100 {

__global__ __launch_bounds__(kThreads) void dense_nvfp4_to_fp8_kernel(
    uint8_t const *__restrict__ packed_nvfp4, uint8_t const *__restrict__ scale,
    uint8_t *__restrict__ output, int64_t rows) {
  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;
  int const row_in_warp = lane >> 3;
  int const scale_group = lane & 7;
  int const source_lane = row_in_warp << 3;
  uint32_t const subgroup_mask = 0xffu << source_lane;
  int64_t row =
      (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp) * kRowsPerWarp + row_in_warp;
  int64_t const row_stride = static_cast<int64_t>(gridDim.x) * kRowsPerBlock;

  for (; row < rows; row += row_stride) {
    uint2 scale_words{0u, 0u};
    if (scale_group == 0) {
      scale_words = *reinterpret_cast<uint2 const *>(scale + row * kScaleGroups);
    }
    uint32_t const scale_word_0 = __shfl_sync(subgroup_mask, scale_words.x, source_lane);
    uint32_t const scale_word_1 = __shfl_sync(subgroup_mask, scale_words.y, source_lane);
    uint32_t const scale_word = scale_group < 4 ? scale_word_0 : scale_word_1;
    uint8_t const scale_byte =
        static_cast<uint8_t>((scale_word >> ((scale_group & 3) * 8)) & 0xffu);

    uint2 const packed =
        *reinterpret_cast<uint2 const *>(packed_nvfp4 + row * kPackedHeadDim + scale_group * 8);
    uint4 const converted = dequantize_group(packed, scale_byte);
    *reinterpret_cast<uint4 *>(output + row * kHeadDim + scale_group * 16) = converted;
  }
}

__global__ __launch_bounds__(kThreads) void sparse_paged_nvfp4_to_fp8_kernel(
    uint8_t const *__restrict__ packed_k, uint8_t const *__restrict__ packed_v,
    uint8_t const *__restrict__ k_scale, uint8_t const *__restrict__ v_scale,
    int64_t const *__restrict__ pair_keys, uint8_t *__restrict__ output_k,
    uint8_t *__restrict__ output_v, int64_t pair_count, int num_kv_heads, int page_size) {
  int64_t const pair_index = blockIdx.x;
  if (pair_index >= pair_count) {
    return;
  }

  int64_t const pair_key = pair_keys[pair_index];
  int const kv_head = static_cast<int>(pair_key % num_kv_heads);
  int64_t const physical_page = pair_key / num_kv_heads;
  int64_t const source_head_page = physical_page * num_kv_heads + kv_head;
  int64_t const output_page = pair_index + num_kv_heads - 1;
  int const groups_per_page = page_size * kScaleGroups;

  for (int group = threadIdx.x; group < groups_per_page; group += blockDim.x) {
    int const token = group / kScaleGroups;
    int const scale_group = group - token * kScaleGroups;
    int64_t const source_row = source_head_page * page_size + token;
    int64_t const output_row = output_page * page_size + token;
    int64_t const packed_offset = source_row * kPackedHeadDim + scale_group * 8;
    int64_t const scale_offset = source_row * kScaleGroups + scale_group;
    int64_t const output_offset = output_row * kHeadDim + scale_group * 16;

    uint2 const packed_k_group = *reinterpret_cast<uint2 const *>(packed_k + packed_offset);
    uint2 const packed_v_group = *reinterpret_cast<uint2 const *>(packed_v + packed_offset);
    uint4 const converted_k = dequantize_group(packed_k_group, k_scale[scale_offset]);
    uint4 const converted_v = dequantize_group(packed_v_group, v_scale[scale_offset]);
    *reinterpret_cast<uint4 *>(output_k + output_offset) = converted_k;
    *reinterpret_cast<uint4 *>(output_v + output_offset) = converted_v;
  }
}

} // namespace minimax::inference::dequant::sm100
