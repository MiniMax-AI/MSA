#pragma once

#include "attn/kernel/include/sm100_m3_decode_attention_projection_mainloop.hpp"

namespace cutlass::m3_decode::collective::attention_projection {

CUTE_DEVICE void apply_norm_and_rope(KernelParams const &params, SharedStorage const &shared,
                                     int head, int token, int lane, F32x4 &values) {
  if (head == 17) {
    return;
  }

  float local_sumsq = 0.0f;
#pragma unroll
  for (int elem = 0; elem < 4; ++elem) {
    local_sumsq += (&values.x)[elem] * (&values.x)[elem];
  }
  float const inv_rms =
      1.0f / sqrtf(warp_xor_sum(local_sumsq) / float(kHeadDim) + params.post_norm_epsilon);

  auto const *staged = head < kNumQHeads
                           ? reinterpret_cast<__nv_bfloat16 const *>(shared.input_raw)
                           : reinterpret_cast<__nv_bfloat16 const *>(shared.q_mailbox);
  F32x4 const weights =
      sm100::load_shared_bf16x4_as_fp32(reinterpret_cast<uint16_t const *>(staged) + lane * 4);
#pragma unroll
  for (int elem = 0; elem < 4; ++elem) {
    (&values.x)[elem] = ((&values.x)[elem] * inv_rms) * (1.0f + (&weights.x)[elem]);
  }

  F32x4 partner;
#pragma unroll
  for (int elem = 0; elem < 4; ++elem) {
    (&partner.x)[elem] = __shfl_xor_sync(0xffffffffu, (&values.x)[elem], 8);
  }
  if (lane < 16) {
    int rope_base = lane * 4;
    bool const first_half = lane < 8;
    if (!first_half)
      rope_base -= kRotaryDim / 2;
    F32x4 const cosine = sm100::load_shared_bf16x4_as_fp32(
        reinterpret_cast<uint16_t const *>(staged + kHeadDim + token * kRotaryDim + rope_base));
    F32x4 const sine = sm100::load_shared_bf16x4_as_fp32(reinterpret_cast<uint16_t const *>(
        staged + kHeadDim + token * kRotaryDim + rope_base + kRotaryDim / 2));
#pragma unroll
    for (int elem = 0; elem < 4; ++elem) {
      float const own = (&values.x)[elem];
      (&values.x)[elem] = first_half
                              ? own * (&cosine.x)[elem] - (&partner.x)[elem] * (&sine.x)[elem]
                              : own * (&cosine.x)[elem] + (&partner.x)[elem] * (&sine.x)[elem];
    }
  }
}

CUTE_DEVICE void store_final_row(KernelParams const &params, int head, int token, int lane,
                                 F32x4 values) {
  if (head < kNumQHeads) {
    uint32_t const packed = sm100::fp32x4_to_fp8x4(values.x, values.y, values.z, values.w);
    reinterpret_cast<uint32_t *>(
        params.q_fp8_out)[(token * kNumQHeads * kHeadDim + head * kHeadDim + lane * 4) / 4] =
        packed;
    return;
  }
  if (head == 18) {
    uint32_t const packed = sm100::fp32x4_to_fp8x4(values.x, values.y, values.z, values.w);
    reinterpret_cast<uint32_t *>(params.qi_fp8_out)[(token * kHeadDim + lane * 4) / 4] = packed;
    return;
  }

  int64_t const slot = params.slot_mapping[token];
  if (slot < 0)
    return;
  float global_scale = head == 16   ? params.k_dequant_global_scale
                       : head == 17 ? params.v_dequant_global_scale
                                    : params.ki_dequant_global_scale;
  float const inv_global_scale = 1.0f / global_scale;
  float local_amax = 0.0f;
#pragma unroll
  for (int elem = 0; elem < 4; ++elem) {
    local_amax = fmaxf(local_amax, fabsf((&values.x)[elem]));
  }
  float const group_amax = subgroup4_xor_max(local_amax);
  float const scale_float = (group_amax * inv_global_scale) / 6.0f;
  uint16_t const scale_pair = sm100::fp32x2_to_fp8e4m3x2(scale_float, scale_float);
  uint8_t const scale_byte = uint8_t(scale_pair);
  float const rounded_scale = sm100::fp8e4m3x2_to_fp32x2(scale_pair).x;
  F32x4 scaled{0.0f, 0.0f, 0.0f, 0.0f};
  if (scale_byte != 0) {
    float const inv_rounded_scale = 1.0f / rounded_scale;
#pragma unroll
    for (int elem = 0; elem < 4; ++elem) {
      (&scaled.x)[elem] = ((&values.x)[elem] * inv_global_scale) * inv_rounded_scale;
    }
  }
  uint16_t const packed = sm100::fp32x4_to_fp4x4(scaled.x, scaled.y, scaled.z, scaled.w);
  uint8_t *data = head == 16   ? params.k_data_cache
                  : head == 17 ? params.v_data_cache
                               : params.ki_data_cache;
  uint8_t *scale = head == 16   ? params.k_scale_cache
                   : head == 17 ? params.v_scale_cache
                                : params.ki_scale_cache;
  int64_t const data_page_stride = head == 16   ? params.k_data_page_stride
                                   : head == 17 ? params.v_data_page_stride
                                                : params.ki_data_page_stride;
  int64_t const scale_page_stride = head == 16   ? params.k_scale_page_stride
                                    : head == 17 ? params.v_scale_page_stride
                                                 : params.ki_scale_page_stride;
  int64_t const page = slot / kCachePageSize;
  int64_t const row = slot % kCachePageSize;
  reinterpret_cast<uint16_t *>(data + page * data_page_stride + row * (kHeadDim / 2))[lane] =
      packed;
  if ((lane & 3) == 0) {
    scale[page * scale_page_stride + row * (kHeadDim / 16) + lane / 4] = scale_byte;
  }
}

template <class Policy>
CUTE_DEVICE void process_q_owner_token(KernelParams const &params, SharedStorage &shared,
                                       BarrierStorage &barriers, HeadSplit hs, int token,
                                       int lane) {
  cute::wait_barrier(barriers.q_mailbox[token], 0);
  if (!Policy::valid_token(token, params)) {
    return;
  }

  int const dim_base = lane * 4;
  F32x4 const p0 = sm100::load_shared_fp32x4(shared.q_mailbox +
                                             (0 * kTokensPerTile + token) * kHeadDim + dim_base);
  F32x4 const p1 = sm100::load_shared_fp32x4(shared.q_mailbox +
                                             (1 * kTokensPerTile + token) * kHeadDim + dim_base);
  F32x4 const p2 = sm100::load_shared_fp32x4(shared.q_mailbox +
                                             (2 * kTokensPerTile + token) * kHeadDim + dim_base);
  F32x4 values{(p0.x + p1.x) + p2.x, (p0.y + p1.y) + p2.y, (p0.z + p1.z) + p2.z,
               (p0.w + p1.w) + p2.w};
  apply_norm_and_rope(params, shared, hs.head, token, lane, values);
  store_final_row(params, hs.head, token, lane, values);
}

} // namespace cutlass::m3_decode::collective::attention_projection
