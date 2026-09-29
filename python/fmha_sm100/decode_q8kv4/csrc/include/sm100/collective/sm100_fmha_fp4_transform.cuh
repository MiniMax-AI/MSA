// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "cute/arch/config.hpp"
#include "cute/arch/copy_sm100.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/arch/simd_sm100.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "nvfp4_to_e4m3.cuh"

#if !defined(MINIMAX_MSA_Q8KV4_BLOCK_SCALE_SHIFT)
#error "The Q8KV4 JIT must define MINIMAX_MSA_Q8KV4_BLOCK_SCALE_SHIFT"
#endif

namespace cutlass::fmha::collective {

// f16x2 {2^-shift, 2^-shift}: the block-scale staging factor (see Traits::kBlockScaleShift).
constexpr uint32_t kBlockScaleMultF16x2 =
    ((15u - MINIMAX_MSA_Q8KV4_BLOCK_SCALE_SHIFT) << 10) * 0x00010001u;

#if MINIMAX_MSA_Q8KV4_BLOCK_SCALE_SHIFT != 0
#if MINIMAX_MSA_Q8KV4_HAS_QMUL4
// Four E4M3 block scales divided by 2^shift: exact through f16, rounded back to E4M3 (scales
// below 2^(shift - 6) become subnormal, below 2^(shift - 10) zero).
CUTLASS_DEVICE
uint32_t stage_e4m3x4_block_scales(uint32_t scale_e4m3x4) {
  uint32_t staged;
  asm("{\n"
      ".reg .b16 lo, hi, staged_lo, staged_hi;\n"
      ".reg .b32 f16_lo, f16_hi;\n"
      "mov.b32 {lo, hi}, %1;\n"
      "cvt.rn.f16x2.e4m3x2 f16_lo, lo;\n"
      "cvt.rn.f16x2.e4m3x2 f16_hi, hi;\n"
      "mul.rn.f16x2 f16_lo, f16_lo, %2;\n"
      "mul.rn.f16x2 f16_hi, f16_hi, %2;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 staged_lo, f16_lo;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 staged_hi, f16_hi;\n"
      "mov.b32 %0, {staged_lo, staged_hi};\n"
      "}\n"
      : "=r"(staged)
      : "r"(scale_e4m3x4), "r"(kBlockScaleMultF16x2));
  return staged;
}
#else
CUTLASS_DEVICE
uint32_t mul_f16x2(uint32_t a, uint32_t b) {
  uint32_t product;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(product) : "r"(a), "r"(b));
  return product;
}
#endif
#endif

CUTLASS_DEVICE
float2 make_f32x2(float x, float y) {
  float2 value;
  value.x = x;
  value.y = y;
  return value;
}

CUTLASS_DEVICE
float2 fadd2(float2 a, float2 b) {
  float2 out;
  cute::add(out, a, b);
  return out;
}

CUTLASS_DEVICE
float2 fmul2(float2 a, float2 b) {
  float2 out;
  cute::mul(out, a, b);
  return out;
}

CUTLASS_DEVICE
float2 ffma2(float2 a, float2 b, float2 c) {
  float2 out;
  cute::fma(out, a, b, c);
  return out;
}

CUTLASS_DEVICE
void ldsm_unpack_fp4_transpose_16x16_x1(uint32_t &out0, uint32_t &out1, void const *smem_ptr) {
  asm volatile("ldmatrix.sync.aligned.shared::cta.m16n16.x1.trans.b8x16.b4x16_p64 "
               "{%0, %1}, [%2];"
               : "=r"(out0), "=r"(out1)
               : "l"(smem_ptr));
}

CUTLASS_DEVICE
int packed_fp4_swizzled_offset(int token, int packed_byte) {
  int const linear_segment = (token & 1) * 4 + packed_byte / 16;
  int const swizzled_segment = linear_segment ^ ((token >> 1) & 7);
  return (token >> 1) * 128 + swizzled_segment * 16 + (packed_byte & 15);
}

CUTLASS_DEVICE
uint32_t byte_permute(uint32_t lhs, uint32_t rhs, uint32_t selector) {
  return __byte_perm(lhs, rhs, selector);
}

CUTLASS_DEVICE
void duplicate_e4m3x4_scale_pairs(uint32_t &scale_dup01, uint32_t &scale_dup23,
                                  uint32_t scale_e4m3x4) {
  scale_dup01 = byte_permute(scale_e4m3x4, scale_e4m3x4, 0x1100u);
  scale_dup23 = byte_permute(scale_e4m3x4, scale_e4m3x4, 0x3322u);
}

CUTLASS_DEVICE
void deinterleave_e4m3x8_pair(uint32_t &dst0, uint32_t &dst1, uint32_t res_lo, uint32_t res_hi) {
  dst0 = byte_permute(res_lo, res_hi, 0x6420u);
  dst1 = byte_permute(res_lo, res_hi, 0x7531u);
}

#if MINIMAX_MSA_Q8KV4_HAS_QMUL4
CUTLASS_DEVICE
void convert_packed_e2m1x4_to_e4m3x4_qmul4(uint32_t &dst, uint16_t fp4x4, uint32_t scale_e4m3x4) {
  fmha_sm100::decode_q8kv4::sm100::common::nvfp4_e2m1x4_mul_e4m3x4(dst, fp4x4,
                                                                                    scale_e4m3x4);
}
#endif

#if !MINIMAX_MSA_Q8KV4_HAS_QMUL4
// FP16 fallback V path. The per-stage prepare pass converts the E4M3 scales to F16 once and
// stores them as f16x2 pairs, so the V loop multiplies directly: the four scales of a group
// arrive as two pre-converted f16x2 words ({s0, s1}, {s2, s3}); the per-token broadcasts are
// {h, h} packs so ptxas folds them into HMUL2 .H0_H0 / .H1_H1 swizzles.
CUTLASS_DEVICE
void convert_unpacked_e2m1x4_pair_to_e4m3x4_f16scales(uint32_t &dst0, uint32_t &dst1,
                                                      uint32_t fp4x4_0, uint32_t fp4x4_1,
                                                      uint32_t scale_f16x2_01,
                                                      uint32_t scale_f16x2_23) {
  uint32_t const packed = fp4x4_1 * 16u + fp4x4_0;
  asm volatile("{\n"
               ".reg .b16 scale_h0, scale_h1, scale_h2, scale_h3;\n"
               ".reg .b32 scale_f16_0, scale_f16_1, scale_f16_2, scale_f16_3;\n"
               ".reg .b8 e2m1_0, e2m1_1, e2m1_2, e2m1_3;\n"
               ".reg .b32 f16_0, f16_1, f16_2, f16_3;\n"
               ".reg .b16 e4m3_0, e4m3_1, e4m3_2, e4m3_3;\n"
               ".reg .b32 result_lo, result_hi;\n"
               "mov.b32 {scale_h0, scale_h1}, %2;\n"
               "mov.b32 {scale_h2, scale_h3}, %3;\n"
               "mov.b32 scale_f16_0, {scale_h0, scale_h0};\n"
               "mov.b32 scale_f16_1, {scale_h1, scale_h1};\n"
               "mov.b32 scale_f16_2, {scale_h2, scale_h2};\n"
               "mov.b32 scale_f16_3, {scale_h3, scale_h3};\n"
               "mov.b32 {e2m1_0, e2m1_1, e2m1_2, e2m1_3}, %4;\n"
               "cvt.rn.f16x2.e2m1x2 f16_0, e2m1_0;\n"
               "cvt.rn.f16x2.e2m1x2 f16_1, e2m1_1;\n"
               "cvt.rn.f16x2.e2m1x2 f16_2, e2m1_2;\n"
               "cvt.rn.f16x2.e2m1x2 f16_3, e2m1_3;\n"
               "mul.rn.f16x2 f16_0, f16_0, scale_f16_0;\n"
               "mul.rn.f16x2 f16_1, f16_1, scale_f16_1;\n"
               "mul.rn.f16x2 f16_2, f16_2, scale_f16_2;\n"
               "mul.rn.f16x2 f16_3, f16_3, scale_f16_3;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e4m3_0, f16_0;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e4m3_1, f16_1;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e4m3_2, f16_2;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e4m3_3, f16_3;\n"
               "mov.b32 result_lo, {e4m3_0, e4m3_1};\n"
               "mov.b32 result_hi, {e4m3_2, e4m3_3};\n"
               "prmt.b32 %0, result_lo, result_hi, 0x6420;\n"
               "prmt.b32 %1, result_lo, result_hi, 0x7531;\n"
               "}\n"
               : "=&r"(dst0), "=&r"(dst1)
               : "r"(scale_f16x2_01), "r"(scale_f16x2_23), "r"(packed));
}

// E4M3x4 scale word -> two f16x2 words ({s0,s1}, {s2,s3}).
CUTLASS_DEVICE
void convert_e4m3x4_scales_to_f16x2_pair(uint32_t &f16x2_01, uint32_t &f16x2_23,
                                         uint32_t scale_e4m3x4) {
  asm volatile("{\n"
               ".reg .b16 pair01, pair23;\n"
               "mov.b32 {pair01, pair23}, %2;\n"
               "cvt.rn.f16x2.e4m3x2 %0, pair01;\n"
               "cvt.rn.f16x2.e4m3x2 %1, pair23;\n"
               "}\n"
               : "=r"(f16x2_01), "=r"(f16x2_23)
               : "r"(scale_e4m3x4));
}
#endif

#if MINIMAX_MSA_Q8KV4_HAS_QMUL4
// QMUL4 V path: E4M3 scale word consumed directly.
CUTLASS_DEVICE
void convert_e2m1x4_pair_to_e4m3x4(uint32_t &dst0, uint32_t &dst1, uint32_t fp4x4_0,
                                   uint32_t fp4x4_1, uint32_t scale_e4m3x4) {
  uint32_t const fp4x8_interleaved = fp4x4_1 * 16u + fp4x4_0;
  uint32_t scale_dup01;
  uint32_t scale_dup23;
  duplicate_e4m3x4_scale_pairs(scale_dup01, scale_dup23, scale_e4m3x4);
  uint32_t res_lo;
  uint32_t res_hi;
  convert_packed_e2m1x4_to_e4m3x4_qmul4(res_lo, static_cast<uint16_t>(fp4x8_interleaved),
                                        scale_dup01);
  convert_packed_e2m1x4_to_e4m3x4_qmul4(res_hi, static_cast<uint16_t>(fp4x8_interleaved >> 16),
                                        scale_dup23);
  deinterleave_e4m3x8_pair(dst0, dst1, res_lo, res_hi);
}
#endif

// PRMT selector that gathers byte `byte_idx` of two scale words into a 16-bit (row0, row1) pair.
CUTLASS_DEVICE
uint32_t scale_pair_selector(int byte_idx) {
  return 0x0011u * static_cast<uint32_t>(byte_idx & 3) + 0x0040u;
}

#if MINIMAX_MSA_Q8KV4_HAS_QMUL4
// Eight consecutive E2M1 values (one 32-bit word) of two tokens sharing one head-dim group:
// scale_e4m3x2 holds (token a scale, token b scale). Produces 4 E4M3 words: a[0:4], a[4:8], b[0:4],
// b[4:8].
CUTLASS_DEVICE
void convert_e2m1x8_token_pair_to_e4m3x8(uint32_t &dst_a_lo, uint32_t &dst_a_hi, uint32_t &dst_b_lo,
                                         uint32_t &dst_b_hi, uint32_t fp4x8_a, uint32_t fp4x8_b,
                                         uint16_t scale_e4m3x2) {
  uint32_t const pair = static_cast<uint32_t>(scale_e4m3x2);
  uint32_t const scale_a = byte_permute(pair, 0u, 0x0000u);
  uint32_t const scale_b = byte_permute(pair, 0u, 0x1111u);
  convert_packed_e2m1x4_to_e4m3x4_qmul4(dst_a_lo, static_cast<uint16_t>(fp4x8_a), scale_a);
  convert_packed_e2m1x4_to_e4m3x4_qmul4(dst_a_hi, static_cast<uint16_t>(fp4x8_a >> 16), scale_a);
  convert_packed_e2m1x4_to_e4m3x4_qmul4(dst_b_lo, static_cast<uint16_t>(fp4x8_b), scale_b);
  convert_packed_e2m1x4_to_e4m3x4_qmul4(dst_b_hi, static_cast<uint16_t>(fp4x8_b >> 16), scale_b);
}
#else
// FP16 fallback of the same: one scale-pair conversion feeds all 16 elements; the per-token
// broadcasts are {h, h} packs so ptxas folds them into HMUL2 .H0_H0 / .H1_H1 swizzles.
CUTLASS_DEVICE
void convert_e2m1x8_token_pair_to_e4m3x8(uint32_t &dst_a_lo, uint32_t &dst_a_hi, uint32_t &dst_b_lo,
                                         uint32_t &dst_b_hi, uint32_t fp4x8_a, uint32_t fp4x8_b,
                                         uint16_t scale_e4m3x2) {
  asm volatile("{\n"
               ".reg .b32 scale_f16x2, scale_f16_a, scale_f16_b;\n"
               ".reg .b16 scale_ha, scale_hb;\n"
               ".reg .b8 a0, a1, a2, a3, b0, b1, b2, b3;\n"
               ".reg .b32 fa0, fa1, fa2, fa3, fb0, fb1, fb2, fb3;\n"
               ".reg .b16 ea0, ea1, ea2, ea3, eb0, eb1, eb2, eb3;\n"
               "cvt.rn.f16x2.e4m3x2 scale_f16x2, %6;\n"
#if MINIMAX_MSA_Q8KV4_BLOCK_SCALE_SHIFT != 0
               "mul.rn.f16x2 scale_f16x2, scale_f16x2, %7;\n"
#endif
               "mov.b32 {scale_ha, scale_hb}, scale_f16x2;\n"
               "mov.b32 scale_f16_a, {scale_ha, scale_ha};\n"
               "mov.b32 scale_f16_b, {scale_hb, scale_hb};\n"
               "mov.b32 {a0, a1, a2, a3}, %4;\n"
               "mov.b32 {b0, b1, b2, b3}, %5;\n"
               "cvt.rn.f16x2.e2m1x2 fa0, a0;\n"
               "cvt.rn.f16x2.e2m1x2 fa1, a1;\n"
               "cvt.rn.f16x2.e2m1x2 fa2, a2;\n"
               "cvt.rn.f16x2.e2m1x2 fa3, a3;\n"
               "cvt.rn.f16x2.e2m1x2 fb0, b0;\n"
               "cvt.rn.f16x2.e2m1x2 fb1, b1;\n"
               "cvt.rn.f16x2.e2m1x2 fb2, b2;\n"
               "cvt.rn.f16x2.e2m1x2 fb3, b3;\n"
               "mul.rn.f16x2 fa0, fa0, scale_f16_a;\n"
               "mul.rn.f16x2 fa1, fa1, scale_f16_a;\n"
               "mul.rn.f16x2 fa2, fa2, scale_f16_a;\n"
               "mul.rn.f16x2 fa3, fa3, scale_f16_a;\n"
               "mul.rn.f16x2 fb0, fb0, scale_f16_b;\n"
               "mul.rn.f16x2 fb1, fb1, scale_f16_b;\n"
               "mul.rn.f16x2 fb2, fb2, scale_f16_b;\n"
               "mul.rn.f16x2 fb3, fb3, scale_f16_b;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 ea0, fa0;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 ea1, fa1;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 ea2, fa2;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 ea3, fa3;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 eb0, fb0;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 eb1, fb1;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 eb2, fb2;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 eb3, fb3;\n"
               "mov.b32 %0, {ea0, ea1};\n"
               "mov.b32 %1, {ea2, ea3};\n"
               "mov.b32 %2, {eb0, eb1};\n"
               "mov.b32 %3, {eb2, eb3};\n"
               "}\n"
               : "=&r"(dst_a_lo), "=&r"(dst_a_hi), "=&r"(dst_b_lo), "=&r"(dst_b_hi)
               : "r"(fp4x8_a), "r"(fp4x8_b), "h"(scale_e4m3x2), "r"(kBlockScaleMultF16x2));
}
#endif

CUTLASS_DEVICE
void tmem_store_16x256b(uint32_t tmem_addr, uint32_t src0, uint32_t src1, uint32_t src2,
                        uint32_t src3) {
  cute::SM100_TMEM_STORE_16dp256b1x::copy(src0, src1, src2, src3, tmem_addr);
}

CUTLASS_DEVICE
void tmem_store_16x128b(uint32_t tmem_addr, uint32_t src0, uint32_t src1) {
  cute::SM100_TMEM_STORE_16dp128b1x::copy(src0, src1, tmem_addr);
}

CUTLASS_DEVICE
void tmem_load_16x256b_x1(uint32_t (&dst)[4], uint32_t tmem_addr) {
  cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr, dst[0], dst[1], dst[2], dst[3]);
}

CUTLASS_DEVICE
void tmem_store_16x256b_x1(uint32_t tmem_addr, uint32_t const (&src)[4]) {
  cute::SM100_TMEM_STORE_16dp256b1x::copy(src[0], src[1], src[2], src[3], tmem_addr);
}

CUTLASS_DEVICE
void tmem_load_16x256b_x2(uint32_t (&dst)[8], uint32_t tmem_addr) {
  cute::SM100_TMEM_LOAD_16dp256b2x::copy(tmem_addr, dst[0], dst[1], dst[2], dst[3], dst[4], dst[5],
                                         dst[6], dst[7]);
}

CUTLASS_DEVICE
void tmem_store_16x256b_x2(uint32_t tmem_addr, uint32_t const (&src)[8]) {
  cute::SM100_TMEM_STORE_16dp256b2x::copy(src[0], src[1], src[2], src[3], src[4], src[5], src[6],
                                          src[7], tmem_addr);
}

CUTLASS_DEVICE
void tmem_load_32x32b_x4(uint32_t (&dst)[4], uint32_t tmem_addr) {
  cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr, dst[0], dst[1], dst[2], dst[3]);
}

CUTLASS_DEVICE
void tmem_store_32x32b_x4(uint32_t tmem_addr, uint32_t const (&src)[4]) {
  cute::SM100_TMEM_STORE_32dp32b4x::copy(src[0], src[1], src[2], src[3], tmem_addr);
}

CUTLASS_DEVICE
void tmem_load_32x32b_x8(uint32_t (&dst)[8], uint32_t tmem_addr) {
  cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr, dst[0], dst[1], dst[2], dst[3], dst[4], dst[5],
                                        dst[6], dst[7]);
}

CUTLASS_DEVICE
void tmem_store_32x32b_x8(uint32_t tmem_addr, uint32_t const (&src)[8]) {
  cute::SM100_TMEM_STORE_32dp32b8x::copy(src[0], src[1], src[2], src[3], src[4], src[5], src[6],
                                         src[7], tmem_addr);
}

CUTLASS_DEVICE
void store_transposed_smem_8b_8x128(uint8_t *smem_base, uint32_t const (&src)[2],
                                    int warp_group_lane) {
  int const warp_idx = warp_group_lane >> 5;
  int const lane_idx = warp_group_lane & 31;
  int const matrix_col = (lane_idx >> 3) & 1;
  int const thread_row = lane_idx & 7;
  int const segment_col = (warp_idx * 2 + matrix_col) ^ thread_row;
  int const smem_offset = thread_row * 128 + segment_col * 16;
  cute::SM100_U8x8_STSM_T::copy(src[0], src[1],
                                *reinterpret_cast<cute::uint128_t *>(smem_base + smem_offset));
}

CUTLASS_DEVICE
void store_transposed_smem_8b_16x128(uint8_t *smem_base, uint32_t const (&src)[4],
                                     int warp_group_lane) {
  int const warp_idx = warp_group_lane >> 5;
  int const lane_idx = warp_group_lane & 31;
  int const matrix_idx = lane_idx >> 3;
  int const matrix_row = matrix_idx & 1;
  int const matrix_col = matrix_idx >> 1;
  int const thread_row = lane_idx & 7;
  int const segment_col = (warp_idx * 2 + matrix_col) ^ thread_row;
  int const smem_offset = (matrix_row * 8 + thread_row) * 128 + segment_col * 16;
  cute::SM100_U8x16_STSM_T::copy(src[0], src[1], src[2], src[3],
                                 *reinterpret_cast<cute::uint128_t *>(smem_base + smem_offset));
}

CUTLASS_DEVICE
void store_transposed_smem_16b_128x8(uint8_t *smem_base, uint32_t const (&src)[4],
                                     int warp_group_lane) {
  int const warp_idx = warp_group_lane >> 5;
  int const lane_idx = warp_group_lane & 31;
  int const slice_idx = warp_idx >> 1;
  int const matrix_col = (warp_idx & 1) * 4 + ((lane_idx >> 3) & 1);
  int const thread_row = lane_idx & 7;
  int const row_offset = thread_row * 128;
  int const slice_offset = slice_idx * 8 * 128;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 2; ++i) {
    int const segment_col = (matrix_col + i * 2) ^ thread_row;
    int const smem_offset = slice_offset + row_offset + segment_col * 16;
    cute::SM90_U16x4_STSM_T::copy(src[i * 2], src[i * 2 + 1],
                                  *reinterpret_cast<cute::uint128_t *>(smem_base + smem_offset));
  }
}

CUTLASS_DEVICE
void store_transposed_smem_16b_128x16(uint8_t *smem_base, uint32_t const (&src)[8],
                                      int warp_group_lane) {
  int const warp_idx = warp_group_lane >> 5;
  int const lane_idx = warp_group_lane & 31;
  int const slice_idx = warp_idx >> 1;
  int const warp_idx_in_slice = warp_idx & 1;
  int const matrix_idx = lane_idx >> 3;
  int const matrix_row = matrix_idx >> 1;
  int const matrix_col = warp_idx_in_slice * 4 + (matrix_idx & 1);
  int const thread_row = lane_idx & 7;
  int const row_offset = (matrix_row * 8 + thread_row) * 128;
  int const slice_offset = slice_idx * 16 * 128;

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 2; ++i) {
    int const segment_col = (matrix_col + i * 2) ^ thread_row;
    int const smem_offset = slice_offset + row_offset + segment_col * 16;
    cute::SM90_U16x8_STSM_T::copy(src[i * 4 + 0], src[i * 4 + 1], src[i * 4 + 2], src[i * 4 + 3],
                                  *reinterpret_cast<cute::uint128_t *>(smem_base + smem_offset));
  }
}

CUTLASS_DEVICE
uint32_t pack_float4_to_e4m3(float x0, float x1, float x2, float x3) {
  using Converter = cutlass::detail::NumericArrayConverterPacked4Element<
      cutlass::float_e4m3_t, float, cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
  cutlass::Array<float, 4> src = cutlass::make_Array(x0, x1, x2, x3);
  cutlass::Array<cutlass::float_e4m3_t, 4> dst = Converter::convert(src);
  return reinterpret_cast<uint32_t const &>(dst);
}

CUTLASS_DEVICE
uint32_t pack_float2_to_bfloat16(float x0, float x1) {
  using Converter =
      cutlass::NumericArrayConverter<cutlass::bfloat16_t, float, 2,
                                     cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
  cutlass::Array<float, 2> src = cutlass::make_Array(x0, x1);
  cutlass::Array<cutlass::bfloat16_t, 2> dst = Converter::convert(src);
  return reinterpret_cast<uint32_t const &>(dst);
}

CUTLASS_DEVICE
void fence_tmem_store() { cutlass::arch::fence_view_async_tmem_store(); }

CUTLASS_DEVICE
void fence_tmem_load() { cutlass::arch::fence_view_async_tmem_load(); }

} // namespace cutlass::fmha::collective
