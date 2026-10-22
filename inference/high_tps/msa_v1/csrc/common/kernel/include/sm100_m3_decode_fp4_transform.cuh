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

namespace cutlass::m3_decode::collective {

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
void ldsm_unpack_fp4_8x16_x2(uint32_t &out0, uint32_t &out1, void const *smem_ptr) {
  cute::SM100_SU4_DU8x16_x2_LDSM_N::copy(*reinterpret_cast<cute::uint128_t const *>(smem_ptr), out0,
                                         out1);
}

CUTLASS_DEVICE
void ldsm_unpack_fp4_transpose_16x16_x1(uint32_t &out0, uint32_t &out1, void const *smem_ptr) {
  asm volatile("ldmatrix.sync.aligned.shared::cta.m16n16.x1.trans.b8x16.b4x16_p64 "
               "{%0, %1}, [%2];"
               : "=r"(out0), "=r"(out1)
               : "l"(smem_ptr));
}

CUTLASS_DEVICE
uint32_t byte_permute(uint32_t lhs, uint32_t rhs, uint32_t selector) {
  return __byte_perm(lhs, rhs, selector);
}

CUTLASS_DEVICE
void convert_e2m1x4_to_e4m3x4_qmul4(uint32_t &dst, uint32_t fp4x4, uint32_t scale_e4m3x4) {
  // LDSM expands each E2M1 value into the low nibble of a byte. QMUL4
  // consumes four adjacent nibbles in a 16-bit operand.
  uint16_t const packed =
      static_cast<uint16_t>((fp4x4 & 0x0000000fu) | ((fp4x4 >> 4) & 0x000000f0u) |
                            ((fp4x4 >> 8) & 0x00000f00u) | ((fp4x4 >> 12) & 0x0000f000u));
  asm volatile("mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, %1, %2;\n"
               : "=r"(dst)
               : "h"(packed), "r"(scale_e4m3x4));
}

template <int ByteIdx>
CUTLASS_DEVICE uint32_t duplicate_e4m3_scale_byte_const(uint32_t scale_pack) {
  static_assert(ByteIdx >= 0 && ByteIdx < 4);
  return byte_permute(scale_pack, 0u, 0x1111u * ByteIdx);
}

template <int ByteIdx>
CUTLASS_DEVICE void convert_e2m1x4_to_e4m3x4_qmul4_select_byte_const(uint32_t &dst, uint32_t fp4x4,
                                                                     uint32_t scale_pack) {
  uint32_t const scale_broadcast = duplicate_e4m3_scale_byte_const<ByteIdx>(scale_pack);
  convert_e2m1x4_to_e4m3x4_qmul4(dst, fp4x4, scale_broadcast);
}

CUTLASS_DEVICE
void convert_e2m1x4_to_e4m3x4_qmul4_b0(uint32_t &dst, uint32_t fp4x4, uint32_t scale_pack) {
  convert_e2m1x4_to_e4m3x4_qmul4_select_byte_const<0>(dst, fp4x4, scale_pack);
}

CUTLASS_DEVICE
void convert_e2m1x4_to_e4m3x4_qmul4_b1(uint32_t &dst, uint32_t fp4x4, uint32_t scale_pack) {
  convert_e2m1x4_to_e4m3x4_qmul4_select_byte_const<1>(dst, fp4x4, scale_pack);
}

CUTLASS_DEVICE
void convert_e2m1x4_to_e4m3x4_qmul4_b2(uint32_t &dst, uint32_t fp4x4, uint32_t scale_pack) {
  convert_e2m1x4_to_e4m3x4_qmul4_select_byte_const<2>(dst, fp4x4, scale_pack);
}

CUTLASS_DEVICE
void convert_e2m1x4_to_e4m3x4_qmul4_b3(uint32_t &dst, uint32_t fp4x4, uint32_t scale_pack) {
  convert_e2m1x4_to_e4m3x4_qmul4_select_byte_const<3>(dst, fp4x4, scale_pack);
}

CUTLASS_DEVICE
void convert_e2m1x4_to_e4m3x4_qmul4_select_byte(uint32_t &dst, uint32_t fp4x4, uint32_t scale_pack,
                                                int byte_idx) {
  if (byte_idx == 0) {
    convert_e2m1x4_to_e4m3x4_qmul4_b0(dst, fp4x4, scale_pack);
  } else if (byte_idx == 1) {
    convert_e2m1x4_to_e4m3x4_qmul4_b1(dst, fp4x4, scale_pack);
  } else if (byte_idx == 2) {
    convert_e2m1x4_to_e4m3x4_qmul4_b2(dst, fp4x4, scale_pack);
  } else {
    convert_e2m1x4_to_e4m3x4_qmul4_b3(dst, fp4x4, scale_pack);
  }
}

CUTLASS_DEVICE
void convert_e2m1x4_pair_to_e4m3x4(uint32_t &dst0, uint32_t &dst1, uint32_t fp4x4_0,
                                   uint32_t fp4x4_1, uint32_t scale_e4m3x4) {
  convert_e2m1x4_to_e4m3x4_qmul4(dst0, fp4x4_0, scale_e4m3x4);
  convert_e2m1x4_to_e4m3x4_qmul4(dst1, fp4x4_1, scale_e4m3x4);
}

CUTLASS_DEVICE
void convert_e2m1x4_pair_to_e4m3x4_select_byte(uint32_t &dst0, uint32_t &dst1, uint32_t fp4x4_0,
                                               uint32_t fp4x4_1, uint32_t scale_pack0,
                                               uint32_t scale_pack1, int byte_idx) {
  convert_e2m1x4_to_e4m3x4_qmul4_select_byte(dst0, fp4x4_0, scale_pack0, byte_idx);
  convert_e2m1x4_to_e4m3x4_qmul4_select_byte(dst1, fp4x4_1, scale_pack1, byte_idx);
}

CUTLASS_DEVICE
void load_e4m3_scale_pair_token_stride4(uint32_t &scale0, uint32_t &scale1, uint8_t const *base,
                                        int group_pair) {
  int const word_offset = group_pair & 4;
  bool const upper_pair = (group_pair & 2) != 0;
  uint32_t const row0 = *reinterpret_cast<uint32_t const *>(base + 0 * 8 + word_offset);
  uint32_t const row1 = *reinterpret_cast<uint32_t const *>(base + 1 * 8 + word_offset);
  uint32_t const row2 = *reinterpret_cast<uint32_t const *>(base + 2 * 8 + word_offset);
  uint32_t const row3 = *reinterpret_cast<uint32_t const *>(base + 3 * 8 + word_offset);
  uint32_t const selector = upper_pair ? 0x7362u : 0x5140u;
  uint32_t const permuted_rows01 = byte_permute(row0, row1, selector);
  uint32_t const permuted_rows23 = byte_permute(row2, row3, selector);

  scale0 = byte_permute(permuted_rows01, permuted_rows23, 0x5410u);
  scale1 = byte_permute(permuted_rows01, permuted_rows23, 0x7632u);
}

CUTLASS_DEVICE
void tmem_store_16x128b(uint32_t tmem_addr, uint32_t src0, uint32_t src1) {
  cute::SM100_TMEM_STORE_16dp128b1x::copy(src0, src1, tmem_addr);
}

CUTLASS_DEVICE
void tmem_load_16x256b(uint32_t (&dst)[4], uint32_t tmem_addr) {
  cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr, dst[0], dst[1], dst[2], dst[3]);
}

CUTLASS_DEVICE
void tmem_load_16x256b_x2(uint32_t (&dst)[8], uint32_t tmem_addr) {
  cute::SM100_TMEM_LOAD_16dp256b2x::copy(tmem_addr, dst[0], dst[1], dst[2], dst[3], dst[4], dst[5],
                                         dst[6], dst[7]);
}

CUTLASS_DEVICE
void tmem_store_16x256b(uint32_t tmem_addr, uint32_t const (&src)[4]) {
  cute::SM100_TMEM_STORE_16dp256b1x::copy(src[0], src[1], src[2], src[3], tmem_addr);
}

CUTLASS_DEVICE
void tmem_store_16x256b_x2(uint32_t tmem_addr, uint32_t const (&src)[8]) {
  cute::SM100_TMEM_STORE_16dp256b2x::copy(src[0], src[1], src[2], src[3], src[4], src[5], src[6],
                                          src[7], tmem_addr);
}

CUTLASS_DEVICE
void tmem_load_32x32b(uint32_t &dst, uint32_t tmem_addr) {
  cute::SM100_TMEM_LOAD_32dp32b1x::copy(tmem_addr, dst);
}

CUTLASS_DEVICE
void tmem_load_32x32b_x8(uint32_t (&dst)[8], uint32_t tmem_addr) {
  cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr, dst[0], dst[1], dst[2], dst[3], dst[4], dst[5],
                                        dst[6], dst[7]);
}

CUTLASS_DEVICE
void tmem_store_32x32b(uint32_t tmem_addr, uint32_t src) {
  cute::SM100_TMEM_STORE_32dp32b1x::copy(src, tmem_addr);
}

CUTLASS_DEVICE
void tmem_store_32x32b_x8(uint32_t tmem_addr, uint32_t const (&src)[8]) {
  cute::SM100_TMEM_STORE_32dp32b8x::copy(src[0], src[1], src[2], src[3], src[4], src[5], src[6],
                                         src[7], tmem_addr);
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
uint2 pack_float8_to_e4m3x8(cutlass::Array<float, 8> const &src) {
  using Converter =
      cutlass::NumericArrayConverter<cutlass::float_e4m3_t, float, 8,
                                     cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
  cutlass::Array<cutlass::float_e4m3_t, 8> dst = Converter::convert(src);
  return reinterpret_cast<uint2 const &>(dst);
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

} // namespace cutlass::m3_decode::collective
