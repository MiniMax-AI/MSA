// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "cute/arch/config.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "sm100/common/nvfp4_to_e4m3.cuh"

namespace fmha_sm100::prefill_q8kv4::detail {

CUTLASS_DEVICE void dequant_e2m1x8_uniform_scale(
    uint32_t& output_lo, uint32_t& output_hi,
    uint32_t packed_fp4, uint32_t scale_e4m3x4) {
  ::fmha_sm100::prefill_q8kv4::sm100::common::
      nvfp4_to_e4m3x4(
      output_lo, static_cast<uint16_t>(packed_fp4), scale_e4m3x4);
  ::fmha_sm100::prefill_q8kv4::sm100::common::
      nvfp4_to_e4m3x4(
      output_hi, static_cast<uint16_t>(packed_fp4 >> 16), scale_e4m3x4);
}

struct Fp4DequantInput {
  uint32_t packed_fp4;
  uint32_t scale_e4m3x4;
};

// One E4M3 block scale replicated into the four lanes of a QMUL4 operand. With block-scale
// staging the scale is divided by 2^shift first: exact through f16, rounded back to E4M3 (scales
// below 2^(shift - 6) become subnormal, below 2^(shift - 10) zero), the decode kernel's rounding.
CUTLASS_DEVICE uint32_t replicate_block_scale(uint8_t scale_e4m3) {
#if FMHA_SM100_PREFILL_Q8KV4_BLOCK_SCALE_SHIFT == 0
  return static_cast<uint32_t>(scale_e4m3) * 0x01010101u;
#else
  constexpr uint32_t kStageF16x2 =
      ((15u - FMHA_SM100_PREFILL_Q8KV4_BLOCK_SCALE_SHIFT) << 10) * 0x00010001u;
  uint16_t const pair = static_cast<uint16_t>(scale_e4m3 * 0x0101u);
  uint16_t staged;
  asm("{\n"
      ".reg .b32 f16x2;\n"
      "cvt.rn.f16x2.e4m3x2 f16x2, %1;\n"
      "mul.rn.f16x2 f16x2, f16x2, %2;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 %0, f16x2;\n"
      "}\n"
      : "=h"(staged)
      : "h"(pair), "r"(kStageF16x2));
  return static_cast<uint32_t>(staged) * 0x00010001u;
#endif
}

CUTLASS_DEVICE Fp4DequantInput load_fp4_dequant_input(
    uint8_t const* packed_fp4, uint8_t scale_e4m3) {
  return {
      *reinterpret_cast<uint32_t const*>(packed_fp4),
      replicate_block_scale(scale_e4m3),
  };
}

CUTLASS_DEVICE void dequant_store_fp8_smem(
    Fp4DequantInput const& input, uint32_t output_address) {
  uint32_t output_lo;
  uint32_t output_hi;
  dequant_e2m1x8_uniform_scale(
      output_lo, output_hi, input.packed_fp4, input.scale_e4m3x4);
  asm volatile("st.shared.v2.b32 [%0], {%1, %2};"
               :
               : "r"(output_address), "r"(output_lo), "r"(output_hi));
}

// Scales are linear (byte token * 8 + group, the K layout) or, with TokenQuadScales, in the
// token-quad order (byte (token / 4) * 32 + group * 4 + token % 4) that vLLM writes for V.
//
// Rows [valid_rows, NumRows) are past the request's KV length and are dequantized with a zero
// scale, so they come out as exact zeros. For V this is required: those tokens have P = 0, but the
// PV MMA still multiplies them, and the cache bytes there are unspecified (vLLM reuses blocks
// across cache groups of different formats), so a NaN block scale (0x7F / 0xFF) would give
// 0 x NaN = NaN.
template <int NumRows, int HeadDim, int ScaleGroupSize, bool TokenQuadScales>
CUTLASS_DEVICE void dequant_fp4_tile_to_fp8_smem(
    uint8_t const* packed_fp4, uint8_t const* scale,
    uint8_t* output_smem, int thread_idx, int valid_rows = NumRows) {
  static_assert(NumRows == 128);
  static_assert(HeadDim == 128);
  static_assert(ScaleGroupSize == 16);

  constexpr int kNumThreads = 128;
  constexpr int kElementsPerThread = 8;
  constexpr int kThreadsPerRow = HeadDim / kElementsPerThread;
  constexpr int kRowsPerIteration = kNumThreads / kThreadsPerRow;
  constexpr int kRowIterations = NumRows / kRowsPerIteration;
  constexpr int kDataRowStride = HeadDim / 2;
  constexpr int kScaleRowStride = HeadDim / ScaleGroupSize;
  constexpr int kDataIterationStride =
      kDataRowStride * kRowsPerIteration;
  constexpr int kScaleIterationStride =
      kScaleRowStride * kRowsPerIteration;
  constexpr int kSwizzleAtomBytes = 8 * 128;
  constexpr int kLinearRowStep =
      (kRowsPerIteration / 8) * kSwizzleAtomBytes;

  static_assert(kThreadsPerRow == 16);
  static_assert(kRowsPerIteration == 8);
  static_assert(kRowIterations == 16);
  // Eight rows are two whole token quads, so both scale orders advance 64 bytes per iteration.
  static_assert(kScaleIterationStride == (kRowsPerIteration / 4) * 32);

  int const row_in_iteration = thread_idx / kThreadsPerRow;
  int const lane_in_row = thread_idx % kThreadsPerRow;
  int const d_base = lane_in_row * kElementsPerThread;
  int const fp4_byte_base = d_base / 2;
  int const scale_group = d_base / ScaleGroupSize;

  int const initial_linear =
      d_base + (row_in_iteration % 8) * 128 +
      (row_in_iteration / 8) * kSwizzleAtomBytes;
  int const xor_offset = (initial_linear & 0x380) >> 3;
  uint32_t output_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(output_smem)) +
      (initial_linear ^ xor_offset);

  uint8_t const* current_data =
      packed_fp4 + row_in_iteration * kDataRowStride + fp4_byte_base;
  int const scale_offset =
      TokenQuadScales
          ? (row_in_iteration / 4) * 32 + scale_group * 4 + row_in_iteration % 4
          : row_in_iteration * kScaleRowStride + scale_group;
  uint8_t const* current_scale = scale + scale_offset;
  // Scale byte of the row `iterations_ahead` iterations past the current one, 0 past valid_rows.
  int row = row_in_iteration;
  auto row_scale = [&](int iterations_ahead) -> uint8_t {
    return row + iterations_ahead * kRowsPerIteration < valid_rows
               ? current_scale[iterations_ahead * kScaleIterationStride]
               : uint8_t{0};
  };

  Fp4DequantInput input_a = load_fp4_dequant_input(current_data, row_scale(0));
  Fp4DequantInput input_b = load_fp4_dequant_input(
      current_data + kDataIterationStride, row_scale(1));
  Fp4DequantInput input_c = load_fp4_dequant_input(
      current_data + 2 * kDataIterationStride, row_scale(2));

  CUTLASS_PRAGMA_UNROLL
  for (int iteration = 0; iteration < kRowIterations - 3; ++iteration) {
    Fp4DequantInput input_d = load_fp4_dequant_input(
        current_data + 3 * kDataIterationStride, row_scale(3));
    dequant_store_fp8_smem(input_a, output_address);
    output_address += kLinearRowStep;
    current_data += kDataIterationStride;
    current_scale += kScaleIterationStride;
    row += kRowsPerIteration;
    input_a = input_b;
    input_b = input_c;
    input_c = input_d;
  }

  dequant_store_fp8_smem(input_a, output_address);
  dequant_store_fp8_smem(
      input_b, output_address + kLinearRowStep);
  dequant_store_fp8_smem(
      input_c, output_address + 2 * kLinearRowStep);
}

template <int NumRows, int HeadDim>
CUTLASS_DEVICE void clear_fp8_tile_smem(
    uint8_t* output_smem, int thread_idx) {
  constexpr int kNumThreads = 128;
  constexpr int kElementsPerThread = 8;
  constexpr int kThreadsPerRow = HeadDim / kElementsPerThread;
  constexpr int kRowsPerIteration = kNumThreads / kThreadsPerRow;
  constexpr int kRowIterations = NumRows / kRowsPerIteration;
  constexpr int kSwizzleAtomBytes = 8 * 128;
  constexpr int kLinearRowStep =
      (kRowsPerIteration / 8) * kSwizzleAtomBytes;

  int const row_in_iteration = thread_idx / kThreadsPerRow;
  int const lane_in_row = thread_idx % kThreadsPerRow;
  int const d_base = lane_in_row * kElementsPerThread;
  int const initial_linear =
      d_base + (row_in_iteration % 8) * 128 +
      (row_in_iteration / 8) * kSwizzleAtomBytes;
  int const xor_offset = (initial_linear & 0x380) >> 3;
  uint32_t output_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(output_smem)) +
      (initial_linear ^ xor_offset);
  uint32_t constexpr kZero = 0;

  CUTLASS_PRAGMA_UNROLL
  for (int iteration = 0; iteration < kRowIterations; ++iteration) {
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};"
                 :
                 : "r"(output_address), "r"(kZero), "r"(kZero));
    output_address += kLinearRowStep;
  }
  cutlass::arch::fence_view_async_shared();
}

}  // namespace fmha_sm100::prefill_q8kv4::detail
