#pragma once

#include <cstdint>

#include "cute/arch/config.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "sm100/common/nvfp4_to_e4m3.cuh"

namespace minimax::msa_v1::attention::prefill::q8kv4::detail {

CUTLASS_DEVICE void dequant_e2m1x8_uniform_scale(
    uint32_t& output_lo, uint32_t& output_hi,
    uint32_t packed_fp4, uint32_t scale_e4m3x4) {
  ::minimax::msa_v1::attention::prefill::q8kv4::sm100::common::
      nvfp4_to_e4m3x4(
      output_lo, static_cast<uint16_t>(packed_fp4), scale_e4m3x4);
  ::minimax::msa_v1::attention::prefill::q8kv4::sm100::common::
      nvfp4_to_e4m3x4(
      output_hi, static_cast<uint16_t>(packed_fp4 >> 16), scale_e4m3x4);
}

struct Fp4DequantInput {
  uint32_t packed_fp4;
  uint32_t scale_e4m3x4;
};

CUTLASS_DEVICE Fp4DequantInput load_fp4_dequant_input(
    uint8_t const* packed_fp4, uint8_t scale_e4m3) {
  return {
      *reinterpret_cast<uint32_t const*>(packed_fp4),
      static_cast<uint32_t>(scale_e4m3) * 0x01010101u,
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

template <int NumRows, int HeadDim, int ScaleGroupSize>
CUTLASS_DEVICE void dequant_fp4_tile_to_fp8_smem(
    uint8_t const* packed_fp4, uint8_t const* scale,
    uint8_t* output_smem, int thread_idx) {
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
  uint8_t const* current_scale =
      scale + row_in_iteration * kScaleRowStride;

  Fp4DequantInput input_a = load_fp4_dequant_input(
      current_data, current_scale[scale_group]);
  Fp4DequantInput input_b = load_fp4_dequant_input(
      current_data + kDataIterationStride,
      current_scale[scale_group + kScaleIterationStride]);
  Fp4DequantInput input_c = load_fp4_dequant_input(
      current_data + 2 * kDataIterationStride,
      current_scale[scale_group + 2 * kScaleIterationStride]);

  CUTLASS_PRAGMA_UNROLL
  for (int iteration = 0; iteration < kRowIterations - 3; ++iteration) {
    Fp4DequantInput input_d = load_fp4_dequant_input(
        current_data + 3 * kDataIterationStride,
        current_scale[scale_group + 3 * kScaleIterationStride]);
    dequant_store_fp8_smem(input_a, output_address);
    output_address += kLinearRowStep;
    current_data += kDataIterationStride;
    current_scale += kScaleIterationStride;
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

}  // namespace minimax::msa_v1::attention::prefill::q8kv4::detail
