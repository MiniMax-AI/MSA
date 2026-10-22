#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>

#include "attn/kernel/include/sm100_m3_decode_problem.hpp"
#include "cutlass/cutlass.h"
#include "moe_mega/kernel/include/sm100_m3_decode_router_traits.hpp"

namespace cutlass::m3_decode::collective {

template <class Traits> struct Sm100M3DecodePostAttentionNorm {
  using Config = Sm100M3DecodeRouterTraits;
  using Params = Sm100M3DecodeKernelParams<Traits>;

  static constexpr int kScaleVectorSize = 32;
  static constexpr int kScaleColumns = Config::kHiddenSize / kScaleVectorSize;
  static constexpr int kScaleColumnsPerCta = kScaleColumns / Config::kClusterCtas;
  static constexpr int kScalePaddedRows = 128;
  static constexpr int kScalePaddedColumns = kScaleColumns;
  static constexpr int kQuantWarps = Traits::kNumThreads / cutlass::NumThreadsPerWarp - 1;

  static_assert(Config::kHiddenSize % kScaleVectorSize == 0,
                "Shared activation hidden size must divide into MXFP8 blocks.");
  static_assert(kScaleColumns % Config::kClusterCtas == 0,
                "Shared activation scale columns must divide across the cluster.");
  static_assert(Config::kHiddenPerCta == kScaleColumnsPerCta * kScaleVectorSize,
                "Each CTA must own complete MXFP8 scale blocks.");
  static_assert(kQuantWarps == 15, "Post-Attention Norm expects fifteen worker warps.");

  CUTE_DEVICE static float warp_max(float value) {
    CUTLASS_PRAGMA_UNROLL
    for (int offset = cutlass::NumThreadsPerWarp / 2; offset > 0; offset /= 2) {
      value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
  }

  CUTE_DEVICE static uint8_t float_to_ue8m0_rp(float value) {
    if (!(value > 0.0f)) {
      return 0;
    }
    uint32_t const magnitude = __float_as_uint(value) & 0x7fffffffu;
    uint32_t exponent = (magnitude + 0x7fffffu) >> 23;
    exponent = magnitude <= 0x400000u ? 0u : exponent;
    return static_cast<uint8_t>(min(exponent, 254u));
  }

  CUTE_DEVICE static float ue8m0_to_inverse_scale(uint8_t value) {
    if (value == 0u) {
      return 0.0f;
    }
    uint32_t const inverse_exponent = 254u - static_cast<uint32_t>(value);
    return __uint_as_float(inverse_exponent << 23);
  }

  CUTE_DEVICE static uint32_t pack_e4m3(float value_0, float value_1, float value_2,
                                        float value_3) {
    uint32_t packed;
    asm volatile("{\n\t"
                 ".reg .b16 lo, hi;\n\t"
                 "cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n\t"
                 "cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n\t"
                 "mov.b32 %0, {lo, hi};\n\t"
                 "}"
                 : "=r"(packed)
                 : "f"(value_0), "f"(value_1), "f"(value_2), "f"(value_3));
    return packed;
  }

  CUTE_DEVICE static int scale_offset(int token, int scale_column) {
    int const row_block = token / kScalePaddedRows;
    int const row_in_block = token % kScalePaddedRows;
    int const column_block = scale_column / 4;
    int const column_in_block = scale_column % 4;
    return row_block * kScalePaddedRows * kScalePaddedColumns + column_block * 512 +
           (row_in_block % 32) * 16 + (row_in_block / 32) * 4 + column_in_block;
  }

  template <class SharedStorage>
  CUTE_DEVICE static void run_warp(Params const &params, SharedStorage const &storage,
                                   float inverse_rms, int token, int local_rank, int quant_warp,
                                   int lane) {
    auto const *residual = reinterpret_cast<__nv_bfloat16 const *>(
        reinterpret_cast<uint8_t const *>(&storage) + Config::kResidualHandoffOffset);
    auto const *gamma = static_cast<__nv_bfloat16 const *>(params.rms_weight_ptr);
    auto *activation = static_cast<uint8_t *>(params.shared_activation_ptr);
    auto *activation_sf = static_cast<uint8_t *>(params.shared_activation_sf_ptr);
    int const token_offset = token * Config::kHiddenSize;
    int const cta_hidden_begin = local_rank * Config::kHiddenPerCta;

    CUTLASS_PRAGMA_UNROLL
    for (int local_scale_column = quant_warp; local_scale_column < kScaleColumnsPerCta;
         local_scale_column += kQuantWarps) {
      int const local_hidden = local_scale_column * kScaleVectorSize + lane;
      int const global_hidden = cta_hidden_begin + local_hidden;
      int const output_offset = token_offset + global_hidden;
      float const value = __bfloat162float(residual[local_hidden]) * inverse_rms *
                          (1.0f + __bfloat162float(gamma[global_hidden]));
      float absmax = warp_max(fabsf(value));
      absmax = __shfl_sync(0xffffffffu, absmax, 0);
      uint8_t const scale = float_to_ue8m0_rp(absmax * (1.0f / 448.0f));
      float const inverse_scale = ue8m0_to_inverse_scale(scale);
      activation[output_offset] =
          __nv_cvt_float_to_fp8(value * inverse_scale, __NV_SATFINITE, __NV_E4M3);
      if (lane == 0) {
        int const scale_column = local_rank * kScaleColumnsPerCta + local_scale_column;
        activation_sf[scale_offset(token, scale_column)] = scale;
      }
    }
  }
};

} // namespace cutlass::m3_decode::collective
