#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace minimax::inference::dequant::sm100 {

#if (__CUDACC_VER_MAJOR__ > 13) || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)
#define MINIMAX_DEQUANT_HAS_QMUL4 1
#else
#define MINIMAX_DEQUANT_HAS_QMUL4 0
#endif

#if MINIMAX_DEQUANT_HAS_QMUL4
__device__ __forceinline__ uint4 convert_nvfp4_group(uint2 packed, uint32_t scale_e4m3x4) {
  uint4 output;
  uint16_t const packed_0 = static_cast<uint16_t>(packed.x);
  uint16_t const packed_1 = static_cast<uint16_t>(packed.x >> 16);
  uint16_t const packed_2 = static_cast<uint16_t>(packed.y);
  uint16_t const packed_3 = static_cast<uint16_t>(packed.y >> 16);
  asm volatile("mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, %4, %8;\n"
               "mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %1, %5, %8;\n"
               "mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %2, %6, %8;\n"
               "mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %3, %7, %8;"
               : "=r"(output.x), "=r"(output.y), "=r"(output.z), "=r"(output.w)
               : "h"(packed_0), "h"(packed_1), "h"(packed_2), "h"(packed_3), "r"(scale_e4m3x4));
  return output;
}
#else
__device__ __forceinline__ void convert_e2m1x8_fallback(uint32_t &output_0, uint32_t &output_1,
                                                        uint32_t packed, uint32_t scale_f16x2) {
  asm volatile("{\n"
               ".reg .b8 b0, b1, b2, b3;\n"
               ".reg .b32 h0, h1, h2, h3;\n"
               ".reg .b16 e0, e1, e2, e3;\n"
               "mov.b32 {b0, b1, b2, b3}, %2;\n"
               "cvt.rn.f16x2.e2m1x2 h0, b0;\n"
               "cvt.rn.f16x2.e2m1x2 h1, b1;\n"
               "cvt.rn.f16x2.e2m1x2 h2, b2;\n"
               "cvt.rn.f16x2.e2m1x2 h3, b3;\n"
               "mul.rn.f16x2 h0, h0, %3;\n"
               "mul.rn.f16x2 h1, h1, %3;\n"
               "mul.rn.f16x2 h2, h2, %3;\n"
               "mul.rn.f16x2 h3, h3, %3;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n"
               "mov.b32 %0, {e0, e1};\n"
               "mov.b32 %1, {e2, e3};\n"
               "}"
               : "=r"(output_0), "=r"(output_1)
               : "r"(packed), "r"(scale_f16x2));
}

__device__ __forceinline__ uint4 convert_nvfp4_group(uint2 packed, uint32_t scale_byte) {
  uint16_t const scale_e4m3x2 = static_cast<uint16_t>(scale_byte | (scale_byte << 8));
  uint32_t scale_f16x2;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(scale_f16x2) : "h"(scale_e4m3x2));

  uint4 output;
  convert_e2m1x8_fallback(output.x, output.y, packed.x, scale_f16x2);
  convert_e2m1x8_fallback(output.z, output.w, packed.y, scale_f16x2);
  return output;
}
#endif

__device__ __forceinline__ uint4 dequantize_group(uint2 packed, uint8_t scale) {
#if MINIMAX_DEQUANT_HAS_QMUL4
  return convert_nvfp4_group(packed, static_cast<uint32_t>(scale) * 0x01010101u);
#else
  return convert_nvfp4_group(packed, static_cast<uint32_t>(scale));
#endif
}

} // namespace minimax::inference::dequant::sm100
