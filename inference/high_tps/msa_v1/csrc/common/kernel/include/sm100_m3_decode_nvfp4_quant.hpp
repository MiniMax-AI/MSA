#pragma once

#include <cuda_fp8.h>

#include <cstdint>

#include "cutlass/cutlass.h"

namespace cutlass::m3_decode::collective {

struct Sm100M3DecodeNvfp4Quant {
  CUTE_DEVICE static uint8_t float_to_e4m3(float value) {
    uint16_t packed;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, 0.0, %1;" : "=h"(packed) : "f"(value));
    return static_cast<uint8_t>(packed);
  }

  CUTE_DEVICE static float e4m3_to_float(uint8_t value) {
    __nv_fp8_e4m3 converted = *reinterpret_cast<__nv_fp8_e4m3 *>(&value);
    return static_cast<float>(converted);
  }

  CUTE_DEVICE static float reciprocal_approx(float value) {
    float result;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
  }

  CUTE_DEVICE static uint8_t pack_e2m1x2_canonical(float value0, float value1) {
    uint32_t packed;
    asm volatile("{\n"
                 "  .reg .b8 pair;\n"
                 "  .reg .b32 magnitude;\n"
                 "  .reg .pred is_zero;\n"
                 "  cvt.rn.satfinite.e2m1x2.f32 pair, %2, %1;\n"
                 "  cvt.u32.u8 %0, pair;\n"
                 "  and.b32 magnitude, %0, 0x07;\n"
                 "  setp.eq.u32 is_zero, magnitude, 0;\n"
                 "  @is_zero and.b32 %0, %0, 0xfffffff7;\n"
                 "  and.b32 magnitude, %0, 0x70;\n"
                 "  setp.eq.u32 is_zero, magnitude, 0;\n"
                 "  @is_zero and.b32 %0, %0, 0xffffff7f;\n"
                 "}\n"
                 : "=r"(packed)
                 : "f"(value0), "f"(value1));
    return static_cast<uint8_t>(packed);
  }
};

} // namespace cutlass::m3_decode::collective
