#pragma once
#include "include/sm100/common/m3_numeric.cuh"
#include "include/sm100/common/params.hpp"
#include <cuda_runtime.h>

namespace mega_moe {
__device__ inline float group_max(float x, int width) {
  for (int d = width / 2; d; d >>= 1)
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, d, width));
  return x;
}
__device__ inline float reciprocal(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ inline uint8_t pack_fp4(float x, float y) {
  uint32_t p;
  asm volatile("{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q, %2, %1; cvt.u32.u8 %0, q; }"
               : "=r"(p)
               : "f"(x), "f"(y));
  if (!(p & 7))
    p &= ~8u;
  if (!(p & 0x70))
    p &= ~0x80u;
  return uint8_t(p);
}
// Routed SwiGLU with clamped inputs and an approximate sigmoid reciprocal.
__device__ inline float gated(float gate, float up) {
  gate = fminf(gate, 7.f);
  up = fminf(fmaxf(up, -7.f), 7.f);
  float den = 1.f + exp2f(gate * (-1.702f * 1.4426950408889634f));
  return (up + 1.f) * gate * reciprocal(den);
}
} // namespace mega_moe
