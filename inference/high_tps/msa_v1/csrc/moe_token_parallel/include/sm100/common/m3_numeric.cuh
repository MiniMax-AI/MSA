#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#ifdef __CUDACC__
#define M3_HD __host__ __device__ __forceinline__
#else
#define M3_HD inline
#endif
namespace m3 {
M3_HD uint32_t bits(float x) {
  uint32_t u;
#ifdef __CUDA_ARCH__
  u = __float_as_uint(x);
#else
  std::memcpy(&u, &x, 4);
#endif
  return u;
}
M3_HD float from_bits(uint32_t u) {
  float x;
#ifdef __CUDA_ARCH__
  x = __uint_as_float(u);
#else
  std::memcpy(&x, &u, 4);
#endif
  return x;
}
M3_HD float bf16(uint16_t x) { return from_bits(uint32_t(x) << 16); }
M3_HD uint16_t to_bf16(float x) {
  uint32_t u = bits(x);
  if ((u & 0x7fffffff) > 0x7f800000)
    return uint16_t((u >> 16) | 0x40);
  return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
// Round to the nearest BF16 value and return it as FP32.
M3_HD float bf16_round(float x) { return bf16(to_bf16(x)); }
M3_HD float e4m3(uint8_t x) {
  int e = (x >> 3) & 15, m = x & 7;
  float y = (e == 15 && m == 7) ? NAN
            : e == 0            ? ldexpf(float(m), -9)
                                : ldexpf(float(8 + m), e - 10);
  return x & 128 ? -y : y;
}
M3_HD uint8_t to_e4m3(float x) {
#ifdef __CUDA_ARCH__
  uint16_t v;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(v) : "f"(x));
  return uint8_t(v);
#else
  uint8_t sign = uint8_t((bits(x) >> 24) & 128);
  float a = fabsf(x);
  if (std::isnan(a))
    return sign | 127;
  if (a >= 448.f)
    return sign | 126;
  int lo = 0, hi = 126;
  while (lo + 1 < hi) {
    int m = (lo + hi) / 2;
    if (e4m3(m) <= a)
      lo = m;
    else
      hi = m;
  }
  float dl = a - e4m3(lo), dh = e4m3(hi) - a;
  int c = dl < dh ? lo : dh < dl ? hi : ((lo & 1) ? hi : lo);
  return sign | uint8_t(c);
#endif
}
// Round UE8M0 scales up while preserving subnormal and code-zero behavior.
M3_HD uint8_t ue8m0_up(float x) {
  uint32_t u = bits(x), e = (u >> 23) & 255, m = u & 0x7fffff;
  uint32_t bump = m != 0;
  if (e == 0 && m <= 0x400000)
    bump = 0;
  uint32_t c = e + bump;
  return x <= 0 ? 0 : uint8_t(c > 254 ? 254 : c);
}
M3_HD float input_inv_scale(uint8_t c) { return c == 0 ? 0.f : from_bits(uint32_t(254 - c) << 23); }
} // namespace m3
