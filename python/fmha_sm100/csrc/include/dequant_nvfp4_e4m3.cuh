// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <cassert>
#include <cuda_fp16.h>

namespace k1 {

struct Geom {
  static constexpr int kD          = 128;      // head_size
  static constexpr int kN          = 128;      // page / block size
  static constexpr int kDataRow    = kD / 2;   // 64  packed bytes per (head, token)
  static constexpr int kScaleRow   = kD / 16;  // 8   e4m3 block scales per (head, token)
  static constexpr int kDataSide   = kN * kDataRow;   // 8192  per (page, head, side)
  static constexpr int kScaleSide  = kN * kScaleRow;  // 1024  per (page, head, side)
  static constexpr int kTileElems  = kN * kD;         // 16384
  static constexpr int kTileBytes  = kN * kD;         // 16384 e4m3 bytes = one MMA stage
  static constexpr int kChunkBytes = 16;
  static constexpr int kChunkElems = 32;
  static constexpr int kChunks     = kDataSide / kChunkBytes;   // 512
};

template <int kHkv>
struct PageMap {
  static constexpr int kDataPerSide  = kHkv * Geom::kDataSide;    // 32768 / 8192
  static constexpr int kScalePerSide = kHkv * Geom::kScaleSide;   //  4096 / 1024
  static constexpr int kSideBytes    = kDataPerSide + kScalePerSide;  // 36864 / 9216
  static constexpr int kPageBytes    = 2 * kSideBytes;                // 73728 / 18432

  __host__ __device__ static constexpr int k_data (int head) { return head * Geom::kDataSide; }
  __host__ __device__ static constexpr int k_scale(int head) { return kDataPerSide + head * Geom::kScaleSide; }
  __host__ __device__ static constexpr int v_data (int head) { return kSideBytes + head * Geom::kDataSide; }
  __host__ __device__ static constexpr int v_scale(int head) { return kSideBytes + kDataPerSide + head * Geom::kScaleSide; }

  template <bool kIsV>
  __host__ __device__ static constexpr int data (int head) { return kIsV ? v_data(head)  : k_data(head);  }
  template <bool kIsV>
  __host__ __device__ static constexpr int scale(int head) { return kIsV ? v_scale(head) : k_scale(head); }
};

__host__ __device__ __forceinline__ constexpr int smem_kv_byte(int token, int chan) {
  return 128 * token + 16 * ((chan >> 4) ^ (token & 7)) + (chan & 15);
}

template <bool kIsV, int kNumWarps, bool kScrubNaN = true, int kPrefetch = 2>
__device__ __forceinline__ void dequant_page_tile(
    const uint8_t* __restrict__ sdata,
    const uint8_t* __restrict__ sscale,
    uint8_t*       __restrict__ sout,
    float                       inv_hoist,
    int                         tid)
{
  static_assert(kNumWarps == 1 || kNumWarps == 2 || kNumWarps == 4 || kNumWarps == 8,
                "kNumWarps must be 1, 2, 4 or 8");
  constexpr int kThreads   = 32 * kNumWarps;
  constexpr int kPerThread = Geom::kChunks / kThreads;   // 16 / kNumWarps
  constexpr int kPre       = (kPrefetch < kPerThread) ? kPrefetch : kPerThread;
  static_assert(kPre >= 1, "kPrefetch must be >= 1");
  static_assert(Geom::kChunks % kThreads == 0, "chunk count must divide evenly");

  assert((reinterpret_cast<uintptr_t>(sout) & 1023u) == 0
         && "k1::dequant_page_tile: sout must be 1024 B aligned (Sw<3,4,3> is absolute)");

  uint32_t invg2;
  {
    __half  h  = __float2half_rn(inv_hoist);
    __half2 h2 = __half2half2(h);
    invg2 = *reinterpret_cast<const uint32_t*>(&h2);
  }

  const int chunk0 = tid;
  const int token0 = chunk0 >> 2;
  const int cg     = chunk0 & 3;
  const uint8_t* dp = sdata + Geom::kChunkBytes * chunk0;          // += 16*kThreads
  const uint8_t* sp8;
  uint32_t       vsel = 0;
  if constexpr (!kIsV) {
    sp8 = sscale + 2 * chunk0;                                     // += 2*kThreads
  } else {
    const int r = token0 & 3;
    sp8  = sscale + 32 * (token0 >> 2) + 8 * cg;                   // += 2*kThreads
    vsel = static_cast<uint32_t>(r | ((4 + r) << 4));
  }
  const int g0 = (2 * cg) ^ (token0 & 7);
  uint8_t* op0 = sout + 128 * token0 + 16 * g0;          // += 32*kThreads
  uint8_t* op1 = sout + 128 * token0 + 16 * (g0 ^ 1);    // += 32*kThreads

  uint4    pk_q[kPre];
  uint32_t s16_q[kPre];
  auto fetch = [&](int slot, int i) {
    pk_q[slot] = *reinterpret_cast<const uint4*>(dp + i * (16 * kThreads));
    if constexpr (!kIsV) {
      s16_q[slot] = *reinterpret_cast<const uint16_t*>(sp8 + i * (2 * kThreads));
    } else {
      const uint2 w = *reinterpret_cast<const uint2*>(sp8 + i * (2 * kThreads));
      asm("prmt.b32 %0, %1, %2, %3;" : "=r"(s16_q[slot]) : "r"(w.x), "r"(w.y), "r"(vsel));
    }
  };
  #pragma unroll
  for (int j = 0; j < kPre; ++j) fetch(j, j);

  #pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    const int slot = i % kPre;
    const uint4    pk  = pk_q[slot];
    const uint32_t s16 = s16_q[slot];
    if (i + kPre < kPerThread) fetch(slot, i + kPre);   // refill this slot immediately

    uint32_t sp;   // f16x2 = { scale[2cg], scale[2cg+1] }
    asm("{ .reg .b16 c; cvt.u16.u32 c, %1; cvt.rn.f16x2.e4m3x2 %0, c; }"
        : "=r"(sp) : "r"(s16));
    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(sp) : "r"(invg2));
    if constexpr (kScrubNaN) {
      asm("{ .reg .b32 z; mov.b32 z, 0; max.f16x2 %0, %0, z; }" : "+r"(sp));
    }
    uint32_t o[8];

    uint32_t sc0, sc1;
    asm("{ .reg .b16 a,b; mov.b32 {a,b}, %2; mov.b32 %0, {a,a}; mov.b32 %1, {b,b}; }"
        : "=r"(sc0), "=r"(sc1) : "r"(sp));

    uint32_t h[16];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint32_t word = (k == 0) ? pk.x : (k == 1) ? pk.y : (k == 2) ? pk.z : pk.w;
      asm("{ .reg .b8 b0,b1,b2,b3;\n\t"
          "  mov.b32 {b0,b1,b2,b3}, %4;\n\t"
          "  cvt.rn.f16x2.e2m1x2 %0, b0;\n\t"
          "  cvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
          "  cvt.rn.f16x2.e2m1x2 %2, b2;\n\t"
          "  cvt.rn.f16x2.e2m1x2 %3, b3; }"
          : "=r"(h[4 * k + 0]), "=r"(h[4 * k + 1]), "=r"(h[4 * k + 2]), "=r"(h[4 * k + 3])
          : "r"(word));
    }

    #pragma unroll
    for (int k = 0; k < 8; ++k)  asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(h[k])     : "r"(sc0));
    #pragma unroll
    for (int k = 0; k < 8; ++k)  asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(h[8 + k]) : "r"(sc1));

    #pragma unroll
    for (int k = 0; k < 8; ++k)
      asm("{ .reg .b16 lo, hi;\n\t"
          "  cvt.rn.satfinite.e4m3x2.f16x2 lo, %1;\n\t"
          "  cvt.rn.satfinite.e4m3x2.f16x2 hi, %2;\n\t"
          "  mov.b32 %0, {lo, hi}; }"
          : "=r"(o[k]) : "r"(h[2 * k]), "r"(h[2 * k + 1]));

    *reinterpret_cast<uint4*>(op0 + i * (32 * kThreads)) = make_uint4(o[0], o[1], o[2], o[3]);
    *reinterpret_cast<uint4*>(op1 + i * (32 * kThreads)) = make_uint4(o[4], o[5], o[6], o[7]);
  }
}

}  // namespace k1
