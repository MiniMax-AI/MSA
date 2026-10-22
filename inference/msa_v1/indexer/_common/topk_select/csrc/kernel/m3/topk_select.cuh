/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Block-level top-K. Every candidate packs into ONE unique u32 key -- 16-bit
// score high, ~index low -- so a max reduction is a full arg-max and the slot
// array is already the sorted output: no ties, no histogram, no index recovery.
// The score is quantized linearly over the row's [min, max], which resolves the
// exact top-K boundary while keeping the key a single word.

namespace m3 {

constexpr int kTopK = 16;
constexpr int kSelK = kTopK - 1; // ranked slots; the 16th is the forced tail
constexpr int kThreads = 256;
constexpr int kScoreWarps = kThreads / 32;
constexpr int kMaximumColumns = 16384;
constexpr int kBufferedKeysPerThread = 32;

constexpr int kPrefetchKeys = 8;
static_assert(kMaximumColumns <= 65536);
static_assert(kScoreWarps * kTopK == 32 * 4, "the merge warp holds 4 keys per lane");

struct alignas(16) SelectSmem {
  unsigned wsel[kScoreWarps][kTopK];
  int gpart[kScoreWarps][2];
};

namespace topk_sel {

// Descending sort of a register array: Batcher odd-even merge, which is defined
// for EVERY N -- a bitonic network is not, and at a non-power-of-two N it both
// indexes past the array and skips its last merge stage. t[0] ends up the lane
// maximum; callers read only t[0 .. min(N, kSelK) - 1] and ptxas drops the
// comparators that can reach nothing else.
template <int N> __device__ __forceinline__ void sort_desc_u32(unsigned (&t)[N]) {
#pragma unroll
  for (int p = 1; p < N; p += p)
#pragma unroll
    for (int k = p; k >= 1; k /= 2)
#pragma unroll
      for (int j = k % p; j + k < N; j += 2 * k)
#pragma unroll
        for (int i = 0; i < k && i + j + k < N; ++i)
          if ((i + j) / (p * 2) == (i + j + k) / (p * 2)) {
            const unsigned a = t[i + j], b = t[i + j + k];
            t[i + j] = a > b ? a : b;
            t[i + j + k] = a > b ? b : a;
          }
}

// The unique selection key: 16-bit fixed-point score in [1, 65535] (high half)
// + ~id (low half, so a lower id wins ties). `scale` = 65534/(hi-lo), or 0 for a
// degenerate range, where every score quantizes to 1 and the id decides.
// `bias` = 1.5 - lo*scale folds the shift and the rounding into one fma.
// Callers pack empty slots as 0.
__device__ __forceinline__ unsigned pack_key(float v, float scale, float bias, int id) {
  unsigned q = static_cast<unsigned>(fmaf(v, scale, bias));
  q = q > 65535u ? 65535u : q;
  return (q << 16) | ((~static_cast<unsigned>(id)) & 0xFFFFu);
}

// Pop the round winner: if this thread held the max (w), shift key[1..SH-1] down
// and zero the top. Branchless and static-indexed, so key[] stays in registers.
// SH <= ASZ is the live prefix.
template <int SH, int ASZ>
__device__ __forceinline__ void winner_shift_down(unsigned (&key)[ASZ], bool w) {
#pragma unroll
  for (int j = 0; j < SH - 1; ++j)
    key[j] = w ? key[j + 1] : key[j];
  key[SH - 1] = w ? 0u : key[SH - 1];
}

__device__ __forceinline__ int ordered_int(float value) {
  const int bits = __float_as_int(value);
  return bits ^ ((bits >> 31) & 0x7FFFFFFF);
}

__device__ __forceinline__ float ordered_float(int key) {
  return __int_as_float(key ^ ((key >> 31) & 0x7FFFFFFF));
}

__device__ __forceinline__ void warp_max_min(int &kmax, int &kmin) {
  kmax = __reduce_max_sync(0xFFFFFFFFu, kmax);
  kmin = __reduce_min_sync(0xFFFFFFFFu, kmin);
}

template <int SH, int ASZ>
__device__ __forceinline__ void warp_select_to(unsigned (&key)[ASZ], unsigned *dst) {
  const bool leader = (static_cast<int>(threadIdx.x) & 31) == 0;
#pragma unroll
  for (int r = 0; r < kSelK; ++r) {
    unsigned M;
    asm("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(M) : "r"(key[0]));
    if (leader)
      dst[r] = M;
    winner_shift_down<SH>(key, key[0] == M);
  }
}

template <int SH, int ASZ> __device__ __forceinline__ unsigned warp_select(unsigned (&key)[ASZ]) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  unsigned mine = 0u;
#pragma unroll
  for (int r = 0; r < kSelK; ++r) {
    unsigned M;
    asm("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(M) : "r"(key[0]));
    mine = (lane == r) ? M : mine;
    winner_shift_down<SH>(key, key[0] == M);
  }
  return mine;
}

// Padding lanes (id >= nvm) get 0. Ends just after the block-reduce barrier.
template <int KPT>
__device__ __forceinline__ void
load_quantize_keys(int nvm, const float *__restrict__ rsc1, const float (&pre)[kPrefetchKeys],
                   SelectSmem &s, unsigned (&key)[KPT], float &scale, float &bias) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  // (1) KPT scores/thread (raw bits) + per-thread max/min, padding excluded.

  float mymax = -INFINITY, mymin = INFINITY;
#pragma unroll
  for (int i = 0; i < KPT; ++i) {
    const int id = tid + i * kThreads;
    const float v = (id < nvm)
                        ? (i < kPrefetchKeys ? pre[i < kPrefetchKeys ? i : 0] : __ldcg(rsc1 + id))
                        : -INFINITY;
    key[i] = __float_as_uint(v);
    if (id < nvm) {
      mymax = fmaxf(mymax, v);
      mymin = fminf(mymin, v);
    }
  }
  if constexpr (KPT == kBufferedKeysPerThread) {
#pragma unroll 1
    for (int column = KPT * kThreads + tid; column < nvm; column += kThreads) {
      const float value = __ldcg(rsc1 + column);
      mymax = fmaxf(mymax, value);
      mymin = fminf(mymin, value);
    }
  }

  int kmax = ordered_int(mymax), kmin = ordered_int(mymin);
  warp_max_min(kmax, kmin);
  if (lane == 0) {
    s.gpart[warp][0] = kmax;
    s.gpart[warp][1] = kmin;
  }
  __syncthreads();
  const int2 partial = reinterpret_cast<const int2 *>(s.gpart)[lane & (kScoreWarps - 1)];
  kmax = partial.x;
  kmin = partial.y;
  warp_max_min(kmax, kmin);
  const float rmax = ordered_float(kmax), rmin = ordered_float(kmin);
  // (2) quantize.
  scale = (rmax > rmin) ? (65534.0f / (rmax - rmin)) : 0.0f;
  bias = fmaf(-rmin, scale, 1.5f);
#pragma unroll
  for (int i = 0; i < KPT; ++i) {
    const int id = tid + i * kThreads;
    key[i] = (id < nvm) ? pack_key(__uint_as_float(key[i]), scale, bias, id) : 0u;
  }
}

} // namespace topk_sel

template <int KPT>
__device__ __forceinline__ void block_topk_select(int nvm, const float *__restrict__ rsc1,
                                                  const float (&pre)[kPrefetchKeys], SelectSmem &s,
                                                  int *__restrict__ out_row) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  // A thread contributes at most kSelK entries to the global top-kSelK, so
  // after the sort only the leading min(KPT, kSelK) registers can be selected.
  constexpr int SH = KPT < kSelK ? KPT : kSelK;

  unsigned key[KPT];
  float scale, bias;
  topk_sel::load_quantize_keys<KPT>(nvm, rsc1, pre, s, key, scale, bias);
  topk_sel::sort_desc_u32<KPT>(key); // key[0] = this thread's max

  if constexpr (KPT == kBufferedKeysPerThread) {
#pragma unroll 1
    for (int column = KPT * kThreads + tid; column < nvm; column += kThreads) {
      unsigned candidate = topk_sel::pack_key(__ldcg(rsc1 + column), scale, bias, column);
#pragma unroll
      for (int slot = 0; slot < kSelK; ++slot) {
        const unsigned retained = key[slot];
        key[slot] = candidate > retained ? candidate : retained;
        candidate = candidate > retained ? retained : candidate;
      }
    }
  }

  if (lane == 0)
    s.wsel[warp][kSelK] = 0u;
  topk_sel::warp_select_to<SH>(key, s.wsel[warp]);
  __syncthreads();

  if (warp == 0) {
    const uint4 winners = reinterpret_cast<const uint4 *>(&s.wsel[0][0])[lane];
    unsigned run[4] = {winners.x, winners.y, winners.z, winners.w};
    const unsigned top = topk_sel::warp_select<4>(run);
    if (lane < kTopK)
      out_row[lane] = (lane == kSelK) ? nvm : static_cast<int>((~top) & 0xFFFFu);
  }
}

} // namespace m3
