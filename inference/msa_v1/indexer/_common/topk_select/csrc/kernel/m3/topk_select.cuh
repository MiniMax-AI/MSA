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
constexpr int kSelK = kTopK - 1;  // ranked slots; the 16th is the forced tail
constexpr int kThreads = 256;
constexpr int kScoreWarps = kThreads / 32;

struct alignas(16) SelectSmem {
  unsigned slot[kTopK];         // atomicMax round winners
  int sel_ids[kTopK];           // decoded output ids
  float gpart[2][kScoreWarps];  // per-warp [max, min] partials
};

namespace topk_sel {

// Descending sort of a register array: Batcher odd-even merge, which is defined
// for EVERY N -- a bitonic network is not, and at a non-power-of-two N it both
// indexes past the array and skips its last merge stage. t[0] ends up the lane
// maximum; callers read only t[0 .. min(N, kSelK) - 1] and ptxas drops the
// comparators that can reach nothing else.
template <int N>
__device__ __forceinline__ void sort_desc_u32(unsigned (&t)[N]) {
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
__device__ __forceinline__ unsigned pack_key(float v, float scale, float bias,
                                             int id) {
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
  for (int j = 0; j < SH - 1; ++j) key[j] = w ? key[j + 1] : key[j];
  key[SH - 1] = w ? 0u : key[SH - 1];
}

// KPT scores/thread -> block-reduced row [min, max] -> KPT unique keys.
// Padding lanes (id >= nvm) get 0. Ends just after the block-reduce barrier.
template <int KPT>
__device__ __forceinline__ void load_quantize_keys(
    int nvm, const float* __restrict__ rsc1, SelectSmem& s,
    unsigned (&key)[KPT]) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  // (1) KPT scores/thread (raw bits) + per-thread max/min, padding excluded.
  float mymax = -INFINITY, mymin = INFINITY;
#pragma unroll
  for (int i = 0; i < KPT; ++i) {
    const int id = tid + i * kThreads;
    const float v = (id < nvm) ? __ldcg(rsc1 + id) : -INFINITY;
    key[i] = __float_as_uint(v);
    if (id < nvm) {
      mymax = fmaxf(mymax, v);
      mymin = fminf(mymin, v);
    }
  }
  // Block reduce to the row max/min: one partial per warp in smem, then every
  // warp re-reduces the kScoreWarps partials across its own lanes.
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
    mymax = fmaxf(mymax, __shfl_xor_sync(0xFFFFFFFFu, mymax, d));
    mymin = fminf(mymin, __shfl_xor_sync(0xFFFFFFFFu, mymin, d));
  }
  if (lane == 0) {
    s.gpart[0][warp] = mymax;
    s.gpart[1][warp] = mymin;
  }
  __syncthreads();
  float rmax = (lane < kScoreWarps) ? s.gpart[0][lane] : -INFINITY;
  float rmin = (lane < kScoreWarps) ? s.gpart[1][lane] : INFINITY;
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
    rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFFu, rmax, d));
    rmin = fminf(rmin, __shfl_xor_sync(0xFFFFFFFFu, rmin, d));
  }
  // (2) quantize.
  const float scale = (rmax > rmin) ? (65534.0f / (rmax - rmin)) : 0.0f;
  const float bias = fmaf(-rmin, scale, 1.5f);
#pragma unroll
  for (int i = 0; i < KPT; ++i) {
    const int id = tid + i * kThreads;
    key[i] =
        (id < nvm) ? pack_key(__uint_as_float(key[i]), scale, bias, id) : 0u;
  }
}

}  // namespace topk_sel

// Rank the nvm candidates [0, nvm) into s.sel_ids[0..kSelK-1] by (score desc,
// id asc), KPT keys per thread. KPT = ceil(nvm/kThreads) rounded up to a rung
// the caller instantiates, so key[KPT] never spills. Ends on a barrier, so
// sel_ids is visible to the output write.
template <int KPT>
__device__ __forceinline__ void block_topk_atomicmax(
    int nvm, const float* __restrict__ rsc1, SelectSmem& s) {
  const int tid = static_cast<int>(threadIdx.x);
  // A thread contributes at most kSelK entries to the global top-kSelK, so
  // after the sort only the leading min(KPT, kSelK) registers can be selected.
  constexpr int SH = KPT < kSelK ? KPT : kSelK;

  unsigned key[KPT];
  topk_sel::load_quantize_keys<KPT>(nvm, rsc1, s, key);
  topk_sel::sort_desc_u32<KPT>(key);  // key[0] = this thread's max

  if (tid < kSelK) s.slot[tid] = 0u;
  __syncthreads();
  for (int r = 0; r < kSelK; ++r) {
    atomicMax(&s.slot[r], key[0]);
    __syncthreads();
    const unsigned M = s.slot[r];
    const bool w = (key[0] == M);
    topk_sel::winner_shift_down<SH>(key, w);
  }
  if (tid < kSelK) s.sel_ids[tid] = static_cast<int>((~s.slot[tid]) & 0xFFFFu);
  __syncthreads();
}

}  // namespace m3
