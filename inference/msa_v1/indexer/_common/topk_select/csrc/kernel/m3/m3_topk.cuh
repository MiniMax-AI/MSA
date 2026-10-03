/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// m3_topk -- top-K select over one row of scores.
//
// Contract (kTopK = 16): a row of `n` scores yields 16 ids. Column n-1 is always
// selected and written to the last slot without being ranked; the ranking picks
// kSelK = 15 out of the remaining nvm = n-1 columns. A row of n <= kTopK emits
// its n ids in order and pads with -1, so the tail lands at slot n-1.
//
// `n` is PER ROW, always: it is `row_n[row]`, and there is no fixed-length path.
// A paged decode scores a row only over the pages before that query's own local
// page, so n varies with the sequence length; the scalar `n_max` only bounds
// them, and is the only `n` the host and the kernel entry can use. Padding the
// tail is no escape: the key quantizes over the row's own [min, max]
// (topk_select.cuh), so a -1e30 pad collapses every real score onto one key.

#pragma once

#include "topk_select.cuh"

namespace m3 {

// Rows per CTA for the warp family. A constant on purpose: it depends on neither
// the SM count nor `rows`.
constexpr int kRowsPerBlock = 4;

// The contract's ceiling: the widest row the select ranks.
constexpr int kMaxNvm = 8192;

// KPT rungs the block select is instantiated at, narrowest first. A row takes
// the first rung whose KPT*kThreads covers its own nvm, so the rung IS the size
// of that row's per-thread register array.
#ifndef M3_RUNGS
#define M3_RUNGS 2, 4, 8, 12, 16, 20, 24, 28, 32
#endif
constexpr int kRung[] = {M3_RUNGS};
constexpr int kRungs = static_cast<int>(sizeof(kRung) / sizeof(kRung[0]));
static_assert(kRung[kRungs - 1] * kThreads >= kMaxNvm,
              "the last rung has to reach kMaxNvm");

// The kernel's thresholds, host-callable, so a caller can label a point with the
// family and width it takes. nvm = n-1. Fed n_max, they describe the launch
// shape, not what any one row of that launch takes.
inline const char* m3_family(int n) {
  if (n <= kTopK) return "trivial";
  if (n - 1 <= kThreads) return "warp";
  return "block";
}

inline int m3_width(int n) {
  const int nvm = n - 1;
  if (n <= kTopK) return 0;
  if (nvm <= 32) return 1;
  if (nvm <= 64) return 2;
  if (nvm <= 128) return 4;
  if (nvm <= 256) return 8;
  for (int i = 0; i < kRungs; ++i)
    if (nvm <= kRung[i] * kThreads) return kRung[i];
  return kRung[kRungs - 1];
}

// The widest row any launch ranks. Past it the select would drop candidates.
inline int m3_max_n() { return kMaxNvm + 1; }

// ---------------------------------------------------------------------------
// What the two families do identically: read the row's own n, answer it outright
// when it is short, and lay out the 16 output slots. `slot` is the calling
// thread's index inside whatever unit owns the row -- the CTA in the block
// family, the warp in the warp family -- which is what lets one body serve both.
// Every thread of that unit reads the same `n`, so the width branch each family
// takes is uniform over that unit.
// ---------------------------------------------------------------------------

// This row's nvm = n-1, the count the select ranks -- or 0 once the row is
// already written, which is every row of n <= kTopK: fewer columns than slots,
// so the answer is the n ids in order padded with -1, and nothing is ranked.
__device__ __forceinline__ int row_nvm(const int* __restrict__ row_n,
                                       int* __restrict__ out_ids, int row,
                                       int slot) {
  const int n = __ldg(row_n + row);
  if (n > kTopK) return n - 1;  // column n-1 is forced-selected, never ranked
  if (slot < kTopK) out_ids[row * kTopK + slot] = (slot < n) ? slot : -1;
  return 0;
}

__device__ __forceinline__ const float* row_scores(
    const float* __restrict__ scores, int row, int row_stride) {
  return scores + static_cast<size_t>(row) * static_cast<size_t>(row_stride);
}

// The 15 ranked ids, then the forced tail in the last slot.
__device__ __forceinline__ void write_row(int* __restrict__ out_ids, int row,
                                          int slot, int nvm,
                                          const int* __restrict__ sel) {
  if (slot < kTopK)
    out_ids[row * kTopK + slot] = (slot == kSelK) ? nvm : sel[slot];
}

// One warp ranks one row: load -> warp-reduce [min,max] -> pack_key -> sort ->
// kSelK rounds of redux.sync.max, winner shifts down. No block barrier, and the
// result goes to a per-warp `out15` rather than shared state, so warps in one CTA
// never interact -- which is what lets each warp pick its width from its own row.
// __noinline__ keeps the four instantiations out of the caller's inline body.
template <int N>
__device__ __noinline__ void warp_peel_row(int nvm,
                                           const float* __restrict__ rsc1,
                                           int* __restrict__ out15) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  constexpr int SH = N < kSelK ? N : kSelK;
  float sc[N];
  float mymax = -INFINITY, mymin = INFINITY;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const int idx = i * 32 + lane;
    const float v = (idx < nvm) ? __ldcg(rsc1 + idx) : -INFINITY;
    sc[i] = v;
    if (idx < nvm) {
      mymax = fmaxf(mymax, v);
      mymin = fminf(mymin, v);
    }
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
    mymax = fmaxf(mymax, __shfl_xor_sync(0xFFFFFFFFu, mymax, d));
    mymin = fminf(mymin, __shfl_xor_sync(0xFFFFFFFFu, mymin, d));
  }
  const float scale = (mymax > mymin) ? (65534.0f / (mymax - mymin)) : 0.0f;
  const float bias = fmaf(-mymin, scale, 1.5f);
  unsigned key[N];
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const int idx = i * 32 + lane;
    key[i] = (idx < nvm) ? topk_sel::pack_key(sc[i], scale, bias, idx) : 0u;
  }
  topk_sel::sort_desc_u32<N>(key);
#pragma unroll
  for (int r = 0; r < kSelK; ++r) {
    unsigned M;
    asm("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(M) : "r"(key[0]));
    if (lane == 0) out15[r] = static_cast<int>((~M) & 0xFFFFu);
    const bool w = (key[0] == M);
    topk_sel::winner_shift_down<SH>(key, w);
  }
  __syncwarp();  // lane 0's writes visible to the rest of this warp
}

// One row, one CTA: take the narrowest rung that covers this row's nvm.
template <int I>
__device__ __forceinline__ void block_rank_row(int nvm,
                                               const float* __restrict__ p,
                                               SelectSmem& s) {
  constexpr int KPT = kRung[I];
  constexpr bool kLast = (I + 1 == kRungs);  // covers kMaxNvm by static_assert
  if (kLast || nvm <= KPT * kThreads) {
    block_topk_atomicmax<KPT>(nvm, p, s);
  } else if constexpr (!kLast) {  // discarded at the last rung, so it terminates
    block_rank_row<I + 1>(nvm, p, s);
  }
}

// One warp per row: take the narrowest width that covers this row's nvm.
__device__ __forceinline__ void warp_rank_row(int nvm,
                                              const float* __restrict__ p,
                                              int* __restrict__ out15) {
  if (nvm <= 32)
    warp_peel_row<1>(nvm, p, out15);
  else if (nvm <= 64)
    warp_peel_row<2>(nvm, p, out15);
  else if (nvm <= 128)
    warp_peel_row<4>(nvm, p, out15);
  else
    warp_peel_row<8>(nvm, p, out15);
}

// Rows per CTA -- what the warp family packs into one CTA, reported so a caller
// can label a launch with it. The grid is NOT derived from this (it is `rows`);
// this is the kernel's own family test, so a label built from it cannot disagree
// with the mapping the kernel actually uses.
inline int m3_rows_per_block(int n_max) {
  return ((n_max - 1) <= kThreads) ? kRowsPerBlock : 1;
}

// FAMILY comes from n_max and is grid-uniform -- it fixes the thread -> row
// mapping, which the whole grid has to agree on:
//   n_max-1 <= kThreads : warp family, one warp per row, kRowsPerBlock per CTA
//   n_max-1 >  kThreads : block family, one row per CTA, all kThreads on it
// A row of <= 256 candidates is ranked warp-locally because one row per CTA
// would idle 7 of 8 warps.
//
// WIDTH comes from that row's own nvm: per CTA in the block family, per warp in
// the warp family. A short row in a block-family launch takes the narrowest
// block width rather than a warp path -- correct, just wider than it needs.
__global__ __launch_bounds__(kThreads) void m3_topk_kernel(const float *__restrict__ scores,
                                                           const int *__restrict__ row_n,
                                                           int *__restrict__ out_ids, int n_max,
                                                           int row_stride, int rows,
                                                           bool pdl = false) {
  __shared__ union Smem {
    SelectSmem s;
    int sel[kRowsPerBlock][kSelK];
  } u;
  const int tid = static_cast<int>(threadIdx.x);

  if (n_max - 1 <= kThreads) {
    // ---- warp family: one warp per row, no block barrier anywhere ----
    // This branch IS m3_rows_per_block's test, so the rows per CTA here is
    // kRowsPerBlock -- the same one the host built the grid from.
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;
    // A warp past kRowsPerBlock is idle and the tail CTA is bounded per warp;
    // neither may touch row_n, so the guard has to come before row_nvm rather
    // than fold into it. Both leave nvm at 0 -- the same "nothing to rank" the
    // short-row branch leaves -- so one test below covers all three cases.
    const bool owns_row = (warp < kRowsPerBlock) && (row < rows);
    const int nvm = owns_row ? row_nvm(row_n, out_ids, row, lane) : 0;
    if (pdl)
      cudaGridDependencySynchronize();
    if (nvm > 0) {
      warp_rank_row(nvm, row_scores(scores, row, row_stride), u.sel[warp]);
      write_row(out_ids, row, lane, nvm, u.sel[warp]);
    }
  } else {
    // ---- block family: one row per CTA, so blockIdx.x IS the row and every
    // thread of the CTA reads the same n -- the branch below is CTA-uniform,
    // which is what keeps the __syncthreads() inside block_rank_row convergent.
    const int row = static_cast<int>(blockIdx.x);
    const int nvm = row_nvm(row_n, out_ids, row, tid);
    if (pdl)
      cudaGridDependencySynchronize();
    if (nvm > 0) {
      block_rank_row<0>(nvm, row_scores(scores, row, row_stride), u.s);
      write_row(out_ids, row, tid, nvm, u.s.sel_ids);
    }
  }
}

inline int m3_occ() {
  int blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m3_topk_kernel,
                                                kThreads, 0);
  return blocks;
}

// THE GRID RULE: one CTA per row. Always -- the grid depends on `rows` alone, so
// the host never has to know the family to size a launch.
//
// The block family already wants exactly that: one row per CTA. The warp family
// packs kRowsPerBlock rows into a CTA, so one CTA per row over-launches 4x, and
// the surplus CTAs compute a row index past `rows`, fail the kernel's own
// `row < rows` guard and retire without touching memory. The output is
// bit-identical either way; what the surplus costs is scheduling, because a CTA
// still has to land on an SM and be given its registers before it can find out
// it has nothing to do.
//
// Measured on B300 against a packed ceil(rows/rpb) grid (pytopk/test_gridsize.py
// A/Bs the two over this same kernel): at rows <= 1024, the batch size this is
// built for, the warp family pays +0.9% at the median and +2.8% at its worst
// point, and the whole bs x kv matrix moves +0.03%. That is the price of the
// simpler launch and it is worth paying. It does NOT stay that cheap -- past
// rows ~3072 the retiring CTAs stop fitting in the shadow of the working ones:
// +7 to +15% at 4096 rows, +31 to +45% at 8192. A caller working at that scale
// wants the packed grid back.
inline void m3_launch(const float* scores, const int* row_n, int* out, int n_max,
                      int row_stride, int rows, cudaStream_t st) {
  m3_topk_kernel<<<rows, kThreads, 0, st>>>(scores, row_n, out, n_max,
                                            row_stride, rows);
}

}  // namespace m3
