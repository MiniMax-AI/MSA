/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <climits>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace cutlass::m3_decode::collective::runtime_topk {

// Register odd-even merge, packed-key selection, and runtime width dispatch.
// Only the first 256 threads participate;
// the enclosing CTA also owns attention work. No CTA-wide barrier is legal here.
struct alignas(16) Scratch {
  uint32_t slots[16];
  uint32_t warp_candidates[8][16];
  int selected[16];
  uint32_t extrema[2][8];
  uint64_t exact_partials[8];
};

template <int Threads> __device__ __forceinline__ void sync() {
  if constexpr (Threads == 32) {
    __syncwarp();
  } else {
    asm volatile("bar.sync 2, 256;" ::: "memory");
  }
}

template <int N> __device__ __forceinline__ void sort_descending(uint32_t (&keys)[N]) {
#pragma unroll
  for (int p = 1; p < N; p *= 2) {
#pragma unroll
    for (int k = p; k >= 1; k /= 2) {
#pragma unroll
      for (int j = k % p; j + k < N; j += 2 * k) {
#pragma unroll
        for (int i = 0; i < k && i + j + k < N; ++i) {
          if ((i + j) / (p * 2) == (i + j + k) / (p * 2)) {
            uint32_t const a = keys[i + j];
            uint32_t const b = keys[i + j + k];
            keys[i + j] = a > b ? a : b;
            keys[i + j + k] = a > b ? b : a;
          }
        }
      }
    }
  }
}

__device__ __forceinline__ uint32_t warp_max(uint32_t value) {
  return __reduce_max_sync(0xffffffffu, value);
}

__device__ __forceinline__ uint64_t warp_max(uint64_t value) {
  uint32_t const high = static_cast<uint32_t>(value >> 32);
  uint32_t const best_high = warp_max(high);
  uint32_t const low = warp_max(high == best_high ? static_cast<uint32_t>(value) : 0u);
  return (static_cast<uint64_t>(best_high) << 32) | low;
}

__device__ __forceinline__ uint32_t bucket(float score, float scale, float bias) {
  return static_cast<uint32_t>(fminf(65535.0f, fmaxf(1.0f, fmaf(score, scale, bias))));
}

__device__ __forceinline__ uint32_t ordered_score(float score) {
  // Canonicalize signed zero: equal FP32 values must tie by page ID.
  uint32_t bits = __float_as_uint(score);
  bits = (bits & 0x7fffffffu) == 0 ? 0u : bits;
  return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}

__device__ __forceinline__ float score_from_ordered(uint32_t ordered) {
  return __uint_as_float(ordered & 0x80000000u ? ordered ^ 0x80000000u : ~ordered);
}

__device__ __forceinline__ uint64_t exact_key(float score, int page) {
  return (static_cast<uint64_t>(ordered_score(score)) << 32) | static_cast<uint32_t>(~page);
}

template <int Threads>
__device__ __noinline__ void refine_boundary(float const *scores, int ranked_pages,
                                             Scratch &scratch, float scale, float bias) {
  int const tid = static_cast<int>(threadIdx.x);
  int const lane = tid & 31;
  int const warp = tid >> 5;
  uint32_t const boundary = scratch.slots[14] >> 16;
  int strict_count = 0;
  while (strict_count < 15 && (scratch.slots[strict_count] >> 16) > boundary) {
    ++strict_count;
  }
  uint64_t previous = ~uint64_t(0);
  for (int selected = strict_count; selected < 15; ++selected) {
    uint64_t best = 0;
    // Only a colliding boundary bucket is re-ranked. Streaming the bucket
    // bounds register use even for the widest row and the all-equal case.
    for (int page = tid; page < ranked_pages; page += Threads) {
      float const score = __ldcg(scores + page);
      uint64_t const key = exact_key(score, page);
      if (bucket(score, scale, bias) == boundary && key < previous) {
        best = best > key ? best : key;
      }
    }
    best = warp_max(best);
    if constexpr (Threads != 32) {
      if (lane == 0) {
        scratch.exact_partials[warp] = best;
      }
      sync<Threads>();
      best = warp_max(lane < Threads / 32 ? scratch.exact_partials[lane] : 0);
    }
    if (tid == 0) {
      scratch.selected[selected] = static_cast<int>(~static_cast<uint32_t>(best));
    }
    previous = best;
    sync<Threads>();
  }
}

template <int Threads, int Width>
__device__ __noinline__ void rank_row(float const *scores, int ranked_pages, Scratch &scratch) {
  int const tid = static_cast<int>(threadIdx.x);
  int const lane = tid & 31;
  int const warp = tid >> 5;
  uint32_t keys[Width];
  uint32_t high_key = 0;
  uint32_t low_key = ~uint32_t(0);
#pragma unroll
  for (int item = 0; item < Width; ++item) {
    int const page = tid + item * Threads;
    float const score = page < ranked_pages ? __ldcg(scores + page) : -CUDART_INF_F;
    keys[item] = __float_as_uint(score);
    if (page < ranked_pages) {
      uint32_t const ordered = ordered_score(score);
      high_key = max(high_key, ordered);
      low_key = min(low_key, ordered);
    }
  }
#pragma unroll
  for (int delta = 16; delta > 0; delta /= 2) {
    high_key = max(high_key, __shfl_xor_sync(0xffffffffu, high_key, delta));
    low_key = min(low_key, __shfl_xor_sync(0xffffffffu, low_key, delta));
  }
  if constexpr (Threads != 32) {
    if (lane == 0) {
      scratch.extrema[0][warp] = high_key;
      scratch.extrema[1][warp] = low_key;
    }
    sync<Threads>();
    high_key = lane < Threads / 32 ? scratch.extrema[0][lane] : 0;
    low_key = lane < Threads / 32 ? scratch.extrema[1][lane] : ~uint32_t(0);
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {
      high_key = max(high_key, __shfl_xor_sync(0xffffffffu, high_key, delta));
      low_key = min(low_key, __shfl_xor_sync(0xffffffffu, low_key, delta));
    }
  }
  float const high = score_from_ordered(high_key);
  float const low = score_from_ordered(low_key);
  float scale = high > low ? 65534.0f / (high - low) : 0.0f;
  float bias = fmaf(-low, scale, 1.5f);
  // Extreme finite ranges may overflow the scaling arithmetic. Collapsing
  // these to one bucket deliberately activates the exact FP32 fallback.
  if (!::isfinite(scale) || !::isfinite(bias)) {
    scale = 0.0f;
    bias = 1.5f;
  }
#pragma unroll
  for (int item = 0; item < Width; ++item) {
    int const page = tid + item * Threads;
    keys[item] = page < ranked_pages ? (bucket(__uint_as_float(keys[item]), scale, bias) << 16) |
                                           (static_cast<uint32_t>(~page) & 0xffffu)
                                     : 0u;
  }
  sort_descending(keys);
  // Keep the 16th historical candidate for the exact boundary check. Each
  // warp contributes its sorted top 16, then warp zero merges those lists.
  // This is the standalone selector's two-level structure, with named-barrier
  // synchronization and one additional candidate instead of its forced tail.
  constexpr int kLive = Width < 16 ? Width : 16;
#pragma unroll
  for (int round = 0; round < 16; ++round) {
    uint32_t const winner = warp_max(keys[0]);
    if (lane == 0) {
      if constexpr (Threads == 32) {
        scratch.slots[round] = winner;
      } else {
        scratch.warp_candidates[warp][round] = winner;
      }
    }
    bool const won = keys[0] == winner;
#pragma unroll
    for (int item = 0; item < kLive - 1; ++item) {
      keys[item] = won ? keys[item + 1] : keys[item];
    }
    keys[kLive - 1] = won ? 0u : keys[kLive - 1];
  }
  sync<Threads>();
  if constexpr (Threads != 32) {
    if (warp == 0) {
      uint4 const winners = reinterpret_cast<uint4 const *>(scratch.warp_candidates)[lane];
      uint32_t run[4] = {winners.x, winners.y, winners.z, winners.w};
#pragma unroll
      for (int round = 0; round < 16; ++round) {
        uint32_t const winner = warp_max(run[0]);
        if (lane == 0) {
          scratch.slots[round] = winner;
        }
        bool const won = run[0] == winner;
#pragma unroll
        for (int item = 0; item < 3; ++item) {
          run[item] = won ? run[item + 1] : run[item];
        }
        run[3] = won ? 0u : run[3];
      }
    }
    sync<Threads>();
  }
  if (tid < 15) {
    scratch.selected[tid] = static_cast<int>(~scratch.slots[tid] & 0xffffu);
  }
  sync<Threads>();
  if (high_key != low_key && (scratch.slots[14] >> 16) == (scratch.slots[15] >> 16)) {
    refine_boundary<Threads>(scores, ranked_pages, scratch, scale, bias);
  }
}

template <int Width = 2>
__device__ __forceinline__ void rank_block(float const *scores, int ranked_pages,
                                           Scratch &scratch) {
  if (ranked_pages <= Width * 256 || Width == 32) {
    rank_row<256, Width>(scores, ranked_pages, scratch);
  } else if constexpr (Width < 32) {
    rank_block<(Width < 8 ? Width * 2 : Width + 4)>(scores, ranked_pages, scratch);
  }
}

__device__ __forceinline__ void select(float const *scores, int visible_pages, Scratch &scratch,
                                       int *output) {
  int const tid = static_cast<int>(threadIdx.x);
  if (tid >= 256) {
    return;
  }
  if (visible_pages <= 16) {
    if (tid < 16) {
      output[tid] = tid < visible_pages ? tid : -1;
    }
    __syncwarp();
    return;
  }
  int const ranked_pages = visible_pages - 1;
  if (ranked_pages <= 256) {
    if (tid >= 32) {
      return;
    }
    if (ranked_pages <= 32) {
      rank_row<32, 1>(scores, ranked_pages, scratch);
    } else if (ranked_pages <= 64) {
      rank_row<32, 2>(scores, ranked_pages, scratch);
    } else if (ranked_pages <= 128) {
      rank_row<32, 4>(scores, ranked_pages, scratch);
    } else {
      rank_row<32, 8>(scores, ranked_pages, scratch);
    }
  } else {
    rank_block(scores, ranked_pages, scratch);
  }
  if (tid < 32) {
    int page = tid < 15 ? scratch.selected[tid] : (tid == 15 ? ranked_pages : INT_MAX);
#pragma unroll
    for (int span = 2; span <= 16; span *= 2) {
#pragma unroll
      for (int stride = span / 2; stride > 0; stride /= 2) {
        int const peer = __shfl_xor_sync(0xffffffffu, page, stride);
        page = ((tid & span) == 0) == ((tid & stride) == 0) ? min(page, peer) : max(page, peer);
      }
    }
    if (tid < 16) {
      output[tid] = page;
    }
    __syncwarp();
  }
}

} // namespace cutlass::m3_decode::collective::runtime_topk
