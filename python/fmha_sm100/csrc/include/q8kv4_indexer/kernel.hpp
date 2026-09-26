// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include "q8kv4_indexer/collective.hpp"

namespace q8kv4_indexer {

template <class Traits>
CUTE_DEVICE int indexer_gemm_scheduler_tile_count(IndexerGemmParams const &params, int batch_idx) {
  int const kv_length = params.kv_lengths_ptr[batch_idx];
  int const page_count_unclamped =
      kv_length > 0 ? (kv_length + Traits::kPageTokens - 1) / Traits::kPageTokens : 0;
  int const page_count = min(page_count_unclamped, params.max_pages);
  int const scored_pages = max(page_count - 1, 0);
  return (scored_pages + Traits::kPagesPerWorkTile - 1) / Traits::kPagesPerWorkTile;
}

template <class Traits>
__global__ void __launch_bounds__(Traits::kPrepareThreads)
    prepare_indexer_gemm_scheduler(const __grid_constant__ IndexerGemmParams params) {
  __shared__ int32_t warp_prefix[Traits::kPrepareWarps];
  int const thread_idx = static_cast<int>(threadIdx.x);
  int const lane_idx = thread_idx & 31;
  int const warp_idx = thread_idx >> 5;
  int const active_warps = static_cast<int>(blockDim.x) >> 5;
  int tile_count = 0;
  if (thread_idx < params.batch) {
    tile_count = indexer_gemm_scheduler_tile_count<Traits>(params, thread_idx);
  }
  for (int offset = 1; offset < 32; offset <<= 1) {
    int const addend = __shfl_up_sync(0xffffffffu, tile_count, offset);
    if (lane_idx >= offset) {
      tile_count += addend;
    }
  }
  if (lane_idx == 31) {
    warp_prefix[warp_idx] = tile_count;
  }
  __syncthreads();
  if (warp_idx == 0) {
    int warp_count = lane_idx < active_warps ? warp_prefix[lane_idx] : 0;
    for (int offset = 1; offset < 32; offset <<= 1) {
      int const addend = __shfl_up_sync(0xffffffffu, warp_count, offset);
      if (lane_idx >= offset) {
        warp_count += addend;
      }
    }
    if (lane_idx < active_warps) {
      warp_prefix[lane_idx] = warp_count;
    }
  }
  __syncthreads();
  tile_count += warp_idx > 0 ? warp_prefix[warp_idx - 1] : 0;
  if (thread_idx == 0) {
    params.scheduler_workspace_ptr[0] = 0;
    params.scheduler_workspace_ptr[Traits::kPrepareThreads + 1] = 0;
  }
  if (thread_idx < params.batch) {
    params.scheduler_workspace_ptr[thread_idx + 1] = tile_count;
  }
}

template <class Traits>
__global__ void __launch_bounds__(256)
    prepare_indexer_gemm_scheduler_counts(const __grid_constant__ IndexerGemmParams params) {
  int const batch_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (batch_idx == 0) {
    params.scheduler_workspace_ptr[-1] = 0;
    params.scheduler_workspace_ptr[0] = 0;
  }
  if (batch_idx < params.batch) {
    params.scheduler_workspace_ptr[batch_idx + 1] =
        indexer_gemm_scheduler_tile_count<Traits>(params, batch_idx);
  }
}

__global__ void reset_indexer_gemm_scheduler_counter(int32_t *counter) { *counter = 0; }

template <class Traits, int SchedulerCounterOffset>
__global__ void __launch_bounds__(Traits::kThreads)
    indexer_gemm_kernel(const __grid_constant__ IndexerGemmParams params) {
  using Collective = IndexerGemmCollective<Traits, SchedulerCounterOffset>;
  using SharedStorage = typename Collective::SharedStorage;
  extern __shared__ __align__(1024) uint8_t shared_memory[];
  SharedStorage &storage = *reinterpret_cast<SharedStorage *>(shared_memory);
  Collective::run(params, storage);
}

template <class Traits, int SchedulerCounterOffset> struct IndexerGemmKernel {
  using Collective = IndexerGemmCollective<Traits, SchedulerCounterOffset>;
  using SharedStorage = typename Collective::SharedStorage;
  static constexpr int kThreadCount = Traits::kThreads;
  static constexpr int kSharedStorageBytes = sizeof(SharedStorage);
};

} // namespace q8kv4_indexer
