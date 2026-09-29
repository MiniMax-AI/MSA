// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "utils.cuh"

namespace flashinfer {

constexpr float kPerIterOverhead = 43;
constexpr float kTileGlobalOverhead = 110;
constexpr float kSmGlobalOverhead = 165;
constexpr int kMaxSms = 256;

struct PlanSharedState {
  int sm_cost[kMaxSms];
  int sm_task_count[kMaxSms];
};

// Layout: [63:32] query tile | [31:16] query head | [15:0] batch.
__device__ __forceinline__ uint64_t pack_work_info(int qo_tile, int head, int batch) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(qo_tile)) << 32) |
         (static_cast<uint64_t>(head & 0xffff) << 16) | static_cast<uint64_t>(batch & 0xffff);
}

__device__ __forceinline__ uint64_t pack_work_range(int start, int end) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(end)) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(start));
}

__device__ __forceinline__ int compute_kv_iters(int batch_idx, int qo_tile_idx, int *qo_lens,
                                                int *kv_lens, int *qo_offsets, int qo_tile_size,
                                                int kv_tile_size, bool causal,
                                                int pack_factor = 1) {
  int kv_len = kv_lens[batch_idx];
  if (!causal) {
    return (kv_len + kv_tile_size - 1) / kv_tile_size;
  }
  int offset_q = qo_offsets ? qo_offsets[batch_idx] : (kv_len - qo_lens[batch_idx]);
  int packed_q_end = (qo_tile_idx + 1) * qo_tile_size;
  int q_end = pack_factor > 1 ? (packed_q_end - 1) / pack_factor + 1 : packed_q_end;
  int effective_kv_len = q_end + offset_q;
  effective_kv_len = ::min(effective_kv_len, kv_len);
  if (effective_kv_len <= 0) {
    return 0;
  }
  return (effective_kv_len + kv_tile_size - 1) / kv_tile_size;
}

__device__ void direct_greedy(PlanSharedState &state, int num_buckets, int *qo_lens, int *kv_lens,
                              int *qo_offsets, int qo_tile_size, int kv_tile_size, int batch_size,
                              int num_heads, bool causal, int num_kv_splits,
                              uint64_t *packed_work_range, uint64_t *packed_work_info,
                              int *kv_tile_begin_indices, int *kv_tile_end_indices,
                              int *kv_split_indices, int *num_kv_splits_per_row,
                              int *qo_segment_offsets, int pack_factor) {
  int tid = static_cast<int>(threadIdx.x);

  __shared__ int max_qo_tiles;
  if (tid == 0) {
    int max_tiles = 0;
    for (int batch = 0; batch < batch_size; ++batch) {
      max_tiles = ::max(max_tiles, ceil_div(qo_lens[batch], qo_tile_size));
    }
    max_qo_tiles = max_tiles;
  }
  __syncthreads();

  if (tid < num_buckets) {
    state.sm_cost[tid] = kSmGlobalOverhead;
    state.sm_task_count[tid] = 0;
  }
  __syncthreads();

  // Pass one derives the work-list offsets from the same greedy assignment used below.
  for (int qo_tile = max_qo_tiles - 1; qo_tile >= 0; --qo_tile) {
    for (int batch = 0; batch < batch_size; ++batch) {
      if (qo_tile >= ceil_div(qo_lens[batch], qo_tile_size)) {
        continue;
      }
      int kv_iters = compute_kv_iters(batch, qo_tile, qo_lens, kv_lens, qo_offsets, qo_tile_size,
                                      kv_tile_size, causal, pack_factor);
      if (kv_iters <= 0) {
        continue;
      }
      int split_count = num_kv_splits <= 1 ? 1 : ::min(num_kv_splits, kv_iters);
      int split_extent = (kv_iters + split_count - 1) / split_count;
      for (int split = 0; split < split_count; ++split) {
        int kv_begin = split * split_extent;
        int kv_end = ::min(kv_begin + split_extent, kv_iters);
        int tile_cost =
            static_cast<int>(kPerIterOverhead * (kv_end - kv_begin) + kTileGlobalOverhead);

        for (int head_offset = 0; head_offset < num_heads;) {
          int assignment_count = ::min(num_heads - head_offset, num_buckets);
          int my_cost = tid < num_buckets ? state.sm_cost[tid] : 0x7fffffff;
          int my_rank = 0;
          if (tid < num_buckets) {
            for (int bucket = 0; bucket < num_buckets; ++bucket) {
              int other_cost = state.sm_cost[bucket];
              if (other_cost < my_cost || (other_cost == my_cost && bucket < tid)) {
                ++my_rank;
              }
            }
          }
          bool assigned = tid < num_buckets && my_rank < assignment_count;
          __syncthreads();
          if (assigned) {
            state.sm_cost[tid] += tile_cost;
            ++state.sm_task_count[tid];
          }
          __syncthreads();
          head_offset += assignment_count;
        }
      }
    }
  }

  __shared__ int work_offsets[kMaxSms + 1];
  if (tid == 0) {
    int offset = 0;
    for (int bucket = 0; bucket < num_buckets; ++bucket) {
      work_offsets[bucket] = offset;
      offset += state.sm_task_count[bucket];
      state.sm_task_count[bucket] = 0;
    }
    work_offsets[num_buckets] = offset;
  }
  __syncthreads();

  if (tid < num_buckets) {
    state.sm_cost[tid] = kSmGlobalOverhead;
  }
  __syncthreads();

  // Pass two repeats the deterministic assignment and materializes the work list.
  for (int qo_tile = max_qo_tiles - 1; qo_tile >= 0; --qo_tile) {
    for (int batch = 0; batch < batch_size; ++batch) {
      if (qo_tile >= ceil_div(qo_lens[batch], qo_tile_size)) {
        continue;
      }
      int kv_iters = compute_kv_iters(batch, qo_tile, qo_lens, kv_lens, qo_offsets, qo_tile_size,
                                      kv_tile_size, causal, pack_factor);
      if (kv_iters <= 0) {
        continue;
      }
      int split_count = num_kv_splits <= 1 ? 1 : ::min(num_kv_splits, kv_iters);
      int split_extent = (kv_iters + split_count - 1) / split_count;
      if (tid == 0 && num_kv_splits_per_row && qo_segment_offsets) {
        int segment_offset = qo_segment_offsets[batch];
        int row_begin = qo_tile * qo_tile_size;
        int row_end = ::min(row_begin + qo_tile_size, qo_lens[batch]);
        for (int row = row_begin; row < row_end; ++row) {
          num_kv_splits_per_row[segment_offset + row] = split_count;
        }
      }

      for (int split = 0; split < split_count; ++split) {
        int kv_begin = split * split_extent;
        int kv_end = ::min(kv_begin + split_extent, kv_iters);
        int tile_cost =
            static_cast<int>(kPerIterOverhead * (kv_end - kv_begin) + kTileGlobalOverhead);

        for (int head_offset = 0; head_offset < num_heads;) {
          int assignment_count = ::min(num_heads - head_offset, num_buckets);
          int my_cost = tid < num_buckets ? state.sm_cost[tid] : 0x7fffffff;
          int my_rank = 0;
          if (tid < num_buckets) {
            for (int bucket = 0; bucket < num_buckets; ++bucket) {
              int other_cost = state.sm_cost[bucket];
              if (other_cost < my_cost || (other_cost == my_cost && bucket < tid)) {
                ++my_rank;
              }
            }
          }
          bool assigned = tid < num_buckets && my_rank < assignment_count;
          __syncthreads();
          if (assigned) {
            state.sm_cost[tid] += tile_cost;
            int position = work_offsets[tid] + state.sm_task_count[tid]++;
            packed_work_info[position] = pack_work_info(qo_tile, head_offset + my_rank, batch);
            if (kv_tile_begin_indices) {
              kv_tile_begin_indices[position] = kv_begin;
            }
            if (kv_tile_end_indices) {
              kv_tile_end_indices[position] = kv_end;
            }
            if (kv_split_indices) {
              kv_split_indices[position] = split;
            }
          }
          __syncthreads();
          head_offset += assignment_count;
        }
      }
    }
  }

  if (tid < num_buckets) {
    packed_work_range[tid] =
        pack_work_range(work_offsets[tid], work_offsets[tid] + state.sm_task_count[tid]);
  }
}

__global__ void plan_kernel(int *qo_segment_offsets, int *qo_lens, int *kv_lens,
                            uint64_t *packed_work_range, uint64_t *packed_work_info,
                            int qo_tile_size, int kv_tile_size, int batch_size, int num_heads,
                            int num_buckets, bool causal, int *qo_offsets, int num_kv_splits,
                            int *kv_tile_begin_indices, int *kv_tile_end_indices,
                            int *kv_split_indices, int *num_kv_splits_per_row, float *workspace_lse,
                            int lse_total_size, int pack_factor) {
  __shared__ PlanSharedState state;

  if (num_kv_splits_per_row && qo_segment_offsets) {
    int total_rows = qo_segment_offsets[batch_size];
    for (int row = static_cast<int>(threadIdx.x); row < total_rows;
         row += static_cast<int>(blockDim.x)) {
      num_kv_splits_per_row[row] = 1;
    }
    __syncthreads();
  }

  if (workspace_lse) {
    for (int index = static_cast<int>(threadIdx.x); index < lse_total_size;
         index += static_cast<int>(blockDim.x)) {
      workspace_lse[index] = -INFINITY;
    }
  }

  direct_greedy(state, num_buckets, qo_lens, kv_lens, qo_offsets, qo_tile_size, kv_tile_size,
                batch_size, num_heads, causal, num_kv_splits, packed_work_range, packed_work_info,
                kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices, num_kv_splits_per_row,
                qo_segment_offsets, pack_factor);
}

cudaError_t plan_kernel_wrapper(int *qo_segment_offsets, int *qo_lens, int *kv_lens,
                                uint64_t *packed_work_range, uint64_t *packed_work_info,
                                int qo_tile_size, int kv_tile_size, int batch_size, int num_heads,
                                int num_buckets, bool causal, int *qo_offsets, cudaStream_t stream,
                                int num_kv_splits = 1, int *kv_tile_begin_indices = nullptr,
                                int *kv_tile_end_indices = nullptr, int *kv_split_indices = nullptr,
                                int *num_kv_splits_per_row = nullptr,
                                float *workspace_lse = nullptr, int lse_total_size = 0,
                                int pack_factor = 1) {
  plan_kernel<<<1, kMaxSms, 0, stream>>>(
      qo_segment_offsets, qo_lens, kv_lens, packed_work_range, packed_work_info, qo_tile_size,
      kv_tile_size, batch_size, num_heads, num_buckets, causal, qo_offsets, num_kv_splits,
      kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices, num_kv_splits_per_row,
      workspace_lse, lse_total_size, pack_factor);
  FLASHINFER_CUDA_CALL(cudaGetLastError());
  return cudaSuccess;
}

} // namespace flashinfer
