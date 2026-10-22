#pragma once

#include <climits>
#include <cstdint>

#include "attn/kernel/include/sm100_m3_decode_problem.hpp"
#include "common/kernel/include/sm100_m3_decode_pipeline.hpp"
#include "common/kernel/include/sm100_m3_decode_storage.hpp"
#include "common/kernel/include/sm100_m3_decode_traits.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/cutlass.h"

namespace cutlass::m3_decode::collective {

template <class Traits> struct Sm100M3DecodeAttentionPageOffsets {
  using Storage = SharedStorage<Traits>;
  using Params = Sm100M3DecodeKernelParams<Traits>;
  using Barriers = BarrierLayout<Traits>;

  struct State {
    int group_event = 0;
  };

  CUTLASS_DEVICE static uint64_t *page_offsets_full_barrier(Storage &storage, int stage) {
    return storage.attention_pipelines().ptr(Barriers::kPageOffsetsFullArv32 + stage);
  }

  CUTLASS_DEVICE static uint64_t *page_offsets_empty_barrier(Storage &storage, int stage) {
    return storage.attention_pipelines().ptr(Barriers::kPageOffsetsEmptyArv32 + stage);
  }

  CUTLASS_DEVICE static int kv_length_for_batch(Params const &params, int batch_idx,
                                                int kv_head_idx, int q_token_idx) {
    return m3_decode_kv_length_for_batch<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
  }

  CUTLASS_DEVICE static int page_count_for_batch(Params const &params, int batch_idx,
                                                 int kv_head_idx, int q_token_idx) {
    return m3_decode_kv_tile_count_for_batch<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
  }

  CUTLASS_DEVICE static int page_group_count(Params const &params, int batch_idx, int kv_head_idx,
                                             int q_token_idx) {
    int const pages = page_count_for_batch(params, batch_idx, kv_head_idx, q_token_idx);
    return (pages + Traits::kPageOffsetsPerStage - 1) / Traits::kPageOffsetsPerStage;
  }

  CUTLASS_DEVICE static Sm100M3DecodeKvTileRange
  page_range_for_tile_range(Params const &params, int batch_idx, int kv_head_idx, int q_token_idx,
                            int kv_tile_begin, int kv_tile_end) {
    int const pages = page_count_for_batch(params, batch_idx, kv_head_idx, q_token_idx);
    return make_kv_tile_range(pages, kv_tile_begin, kv_tile_end);
  }

  CUTLASS_DEVICE static uint32_t producer_empty_phase(int group_event) {
    return static_cast<uint32_t>(1 ^ ((group_event / Traits::kNumPageOffsetStages) & 1));
  }

  CUTLASS_DEVICE static uint32_t consumer_full_phase(int group_event) {
    return static_cast<uint32_t>((group_event / Traits::kNumPageOffsetStages) & 1);
  }

  CUTLASS_DEVICE static void cp_async_page_offset(int32_t *dst, int32_t const *src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    cutlass::arch::cp_async<sizeof(int32_t), cutlass::arch::CacheOperation::Always>(dst, src);
#else
    *dst = *src;
#endif
  }

  CUTLASS_DEVICE static void commit_cp_async_group(uint64_t *full_barrier) {
    cutlass::arch::cpasync_barrier_arrive(full_barrier);
    Sm100M3DecodeBarrier::arrive(full_barrier);
  }

  CUTLASS_DEVICE static void commit_scalar_group(uint64_t *full_barrier) {
    cutlass::arch::fence_view_async_shared();
    Sm100M3DecodeBarrier::arrive(full_barrier);
  }

  CUTLASS_DEVICE void load_group(Storage &storage, Params const &params, int batch_idx,
                                 int kv_head_idx, int q_token_idx, int group_event, int page_group,
                                 int pages_to_stage, int lane_idx) const {
    int const stage = group_event % Traits::kNumPageOffsetStages;
    uint64_t *empty_barrier = page_offsets_empty_barrier(storage, stage);
    uint64_t *full_barrier = page_offsets_full_barrier(storage, stage);

    Sm100M3DecodeBarrier::wait(empty_barrier, producer_empty_phase(group_event), 600);

    int const page_idx = page_group * Traits::kPageOffsetsPerStage + lane_idx;
    int lookup = batch_idx * params.kv_page_stride;
    if (page_idx < pages_to_stage) {
      lookup += storage.indexer_topk[page_idx];
    }
    cp_async_page_offset(&storage.smem_page_offsets_kv.data[stage][lane_idx],
                         params.kv_indices_ptr + lookup);
    commit_cp_async_group(full_barrier);
  }

  CUTLASS_DEVICE void run_tile(Storage &storage, Params const &params, int batch_idx,
                               int kv_head_idx, int q_token_idx, int lane_idx, State &state,
                               int full_tiles, int kv_tile_begin = 0,
                               int kv_tile_end = INT_MAX) const {
    Sm100M3DecodeKvTileRange const page_range =
        make_kv_tile_range(full_tiles, kv_tile_begin, kv_tile_end);
    if (page_range.count <= 0) {
      return;
    }
    int const begin_group = page_range.begin / Traits::kPageOffsetsPerStage;
    int const end_group =
        (page_range.end + Traits::kPageOffsetsPerStage - 1) / Traits::kPageOffsetsPerStage;
    CUTLASS_PRAGMA_NO_UNROLL
    for (int group = begin_group; group < end_group; ++group) {
      load_group(storage, params, batch_idx, kv_head_idx, q_token_idx, state.group_event++, group,
                 page_range.end, lane_idx);
    }
  }

  CUTLASS_DEVICE void operator()(Storage &storage, Params const &params, int batch_idx,
                                 int kv_head_idx, int q_token_idx, int lane_idx) const {
    State state;
    int const full_tiles = page_count_for_batch(params, batch_idx, kv_head_idx, q_token_idx);
    run_tile(storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx, state, full_tiles);
  }
};

} // namespace cutlass::m3_decode::collective
