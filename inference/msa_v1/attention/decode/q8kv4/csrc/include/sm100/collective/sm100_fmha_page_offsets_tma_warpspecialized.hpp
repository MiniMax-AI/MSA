#pragma once

#include <climits>
#include <cstdint>

#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "fmha_common.hpp"
#include "sm100_fmha_pipeline.hpp"
#include "sm100_fmha_q8kv4_traits.hpp"
#include "sm100_fmha_storage.hpp"

namespace cutlass::fmha::collective {

template <class Traits> struct Sm100FmhaPageOffsetsTmaWarpspecialized {
  using Storage = SharedStorage<Traits>;
  using Params = Sm100FmhaFwdKernelParams<Traits>;
  using Barriers = BarrierLayout<Traits>;

  struct State {
    int group_event = 0;
  };

  CUTLASS_DEVICE static uint64_t *page_offsets_full_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kPageOffsetsFullArv32 + stage);
  }

  CUTLASS_DEVICE static uint64_t *page_offsets_empty_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kPageOffsetsEmptyArv32 + stage);
  }

  CUTLASS_DEVICE static int page_count_for_batch(Params const &params, int batch_idx,
                                                 int kv_head_idx, int q_token_idx) {
    return fmha_fwd_kv_tile_count_for_batch<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
  }

  CUTLASS_DEVICE static Sm100FmhaKvTileRange
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

  CUTLASS_DEVICE static void commit_scalar_group(uint64_t *full_barrier) {
    cutlass::arch::fence_view_async_shared();
    Sm100FmhaBarrier::arrive(full_barrier);
  }

  CUTLASS_DEVICE void load_group(Storage &storage, Params const &params, int batch_idx,
                                 int kv_head_idx, int q_token_idx, int group_event, int page_group,
                                 int pages_this_batch, int lane_idx) const {
    int const page_idx = page_group * Traits::kPageOffsetsPerStage + lane_idx;
    int lookup = batch_idx * params.kv_page_stride;
    {
      if (page_idx < pages_this_batch && page_idx < params.kv_block_num) {
        int const q_token_global =
            fmha_fwd_q_token_global_index<Traits>(params, batch_idx, q_token_idx);
        int const block_offset =
            (q_token_global * params.num_kv_heads + kv_head_idx) * params.kv_block_num + page_idx;
        int const logical_page = __ldg(params.kv_block_indexes_ptr + block_offset);
        int const page_for_lookup =
            logical_page >= 0 && logical_page < params.kv_page_stride ? logical_page : 0;
        lookup += page_for_lookup;
      }
    }
#if MINIMAX_MSA_Q8KV4_HAS_QMUL4
    // Keep the uniform load on the QMUL4 path; predication regresses its hot loop.
    int const physical_page = __ldg(params.kv_indices_ptr + lookup);
#else
    int physical_page = 0;
    if (page_idx < pages_this_batch) {
      physical_page = __ldg(params.kv_indices_ptr + lookup);
    }
#endif

    // K and V consume the same physical-page mapping. Publish it once and keep
    // the stage live until the final V tile in this page group has consumed it.
    int const stage = group_event % Traits::kNumPageOffsetStages;
    uint64_t *empty_barrier = page_offsets_empty_barrier(storage, stage);
    uint64_t *full_barrier = page_offsets_full_barrier(storage, stage);
    Sm100FmhaBarrier::wait(empty_barrier, producer_empty_phase(group_event), 600);
    storage.smem_page_offsets_kv.data[stage][lane_idx] = physical_page;
    commit_scalar_group(full_barrier);
  }

  CUTLASS_DEVICE void run_tile(Storage &storage, Params const &params, int batch_idx,
                               int kv_head_idx, int q_token_idx, int lane_idx, State &state,
                               int kv_tile_begin = 0, int kv_tile_end = INT_MAX) const {
    Sm100FmhaKvTileRange const page_range = page_range_for_tile_range(
        params, batch_idx, kv_head_idx, q_token_idx, kv_tile_begin, kv_tile_end);
    if (page_range.count <= 0) {
      return;
    }
    int const begin_group = page_range.begin / Traits::kPageOffsetsPerStage;
    int const end_group =
        (page_range.end + Traits::kPageOffsetsPerStage - 1) / Traits::kPageOffsetsPerStage;
    CUTLASS_PRAGMA_NO_UNROLL
    for (int group = begin_group; group < end_group; ++group) {
      load_group(storage, params, batch_idx, kv_head_idx, q_token_idx, state.group_event, group,
                 page_range.end, lane_idx);
      ++state.group_event;
    }
  }

  CUTLASS_DEVICE void operator()(Storage &storage, Params const &params, int batch_idx,
                                 int kv_head_idx, int q_token_idx, int lane_idx) const {
    State state;
    run_tile(storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx, state);
  }
};

} // namespace cutlass::fmha::collective
