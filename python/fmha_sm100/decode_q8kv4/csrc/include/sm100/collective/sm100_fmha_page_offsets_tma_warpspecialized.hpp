// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <climits>
#include <cstdint>

#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "fmha_common.hpp"
#include "sm100_fmha_pipeline.hpp"
#include "sm100_fmha_q8kv4_traits.hpp"
#include "sm100_fmha_selection_ring.hpp"
#include "sm100_fmha_storage.hpp"

namespace cutlass::fmha::collective {

template <class Traits> struct Sm100FmhaPageOffsetsTmaWarpspecialized {
  using Storage = SharedStorage<Traits>;
  using Params = Sm100FmhaFwdKernelParams<Traits>;
  using Barriers = BarrierLayout<Traits>;

  struct State {
    int group_event = 0;
    int selection_event = 0;
  };

  CUTLASS_DEVICE static uint64_t *page_offsets_full_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kPageOffsetsFullArv32 + stage);
  }

  CUTLASS_DEVICE static uint64_t *page_offsets_empty_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kPageOffsetsEmptyArv32 + stage);
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

  // `logical_page` is this lane's list entry for the group, loaded once per item by run_tile;
  // `page_base` / `page_count` are the request's range in the flat physical-page list.
  CUTLASS_DEVICE void load_group(Storage &storage, Params const &params, int page_base,
                                 int page_count, int group_event, int page_group,
                                 int pages_this_batch, int logical_page, int lane_idx) const {
    int const page_idx = page_group * Traits::kPageOffsetsPerStage + lane_idx;
    int lookup = page_base;
    if (page_idx < pages_this_batch) {
      int const page_for_lookup = logical_page >= 0 && logical_page < page_count ? logical_page : 0;
      lookup += page_for_lookup;
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
    // Producer of the per-item selection: load the list once, derive the selection, publish it
    // for the other warps, then reuse the same entries for the physical-page lookups.
    Sm100FmhaSelectionLanes const lanes = fmha_fwd_load_selection_lanes<Traits>(
        params, batch_idx, kv_head_idx, q_token_idx, lane_idx);
    // Issued alongside the list loads so the page lookups stay one dependent load deep.
    int const page_base = __ldg(params.kv_indptr_ptr + batch_idx);
    int const page_count = __ldg(params.kv_indptr_ptr + batch_idx + 1) - page_base;
    Sm100FmhaSparseSelection const selection =
        fmha_fwd_sparse_selection<Traits>(params, batch_idx, q_token_idx, lanes);
    Sm100FmhaSelectionRing<Traits>::publish(storage, selection, lane_idx, state.selection_event);
    Sm100FmhaKvTileRange const page_range =
        make_kv_tile_range(selection.selected_pages, kv_tile_begin, kv_tile_end);
    if (page_range.count <= 0) {
      return;
    }
    int const begin_group = page_range.begin / Traits::kPageOffsetsPerStage;
    int const end_group =
        (page_range.end + Traits::kPageOffsetsPerStage - 1) / Traits::kPageOffsetsPerStage;
    CUTLASS_PRAGMA_NO_UNROLL
    for (int group = begin_group; group < end_group; ++group) {
      load_group(storage, params, page_base, page_count, state.group_event, group, page_range.end,
                 group == 0 ? lanes.page0 : lanes.page1, lane_idx);
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
