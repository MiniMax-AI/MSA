#pragma once

#include <climits>
#include <cstdint>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "fmha_common.hpp"
#include "sm100_fmha_page_offsets_tma_warpspecialized.hpp"
#include "sm100_fmha_pipeline.hpp"
#include "sm100_fmha_q8kv4_traits.hpp"
#include "sm100_fmha_storage.hpp"

namespace cutlass::fmha::collective {

template <class Traits> struct Sm100FmhaLoadTmaWarpspecialized {
  using Storage = SharedStorage<Traits>;
  using Barriers = BarrierLayout<Traits>;
  using Params = Sm100FmhaFwdKernelParams<Traits>;
  using PageOffsetLoader = Sm100FmhaPageOffsetsTmaWarpspecialized<Traits>;

  struct Arguments {
    FMHACutlassSM100Params fmha;
  };

  static Params to_underlying_arguments(Arguments const &args, int max_active_ctas) {
    Params params{};
    cudaError_t build_status = FMHACutlassSM100ParamsBuilder<Traits>::build(args.fmha, params);
    if (build_status == cudaSuccess && max_active_ctas > 0) {
      FMHACutlassSM100ParamsBuilder<Traits>::apply_scheduler_policy(args.fmha, max_active_ctas,
                                                                    params);
    }
    return params;
  }

  struct State {
    int q_event = 0;
    int raw_event = 0;
    int page_event_base = 0;
    bool grid_dependency_synchronized = false;
  };

  struct TilePage {
    int page_group = 0;
    int page_event = 0;
    int page_stage = 0;
    int page_lane = 0;
    int token_in_page = 0;
    bool valid = false;
  };

  CUTLASS_DEVICE static uint64_t *q_full_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kQFullArv1 + stage);
  }

  CUTLASS_DEVICE static uint64_t *q_empty_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kQEmptyArv1 + stage);
  }

  CUTLASS_DEVICE static uint64_t *kv_full_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kKvFullArv1 + stage);
  }

  CUTLASS_DEVICE static uint64_t *kv_empty_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kKvEmptyArv1 + stage);
  }

  CUTLASS_DEVICE static uint64_t *page_offsets_full_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kPageOffsetsFullArv32 + stage);
  }

  CUTLASS_DEVICE static uint64_t *page_offsets_empty_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kPageOffsetsEmptyArv32 + stage);
  }

  CUTLASS_DEVICE static int kv_length_for_batch(Params const &params, int batch_idx,
                                                int kv_head_idx, int q_token_idx) {
    return fmha_fwd_kv_length_for_batch<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
  }

  CUTLASS_DEVICE static int kv_tile_count(Params const &params, int batch_idx, int kv_head_idx,
                                          int q_token_idx) {
    return fmha_fwd_kv_tile_count_for_batch<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
  }

  CUTLASS_DEVICE static TilePage resolve_tile_page(Params const &params, int batch_idx,
                                                   int kv_head_idx, int q_token_idx, int tile_idx,
                                                   int kv_len, int page_event_base = 0,
                                                   int page_group_offset = 0) {
    TilePage page;
    int const logical_token = tile_idx * Traits::kTileKv;
    if (logical_token >= kv_len) {
      return page;
    }

    int const logical_page = logical_token / Traits::kPageSize;
    int const pages_this_batch = (kv_len + Traits::kPageSize - 1) / Traits::kPageSize;
    if (logical_page >= pages_this_batch) {
      return page;
    }

    int const global_page_group = logical_page / Traits::kPageOffsetsPerStage;
    page.page_group = global_page_group - page_group_offset;
    if (page.page_group < 0) {
      return TilePage{};
    }
    page.page_event = page_event_base + page.page_group;
    page.page_stage = page.page_event % Traits::kNumPageOffsetStages;
    page.page_lane = logical_page % Traits::kPageOffsetsPerStage;
    page.token_in_page = logical_token - logical_page * Traits::kPageSize;
    page.valid = true;
    return page;
  }

  CUTLASS_DEVICE static void release_page_offsets(Storage &storage, TilePage const &page) {
    if (page.valid) {
      Sm100FmhaBarrier::arrive_cluster_zero(page_offsets_empty_barrier(storage, page.page_stage));
    }
  }

  CUTLASS_DEVICE static bool is_page_offsets_group_tail(int tile_idx) {
    return (tile_idx % Traits::kPageOffsetsPerStage) == (Traits::kPageOffsetsPerStage - 1);
  }

  CUTLASS_DEVICE static bool should_wait_for_page_offsets(int local_tile_idx, int tile_idx) {
    return local_tile_idx == 0 || (tile_idx % Traits::kPageOffsetsPerStage) == 0;
  }

  CUTLASS_DEVICE static bool should_release_v_page_offsets(int local_tile_idx, int tiles,
                                                           int tile_idx) {
    return (local_tile_idx + 1 < tiles) && is_page_offsets_group_tail(tile_idx);
  }

  CUTLASS_DEVICE static void load_q_tile(Storage &storage, Params const &params, int batch_idx,
                                         int kv_head_idx, int q_token_idx, int q_stage,
                                         uint32_t producer_phase, int lane_idx,
                                         bool wait_for_primary_grid, bool lane_predicate) {
    int const stage = q_stage % Traits::kNumStagesQ;
    uint64_t *empty_barrier = q_empty_barrier(storage, stage);
    uint64_t *full_barrier = q_full_barrier(storage, stage);

    Sm100FmhaBarrier::wait(empty_barrier, producer_phase, 500);
    Sm100FmhaBarrier::expect_tx_cluster_lane0(full_barrier, Traits::kSmemQBytesPerStage,
                                              static_cast<uint32_t>(lane_idx));
    if (wait_for_primary_grid) {
      cudaGridDependencySynchronize();
    }
    int const q_token_global =
        fmha_fwd_q_token_global_index<Traits>(params, batch_idx, q_token_idx);
    Sm100FmhaTma::load_5d_predicated(&params.tma.q, storage.smem_q.stage_ptr(stage), full_barrier,
                                     0, 0, kv_head_idx, q_token_global, 0,
                                     static_cast<uint32_t>(lane_predicate));
  }

  CUTLASS_DEVICE static void
  load_raw_kv_tile(Storage &storage, Params const &params, int batch_idx, int kv_head_idx,
                   int q_token_idx, int tile_idx, int kv_len, int kv_stage, uint32_t producer_phase,
                   bool is_v, bool wait_page_offsets_before, bool release_page_offsets_after,
                   int page_event_base, int page_group_offset, int lane_idx, bool lane_predicate) {
    int const stage = kv_stage % Traits::kNumStagesRawKv;
    uint64_t *empty_barrier = kv_empty_barrier(storage, stage);
    uint64_t *full_barrier = kv_full_barrier(storage, stage);
    TilePage const page = resolve_tile_page(params, batch_idx, kv_head_idx, q_token_idx, tile_idx,
                                            kv_len, page_event_base, page_group_offset);

    if (page.valid && wait_page_offsets_before) {
      Sm100FmhaBarrier::wait(page_offsets_full_barrier(storage, page.page_stage),
                             PageOffsetLoader::consumer_full_phase(page.page_event), 503);
    }
    Sm100FmhaBarrier::wait(empty_barrier, producer_phase, 504);

    uint32_t tx_bytes = 0;
    tx_bytes += Traits::kTileKv * (Traits::kHeadDim / 2);
    tx_bytes += Traits::kRawKvScaleBytesPerStage;
    if (!page.valid) {
      if (lane_predicate) {
        Sm100FmhaBarrier::arrive(full_barrier);
      }
    } else {
      int const physical_page = storage.smem_page_offsets_kv.data[page.page_stage][page.page_lane];

      if (tx_bytes == 0) {
        if (lane_predicate) {
          Sm100FmhaBarrier::arrive(full_barrier);
        }
      } else {
        Sm100FmhaBarrier::expect_tx_cluster_lane0(full_barrier, tx_bytes,
                                                  static_cast<uint32_t>(lane_idx));
        CUtensorMap const *kv_desc = is_v ? &params.tma.v : &params.tma.k;
        CUtensorMap const *scale_desc = is_v ? &params.tma.v_scale : &params.tma.k_scale;
        uint32_t const pred = static_cast<uint32_t>(lane_predicate);

        Sm100FmhaTma::load_4d_predicated(kv_desc, storage.smem_kv.raw_ptr(stage), full_barrier, 0,
                                         is_v ? page.token_in_page : page.token_in_page / 2,
                                         kv_head_idx, physical_page, pred);
        Sm100FmhaTma::load_4d_predicated(scale_desc, storage.smem_kv.scale_ptr(stage), full_barrier,
                                         0, page.token_in_page / Traits::kScaleGroupSize,
                                         kv_head_idx, physical_page, pred);
      }
    }

    if (release_page_offsets_after) {
      release_page_offsets(storage, page);
    }
  }

  CUTLASS_DEVICE void operator()(Storage &storage, Params const &params, int batch_idx,
                                 int kv_head_idx, int q_token_idx, int lane_idx,
                                 bool lane_predicate) const {
    State state;
    run_tile(storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx, lane_predicate, state);
  }

  CUTLASS_DEVICE void run_tile(Storage &storage, Params const &params, int batch_idx,
                               int kv_head_idx, int q_token_idx, int lane_idx, bool lane_predicate,
                               State &state, int kv_tile_begin = 0,
                               int kv_tile_end = INT_MAX) const {
    int const full_tiles = kv_tile_count(params, batch_idx, kv_head_idx, q_token_idx);
    int const kv_len = full_tiles * Traits::kTileKv;
    Sm100FmhaKvTileRange const tile_range =
        make_kv_tile_range(full_tiles, kv_tile_begin, kv_tile_end);
    if (tile_range.count <= 0) {
      return;
    }

    auto q_stage = [](int event) { return event % Traits::kNumStagesQ; };
    auto q_empty_phase = [](int event) {
      return static_cast<uint32_t>(1 ^ ((event / Traits::kNumStagesQ) & 1));
    };
    bool const wait_for_primary_grid = !state.grid_dependency_synchronized;
    state.grid_dependency_synchronized = true;

    load_q_tile(storage, params, batch_idx, kv_head_idx, q_token_idx, q_stage(state.q_event),
                q_empty_phase(state.q_event), lane_idx, wait_for_primary_grid, lane_predicate);
    ++state.q_event;

    auto raw_stage = [](int event) { return event % Traits::kNumStagesRawKv; };
    auto raw_empty_phase = [](int event) {
      return static_cast<uint32_t>(1 ^ ((event / Traits::kNumStagesRawKv) & 1));
    };
    int const page_event_base = state.page_event_base;
    int const page_group_offset = tile_range.begin / Traits::kPageOffsetsPerStage;
    int const tiles = tile_range.count;
    auto load_event = [&](int local_tile_idx, bool is_v) {
      int const tile_idx = tile_range.begin + local_tile_idx;
      int const event = state.raw_event++;
      bool const wait_page_offsets_before =
          !is_v && should_wait_for_page_offsets(local_tile_idx, tile_idx);
      bool const release_page_offsets_after =
          is_v ? should_release_v_page_offsets(local_tile_idx, tiles, tile_idx) : false;
      load_raw_kv_tile(storage, params, batch_idx, kv_head_idx, q_token_idx, tile_idx, kv_len,
                       raw_stage(event), raw_empty_phase(event), is_v, wait_page_offsets_before,
                       release_page_offsets_after, page_event_base, page_group_offset, lane_idx,
                       lane_predicate);
    };

    // Raw-KV event stream: first K0, steady V(i)+K(i+1), last V(N-1).
    load_event(0, false);
    CUTLASS_PRAGMA_NO_UNROLL
    for (int tile = 0; tile + 1 < tiles; ++tile) {
      load_event(tile, true);
      load_event(tile + 1, false);
    }
    load_event(tiles - 1, true);
    release_page_offsets(storage, resolve_tile_page(params, batch_idx, kv_head_idx, q_token_idx,
                                                    tile_range.end - 1, kv_len, page_event_base,
                                                    page_group_offset));
    state.page_event_base +=
        ((tile_range.end + Traits::kPageOffsetsPerStage - 1) / Traits::kPageOffsetsPerStage -
         page_group_offset);
  }
};

} // namespace cutlass::fmha::collective
