// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"
#include "fmha_common.hpp"
#include "sm100_fmha_pipeline.hpp"
#include "sm100_fmha_q8kv4_traits.hpp"
#include "sm100_fmha_storage.hpp"

namespace cutlass::fmha::collective {

// Per-item selection records flow from the page-offsets warp to every other working warp through
// a small shared-memory ring. That warp already loads the query's TopK list for its physical-page
// lookups, so the selection costs it a few ballots; consumers replace their own list and length
// loads with one barrier wait and one shared-memory read. Every role enters its per-item code in
// the same item order, so a per-role event counter indexes the ring.
template <class Traits> struct Sm100FmhaSelectionRing {
  using Storage = SharedStorage<Traits>;
  using Barriers = BarrierLayout<Traits>;

  CUTLASS_DEVICE static uint64_t *full_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kSelectionFullArv1 + stage);
  }

  CUTLASS_DEVICE static uint64_t *empty_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kSelectionEmptyArvWarps + stage);
  }

  CUTLASS_DEVICE static int stage_of(int event) { return event % Traits::kNumSelectionStages; }

  CUTLASS_DEVICE static uint32_t full_phase(int event) {
    return static_cast<uint32_t>((event / Traits::kNumSelectionStages) & 1);
  }

  CUTLASS_DEVICE static uint32_t producer_empty_phase(int event) {
    return static_cast<uint32_t>(1 ^ ((event / Traits::kNumSelectionStages) & 1));
  }

  // Record layout: bits 0-7 selected pages (<= 64), bits 8-15 tail limit (1..128), bits 16-23
  // tail tile + 1 (0 when no tile needs the causal tail). One word keeps each consumer at a
  // single load and shuffle, which matters because fourteen warps run this code per item.
  CUTLASS_DEVICE static uint32_t pack(Sm100FmhaSparseSelection const &selection) {
    return static_cast<uint32_t>(selection.selected_pages) |
           (static_cast<uint32_t>(selection.tail_limit) << 8) |
           (static_cast<uint32_t>(selection.tail_tile + 1) << 16);
  }

  CUTLASS_DEVICE static int selected_pages(uint32_t record) { return record & 0xffu; }
  CUTLASS_DEVICE static int tail_limit(uint32_t record) { return (record >> 8) & 0xffu; }
  CUTLASS_DEVICE static int tail_tile(uint32_t record) {
    return static_cast<int>((record >> 16) & 0xffu) - 1;
  }

  // Producer (page-offsets warp): publish the record it derived for its own item.
  CUTLASS_DEVICE static void publish(Storage &storage, Sm100FmhaSparseSelection const &selection,
                                     int lane_idx, int &event) {
    int const stage = stage_of(event);
    Sm100FmhaBarrier::wait(empty_barrier(storage, stage), producer_empty_phase(event), 700);
    if (lane_idx == 0) {
      storage.smem_selection.data[stage] = pack(selection);
      // mbarrier.arrive releases the store above to the consumers' acquiring waits.
      Sm100FmhaBarrier::arrive(full_barrier(storage, stage));
    }
    ++event;
  }

  // Consumer (whole warp): lane 0 reads the record and releases the stage; its arrive follows
  // its own load in program order with release semantics, so no warp-wide sync is needed.
  CUTLASS_DEVICE static uint32_t consume(Storage &storage, int lane_idx, int &event) {
    int const stage = stage_of(event);
    Sm100FmhaBarrier::wait(full_barrier(storage, stage), full_phase(event), 710);
    uint32_t record = 0;
    if (lane_idx == 0) {
      record = *reinterpret_cast<uint32_t const volatile *>(&storage.smem_selection.data[stage]);
      Sm100FmhaBarrier::arrive(empty_barrier(storage, stage));
    }
    ++event;
    return __shfl_sync(0xffffffffu, record, 0);
  }
};

} // namespace cutlass::fmha::collective
