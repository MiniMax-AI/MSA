// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "fmha_tile_scheduler.hpp"
#include "sm100_fmha_q8kv4_traits.hpp"

namespace cutlass::fmha::collective {

template <class Traits> struct TmemLayout {
  enum : uint32_t {
    kS0 = 0,
    kS1 = 16,
    kStats0 = 32,
    kStats1 = 64,
    kO0 = 96,
    kPreparedKv = 112,
    kEnd = Traits::kNumTmemCols,
  };

  static constexpr uint32_t kColsPerPreparedKvStage =
      (Traits::kHeadDim * Traits::kTileKv) / (4 * 128);
  static constexpr uint32_t kPreparedKvCols = kEnd - kPreparedKv;
  static constexpr uint32_t kPreparedKvStages = kPreparedKvCols / kColsPerPreparedKvStage;

  static_assert(kEnd == 512, "SM100 TMEM has 512 columns per SM.");
  static_assert(kColsPerPreparedKvStage == 32,
                "q8kv4 FMHA forward expects 32 TMEM columns per KV stage.");
  static_assert(kPreparedKvStages == Traits::kNumStagesTransform,
                "prepared KV TMEM staging must match the TransformKv pipeline.");
};

template <class Traits> struct BarrierLayout {
  // Slots are grouped by arrive count; offsets follow from the stage counts. With the SM100
  // defaults (Q 2, raw KV 8, transform 12, page offsets 6) this reproduces the original table.
  // Count 1.
  static constexpr int kArv1Offset = 0;
  static constexpr int kQFullArv1 = kArv1Offset + 0;
  static constexpr int kQEmptyArv1 = kQFullArv1 + Traits::kNumStagesQ;
  static constexpr int kKvFullArv1 = kQEmptyArv1 + Traits::kNumStagesQ;
  static constexpr int kKvEmptyArv1 = kKvFullArv1 + Traits::kNumStagesRawKv + 1;
  static constexpr int kTransformedKvEmptyArv1 = kKvEmptyArv1 + Traits::kNumStagesRawKv + 1;
  static constexpr int kWorkIdStorageFullArv1 =
      kTransformedKvEmptyArv1 + Traits::kNumStagesTransform;
  static constexpr int kS0FullArv1 = kWorkIdStorageFullArv1 + 2;
  static constexpr int kPEmptyArv1 = kS0FullArv1 + 2;
  static constexpr int kOFullArv1 = kPEmptyArv1 + 2;
  static constexpr int kMergeStageFullArv1 = kOFullArv1 + 1;
  // Count 32.
  static constexpr int kArv32Offset = kMergeStageFullArv1 + 1;
  static constexpr int kPageOffsetsFullArv32 = kArv32Offset + 0;
  static constexpr int kPageOffsetsEmptyArv32 =
      kPageOffsetsFullArv32 + Traits::kNumPageOffsetStages;
  static constexpr int kWorkIdThrottleFullArv32 =
      kPageOffsetsEmptyArv32 + Traits::kNumPageOffsetStages;
  static constexpr int kWorkIdThrottleEmptyArv32 = kWorkIdThrottleFullArv32 + 2;
  // Count 128.
  static constexpr int kArv128Offset = kWorkIdThrottleEmptyArv32 + 2;
  static constexpr int kTransformedKvFullArv128 = kArv128Offset + 0;
  static constexpr int kS0EmptyArv128 = kTransformedKvFullArv128 + Traits::kNumStagesTransform;
  static constexpr int kSoftmaxLocalFullArv128 = kS0EmptyArv128 + 2;
  static constexpr int kSoftmaxLocalEmptyArv128 = kSoftmaxLocalFullArv128 + 2;
  static constexpr int kPFullArv128 = kSoftmaxLocalEmptyArv128 + 2;
  static constexpr int kOEmptyArv128 = kPFullArv128 + 2;
  // Count 512.
  static constexpr int kArv512Offset = kOEmptyArv128 + 1;
  static constexpr int kWorkIdStorageEmptyArv512 = kArv512Offset + 0;
  // Selection ring (appended so the slot table above keeps its original offsets): full has one
  // producer arrival, empty one arrival per consuming warp.
  static constexpr int kSelectionFullArv1 = kWorkIdStorageEmptyArv512 + 2;
  static constexpr int kSelectionEmptyArvWarps = kSelectionFullArv1 + Traits::kNumSelectionStages;
  static constexpr int kNumUsedSlots = kSelectionEmptyArvWarps + Traits::kNumSelectionStages;

  static_assert(kNumUsedSlots <= Traits::kNumBarrierSlots,
                "barrier layout must fit the pipeline storage.");
  static_assert(Traits::kNumStagesRawKv != 8 || Traits::kNumStagesTransform != 12 ||
                    (kKvEmptyArv1 == 13 && kTransformedKvEmptyArv1 == 22 && kArv32Offset == 42 &&
                     kArv128Offset == 58 && kArv512Offset == 79),
                "default layout must match the original SM100 slot table plus the merge slot.");
};

template <class Traits> struct SharedStorage {
  using Barriers = BarrierLayout<Traits>;
  using SchedulerStorage = typename Sm100FmhaScheduler<Traits>::SharedStorage;

  struct alignas(1024) SmemQ {
    uint8_t data[Traits::kNumStagesQ][Traits::kSmemQBytesPerStage];

    CUTLASS_DEVICE uint8_t *stage_ptr(int stage) { return data[stage]; }
  };

  struct alignas(1024) SmemKv {
    uint8_t data[Traits::kNumStagesRawKv][Traits::kRawKvStageBytes];

    CUTLASS_DEVICE uint8_t *stage_ptr(int stage) { return data[stage]; }

    CUTLASS_DEVICE uint8_t *raw_ptr(int stage) { return data[stage]; }

    CUTLASS_DEVICE uint8_t *scale_ptr(int stage) {
      return data[stage] + Traits::kRawKvDataBytesPerStage;
    }

    CUTLASS_DEVICE uint8_t *scale_scratch_ptr(int stage) {
      return data[stage] + Traits::kRawKvDataBytesPerStage + Traits::kRawKvScaleBytesPerStage;
    }
  };

  struct alignas(128) SmemPageOffsetsKv {
    int32_t data[Traits::kNumPageOffsetStages][Traits::kPageOffsetsPerStage];
  };

  struct alignas(1024) SmemP {
    uint8_t data[Traits::kSmemPBytes];

    CUTLASS_DEVICE uint8_t *stage_ptr(int stage) {
      return data + stage * Traits::kSmemQBytesPerStage;
    }
  };

  struct alignas(128) SmemO {
    uint8_t data[Traits::kSmemOBytes];
  };

  // Staging area of the balanced-schedule merge: the workspace slots of one item (O block then
  // LSEs each), filled by bulk copies and read back by the fold.
  struct alignas(128) SmemMergeStage {
    uint8_t data[Traits::kMergeMaxSlots * Traits::kMergeSlotBytes];

    CUTLASS_DEVICE uint8_t *slot_o(int slot) { return data + slot * Traits::kMergeSlotBytes; }

    CUTLASS_DEVICE float *slot_lse(int slot) {
      return reinterpret_cast<float *>(slot_o(slot) + Traits::kMergeSlotOBytes);
    }
  };

  struct alignas(16) SmemWarpGroupReduction {
    float data[Traits::kWarpGroupReductionFloats];
  };

  // One packed selection record per ring stage (see Sm100FmhaSelectionRing::pack).
  struct alignas(16) SmemSelection {
    uint32_t data[Traits::kNumSelectionStages];
  };

  struct PipelineStorage {
    uint64_t barriers[Traits::kNumBarrierSlots];

    CUTLASS_DEVICE uint64_t *ptr(int slot) { return barriers + slot; }

    CUTLASS_DEVICE static void init_mbarrier(uint64_t *barrier, uint32_t arrive_count) {
      cutlass::arch::ClusterBarrier::init(barrier, arrive_count);
    }

    CUTLASS_DEVICE void init_pipeline_barriers(int warp_idx, int lane_idx) {
      if (warp_idx != 0) {
        return;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int slot = lane_idx; slot < Traits::kNumBarrierSlots;
           slot += cutlass::NumThreadsPerWarp) {
        uint32_t const arrive_count =
            slot >= Barriers::kSelectionEmptyArvWarps ? Traits::kSelectionConsumerWarps
            : slot >= Barriers::kSelectionFullArv1    ? 1u
            : slot < Barriers::kArv32Offset           ? 1u
            : slot < Barriers::kArv128Offset          ? 32u
            : slot < Barriers::kArv512Offset          ? 128u
                                                      : 512u;
        init_mbarrier(barriers + slot, arrive_count);
      }
    }
  };

  SmemQ smem_q;
  SmemKv smem_kv;
  SmemP smem_p;
  SmemPageOffsetsKv smem_page_offsets_kv;
  SmemSelection smem_selection;
  SmemO smem_o;
  SmemMergeStage smem_merge_stage;
  SmemWarpGroupReduction smem_softmax_red0;
  SmemWarpGroupReduction smem_softmax_red1;
  SmemWarpGroupReduction smem_corr_red1;
  alignas(16) SchedulerStorage scheduler_storage;
  uint32_t tmem_sw_state[Traits::kTmemSwStateBytes / sizeof(uint32_t)];
  // Correction warpgroup broadcast slot: last-arriver election of the in-kernel split merge.
  alignas(16) int32_t merge_flag[4];
  PipelineStorage pipelines;

  CUTLASS_DEVICE uint32_t *tmem_state_ptr() { return tmem_sw_state; }

  CUTLASS_DEVICE int32_t *merge_flag_ptr() { return merge_flag; }


  static_assert(sizeof(SmemQ) == Traits::kNumStagesQ * Traits::kSmemQBytesPerStage,
                "SmemQ must match the q8kv4 FMHA forward layout.");
  static_assert(sizeof(SmemKv) == Traits::kNumStagesRawKv * Traits::kRawKvStageBytes,
                "SmemKv must match the q8kv4 FMHA forward layout.");
  static_assert(sizeof(SmemP) == Traits::kSmemPBytes,
                "SmemP must match the q8kv4 FMHA forward layout.");
  static_assert(sizeof(SmemPageOffsetsKv) == 768,
                "SmemPageOffsetsKv must match the q8kv4 FMHA forward layout.");
  static_assert(sizeof(SmemO) == Traits::kSmemOBytes,
                "SmemO must match the q8kv4 FMHA forward layout.");
  static_assert(sizeof(SmemMergeStage) == Traits::kMergeMaxSlots * Traits::kMergeSlotBytes,
                "merge staging must hold every slot of an item.");
  static_assert(sizeof(SmemWarpGroupReduction) == 256,
                "warpgroup reduction buffer must match the q8kv4 FMHA forward layout.");
  static_assert(sizeof(SchedulerStorage) <= 128,
                "scheduler storage should remain a small work-id staging area.");
  static_assert(sizeof(PipelineStorage) == Traits::kBarrierStorageBytes,
                "barrier storage must match the q8kv4 FMHA forward layout.");
};

// Dynamic shared memory of a launch: the whole struct. The 1 KB and 128 B alignment of the
// tiles pads the layout, so the sum of the member sizes falls short of where the barriers end.
template <class Traits> inline constexpr int kSharedStorageBytes = sizeof(SharedStorage<Traits>);

} // namespace cutlass::fmha::collective
