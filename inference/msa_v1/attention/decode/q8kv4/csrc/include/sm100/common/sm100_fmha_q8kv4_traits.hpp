#pragma once

#include "cutlass/cutlass.h"
#include "fmha_fusion.hpp"
#include <cstdint>

namespace cutlass::fmha::collective {

template <int HeadGroup = 16> struct Sm100FmhaQ8Kv4BaseTraits {
  static_assert(HeadGroup == 8 || HeadGroup == 16, "Q8KV4 decode supports GQA 8 or 16.");
  static constexpr int kTileQ = HeadGroup;
  static constexpr int kTileKv = 128;
  static constexpr int kSplitPlanTileQ = 128;
  static constexpr int kSplitPlanTileKv = 256;
  static constexpr int kHeadDim = 128;
  static constexpr int kPageSize = 128;
  static constexpr int kHeadGroup = HeadGroup;
  static constexpr int kScaleGroupSize = 16;

  static constexpr int kNumThreads = 512;
  static constexpr int kNumWarps = kNumThreads / cutlass::NumThreadsPerWarp;

  static constexpr int kNumLoadWarps = 1;
  static constexpr int kNumPageOffsetsWarps = 1;
  static constexpr int kNumMmaWarps = 1;
  static constexpr int kNumPaddingWarps = 1;
  static constexpr int kNumSoftmaxWarps = 4;
  static constexpr int kNumCorrectionWarps = 4;
  static constexpr int kNumTransformKvWarps = 4;

  static constexpr int kSoftmaxWarpStart = 0;
  static constexpr int kCorrectionWarpStart = 4;
  static constexpr int kMmaWarpStart = 8;
  static constexpr int kPageOffsetsWarpStart = 9;
  static constexpr int kPaddingWarpStart = 10;
  static constexpr int kSchedulerWarpStart = kPaddingWarpStart;
  static constexpr int kLoadWarpStart = 11;
  static constexpr int kTransformKvWarpStart = 12;

  static constexpr int kLoadMaxRegisters = 56;
  static constexpr int kPageOffsetsMaxRegisters = 56;
  static constexpr int kMmaMaxRegisters = 56;
  static constexpr int kPaddingMaxRegisters = 56;
  static constexpr int kSoftmaxMaxRegisters = 184;
  static constexpr int kCorrectionMaxRegisters = 128;
  static constexpr int kTransformKvMaxRegisters = 144;

  static constexpr int kNumStagesQ = 2;
#ifndef MINIMAX_MSA_Q8KV4_RAW_KV_STAGES
#define MINIMAX_MSA_Q8KV4_RAW_KV_STAGES 8
#endif
  // Raw (TMA-landed) KV ring depth in half-tile stages; 8 on SM100. Deeper rings need the
  // oversized shared-memory configuration (see set_smem_attribute()).
  static constexpr int kNumStagesRawKv = MINIMAX_MSA_Q8KV4_RAW_KV_STAGES;
  static constexpr int kNumStagesTransform = 12;
  static constexpr int kNumTmemCols = 512;

  static constexpr int kSmemQBytesPerStage = kTileQ * kHeadDim;
  static constexpr int kRawKvDataBytesPerStage = kTileKv * kHeadDim;
  static constexpr int kRawKvScaleBytesPerStage = kTileKv * (kHeadDim / kScaleGroupSize);
  // FP16 fallback: the V scale prepare pass stores F16 pairs (2x scratch) so the V loop needs no
  // scale conversions. The QMUL4 path consumes E4M3 scales directly and keeps the 1x scratch.
  static constexpr bool kF16VScaleScratch = !MINIMAX_MSA_Q8KV4_HAS_QMUL4;
  static constexpr int kRawKvScaleScratchBytesPerStage =
      (kF16VScaleScratch ? 2 : 1) * kRawKvScaleBytesPerStage;
  static constexpr int kRawKvStageBytes =
      kRawKvDataBytesPerStage + kRawKvScaleBytesPerStage + kRawKvScaleScratchBytesPerStage;
  static constexpr int kRawKvTmaBytes = kTileKv * (kHeadDim / 2) + kRawKvScaleBytesPerStage;

  static constexpr int kNumPageOffsetStages = 6;
  static constexpr int kPageOffsetsPerStage = 32;
  static constexpr int kSmemPBytes = kNumStagesQ * kSmemQBytesPerStage;
  static constexpr int kSmemOBytes = kHeadGroup * kHeadDim * 2;
  // Balanced-schedule merge: a segment's workspace slot is one block of 16 rows x 256 B bf16
  // partial O followed by the 16 row LSEs, so the last arriving segment of an item fetches each
  // slot with one bulk copy into shared memory before folding. The planner cuts an item into at
  // most kMergeMaxSlots segments.
  static constexpr int kMergeMaxSlots = 8;
  static constexpr int kMergeSlotOBytes = kHeadGroup * kHeadDim * 2;
  static constexpr int kMergeSlotLseBytes = kHeadGroup * 4;
  static constexpr int kMergeSlotBytes = kMergeSlotOBytes + kMergeSlotLseBytes;
  static constexpr int kWarpGroupReductionFloats = 64;
  static constexpr int kTmemSwStateBytes = 16;
  static constexpr int kNumBarrierSlots = 128;
  static constexpr int kBarrierStorageBytes = kNumBarrierSlots * 8;
  static constexpr bool kPagedOnly = true;
  static constexpr bool kSingleTokenQ = true;
  static constexpr bool kSupportsSplitKv = false;
  static constexpr bool kEnableSplitKvPath = false;
  static constexpr bool kEnableStaticPath = true;

  static_assert(kTransformKvWarpStart + kNumTransformKvWarps == kNumWarps,
                "FMHA forward decode uses all 16 warps in the q8kv4 task layout.");
  static_assert(kSmemQBytesPerStage == kHeadGroup * 128,
                "q8kv4 FMHA forward Q stage should match the expected SmemQ tile.");
  static_assert(kRawKvDataBytesPerStage == 16384,
                "q8kv4 FMHA forward raw KV stage stores one unpacked byte per fp4.");
  static_assert(kRawKvScaleBytesPerStage == 1024,
                "q8kv4 FMHA forward scale stage stores one E4M3 byte per 16 values.");
  static_assert(kRawKvScaleScratchBytesPerStage == (kF16VScaleScratch ? 2048 : 1024),
                "q8kv4 FMHA forward scale scratch stores one reordered scale tile (E4M3 or F16).");
  static_assert(kRawKvStageBytes == 16384 + 1024 + kRawKvScaleScratchBytesPerStage,
                "raw KV stage size must match the transformed-KV path.");
  static_assert(kRawKvTmaBytes == 9216,
                "raw KV TMA transaction bytes must match the SmemKv pipeline.");
  static_assert(kSplitPlanTileKv % kTileKv == 0,
                "split plan KV tile must map to FMHA forward decode KV tiles.");
  static_assert(kSplitPlanTileQ % kTileQ == 0,
                "split plan Q tile must map to FMHA forward decode Q tiles.");
};

template <int TopK, bool EnableSplitKv, int HeadGroup = 16>
struct Sm100FmhaQ8Kv4SparseTraits : Sm100FmhaQ8Kv4BaseTraits<HeadGroup> {
  static constexpr int kSparseTopK = TopK;
  static constexpr int kSparseKvTokens = TopK * Sm100FmhaQ8Kv4BaseTraits<HeadGroup>::kPageSize;
  static constexpr bool kEnableSplitKvPath = EnableSplitKv;
  static constexpr bool kEnableStaticPath = true;
  static_assert(TopK == 16, "first sparse FMHA forward decode variant supports topK=16.");
};

template <bool IsSplitKV, SparseAttnMode kSparseAttnMode, bool IsQ8KV4, int SparseTopK = 16,
          int FixedQTokensPerBatch = 0, int HeadGroup = 16>
struct Sm100FmhaQ8Kv4TraitSelector {
  static_assert(IsQ8KV4, "this SM100 forward decode path is specialized for Q8KV4.");
  static_assert(kSparseAttnMode == SparseAttnMode::Sparse,
                "Q8KV4 decode only supports sparse attention.");
  static_assert(FixedQTokensPerBatch == 0, "Q8KV4 decode query length remains a runtime scalar.");
  using type = Sm100FmhaQ8Kv4SparseTraits<SparseTopK, IsSplitKV, HeadGroup>;
};

} // namespace cutlass::fmha::collective
