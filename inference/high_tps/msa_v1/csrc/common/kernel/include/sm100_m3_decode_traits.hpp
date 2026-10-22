#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"

namespace cutlass::m3_decode::collective {

enum class Sm100M3DecodeKernelPath {
  kMonolithic,
  kMegaAtten,
  kMegaMoe,
};

struct Sm100M3DecodeBaseTraits {
  static constexpr int kTileQ = 16;
  static constexpr int kTileKv = 128;
  static constexpr int kHeadDim = 128;
  static constexpr int kPageSize = 128;
  static constexpr int kHeadGroup = 16;
  static constexpr int kPackFactor = 16;

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
  static constexpr int kLoadWarpStart = 11;
  static constexpr int kTransformKvWarpStart = 12;

  static constexpr int kLoadMaxRegisters = 56;
  static constexpr int kPageOffsetsMaxRegisters = 56;
  static constexpr int kMmaMaxRegisters = 56;
  static constexpr int kPaddingMaxRegisters = 56;
  static constexpr int kSoftmaxMaxRegisters = 184;
  static constexpr int kCorrectionMaxRegisters = 88;
  static constexpr int kTransformKvMaxRegisters = 184;

  static constexpr int kNumStagesQ = 2;
  static constexpr int kNumStagesRawKv = 8;
  static constexpr int kNumStagesTransform = 12;
  static constexpr int kNumTmemCols = 512;

  static constexpr int kSmemQBytesPerStage = kTileQ * kHeadDim;
  static constexpr int kRawKvDataBytesPerStage = kTileKv * kHeadDim;
  static constexpr int kRawKvScaleBytesPerStage = kTileKv * (kHeadDim / kPackFactor);
  static constexpr int kRawKvScaleScratchBytesPerStage = kRawKvScaleBytesPerStage;
  static constexpr int kRawKvStageBytes =
      kRawKvDataBytesPerStage + kRawKvScaleBytesPerStage + kRawKvScaleScratchBytesPerStage;
  static constexpr int kRawKvTmaBytes = kTileKv * (kHeadDim / 2) + kRawKvScaleBytesPerStage;

  static constexpr int kNumPageOffsetStages = 6;
  static constexpr int kPageOffsetsPerStage = 32;
  static constexpr int kSmemPBytes = 4096;
  static constexpr int kSmemOBytes = 4096;
  static constexpr int kWarpGroupReductionFloats = 64;
  static constexpr int kTmemSwStateBytes = 16;
  static constexpr int kNumBarrierSlots = 96;
  static constexpr int kBarrierStorageBytes = kNumBarrierSlots * 8;
  static constexpr int kNumKvSplits = 8;
  static constexpr int kClusterReductionMaxRowsPerCta = 2;
  static constexpr int kClusterReductionMailboxRows = kHeadGroup;
  static constexpr int kClusterReductionStatsPerRow = 2;
  static constexpr int kClusterReductionOBytes =
      kClusterReductionMailboxRows * kHeadDim * static_cast<int>(sizeof(cutlass::bfloat16_t));
  static constexpr int kClusterReductionStatsFloats =
      kClusterReductionMailboxRows * kClusterReductionStatsPerRow;
  static constexpr int kClusterReductionRowPayloadBytes =
      kHeadDim * static_cast<int>(sizeof(cutlass::bfloat16_t)) +
      kClusterReductionStatsPerRow * static_cast<int>(sizeof(float));
  static constexpr int kClusterReductionPayloadBytes =
      kClusterReductionOBytes + kClusterReductionStatsFloats * static_cast<int>(sizeof(float));
  static constexpr int kClusterReductionStorageRows = kTileKv;
  static constexpr int kClusterReductionTransactionBytes =
      kClusterReductionStorageRows * kClusterReductionRowPayloadBytes;
  static constexpr int kClusterReductionSmemBytes = kClusterReductionTransactionBytes;

  static constexpr int kMaxQTokensPerBatch = 8;
  static constexpr int kSparseTopK = 16;

  static_assert(kTransformKvWarpStart + kNumTransformKvWarps == kNumWarps,
                "M3 Decode attention uses all 16 warps in the Q8KV4 task layout.");
  static_assert(kHeadGroup % kNumKvSplits == 0,
                "the fixed split count must divide the Q-head group.");
  static_assert(kSmemQBytesPerStage == 2048,
                "Q8KV4 attention Q stage should match the expected SmemQ tile.");
  static_assert(kRawKvDataBytesPerStage == 16384,
                "Q8KV4 attention raw KV stage stores one unpacked byte per FP4.");
  static_assert(kRawKvScaleBytesPerStage == 1024,
                "Q8KV4 attention scale stage stores one E4M3 byte per 16 values.");
  static_assert(kRawKvScaleScratchBytesPerStage == 1024,
                "Q8KV4 attention scale scratch stores one reordered scale tile.");
  static_assert(kRawKvStageBytes == 18432, "raw KV stage size must match the transformed-KV path.");
  static_assert(kRawKvTmaBytes == 9216,
                "raw KV TMA transaction bytes must match the SmemKv pipeline.");
  static_assert(kClusterReductionTransactionBytes >= kClusterReductionPayloadBytes,
                "cluster reduction storage must cover all written rows.");
};

template <bool RouterGammaPrepacked = true, int IndexerTopKItemsPerThread = 0,
          bool WaveBalancedIndexerPages = false,
          Sm100M3DecodeKernelPath KernelPath = Sm100M3DecodeKernelPath::kMonolithic,
          int MoeQueryTokens = 8, bool MaskPaddedMoeRoutes = false>
struct Sm100M3DecodeTraits : Sm100M3DecodeBaseTraits {
  static_assert(MoeQueryTokens >= 1 && MoeQueryTokens <= 8);
  static_assert(KernelPath == Sm100M3DecodeKernelPath::kMegaMoe || MoeQueryTokens == 8);
  static_assert(!MaskPaddedMoeRoutes || KernelPath == Sm100M3DecodeKernelPath::kMonolithic);
  static constexpr int kMoeQueryTokens = MoeQueryTokens;
  static constexpr bool kMaskPaddedMoeRoutes = MaskPaddedMoeRoutes;
  static constexpr bool kRouterGammaPrepacked = RouterGammaPrepacked;
  static constexpr int kIndexerTopKItemsPerThread = IndexerTopKItemsPerThread;
  static constexpr bool kWaveBalancedIndexerPages = WaveBalancedIndexerPages;
  static constexpr Sm100M3DecodeKernelPath kKernelPath = KernelPath;
};

} // namespace cutlass::m3_decode::collective
