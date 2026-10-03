#pragma once

#include "cutlass/cutlass.h"

namespace minimax::msa_v1::indexer::decode::q8kv4 {

// Include the tile in the type so JIT modules cannot share launch-initialization state.
template <int NumIndexHeads, int QueryColumns> struct IndexerGemmTraitsForHeads {
  static constexpr int kNumIndexHeads = NumIndexHeads;
  // Narrow tiles use smaller CTAs and split each planned span to overlap memory latency.
  static constexpr int kPageWorkerGroups = QueryColumns == 16 ? 3 : 4;
  static constexpr int kCtasPerWorker = QueryColumns == 16 ? 2 : 1;
  static constexpr int kPageWorkerWarps = 4;
  static constexpr int kAccumulatorStages = kPageWorkerGroups;
  static constexpr int kDequantStages = kPageWorkerGroups;
  static constexpr int kDequantGroups = kPageWorkerGroups;
  static constexpr int kDequantWarpsPerGroup = kPageWorkerWarps;
  static constexpr int kConsumerGroups = kPageWorkerGroups;
  static constexpr int kConsumerWarpsPerGroup = kPageWorkerWarps;
  static constexpr int kQueryColumns = QueryColumns;
  static_assert(kQueryColumns >= 16 && kQueryColumns <= 64 && kQueryColumns % 16 == 0);
  static constexpr int kMaxQueryLength = 16;
  static constexpr int kHeadDim = 128;
  static constexpr int kPageTokens = 128;
  static constexpr int kScaleGroupSize = 16;
  static constexpr int kScaleGroups = kHeadDim / kScaleGroupSize;
  static constexpr int kPackedKBytes = kPageTokens * kHeadDim / 2;
  static constexpr int kScaleBytes = kPageTokens * kScaleGroups;
  static constexpr int kPageBytes = kPackedKBytes + kScaleBytes;
  static constexpr int kThreads =
      (kPageWorkerGroups * kPageWorkerWarps + 2) * cutlass::NumThreadsPerWarp;
  static constexpr int kMaxPagesPerCta = 2 * kPageWorkerGroups;
  static constexpr int kMaximumPages = 8192;
};

using IndexerGemmTraits =
    IndexerGemmTraitsForHeads<MINIMAX_MSA_INDEX_HEADS, MINIMAX_MSA_QUERY_COLUMNS>;

} // namespace minimax::msa_v1::indexer::decode::q8kv4
