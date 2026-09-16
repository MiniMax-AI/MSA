#pragma once

#include "cutlass/cutlass.h"

namespace minimax::msa_v1::indexer::decode::tp4_q8kv4 {

struct IndexerGemmTraits {
  static constexpr int kQueryLength = 8;
  static constexpr int kHeadDim = 128;
  static constexpr int kPageTokens = 128;
  static constexpr int kScaleGroupSize = 16;
  static constexpr int kScaleGroups = kHeadDim / kScaleGroupSize;
  static constexpr int kPackedKBytes = kPageTokens * kHeadDim / 2;
  static constexpr int kScaleBytes = kPageTokens * kScaleGroups;
  static constexpr int kPageBytes = kPackedKBytes + kScaleBytes;
  static constexpr int kScoreWarps = 8;
  static constexpr int kTransportWarps = 1;
  static constexpr int kThreads =
      (kScoreWarps + kTransportWarps) * cutlass::NumThreadsPerWarp;
  static constexpr int kMaxPagesPerCta = 4;
  static constexpr int kPagesPerWorkTile = 16;
  static constexpr int kPartialPageCapacity = kPagesPerWorkTile;
  static constexpr int kPartialQueryCapacity = kQueryLength + 1;
  static constexpr int kPartialWarpCapacity = kScoreWarps;
  static constexpr int kSchedulerStages = 2;
  // Keep the inline scan small enough that scheduler setup remains negligible
  // for the latency-sensitive decode range. Larger batches use CUB scan.
  static constexpr int kPrepareThreads = 128;
  static constexpr int kPrepareWarps = kPrepareThreads / 32;
  static constexpr int kMaximumPages = 8192;
};

}  // namespace minimax::msa_v1::indexer::decode::tp4_q8kv4
