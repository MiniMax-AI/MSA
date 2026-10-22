#pragma once

#include "attn/kernel/include/m3_decode_output_projection_handoff.hpp"

namespace cutlass::m3_decode::collective {

struct Sm100M3DecodeRouterTraits {
  static constexpr int kHiddenSize = mega_atten::m3_decode::output_projection::kProblemM;
  static constexpr int kExperts = 128;
  static constexpr int kTopK = 4;
  static constexpr int kClusterCtas = 8;
  static constexpr int kThreads = 512;
  static constexpr int kResidualHandoffOffset = 8192;
  static constexpr int kHiddenPerCta = kHiddenSize / kClusterCtas;
  static constexpr int kScalePartials = 2;
  static constexpr int kEntryReadyBytes = sizeof(uint32_t);
  static constexpr int kDotWarps = kThreads / 32;
  static constexpr int kExpertWaves = kExperts / kDotWarps;
  static constexpr int kExpertsPerOwner = kExperts / kClusterCtas;
  static constexpr int kPeerCount = kClusterCtas - 1;
  static constexpr int kPartialBytesPerPeer = kExpertsPerOwner * sizeof(float);
  static constexpr int kPartialRemoteBytes = kPeerCount * kPartialBytesPerPeer;
  static constexpr int kCandidateBytesPerOwner = kTopK * 16;
  static constexpr int kCandidateRemoteBytes = kPeerCount * kCandidateBytesPerOwner;
  static constexpr int kResidualRemoteBytes = kHiddenPerCta * sizeof(uint16_t);
  static constexpr int kScaleRemoteBytes = kScalePartials * sizeof(float);

  static_assert(kHiddenSize == 6144, "Router hidden size must be 6144.");
  static_assert(kHiddenPerCta == 768, "Router cluster rank must own 768 hidden elements.");
  static_assert(kDotWarps == 16 && kExpertWaves == 8,
                "Router dot requires 16 warps and eight expert waves.");
  static_assert(kExpertsPerOwner == 16, "each Router owner must own 16 experts.");
  static_assert(kPartialRemoteBytes == 448, "Router partial transaction must be 448 bytes.");
  static_assert(kCandidateRemoteBytes == 448, "Router candidate transaction must be 448 bytes.");
  static_assert(kResidualRemoteBytes == 1536, "Router residual transaction must be 1536 bytes.");
  static_assert(kScaleRemoteBytes == 8, "Router scale transaction must be eight bytes.");
  static_assert(kEntryReadyBytes == 4, "Router entry transaction must be four bytes.");
};

} // namespace cutlass::m3_decode::collective
