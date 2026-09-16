#pragma once

#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"

namespace minimax::msa_v1::attention::prefill::q8kv4::detail {

using Element = cutlass::float_e4m3_t;
using Accumulator = float;

inline constexpr int kQHeadsPerKv = 16;
inline constexpr int kHeadDim = 128;
inline constexpr int kPageSize = 128;
inline constexpr int kQueriesPerGroup = 128 / kQHeadsPerKv;
inline constexpr int kQStages = 3;
inline constexpr int kScoreStages = 2;
inline constexpr int kQMetadataStages = 16;

inline constexpr int kTmemScoreOffset = 0;
inline constexpr int kTmemScoreStride = 128;
inline constexpr int kTmemProbabilityOffset = 96;
inline constexpr int kTmemOutputOffset = 256;
inline constexpr int kTmemOutputStride = 128;
inline constexpr int kTmemColumns = 512;

inline constexpr int kPackedKvBytes = kPageSize * kHeadDim / 2;
inline constexpr int kScaleBytes = kPageSize * kHeadDim / 16;

// Warp roles follow the FA4 convention: define ownership once and dispatch
// each role from a small top-level kernel body.
struct WarpSpecializationSm103 {
  static constexpr int kSoftmax0Warp = 0;
  static constexpr int kSoftmax1Warp = 4;
  static constexpr int kEpilogueWarp = 8;
  static constexpr int kMmaWarp = 12;
  static constexpr int kLoadWarp = 13;

  static constexpr int kSoftmaxWarps = 4;
  static constexpr int kEpilogueWarps = 4;
  static constexpr int kLoadWarps = 3;
  static constexpr int kWarps = 16;
  static constexpr int kThreads = kWarps * 32;
  static constexpr int kTmemParticipantWarps = kMmaWarp + 1;

  static constexpr int kLoadKWarp = kLoadWarp;
  static constexpr int kLoadVWarp = kLoadWarp + 1;
  static constexpr int kQLoadWarps = kLoadWarps;
  static constexpr int kQLoadThreads = kQLoadWarps * 32;

  CUTLASS_HOST_DEVICE static constexpr bool is_softmax0_warp(int warp_idx) {
    return warp_idx >= kSoftmax0Warp && warp_idx < kSoftmax0Warp + kSoftmaxWarps;
  }

  CUTLASS_HOST_DEVICE static constexpr bool is_softmax1_warp(int warp_idx) {
    return warp_idx >= kSoftmax1Warp && warp_idx < kSoftmax1Warp + kSoftmaxWarps;
  }

  CUTLASS_HOST_DEVICE static constexpr bool is_softmax_warp(int warp_idx) {
    return is_softmax0_warp(warp_idx) || is_softmax1_warp(warp_idx);
  }

  CUTLASS_HOST_DEVICE static constexpr bool is_epilogue_warp(int warp_idx) {
    return warp_idx >= kEpilogueWarp && warp_idx < kMmaWarp;
  }

  CUTLASS_HOST_DEVICE static constexpr bool is_mma_warp(int warp_idx) {
    return warp_idx == kMmaWarp;
  }

  CUTLASS_HOST_DEVICE static constexpr bool is_load_warp(int warp_idx) {
    return warp_idx >= kLoadWarp && warp_idx < kLoadWarp + kLoadWarps;
  }

  CUTLASS_HOST_DEVICE static constexpr int softmax_stage(int warp_idx) {
    return is_softmax0_warp(warp_idx) ? 0 : 1;
  }

  CUTLASS_HOST_DEVICE static constexpr int softmax_warp_base(int stage) {
    return stage == 0 ? kSoftmax0Warp : kSoftmax1Warp;
  }
};

using WarpSpecialization = WarpSpecializationSm103;

static_assert(WarpSpecialization::kSoftmax0Warp == 0);
static_assert(WarpSpecialization::kSoftmax1Warp ==
              WarpSpecialization::kSoftmax0Warp + WarpSpecialization::kSoftmaxWarps);
static_assert(WarpSpecialization::kEpilogueWarp ==
              WarpSpecialization::kSoftmax1Warp + WarpSpecialization::kSoftmaxWarps);
static_assert(WarpSpecialization::kMmaWarp ==
              WarpSpecialization::kEpilogueWarp + WarpSpecialization::kEpilogueWarps);
static_assert(WarpSpecialization::kLoadWarp == WarpSpecialization::kMmaWarp + 1);
static_assert(WarpSpecialization::kLoadWarp + WarpSpecialization::kLoadWarps ==
              WarpSpecialization::kWarps);

inline constexpr int kSoftmax0Registers = 176;
inline constexpr int kSoftmax1Registers = 176;
inline constexpr int kEpilogueRegisters = 96;
inline constexpr int kOtherRegisters = 64;
static_assert(kSoftmax0Registers + kSoftmax1Registers + kEpilogueRegisters + kOtherRegisters <=
              512);

inline constexpr int kEpilogueBarrierId = 4;
inline constexpr int kKvDequantKBarrierId = 6;
inline constexpr int kKvDequantVBarrierId = 7;
inline constexpr int kTmemBarrierId = 0;
inline constexpr int kQLoadBarrierId = 1;

using QTokenSmemLayout = decltype(cute::coalesce(
    cute::tile_to_shape(cute::UMMA::Layout_K_SW128_Atom<Element>{},
                        cute::Shape<cute::Int<kQHeadsPerKv>, cute::Int<kHeadDim>>{},
                        cute::Step<cute::_1, cute::_2>{}),
    cute::Shape<cute::_1, cute::_1>{}));

} // namespace minimax::msa_v1::attention::prefill::q8kv4::detail
