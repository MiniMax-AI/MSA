#pragma once

/***************************************************************************************************
 * Copyright (c) 2026, MegaGEMM contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SM100 CUTLASS/CuTe C++ port of the MiniMax-M3 TP4 QKV/QIKI 8x8 decode kernel.
 *
 * The launch geometry and the M128N8 block-scaled instruction below are deliberately fixed.  They
 * form workspace ABI `8c8cta_qsplit3_kvsplit4`: eight 8-CTA clusters, 256 threads per CTA, three
 * K splits for Q, and four K splits for K/V/QI/KI. The device implementation is embedded as the
 * Attention Projection prologue of the M3 decode MegaKernel.
 **************************************************************************************************/

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/detail/sm100_tmem_helper.hpp>
#include <cutlass/float8.h>

#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm100_tma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/tensor.hpp>

#include "attn/kernel/include/sm100_m3_decode_attention_projection_numeric.hpp"

namespace cutlass::m3_decode::collective::attention_projection {

using namespace cute;
using sm100::F32x4;

constexpr int kTokensPerTile = 8;
constexpr int kHidden = 6144;
constexpr int kHeadDim = 128;
constexpr int kNumQHeads = 16;
constexpr int kNumHeadSlots = 20;
constexpr int kOutputOwnerCtas = kNumHeadSlots;
constexpr int kOutput = kNumHeadSlots * kHeadDim;
constexpr int kRotaryDim = 64;
constexpr int kCachePageSize = 128;
constexpr int kThreads = 256;
constexpr int kClusterSize = 8;
constexpr int kClusterCount = 8;
constexpr int kInputKPerCta = kHidden / kClusterSize;
constexpr int kCtaK = 128;
constexpr int kSfVec = 32;
constexpr int kAbStages = 6;
constexpr int kPrefetchDistance = 3;
constexpr int kQSplitK = 3;
constexpr int kSpecialSplitK = 4;
constexpr int kNormGroupStride = 36;
constexpr int kNormSmemSize = (kInputKPerCta / kSfVec) * kNormGroupStride;
constexpr int kRawSmemSize = 880;
constexpr int kAttentionProjectionBarrierId = 3;

using ElementAB = cutlass::float_e4m3_t;
using ElementSF = cutlass::float_ue8m0_t;
using DirectM128N8Mxfp8Atom =
    cute::SM100_MMA_MXF8F6F4_SS<ElementAB, ElementAB, float, ElementSF, 128, 8,
                                cute::UMMA::Major::K, cute::UMMA::Major::K>;
using DirectM128N8Mxfp8 = decltype(cute::make_tiled_mma(DirectM128N8Mxfp8Atom{}));
static_assert(cute::size<0>(typename DirectM128N8Mxfp8::Shape_MNK{}) == kHeadDim);
static_assert(cute::size<1>(typename DirectM128N8Mxfp8::Shape_MNK{}) == kTokensPerTile);
static_assert(cute::size<2>(typename DirectM128N8Mxfp8::Shape_MNK{}) == kSfVec);

using MmaTileShape = cute::Shape<cute::_128, cute::_8, cute::_128>;
using MmaShapeA =
    decltype(cute::partition_shape_A(DirectM128N8Mxfp8{}, cute::select<0, 2>(MmaTileShape{})));
using MmaShapeB =
    decltype(cute::partition_shape_B(DirectM128N8Mxfp8{}, cute::select<1, 2>(MmaTileShape{})));
using SmemLayoutA = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_SW128_Atom<ElementAB>{}, MmaShapeA{}));
using SmemLayoutB = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_SW128_Atom<ElementAB>{}, MmaShapeB{}));
using SmemLayoutSFA = decltype(sm100::make_m128n8_sfa_smem_stage_layout<DirectM128N8Mxfp8>());
using SmemLayoutSFB = decltype(sm100::make_small_n_sfb_smem_stage_layout<>());
static_assert(cute::cosize_v<SmemLayoutA> == 128 * 128);
static_assert(cute::cosize_v<SmemLayoutB> == 8 * 128);
static_assert(cute::cosize_v<SmemLayoutSFA> == 512);
static_assert(cute::cosize_v<SmemLayoutSFB> == 512);

using WeightLayout =
    cute::Layout<cute::Shape<cute::Int<kOutput>, cute::Int<kHidden>, cute::_1>,
                 cute::Stride<cute::Int<kHidden>, cute::_1, cute::Int<kOutput * kHidden>>>;
using ActivationLayout =
    cute::Layout<cute::Shape<cute::_8, cute::Int<kHidden>, cute::_1>,
                 cute::Stride<cute::Int<kHidden>, cute::_1, cute::Int<kTokensPerTile * kHidden>>>;
using WeightScaleLayout =
    decltype(cutlass::detail::Sm1xxBlockScaledConfig<kSfVec>::tile_atom_to_shape_SFA(
        cute::make_shape(cute::Int<kOutput>{}, cute::_8{}, cute::Int<kHidden>{}, cute::_1{})));
using ClusterLayoutVMNK = decltype(cute::tiled_divide(
    cute::make_layout(cute::make_shape(cute::_1{}, cute::_8{}, cute::_1{})),
    cute::make_tile(typename DirectM128N8Mxfp8::AtomThrID{})));
using WeightTensor = decltype(cute::make_tensor(
    cute::make_gmem_ptr(static_cast<ElementAB const *>(nullptr)), WeightLayout{}));
using ActivationTensor = decltype(cute::make_tensor(
    cute::make_gmem_ptr(static_cast<ElementAB const *>(nullptr)), ActivationLayout{}));
using WeightScaleTensor = decltype(cute::make_tensor(
    cute::make_gmem_ptr(static_cast<ElementSF const *>(nullptr)), WeightScaleLayout{}));
using WeightTma = decltype(cute::make_tma_atom_A_sm100<ElementAB>(
    cute::SM90_TMA_LOAD{}, WeightTensor{}, SmemLayoutA{}, MmaTileShape{}, DirectM128N8Mxfp8{},
    ClusterLayoutVMNK{}));
using WeightScaleTma = decltype(cute::make_tma_atom_A_sm100<uint16_t>(
    cute::SM90_TMA_LOAD{}, WeightScaleTensor{}, SmemLayoutSFA{}, MmaTileShape{},
    DirectM128N8Mxfp8{}, ClusterLayoutVMNK{}));
using ActivationTma = decltype(cute::make_tma_atom_B_sm100<ElementAB>(
    cute::SM90_TMA_LOAD{}, ActivationTensor{}, SmemLayoutB{}, MmaTileShape{}, DirectM128N8Mxfp8{},
    ClusterLayoutVMNK{}));

struct alignas(128) TmaParams {
  WeightTma weight;
  WeightScaleTma weight_scale;
  ActivationTma activation;
};
static_assert(sizeof(TmaParams) == 768);
static_assert(std::is_trivially_copyable_v<TmaParams>);
static_assert(std::is_standard_layout_v<TmaParams>);

TmaParams make_tma_params(ElementAB const *weight, ElementSF const *weight_scale,
                          ElementAB const *activation) {
  auto m_weight = cute::make_tensor(cute::make_gmem_ptr(weight), WeightLayout{});
  auto m_weight_scale = cute::make_tensor(cute::make_gmem_ptr(weight_scale), WeightScaleLayout{});
  auto m_activation = cute::make_tensor(cute::make_gmem_ptr(activation), ActivationLayout{});
  return {cute::make_tma_atom_A_sm100<ElementAB>(cute::SM90_TMA_LOAD{}, m_weight, SmemLayoutA{},
                                                 MmaTileShape{}, DirectM128N8Mxfp8{},
                                                 ClusterLayoutVMNK{}),
          cute::make_tma_atom_A_sm100<uint16_t>(cute::SM90_TMA_LOAD{}, m_weight_scale,
                                                SmemLayoutSFA{}, MmaTileShape{},
                                                DirectM128N8Mxfp8{}, ClusterLayoutVMNK{}),
          cute::make_tma_atom_B_sm100<ElementAB>(cute::SM90_TMA_LOAD{}, m_activation, SmemLayoutB{},
                                                 MmaTileShape{}, DirectM128N8Mxfp8{},
                                                 ClusterLayoutVMNK{})};
}

struct KernelParams {
  TmaParams tma;
  __nv_bfloat16 const *layer_input;
  __nv_bfloat16 const *input_norm_weight;

  uint8_t *workspace_data;
  uint8_t *workspace_scale;
  uint32_t *workspace_grid_epoch;
  float *workspace_special_partial;
  uint32_t *workspace_special_epochs;

  int64_t const *positions;
  int64_t const *slot_mapping;
  __nv_bfloat16 const *q_norm_weight;
  __nv_bfloat16 const *k_norm_weight;
  __nv_bfloat16 const *qi_norm_weight;
  __nv_bfloat16 const *ki_norm_weight;
  __nv_bfloat16 const *cos_sin_cache;

  uint8_t *q_fp8_out;
  uint8_t *qi_fp8_out;
  uint8_t *k_data_cache;
  uint8_t *k_scale_cache;
  uint8_t *v_data_cache;
  uint8_t *v_scale_cache;
  uint8_t *ki_data_cache;
  uint8_t *ki_scale_cache;

  float input_norm_epsilon;
  float post_norm_epsilon;
  float k_dequant_global_scale;
  float v_dequant_global_scale;
  float ki_dequant_global_scale;

  // Byte distance between physical pages for each cache component. Rows inside a page are dense:
  // data rows are kHeadDim / 2 bytes and scale rows are kHeadDim / 16 bytes.
  int64_t k_data_page_stride;
  int64_t k_scale_page_stride;
  int64_t v_data_page_stride;
  int64_t v_scale_page_stride;
  int64_t ki_data_page_stride;
  int64_t ki_scale_page_stride;
};

struct KernelPolicy {
  CUTE_DEVICE static constexpr bool valid_token(int, KernelParams const &) { return true; }
};

struct alignas(8) BarrierStorage {
  uint64_t ab_full[kAbStages];
  uint64_t ab_empty[kAbStages];
  uint64_t mma_epilog;
  uint64_t tmem_lifetime;
  uint64_t input_mailbox;
  uint64_t q_mailbox[kTokensPerTile];
};

static_assert(alignof(BarrierStorage) == alignof(uint64_t));
static_assert(sizeof(BarrierStorage) == (2 * kAbStages + 3 + kTokensPerTile) * sizeof(uint64_t));
static_assert(sizeof(BarrierStorage) == 184);

// The complete implementation grows this storage in-place.  Keep all offsets compile-time and the
// base 128-byte aligned so the eventual six-stage TMA ring and DSM mailboxes
// retain their contracts.
struct alignas(128) SharedStorage {
  alignas(128) uint8_t ab_ring[kAbStages][128 * 128 + 8 * 128 + 512 + 512];
  // Per-CTA storage; Q split CTAs map their slot into the rank-0/rank-3 owner's DSM.
  alignas(128) float q_mailbox[kQSplitK * kTokensPerTile * kHeadDim];
  alignas(16) float input_warp_sums[8];
  alignas(16) float input_partial[4];
  alignas(16) float input_rms_inv[4];
  alignas(16) float input_norm_multiplier[kNormSmemSize];
  alignas(16) float input_raw[kRawSmemSize];
  // Keep the remaining projection workspace at its established addresses
  // while the mbarriers move to the phase-specific slab.
  alignas(16) uint8_t barrier_layout_padding[208];
  alignas(16) float input_mailbox[7 * 4];
  alignas(16) uint32_t tmem_base;
};

static_assert(sizeof(SharedStorage) < 128 * 1024, "8x8 candidate must fit the SM100 opt-in limit");

} // namespace cutlass::m3_decode::collective::attention_projection
