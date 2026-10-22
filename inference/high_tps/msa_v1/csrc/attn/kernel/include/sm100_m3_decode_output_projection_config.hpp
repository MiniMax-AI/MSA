#pragma once

#include <cstdint>

#include "attn/kernel/include/m3_decode_output_projection_handoff.hpp"
#include "attn/kernel/include/sm100_m3_decode_problem.hpp"
#include "common/kernel/include/sm100_m3_decode_pipeline.hpp"
#include "common/kernel/include/sm100_m3_decode_storage.hpp"
#include "common/kernel/include/sm100_m3_decode_tp4_allreduce.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/arch/copy_sm100.hpp"
#include "cute/arch/copy_sm90_desc.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/atom/copy_traits_sm100.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm100.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

namespace cutlass::m3_decode::collective {

template <class Traits> struct Sm100M3DecodeOutputProjectionConfig {
  using Storage = SharedStorage<Traits>;
  using Params = Sm100M3DecodeKernelParams<Traits>;
  using M3DecodeBarriers = BarrierLayout<Traits>;

  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementAccumulator = float;
  using ElementScale = cutlass::float_ue8m0_t;
  using ElementOutput = cutlass::bfloat16_t;

  static constexpr int kTileM = 128;
  static constexpr int kTileN = 8;
  static constexpr int kTileK = 128;
  static constexpr int kOutputProjectionSplit = 8;
  static constexpr int kProblemM = mega_atten::m3_decode::output_projection::kProblemM;
  static constexpr int kProblemN = mega_atten::m3_decode::output_projection::kProblemN;
  static constexpr int kProblemK = mega_atten::m3_decode::output_projection::kProblemK;
  static constexpr int kOutputMTiles = kProblemM / kTileM;
  static constexpr int kOutputMTilesPerToken = kOutputMTiles / kProblemN;
  static constexpr int kAttentionCtas = kProblemN * kOutputProjectionSplit;
  static constexpr int kDedicatedOutputProjectionCtas = kOutputMTiles;
  static constexpr int kGridCtas = kAttentionCtas + kDedicatedOutputProjectionCtas;
  static constexpr int kKStages = kProblemK / kTileK;
  static constexpr int kStageProducerCount = kProblemN;
  static constexpr int kRowsPerProducer = kKStages / kOutputProjectionSplit;
  // Publish attention activations in four groups of four K stages.
  static constexpr int kHandoffStagesPerGroup =
      mega_atten::m3_decode::output_projection::kHandoffStagesPerGroup;
  static constexpr int kHandoffGroupCount =
      mega_atten::m3_decode::output_projection::kHandoffGroupCount;
  static constexpr int kProducerSplitsPerGroup = kHandoffStagesPerGroup / kRowsPerProducer;
  static constexpr int kHandoffGroupProducerCount = kStageProducerCount * kProducerSplitsPerGroup;
  static constexpr int kWeightLoadWarp = 0;
  static constexpr int kActivationLoadWarp = 1;
  static constexpr int kScaleLoadWarp = 2;
  static constexpr int kHandoffPollWarp = 3;
  static constexpr int kEpilogueWarpStart = 4;
  static constexpr int kEpilogueWarps = 4;
  static constexpr int kTcWarpStart = 8;
  static constexpr int kTcWarps = 4;

  using MmaOp =
      cute::SM100_MMA_MXF8F6F4_SS<ElementA, ElementB, ElementAccumulator, ElementScale, kTileM,
                                  kTileN, cute::UMMA::Major::K, cute::UMMA::Major::K>;
  using MmaTraits = cute::MMA_Traits<MmaOp>;
  using TiledMma = decltype(cute::make_tiled_mma(MmaOp{}));
  static constexpr int kInstructionK = MmaTraits::K;
  static constexpr int kScaleVectorSize = MmaTraits::SFVecSize;
  static constexpr int kMmaPerKTile = kTileK / kInstructionK;
  static constexpr int kScaleColsPerKTile = kTileK / kScaleVectorSize;
  static constexpr int kScaleRowsPadded = 128;
  using TileShape = cute::Shape<cute::Int<kTileM>, cute::Int<kTileN>, cute::Int<kTileK>>;
  using MmaShapeA_MK = decltype(cute::partition_shape_A(
      TiledMma{}, cute::make_shape(cute::Int<kTileM>{}, cute::Int<kTileK>{})));
  using MmaShapeB_NK = decltype(cute::partition_shape_B(
      TiledMma{}, cute::make_shape(cute::Int<kTileN>{}, cute::Int<kTileK>{})));
  using AccumulatorShape = decltype(cute::partition_shape_C(
      TiledMma{}, cute::make_shape(cute::Int<kTileM>{}, cute::Int<kTileN>{})));
  using SmemLayoutAtomA = cute::UMMA::Layout_K_SW128_Atom<ElementA>;
  using SmemLayoutAtomB = cute::UMMA::Layout_K_SW128_Atom<ElementB>;
  using SmemLayoutA = decltype(cute::UMMA::tile_to_mma_shape(
      SmemLayoutAtomA{}, cute::append(MmaShapeA_MK{}, cute::Int<1>{}),
      cute::Step<cute::_1, cute::_2, cute::_3>{}));
  using SmemLayoutB = decltype(cute::UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{}, cute::append(MmaShapeB_NK{}, cute::Int<1>{}),
      cute::Step<cute::_1, cute::_2, cute::_3>{}));
  using ScaleLayoutConfig = cutlass::detail::Sm1xxBlockScaledConfig<kScaleVectorSize>;
  using SmemLayoutSFA = decltype(ScaleLayoutConfig::deduce_smem_layoutSFA(TiledMma{}, TileShape{}));
  using SmemLayoutSFB = decltype(ScaleLayoutConfig::deduce_smem_layoutSFB(TiledMma{}, TileShape{}));
  using UtccpOp = cute::SM100_UTCCP_4x32dp128bit_1cta;

  static constexpr int kSmemAlignment = 128;
  static constexpr int kActivationValueSmemOffset = 0;
  static constexpr int kActivationValueBytesPerStage = kTileN * kTileK;
  static constexpr int kActivationScaleSmemOffset =
      kActivationValueSmemOffset + kActivationValueBytesPerStage;
  static constexpr int kActivationScaleBytesPerStage = kScaleRowsPadded * kScaleColsPerKTile;
  static constexpr int kActivationStageBytes =
      ((kActivationScaleSmemOffset + kActivationScaleBytesPerStage + kSmemAlignment - 1) /
       kSmemAlignment) *
      kSmemAlignment;
  static constexpr int kActivationStages = 8;
  static constexpr int kActivationSmemBaseOffset = kActivationScaleBytesPerStage;
  static constexpr int kActivationStageStride = kActivationStageBytes;
  static constexpr int kPipelineStages = 8;
  static constexpr int kRawKvEventsPerCta = 2 * (Traits::kSparseTopK / kOutputProjectionSplit);
  static constexpr int kInitialWeightStage = kRawKvEventsPerCta % kPipelineStages;

  static constexpr int kWeightValueSmemOffset = 0;
  static constexpr int kWeightValueBytesPerStage = kTileM * kTileK;
  static constexpr int kWeightScaleSmemOffset = kWeightValueSmemOffset + kWeightValueBytesPerStage;
  static constexpr int kWeightScaleBytesPerStage = kScaleRowsPadded * kScaleColsPerKTile;
  static constexpr int kWeightStageBytes =
      ((kWeightScaleSmemOffset + kWeightScaleBytesPerStage + kSmemAlignment - 1) / kSmemAlignment) *
      kSmemAlignment;
  static constexpr int kReusableWeightStages = kPipelineStages;
  // Eight stages overlap the attention tail without extending the live range.
  static constexpr int kPrefetchWeightStages = 8;

  static constexpr int kAccumulatorTmemCols = kTileN;
  static constexpr int kScaleFactorTmemColsA = 4;
  static constexpr int kScaleFactorTmemColsB = 4;
  static constexpr int kScaleFactorBuffers = kPipelineStages;
  static constexpr int kScaleFactorTmemCols = kScaleFactorTmemColsA + kScaleFactorTmemColsB;
  static constexpr int kTotalTmemCols =
      kAccumulatorTmemCols + kScaleFactorBuffers * kScaleFactorTmemCols;
  static constexpr int kWeightFullBarrierBase = 0;
  static constexpr int kWeightEmptyBarrierBase = kWeightFullBarrierBase + kReusableWeightStages;
  static constexpr int kEpilogueFullBarrierSlot = kWeightEmptyBarrierBase + kReusableWeightStages;
  static constexpr int kActivationReadyBarrierBase = kEpilogueFullBarrierSlot + 1;
  static constexpr int kTcReadyBarrierBase = kActivationReadyBarrierBase + kHandoffGroupCount;
  static constexpr int kAccumulatorReadyBarrierSlot = kTcReadyBarrierBase + kPipelineStages;
  static constexpr int kSmemValueElementsA = cute::cosize_v<SmemLayoutA>;
  static constexpr int kSmemValueElementsB = cute::cosize_v<SmemLayoutB>;
  static constexpr int kSmemScaleElementsSFA = cute::cosize_v<SmemLayoutSFA>;
  static constexpr int kSmemScaleElementsSFB = cute::cosize_v<SmemLayoutSFB>;

  static_assert(Traits::kHeadDim == 128, "Output Projection expects attention head dim 128.");
  static_assert(Traits::kHeadGroup == 16, "Output Projection expects 16 attention heads.");
  static_assert(kProblemK == Traits::kHeadGroup * Traits::kHeadDim,
                "Output Projection K must match flattened attention output.");
  static_assert(kProblemM % kTileM == 0, "Output Projection M must use complete CTA M tiles.");
  static_assert(kOutputMTiles % kProblemN == 0,
                "Output Projection M tiles must divide across token clusters.");
  static_assert(kOutputMTilesPerToken == 6,
                "hidden size 6144 requires six Output Projection M tiles per token cluster.");
  static_assert(kAttentionCtas == 64, "attention path requires 64 CTAs.");
  static_assert(kDedicatedOutputProjectionCtas == 48,
                "dedicated Output Projection requires one CTA per M tile.");
  static_assert(kGridCtas == 112, "dedicated Output Projection grid must contain 112 CTAs.");
  static_assert(kOutputMTilesPerToken <= kOutputProjectionSplit,
                "Output Projection M tiles must fit the split=8 CTA cluster.");
  static_assert(Traits::kMaxQTokensPerBatch == kProblemN,
                "Output Projection token count must match the attention grid.");
  static_assert(Traits::kNumKvSplits == kOutputProjectionSplit,
                "Output Projection requires split=8 attention clusters.");
  static_assert(Sm100M3DecodeTp4AllReduce<Traits>::kHiddenSize == kProblemM,
                "Output Projection M must match the TP4 hidden size.");
  static_assert(kProblemK % kTileK == 0, "Output Projection K must use complete CTA K tiles.");
  static_assert(kTileK % kInstructionK == 0,
                "Output Projection K tile must contain complete MMA instructions.");
  static_assert(kInstructionK == 32, "MXFP8 blockscaled MMA consumes K=32 per instruction.");
  static_assert(kScaleVectorSize == 32, "MXFP8 scale vector must cover 32 K elements.");
  static_assert(kMmaPerKTile == 4, "Output Projection cta_k=128 must issue four MMA steps.");
  static_assert(kTcWarps == kMmaPerKTile, "Output Projection requires one TC warp per MMA step.");
  static_assert(kSmemValueElementsB == kActivationValueBytesPerStage,
                "activation value SMEM layout must cover one 8x128 tile.");
  static_assert(kActivationScaleBytesPerStage == 512,
                "activation C-UTCCP scale stage should be 512 bytes.");
  static_assert(kSmemScaleElementsSFB == kActivationScaleBytesPerStage,
                "activation SFB SMEM layout must describe one 512B block.");
  static_assert(kActivationStageBytes == 1536,
                "activation SMEM stage should contain value plus padded scales.");
  static_assert(kActivationStageStride >= kActivationStageBytes,
                "activation stage stride must cover value and scale payloads.");
  static_assert(kActivationSmemBaseOffset + (kActivationStages - 1) * kActivationStageStride +
                        kActivationStageBytes <=
                    Traits::kClusterReductionSmemBytes,
                "activation stages must fit the recycled combine SMEM.");
  static_assert(kSmemValueElementsA == kWeightValueBytesPerStage,
                "weight value SMEM layout must cover one 128x128 tile.");
  static_assert(kWeightScaleBytesPerStage == 512,
                "weight C-UTCCP scale stage should be 512 bytes.");
  static_assert(kSmemScaleElementsSFA == kWeightScaleBytesPerStage,
                "weight SFA SMEM layout must describe one 512B block.");
  static_assert(kWeightStageBytes == 16896,
                "weight stage should be one 128x128 value tile plus scales.");
  static_assert(kWeightStageBytes <= Traits::kRawKvStageBytes,
                "Output Projection weight must fit one recycled KV stage.");
  static_assert(kReusableWeightStages <= Traits::kNumStagesRawKv,
                "Output Projection stages must fit the recycled KV buffer.");
  static_assert(kKStages == 16, "Output Projection expects sixteen K stages for K=2048.");
  static_assert(kKStages == mega_atten::m3_decode::output_projection::kActivationStageCount,
                "Output Projection handoff stages must match K.");
  static_assert(kKStages % kOutputProjectionSplit == 0,
                "Output Projection stages must divide across split producers.");
  static_assert(kStageProducerCount < 256,
                "Output Projection producer count must fit the packed counter.");
  static_assert(kHandoffStagesPerGroup == 4,
                "Output Projection handoff groups must use four K stages.");
  static_assert(kKStages % kHandoffStagesPerGroup == 0,
                "Output Projection stages must divide into handoff groups.");
  static_assert(kHandoffStagesPerGroup % kRowsPerProducer == 0,
                "handoff groups must contain complete producer rows.");
  static_assert(kPrefetchWeightStages == 8,
                "Output Projection prefetch must cover eight weight stages.");
  static_assert(Traits::kSparseTopK % kOutputProjectionSplit == 0,
                "Output Projection expects an even sparse tile split.");
  static_assert(kRawKvEventsPerCta == 4,
                "split=8 Output Projection expects four raw-KV events per CTA.");
  static_assert(kTotalTmemCols == 72, "Output Projection requires 72 TMEM columns.");
  static_assert(kTotalTmemCols <= Traits::kNumTmemCols,
                "Output Projection TMEM must fit the existing allocation.");
  static_assert(kEpilogueFullBarrierSlot < Storage::ReusablePipelineStorage::kNumBarriers,
                "Output Projection barriers must fit shared storage.");
  static_assert(kActivationReadyBarrierBase + kHandoffGroupCount <=
                    Storage::ReusablePipelineStorage::kNumBarriers,
                "Output Projection activation barriers must fit shared storage.");
  static_assert(kAccumulatorReadyBarrierSlot < Storage::ReusablePipelineStorage::kNumBarriers,
                "Output Projection TC barriers must fit shared storage.");

  struct SmemOverlay {
    Storage *storage;

    CUTLASS_DEVICE uint8_t *activation_stage_base(int stage) const {
      return reinterpret_cast<uint8_t *>(&storage->smem_cluster_reduction) +
             kActivationSmemBaseOffset + stage * kActivationStageStride;
    }

    CUTLASS_DEVICE uint8_t *activation_value_ptr(int stage) const {
      return activation_stage_base(stage) + kActivationValueSmemOffset;
    }

    CUTLASS_DEVICE uint8_t *activation_scale_ptr(int stage) const {
      return activation_stage_base(stage) + kActivationScaleSmemOffset;
    }

    CUTLASS_DEVICE uint8_t *weight_stage_base(int stage) const {
      return storage->smem_kv.raw_ptr(stage);
    }

    CUTLASS_DEVICE uint8_t *weight_value_ptr(int stage) const {
      return weight_stage_base(stage) + kWeightValueSmemOffset;
    }

    CUTLASS_DEVICE uint8_t *weight_scale_ptr(int stage) const {
      return weight_stage_base(stage) + kWeightScaleSmemOffset;
    }
  };

  CUTLASS_DEVICE static SmemOverlay smem_overlay(Storage &storage) { return SmemOverlay{&storage}; }

  CUTLASS_DEVICE static bool is_attention_cta() {
    return m3_decode_logical_block_idx() < kAttentionCtas;
  }

  CUTLASS_DEVICE static bool is_output_projection_cta() { return !is_attention_cta(); }

  CUTLASS_DEVICE static int output_projection_m_tile() {
    return m3_decode_logical_block_idx() - kAttentionCtas;
  }
};

} // namespace cutlass::m3_decode::collective
