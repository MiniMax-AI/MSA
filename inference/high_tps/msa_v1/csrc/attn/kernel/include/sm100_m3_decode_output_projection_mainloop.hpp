#pragma once

#include "attn/kernel/include/sm100_m3_decode_output_projection_attention_handoff.hpp"

namespace cutlass::m3_decode::collective {

template <class Traits>
struct Sm100M3DecodeOutputProjectionMainloop : Sm100M3DecodeOutputProjectionHandoff<Traits> {
  using Base = Sm100M3DecodeOutputProjectionHandoff<Traits>;
  using Base::handoff_consumer_epochs;
  using Base::is_output_projection_cta;
  using Base::kAccumulatorReadyBarrierSlot;
  using Base::kActivationReadyBarrierBase;
  using Base::kActivationStages;
  using Base::kActivationStageStride;
  using Base::kActivationValueBytesPerStage;
  using Base::kEpilogueFullBarrierSlot;
  using Base::kHandoffGroupCount;
  using Base::kHandoffStagesPerGroup;
  using Base::kInitialWeightStage;
  using Base::kKStages;
  using Base::kPrefetchWeightStages;
  using Base::kReusableWeightStages;
  using Base::kScaleFactorBuffers;
  using Base::kScaleFactorTmemCols;
  using Base::kTcReadyBarrierBase;
  using Base::kTcWarps;
  using Base::kTileK;
  using Base::kTileM;
  using Base::kTileN;
  using Base::kWeightEmptyBarrierBase;
  using Base::kWeightFullBarrierBase;
  using Base::kWeightScaleBytesPerStage;
  using Base::kWeightValueBytesPerStage;
  using Base::load_relaxed_gpu;
  using Base::output_projection_m_tile;
  using Base::smem_overlay;
  using Base::store_relaxed_gpu;
  using Base::wait_activation_group;
  using typename Base::AccumulatorShape;
  using typename Base::ElementA;
  using typename Base::ElementB;
  using typename Base::ElementScale;
  using typename Base::M3DecodeBarriers;
  using typename Base::Params;
  using typename Base::SmemLayoutA;
  using typename Base::SmemLayoutB;
  using typename Base::SmemLayoutSFA;
  using typename Base::SmemLayoutSFB;
  using typename Base::SmemOverlay;
  using typename Base::Storage;
  using typename Base::TiledMma;
  using typename Base::UtccpOp;
  static constexpr int kPipelineBarrierCount = kAccumulatorReadyBarrierSlot + 1;
  static_assert(kPipelineBarrierCount == Storage::kOutputProjectionBarrierCount);

  struct ScaleTensorSet {
    decltype(cute::make_tensor<typename TiledMma::FrgTypeC>(AccumulatorShape{})) acc;
    decltype(cute::make_tensor(cute::make_smem_ptr(static_cast<ElementScale *>(nullptr)),
                               SmemLayoutSFA{})) sfa;
    decltype(cute::make_tensor(cute::make_smem_ptr(static_cast<ElementScale *>(nullptr)),
                               SmemLayoutSFB{})) sfb;
    decltype(cute::make_tensor<typename TiledMma::FrgTypeSFA>(
        cute::shape(SmemLayoutSFA{}))) tmem_sfa;
    decltype(cute::make_tensor<typename TiledMma::FrgTypeSFB>(
        cute::shape(SmemLayoutSFB{}))) tmem_sfb;
  };

  CUTLASS_DEVICE static ScaleTensorSet make_scale_tensors(SmemOverlay overlay, int weight_stage,
                                                          int activation_stage) {
    auto acc = cute::make_tensor<typename TiledMma::FrgTypeC>(AccumulatorShape{});
    auto sfa = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<ElementScale *>(
                                     overlay.weight_scale_ptr(weight_stage))),
                                 SmemLayoutSFA{});
    auto sfb = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<ElementScale *>(
                                     overlay.activation_scale_ptr(activation_stage))),
                                 SmemLayoutSFB{});
    auto tmem_sfa = cute::make_tensor<typename TiledMma::FrgTypeSFA>(cute::shape(SmemLayoutSFA{}));
    auto tmem_sfb = cute::make_tensor<typename TiledMma::FrgTypeSFB>(cute::shape(SmemLayoutSFB{}));
    return {acc, sfa, sfb, tmem_sfa, tmem_sfb};
  }

  CUTLASS_DEVICE static void bind_tmem_base(ScaleTensorSet &tensors, uint32_t tmem_base,
                                            int scale_buffer) {
    tensors.acc.data() = tmem_base;
    tensors.tmem_sfa.data() = tensors.acc.data().get() +
                              cutlass::detail::find_tmem_tensor_col_offset(tensors.acc) +
                              scale_buffer * kScaleFactorTmemCols;
    tensors.tmem_sfb.data() = tensors.tmem_sfa.data().get() +
                              cutlass::detail::find_tmem_tensor_col_offset(tensors.tmem_sfa);
  }

  template <class FragmentA, class FragmentB>
  CUTLASS_DEVICE static void issue_mma_block(ScaleTensorSet &tensors, FragmentA const &frag_a,
                                             FragmentB const &frag_b, TiledMma &tiled_mma,
                                             int k_block, int weight_stage_idx,
                                             int activation_stage_idx) {
    tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
    cute::gemm(tiled_mma.with(tiled_mma.accumulate_, tensors.tmem_sfa(cute::_, cute::_, k_block),
                              tensors.tmem_sfb(cute::_, cute::_, k_block)),
               frag_a(cute::_, cute::_, k_block, weight_stage_idx),
               frag_b(cute::_, cute::_, k_block, activation_stage_idx), tensors.acc);
  }

  CUTLASS_DEVICE static void initialize_accumulator(uint32_t tmem_base, int tc_warp_idx) {
    uint32_t zeros[8] = {};
    uint32_t const datapath_offset = static_cast<uint32_t>(tc_warp_idx * 32) << 16;
    cute::SM100_TMEM_STORE_32dp32b8x::copy(zeros[0], zeros[1], zeros[2], zeros[3], zeros[4],
                                           zeros[5], zeros[6], zeros[7],
                                           tmem_base + datapath_offset);
    cutlass::arch::fence_view_async_tmem_store();
  }

  template <class SourceTensor, class DestinationTensor>
  CUTLASS_DEVICE static void issue_utccp(SourceTensor const &src, DestinationTensor const &dst) {
    UtccpOp::copy(src[0], cute::raw_pointer_cast(dst.data()));
  }

  CUTLASS_DEVICE static uint64_t *weight_full_barrier(Storage &storage, int stage) {
    return storage.phase_barrier_ptr(Storage::kOutputProjectionBarrierBase +
                                     kWeightFullBarrierBase + stage);
  }

  CUTLASS_DEVICE static uint64_t *weight_empty_barrier(Storage &storage, int stage) {
    return storage.phase_barrier_ptr(Storage::kOutputProjectionBarrierBase +
                                     kWeightEmptyBarrierBase + stage);
  }

  CUTLASS_DEVICE static uint64_t *epilogue_full_barrier(Storage &storage) {
    return storage.phase_barrier_ptr(Storage::kOutputProjectionBarrierBase +
                                     kEpilogueFullBarrierSlot);
  }

  CUTLASS_DEVICE static uint64_t *activation_ready_barrier(Storage &storage, int stage) {
    return storage.phase_barrier_ptr(Storage::kOutputProjectionBarrierBase +
                                     kActivationReadyBarrierBase + stage);
  }

  CUTLASS_DEVICE static uint64_t *tc_ready_barrier(Storage &storage, int stage) {
    return storage.phase_barrier_ptr(Storage::kOutputProjectionBarrierBase + kTcReadyBarrierBase +
                                     stage);
  }

  CUTLASS_DEVICE static uint64_t *accumulator_ready_barrier(Storage &storage) {
    return storage.phase_barrier_ptr(Storage::kOutputProjectionBarrierBase +
                                     kAccumulatorReadyBarrierSlot);
  }

  template <int Stages> CUTLASS_DEVICE static uint32_t full_phase(int event) {
    return static_cast<uint32_t>((event / Stages) & 1);
  }

  template <int Stages> CUTLASS_DEVICE static uint32_t empty_phase(int event) {
    return static_cast<uint32_t>(1 ^ ((event / Stages) & 1));
  }

  CUTLASS_DEVICE static void init_output_projection_pipeline(Storage &storage) {
    if (cute::elect_one_sync()) {
      for (int stage = 0; stage < kReusableWeightStages; ++stage) {
        cute::initialize_barrier(weight_full_barrier(storage, stage)[0], 2);
        cute::initialize_barrier(weight_empty_barrier(storage, stage)[0], kTcWarps);
        cute::initialize_barrier(tc_ready_barrier(storage, stage)[0], 1);
      }
      for (int group = 0; group < kHandoffGroupCount; ++group) {
        cute::initialize_barrier(activation_ready_barrier(storage, group)[0], 1);
      }
      cute::initialize_barrier(accumulator_ready_barrier(storage)[0], kTcWarps);
      cute::initialize_barrier(epilogue_full_barrier(storage)[0], kTcWarps);
    }
    cutlass::arch::fence_barrier_init();
    __syncwarp();
  }

  CUTLASS_DEVICE static int weight_stage(int k_stage) {
    return (k_stage + kInitialWeightStage) % kReusableWeightStages;
  }

  CUTLASS_DEVICE static int activation_stage(int k_stage) { return k_stage % kActivationStages; }

  CUTLASS_DEVICE static void issue_weight_load(Storage &storage, Params const &params,
                                               SmemOverlay overlay, int m_tile, int k_stage) {
    int const stage = weight_stage(k_stage);
    Sm100M3DecodeBarrier::wait(weight_empty_barrier(storage, stage),
                               empty_phase<kReusableWeightStages>(k_stage));
    if (cute::elect_one_sync()) {
      uint64_t *full = weight_full_barrier(storage, stage);
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
          full, kWeightValueBytesPerStage + kWeightScaleBytesPerStage);
      Sm100M3DecodeTma::load_2d_predicated(
          &params.tma.output_projection_w, overlay.weight_value_ptr(stage), full, k_stage * kTileK,
          m_tile * kTileM, 1, cute::TMA::CacheHintSm90::EVICT_FIRST);
      Sm100M3DecodeTma::load_3d_predicated(
          &params.tma.output_projection_w_sf, overlay.weight_scale_ptr(stage), full, 0, 0,
          m_tile * kKStages + k_stage, 1, cute::TMA::CacheHintSm90::EVICT_FIRST);
    }
  }

  CUTLASS_DEVICE static void load_weight_pipeline(Storage &storage, Params const &params,
                                                  SmemOverlay overlay, int m_tile) {
    CUTLASS_PRAGMA_UNROLL
    for (int k_stage = kPrefetchWeightStages; k_stage < kKStages; ++k_stage) {
      issue_weight_load(storage, params, overlay, m_tile, k_stage);
    }
  }

  CUTLASS_DEVICE static void prefetch_weight_task(Storage &storage, Params const &params,
                                                  int lane_idx, int raw_event_count, int m_tile) {
    init_output_projection_pipeline(storage);
    if (cute::elect_one_sync()) {
      Sm100M3DecodeTma::prefetch_tensormap(&params.tma.output_projection_w);
      Sm100M3DecodeTma::prefetch_tensormap(&params.tma.output_projection_w_sf);
      Sm100M3DecodeTma::prefetch_tensormap(&params.tma.output_projection_o);
      Sm100M3DecodeTma::prefetch_tensormap(&params.tma.output_projection_o_sf_padded);
    }

    auto overlay = smem_overlay(storage);
    CUTLASS_PRAGMA_UNROLL
    for (int k_stage = 0; k_stage < kPrefetchWeightStages; ++k_stage) {
      int const stage = weight_stage(k_stage);
      if (stage < raw_event_count) {
        int const final_raw_event =
            stage +
            ((raw_event_count - 1 - stage) / Traits::kNumStagesRawKv) * Traits::kNumStagesRawKv;
        Sm100M3DecodeBarrier::wait(
            storage.attention_pipelines().ptr(M3DecodeBarriers::kKvEmptyArv1 + stage),
            full_phase<Traits::kNumStagesRawKv>(final_raw_event));
      }
      issue_weight_load(storage, params, overlay, m_tile, k_stage);
    }
    int const l2_prefetch_stage = kPrefetchWeightStages + lane_idx;
    if (l2_prefetch_stage < kKStages) {
      cute::SM90_TMA_LOAD_2D::PREFETCH::copy(&params.tma.output_projection_w,
                                             l2_prefetch_stage * kTileK, m_tile * kTileM);
    }
  }

  CUTLASS_DEVICE static void prefetch_weight(Storage &storage, Params const &params, int lane_idx,
                                             int raw_event_count) {
    if (!is_output_projection_cta()) {
      return;
    }
    prefetch_weight_task(storage, params, lane_idx, raw_event_count, output_projection_m_tile());
  }

  CUTLASS_DEVICE static void poll_activation_pipeline(Storage &storage, Params const &params,
                                                      int m_tile) {
    if (cute::elect_one_sync()) {
      uint32_t *consumer_epoch = handoff_consumer_epochs(params) + m_tile;
      uint32_t const epoch = load_relaxed_gpu(consumer_epoch);
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < kHandoffGroupCount; ++group) {
        wait_activation_group(params, group, epoch);
        cutlass::arch::ClusterBarrier::arrive(activation_ready_barrier(storage, group));
      }
      store_relaxed_gpu(consumer_epoch, epoch + 1);
    }
  }

  CUTLASS_DEVICE static void load_activation_pipeline(Storage &storage, Params const &params,
                                                      SmemOverlay overlay) {
    CUTLASS_PRAGMA_UNROLL
    for (int k_stage = 0; k_stage < kKStages; ++k_stage) {
      int const weight_stage_idx = weight_stage(k_stage);
      int const activation_stage_idx = activation_stage(k_stage);
      if (k_stage >= kActivationStages) {
        int const previous_event = k_stage - kActivationStages;
        Sm100M3DecodeBarrier::wait(weight_empty_barrier(storage, weight_stage(previous_event)),
                                   full_phase<kReusableWeightStages>(previous_event));
      }

      if (k_stage % kHandoffStagesPerGroup == 0) {
        Sm100M3DecodeBarrier::wait(
            activation_ready_barrier(storage, k_stage / kHandoffStagesPerGroup), 0);
      }

      uint64_t *full = weight_full_barrier(storage, weight_stage_idx);
      uint8_t *act_smem = overlay.activation_value_ptr(activation_stage_idx);
      uint8_t *act_sf_smem = overlay.activation_scale_ptr(activation_stage_idx);
      if (cute::elect_one_sync()) {
        constexpr uint32_t kActivationTransactionBytes =
            kActivationValueBytesPerStage + kTileN * 16;
        cutlass::arch::ClusterTransactionBarrier::expect_transaction(full,
                                                                     kActivationTransactionBytes);
        Sm100M3DecodeTma::load_3d_predicated(&params.tma.output_projection_o, act_smem, full, 0,
                                             k_stage, 0, 1, cute::TMA::CacheHintSm90::EVICT_LAST);
        Sm100M3DecodeTma::load_3d_predicated(&params.tma.output_projection_o_sf_padded, act_sf_smem,
                                             full, 0, 0, k_stage, 1,
                                             cute::TMA::CacheHintSm90::EVICT_LAST);
      }

      if (cute::elect_one_sync()) {
        cutlass::arch::ClusterBarrier::arrive(full);
      }
    }
  }

  CUTLASS_DEVICE static void run_scale_pipeline(Storage &storage, SmemOverlay overlay) {
    auto tensors = make_scale_tensors(overlay, 0, 0);
    uint32_t const tmem_base = storage.tmem_state_ptr()[0];
    bind_tmem_base(tensors, tmem_base, 0);

    auto sfa_staged_layout = cute::make_layout(
        cute::append(cute::shape(SmemLayoutSFA{}), cute::Int<kReusableWeightStages>{}),
        cute::append(cute::stride(SmemLayoutSFA{}), cute::Int<Traits::kRawKvStageBytes>{}));
    auto sfb_staged_layout = cute::make_layout(
        cute::append(cute::shape(SmemLayoutSFB{}), cute::Int<kActivationStages>{}),
        cute::append(cute::stride(SmemLayoutSFB{}), cute::Int<kActivationStageStride>{}));
    auto sfa_staged = cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<ElementScale *>(overlay.weight_scale_ptr(0))),
        sfa_staged_layout);
    auto sfb_staged = cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<ElementScale *>(overlay.activation_scale_ptr(0))),
        sfb_staged_layout);
    auto sfa_compact =
        cute::make_tensor(sfa_staged.data(), cute::filter_zeros(sfa_staged.layout()));
    auto sfb_compact =
        cute::make_tensor(sfb_staged.data(), cute::filter_zeros(sfb_staged.layout()));
    auto tmem_sfa_compact =
        cute::make_tensor(tensors.tmem_sfa.data(), cute::filter_zeros(tensors.tmem_sfa.layout()));
    auto tmem_sfb_compact =
        cute::make_tensor(tensors.tmem_sfb.data(), cute::filter_zeros(tensors.tmem_sfb.layout()));
    auto copy_sfa = cute::make_utccp_copy(UtccpOp{}, tmem_sfa_compact);
    auto copy_sfb = cute::make_utccp_copy(UtccpOp{}, tmem_sfb_compact);
    auto thr_sfa = copy_sfa.get_slice(0);
    auto thr_sfb = copy_sfb.get_slice(0);
    auto src_sfa = cute::get_utccp_smem_desc_tensor<UtccpOp>(thr_sfa.partition_S(sfa_compact));
    auto src_sfb = cute::get_utccp_smem_desc_tensor<UtccpOp>(thr_sfb.partition_S(sfb_compact));
    CUTLASS_PRAGMA_UNROLL
    for (int k_stage = 0; k_stage < kKStages; ++k_stage) {
      int const weight_stage_idx = weight_stage(k_stage);
      int const activation_stage_idx = activation_stage(k_stage);
      Sm100M3DecodeBarrier::wait(weight_full_barrier(storage, weight_stage_idx),
                                 full_phase<kReusableWeightStages>(k_stage));
      bind_tmem_base(tensors, tmem_base, k_stage % kScaleFactorBuffers);

      if (cute::elect_one_sync()) {
        issue_utccp(src_sfa(cute::_, cute::_, cute::_, cute::_, weight_stage_idx),
                    tensors.tmem_sfa);
        issue_utccp(src_sfb(cute::_, cute::_, cute::_, cute::_, activation_stage_idx),
                    tensors.tmem_sfb);
      }
      cutlass::arch::umma_arrive(tc_ready_barrier(storage, weight_stage_idx));
    }
  }

  CUTLASS_DEVICE static void run_tc_pipeline(Storage &storage, SmemOverlay overlay,
                                             int tc_warp_idx) {
    auto tiled_mma = TiledMma{};
    auto tensors = make_scale_tensors(overlay, 0, 0);
    uint32_t const tmem_base = storage.tmem_state_ptr()[0];
    bind_tmem_base(tensors, tmem_base, 0);

    auto a_layout_atom = SmemLayoutA{};
    auto b_layout_atom = SmemLayoutB{};
    auto a_staged_inner =
        cute::make_layout(cute::append(cute::take<0, 3>(cute::shape(a_layout_atom.layout_b())),
                                       cute::Int<kReusableWeightStages>{}),
                          cute::append(cute::take<0, 3>(cute::stride(a_layout_atom.layout_b())),
                                       cute::Int<Traits::kRawKvStageBytes>{}));
    auto b_staged_inner =
        cute::make_layout(cute::append(cute::take<0, 3>(cute::shape(b_layout_atom.layout_b())),
                                       cute::Int<kActivationStages>{}),
                          cute::append(cute::take<0, 3>(cute::stride(b_layout_atom.layout_b())),
                                       cute::Int<kActivationStageStride>{}));
    auto a_staged_layout = cute::make_composed_layout(a_layout_atom.layout_a(),
                                                      a_layout_atom.offset(), a_staged_inner);
    auto b_staged_layout = cute::make_composed_layout(b_layout_atom.layout_a(),
                                                      b_layout_atom.offset(), b_staged_inner);
    auto a_staged = cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<ElementA *>(overlay.weight_value_ptr(0))),
        a_staged_layout);
    auto b_staged = cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<ElementB *>(overlay.activation_value_ptr(0))),
        b_staged_layout);
    auto frag_a = TiledMma::make_fragment_A(a_staged);
    auto frag_b = TiledMma::make_fragment_B(b_staged);
    CUTLASS_PRAGMA_UNROLL
    for (int k_stage = 0; k_stage < kKStages; ++k_stage) {
      int const weight_stage_idx = weight_stage(k_stage);
      int const activation_stage_idx = activation_stage(k_stage);
      uint64_t *tc_ready = tc_ready_barrier(storage, weight_stage_idx);
      cutlass::BarrierStatus const tc_ready_token = Sm100M3DecodeBarrier::try_wait_token(
          tc_ready, full_phase<kReusableWeightStages>(k_stage));
      Sm100M3DecodeBarrier::wait(tc_ready_barrier(storage, weight_stage_idx),
                                 full_phase<kReusableWeightStages>(k_stage), tc_ready_token);
      bind_tmem_base(tensors, tmem_base, k_stage % kScaleFactorBuffers);

      if (k_stage == 0) {
        initialize_accumulator(tmem_base, tc_warp_idx);
        if (cute::elect_one_sync()) {
          cutlass::arch::ClusterBarrier::arrive(accumulator_ready_barrier(storage));
        }
        Sm100M3DecodeBarrier::wait(accumulator_ready_barrier(storage), 0);
      }

      issue_mma_block(tensors, frag_a, frag_b, tiled_mma, tc_warp_idx, weight_stage_idx,
                      activation_stage_idx);
      cutlass::arch::umma_arrive(weight_empty_barrier(storage, weight_stage_idx));
    }
    cutlass::arch::umma_arrive(epilogue_full_barrier(storage));
  }
};

} // namespace cutlass::m3_decode::collective
