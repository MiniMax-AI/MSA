#pragma once

#include "attn/kernel/include/sm100_m3_decode_attention_projection_input_norm_quant.hpp"

namespace cutlass::m3_decode::collective::attention_projection {

struct HeadSplit {
  int head;
  int split;
  int split_count;
};

CUTE_DEVICE HeadSplit head_split_for_cta(int cluster_id, int rank) {
  if (rank < 3) {
    return {2 * cluster_id, rank, kQSplitK};
  }
  if (rank < 6) {
    return {2 * cluster_id + 1, rank - 3, kQSplitK};
  }
  return {16 + cluster_id / 2, 2 * (cluster_id & 1) + rank - 6, kSpecialSplitK};
}

template <int StartTile, int CommonTileCount, int QTailTileCount = 0>
CUTE_DEVICE void produce_weight_tma_phase(KernelParams const &params, SharedStorage &shared,
                                          BarrierStorage &barriers, HeadSplit hs, uint32_t leader) {
  static_assert(StartTile >= 0);
  static_assert(CommonTileCount > 0);
  static_assert(QTailTileCount >= 0);
  static_assert(StartTile + CommonTileCount + QTailTileCount <= 16);
  using X = cute::Underscore;
  auto m_weight = params.tma.weight.get_tma_tensor(
      cute::make_shape(cute::Int<kOutput>{}, cute::Int<kHidden>{}, cute::_1{}));
  auto g_weight = cute::local_tile(m_weight, MmaTileShape{}, cute::make_coord(_, _, _),
                                   cute::Step<cute::_1, X, cute::_1>{});
  auto m_sfa = params.tma.weight_scale.get_tma_tensor(cute::shape(WeightScaleLayout{}));
  auto g_sfa = cute::local_tile(m_sfa, MmaTileShape{}, cute::make_coord(_, _, _),
                                cute::Step<cute::_1, X, cute::_1>{});
  auto cta_mma = DirectM128N8Mxfp8{}.get_slice(0);
  auto tCgWeight = cta_mma.partition_A(g_weight);
  auto tCgSfa = cta_mma.partition_A(g_sfa);
  int const tiles_per_split = hs.split_count == kQSplitK ? 16 : 12;

  auto produce_tile = [&](auto tile_index) {
    constexpr int tile = decltype(tile_index)::value;
    constexpr int stage = tile % kAbStages;
    constexpr int producer_phase = 1 ^ ((tile / kAbStages) & 1);
    int const global_tile = hs.split * tiles_per_split + tile;
    uint8_t *stage_base = shared.ab_ring[stage];
    auto s_weight = cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<ElementAB *>(stage_base)), SmemLayoutA{});
    auto s_sfa = cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<ElementSF *>(stage_base + 128 * 128 + 8 * 128)),
        SmemLayoutSFA{});
    auto [tWeightG, tWeightS] = cute::tma_partition(
        params.tma.weight, cute::group_modes<0, 3>(s_weight), cute::group_modes<0, 3>(tCgWeight));
    auto [tSfaG, tSfaS] = cute::tma_partition(
        params.tma.weight_scale, cute::group_modes<0, 3>(s_sfa), cute::group_modes<0, 3>(tCgSfa));

    if (leader) {
      if constexpr (tile == 0) {
#pragma unroll
        for (int prefetch_tile = 0; prefetch_tile < kPrefetchDistance; ++prefetch_tile) {
          int const prefetch_global = hs.split * tiles_per_split + prefetch_tile;
          cute::prefetch(params.tma.weight, tWeightG(_, hs.head, prefetch_global, 0));
          auto prefetch_sfa =
              cute::group_modes<0, 2>(cute::filter_zeros(tSfaG(_, hs.head, prefetch_global, _)));
          cute::prefetch(params.tma.weight_scale, prefetch_sfa);
        }
      }
      cute::wait_barrier(barriers.ab_empty[stage], producer_phase);
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&barriers.ab_full[stage],
                                                                     128 * 128 + 512);
      cute::copy(
          params.tma.weight.with(barriers.ab_full[stage], 0, cute::TMA::CacheHintSm90::EVICT_FIRST),
          tWeightG(_, hs.head, global_tile, 0), tWeightS);
      auto tSfaGTile = cute::filter_zeros(tSfaG(_, hs.head, global_tile, _));
      auto tSfaSTile = cute::filter_zeros(tSfaS);
      auto tSfaGGrouped = cute::group_modes<0, 2>(tSfaGTile);
      cute::copy(params.tma.weight_scale.with(barriers.ab_full[stage], 0,
                                              cute::TMA::CacheHintSm90::EVICT_FIRST),
                 tSfaGGrouped, tSfaSTile);
      constexpr int future_tile = tile + kPrefetchDistance;
      if constexpr (future_tile < 9) {
        int const future_global = hs.split * tiles_per_split + future_tile;
        cute::prefetch(params.tma.weight, tWeightG(_, hs.head, future_global, 0));
        auto future_sfa =
            cute::group_modes<0, 2>(cute::filter_zeros(tSfaG(_, hs.head, future_global, _)));
        cute::prefetch(params.tma.weight_scale, future_sfa);
      }
    }
  };

  cute::for_each(cute::make_seq<CommonTileCount>{}, [&](auto tile_offset) {
    produce_tile(cute::Int<StartTile + decltype(tile_offset)::value>{});
  });
  if constexpr (QTailTileCount > 0) {
    if (hs.split_count == kQSplitK) {
      cute::for_each(cute::make_seq<QTailTileCount>{}, [&](auto tile_offset) {
        produce_tile(cute::Int<StartTile + CommonTileCount + decltype(tile_offset)::value>{});
      });
    }
  }
}

CUTE_DEVICE void produce_activation_tma(KernelParams const &params, SharedStorage &shared,
                                        BarrierStorage &barriers, HeadSplit hs, int lane,
                                        uint32_t leader) {
  using X = cute::Underscore;
  auto m_activation = params.tma.activation.get_tma_tensor(
      cute::make_shape(cute::_8{}, cute::Int<kHidden>{}, cute::_1{}));
  auto g_activation = cute::local_tile(m_activation, MmaTileShape{}, cute::make_coord(_, _, _),
                                       cute::Step<X, cute::_1, cute::_1>{});
  auto cta_mma = DirectM128N8Mxfp8{}.get_slice(0);
  auto tCgActivation = cta_mma.partition_B(g_activation);
  int const tiles_per_split = hs.split_count == kQSplitK ? 16 : 12;

  for (int tile = 0; tile < tiles_per_split; ++tile) {
    int const stage = tile % kAbStages;
    int const producer_phase = 1 ^ ((tile / kAbStages) & 1);
    int const global_tile = hs.split * tiles_per_split + tile;
    uint8_t *stage_base = shared.ab_ring[stage];
    auto s_activation = cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<ElementAB *>(stage_base + 128 * 128)), SmemLayoutB{});
    auto [tActivationG, tActivationS] =
        cute::tma_partition(params.tma.activation, cute::group_modes<0, 3>(s_activation),
                            cute::group_modes<0, 3>(tCgActivation));

    // Match the DSL producer protocol: every W1 lane observes the empty
    // stage, writes its one live SFB byte, and publishes that store to the
    // async proxy before the elected lane launches the activation TMA.
    cute::wait_barrier(barriers.ab_empty[stage], producer_phase);
    int const n = lane / (kCtaK / kSfVec);
    int const group = lane % (kCtaK / kSfVec);
    uint8_t *stage_sfb = stage_base + 128 * 128 + 8 * 128 + 512;
    stage_sfb[n * 16 + group] =
        params.workspace_scale[n * (kHidden / kSfVec) + global_tile * (kCtaK / kSfVec) + group];
    sm100::fence_proxy_async_shared();
    if (leader) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&barriers.ab_full[stage],
                                                                     kTokensPerTile * kCtaK);
      cute::copy(params.tma.activation.with(barriers.ab_full[stage]),
                 tActivationG(_, 0, global_tile, 0), tActivationS);
    }
  }
}

CUTE_DEVICE void prefetch_weight_tail(KernelParams const &params, SharedStorage &shared,
                                      HeadSplit hs) {
  using X = cute::Underscore;
  auto m_weight = params.tma.weight.get_tma_tensor(
      cute::make_shape(cute::Int<kOutput>{}, cute::Int<kHidden>{}, cute::_1{}));
  auto g_weight = cute::local_tile(m_weight, MmaTileShape{}, cute::make_coord(_, _, _),
                                   cute::Step<cute::_1, X, cute::_1>{});
  auto m_sfa = params.tma.weight_scale.get_tma_tensor(cute::shape(WeightScaleLayout{}));
  auto g_sfa = cute::local_tile(m_sfa, MmaTileShape{}, cute::make_coord(_, _, _),
                                cute::Step<cute::_1, X, cute::_1>{});
  auto cta_mma = DirectM128N8Mxfp8{}.get_slice(0);
  auto tCgWeight = cta_mma.partition_A(g_weight);
  auto tCgSfa = cta_mma.partition_A(g_sfa);
  auto s_weight = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<ElementAB *>(shared.ab_ring[0])), SmemLayoutA{});
  auto s_sfa = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<ElementSF *>(shared.ab_ring[0] + 128 * 128 + 8 * 128)),
      SmemLayoutSFA{});
  auto [tWeightG, tWeightS] = cute::tma_partition(
      params.tma.weight, cute::group_modes<0, 3>(s_weight), cute::group_modes<0, 3>(tCgWeight));
  auto [tSfaG, tSfaS] = cute::tma_partition(params.tma.weight_scale, cute::group_modes<0, 3>(s_sfa),
                                            cute::group_modes<0, 3>(tCgSfa));
  (void)tWeightS;
  (void)tSfaS;
  int const tiles = hs.split_count == kQSplitK ? 16 : 12;
  if (cute::elect_one_sync()) {
    for (int tile = 9; tile < tiles; ++tile) {
      int const global_tile = hs.split * tiles + tile;
      cute::prefetch(params.tma.weight, tWeightG(_, hs.head, global_tile, 0));
      auto tail_sfa =
          cute::group_modes<0, 2>(cute::filter_zeros(tSfaG(_, hs.head, global_tile, _)));
      cute::prefetch(params.tma.weight_scale, tail_sfa);
    }
  }
}

CUTE_DEVICE void initialize_ab6_and_tma_descriptors(KernelParams const &params,
                                                    SharedStorage &shared, BarrierStorage &barriers,
                                                    HeadSplit hs, int rank, int tid) {
  // The padded SFB bytes not written by W1 must remain canonical zero in every stage.
  if (tid < kAbStages * (512 / int(sizeof(uint4)))) {
    int const stage = tid / (512 / int(sizeof(uint4)));
    int const vector = tid % (512 / int(sizeof(uint4)));
    auto *stage_sfb = reinterpret_cast<uint4 *>(shared.ab_ring[stage] + 128 * 128 + 8 * 128 + 512);
    stage_sfb[vector] = make_uint4(0, 0, 0, 0);
  }
  if (tid == 0) {
#pragma unroll
    for (int stage = 0; stage < kAbStages; ++stage) {
      cute::initialize_barrier(barriers.ab_full[stage], 2);
      cute::initialize_barrier(barriers.ab_empty[stage], 1);
    }
    cute::initialize_barrier(barriers.mma_epilog, 1);
    cute::initialize_barrier(barriers.tmem_lifetime, 32 + 128);
    cute::initialize_barrier(barriers.input_mailbox, 1);
    for (int token = 0; token < kTokensPerTile; ++token) {
      cute::initialize_barrier(barriers.q_mailbox[token], 1);
    }
    cute::prefetch_tma_descriptor(params.tma.weight.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma.weight_scale.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma.activation.get_tma_descriptor());
  }
  sm100::fence_mbarrier_init();
  cutlass::arch::NamedBarrier(kThreads, kAttentionProjectionBarrierId).arrive_and_wait();
  if (tid == 0) {
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
        &barriers.input_mailbox, (kClusterSize - 1) * int(sizeof(float)));
    if (rank == 0 || rank == 3) {
      for (int token = 0; token < kTokensPerTile; ++token) {
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
            &barriers.q_mailbox[token], kQSplitK * kHeadDim * int(sizeof(float)));
      }
    }
    // Only tid0 mutates the receiver mbarriers after the init fence.  Publish those
    // expect-tx updates explicitly, then use the lighter control-only cluster rendezvous.
    asm volatile("fence.acq_rel.cluster;" : : : "memory");
  }
  cutlass::arch::NamedBarrier(kThreads, kAttentionProjectionBarrierId).arrive_and_wait();
}

CUTE_DEVICE void issue_early_weight_tma(KernelParams const &params, SharedStorage &shared,
                                        BarrierStorage &barriers, HeadSplit hs, int tid) {
  // W0's six-stage prologue deliberately runs before the all-warp input phase, overlapping the
  // cold weight/SFA TMA traffic with input RMSNorm and activation production.
  if ((tid >> 5) == 0) {
    uint32_t const producer_leader = cute::elect_one_sync();
    produce_weight_tma_phase<0, kAbStages>(params, shared, barriers, hs, producer_leader);
  }
}

template <int TileCount>
CUTE_DEVICE void drain_ab6_producer_constexpr(BarrierStorage &barriers, uint32_t leader) {
  if (leader) {
    cute::for_each(cute::make_seq<kAbStages>{}, [&](auto tail_index) {
      constexpr int logical_tile = TileCount + decltype(tail_index)::value;
      constexpr int stage = logical_tile % kAbStages;
      constexpr int producer_phase = 1 ^ ((logical_tile / kAbStages) & 1);
      cute::wait_barrier(barriers.ab_empty[stage], producer_phase);
    });
  }
}

CUTE_DEVICE void drain_ab6_producer(BarrierStorage &barriers, int tile_count, uint32_t leader) {
  if (leader) {
#pragma unroll
    for (int tail = 0; tail < kAbStages; ++tail) {
      int const logical_tile = tile_count + tail;
      int const stage = logical_tile % kAbStages;
      int const producer_phase = 1 ^ ((logical_tile / kAbStages) & 1);
      cute::wait_barrier(barriers.ab_empty[stage], producer_phase);
    }
  }
}

template <class Policy>
CUTE_DEVICE void stage_special_post_data(KernelParams const &params, SharedStorage &shared,
                                         HeadSplit hs, int lane) {
  if (hs.head == 17 || hs.split != 0)
    return;
  __nv_bfloat16 const *norm = params.q_norm_weight;
  auto *stage = reinterpret_cast<uint16_t *>(shared.input_raw);
  if (hs.head >= kNumQHeads) {
    norm = params.k_norm_weight;
    if (hs.head == 18)
      norm = params.qi_norm_weight;
    if (hs.head == 19)
      norm = params.ki_norm_weight;
    stage = reinterpret_cast<uint16_t *>(shared.q_mailbox);
  }
  sm100::copy_global_to_shared_bf16x4(reinterpret_cast<uint16_t const *>(norm) + lane * 4,
                                      stage + lane * 4);
#pragma unroll
  for (int round = 0; round < 4; ++round) {
    int const copy_index = round * 32 + lane;
    int const token = copy_index / (kRotaryDim / 4);
    int const vector = copy_index % (kRotaryDim / 4);
    if (Policy::valid_token(token, params)) {
      int64_t const position = params.positions[token];
      sm100::copy_global_to_shared_bf16x4(reinterpret_cast<uint16_t const *>(params.cos_sin_cache) +
                                              position * kRotaryDim + vector * 4,
                                          stage + kHeadDim + token * kRotaryDim + vector * 4);
    }
  }
}

template <class Policy>
CUTE_DEVICE void project_ab6_m128n8_mxfp8(KernelParams const &params, SharedStorage &shared,
                                          BarrierStorage &barriers, HeadSplit hs, int tid) {
  int const warp = cutlass::canonical_warp_idx_sync();
  int const lane = tid & 31;
  int const tiles_per_split = hs.split_count == kQSplitK ? 16 : 12;

  // W0 continues the rolling weight/SFA producer after its six-stage input-overlap prologue.
  if (warp == 0) {
    uint32_t const producer_leader = cute::elect_one_sync();
    produce_weight_tma_phase<kAbStages, kAbStages, 4>(params, shared, barriers, hs,
                                                      producer_leader);
    if (hs.split_count == kQSplitK) {
      drain_ab6_producer_constexpr<16>(barriers, producer_leader);
    } else {
      drain_ab6_producer_constexpr<12>(barriers, producer_leader);
    }
  }
  // W1 publishes the workspace activation and its hand-packed narrow-N scale bytes.
  if (warp == 1) {
    uint32_t const producer_leader = cute::elect_one_sync();
    produce_activation_tma(params, shared, barriers, hs, lane, producer_leader);
    drain_ab6_producer(barriers, tiles_per_split, producer_leader);
  }
  if (warp == 3) {
    stage_special_post_data<Policy>(params, shared, hs, lane);
  }

  if (warp == 2) {
    auto acc_shape = cute::partition_shape_C(DirectM128N8Mxfp8{}, cute::take<0, 2>(MmaTileShape{}));
    auto accum = DirectM128N8Mxfp8::make_fragment_C(acc_shape);
    cute::TMEM::Allocator1Sm allocator{};
    allocator.allocate(256, &shared.tmem_base);
    // Phase zero: W2's 32 lanes plus the 128 spill lanes establish that TMEM is allocated.
    cute::arrive_barrier(barriers.tmem_lifetime);
    allocator.release_allocation_lock();

    accum.data() = shared.tmem_base;
    auto tSFA = sm100::make_m128n8_sfa_tmem_tensor<DirectM128N8Mxfp8>(
        shared.tmem_base + cutlass::detail::find_tmem_tensor_col_offset(accum));
    auto tSFB = sm100::make_m128n8_sfb_tmem_tensor<DirectM128N8Mxfp8>(
        tSFA.data().get() + cutlass::detail::find_tmem_tensor_col_offset(tSFA));
    auto tSFACompact = cute::make_tensor(tSFA.data(), cute::filter_zeros(tSFA.layout()));
    auto tSFBCompact = cute::make_tensor(tSFB.data(), cute::filter_zeros(tSFB.layout()));
    auto copySfa = sm100::make_sf_s2t_copy(tSFACompact);
    auto copySfb = sm100::make_sf_s2t_copy(tSFBCompact);
    auto tSfaMma = sm100::make_sf_tmem_mma_view(tSFA);
    auto tSfbMma = sm100::make_sf_tmem_mma_view(tSFB);
    DirectM128N8Mxfp8 tiled_mma;
    tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;
    bool next_stage_ready = false;

    bool const is_q = hs.split_count == kQSplitK;
    auto consume_tile = [&](auto tile_index) {
      constexpr int tile = decltype(tile_index)::value;
      constexpr int stage = tile % kAbStages;
      constexpr int consumer_phase = (tile / kAbStages) & 1;
      if (!next_stage_ready) {
        cute::wait_barrier(barriers.ab_full[stage], consumer_phase);
      }
      next_stage_ready = false;
      if constexpr (tile < 11 || (tile >= 12 && tile < 15)) {
        constexpr int next_tile = tile + 1;
        next_stage_ready = cutlass::arch::ClusterBarrier::try_wait(
            &barriers.ab_full[next_tile % kAbStages], (next_tile / kAbStages) & 1);
      } else if constexpr (tile == 11) {
        if (is_q) {
          constexpr int next_tile = 12;
          next_stage_ready = cutlass::arch::ClusterBarrier::try_wait(
              &barriers.ab_full[next_tile % kAbStages], (next_tile / kAbStages) & 1);
        }
      }
      uint8_t *stage_base = shared.ab_ring[stage];
      auto sA = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<ElementAB *>(stage_base)),
                                  SmemLayoutA{});
      auto sB = cute::make_tensor(
          cute::make_smem_ptr(reinterpret_cast<ElementAB *>(stage_base + 128 * 128)),
          SmemLayoutB{});
      auto sSFA = cute::make_tensor(
          cute::make_smem_ptr(reinterpret_cast<ElementSF *>(stage_base + 128 * 128 + 8 * 128)),
          SmemLayoutSFA{});
      auto sSFB = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<ElementSF *>(
                                        stage_base + 128 * 128 + 8 * 128 + 512)),
                                    SmemLayoutSFB{});
      auto tCrA = DirectM128N8Mxfp8::make_fragment_A(sA);
      auto tCrB = DirectM128N8Mxfp8::make_fragment_B(sB);
      auto sSFACompact = cute::make_tensor(sSFA.data(), cute::filter_zeros(sSFA.layout()));
      auto sSFBCompact = cute::make_tensor(sSFB.data(), cute::filter_zeros(sSFB.layout()));
      auto copySfaSource_ = copySfa.get_slice(0).partition_S(sSFACompact);
      auto copySfbSource_ = copySfb.get_slice(0).partition_S(sSFBCompact);
      auto copySfaSource =
          cute::get_utccp_smem_desc_tensor<cute::SM100_UTCCP_4x32dp128bit_1cta>(copySfaSource_);
      auto copySfbSource =
          cute::get_utccp_smem_desc_tensor<cute::SM100_UTCCP_4x32dp128bit_1cta>(copySfbSource_);
      auto copySfaDestination = copySfa.get_slice(0).partition_D(tSFACompact);
      auto copySfbDestination = copySfb.get_slice(0).partition_D(tSFBCompact);

      if (cute::elect_one_sync()) {
        cute::copy(copySfa, copySfaSource, copySfaDestination);
        cute::copy(copySfb, copySfbSource, copySfbDestination);
      }
#pragma unroll
      for (int k_block = 0; k_block < 4; ++k_block) {
        cute::gemm(
            tiled_mma.with(tiled_mma.accumulate_, tSfaMma(_, _, k_block), tSfbMma(_, _, k_block)),
            tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
        tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
      }
      // The empty stage becomes visible only after the async UMMA has stopped reading it.
      cutlass::arch::umma_arrive(&barriers.ab_empty[stage]);
    };

    cute::for_each(cute::make_seq<12>{}, consume_tile);
    if (is_q) {
      cute::for_each(cute::make_seq<4>{}, [&](auto tail_index) {
        consume_tile(cute::Int<12 + decltype(tail_index)::value>{});
      });
    }
    cutlass::arch::umma_arrive(&barriers.mma_epilog);
    // Phase one protects deallocation until W4-W7 finish their TMEM loads.
    cute::arrive_barrier(barriers.tmem_lifetime);
    cute::wait_barrier(barriers.tmem_lifetime, 1);
    allocator.free(shared.tmem_base, 256);
  } else if (warp >= 4) {
    cute::arrive_barrier(barriers.tmem_lifetime);
    cute::wait_barrier(barriers.tmem_lifetime, 0);

    int const epi_tid = tid - 128;
    cute::wait_barrier(barriers.mma_epilog, 0);
    uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
    uint32_t const tmem_address = shared.tmem_base + uint32_t(epi_tid >> 5) * 0x200000u;
    cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_address, r0, r1, r2, r3, r4, r5,
                                                             r6, r7);
    float const registers[8] = {__uint_as_float(r0), __uint_as_float(r1), __uint_as_float(r2),
                                __uint_as_float(r3), __uint_as_float(r4), __uint_as_float(r5),
                                __uint_as_float(r6), __uint_as_float(r7)};
    int const spill_lane = epi_tid & 31;
    if (hs.head < kNumQHeads) {
      int const lane_group = spill_lane & ~3;
      int const owner_rank = (hs.head & 1) ? 3 : 0;
      float *destination = shared.q_mailbox + hs.split * kTokensPerTile * kHeadDim + epi_tid;
      // The destination rank and split base are invariant across all eight output tokens.  Mapping
      // them once removes fourteen redundant MAPA instructions from the Q spill path while keeping
      // the FP32x4 transaction and completion-byte contracts unchanged.
      uint32_t const remote_destination = sm100::map_shared_rank(destination, uint32_t(owner_rank));
      uint32_t const remote_barrier =
          sm100::map_shared_rank(barriers.q_mailbox, uint32_t(owner_rank));
#pragma unroll
      for (int output_token = 0; output_token < kTokensPerTile; ++output_token) {
        float const value = registers[output_token];
        F32x4 packed{__shfl_sync(0xffffffffu, value, lane_group),
                     __shfl_sync(0xffffffffu, value, lane_group + 1),
                     __shfl_sync(0xffffffffu, value, lane_group + 2),
                     __shfl_sync(0xffffffffu, value, lane_group + 3)};
        if ((spill_lane & 3) == 0) {
          uint32_t const destination_offset =
              remote_destination + uint32_t(output_token * kHeadDim * int(sizeof(float)));
          uint32_t const barrier_offset =
              remote_barrier + uint32_t(output_token * int(sizeof(uint64_t)));
          asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 "
                       "[%0], {%2, %3, %4, %5}, [%1];"
                       :
                       : "r"(destination_offset), "r"(barrier_offset), "f"(packed.x), "f"(packed.y),
                         "f"(packed.z), "f"(packed.w)
                       : "memory");
        }
      }
    } else {
      int const special_index = hs.head - kNumQHeads;
#pragma unroll
      for (int output_token = 0; output_token < kTokensPerTile; ++output_token) {
        params.workspace_special_partial
            [((special_index * kSpecialSplitK + hs.split) * kTokensPerTile + output_token) *
                 kHeadDim +
             epi_tid] = registers[output_token];
      }
      cutlass::arch::NamedBarrier(128, 2).arrive_and_wait();
      sm100::special_epoch_release_acquire<kSpecialSplitK>(params.workspace_special_epochs +
                                                               special_index * 32,
                                                           uint32_t(hs.split), uint32_t(epi_tid));
    }
    // Register dependencies protect the loaded values; the wait is required only before
    // W2 is allowed to reuse/free this CTA's TMEM allocation.
    sm100::fence_view_async_tmem_load();
    cute::arrive_barrier(barriers.tmem_lifetime);
  }
}

} // namespace cutlass::m3_decode::collective::attention_projection
