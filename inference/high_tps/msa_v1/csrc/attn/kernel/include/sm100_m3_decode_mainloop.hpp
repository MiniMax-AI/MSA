#pragma once

#include "attn/kernel/include/sm100_m3_decode_attention_correction.hpp"
#include "attn/kernel/include/sm100_m3_decode_attention_kv_transform.hpp"
#include "attn/kernel/include/sm100_m3_decode_attention_load.hpp"
#include "attn/kernel/include/sm100_m3_decode_attention_mma.hpp"
#include "attn/kernel/include/sm100_m3_decode_attention_page_offsets.hpp"
#include "attn/kernel/include/sm100_m3_decode_attention_projection.hpp"
#include "attn/kernel/include/sm100_m3_decode_attention_softmax.hpp"
#include "attn/kernel/include/sm100_m3_decode_indexer.hpp"
#include "attn/kernel/include/sm100_m3_decode_output_projection.hpp"
#include "attn/kernel/include/sm100_m3_decode_problem.hpp"
#include "common/kernel/include/sm100_m3_decode_execution.hpp"
#include "common/kernel/include/sm100_m3_decode_scheduler.hpp"
#include "common/kernel/include/sm100_m3_decode_storage.hpp"
#include "common/kernel/include/sm100_m3_decode_tp4_allreduce.hpp"
#include "common/kernel/include/sm100_m3_decode_traits.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"
namespace cutlass::m3_decode::collective {

static constexpr int kTmemDeallocBarrierId = 12;

template <class Traits> CUTLASS_DEVICE void configure_softmax_register_budget() {
  cutlass::arch::warpgroup_reg_alloc<Traits::kSoftmaxMaxRegisters>();
}

template <class Traits> CUTLASS_DEVICE void configure_correction_register_budget() {
  cutlass::arch::warpgroup_reg_dealloc<Traits::kCorrectionMaxRegisters>();
}

template <class Traits> CUTLASS_DEVICE void configure_worker_register_budget() {
  static_assert(Traits::kMmaMaxRegisters == Traits::kPageOffsetsMaxRegisters &&
                    Traits::kMmaMaxRegisters == Traits::kPaddingMaxRegisters &&
                    Traits::kMmaMaxRegisters == Traits::kLoadMaxRegisters,
                "worker-role warpgroup must use one uniform register budget.");
  cutlass::arch::warpgroup_reg_dealloc<Traits::kMmaMaxRegisters>();
}

template <class Traits> CUTLASS_DEVICE void configure_transform_register_budget() {
  cutlass::arch::warpgroup_reg_alloc<Traits::kTransformKvMaxRegisters>();
}

template <class Traits>
CUTLASS_DEVICE void release_tmem_allocation(SharedStorage<Traits> &storage, int warp_idx,
                                            int lane_idx) {
  Sm100M3DecodeNamedBarrier::sync(128, kTmemDeallocBarrierId);
  int const warp_group_lane =
      (warp_idx - Traits::kCorrectionWarpStart) * cutlass::NumThreadsPerWarp + lane_idx;
  if (warp_group_lane < cutlass::NumThreadsPerWarp) {
    cute::TMEM::Allocator1Sm allocator;
    uint32_t const tmem_base = __shfl_sync(0xffffffffu, storage.tmem_state_ptr()[0], 0, 32);
    allocator.free(tmem_base, Traits::kNumTmemCols);
  }
}

template <class Traits>
CUTLASS_DEVICE Sm100M3DecodeKvTileRange cluster_split_tile_range(int full_tiles, int split_idx) {
  constexpr int kNumKvSplits = Traits::kNumKvSplits;
  int const tiles_per_split = (full_tiles + kNumKvSplits - 1) / kNumKvSplits;
  int const begin = split_idx * tiles_per_split;
  int const end = begin + tiles_per_split;
  return make_kv_tile_range(full_tiles, begin, end);
}

template <class Traits>
CUTLASS_DEVICE void init_cluster_reduction_barrier(SharedStorage<Traits> &storage, int thread_idx) {
  if (thread_idx == 0) {
    cutlass::arch::ClusterTransactionBarrier::init(storage.cluster_reduction_barrier_ptr(), 1);
  }
}

template <class Traits>
CUTLASS_DEVICE void expect_cluster_reduction_transactions(SharedStorage<Traits> &storage,
                                                          int thread_idx) {
  if (thread_idx == 0) {
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
        storage.cluster_reduction_barrier_ptr(), Traits::kClusterReductionTransactionBytes);
  }
}

template <class Traits>
CUTLASS_DEVICE void configure_output_projection_register_budget(int warp_idx) {
  if (warp_idx < 4) {
    configure_softmax_register_budget<Traits>();
  } else if (warp_idx < 8) {
    configure_correction_register_budget<Traits>();
  } else if (warp_idx < 12) {
    configure_worker_register_budget<Traits>();
  } else {
    configure_transform_register_budget<Traits>();
  }
}

template <class Traits, bool ConfigureRegisterBudget = true>
CUTLASS_DEVICE void
run_output_projection_task_device(Sm100M3DecodeKernelParams<Traits> const &params,
                                  SharedStorage<Traits> &storage, int thread_idx, int warp_idx,
                                  int lane_idx, int m_tile, bool tmem_preallocated = false) {
  using OutputProjection = Sm100M3DecodeOutputProjection<Traits>;
  using Tp4Tail = Sm100M3DecodeTp4AllReduce<Traits>;

  if constexpr (ConfigureRegisterBudget) {
    configure_output_projection_register_budget<Traits>(warp_idx);
  }
  if (!tmem_preallocated && thread_idx < cutlass::NumThreadsPerWarp) {
    cute::TMEM::Allocator1Sm allocator;
    allocator.allocate(Traits::kNumTmemCols, storage.tmem_state_ptr());
  }
  if (warp_idx == OutputProjection::kWeightLoadWarp) {
    OutputProjection::prefetch_weight_task(storage, params, lane_idx, 0, m_tile);
  }
  OutputProjection::run_task(storage, params, thread_idx, warp_idx, m_tile);

  if (warp_idx >= Traits::kCorrectionWarpStart &&
      warp_idx < Traits::kCorrectionWarpStart + Traits::kNumCorrectionWarps) {
    Sm100M3DecodeNamedBarrier::sync(128, kTmemDeallocBarrierId);
    int const warp_group_lane =
        (warp_idx - Traits::kCorrectionWarpStart) * cutlass::NumThreadsPerWarp + lane_idx;
    if (warp_group_lane == 0) {
      if constexpr (Traits::kKernelPath == Sm100M3DecodeKernelPath::kMegaAtten) {
        Sm100M3DecodeTp4Mailbox::finish_publish(params, m_tile);
      } else {
        Tp4Tail::publish_output_projection_completion(params);
      }
    }
    if (warp_group_lane < cutlass::NumThreadsPerWarp) {
      cute::TMEM::Allocator1Sm allocator;
      uint32_t const tmem_base = __shfl_sync(0xffffffffu, storage.tmem_state_ptr()[0], 0, 32);
      allocator.free(tmem_base, Traits::kNumTmemCols);
    }
  }
  __syncthreads();
}

template <class Traits, bool RunMoeTail, class Fusion>
CUTLASS_DEVICE void run_attention_task_device(Sm100M3DecodeKernelParams<Traits> const &params,
                                              SharedStorage<Traits> &storage, uint8_t *smem,
                                              int thread_idx, int warp_idx, int lane_idx,
                                              int batch_idx, int kv_head_idx, int q_token_idx,
                                              int split_idx, bool tmem_preallocated = false) {
  using OutputProjection = Sm100M3DecodeOutputProjection<Traits>;
  using Scheduler = Sm100M3DecodeScheduler<Traits>;
  using Tp4Tail = Sm100M3DecodeTp4AllReduce<Traits>;

  if (!tmem_preallocated) {
    Sm100M3DecodeTma::prefetch_tensormap(&params.tma.q);
    Sm100M3DecodeTma::prefetch_tensormap(&params.tma.k);
    Sm100M3DecodeTma::prefetch_tensormap(&params.tma.v);
  }

  if (!tmem_preallocated) {
    storage.attention_pipelines().init_pipeline_barriers(warp_idx, lane_idx);
    init_cluster_reduction_barrier<Traits>(storage, thread_idx);
    cutlass::arch::fence_barrier_init();
  }

  if (!tmem_preallocated && thread_idx < cutlass::NumThreadsPerWarp) {
    cute::TMEM::Allocator1Sm allocator;
    allocator.allocate(Traits::kNumTmemCols, storage.tmem_state_ptr());
  }
  if (thread_idx == 0) {
    int const visible_kv_len = m3_decode_visible_kv_length<Traits>(params, batch_idx, q_token_idx);
    int const visible_pages = (visible_kv_len + Traits::kPageSize - 1) / Traits::kPageSize;
    int const selected_pages =
        visible_pages < Traits::kSparseTopK ? visible_pages : Traits::kSparseTopK;
    int const last_selected_page = storage.indexer_topk[selected_pages - 1];
    int const full_selected_kv_len = selected_pages * Traits::kPageSize;
    int const tail_correction =
        (Traits::kPageSize - (visible_kv_len % Traits::kPageSize)) % Traits::kPageSize;
    int const needs_tail_mask = int(last_selected_page == visible_pages - 1);
    Sm100M3DecodeKvMetadata const metadata{full_selected_kv_len - needs_tail_mask * tail_correction,
                                           selected_pages};
    storage.smem_kv_metadata.kv_length = metadata.kv_length;
    storage.smem_kv_metadata.tile_count = metadata.tile_count;
  }
  __syncthreads();
  if (!tmem_preallocated) {
    expect_cluster_reduction_transactions<Traits>(storage, thread_idx);
    cute::cluster_sync();
  }

  int const kv_length = storage.smem_kv_metadata.kv_length;
  int const full_tiles = storage.smem_kv_metadata.tile_count;
  Sm100M3DecodeKvTileRange const tile_range =
      cluster_split_tile_range<Traits>(full_tiles, split_idx);
  if (warp_idx == Traits::kPageOffsetsWarpStart) {
    if constexpr (RunMoeTail) {
      configure_worker_register_budget<Traits>();
    }
    typename Sm100M3DecodeAttentionPageOffsets<Traits>::State state;
    Sm100M3DecodeAttentionPageOffsets<Traits>{}.run_tile(storage, params, batch_idx, kv_head_idx,
                                                         q_token_idx, lane_idx, state, full_tiles,
                                                         tile_range.begin, tile_range.end);
  } else if (warp_idx == Traits::kLoadWarpStart) {
    if constexpr (RunMoeTail) {
      configure_worker_register_budget<Traits>();
    }
    typename Sm100M3DecodeAttentionLoad<Traits>::State state;
    Sm100M3DecodeAttentionLoad<Traits>{}.run_tile(
        storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx,
        static_cast<bool>(cute::elect_one_sync()), state, full_tiles, kv_length, tile_range.begin,
        tile_range.end);
  } else if (warp_idx >= Traits::kSoftmaxWarpStart &&
             warp_idx < Traits::kSoftmaxWarpStart + Traits::kNumSoftmaxWarps) {
    if constexpr (RunMoeTail) {
      configure_softmax_register_budget<Traits>();
    }
    typename Sm100M3DecodeAttentionSoftmax<Traits>::State state;
    Sm100M3DecodeAttentionSoftmax<Traits>{}.run_tile(
        storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx,
        warp_idx - Traits::kSoftmaxWarpStart, state, kv_length, full_tiles, tile_range.begin,
        tile_range.end);
  } else if (warp_idx >= Traits::kCorrectionWarpStart &&
             warp_idx < Traits::kCorrectionWarpStart + Traits::kNumCorrectionWarps) {
    if constexpr (RunMoeTail) {
      configure_correction_register_budget<Traits>();
    }
    typename Sm100M3DecodeAttentionCorrection<Traits>::State state;
    Sm100M3DecodeAttentionCorrection<Traits>{}.run_tile(
        storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx,
        warp_idx - Traits::kCorrectionWarpStart, state, full_tiles, tile_range.begin,
        tile_range.end, split_idx);
    OutputProjection::publish_activation_stages(params, split_idx,
                                                warp_idx - Traits::kCorrectionWarpStart);
  } else if (warp_idx >= Traits::kTransformKvWarpStart &&
             warp_idx < Traits::kTransformKvWarpStart + Traits::kNumTransformKvWarps) {
    if constexpr (RunMoeTail) {
      configure_transform_register_budget<Traits>();
    }
    typename Sm100M3DecodeAttentionKvTransform<Traits>::State state;
    Sm100M3DecodeAttentionKvTransform<Traits>{}.run_tile(
        storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx,
        warp_idx - Traits::kTransformKvWarpStart, state, full_tiles, tile_range.begin,
        tile_range.end);
  } else if (warp_idx == Traits::kMmaWarpStart) {
    if constexpr (RunMoeTail) {
      configure_worker_register_budget<Traits>();
    }
    typename Sm100M3DecodeAttentionMma<Traits>::State state;
    Sm100M3DecodeAttentionMma<Traits>{}.run_tile(storage, params, batch_idx, kv_head_idx,
                                                 q_token_idx, lane_idx, state, full_tiles,
                                                 tile_range.begin, tile_range.end);
  } else if (warp_idx == Traits::kPaddingWarpStart) {
    if constexpr (RunMoeTail) {
      configure_worker_register_budget<Traits>();
    }
  }

  if (warp_idx >= Traits::kCorrectionWarpStart &&
      warp_idx < Traits::kCorrectionWarpStart + Traits::kNumCorrectionWarps) {
    release_tmem_allocation<Traits>(storage, warp_idx, lane_idx);
  }

  __syncthreads();
  Fusion::finish_attention(params, storage, smem, thread_idx, q_token_idx);
}

template <class Traits> struct AttentionOnlyTasks {
  CUTLASS_DEVICE static void finish_attention(Sm100M3DecodeKernelParams<Traits> const &params,
                                              SharedStorage<Traits> &storage, uint8_t *smem,
                                              int thread_idx, int q_token_idx) {
    Sm100M3DecodeTp4Mailbox::consume(params, smem, q_token_idx, thread_idx,
                                     static_cast<int>(cute::block_rank_in_cluster()));
  }
  CUTLASS_DEVICE static void run(Sm100M3DecodeKernelParams<Traits> const &params,
                                 SharedStorage<Traits> &storage, uint8_t *smem, int thread_idx,
                                 int warp_idx, int lane_idx, int cluster_size, int cluster_rank,
                                 int cluster_ordinal, uint32_t) {}
};
template <class Traits, bool RunMoeTail, class Fusion> struct AttentionInitialTask {
  CUTLASS_DEVICE static uint32_t run(Sm100M3DecodeKernelParams<Traits> const &params,
                                     SharedStorage<Traits> &storage, uint8_t *smem, int thread_idx,
                                     int warp_idx, int lane_idx, int cluster_size, int cluster_rank,
                                     int cluster_ordinal) {
    using Scheduler = Sm100M3DecodeScheduler<Traits>;
    uint32_t shared_epoch = 0;

    bool const is_shared_cluster = cluster_size == Scheduler::kFallbackClusterSize &&
                                   cluster_ordinal >= Scheduler::kSharedFallbackClusterStart &&
                                   cluster_ordinal < Scheduler::kRoutedFallbackClusterStart;
    int const shared_worker = (cluster_ordinal - Scheduler::kSharedFallbackClusterStart) *
                                  Scheduler::kFallbackClusterSize +
                              cluster_rank;
    bool const is_shared_primary = is_shared_cluster && shared_worker < Scheduler::kSharedCtas;

    bool wait_attention_projection_after_indexer = false;
    constexpr int kAttentionProjectionPreferredClusterStart =
        Scheduler::kPreferredClusters -
        cutlass::m3_decode::collective::attention_projection::kClusterCount - 1;
    constexpr int kAttentionProjectionPreferredClusterEnd =
        kAttentionProjectionPreferredClusterStart +
        cutlass::m3_decode::collective::attention_projection::kClusterCount;
    bool const is_attention_projection_producer =
        cluster_size == Scheduler::kPreferredClusterSize &&
        cluster_ordinal >= kAttentionProjectionPreferredClusterStart &&
        cluster_ordinal < kAttentionProjectionPreferredClusterEnd;
    if (__builtin_expect(is_attention_projection_producer, false)) {
      int const attention_projection_cluster_id =
          kAttentionProjectionPreferredClusterEnd - 1 - cluster_ordinal;
      using AttentionProjectionBarrierStorage =
          cutlass::m3_decode::collective::attention_projection::BarrierStorage;
      static_assert(sizeof(AttentionProjectionBarrierStorage) ==
                    SharedStorage<Traits>::kAttentionProjectionBarrierCount * sizeof(uint64_t));
      auto &attention_projection_barriers = *reinterpret_cast<AttentionProjectionBarrierStorage *>(
          storage.phase_barrier_ptr(SharedStorage<Traits>::kAttentionProjectionBarrierBase));
      bool const is_attention_projection_output_owner =
          cutlass::m3_decode::collective::attention_projection::run(
              params.attention_projection, smem, attention_projection_barriers, thread_idx,
              attention_projection_cluster_id, cluster_rank,
              Scheduler::attention_projection_qi_ready_address(params),
              Scheduler::attention_projection_ki_ready_address(params),
              Scheduler::current_generation_prefix(params));
      bool const is_attention_projection_leader =
          attention_projection_cluster_id == 0 && cluster_rank == 0;
      Scheduler::publish_attention_projection_ready(params, thread_idx,
                                                    is_attention_projection_leader);
      if (!is_attention_projection_leader && !is_attention_projection_output_owner) {
        Scheduler::wait_attention_projection_ready(params, thread_idx);
      }
    } else {
      Scheduler::wait_attention_projection_indexer_ready(params, thread_idx);
      wait_attention_projection_after_indexer = true;
    }

    if constexpr (RunMoeTail) {
      shared_epoch = Fusion::epoch(params);
    }

    bool const attention_cluster = cluster_size == Scheduler::kPreferredClusterSize &&
                                   cluster_ordinal < Scheduler::kAttentionTasks;
    bool const retain_tmem = cluster_size == Scheduler::kPreferredClusterSize &&
                             cluster_ordinal < Scheduler::kOutputProjectionPreferredClusterEnd;
    Sm100M3DecodeIndexer<Traits>::run(params, storage, smem, thread_idx, warp_idx, lane_idx,
                                      cluster_size, cluster_rank, attention_cluster,
                                      cluster_ordinal, retain_tmem);
    constexpr bool kOverlapSharedWithLongIndexer = Traits::kIndexerTopKItemsPerThread >= 16;
    bool const indexer_joined_projection = cluster_size == Scheduler::kPreferredClusterSize &&
                                           cluster_ordinal < Scheduler::kAttentionTasks;
    if (wait_attention_projection_after_indexer && !indexer_joined_projection &&
        !kOverlapSharedWithLongIndexer) {
      Scheduler::wait_attention_projection_ready(params, thread_idx);
    }
    if (cluster_size == Scheduler::kPreferredClusterSize) {
      if (cluster_ordinal < Scheduler::kAttentionTasks) {
        run_attention_task_device<Traits, RunMoeTail, Fusion>(
            params, storage, smem, thread_idx, warp_idx, lane_idx, static_cast<int>(blockIdx.z),
            static_cast<int>(blockIdx.y), cluster_ordinal, cluster_rank, true);
      } else if (cluster_ordinal < Scheduler::kOutputProjectionPreferredClusterEnd) {
        run_output_projection_task_device<Traits, RunMoeTail>(
            params, storage, thread_idx, warp_idx, lane_idx,
            (cluster_ordinal - Scheduler::kAttentionTasks) * Scheduler::kPreferredClusterSize +
                cluster_rank,
            true);
      } else if constexpr (RunMoeTail) {
        configure_output_projection_register_budget<Traits>(warp_idx);
      }
    } else if constexpr (RunMoeTail) {
      configure_output_projection_register_budget<Traits>(warp_idx);
      if (is_shared_primary) {
        Fusion::run_shared_fc1(storage, params, thread_idx, warp_idx, shared_epoch, shared_worker);
      }
    }

    return shared_epoch;
  }
};
template <class Traits> struct Sm100M3DecodeMegaAttenMainloop {
  using DeviceParams = Sm100M3DecodeKernelParams<Traits>;
  CUTLASS_DEVICE static void run_device(DeviceParams const &params, uint8_t *smem) {
    run_decode_schedule<Traits, AttentionInitialTask<Traits, false, AttentionOnlyTasks<Traits>>,
                        AttentionOnlyTasks<Traits>>(params, smem);
  }
};

} // namespace cutlass::m3_decode::collective
