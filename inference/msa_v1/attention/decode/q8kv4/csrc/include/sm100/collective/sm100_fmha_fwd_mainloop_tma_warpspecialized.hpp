#pragma once

#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"
#include "fmha_common.hpp"
#include "sm100_fmha_correction_tma_warpspecialized.hpp"
#include "sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "sm100_fmha_kv_transform_tma_warpspecialized.hpp"
#include "sm100_fmha_load_tma_warpspecialized.hpp"
#include "sm100_fmha_mma_tma_warpspecialized.hpp"
#include "sm100_fmha_page_offsets_tma_warpspecialized.hpp"
#include "sm100_fmha_q8kv4_traits.hpp"
#include "sm100_fmha_softmax_tma_warpspecialized.hpp"
#include "sm100_fmha_storage.hpp"

namespace cutlass::fmha::collective {

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
                                            int lane_idx, int warp_group_start) {
  Sm100FmhaNamedBarrier::sync(128, kTmemDeallocBarrierId);
  int const warp_group_lane = (warp_idx - warp_group_start) * cutlass::NumThreadsPerWarp + lane_idx;
  if (warp_group_lane < cutlass::NumThreadsPerWarp) {
    cute::TMEM::Allocator1Sm allocator;
    uint32_t const tmem_base = __shfl_sync(0xffffffffu, storage.tmem_state_ptr()[0], 0, 32);
    allocator.free(tmem_base, Traits::kNumTmemCols);
  }
}

template <class PipelineState>
CUTLASS_DEVICE void advance_scheduler_consumer(PipelineState &state, bool increment_pipe) {
  if (increment_pipe) {
    ++state;
  }
}

template <class Traits, bool EnableSplitKvPath = Traits::kEnableSplitKvPath>
struct Sm100FmhaSplitPathPredicates {

  CUTLASS_DEVICE static bool use_workspace_split(Sm100FmhaFwdKernelParams<Traits> const &) {
    return false;
  }
};

template <class Traits> struct Sm100FmhaSplitPathPredicates<Traits, true> {

  CUTLASS_DEVICE static bool use_workspace_split(Sm100FmhaFwdKernelParams<Traits> const &params) {
    return params.num_kv_splits > 1 && params.workspace_o_ptr != nullptr;
  }
};

template <class Traits>
CUTLASS_DEVICE bool use_workspace_split(Sm100FmhaFwdKernelParams<Traits> const &params) {
  return Sm100FmhaSplitPathPredicates<Traits>::use_workspace_split(params);
}

template <class Traits> CUTLASS_DEVICE constexpr int split_plan_q_tokens_per_work() { return 1; }

template <class Traits>
CUTLASS_DEVICE int split_plan_q_token_end(Sm100FmhaFwdKernelParams<Traits> const &params,
                                          int batch_idx, int q_token_begin) {
  int const q_token_count = fmha_fwd_q_token_count<Traits>(params, batch_idx);
  int const q_token_end = q_token_begin + split_plan_q_tokens_per_work<Traits>();
  return q_token_end < q_token_count ? q_token_end : q_token_count;
}

template <class Traits>
CUTLASS_DEVICE void run_fwd_mainloop_device(Sm100FmhaFwdKernelParams<Traits> const &params,
                                            uint8_t *smem) {

  using Storage = SharedStorage<Traits>;
  Storage &storage = *reinterpret_cast<Storage *>(smem);

  int const thread_idx = static_cast<int>(threadIdx.x);
  int const warp_idx = __shfl_sync(0xffffffffu, thread_idx / cutlass::NumThreadsPerWarp, 0, 32);
  int const lane_idx = thread_idx % cutlass::NumThreadsPerWarp;
  int const grid_x_idx = static_cast<int>(blockIdx.x);
  int const kv_head_idx = static_cast<int>(blockIdx.y);
  int const batch_idx = static_cast<int>(blockIdx.z);
  Sm100FmhaTma::prefetch_tensormap(&params.tma.q);
  Sm100FmhaTma::prefetch_tensormap(&params.tma.k);
  Sm100FmhaTma::prefetch_tensormap(&params.tma.v);

  storage.pipelines.init_pipeline_barriers(warp_idx, lane_idx);
  cutlass::arch::fence_barrier_init();

  if (thread_idx < cutlass::NumThreadsPerWarp) {
    cute::TMEM::Allocator1Sm allocator;
    allocator.allocate(Traits::kNumTmemCols, storage.tmem_state_ptr());
    allocator.release_allocation_lock();
  }

  __syncthreads();

  {
    bool const workspace_split = use_workspace_split<Traits>(params);
    bool const precomputed_schedule = params.use_precomputed_scheduler;
    bool const use_persistent_scheduler = params.use_persistent_scheduler;
    if (workspace_split || precomputed_schedule) {
      using SchedulerAdapter = Sm100FmhaScheduler<Traits>;
      int const cta_idx = static_cast<int>(blockIdx.x);
      auto const work_range = SchedulerAdapter::precomputed_work_range(params, cta_idx);

      if (warp_idx == Traits::kPageOffsetsWarpStart) {
        configure_worker_register_budget<Traits>();
        typename Sm100FmhaPageOffsetsTmaWarpspecialized<Traits>::State state;
        CUTLASS_PRAGMA_NO_UNROLL
        for (int work_idx = work_range.begin; work_idx < work_range.end; ++work_idx) {
          auto const work_tile = SchedulerAdapter::split_work_tile(params, work_idx);
          int const q_token_end =
              split_plan_q_token_end<Traits>(params, work_tile.batch_idx, work_tile.q_token_idx);
          CUTLASS_PRAGMA_NO_UNROLL
          for (int q_token = work_tile.q_token_idx; q_token < q_token_end; ++q_token) {
            Sm100FmhaPageOffsetsTmaWarpspecialized<Traits>{}.run_tile(
                storage, params, work_tile.batch_idx, work_tile.kv_head_idx, q_token, lane_idx,
                state, work_tile.kv_tile_begin, work_tile.kv_tile_end);
          }
        }
      } else if (warp_idx == Traits::kLoadWarpStart) {
        configure_worker_register_budget<Traits>();
        typename Sm100FmhaLoadTmaWarpspecialized<Traits>::State state;
        CUTLASS_PRAGMA_NO_UNROLL
        for (int work_idx = work_range.begin; work_idx < work_range.end; ++work_idx) {
          auto const work_tile = SchedulerAdapter::split_work_tile(params, work_idx);
          int const q_token_end =
              split_plan_q_token_end<Traits>(params, work_tile.batch_idx, work_tile.q_token_idx);
          CUTLASS_PRAGMA_NO_UNROLL
          for (int q_token = work_tile.q_token_idx; q_token < q_token_end; ++q_token) {
            Sm100FmhaLoadTmaWarpspecialized<Traits>{}.run_tile(
                storage, params, work_tile.batch_idx, work_tile.kv_head_idx, q_token, lane_idx,
                static_cast<bool>(cute::elect_one_sync()), state, work_tile.kv_tile_begin,
                work_tile.kv_tile_end);
          }
        }
      } else if (warp_idx >= Traits::kSoftmaxWarpStart &&
                 warp_idx < Traits::kSoftmaxWarpStart + Traits::kNumSoftmaxWarps) {
        configure_softmax_register_budget<Traits>();
        typename Sm100FmhaSoftmaxTmaWarpspecialized<Traits>::State state;
        CUTLASS_PRAGMA_NO_UNROLL
        for (int work_idx = work_range.begin; work_idx < work_range.end; ++work_idx) {
          auto const work_tile = SchedulerAdapter::split_work_tile(params, work_idx);
          int const q_token_end =
              split_plan_q_token_end<Traits>(params, work_tile.batch_idx, work_tile.q_token_idx);
          CUTLASS_PRAGMA_NO_UNROLL
          for (int q_token = work_tile.q_token_idx; q_token < q_token_end; ++q_token) {
            Sm100FmhaSoftmaxTmaWarpspecialized<Traits>{}.run_tile(
                storage, params, work_tile.batch_idx, work_tile.kv_head_idx, q_token, lane_idx,
                warp_idx - Traits::kSoftmaxWarpStart, state, work_tile.kv_tile_begin,
                work_tile.kv_tile_end);
          }
        }
      } else if (warp_idx >= Traits::kCorrectionWarpStart &&
                 warp_idx < Traits::kCorrectionWarpStart + Traits::kNumCorrectionWarps) {
        configure_correction_register_budget<Traits>();
        typename Sm100FmhaCorrectionTmaWarpspecialized<Traits>::State state;
        CUTLASS_PRAGMA_NO_UNROLL
        for (int work_idx = work_range.begin; work_idx < work_range.end; ++work_idx) {
          auto const work_tile = SchedulerAdapter::split_work_tile(params, work_idx);
          int const q_token_end =
              split_plan_q_token_end<Traits>(params, work_tile.batch_idx, work_tile.q_token_idx);
          CUTLASS_PRAGMA_NO_UNROLL
          for (int q_token = work_tile.q_token_idx; q_token < q_token_end; ++q_token) {
            Sm100FmhaCorrectionTmaWarpspecialized<Traits>{}.run_tile(
                storage, params, work_tile.batch_idx, work_tile.kv_head_idx, q_token, lane_idx,
                warp_idx - Traits::kCorrectionWarpStart, state, work_tile.kv_tile_begin,
                work_tile.kv_tile_end, work_tile.kv_split_idx, work_tile.kv_split_count);
          }
        }
      } else if (warp_idx >= Traits::kTransformKvWarpStart &&
                 warp_idx < Traits::kTransformKvWarpStart + Traits::kNumTransformKvWarps) {
        configure_transform_register_budget<Traits>();
        typename Sm100FmhaKvTransformTmaWarpspecialized<Traits>::State state;
        CUTLASS_PRAGMA_NO_UNROLL
        for (int work_idx = work_range.begin; work_idx < work_range.end; ++work_idx) {
          auto const work_tile = SchedulerAdapter::split_work_tile(params, work_idx);
          int const q_token_end =
              split_plan_q_token_end<Traits>(params, work_tile.batch_idx, work_tile.q_token_idx);
          CUTLASS_PRAGMA_NO_UNROLL
          for (int q_token = work_tile.q_token_idx; q_token < q_token_end; ++q_token) {
            Sm100FmhaKvTransformTmaWarpspecialized<Traits>{}.run_tile(
                storage, params, work_tile.batch_idx, work_tile.kv_head_idx, q_token, lane_idx,
                warp_idx - Traits::kTransformKvWarpStart, state, work_tile.kv_tile_begin,
                work_tile.kv_tile_end);
          }
        }
      } else if (warp_idx == Traits::kMmaWarpStart) {
        configure_worker_register_budget<Traits>();
        typename Sm100FmhaMmaTmaWarpspecialized<Traits>::State state;
        CUTLASS_PRAGMA_NO_UNROLL
        for (int work_idx = work_range.begin; work_idx < work_range.end; ++work_idx) {
          auto const work_tile = SchedulerAdapter::split_work_tile(params, work_idx);
          int const q_token_end =
              split_plan_q_token_end<Traits>(params, work_tile.batch_idx, work_tile.q_token_idx);
          CUTLASS_PRAGMA_NO_UNROLL
          for (int q_token = work_tile.q_token_idx; q_token < q_token_end; ++q_token) {
            Sm100FmhaMmaTmaWarpspecialized<Traits>{}.run_tile(
                storage, params, work_tile.batch_idx, work_tile.kv_head_idx, q_token, lane_idx,
                state, work_tile.kv_tile_begin, work_tile.kv_tile_end);
          }
        }
        Sm100FmhaMmaTmaWarpspecialized<Traits>::commit_acquired_s_tail(storage, state);
      } else if (warp_idx == Traits::kPaddingWarpStart) {
        configure_worker_register_budget<Traits>();
      }
    } else if (Traits::kEnableStaticPath && !use_persistent_scheduler) {
      int const q_token_idx = grid_x_idx;
      if (warp_idx == Traits::kPageOffsetsWarpStart) {
        configure_worker_register_budget<Traits>();
        typename Sm100FmhaPageOffsetsTmaWarpspecialized<Traits>::State state;
        Sm100FmhaPageOffsetsTmaWarpspecialized<Traits>{}.run_tile(
            storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx, state);
      } else if (warp_idx == Traits::kLoadWarpStart) {
        configure_worker_register_budget<Traits>();
        typename Sm100FmhaLoadTmaWarpspecialized<Traits>::State state;
        Sm100FmhaLoadTmaWarpspecialized<Traits>{}.run_tile(
            storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx,
            static_cast<bool>(cute::elect_one_sync()), state);
      } else if (warp_idx >= Traits::kSoftmaxWarpStart &&
                 warp_idx < Traits::kSoftmaxWarpStart + Traits::kNumSoftmaxWarps) {
        configure_softmax_register_budget<Traits>();
        typename Sm100FmhaSoftmaxTmaWarpspecialized<Traits>::State state;
        Sm100FmhaSoftmaxTmaWarpspecialized<Traits>{}.run_tile(
            storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx,
            warp_idx - Traits::kSoftmaxWarpStart, state);
      } else if (warp_idx >= Traits::kCorrectionWarpStart &&
                 warp_idx < Traits::kCorrectionWarpStart + Traits::kNumCorrectionWarps) {
        configure_correction_register_budget<Traits>();
        typename Sm100FmhaCorrectionTmaWarpspecialized<Traits>::State state;
        Sm100FmhaCorrectionTmaWarpspecialized<Traits>{}.run_tile(
            storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx,
            warp_idx - Traits::kCorrectionWarpStart, state);
      } else if (warp_idx >= Traits::kTransformKvWarpStart &&
                 warp_idx < Traits::kTransformKvWarpStart + Traits::kNumTransformKvWarps) {
        configure_transform_register_budget<Traits>();
        typename Sm100FmhaKvTransformTmaWarpspecialized<Traits>::State state;
        Sm100FmhaKvTransformTmaWarpspecialized<Traits>{}.run_tile(
            storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx,
            warp_idx - Traits::kTransformKvWarpStart, state);
      } else if (warp_idx == Traits::kMmaWarpStart) {
        configure_worker_register_budget<Traits>();
        typename Sm100FmhaMmaTmaWarpspecialized<Traits>::State state;
        Sm100FmhaMmaTmaWarpspecialized<Traits>{}.run_tile(storage, params, batch_idx, kv_head_idx,
                                                          q_token_idx, lane_idx, state);
      } else if (warp_idx == Traits::kPaddingWarpStart) {
        configure_worker_register_budget<Traits>();
      }
    } else {
      using SchedulerAdapter = Sm100FmhaScheduler<Traits>;
      using SchedulerImpl = typename SchedulerAdapter::Scheduler;
      using ClusterShape = typename SchedulerAdapter::ClusterShape;
      using WorkTileInfo = typename SchedulerAdapter::WorkTileInfo;
      using WorkIdPipeline = typename SchedulerAdapter::Pipeline;
      using WorkIdPipelineState = typename SchedulerAdapter::PipelineState;
      using ThrottlePipeline = typename SchedulerAdapter::ThrottlePipeline;
      using ThrottlePipelineState = typename SchedulerAdapter::ThrottlePipelineState;

      typename WorkIdPipeline::Params work_id_pipeline_params;
      work_id_pipeline_params.role = warp_idx == Traits::kSchedulerWarpStart
                                         ? WorkIdPipeline::ThreadCategory::ProducerConsumer
                                         : WorkIdPipeline::ThreadCategory::Consumer;
      work_id_pipeline_params.initializing_warp = Traits::kSchedulerWarpStart;
      work_id_pipeline_params.producer_arv_count = 1;
      work_id_pipeline_params.consumer_arv_count = Traits::kNumThreads;
      work_id_pipeline_params.producer_blockid = 0;
      work_id_pipeline_params.transaction_bytes = sizeof(typename SchedulerImpl::CLCResponse);
      WorkIdPipeline work_id_pipeline(storage.scheduler_storage.pipeline(), work_id_pipeline_params,
                                      ClusterShape{});

      typename ThrottlePipeline::Params throttle_pipeline_params;
      if (warp_idx == Traits::kLoadWarpStart) {
        throttle_pipeline_params.role = ThrottlePipeline::ThreadCategory::Producer;
      } else if (warp_idx == Traits::kSchedulerWarpStart) {
        throttle_pipeline_params.role = ThrottlePipeline::ThreadCategory::Consumer;
      }
      throttle_pipeline_params.producer_arv_count = cutlass::NumThreadsPerWarp;
      throttle_pipeline_params.consumer_arv_count = cutlass::NumThreadsPerWarp;
      throttle_pipeline_params.dst_blockid = 0;
      throttle_pipeline_params.initializing_warp = Traits::kSchedulerWarpStart;
      ThrottlePipeline throttle_pipeline(storage.scheduler_storage.throttle_pipeline(),
                                         throttle_pipeline_params);

      __syncthreads();

      SchedulerImpl scheduler(storage.scheduler_storage.data(), params.scheduler, dim3(0, 0, 0));
      WorkTileInfo work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});

      if (warp_idx == Traits::kPageOffsetsWarpStart) {
        configure_worker_register_budget<Traits>();
        typename Sm100FmhaPageOffsetsTmaWarpspecialized<Traits>::State state;
        WorkIdPipelineState work_id_consumer_state;
        do {
          auto work_tile =
              SchedulerAdapter::work_tile_from_info(work_tile_info, params.q_tokens_per_batch);
          Sm100FmhaPageOffsetsTmaWarpspecialized<Traits>{}.run_tile(
              storage, params, work_tile.batch_idx, work_tile.kv_head_idx, work_tile.q_token_idx,
              lane_idx, state);
          auto next_work =
              scheduler.fetch_next_work(work_tile_info, work_id_pipeline, work_id_consumer_state);
          work_tile_info = cute::get<0>(next_work);
          advance_scheduler_consumer(work_id_consumer_state, cute::get<1>(next_work));
        } while (work_tile_info.is_valid_tile);
      } else if (warp_idx == Traits::kLoadWarpStart) {
        configure_worker_register_budget<Traits>();
        typename Sm100FmhaLoadTmaWarpspecialized<Traits>::State state;
        WorkIdPipelineState work_id_consumer_state;
        ThrottlePipelineState throttle_producer_state =
            cutlass::make_producer_start_state<ThrottlePipeline>();
        do {
          throttle_pipeline.producer_acquire(throttle_producer_state);
          throttle_pipeline.producer_commit(throttle_producer_state);
          ++throttle_producer_state;
          auto work_tile =
              SchedulerAdapter::work_tile_from_info(work_tile_info, params.q_tokens_per_batch);
          Sm100FmhaLoadTmaWarpspecialized<Traits>{}.run_tile(
              storage, params, work_tile.batch_idx, work_tile.kv_head_idx, work_tile.q_token_idx,
              lane_idx, static_cast<bool>(cute::elect_one_sync()), state);
          auto next_work =
              scheduler.fetch_next_work(work_tile_info, work_id_pipeline, work_id_consumer_state);
          work_tile_info = cute::get<0>(next_work);
          advance_scheduler_consumer(work_id_consumer_state, cute::get<1>(next_work));
        } while (work_tile_info.is_valid_tile);
      } else if (warp_idx >= Traits::kSoftmaxWarpStart &&
                 warp_idx < Traits::kSoftmaxWarpStart + Traits::kNumSoftmaxWarps) {
        configure_softmax_register_budget<Traits>();
        typename Sm100FmhaSoftmaxTmaWarpspecialized<Traits>::State state;
        WorkIdPipelineState work_id_consumer_state;
        do {
          auto work_tile =
              SchedulerAdapter::work_tile_from_info(work_tile_info, params.q_tokens_per_batch);
          Sm100FmhaSoftmaxTmaWarpspecialized<Traits>{}.run_tile(
              storage, params, work_tile.batch_idx, work_tile.kv_head_idx, work_tile.q_token_idx,
              lane_idx, warp_idx - Traits::kSoftmaxWarpStart, state);
          auto next_work =
              scheduler.fetch_next_work(work_tile_info, work_id_pipeline, work_id_consumer_state);
          work_tile_info = cute::get<0>(next_work);
          advance_scheduler_consumer(work_id_consumer_state, cute::get<1>(next_work));
        } while (work_tile_info.is_valid_tile);
      } else if (warp_idx >= Traits::kCorrectionWarpStart &&
                 warp_idx < Traits::kCorrectionWarpStart + Traits::kNumCorrectionWarps) {
        configure_correction_register_budget<Traits>();
        typename Sm100FmhaCorrectionTmaWarpspecialized<Traits>::State state;
        WorkIdPipelineState work_id_consumer_state;
        do {
          auto work_tile =
              SchedulerAdapter::work_tile_from_info(work_tile_info, params.q_tokens_per_batch);
          Sm100FmhaCorrectionTmaWarpspecialized<Traits>{}.run_tile(
              storage, params, work_tile.batch_idx, work_tile.kv_head_idx, work_tile.q_token_idx,
              lane_idx, warp_idx - Traits::kCorrectionWarpStart, state);
          auto next_work =
              scheduler.fetch_next_work(work_tile_info, work_id_pipeline, work_id_consumer_state);
          work_tile_info = cute::get<0>(next_work);
          advance_scheduler_consumer(work_id_consumer_state, cute::get<1>(next_work));
        } while (work_tile_info.is_valid_tile);
      } else if (warp_idx >= Traits::kTransformKvWarpStart &&
                 warp_idx < Traits::kTransformKvWarpStart + Traits::kNumTransformKvWarps) {
        configure_transform_register_budget<Traits>();
        typename Sm100FmhaKvTransformTmaWarpspecialized<Traits>::State state;
        WorkIdPipelineState work_id_consumer_state;
        do {
          auto work_tile =
              SchedulerAdapter::work_tile_from_info(work_tile_info, params.q_tokens_per_batch);
          Sm100FmhaKvTransformTmaWarpspecialized<Traits>{}.run_tile(
              storage, params, work_tile.batch_idx, work_tile.kv_head_idx, work_tile.q_token_idx,
              lane_idx, warp_idx - Traits::kTransformKvWarpStart, state);
          auto next_work =
              scheduler.fetch_next_work(work_tile_info, work_id_pipeline, work_id_consumer_state);
          work_tile_info = cute::get<0>(next_work);
          advance_scheduler_consumer(work_id_consumer_state, cute::get<1>(next_work));
        } while (work_tile_info.is_valid_tile);
      } else if (warp_idx == Traits::kMmaWarpStart) {
        configure_worker_register_budget<Traits>();
        typename Sm100FmhaMmaTmaWarpspecialized<Traits>::State state;
        WorkIdPipelineState work_id_consumer_state;
        do {
          auto work_tile =
              SchedulerAdapter::work_tile_from_info(work_tile_info, params.q_tokens_per_batch);
          Sm100FmhaMmaTmaWarpspecialized<Traits>{}.run_tile(storage, params, work_tile.batch_idx,
                                                            work_tile.kv_head_idx,
                                                            work_tile.q_token_idx, lane_idx, state);
          auto next_work =
              scheduler.fetch_next_work(work_tile_info, work_id_pipeline, work_id_consumer_state);
          work_tile_info = cute::get<0>(next_work);
          advance_scheduler_consumer(work_id_consumer_state, cute::get<1>(next_work));
        } while (work_tile_info.is_valid_tile);
        Sm100FmhaMmaTmaWarpspecialized<Traits>::commit_acquired_s_tail(storage, state);
      } else if (warp_idx == Traits::kSchedulerWarpStart) {
        configure_worker_register_budget<Traits>();
        WorkIdPipelineState work_id_consumer_state;
        WorkIdPipelineState work_id_producer_state =
            cutlass::make_producer_start_state<WorkIdPipeline>();
        ThrottlePipelineState throttle_consumer_state;
        bool requires_work_id_query = true;
        do {
          if (requires_work_id_query) {
            throttle_pipeline.consumer_wait(throttle_consumer_state);
            throttle_pipeline.consumer_release(throttle_consumer_state);
            ++throttle_consumer_state;
            work_id_producer_state =
                scheduler.advance_to_next_work(work_id_pipeline, work_id_producer_state);
          }
          auto next_work =
              scheduler.fetch_next_work(work_tile_info, work_id_pipeline, work_id_consumer_state);
          work_tile_info = cute::get<0>(next_work);
          requires_work_id_query = cute::get<1>(next_work);
          advance_scheduler_consumer(work_id_consumer_state, requires_work_id_query);
        } while (work_tile_info.is_valid_tile);
        work_id_pipeline.producer_tail(work_id_producer_state);
      }
    }

    // Release TMEM only after every warp has completed its final work tile.
    __syncthreads();
    constexpr int kTmemDeallocWarpStart = Traits::kSoftmaxWarpStart;
    if (warp_idx >= kTmemDeallocWarpStart &&
        warp_idx < kTmemDeallocWarpStart + Traits::kNumSoftmaxWarps) {
      release_tmem_allocation<Traits>(storage, warp_idx, lane_idx, kTmemDeallocWarpStart);
    }
    __syncthreads();
    if (thread_idx == 0) {
      cudaTriggerProgrammaticLaunchCompletion();
    }
  }
}

template <class Traits> struct Sm100FmhaFwdQ8Kv4MainloopTmaWarpspecialized {
  using FmhaTraits = Traits;
  using Storage = SharedStorage<Traits>;
  using TensorStorage = Storage;
  using TileShape = cute::Shape<cute::Int<Traits::kTileQ>, cute::Int<Traits::kTileKv>,
                                cute::Int<Traits::kHeadDim>>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  static constexpr bool IsSplitKV = Traits::kEnableSplitKvPath;
  static constexpr bool IsQ8KV4 = true;

  struct Arguments {
    typename Sm100FmhaLoadTmaWarpspecialized<Traits>::Arguments load;
  };

  struct Params {
    Sm100FmhaFwdKernelParams<Traits> kernel;
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const &, Arguments const &args, void *,
                                        int max_active_ctas) {
    return Params{Sm100FmhaLoadTmaWarpspecialized<Traits>::to_underlying_arguments(
        args.load, max_active_ctas)};
  }

  template <class... Args> CUTLASS_DEVICE void page_offsets(Args &&...args) const {
    Sm100FmhaPageOffsetsTmaWarpspecialized<Traits>{}(static_cast<Args &&>(args)...);
  }

  template <class... Args> CUTLASS_DEVICE void load(Args &&...args) const {
    Sm100FmhaLoadTmaWarpspecialized<Traits>{}(static_cast<Args &&>(args)...);
  }

  template <class... Args> CUTLASS_DEVICE void transform_kv(Args &&...args) const {
    Sm100FmhaKvTransformTmaWarpspecialized<Traits>{}.run_schedule(static_cast<Args &&>(args)...);
  }

  template <class... Args> CUTLASS_DEVICE void mma(Args &&...args) const {
    Sm100FmhaMmaTmaWarpspecialized<Traits>{}(static_cast<Args &&>(args)...);
  }

  template <class... Args> CUTLASS_DEVICE void softmax_correction(Args &&...args) const {
    Sm100FmhaSoftmaxTmaWarpspecialized<Traits>{}(static_cast<Args &&>(args)...);
  }

  using DeviceParams = Sm100FmhaFwdKernelParams<Traits>;

  CUTLASS_DEVICE static void run_device(DeviceParams const &params, uint8_t *smem) {
    run_fwd_mainloop_device<Traits>(params, smem);
  }

  template <class... Args> CUTLASS_DEVICE void epilogue(Args &&...args) const {
    Sm100FmhaFwdEpilogueTmaWarpspecialized<Traits>{}(static_cast<Args &&>(args)...);
  }
};

template <class ElementOrTraits, class ElementQK = void, class ElementPV = void,
          class TileShapeQK = void, class TileShapePV = void, class StrideQ = void,
          class StrideK = void, class StrideV = void, class Mask = void, class ThreadShape = void,
          bool IsSplitKV_ = false, int KVPageSize = -1,
          SparseAttnMode kSparseAttnMode = SparseAttnMode::Off, bool IsQ8KV4_ = true,
          int SparseTopK = 16, int FixedQTokensPerBatch = 0>
struct Sm100FmhaFwdMainloopTmaWarpspecialized
    : Sm100FmhaFwdQ8Kv4MainloopTmaWarpspecialized<typename Sm100FmhaQ8Kv4TraitSelector<
          IsSplitKV_, kSparseAttnMode, IsQ8KV4_, SparseTopK, FixedQTokensPerBatch,
          pack_factor_of<Mask>::value>::type> {
  static_assert(IsQ8KV4_, "fwd_decode currently provides only the q8kv4 mainloop specialization.");

  using Q8Kv4Traits =
      typename Sm100FmhaQ8Kv4TraitSelector<IsSplitKV_, kSparseAttnMode, IsQ8KV4_, SparseTopK,
                                           FixedQTokensPerBatch, pack_factor_of<Mask>::value>::type;
  using Base = Sm100FmhaFwdQ8Kv4MainloopTmaWarpspecialized<Q8Kv4Traits>;
  using FmhaTraits = typename Base::FmhaTraits;

  using Element = ElementOrTraits;
  using ElementAccumulatorQK = ElementQK;
  using ElementAccumulatorPV = ElementPV;
  using TileShapeQKType = TileShapeQK;
  using TileShapePVType = TileShapePV;
  using StrideQType = StrideQ;
  using StrideKType = StrideK;
  using StrideVType = StrideV;
  using MaskType = Mask;
  using ThreadShapeType = ThreadShape;
};

template <class Traits>
struct Sm100FmhaFwdMainloopTmaWarpspecialized<Traits, void, void, void, void, void, void, void,
                                              void, void, false, -1, SparseAttnMode::Off, true, 16,
                                              0>
    : Sm100FmhaFwdQ8Kv4MainloopTmaWarpspecialized<Traits> {};

} // namespace cutlass::fmha::collective
