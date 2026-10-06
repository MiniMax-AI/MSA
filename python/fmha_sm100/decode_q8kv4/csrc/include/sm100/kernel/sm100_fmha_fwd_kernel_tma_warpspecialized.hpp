// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <mutex>

#include <cuda_runtime.h>

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/config.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/kernel_hardware_info.h"
#include "decode_attention_params.hpp"
#include "fmha_common.hpp"
#include "sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "sm100_fmha_pipeline.hpp"
#include "sm100_fmha_q8kv4_traits.hpp"
#include "sm100_fmha_storage.hpp"

namespace cutlass::fmha::kernel {

using namespace cutlass::fmha::collective;

template <class Traits>
__global__ __launch_bounds__(Traits::kNumThreads, 1) void sm100_fmha_fwd_device_kernel(
    CUTE_GRID_CONSTANT Sm100FmhaFwdKernelParams<Traits> const params) {
  extern __shared__ uint8_t smem[];
  Sm100FmhaFwdMainloopTmaWarpspecialized<Traits>::run_device(params, smem);
}

template <class Traits> struct Sm100FmhaFwdQ8Kv4KernelTmaWarpspecialized {
  using Storage = SharedStorage<Traits>;
  using SharedStorage = Storage;
  using Arguments = FMHACutlassSM100Params;
  using Params = Sm100FmhaFwdKernelParams<Traits>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using ArchTag = cutlass::arch::Sm100;

  static constexpr int MaxThreadsPerBlock = Traits::kNumThreads;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = kSharedStorageBytes<Traits>;

  struct ActiveCtaCapacity {
    int device_id = -1;
    int cluster_size = 1;
    int blocks_per_sm = 0;
    int sm_count = 0;
    int max_active_ctas = 0;
  };

  static dim3 get_grid_shape(const Arguments &params) {
    if (params.num_kv_splits > 1) {
      if (params.workspace_o_ptr == nullptr) {
        int const q_tokens = FMHACutlassSM100ParamsBuilder<Traits>::q_tokens_per_batch(params);
        return dim3(params.num_kv_splits * q_tokens, params.num_kv_heads, params.batch_size);
      }
      return dim3(params.num_ctas, 1, 1);
    }
    int const q_tokens = FMHACutlassSM100ParamsBuilder<Traits>::q_tokens_per_batch(params);
    return dim3(q_tokens, params.num_kv_heads, params.batch_size);
  }

  static dim3 get_grid_shape(const Params &params) {
    if (params.num_kv_splits > 1) {
      if (params.workspace_o_ptr == nullptr) {
        return dim3(params.num_kv_splits * params.q_tokens_per_batch, params.num_kv_heads,
                    params.batch_size);
      }
      return dim3(params.num_ctas, 1, 1);
    }
    if (params.use_precomputed_scheduler) {
      return dim3(params.num_ctas, 1, 1);
    }
    if (params.use_persistent_scheduler) {
      return dim3(1, params.num_kv_heads, params.batch_size * params.q_tokens_per_batch);
    }
    return dim3(params.q_tokens_per_batch, params.num_kv_heads, params.batch_size);
  }

  static dim3 get_block_shape() { return dim3(Traits::kNumThreads, 1, 1); }

  static int get_smem_size() { return SharedStorageSize; }

  // Layouts above the opt-in limit need the oversized shared-memory configuration (Rubin:
  // up to 327 KB per block, L1 reduced to 8 KB), selected through the shared-memory-mode
  // function attribute; MaxDynamicSharedMemorySize is rejected for such sizes.
  static cudaError_t set_smem_attribute() {
    int device_id = 0;
    cudaError_t status = cudaGetDevice(&device_id);
    if (status != cudaSuccess) {
      return status;
    }
    int optin_limit = 0;
    status =
        cudaDeviceGetAttribute(&optin_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
    if (status != cudaSuccess) {
      return status;
    }
    if (get_smem_size() <= optin_limit) {
      return cudaFuncSetAttribute(sm100_fmha_fwd_device_kernel<Traits>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, get_smem_size());
    }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13040
    int oversized_limit = 0;
    status = cudaDeviceGetAttribute(&oversized_limit, cudaDevAttrOversizedSharedMemoryPerBlock,
                                    device_id);
    if (status != cudaSuccess) {
      return status;
    }
    if (get_smem_size() > oversized_limit) {
      return cudaErrorInvalidValue;
    }
    return cudaFuncSetAttribute(sm100_fmha_fwd_device_kernel<Traits>,
                                cudaFuncAttributeSharedMemoryMode,
                                cudaSharedMemoryModeAllowOversizedSharedMemory);
#else
    return cudaErrorInvalidValue;
#endif
  }

  static cudaError_t query_active_cta_capacity(ActiveCtaCapacity &capacity) {
    constexpr int kClusterSize = 1;
    int device_id = 0;
    cudaError_t status = cudaGetDevice(&device_id);
    if (status != cudaSuccess) {
      return status;
    }

    static std::mutex cache_mutex;
    static ActiveCtaCapacity cached_capacity;
    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      if (cached_capacity.device_id == device_id && cached_capacity.cluster_size == kClusterSize &&
          cached_capacity.max_active_ctas > 0) {
        capacity = cached_capacity;
        return cudaSuccess;
      }
    }

    status = set_smem_attribute();
    if (status != cudaSuccess) {
      return status;
    }

    int blocks_per_sm = 0;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, sm100_fmha_fwd_device_kernel<Traits>, Traits::kNumThreads, get_smem_size());
    if (status != cudaSuccess) {
      return status;
    }

    int sm_count = 0;
    status = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
    if (status != cudaSuccess) {
      return status;
    }

    capacity.device_id = device_id;
    capacity.cluster_size = kClusterSize;
    capacity.blocks_per_sm = blocks_per_sm;
    capacity.sm_count = sm_count;
    capacity.max_active_ctas = blocks_per_sm * sm_count;
    if (capacity.max_active_ctas <= 0) {
      return cudaErrorInvalidValue;
    }

    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      cached_capacity = capacity;
    }
    return cudaSuccess;
  }

  static cudaError_t can_implement_error(const Arguments &params) {
    if (params.k_scale_ptr == nullptr || params.v_scale_ptr == nullptr) {
      return cudaErrorInvalidValue;
    }
    if (params.kv_indices_ptr == nullptr) {
      return cudaErrorInvalidValue;
    }
    {
      if (params.kv_block_indexes_ptr == nullptr || params.kv_block_num < 1 ||
          params.kv_block_num > Traits::kMaxSparseTopK) {
        return cudaErrorInvalidValue;
      }
    }
    if (params.o_direct_ptr == nullptr || params.num_qo_heads_orig <= 0) {
      return cudaErrorInvalidValue;
    }
    if (params.head_dim_qk != Traits::kHeadDim || params.head_dim_vo != Traits::kHeadDim) {
      return cudaErrorInvalidValue;
    }
    if (params.pack_factor != Traits::kHeadGroup) {
      return cudaErrorInvalidValue;
    }
    if (params.max_qo_len <= 0 || params.max_qo_len % Traits::kHeadGroup != 0) {
      return cudaErrorInvalidValue;
    }
    int const q_tokens = FMHACutlassSM100ParamsBuilder<Traits>::q_tokens_per_batch(params);

    if (q_tokens <= 0) {
      return cudaErrorInvalidValue;
    }
    if (params.num_kv_splits > 1) {

      {
        if (params.packed_work_range_ptr == nullptr || params.packed_work_info_ptr == nullptr ||
            params.kv_tile_begin_ptr == nullptr || params.kv_tile_end_ptr == nullptr ||
            params.kv_split_ptr == nullptr || params.workspace_o_ptr == nullptr ||
            (params.workspace_lse_ptr == nullptr && params.merge_counter_ptr == nullptr) ||
            params.num_ctas <= 0 || params.total_qo_len <= 0 ||
            params.num_qo_heads != params.num_kv_heads) {
          return cudaErrorInvalidValue;
        }
      }
    } else if (params.num_kv_splits != 1) {
      return cudaErrorInvalidValue;
    }
    if (params.num_kv_heads <= 0 || params.num_qo_heads <= 0) {
      return cudaErrorInvalidValue;
    }
    if (params.total_page_num <= 0 || params.batch_size <= 0) {
      return cudaErrorInvalidValue;
    }
    // The TMA descriptors take the head strides, which only need to keep each head's packed
    // rows apart and stay 16-byte aligned (e.g. a head's K and V slots interleaved per head).
    constexpr int kHeadDataBytes = Traits::kPageSize * (Traits::kHeadDim / 2);
    if (params.k_stride_h < kHeadDataBytes || params.k_stride_h % 16 != 0 ||
        params.v_stride_h < kHeadDataBytes || params.v_stride_h % 16 != 0) {
      return cudaErrorInvalidValue;
    }

    int head_group =
        params.h_r_original > 0 ? params.h_r_original : params.num_qo_heads / params.num_kv_heads;
    if (head_group != Traits::kHeadGroup) {
      return cudaErrorInvalidValue;
    }
    if (params.num_qo_heads_orig != params.num_kv_heads * Traits::kHeadGroup) {
      return cudaErrorInvalidValue;
    }
    if (params.o_direct_ptr == nullptr && params.o_ptr == nullptr) {
      return cudaErrorInvalidValue;
    }
    return cudaSuccess;
  }

  static bool can_implement(const Arguments &params) {
    return can_implement_error(params) == cudaSuccess;
  }

  static size_t get_workspace_size(Arguments const &) { return 0; }

  static cutlass::Status initialize_workspace(Arguments const &, void *, cudaStream_t) {
    return cutlass::Status::kSuccess;
  }

  static Params to_underlying_arguments(Arguments const &params, void *) {
    Params kernel_params;
    cudaError_t build_status = FMHACutlassSM100ParamsBuilder<Traits>::build(params, kernel_params);
    if (build_status != cudaSuccess) {
      return kernel_params;
    }

    ActiveCtaCapacity active_cta_capacity;
    cudaError_t capacity_status = query_active_cta_capacity(active_cta_capacity);
    if (capacity_status == cudaSuccess) {
      FMHACutlassSM100ParamsBuilder<Traits>::apply_scheduler_policy(
          params, active_cta_capacity.max_active_ctas, kernel_params);
    }
    return kernel_params;
  }

  static cudaError_t run(Params &params, cudaStream_t stream = nullptr) {
    cudaError_t attr_status = set_smem_attribute();
    if (attr_status != cudaSuccess) {
      return attr_status;
    }

    cudaLaunchAttribute attrs[1]{};
    constexpr int kClusterSize = 1;
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = kClusterSize;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim = get_grid_shape(params);
    launch_config.blockDim = get_block_shape();
    launch_config.dynamicSmemBytes = get_smem_size();
    launch_config.stream = stream;
    launch_config.attrs = attrs;
    launch_config.numAttrs = 1;

    cudaError_t launch_status =
        cudaLaunchKernelEx(&launch_config, sm100_fmha_fwd_device_kernel<Traits>, params);
    if (launch_status != cudaSuccess) {
      return launch_status;
    }
    return cudaGetLastError();
  }
};

template <class ProblemShapeOrTraits, class CollectiveMainloop = void,
          class CollectiveEpilogue = void, class TileScheduler = void, class KernelSchedule = void>
struct Sm100FmhaFwdKernelTmaWarpspecialized;

template <class Traits>
struct Sm100FmhaFwdKernelTmaWarpspecialized<Traits, void, void, void, void>
    : Sm100FmhaFwdQ8Kv4KernelTmaWarpspecialized<Traits> {};

template <class ProblemShape, class CollectiveMainloop, class CollectiveEpilogue,
          class TileScheduler, class KernelSchedule>
struct Sm100FmhaFwdKernelTmaWarpspecialized
    : Sm100FmhaFwdQ8Kv4KernelTmaWarpspecialized<typename CollectiveMainloop::FmhaTraits> {
  using FmhaTraits = typename CollectiveMainloop::FmhaTraits;
  using Base = Sm100FmhaFwdQ8Kv4KernelTmaWarpspecialized<FmhaTraits>;
  using ProblemShapeType = ProblemShape;
  using Mainloop = CollectiveMainloop;
  using Epilogue = CollectiveEpilogue;
  using Scheduler = TileScheduler;
  using Schedule = KernelSchedule;
  using MainloopTensorStorage = typename Mainloop::TensorStorage;
  using SharedStorage = typename Base::SharedStorage;
  using ClusterShape = typename Base::ClusterShape;
  using ArchTag = typename Base::ArchTag;

  static constexpr int MaxThreadsPerBlock = Base::MaxThreadsPerBlock;
  static constexpr int MinBlocksPerMultiprocessor = Base::MinBlocksPerMultiprocessor;
  static constexpr int SharedStorageSize = Base::SharedStorageSize;

  struct Arguments {
    ProblemShape problem_shape;
    typename Mainloop::Arguments mainloop;
    typename Epilogue::Arguments epilogue;
    typename Scheduler::Arguments tile_scheduler;
    cutlass::KernelHardwareInfo hw_info;
  };

  struct Params {
    ProblemShape problem_shape;
    typename Mainloop::Params mainloop;
    typename Epilogue::Params epilogue;
    typename Scheduler::Params tile_scheduler;
  };

  static size_t get_workspace_size(Arguments const &) { return 0; }

  static cutlass::Status initialize_workspace(Arguments const &, void *, cudaStream_t) {
    return cutlass::Status::kSuccess;
  }

  static bool can_implement(Arguments const &args) {
    return Base::can_implement(args.mainloop.load.fmha);
  }

  static Params to_underlying_arguments(Arguments const &args, void *workspace) {
    (void)workspace;
    Params params{};
    params.problem_shape = args.problem_shape;
    typename Base::ActiveCtaCapacity active_cta_capacity;
    int max_active_ctas = 0;
    cudaError_t capacity_status = Base::query_active_cta_capacity(active_cta_capacity);
    if (capacity_status == cudaSuccess) {
      max_active_ctas = active_cta_capacity.max_active_ctas;
    }
    params.mainloop = Mainloop::to_underlying_arguments(args.problem_shape, args.mainloop,
                                                        workspace, max_active_ctas);
    params.epilogue =
        Epilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace);
    params.tile_scheduler = Scheduler::to_underlying_arguments(args.tile_scheduler, args.hw_info);
    return params;
  }

  static dim3 get_grid_shape(Params const &params) {
    return Base::get_grid_shape(params.mainloop.kernel);
  }

  static dim3 get_block_shape() { return Base::get_block_shape(); }

  static cudaError_t run(Params &params, cudaStream_t stream = nullptr) {
    return Base::run(params.mainloop.kernel, stream);
  }
};

} // namespace cutlass::fmha::kernel
