#pragma once

#include "common/kernel/include/sm100_m3_decode_scheduler.hpp"
#include "common/kernel/include/tp4_nvls_layout.hpp"

#include <cuda_runtime.h>

#include "common/kernel/include/m3_decode_sm100_params.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/config.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/reg_reconfig.h"

#include "attn/kernel/include/sm100_m3_decode_problem.hpp"
#include "common/kernel/include/sm100_m3_decode_pipeline.hpp"
#include "common/kernel/include/sm100_m3_decode_storage.hpp"
#include "common/kernel/include/sm100_m3_decode_tma_descriptors.hpp"
#include "common/kernel/include/sm100_m3_decode_traits.hpp"

namespace cutlass::m3_decode::collective {
template <class Traits> struct Sm100M3DecodeMainloop;
template <class Traits> struct Sm100M3DecodeMegaAttenMainloop;
template <class Traits> struct Sm100M3DecodeMegaMoeMainloop;

} // namespace cutlass::m3_decode::collective
namespace cutlass::m3_decode::kernel {

using namespace cutlass::m3_decode::collective;

enum class M3DecodeKernelMode {
  kMonolithic,
  kMegaAtten,
  kMegaMoe,
};

template <class Traits, M3DecodeKernelMode Mode>
__global__ __launch_bounds__(Traits::kNumThreads, 1) void sm100_m3_decode_kernel(
    CUTE_GRID_CONSTANT Sm100M3DecodeKernelParams<Traits> const params) {
  extern __shared__ __align__(1024) uint8_t smem[];
  if constexpr (Mode == M3DecodeKernelMode::kMonolithic) {
    Sm100M3DecodeMainloop<Traits>::run_device(params, smem);
  } else if constexpr (Mode == M3DecodeKernelMode::kMegaAtten) {
    Sm100M3DecodeMegaAttenMainloop<Traits>::run_device(params, smem);
  } else {
    Sm100M3DecodeMegaMoeMainloop<Traits>::run_device(params, smem);
  }
}

template <class Traits, M3DecodeKernelMode Mode> struct Sm100M3DecodeKernel {
  using Storage = SharedStorage<Traits>;
  using Arguments = M3DecodeMegaKernelParams;
  using Params = Sm100M3DecodeKernelParams<Traits>;
  using Scheduler = Sm100M3DecodeScheduler<Traits>;

  static dim3 get_grid_shape(const Params &params) {
    return dim3(Scheduler::kGridCtas, params.num_kv_heads, params.batch_size);
  }

  static dim3 get_block_shape() { return dim3(Traits::kNumThreads, 1, 1); }

  static int get_smem_size() { return Storage::kActiveSmemBytes; }

  static int get_cluster_size(const Params &) { return Scheduler::kFallbackClusterSize; }

  // Each cache component may sit inside a larger per-page slab; only the page stride varies.
  static bool cache_page_strides_ok(const Arguments &params) {
    constexpr int64_t kTmaMaxStride = int64_t{1} << 40;
    constexpr int64_t kDataPageBytes = Traits::kPageSize * (Traits::kHeadDim / 2);
    constexpr int64_t kScalePageBytes = Traits::kPageSize * (Traits::kHeadDim / 16);
    auto ok = [](int64_t stride, int64_t minimum, int64_t alignment) {
      return stride >= minimum && stride % alignment == 0 && stride < kTmaMaxStride;
    };
    // K/V use the 16U4_ALIGN16B TMA type, which needs 32-byte strides and base addresses.
    auto aligned32 = [](void const *ptr) { return reinterpret_cast<uintptr_t>(ptr) % 32 == 0; };
    return params.k_stride_h == kDataPageBytes && params.v_stride_h == kDataPageBytes &&
           aligned32(params.k_ptr) && aligned32(params.v_ptr) &&
           ok(params.k_stride_n, kDataPageBytes * params.num_kv_heads, 32) &&
           ok(params.v_stride_n, kDataPageBytes * params.num_kv_heads, 32) &&
           ok(params.k_scale_page_stride, kScalePageBytes * params.num_kv_heads, 16) &&
           ok(params.v_scale_page_stride, kScalePageBytes * params.num_kv_heads, 16) &&
           ok(params.indexer_k_page_stride, kDataPageBytes, 16) &&
           ok(params.indexer_k_scale_page_stride, kScalePageBytes, 16);
  }

  static cudaError_t can_implement_error(const Arguments &params) {
    if constexpr (Mode == M3DecodeKernelMode::kMegaMoe) {
      if (params.batch_size != 1 || params.total_qo_len_orig != Traits::kMoeQueryTokens) {
        return cudaErrorInvalidValue;
      }
      if (params.post_attention_norm_ptr == nullptr || params.attention_residual_ptr == nullptr ||
          params.next_hidden_ptr == nullptr || params.tp_unicast_ptr == nullptr ||
          params.tp_multicast_ptr == nullptr || params.tp_rank < 0 || params.tp_rank >= 4 ||
          params.router_weight_ptr == nullptr || params.router_correction_bias_ptr == nullptr ||
          params.router_topk_idx_ptr == nullptr || params.router_topk_weight_ptr == nullptr ||
          params.shared_activation_ptr == nullptr || params.shared_activation_sf_ptr == nullptr ||
          params.shared_fc1_w_ptr == nullptr || params.shared_fc1_w_sf_ptr == nullptr ||
          params.shared_mid_ptr == nullptr || params.shared_fc2_w_ptr == nullptr ||
          params.shared_fc2_w_sf_ptr == nullptr ||
          params.routed_active_expert_count_ptr == nullptr ||
          params.routed_route_count_ptr == nullptr ||
          params.routed_active_expert_ids_ptr == nullptr ||
          params.routed_expert_route_offsets_ptr == nullptr ||
          params.routed_route_token_ids_ptr == nullptr ||
          params.routed_route_topk_ids_ptr == nullptr ||
          params.routed_route_expert_ids_ptr == nullptr ||
          params.routed_route_active_slots_ptr == nullptr ||
          params.routed_route_scales_ptr == nullptr || params.routed_activation_ptr == nullptr ||
          params.routed_activation_sf_ptr == nullptr || params.routed_fc1_w_ptr == nullptr ||
          params.routed_fc1_w_sf_ptr == nullptr || params.routed_fc1_alpha_ptr == nullptr ||
          params.routed_mid_ptr == nullptr || params.routed_mid_sf_ptr == nullptr ||
          params.routed_fc2_w_ptr == nullptr || params.routed_fc2_w_sf_ptr == nullptr ||
          params.routed_fc2_alpha_ptr == nullptr) {
        return cudaErrorInvalidValue;
      }
      return cudaSuccess;
    }

    if constexpr (Mode == M3DecodeKernelMode::kMegaAtten) {
      if (params.k_scale_ptr == nullptr || params.v_scale_ptr == nullptr ||
          params.kv_indices_ptr == nullptr || params.indexer_q_ptr == nullptr ||
          params.indexer_k_ptr == nullptr || params.indexer_k_scale_ptr == nullptr ||
          params.indexer_page_table_ptr == nullptr ||
          params.attention_projection_input_norm_weight_ptr == nullptr ||
          params.attention_projection_w_ptr == nullptr ||
          params.attention_projection_w_sf_ptr == nullptr ||
          params.attention_projection_workspace_data_ptr == nullptr ||
          params.attention_projection_workspace_scale_ptr == nullptr ||
          params.attention_projection_grid_epoch_ptr == nullptr ||
          params.attention_projection_special_partial_ptr == nullptr ||
          params.attention_projection_special_epochs_ptr == nullptr ||
          params.attention_projection_positions_ptr == nullptr ||
          params.attention_projection_slot_mapping_ptr == nullptr ||
          params.attention_projection_q_norm_weight_ptr == nullptr ||
          params.attention_projection_k_norm_weight_ptr == nullptr ||
          params.attention_projection_qi_norm_weight_ptr == nullptr ||
          params.attention_projection_ki_norm_weight_ptr == nullptr ||
          params.attention_projection_cos_sin_cache_ptr == nullptr ||
          params.o_direct_ptr == nullptr || params.o_sf_ptr == nullptr ||
          params.output_projection_w_ptr == nullptr ||
          params.output_projection_w_sf_ptr == nullptr ||
          params.output_projection_barrier_ptr == nullptr || params.tp_unicast_ptr == nullptr ||
          params.tp_multicast_ptr == nullptr || params.residual_in_ptr == nullptr ||
          params.attention_residual_ptr == nullptr || params.post_attention_norm_ptr == nullptr ||
          params.rms_weight_ptr == nullptr) {
        return cudaErrorInvalidValue;
      }
      if (params.kv_block_num != Traits::kSparseTopK || params.kv_page_stride <= 0 ||
          params.kv_page_stride > mega_atten::m3_decode::tp4_nvls::kIndexerMaxPages ||
          params.head_dim_qk != Traits::kHeadDim || params.head_dim_vo != Traits::kHeadDim ||
          params.num_kv_heads != 1 || params.num_qo_heads_orig != Traits::kHeadGroup ||
          params.batch_size != 1 || params.total_qo_len_orig != 8 ||
          !cache_page_strides_ok(params)) {
        return cudaErrorInvalidValue;
      }
      return cudaSuccess;
    }

    if (params.k_scale_ptr == nullptr || params.v_scale_ptr == nullptr) {
      return cudaErrorInvalidValue;
    }
    if (params.kv_indices_ptr == nullptr) {
      return cudaErrorInvalidValue;
    }
    if (params.kv_block_num != Traits::kSparseTopK || params.indexer_q_ptr == nullptr ||
        params.indexer_k_ptr == nullptr || params.indexer_k_scale_ptr == nullptr ||
        params.indexer_page_table_ptr == nullptr || params.kv_page_stride <= 0 ||
        params.kv_page_stride > mega_atten::m3_decode::tp4_nvls::kIndexerMaxPages) {
      return cudaErrorInvalidValue;
    }
    if (params.residual_in_ptr == nullptr ||
        params.attention_projection_input_norm_weight_ptr == nullptr ||
        params.attention_projection_w_ptr == nullptr ||
        params.attention_projection_w_sf_ptr == nullptr ||
        params.attention_projection_workspace_data_ptr == nullptr ||
        params.attention_projection_workspace_scale_ptr == nullptr ||
        params.attention_projection_grid_epoch_ptr == nullptr ||
        params.attention_projection_special_partial_ptr == nullptr ||
        params.attention_projection_special_epochs_ptr == nullptr ||
        params.attention_projection_positions_ptr == nullptr ||
        params.attention_projection_slot_mapping_ptr == nullptr ||
        params.attention_projection_q_norm_weight_ptr == nullptr ||
        params.attention_projection_k_norm_weight_ptr == nullptr ||
        params.attention_projection_qi_norm_weight_ptr == nullptr ||
        params.attention_projection_ki_norm_weight_ptr == nullptr ||
        params.attention_projection_cos_sin_cache_ptr == nullptr ||
        params.attention_projection_input_norm_eps <= 0.0f ||
        params.attention_projection_post_norm_eps <= 0.0f ||
        params.attention_projection_k_dequant_scale <= 0.0f ||
        params.attention_projection_v_dequant_scale <= 0.0f ||
        params.attention_projection_ki_dequant_scale <= 0.0f) {
      return cudaErrorInvalidValue;
    }
    if (params.o_direct_ptr == nullptr || params.num_qo_heads_orig <= 0) {
      return cudaErrorInvalidValue;
    }
    if (params.head_dim_qk != Traits::kHeadDim || params.head_dim_vo != Traits::kHeadDim) {
      return cudaErrorInvalidValue;
    }
    int const q_tokens = M3DecodeMegaKernelParamsBuilder<Traits>::q_tokens_per_batch(params);
    if (q_tokens <= 0 || q_tokens > Traits::kMaxQTokensPerBatch) {
      return cudaErrorInvalidValue;
    }
    if (params.num_kv_heads <= 0 || params.num_qo_heads <= 0) {
      return cudaErrorInvalidValue;
    }
    if (params.total_page_num <= 0 || params.batch_size <= 0) {
      return cudaErrorInvalidValue;
    }
    if (!cache_page_strides_ok(params)) {
      return cudaErrorInvalidValue;
    }

    if (params.num_qo_heads_orig != params.num_kv_heads * Traits::kHeadGroup) {
      return cudaErrorInvalidValue;
    }
    if (params.o_sf_ptr == nullptr) {
      return cudaErrorInvalidValue;
    }
    if (params.router_weight_ptr == nullptr || params.router_correction_bias_ptr == nullptr ||
        params.router_topk_idx_ptr == nullptr || params.router_topk_weight_ptr == nullptr) {
      return cudaErrorInvalidValue;
    }
    if (params.shared_activation_ptr == nullptr || params.shared_activation_sf_ptr == nullptr) {
      return cudaErrorInvalidValue;
    }
    if (params.shared_fc1_w_ptr == nullptr || params.shared_fc1_w_sf_ptr == nullptr ||
        params.shared_mid_ptr == nullptr || params.shared_fc2_w_ptr == nullptr ||
        params.shared_fc2_w_sf_ptr == nullptr) {
      return cudaErrorInvalidValue;
    }
    if (params.routed_active_expert_count_ptr == nullptr ||
        params.routed_route_count_ptr == nullptr ||
        params.routed_active_expert_ids_ptr == nullptr ||
        params.routed_expert_route_offsets_ptr == nullptr ||
        params.routed_route_token_ids_ptr == nullptr ||
        params.routed_route_topk_ids_ptr == nullptr ||
        params.routed_route_expert_ids_ptr == nullptr ||
        params.routed_route_active_slots_ptr == nullptr ||
        params.routed_route_scales_ptr == nullptr || params.routed_activation_ptr == nullptr ||
        params.routed_activation_sf_ptr == nullptr || params.routed_fc1_w_ptr == nullptr ||
        params.routed_fc1_w_sf_ptr == nullptr || params.routed_fc1_alpha_ptr == nullptr ||
        params.routed_mid_ptr == nullptr || params.routed_mid_sf_ptr == nullptr) {
      return cudaErrorInvalidValue;
    }
    if (params.routed_fc2_w_ptr == nullptr || params.routed_fc2_w_sf_ptr == nullptr ||
        params.routed_fc2_alpha_ptr == nullptr) {
      return cudaErrorInvalidValue;
    }
    return cudaSuccess;
  }

  static bool can_implement(const Arguments &params) {
    return can_implement_error(params) == cudaSuccess;
  }

  static cudaError_t initialize_params(Arguments const &params, Params &kernel_params) {
    if constexpr (Mode == M3DecodeKernelMode::kMegaAtten) {
      return M3DecodeMegaKernelParamsBuilder<Traits>::build_mega_atten(params, kernel_params);
    } else if constexpr (Mode == M3DecodeKernelMode::kMegaMoe) {
      return M3DecodeMegaKernelParamsBuilder<Traits>::build_mega_moe(params, kernel_params);
    } else {
      return M3DecodeMegaKernelParamsBuilder<Traits>::build(params, kernel_params);
    }
  }

  static Params to_underlying_arguments(Arguments const &params, void *) {
    Params kernel_params;
    initialize_params(params, kernel_params);
    return kernel_params;
  }

  static cudaError_t run(Params &params, cudaStream_t stream = nullptr) {
    cudaError_t attr_status =
        cudaFuncSetAttribute(sm100_m3_decode_kernel<Traits, Mode>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, get_smem_size());
    if (attr_status != cudaSuccess) {
      return attr_status;
    }
    cudaError_t cluster_attr_status = cudaFuncSetAttribute(
        sm100_m3_decode_kernel<Traits, Mode>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (cluster_attr_status != cudaSuccess) {
      return cluster_attr_status;
    }

    cudaLaunchAttribute attrs[2]{};
    int const cluster_size = get_cluster_size(params);
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[1].id = cudaLaunchAttributePreferredClusterDimension;
    attrs[0].val.clusterDim.x = cluster_size;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].val.preferredClusterDim.x = Scheduler::kPreferredClusterSize;
    attrs[1].val.preferredClusterDim.y = 1;
    attrs[1].val.preferredClusterDim.z = 1;
    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim = get_grid_shape(params);
    launch_config.blockDim = get_block_shape();
    launch_config.dynamicSmemBytes = get_smem_size();
    launch_config.stream = stream;
    launch_config.attrs = attrs;
    launch_config.numAttrs = 2;

    if (launch_config.gridDim.x != Scheduler::kGridCtas ||
        launch_config.gridDim.y != static_cast<unsigned>(params.num_kv_heads) ||
        launch_config.gridDim.z != static_cast<unsigned>(params.batch_size) ||
        cluster_size != Scheduler::kFallbackClusterSize) {
      return cudaErrorInvalidConfiguration;
    }

    cudaError_t launch_status =
        cudaLaunchKernelEx(&launch_config, sm100_m3_decode_kernel<Traits, Mode>, params);
    if (launch_status != cudaSuccess) {
      return launch_status;
    }
    return cudaGetLastError();
  }
};

template <class Traits>
using Sm100M3DecodeMegaKernel = Sm100M3DecodeKernel<Traits, M3DecodeKernelMode::kMonolithic>;

template <class Traits>
using Sm100M3DecodeMegaAttenKernel = Sm100M3DecodeKernel<Traits, M3DecodeKernelMode::kMegaAtten>;

template <class Traits>
using Sm100M3DecodeMegaMoeKernel = Sm100M3DecodeKernel<Traits, M3DecodeKernelMode::kMegaMoe>;

} // namespace cutlass::m3_decode::kernel
