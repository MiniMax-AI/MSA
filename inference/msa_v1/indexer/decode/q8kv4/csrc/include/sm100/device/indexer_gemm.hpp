#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>

#include "sm100/kernel/sm100_indexer_gemm_kernel.hpp"

namespace minimax::msa_v1::indexer::decode::q8kv4 {

template <class Traits> struct IndexerGemmRunner {
  using Kernel = IndexerGemmKernel<Traits>;

  static bool can_run(IndexerGemmArguments const &arguments) {
    return arguments.q_ptr != nullptr && arguments.packed_k_ptr != nullptr &&
           arguments.k_scale_ptr != nullptr && arguments.page_table_ptr != nullptr &&
           arguments.kv_lengths_ptr != nullptr && arguments.scheduler_workspace_ptr != nullptr &&
           arguments.output_ptr != nullptr && arguments.query_length >= 1 &&
           arguments.query_length <= Traits::kMaxQueryLength && arguments.batch > 0 &&
           arguments.max_pages > 0 && arguments.max_pages <= Traits::kMaximumPages &&
           arguments.physical_pages > 0 && arguments.sm_count > 0;
  }

  static cudaError_t encode_tma(CUtensorMap &descriptor, void *pointer,
                                uint64_t const (&dimensions)[3], uint64_t const (&strides)[2],
                                uint32_t const (&box)[3], CUtensorMapSwizzle swizzle) {
    uint32_t element_strides[3] = {1, 1, 1};
    CUresult const result = cuTensorMapEncodeTiled(
        &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, pointer, dimensions, strides, box,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
  }

  static cudaError_t to_plan_params(IndexerGemmArguments const &arguments,
                                    IndexerGemmParams &params) {
    params.page_table_ptr = arguments.page_table_ptr;
    params.kv_lengths_ptr = arguments.kv_lengths_ptr;
    params.scheduler_workspace_ptr = arguments.scheduler_workspace_ptr;
    params.sm_count = arguments.sm_count;
    params.batch = arguments.batch;
    params.max_pages = arguments.max_pages;

    return cudaSuccess;
  }

  static cudaError_t to_underlying_arguments(IndexerGemmArguments const &arguments,
                                             IndexerGemmParams &params) {
    cudaError_t status = to_plan_params(arguments, params);
    if (status != cudaSuccess) {
      return status;
    }
    params.q_ptr = static_cast<uint8_t const *>(arguments.q_ptr);
    params.query_length = arguments.query_length;
    params.output_ptr = arguments.output_ptr;
    params.sm_count = arguments.sm_count;

    uint64_t packed_dimensions[3] = {static_cast<uint64_t>(Traits::kHeadDim),
                                     static_cast<uint64_t>(Traits::kHeadDim / 2),
                                     static_cast<uint64_t>(arguments.physical_pages)};
    uint64_t packed_strides[2] = {static_cast<uint64_t>(Traits::kHeadDim),
                                  static_cast<uint64_t>(Traits::kPackedKBytes)};
    uint32_t packed_box[3] = {static_cast<uint32_t>(Traits::kHeadDim),
                              static_cast<uint32_t>(Traits::kHeadDim / 2), 1u};
    status = encode_tma(params.packed_k, const_cast<void *>(arguments.packed_k_ptr),
                        packed_dimensions, packed_strides, packed_box, CU_TENSOR_MAP_SWIZZLE_128B);
    if (status != cudaSuccess) {
      return status;
    }

    uint64_t scale_dimensions[3] = {static_cast<uint64_t>(Traits::kScaleBytes / 8), 8u,
                                    static_cast<uint64_t>(arguments.physical_pages)};
    uint64_t scale_strides[2] = {static_cast<uint64_t>(Traits::kScaleBytes / 8),
                                 static_cast<uint64_t>(Traits::kScaleBytes)};
    uint32_t scale_box[3] = {static_cast<uint32_t>(Traits::kScaleBytes / 8), 8u, 1u};
    return encode_tma(params.k_scale, const_cast<void *>(arguments.k_scale_ptr), scale_dimensions,
                      scale_strides, scale_box, CU_TENSOR_MAP_SWIZZLE_NONE);
  }

  static cudaError_t initialize() {
    static std::once_flag attribute_once;
    static cudaError_t attribute_status = cudaSuccess;
    std::call_once(attribute_once, [] {
      attribute_status = cudaFuncSetAttribute(indexer_gemm_kernel<Traits>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              Kernel::kSharedStorageBytes);
    });
    if (attribute_status != cudaSuccess) {
      return attribute_status;
    }
    return cudaSuccess;
  }

  static cudaError_t run(IndexerGemmParams const &params, cudaStream_t stream) {
    cudaError_t const initialize_status = initialize();
    if (initialize_status != cudaSuccess) {
      return initialize_status;
    }
    dim3 const grid(static_cast<unsigned int>(params.sm_count * Traits::kCtasPerWorker), 1, 1);
    cudaLaunchAttribute attributes[1]{};
    attributes[0].id = cudaLaunchAttributeClusterDimension;
    attributes[0].val.clusterDim.x = 1;
    attributes[0].val.clusterDim.y = 1;
    attributes[0].val.clusterDim.z = 1;
    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim = grid;
    launch_config.blockDim = dim3(Kernel::kThreadCount, 1, 1);
    launch_config.dynamicSmemBytes = Kernel::kSharedStorageBytes;
    launch_config.stream = stream;
    launch_config.attrs = attributes;
    launch_config.numAttrs = 1;
    cudaError_t const launch_status =
        cudaLaunchKernelEx(&launch_config, indexer_gemm_kernel<Traits>, params);
    return launch_status == cudaSuccess ? cudaGetLastError() : launch_status;
  }
};

} // namespace minimax::msa_v1::indexer::decode::q8kv4
