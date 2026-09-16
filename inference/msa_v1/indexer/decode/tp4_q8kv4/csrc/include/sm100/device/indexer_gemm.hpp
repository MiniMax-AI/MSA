#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <mutex>

#include "sm100/kernel/sm100_indexer_gemm_kernel.hpp"

namespace minimax::msa_v1::indexer::decode::tp4_q8kv4 {

template <class Traits>
struct IndexerGemmRunner {
  static constexpr int kInlineCounterOffset = Traits::kPrepareThreads + 1;
  static constexpr int kDynamicCounterOffset = -1;
  using InlineKernel = IndexerGemmKernel<Traits, kInlineCounterOffset>;
  using DynamicKernel = IndexerGemmKernel<Traits, kDynamicCounterOffset>;

  static bool can_plan(IndexerGemmArguments const& arguments) {
    return arguments.page_table_ptr != nullptr &&
           arguments.kv_lengths_ptr != nullptr &&
           arguments.scheduler_workspace_ptr != nullptr &&
           arguments.batch > 0 && arguments.max_pages > 0 &&
           arguments.max_pages <= Traits::kMaximumPages &&
           (arguments.batch <= Traits::kPrepareThreads ||
            (arguments.scheduler_temp_storage_ptr != nullptr &&
             arguments.scheduler_temp_storage_bytes > 0));
  }

  static bool can_run(IndexerGemmArguments const& arguments) {
    return arguments.q_ptr != nullptr && arguments.packed_k_ptr != nullptr &&
           arguments.k_scale_ptr != nullptr &&
           arguments.page_table_ptr != nullptr &&
           arguments.kv_lengths_ptr != nullptr &&
           arguments.scheduler_workspace_ptr != nullptr &&
           arguments.output_ptr != nullptr && arguments.batch > 0 &&
           arguments.max_pages > 0 &&
           arguments.max_pages <= Traits::kMaximumPages &&
           arguments.physical_pages > 0 && arguments.sm_count > 0;
  }

  static cudaError_t scheduler_temp_storage_bytes(
      int batch, size_t& temp_storage_bytes) {
    temp_storage_bytes = 0;
    if (batch <= Traits::kPrepareThreads) {
      return cudaSuccess;
    }
    return get_indexer_gemm_scheduler_temp_storage_bytes(
        batch, temp_storage_bytes);
  }

  static cudaError_t encode_tma(CUtensorMap& descriptor, void* pointer,
                                uint64_t const (&dimensions)[3],
                                uint64_t const (&strides)[2],
                                uint32_t const (&box)[3],
                                CUtensorMapSwizzle swizzle) {
    uint32_t element_strides[3] = {1, 1, 1};
    CUresult const result = cuTensorMapEncodeTiled(
        &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, pointer, dimensions,
        strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
  }

  static cudaError_t to_plan_params(IndexerGemmArguments const& arguments,
                                    IndexerGemmParams& params) {
    params.page_table_ptr = arguments.page_table_ptr;
    params.kv_lengths_ptr = arguments.kv_lengths_ptr;
    params.scheduler_workspace_ptr =
        arguments.scheduler_workspace_ptr +
        (arguments.batch > Traits::kPrepareThreads ? 1 : 0);
    params.batch = arguments.batch;
    params.max_pages = arguments.max_pages;

    return cudaSuccess;
  }

  static cudaError_t to_underlying_arguments(
      IndexerGemmArguments const& arguments, IndexerGemmParams& params) {
    cudaError_t status = to_plan_params(arguments, params);
    if (status != cudaSuccess) {
      return status;
    }
    params.q_ptr = static_cast<uint8_t const*>(arguments.q_ptr);
    params.output_ptr = arguments.output_ptr;
    params.sm_count = arguments.sm_count;

    uint64_t packed_dimensions[3] = {
        static_cast<uint64_t>(Traits::kHeadDim),
        static_cast<uint64_t>(Traits::kHeadDim / 2),
        static_cast<uint64_t>(arguments.physical_pages)};
    uint64_t packed_strides[2] = {
        static_cast<uint64_t>(Traits::kHeadDim),
        static_cast<uint64_t>(Traits::kPackedKBytes)};
    uint32_t packed_box[3] = {
        static_cast<uint32_t>(Traits::kHeadDim),
        static_cast<uint32_t>(Traits::kHeadDim / 2), 1u};
    status = encode_tma(
        params.packed_k, const_cast<void*>(arguments.packed_k_ptr),
        packed_dimensions, packed_strides, packed_box,
        CU_TENSOR_MAP_SWIZZLE_128B);
    if (status != cudaSuccess) {
      return status;
    }

    uint64_t scale_dimensions[3] = {
        static_cast<uint64_t>(Traits::kScaleBytes / 8), 8u,
        static_cast<uint64_t>(arguments.physical_pages)};
    uint64_t scale_strides[2] = {
        static_cast<uint64_t>(Traits::kScaleBytes / 8),
        static_cast<uint64_t>(Traits::kScaleBytes)};
    uint32_t scale_box[3] = {
        static_cast<uint32_t>(Traits::kScaleBytes / 8), 8u, 1u};
    return encode_tma(params.k_scale,
                      const_cast<void*>(arguments.k_scale_ptr),
                      scale_dimensions, scale_strides, scale_box,
                      CU_TENSOR_MAP_SWIZZLE_NONE);
  }

  static cudaError_t initialize() {
    static std::once_flag attribute_once;
    static cudaError_t attribute_status = cudaSuccess;
    std::call_once(attribute_once, [] {
      attribute_status = cudaFuncSetAttribute(
          indexer_gemm_kernel<Traits, kInlineCounterOffset>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          InlineKernel::kSharedStorageBytes);
      if (attribute_status == cudaSuccess) {
        attribute_status = cudaFuncSetAttribute(
            indexer_gemm_kernel<Traits, kDynamicCounterOffset>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            DynamicKernel::kSharedStorageBytes);
      }
    });
    if (attribute_status != cudaSuccess) {
      return attribute_status;
    }
    return cudaSuccess;
  }

  static cudaError_t plan(IndexerGemmParams const& params,
                          void* scheduler_temp_storage,
                          size_t scheduler_temp_storage_bytes,
                          cudaStream_t stream) {
    cudaError_t status = initialize();
    if (status != cudaSuccess) {
      return status;
    }
    if (params.batch <= Traits::kPrepareThreads) {
      int const prepare_threads = ((params.batch + 31) / 32) * 32;
      prepare_indexer_gemm_scheduler<Traits>
          <<<1, prepare_threads, 0, stream>>>(params);
      status = cudaGetLastError();
    } else {
      int constexpr prepare_threads = 256;
      int const prepare_blocks =
          (params.batch - 1) / prepare_threads + 1;
      prepare_indexer_gemm_scheduler_counts<Traits>
          <<<prepare_blocks, prepare_threads, 0, stream>>>(params);
      status = cudaGetLastError();
      if (status == cudaSuccess) {
        int32_t* prefix = params.scheduler_workspace_ptr + 1;
        status = run_indexer_gemm_scheduler_scan(
            prefix, params.batch, scheduler_temp_storage,
            scheduler_temp_storage_bytes, stream);
      }
    }
    return status;
  }

  static cudaError_t run(IndexerGemmParams const& params,
                         cudaStream_t stream) {
    int const counter_offset =
        params.batch <= Traits::kPrepareThreads ? kInlineCounterOffset
                                                : kDynamicCounterOffset;
    reset_indexer_gemm_scheduler_counter<<<1, 1, 0, stream>>>(
        params.scheduler_workspace_ptr + counter_offset);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
      return status;
    }
    dim3 const grid(params.sm_count * 4, 1, 1);
    cudaLaunchAttribute attributes[1]{};
    attributes[0].id = cudaLaunchAttributeClusterDimension;
    attributes[0].val.clusterDim.x = 1;
    attributes[0].val.clusterDim.y = 1;
    attributes[0].val.clusterDim.z = 1;
    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim = grid;
    launch_config.blockDim = dim3(InlineKernel::kThreadCount, 1, 1);
    launch_config.dynamicSmemBytes = InlineKernel::kSharedStorageBytes;
    launch_config.stream = stream;
    launch_config.attrs = attributes;
    launch_config.numAttrs = 1;
    cudaError_t const launch_status =
        params.batch <= Traits::kPrepareThreads
            ? cudaLaunchKernelEx(
                  &launch_config,
                  indexer_gemm_kernel<Traits, kInlineCounterOffset>, params)
            : cudaLaunchKernelEx(
                  &launch_config,
                  indexer_gemm_kernel<Traits, kDynamicCounterOffset>, params);
    return launch_status == cudaSuccess ? cudaGetLastError() : launch_status;
  }
};

}  // namespace minimax::msa_v1::indexer::decode::tp4_q8kv4
