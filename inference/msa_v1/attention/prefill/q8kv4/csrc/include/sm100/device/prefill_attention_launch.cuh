#pragma once

#include "sm100/kernel/sm100_prefill_attention_kernel.cuh"

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/builders/sm100_common.inl"

namespace minimax::msa_v1::attention::prefill::q8kv4 {
namespace {

cudaError_t encode_page_head_tma(cute::TmaDescriptor &descriptor, uint8_t const *base,
                                 int physical_pages, int num_kv_heads, int tile_rows) {
  cuuint64_t const global_dimensions[2] = {
      128,
      static_cast<cuuint64_t>(physical_pages) * num_kv_heads * tile_rows,
  };
  cuuint64_t const global_strides[1] = {128};
  cuuint32_t const box_dimensions[2] = {
      128,
      static_cast<cuuint32_t>(tile_rows),
  };
  cuuint32_t const element_strides[2] = {1, 1};
  CUresult const result = cuTensorMapEncodeTiled(
      reinterpret_cast<CUtensorMap *>(&descriptor), CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
      const_cast<uint8_t *>(base), global_dimensions, global_strides, box_dimensions,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

} // namespace

cudaError_t launch_prefill_attention(PrefillArguments const &arguments, cudaStream_t stream) {
  using namespace cute;
  using namespace detail;

  using TiledMmaQK =
      decltype(make_tiled_mma(SM100_MMA_F8F6F4_SS<Element, Element, Accumulator, 128, 128,
                                                  UMMA::Major::K, UMMA::Major::K>{}));
  using TiledMmaPV =
      decltype(make_tiled_mma(SM100_MMA_F8F6F4_TS<Element, Element, Accumulator, 128, 128,
                                                  UMMA::Major::K, UMMA::Major::MN>{}));

  using QShape = decltype(make_shape(Int<128>{}, Int<128>{}, Int<kQStages>{}));
  using KShape = decltype(make_shape(Int<128>{}, Int<128>{}, _1{}));
  using VShape = decltype(make_shape(Int<128>{}, Int<128>{}, _1{}));
  using QAtom = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
                         UMMA::Major::K, Element, decltype(shape<0>(QShape{})),
                         decltype(shape<1>(QShape{}))>());
  using KAtom = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
                         UMMA::Major::K, Element, decltype(shape<0>(KShape{})),
                         decltype(shape<1>(KShape{}))>());
  using VAtom = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
                         UMMA::Major::MN, Element, decltype(shape<0>(VShape{})),
                         decltype(shape<1>(VShape{}))>());
  using QMmaShape = decltype(partition_shape_A(TiledMmaQK{}, QShape{}));
  using KMmaShape = decltype(partition_shape_B(TiledMmaQK{}, KShape{}));
  using VMmaShape = decltype(partition_shape_B(TiledMmaPV{}, VShape{}));
  using QSmemLayout = decltype(UMMA::tile_to_mma_shape(QAtom{}, QMmaShape{}));
  using KSmemLayout = decltype(UMMA::tile_to_mma_shape(KAtom{}, KMmaShape{}));
  using VSmemLayout = decltype(UMMA::tile_to_mma_shape(VAtom{}, VMmaShape{}));
  using QLogicalLayout = decltype(tile_to_shape(QAtom{}, QShape{}, Step<_1, _2, _3>{}));
  using KLogicalLayout = decltype(tile_to_shape(KAtom{}, KShape{}, Step<_1, _2, _3>{}));
  using VLogicalLayout = decltype(tile_to_shape(VAtom{}, VShape{}, Step<_1, _2, _3>{}));
  static_assert(cosize_v<QSmemLayout> == cosize_v<QLogicalLayout>);
  static_assert(cosize_v<KSmemLayout> == cosize_v<KLogicalLayout>);
  static_assert(cosize_v<VSmemLayout> == cosize_v<VLogicalLayout>);

  auto q_gmem_shape = make_shape(arguments.total_q * arguments.num_q_heads, Int<kHeadDim>{});
  auto q_gmem_stride = make_stride(Int<kHeadDim>{}, _1{});
  auto tma_q =
      make_tma_copy(SM90_TMA_LOAD{},
                    make_tensor(make_gmem_ptr(reinterpret_cast<Element const *>(arguments.q_ptr)),
                                make_layout(q_gmem_shape, q_gmem_stride)),
                    QTokenSmemLayout{});

  cute::TmaDescriptor tma_packed_k{};
  cute::TmaDescriptor tma_packed_v{};
  cute::TmaDescriptor tma_k_scale{};
  cute::TmaDescriptor tma_v_scale{};
  cudaError_t status = encode_page_head_tma(tma_packed_k, arguments.packed_k_ptr,
                                            arguments.physical_pages, arguments.num_kv_heads, 64);
  if (status != cudaSuccess) {
    return status;
  }
  status = encode_page_head_tma(tma_packed_v, arguments.packed_v_ptr, arguments.physical_pages,
                                arguments.num_kv_heads, 64);
  if (status != cudaSuccess) {
    return status;
  }
  status = encode_page_head_tma(tma_k_scale, arguments.k_scale_ptr, arguments.physical_pages,
                                arguments.num_kv_heads, 8);
  if (status != cudaSuccess) {
    return status;
  }
  status = encode_page_head_tma(tma_v_scale, arguments.v_scale_ptr, arguments.physical_pages,
                                arguments.num_kv_heads, 8);
  if (status != cudaSuccess) {
    return status;
  }

  using Storage = SharedStorage<QSmemLayout, KSmemLayout, VSmemLayout>;
  using Params = KernelParams<decltype(tma_q), decltype(q_gmem_shape)>;
  Params kernel_params{arguments,    tma_q,       q_gmem_shape, tma_packed_k,
                       tma_packed_v, tma_k_scale, tma_v_scale};
  auto *kernel =
      &prefill_attention_kernel<Storage, TiledMmaQK, TiledMmaPV, QSmemLayout, KSmemLayout,
                                VSmemLayout, decltype(tma_q), decltype(q_gmem_shape)>;
  status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Storage));
  if (status != cudaSuccess) {
    return status;
  }

  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(arguments.work_capacity, 1, 1);
  config.blockDim = dim3(WarpSpecialization::kThreads, 1, 1);
  config.dynamicSmemBytes = sizeof(Storage);
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, kernel, kernel_params);
}

} // namespace minimax::msa_v1::attention::prefill::q8kv4
