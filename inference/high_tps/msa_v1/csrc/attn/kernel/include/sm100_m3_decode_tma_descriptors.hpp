#pragma once

#include "common/kernel/include/sm100_m3_decode_tma_common.hpp"
#include "common/kernel/include/sm100_m3_decode_tma_descriptors.hpp"

namespace cutlass::m3_decode::collective {

template <class Traits>
cudaError_t
M3DecodeMegaKernelParamsBuilder<Traits>::build_q_desc(const M3DecodeMegaKernelParams &src,
                                                      TmaDescriptors &dst) {
  int const q_tokens = q_tokens_total(src);
  if (q_tokens <= 0) {
    return cudaErrorInvalidValue;
  }

  uint64_t global_dim[5] = {
      static_cast<uint64_t>(Traits::kHeadDim), static_cast<uint64_t>(Traits::kHeadGroup),
      static_cast<uint64_t>(src.num_kv_heads), static_cast<uint64_t>(q_tokens), 1u};
  uint64_t global_stride_bytes[4] = {static_cast<uint64_t>(src.q_stride_h),
                                     static_cast<uint64_t>(src.q_stride_h * Traits::kHeadGroup),
                                     static_cast<uint64_t>(src.q_stride_n),
                                     static_cast<uint64_t>(src.q_stride_n * q_tokens)};
  uint32_t box_dim[5] = {static_cast<uint32_t>(Traits::kHeadDim),
                         static_cast<uint32_t>(Traits::kHeadGroup), 1u, 1u, 1u};

  return encode_tma(dst.q, Sm100M3DecodeTmaDtype::E4M3, 5, src.q_ptr, global_dim,
                    global_stride_bytes, box_dim, CU_TENSOR_MAP_SWIZZLE_128B);
}

template <class Traits>
cudaError_t
M3DecodeMegaKernelParamsBuilder<Traits>::build_kv_desc(CUtensorMap &desc, void *ptr,
                                                       int num_kv_heads, int stride_head_bytes,
                                                       int64_t stride_page_bytes, int total_pages) {
  uint64_t global_dim[4] = {
      static_cast<uint64_t>(Traits::kHeadDim), static_cast<uint64_t>(Traits::kPageSize),
      static_cast<uint64_t>(num_kv_heads), static_cast<uint64_t>(total_pages)};
  uint64_t global_stride_bytes[3] = {static_cast<uint64_t>(Traits::kHeadDim / 2),
                                     static_cast<uint64_t>(stride_head_bytes),
                                     static_cast<uint64_t>(stride_page_bytes)};
  uint32_t box_dim[4] = {static_cast<uint32_t>(Traits::kHeadDim),
                         static_cast<uint32_t>(Traits::kTileKv), 1u, 1u};

  return encode_tma(desc, Sm100M3DecodeTmaDtype::E2M1, 4, ptr, global_dim, global_stride_bytes,
                    box_dim, CU_TENSOR_MAP_SWIZZLE_128B, true);
}

template <class Traits>
cudaError_t
M3DecodeMegaKernelParamsBuilder<Traits>::build_indexer_k_desc(const M3DecodeMegaKernelParams &src,
                                                              TmaDescriptors &dst) {
  if (src.indexer_k_ptr == nullptr || src.total_page_num <= 0) {
    return cudaErrorInvalidValue;
  }

  constexpr uint64_t kPackedRowBytes = Traits::kHeadDim / 2;
  uint64_t global_dim[3] = {static_cast<uint64_t>(Traits::kHeadDim), kPackedRowBytes,
                            static_cast<uint64_t>(src.total_page_num)};
  // Rows are dense inside a page; consecutive pages are indexer_k_page_stride bytes apart.
  uint64_t global_stride_bytes[2] = {static_cast<uint64_t>(Traits::kHeadDim),
                                     static_cast<uint64_t>(src.indexer_k_page_stride)};
  uint32_t box_dim[3] = {static_cast<uint32_t>(Traits::kHeadDim),
                         static_cast<uint32_t>(kPackedRowBytes), 1u};
  return encode_tma(dst.indexer_k, Sm100M3DecodeTmaDtype::U8, 3, src.indexer_k_ptr, global_dim,
                    global_stride_bytes, box_dim, CU_TENSOR_MAP_SWIZZLE_128B);
}

template <class Traits>
cudaError_t M3DecodeMegaKernelParamsBuilder<Traits>::build_indexer_k_scale_desc(
    const M3DecodeMegaKernelParams &src, TmaDescriptors &dst) {
  if (src.indexer_k_scale_ptr == nullptr || src.total_page_num <= 0) {
    return cudaErrorInvalidValue;
  }

  constexpr uint64_t kScaleGroups = Traits::kHeadDim / Traits::kPackFactor;
  constexpr uint64_t kScaleReshape = 16;
  static_assert(kScaleGroups == 8, "Indexer expects 8 scale bytes per token.");
  static_assert(Traits::kPageSize % kScaleReshape == 0,
                "Indexer scale reshape must divide the page.");

  uint64_t global_dim[3] = {kScaleGroups * kScaleReshape,
                            static_cast<uint64_t>(Traits::kPageSize) / kScaleReshape,
                            static_cast<uint64_t>(src.total_page_num)};
  uint64_t global_stride_bytes[2] = {kScaleGroups * kScaleReshape,
                                     static_cast<uint64_t>(src.indexer_k_scale_page_stride)};
  uint32_t box_dim[3] = {static_cast<uint32_t>(kScaleGroups * kScaleReshape),
                         static_cast<uint32_t>(Traits::kPageSize / kScaleReshape), 1u};
  return encode_tma(dst.indexer_k_scale, Sm100M3DecodeTmaDtype::U8, 3, src.indexer_k_scale_ptr,
                    global_dim, global_stride_bytes, box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
}

template <class Traits>
cudaError_t M3DecodeMegaKernelParamsBuilder<Traits>::build_scale_desc(CUtensorMap &desc, void *ptr,
                                                                      int num_kv_heads,
                                                                      int64_t stride_page_bytes,
                                                                      int total_pages) {
  constexpr int kScaleGroups = Traits::kHeadDim / Traits::kPackFactor;
  constexpr int kScaleReshape = 16;
  static_assert(kScaleGroups == 8, "Q8KV4 attention expects 8 scale bytes per token.");
  static_assert(Traits::kTileKv % kScaleReshape == 0, "scale TMA reshape must divide the KV tile.");

  uint64_t global_dim[4] = {static_cast<uint64_t>(kScaleGroups * kScaleReshape),
                            static_cast<uint64_t>(Traits::kPageSize / kScaleReshape),
                            static_cast<uint64_t>(num_kv_heads),
                            static_cast<uint64_t>(total_pages)};
  // Scale rows are dense inside a page and heads are dense inside a page; consecutive pages are
  // stride_page_bytes apart.
  uint64_t global_stride_bytes[3] = {static_cast<uint64_t>(kScaleGroups * kScaleReshape),
                                     static_cast<uint64_t>(Traits::kPageSize * kScaleGroups),
                                     static_cast<uint64_t>(stride_page_bytes)};
  uint32_t box_dim[4] = {static_cast<uint32_t>(kScaleGroups * kScaleReshape),
                         static_cast<uint32_t>(Traits::kTileKv / kScaleReshape), 1u, 1u};

  return encode_tma(desc, Sm100M3DecodeTmaDtype::E4M3, 4, ptr, global_dim, global_stride_bytes,
                    box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
}

template <class Traits>
cudaError_t
M3DecodeMegaKernelParamsBuilder<Traits>::build_o_desc(const M3DecodeMegaKernelParams &src,
                                                      TmaDescriptors &dst) {
  if (src.o_direct_ptr == nullptr) {
    return cudaErrorInvalidValue;
  }

  constexpr int kOChunkDim = 64;
  constexpr int kOChunks = Traits::kHeadDim / kOChunkDim;
  int const q_tokens = o_tokens_total(src);
  if (q_tokens <= 0) {
    return cudaErrorInvalidValue;
  }
  static_assert(Traits::kHeadDim % kOChunkDim == 0,
                "O TMA chunks must evenly cover the head dimension.");
  static_assert(kOChunkDim * 2 == 128, "BF16 O TMA innermost dimension must fit 128B swizzle.");

  uint64_t global_dim[5] = {static_cast<uint64_t>(kOChunkDim), static_cast<uint64_t>(kOChunks),
                            static_cast<uint64_t>(src.num_qo_heads_orig),
                            static_cast<uint64_t>(q_tokens), 1u};
  uint64_t global_stride_bytes[4] = {
      static_cast<uint64_t>(kOChunkDim * 2),
      static_cast<uint64_t>(Traits::kHeadDim * 2),
      static_cast<uint64_t>(src.num_qo_heads_orig * Traits::kHeadDim * 2),
      static_cast<uint64_t>(q_tokens * src.num_qo_heads_orig * Traits::kHeadDim * 2),
  };
  uint32_t box_dim[5] = {static_cast<uint32_t>(kOChunkDim), static_cast<uint32_t>(kOChunks),
                         static_cast<uint32_t>(Traits::kHeadGroup), 1u, 1u};

  return encode_tma(dst.o, Sm100M3DecodeTmaDtype::BF16, 5, src.o_direct_ptr, global_dim,
                    global_stride_bytes, box_dim, CU_TENSOR_MAP_SWIZZLE_128B);
}

template <class Traits>
cudaError_t M3DecodeMegaKernelParamsBuilder<Traits>::build_output_projection_descs(
    const M3DecodeMegaKernelParams &src, TmaDescriptors &dst) {
  constexpr uint64_t kTileM = 128;
  constexpr uint64_t kTileK = 128;
  constexpr uint64_t kScaleTileRows = 32;
  constexpr uint64_t kScaleTileRowBytes = 16;
  constexpr uint64_t kScaleTileBytes = kScaleTileRows * kScaleTileRowBytes;
  constexpr uint64_t kActivationHeads = 16;
  constexpr uint64_t kProblemM = mega_atten::m3_decode::output_projection::kProblemM;
  constexpr uint64_t kProblemN = mega_atten::m3_decode::output_projection::kProblemN;
  constexpr uint64_t kProblemK = mega_atten::m3_decode::output_projection::kProblemK;

  uint64_t w_global_dim[2] = {kProblemK, kProblemM};
  uint64_t w_global_stride[1] = {kProblemK};
  uint32_t w_box_dim[2] = {static_cast<uint32_t>(kTileK), static_cast<uint32_t>(kTileM)};
  cudaError_t status = encode_tma(dst.output_projection_w, Sm100M3DecodeTmaDtype::E4M3, 2,
                                  src.output_projection_w_ptr, w_global_dim, w_global_stride,
                                  w_box_dim, CU_TENSOR_MAP_SWIZZLE_128B);
  if (status != cudaSuccess) {
    return status;
  }

  uint64_t const scale_tiles = (kProblemM / kTileM) * (kProblemK / kTileK);
  uint64_t w_sf_global_dim[3] = {kScaleTileRowBytes, kScaleTileRows, scale_tiles};
  uint64_t w_sf_global_stride[2] = {kScaleTileRowBytes, kScaleTileBytes};
  uint32_t w_sf_box_dim[3] = {static_cast<uint32_t>(kScaleTileRowBytes),
                              static_cast<uint32_t>(kScaleTileRows), 1u};
  status = encode_tma(dst.output_projection_w_sf, Sm100M3DecodeTmaDtype::E4M3, 3,
                      src.output_projection_w_sf_ptr, w_sf_global_dim, w_sf_global_stride,
                      w_sf_box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
  if (status != cudaSuccess) {
    return status;
  }

  uint64_t o_global_dim[3] = {kTileK, kActivationHeads, kProblemN};
  uint64_t o_global_stride[2] = {kTileK, kProblemK};
  uint32_t o_box_dim[3] = {static_cast<uint32_t>(kTileK), 1u, static_cast<uint32_t>(kProblemN)};
  status = encode_tma(dst.output_projection_o, Sm100M3DecodeTmaDtype::E4M3, 3, src.o_direct_ptr,
                      o_global_dim, o_global_stride, o_box_dim, CU_TENSOR_MAP_SWIZZLE_128B);
  if (status != cudaSuccess) {
    return status;
  }

  uint64_t padded_sfb_global_dim[3] = {16u, kProblemN, kProblemK / kTileK};
  uint64_t padded_sfb_global_stride[2] = {16u, 512u};
  uint32_t padded_sfb_box_dim[3] = {16u, static_cast<uint32_t>(kProblemN), 1u};
  return encode_tma(dst.output_projection_o_sf_padded, Sm100M3DecodeTmaDtype::E4M3, 3, src.o_sf_ptr,
                    padded_sfb_global_dim, padded_sfb_global_stride, padded_sfb_box_dim,
                    CU_TENSOR_MAP_SWIZZLE_NONE);
}

template <class Traits>
void M3DecodeMegaKernelParamsBuilder<Traits>::build_attention_projection_params(
    const M3DecodeMegaKernelParams &src, Sm100M3DecodeKernelParams<Traits> &dst) {
  dst.attention_projection = cutlass::m3_decode::collective::attention_projection::KernelParams{
      cutlass::m3_decode::collective::attention_projection::make_tma_params(
          static_cast<cutlass::m3_decode::collective::attention_projection::ElementAB const *>(
              src.attention_projection_w_ptr),
          static_cast<cutlass::m3_decode::collective::attention_projection::ElementSF const *>(
              src.attention_projection_w_sf_ptr),
          static_cast<cutlass::m3_decode::collective::attention_projection::ElementAB const *>(
              src.attention_projection_workspace_data_ptr)),
      static_cast<__nv_bfloat16 const *>(src.residual_in_ptr),
      static_cast<__nv_bfloat16 const *>(src.attention_projection_input_norm_weight_ptr),
      static_cast<uint8_t *>(src.attention_projection_workspace_data_ptr),
      static_cast<uint8_t *>(src.attention_projection_workspace_scale_ptr),
      static_cast<uint32_t *>(src.attention_projection_grid_epoch_ptr),
      static_cast<float *>(src.attention_projection_special_partial_ptr),
      static_cast<uint32_t *>(src.attention_projection_special_epochs_ptr),
      static_cast<int64_t const *>(src.attention_projection_positions_ptr),
      static_cast<int64_t const *>(src.attention_projection_slot_mapping_ptr),
      static_cast<__nv_bfloat16 const *>(src.attention_projection_q_norm_weight_ptr),
      static_cast<__nv_bfloat16 const *>(src.attention_projection_k_norm_weight_ptr),
      static_cast<__nv_bfloat16 const *>(src.attention_projection_qi_norm_weight_ptr),
      static_cast<__nv_bfloat16 const *>(src.attention_projection_ki_norm_weight_ptr),
      static_cast<__nv_bfloat16 const *>(src.attention_projection_cos_sin_cache_ptr),
      static_cast<uint8_t *>(src.q_ptr),
      static_cast<uint8_t *>(src.indexer_q_ptr),
      static_cast<uint8_t *>(src.k_ptr),
      static_cast<uint8_t *>(src.k_scale_ptr),
      static_cast<uint8_t *>(src.v_ptr),
      static_cast<uint8_t *>(src.v_scale_ptr),
      static_cast<uint8_t *>(src.indexer_k_ptr),
      static_cast<uint8_t *>(src.indexer_k_scale_ptr),
      src.attention_projection_input_norm_eps,
      src.attention_projection_post_norm_eps,
      src.scale_k,
      src.scale_v,
      src.attention_projection_ki_dequant_scale,
      src.k_stride_n,
      src.k_scale_page_stride,
      src.v_stride_n,
      src.v_scale_page_stride,
      src.indexer_k_page_stride,
      src.indexer_k_scale_page_stride};
}

template <class Traits>
cudaError_t M3DecodeMegaKernelParamsBuilder<Traits>::build_attention_descs(
    const M3DecodeMegaKernelParams &src, Sm100M3DecodeKernelParams<Traits> &dst) {
  cudaError_t status = build_q_desc(src, dst.tma);
  if (status != cudaSuccess) {
    return status;
  }
  status = build_kv_desc(dst.tma.k, src.k_ptr, src.num_kv_heads, src.k_stride_h, src.k_stride_n,
                         src.total_page_num);
  if (status != cudaSuccess) {
    return status;
  }
  status = build_kv_desc(dst.tma.v, src.v_ptr, src.num_kv_heads, src.v_stride_h, src.v_stride_n,
                         src.total_page_num);
  if (status != cudaSuccess) {
    return status;
  }
  status = build_scale_desc(dst.tma.k_scale, src.k_scale_ptr, src.num_kv_heads,
                            src.k_scale_page_stride, src.total_page_num);
  if (status != cudaSuccess) {
    return status;
  }
  status = build_scale_desc(dst.tma.v_scale, src.v_scale_ptr, src.num_kv_heads,
                            src.v_scale_page_stride, src.total_page_num);
  if (status != cudaSuccess) {
    return status;
  }
  status = build_indexer_k_desc(src, dst.tma);
  if (status != cudaSuccess) {
    return status;
  }
  status = build_indexer_k_scale_desc(src, dst.tma);
  if (status != cudaSuccess) {
    return status;
  }
  status = build_o_desc(src, dst.tma);
  if (status != cudaSuccess) {
    return status;
  }
  status = build_output_projection_descs(src, dst.tma);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaSuccess;
}

template <class Traits>
cudaError_t
M3DecodeMegaKernelParamsBuilder<Traits>::build_mega_atten(const M3DecodeMegaKernelParams &src,
                                                          Sm100M3DecodeKernelParams<Traits> &dst) {
  copy_common_params(src, dst);
  build_attention_projection_params(src, dst);
  return build_attention_descs(src, dst);
}

} // namespace cutlass::m3_decode::collective
