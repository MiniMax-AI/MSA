#pragma once

#include "common/kernel/include/sm100_m3_decode_tma_descriptors.hpp"

namespace cutlass::m3_decode::collective {

template <class Traits>
int M3DecodeMegaKernelParamsBuilder<Traits>::q_tokens_per_batch(
    const M3DecodeMegaKernelParams &src) {
  int const rows_per_batch = src.batch_size * Traits::kHeadGroup;
  if (rows_per_batch <= 0 || src.total_qo_len <= 0 || src.total_qo_len % rows_per_batch != 0) {
    return 0;
  }
  return src.total_qo_len / rows_per_batch;
}

template <class Traits>
int M3DecodeMegaKernelParamsBuilder<Traits>::q_tokens_total(const M3DecodeMegaKernelParams &src) {
  if (src.total_qo_len <= 0 || src.total_qo_len % Traits::kHeadGroup != 0) {
    return 0;
  }
  return src.total_qo_len / Traits::kHeadGroup;
}

template <class Traits>
int M3DecodeMegaKernelParamsBuilder<Traits>::o_tokens_total(const M3DecodeMegaKernelParams &src) {
  if (src.total_qo_len_orig > 0) {
    return src.total_qo_len_orig;
  }
  return q_tokens_total(src);
}

template <class Traits>
cudaError_t M3DecodeMegaKernelParamsBuilder<Traits>::encode_tma(
    CUtensorMap &desc, Sm100M3DecodeTmaDtype dtype, int rank, void *gmem_ptr,
    uint64_t const *global_dim, uint64_t const *global_stride_bytes, uint32_t const *box_dim,
    CUtensorMapSwizzle swizzle, bool unpack4b) {
  if (gmem_ptr == nullptr || rank < 2 || rank > 5) {
    return cudaErrorInvalidValue;
  }

  CUtensorMapDataType data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  if (dtype == Sm100M3DecodeTmaDtype::E2M1) {
    data_type =
        unpack4b ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B : CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
  } else if (dtype == Sm100M3DecodeTmaDtype::BF16) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  }

  uint32_t element_stride[5] = {1, 1, 1, 1, 1};
  CUresult result =
      cuTensorMapEncodeTiled(&desc, data_type, rank, gmem_ptr, global_dim, global_stride_bytes,
                             box_dim, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

template <class Traits>
cudaError_t M3DecodeMegaKernelParamsBuilder<Traits>::build_mxfp8_weight_descs(
    CUtensorMap &weight_desc, CUtensorMap &scale_desc, void *weight_ptr, void *scale_ptr,
    uint64_t problem_m, uint64_t problem_k) {
  constexpr uint64_t kTileM = 128;
  constexpr uint64_t kTileK = 128;
  constexpr uint64_t kScaleTileRows = 32;
  constexpr uint64_t kScaleTileRowBytes = 16;
  constexpr uint64_t kScaleTileBytes = kScaleTileRows * kScaleTileRowBytes;

  uint64_t weight_global_dim[2] = {problem_k, problem_m};
  uint64_t weight_global_stride[1] = {problem_k};
  uint32_t weight_box_dim[2] = {static_cast<uint32_t>(kTileK), static_cast<uint32_t>(kTileM)};
  cudaError_t status =
      encode_tma(weight_desc, Sm100M3DecodeTmaDtype::E4M3, 2, weight_ptr, weight_global_dim,
                 weight_global_stride, weight_box_dim, CU_TENSOR_MAP_SWIZZLE_128B);
  if (status != cudaSuccess) {
    return status;
  }

  uint64_t const scale_tiles = (problem_m / kTileM) * (problem_k / kTileK);
  uint64_t scale_global_dim[3] = {kScaleTileRowBytes, kScaleTileRows, scale_tiles};
  uint64_t scale_global_stride[2] = {kScaleTileRowBytes, kScaleTileBytes};
  uint32_t scale_box_dim[3] = {static_cast<uint32_t>(kScaleTileRowBytes),
                               static_cast<uint32_t>(kScaleTileRows), 1u};
  return encode_tma(scale_desc, Sm100M3DecodeTmaDtype::E4M3, 3, scale_ptr, scale_global_dim,
                    scale_global_stride, scale_box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
}

template <class Traits>
cudaError_t M3DecodeMegaKernelParamsBuilder<Traits>::build_mxfp8_activation_descs(
    CUtensorMap &activation_desc, CUtensorMap &scale_desc, void *activation_ptr, void *scale_ptr,
    uint64_t problem_k) {
  constexpr uint64_t kTileK = 128;
  constexpr uint64_t kTokens = 8;
  uint64_t activation_global_dim[2] = {problem_k, Traits::kMoeQueryTokens};
  uint64_t activation_global_stride[1] = {problem_k};
  uint32_t activation_box_dim[2] = {static_cast<uint32_t>(kTileK), static_cast<uint32_t>(kTokens)};
  cudaError_t status = encode_tma(activation_desc, Sm100M3DecodeTmaDtype::E4M3, 2, activation_ptr,
                                  activation_global_dim, activation_global_stride,
                                  activation_box_dim, CU_TENSOR_MAP_SWIZZLE_128B);
  if (status != cudaSuccess) {
    return status;
  }

  uint64_t scale_global_dim[3] = {16u, Traits::kMoeQueryTokens, problem_k / kTileK};
  uint64_t scale_global_stride[2] = {16u, 512u};
  uint32_t scale_box_dim[3] = {16u, static_cast<uint32_t>(kTokens), 1u};
  return encode_tma(scale_desc, Sm100M3DecodeTmaDtype::E4M3, 3, scale_ptr, scale_global_dim,
                    scale_global_stride, scale_box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
}

template <class Traits>
void M3DecodeMegaKernelParamsBuilder<Traits>::copy_common_params(
    const M3DecodeMegaKernelParams &src, Sm100M3DecodeKernelParams<Traits> &dst) {
  dst.q_ptr = src.q_ptr;
  dst.k_ptr = src.k_ptr;
  dst.v_ptr = src.v_ptr;
  dst.k_scale_ptr = src.k_scale_ptr;
  dst.v_scale_ptr = src.v_scale_ptr;
  dst.indexer_q_ptr = src.indexer_q_ptr;
  dst.indexer_k_ptr = src.indexer_k_ptr;
  dst.indexer_k_scale_ptr = src.indexer_k_scale_ptr;
  dst.indexer_page_table_ptr = src.indexer_page_table_ptr;
  dst.o_ptr = src.o_direct_ptr;
  dst.o_sf_ptr = src.o_sf_ptr;
  dst.output_projection_barrier_ptr = src.output_projection_barrier_ptr;
  dst.tp_unicast_ptr = src.tp_unicast_ptr;
  dst.tp_multicast_ptr = src.tp_multicast_ptr;
  dst.tp_rank = src.tp_rank;
  dst.logical_q_len = src.logical_q_len;
  dst.wait_timeout_ns = src.wait_timeout_ns;
  dst.residual_in_ptr = src.residual_in_ptr;
  dst.residual_out_ptr = src.residual_out_ptr;
  dst.attention_residual_ptr =
      src.attention_residual_ptr != nullptr ? src.attention_residual_ptr : src.residual_out_ptr;
  dst.post_attention_norm_ptr = src.post_attention_norm_ptr;
  dst.next_hidden_ptr = src.next_hidden_ptr != nullptr ? src.next_hidden_ptr : src.residual_out_ptr;
  dst.rms_weight_ptr = src.rms_weight_ptr;
  dst.shared_activation_ptr = src.shared_activation_ptr;
  dst.shared_activation_sf_ptr = src.shared_activation_sf_ptr;
  dst.shared_fc1_w_ptr = src.shared_fc1_w_ptr;
  dst.shared_fc1_w_sf_ptr = src.shared_fc1_w_sf_ptr;
  dst.shared_mid_ptr = src.shared_mid_ptr;
  dst.shared_fc2_w_ptr = src.shared_fc2_w_ptr;
  dst.shared_fc2_w_sf_ptr = src.shared_fc2_w_sf_ptr;
  dst.router_weight_ptr = src.router_weight_ptr;
  dst.router_correction_bias_ptr = src.router_correction_bias_ptr;
  dst.router_topk_idx_ptr = src.router_topk_idx_ptr;
  dst.router_topk_weight_ptr = src.router_topk_weight_ptr;
  dst.routed_active_expert_count_ptr = src.routed_active_expert_count_ptr;
  dst.routed_route_count_ptr = src.routed_route_count_ptr;
  dst.routed_active_expert_ids_ptr = src.routed_active_expert_ids_ptr;
  dst.routed_expert_route_offsets_ptr = src.routed_expert_route_offsets_ptr;
  dst.routed_route_token_ids_ptr = src.routed_route_token_ids_ptr;
  dst.routed_route_topk_ids_ptr = src.routed_route_topk_ids_ptr;
  dst.routed_route_expert_ids_ptr = src.routed_route_expert_ids_ptr;
  dst.routed_route_active_slots_ptr = src.routed_route_active_slots_ptr;
  dst.routed_route_scales_ptr = src.routed_route_scales_ptr;
  dst.routed_activation_ptr = src.routed_activation_ptr;
  dst.routed_activation_sf_ptr = src.routed_activation_sf_ptr;
  dst.routed_fc1_w_ptr = src.routed_fc1_w_ptr;
  dst.routed_fc1_w_sf_ptr = src.routed_fc1_w_sf_ptr;
  dst.routed_fc1_alpha_ptr = src.routed_fc1_alpha_ptr;
  dst.routed_mid_ptr = src.routed_mid_ptr;
  dst.routed_mid_sf_ptr = src.routed_mid_sf_ptr;
  dst.routed_fc2_w_ptr = src.routed_fc2_w_ptr;
  dst.routed_fc2_w_sf_ptr = src.routed_fc2_w_sf_ptr;
  dst.routed_fc2_alpha_ptr = src.routed_fc2_alpha_ptr;
  dst.kv_indices_ptr = src.kv_indices_ptr;
  dst.kv_block_indexes_ptr = src.kv_block_indexes_ptr;
  dst.kv_segment_lens_ptr = src.kv_segment_lens_ptr;
  dst.batch_size = src.batch_size;
  dst.num_kv_heads = src.num_kv_heads;
  dst.num_qo_heads_orig = src.num_qo_heads_orig;
  dst.q_tokens_per_batch = q_tokens_per_batch(src);
  dst.kv_page_stride = src.kv_page_stride;
  dst.kv_block_num = src.kv_block_num;
  dst.scale_v = src.scale_v;
  dst.scale_o = src.o_scale;
  dst.scale_softmax_log2 = src.scale_q * src.scale_k * src.sm_scale * 1.4426950408889634f;
  dst.rms_eps = src.rms_eps;
}

} // namespace cutlass::m3_decode::collective
