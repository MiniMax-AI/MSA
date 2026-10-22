#pragma once
#include <cuda_runtime.h>

#include <cstdint>

struct M3DecodeMegaKernelParams {
  void *q_ptr;
  void *k_ptr;
  void *v_ptr;
  int *kv_segment_lens_ptr;
  float sm_scale;
  float scale_q;
  float scale_k;
  float scale_v;
  float o_scale;
  int num_qo_heads;
  int num_kv_heads;
  int head_dim_qk;
  int head_dim_vo;
  int q_stride_n;
  int q_stride_h;
  // K/V byte strides. k/v use [pages, heads, 128, 64] packed rows; the page stride may exceed the
  // dense page size so K/V can live inside a larger per-page slab.
  int64_t k_stride_n;
  int k_stride_h;
  int64_t v_stride_n;
  int v_stride_h;
  int batch_size;
  int total_qo_len;
  cudaStream_t stream;
  int *kv_indices_ptr;
  int kv_page_stride;
  int total_page_num;
  int *kv_block_indexes_ptr;
  int kv_block_num;
  // Optional fused Indexer operands. The fused sparse variant generates
  // attention TopK page ids inside the MegaKernel.
  void *indexer_q_ptr = nullptr;
  void *indexer_k_ptr = nullptr;
  void *indexer_k_scale_ptr = nullptr;
  // Byte distance between consecutive physical pages of each cache component. Inside a page the
  // data rows (64 B) and scale rows (8 B) are dense.
  int64_t indexer_k_page_stride = 0;
  int64_t indexer_k_scale_page_stride = 0;
  int64_t k_scale_page_stride = 0;
  int64_t v_scale_page_stride = 0;
  int *indexer_page_table_ptr = nullptr;
  // Optional fused Attention Projection operands. The original layer input aliases
  // residual_in_ptr; Q/QI/K/V/KI outputs alias the existing attention and
  // Indexer operands.
  void *attention_projection_input_norm_weight_ptr = nullptr;
  void *attention_projection_w_ptr = nullptr;
  void *attention_projection_w_sf_ptr = nullptr;
  void *attention_projection_workspace_data_ptr = nullptr;
  void *attention_projection_workspace_scale_ptr = nullptr;
  void *attention_projection_grid_epoch_ptr = nullptr;
  void *attention_projection_special_partial_ptr = nullptr;
  void *attention_projection_special_epochs_ptr = nullptr;
  void *attention_projection_positions_ptr = nullptr;
  void *attention_projection_slot_mapping_ptr = nullptr;
  void *attention_projection_q_norm_weight_ptr = nullptr;
  void *attention_projection_k_norm_weight_ptr = nullptr;
  void *attention_projection_qi_norm_weight_ptr = nullptr;
  void *attention_projection_ki_norm_weight_ptr = nullptr;
  void *attention_projection_cos_sin_cache_ptr = nullptr;
  float attention_projection_input_norm_eps = 0.0f;
  float attention_projection_post_norm_eps = 0.0f;
  float attention_projection_k_dequant_scale = 1.0f;
  float attention_projection_v_dequant_scale = 1.0f;
  float attention_projection_ki_dequant_scale = 1.0f;
  int total_qo_len_orig = 0;
  int logical_q_len = 8;
  uint64_t wait_timeout_ns = 0;
  void *o_direct_ptr = nullptr;
  void *o_sf_ptr = nullptr;
  int num_qo_heads_orig = 0;
  // NVFP4 KV: per-group scale factors for fp4→fp8 dequantization.
  // When k_scale_ptr != nullptr, k_ptr/v_ptr point to fp4x2 (uint8) data.
  // Paged scale shape: (total_pages, num_kv_heads, page_size, head_dim/16),
  // dtype=fp8_e4m3.
  void *k_scale_ptr = nullptr;
  void *v_scale_ptr = nullptr;
  // Output Projection inputs and TP4 output payload.
  void *output_projection_w_ptr = nullptr;
  void *output_projection_w_sf_ptr = nullptr;
  void *output_projection_barrier_ptr = nullptr;
  // TP4 AllReduce + residual RMSNorm tail.
  void *tp_unicast_ptr = nullptr;
  void *tp_multicast_ptr = nullptr;
  int tp_rank = 0;
  void *residual_in_ptr = nullptr;
  void *residual_out_ptr = nullptr;
  void *attention_residual_ptr = nullptr;
  void *post_attention_norm_ptr = nullptr;
  void *next_hidden_ptr = nullptr;
  void *rms_weight_ptr = nullptr;
  // Shared-expert RMSNorm output in contiguous E4M3 plus blocked E8M0 scale bytes.
  void *shared_activation_ptr = nullptr;
  void *shared_activation_sf_ptr = nullptr;
  // Shared-expert FC1/FC2 inputs and the fused FC1 handoff workspace.
  void *shared_fc1_w_ptr = nullptr;
  void *shared_fc1_w_sf_ptr = nullptr;
  void *shared_mid_ptr = nullptr;
  void *shared_fc2_w_ptr = nullptr;
  void *shared_fc2_w_sf_ptr = nullptr;
  // FP32 Router/Gate inputs and final Top-4 outputs.
  void *router_weight_ptr = nullptr;
  void *router_correction_bias_ptr = nullptr;
  void *router_topk_idx_ptr = nullptr;
  void *router_topk_weight_ptr = nullptr;
  // Routed-expert compact schedule and token-major NVFP4 activation.
  void *routed_active_expert_count_ptr = nullptr;
  void *routed_route_count_ptr = nullptr;
  void *routed_active_expert_ids_ptr = nullptr;
  void *routed_expert_route_offsets_ptr = nullptr;
  void *routed_route_token_ids_ptr = nullptr;
  void *routed_route_topk_ids_ptr = nullptr;
  void *routed_route_expert_ids_ptr = nullptr;
  void *routed_route_active_slots_ptr = nullptr;
  void *routed_route_scales_ptr = nullptr;
  void *routed_activation_ptr = nullptr;
  void *routed_activation_sf_ptr = nullptr;
  // Routed Expert FC1 production operands and its padded-group output.
  void *routed_fc1_w_ptr = nullptr;
  void *routed_fc1_w_sf_ptr = nullptr;
  void *routed_fc1_alpha_ptr = nullptr;
  void *routed_mid_ptr = nullptr;
  void *routed_mid_sf_ptr = nullptr;
  // Routed Expert FC2 production operands. The final BF16 output aliases residual_out.
  void *routed_fc2_w_ptr = nullptr;
  void *routed_fc2_w_sf_ptr = nullptr;
  void *routed_fc2_alpha_ptr = nullptr;
  float rms_eps = 0.0f;
};
