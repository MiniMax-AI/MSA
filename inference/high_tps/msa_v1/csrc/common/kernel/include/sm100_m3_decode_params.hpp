#pragma once

#include <cuda.h>

#include "attn/kernel/include/sm100_m3_decode_attention_projection_config.hpp"
#include "common/kernel/include/sm100_m3_decode_traits.hpp"

namespace cutlass::m3_decode::collective {

struct Sm100M3DecodeAttentionTmaDescriptors {
  CUtensorMap q{};
  CUtensorMap k{};
  CUtensorMap v{};
  CUtensorMap indexer_k{};
  CUtensorMap indexer_k_scale{};
  CUtensorMap k_scale{};
  CUtensorMap v_scale{};
  CUtensorMap o{};
  CUtensorMap output_projection_w{};
  CUtensorMap output_projection_w_sf{};
  CUtensorMap output_projection_o{};
  CUtensorMap output_projection_o_sf_padded{};
};

struct Sm100M3DecodeMoeTmaDescriptors {
  CUtensorMap shared_fc1_w{};
  CUtensorMap shared_fc1_w_sf{};
  CUtensorMap shared_activation{};
  CUtensorMap shared_activation_sf{};
  CUtensorMap shared_fc2_w{};
  CUtensorMap shared_fc2_w_sf{};
  CUtensorMap shared_mid{};
  CUtensorMap shared_mid_sf{};
  CUtensorMap routed_fc1_w{};
  CUtensorMap routed_fc1_w_sf{};
  CUtensorMap routed_activation{};
  CUtensorMap routed_fc2_w{};
  CUtensorMap routed_fc2_w_sf{};
  CUtensorMap routed_mid{};
  CUtensorMap routed_mid_sf{};
};

template <Sm100M3DecodeKernelPath Path> struct Sm100M3DecodeTmaDescriptors;

template <>
struct Sm100M3DecodeTmaDescriptors<Sm100M3DecodeKernelPath::kMonolithic>
    : Sm100M3DecodeAttentionTmaDescriptors, Sm100M3DecodeMoeTmaDescriptors {};

template <>
struct Sm100M3DecodeTmaDescriptors<Sm100M3DecodeKernelPath::kMegaAtten>
    : Sm100M3DecodeAttentionTmaDescriptors {};

template <>
struct Sm100M3DecodeTmaDescriptors<Sm100M3DecodeKernelPath::kMegaMoe>
    : Sm100M3DecodeMoeTmaDescriptors {};

template <Sm100M3DecodeKernelPath Path> struct Sm100M3DecodeAttentionProjectionParams {};

template <> struct Sm100M3DecodeAttentionProjectionParams<Sm100M3DecodeKernelPath::kMonolithic> {
  attention_projection::KernelParams attention_projection;
};

template <> struct Sm100M3DecodeAttentionProjectionParams<Sm100M3DecodeKernelPath::kMegaAtten> {
  attention_projection::KernelParams attention_projection;
};

template <class Traits>
struct Sm100M3DecodeKernelParams : Sm100M3DecodeAttentionProjectionParams<Traits::kKernelPath> {
  using TmaDescriptors = Sm100M3DecodeTmaDescriptors<Traits::kKernelPath>;

  TmaDescriptors tma;

  void *q_ptr = nullptr;
  void *k_ptr = nullptr;
  void *v_ptr = nullptr;
  void *k_scale_ptr = nullptr;
  void *v_scale_ptr = nullptr;
  void *indexer_q_ptr = nullptr;
  void *indexer_k_ptr = nullptr;
  void *indexer_k_scale_ptr = nullptr;
  int const *indexer_page_table_ptr = nullptr;
  void *o_ptr = nullptr;
  void *o_sf_ptr = nullptr;
  void *output_projection_barrier_ptr = nullptr;
  void *tp_unicast_ptr = nullptr;
  void *tp_multicast_ptr = nullptr;
  int tp_rank = 0;
  int logical_q_len = 8;
  // Every cross-CTA or cross-rank wait traps after this many ns from launch start; 0 = never.
  uint64_t wait_timeout_ns = 0;
  void const *residual_in_ptr = nullptr;
  void *residual_out_ptr = nullptr;
  void *attention_residual_ptr = nullptr;
  void *post_attention_norm_ptr = nullptr;
  void *next_hidden_ptr = nullptr;
  void const *rms_weight_ptr = nullptr;
  void *shared_activation_ptr = nullptr;
  void *shared_activation_sf_ptr = nullptr;
  void const *shared_fc1_w_ptr = nullptr;
  void const *shared_fc1_w_sf_ptr = nullptr;
  void *shared_mid_ptr = nullptr;
  void const *shared_fc2_w_ptr = nullptr;
  void const *shared_fc2_w_sf_ptr = nullptr;
  void const *router_weight_ptr = nullptr;
  void const *router_correction_bias_ptr = nullptr;
  void *router_topk_idx_ptr = nullptr;
  void *router_topk_weight_ptr = nullptr;
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
  void const *routed_fc1_w_ptr = nullptr;
  void const *routed_fc1_w_sf_ptr = nullptr;
  void const *routed_fc1_alpha_ptr = nullptr;
  void *routed_mid_ptr = nullptr;
  void *routed_mid_sf_ptr = nullptr;
  void const *routed_fc2_w_ptr = nullptr;
  void const *routed_fc2_w_sf_ptr = nullptr;
  void const *routed_fc2_alpha_ptr = nullptr;

  int const *kv_indices_ptr = nullptr;
  int const *kv_block_indexes_ptr = nullptr;
  int const *kv_segment_lens_ptr = nullptr;

  int batch_size = 0;
  int num_kv_heads = 0;
  int num_qo_heads_orig = 0;
  int q_tokens_per_batch = 1;
  int kv_page_stride = 0;
  int kv_block_num = 0;
  float scale_v = 1.0f;
  float scale_o = 1.0f;
  float scale_softmax_log2 = 1.0f;
  float rms_eps = 0.0f;
};

} // namespace cutlass::m3_decode::collective
