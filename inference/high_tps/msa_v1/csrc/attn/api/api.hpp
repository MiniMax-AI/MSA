#pragma once

#include "common/api/types.hpp"

namespace mega_atten::m3_decode {
std::tuple<at::Tensor, at::Tensor> run_m3_decode_mega_atten_impl(
    at::Tensor q, at::Tensor k, at::Tensor v, PlanInfo &plan_info, int64_t topk,
    at::Tensor kv_indices, at::Tensor kv_block_indexes, at::Tensor k_scale_tensor,
    at::Tensor v_scale_tensor, std::optional<at::Tensor> out, std::optional<at::Tensor> out_sf,
    std::optional<float> sm_scale, std::optional<float> q_scale, std::optional<float> k_scale,
    std::optional<float> v_scale, std::optional<float> o_scale, at::Tensor output_projection_weight,
    at::Tensor output_projection_weight_sf, int64_t tp_unicast_ptr, int64_t tp_multicast_ptr,
    int64_t tp_rank, at::Tensor residual_in, at::Tensor attention_residual,
    at::Tensor post_attention_norm, at::Tensor rms_weight, std::optional<at::Tensor> indexer_q,
    std::optional<at::Tensor> indexer_k, std::optional<at::Tensor> indexer_k_scale,
    std::optional<at::Tensor> indexer_page_table,
    AttentionProjectionInfo const &attention_projection, double rms_eps, int64_t wait_timeout_ns,
    bool enable_placement);

} // namespace mega_atten::m3_decode
