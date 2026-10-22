#pragma once

#include "common/api/types.hpp"

namespace mega_atten::m3_decode {
void validate_output_projection_args(at::Tensor const &q,
                                     at::Tensor const &output_projection_weight,
                                     at::Tensor const &output_projection_weight_sf);

void validate_attention_projection_args(at::Tensor const &q,
                                        AttentionProjectionInfo const &attention_projection);

void validate_tp_post_attention_norm_args(at::Tensor const &q, int64_t tp_unicast_ptr,
                                          int64_t tp_multicast_ptr, at::Tensor const &residual_in,
                                          at::Tensor const &residual_out,
                                          at::Tensor const &rms_weight, double rms_eps);

// Validates the paged NVFP4 K/V/KI caches written and read by the fused decode kernels.
// Every component is [P, 1, 128, row] (KI and its scale may also be [P, 128, row]) with dense
// rows inside a page and a page stride >= dense page bytes that is a multiple of 16 bytes
// (32 bytes, with a 32-byte-aligned base, for K/V data).
void validate_paged_cache_args(at::Tensor const &q, at::Tensor const &k, at::Tensor const &v,
                               at::Tensor const &k_scale, at::Tensor const &v_scale,
                               at::Tensor const &indexer_k, at::Tensor const &indexer_k_scale);

} // namespace mega_atten::m3_decode
