// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

// Internal C++ API for Q8KV4 paged sparse decode attention.
#pragma once

#include <torch/torch.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace fmha_sm100::decode_q8kv4 {

// Widest TopK list the kernel accepts; the JIT instantiates the kernel with the same bound
// (jit.MAX_TOPK) and its per-item tail mask holds one bit per page.
constexpr int kMaxSparseTopK = 64;
// Largest block-scale staging shift (an E4M3 exponent offset); the JIT compiles one kernel per
// shift, see Traits::kBlockScaleShift.
constexpr int kMaxBlockScaleShift = 7;

struct PlanData {
  at::Tensor packed_work_range, packed_work_info;
  at::Tensor kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices;
  int num_kv_splits = 1;
  at::Tensor workspace_o, workspace_lse, num_kv_splits_per_row;
  at::Tensor kv_split_count,
      merge_counter; // balanced schedule: per-work segment count, per-item counters
  at::Tensor qo_segment_offsets, kv_segment_offsets;
  at::Tensor qo_segment_lens, kv_segment_lens, qo_offset;
  int max_qo_len = 0;
  int qo_tile_size = 0;
  int pack_factor = 1, orig_num_qo_heads = -1;
  int q_tokens_per_batch = 0;
  bool qo_len_uniform = false;
  at::Tensor cute_workspace_buffer;
  std::vector<at::Tensor> host_staging_buffers;
  int kv_block_num = 0;
  // Balanced (stream-K style) schedule: items of the fractional tail wave are cut into
  // KV segments spread across all CTAs; segments merge through the split-KV workspace.
  bool stream_k = false;
  int64_t merge_item_base = 0;
};

struct PlanInfo : PlanData {
  PlanInfo() = default;
  PlanInfo(const PlanInfo &) = delete;
  PlanInfo &operator=(const PlanInfo &) = delete;
  PlanInfo(PlanInfo &&) noexcept = default;
  PlanInfo &operator=(PlanInfo &&) noexcept = default;
};

PlanInfo _make_decode_plan_impl(at::Tensor qo_segment_lens, at::Tensor kv_segment_lens,
                                int num_qo_heads, int num_kv_heads = -1,
                                std::optional<at::Tensor> qo_offset = std::nullopt,
                                int num_kv_splits = -1, int page_size = -1, int kv_block_num = -1,
                                int usable_SM_count = -1, std::optional<int> device = std::nullopt,
                                const std::string &split_mode = "streamk");

at::Tensor _run_decode_impl(at::Tensor q, at::Tensor k, at::Tensor v, PlanInfo &plan_info,
                            at::Tensor seq_lens, at::Tensor kv_indices, at::Tensor kv_indptr,
                            at::Tensor topk_indices, at::Tensor k_scale, at::Tensor v_scale,
                            at::Tensor out, float sm_scale, at::Tensor k_global_scale,
                            at::Tensor v_global_scale, int block_scale_shift);

std::unique_ptr<PlanInfo> make_decode_plan(at::Tensor qo_segment_lens, at::Tensor kv_segment_lens,
                                           int num_qo_heads, int num_kv_heads, int num_kv_splits,
                                           int page_size, int topk, int usable_sm_count,
                                           std::optional<int> device = std::nullopt,
                                           const std::string &split_mode = "streamk");

// kv_indices is the flat physical-page list and kv_indptr [batch + 1] each request's base into
// it. K/V data ([pages, heads, 128, 64] uint8) and block scales ([pages, heads, 128, 8] E4M3 as
// uint8) are strided views: token rows contiguous, page and head strides free (multiples of 16
// bytes), so packed pages holding all heads' data blocks followed by their scale blocks need no
// copy. V scale blocks are in token-quad order, see FMHACutlassSM100Params.
// k_global_scale / v_global_scale are one-element fp32 CUDA tensors read by the kernels (value =
// code x block_scale x global_scale); block_scale_shift selects the kernel that divides the block
// scales by 2^shift before the dequant product (0: products already fit E4M3; 3: block scales use
// the full E4M3 range, the vLLM / TransformerEngine convention).
at::Tensor run_decode(at::Tensor q, at::Tensor k, at::Tensor v, PlanInfo &plan_info,
                      at::Tensor seq_lens, at::Tensor kv_indices, at::Tensor kv_indptr,
                      at::Tensor topk_indices, at::Tensor k_scale, at::Tensor v_scale,
                      at::Tensor out, float sm_scale, at::Tensor k_global_scale,
                      at::Tensor v_global_scale, int block_scale_shift);

} // namespace fmha_sm100::decode_q8kv4
