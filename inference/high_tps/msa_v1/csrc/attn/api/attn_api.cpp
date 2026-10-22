// m3_decode_api.cpp - C++ implementation of the M3 decode kernel API.
//
// Kernel calls go through tvm_ffi; tensor management uses ATen (libtorch).

#include <ATen/DLConvertor.h>
#include <ATen/cuda/CUDAContext.h>
#include <Python.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <optional>
#include <utility>

#include "attn/api/api.hpp"
#include "attn/api/validation.hpp"
#include "attn/kernel/include/m3_decode_output_projection_handoff.hpp"
#include "common/api/support.hpp"
#include "common/api/validation.hpp"
#include "common/kernel/include/tp4_nvls_layout.hpp"
#include "common/runtime/m3_decode_variant_manager.hpp"

namespace mega_atten::m3_decode {
std::tuple<at::Tensor, at::Tensor> run_m3_decode_mega_atten_impl(
    at::Tensor q, at::Tensor k, at::Tensor v, PlanInfo &plan, int64_t topk, at::Tensor kv_indices,
    at::Tensor kv_block_indexes, at::Tensor k_scale_tensor, at::Tensor v_scale_tensor,
    std::optional<at::Tensor> out, std::optional<at::Tensor> out_sf, std::optional<float> sm_scale,
    std::optional<float> q_scale_opt, std::optional<float> k_scale_opt,
    std::optional<float> v_scale_opt, std::optional<float> o_scale_opt,
    at::Tensor output_projection_weight, at::Tensor output_projection_weight_sf,
    int64_t tp_unicast_ptr, int64_t tp_multicast_ptr, int64_t tp_rank, at::Tensor residual_in,
    at::Tensor attention_residual, at::Tensor post_attention_norm, at::Tensor rms_weight,
    std::optional<at::Tensor> indexer_q, std::optional<at::Tensor> indexer_k,
    std::optional<at::Tensor> indexer_k_scale, std::optional<at::Tensor> indexer_page_table,
    AttentionProjectionInfo const &attention_projection, double rms_eps, int64_t wait_timeout_ns,
    bool enable_placement) {
  TORCH_CHECK(topk == 16, "m3_decode_mega_atten requires sparse attention topK=16; got ", topk);
  TORCH_CHECK(wait_timeout_ns >= 0, "wait_timeout_ns must be non-negative; got ", wait_timeout_ns);
  TORCH_CHECK(indexer_q.has_value() && indexer_q->defined() && indexer_k.has_value() &&
                  indexer_k->defined() && indexer_k_scale.has_value() &&
                  indexer_k_scale->defined() && indexer_page_table.has_value() &&
                  indexer_page_table->defined(),
              "m3_decode_mega_atten requires all Indexer operands");
  TORCH_CHECK(attention_projection.enabled(),
              "m3_decode_mega_atten requires Attention Projection operands");

  int64_t const nnz_qo = q.size(0);
  int const num_qo_heads = q.size(1);
  int const head_dim_qk = q.size(2);
  int const num_kv_heads = k.size(1);
  int const page_size = k.size(2);
  int const batch_size = plan.kv_segment_lens.size(0);
  int64_t const kv_page_stride = kv_indices.size(1);

  TORCH_CHECK(indexer_q->device() == q.device() && is_fp8_byte_storage_tensor(*indexer_q) &&
                  indexer_q->is_contiguous() && indexer_q->dim() == 2 && indexer_q->size(0) == 8 &&
                  indexer_q->size(1) == 128,
              "indexer_q must be contiguous E4M3/uint8 [8, 128]");
  validate_paged_cache_args(q, k, v, k_scale_tensor, v_scale_tensor, *indexer_k, *indexer_k_scale);
  TORCH_CHECK(
      indexer_page_table->device() == q.device() && indexer_page_table->scalar_type() == at::kInt &&
          indexer_page_table->is_contiguous() && indexer_page_table->numel() >= kv_page_stride,
      "indexer_page_table must cover every logical KV page");

  validate_output_projection_args(q, output_projection_weight, output_projection_weight_sf);
  validate_attention_projection_args(q, attention_projection);
  validate_tp_post_attention_norm_args(q, tp_unicast_ptr, tp_multicast_ptr, residual_in,
                                       attention_residual, rms_weight, rms_eps);
  validate_hidden_state(post_attention_norm, q.device(), "post_attention_norm");
  TORCH_CHECK(!tensors_overlap(attention_residual, post_attention_norm) &&
                  !tensors_overlap(residual_in, post_attention_norm) &&
                  !tensors_overlap(rms_weight, post_attention_norm),
              "MegaAtten boundary outputs and inputs must be disjoint");
  TORCH_CHECK(tp_rank >= 0 && tp_rank < tp4_nvls::kWorldSize,
              "m3_decode_mega_atten requires tp_rank in [0, 4)");

  at::Tensor o =
      out.has_value() && out->defined()
          ? *out
          : torch::empty({nnz_qo, num_qo_heads, head_dim_qk},
                         torch::TensorOptions().dtype(at::kFloat8_e4m3fn).device(q.device()));
  int64_t sf_size = compute_mxfp8_sf_size(nnz_qo, static_cast<int64_t>(num_qo_heads) * head_dim_qk);
  sf_size = std::max<int64_t>(sf_size, 16 * 512);
  at::Tensor o_sf =
      out_sf.has_value() && out_sf->defined()
          ? *out_sf
          : torch::empty({sf_size}, torch::TensorOptions().dtype(torch::kUInt8).device(q.device()));
  TORCH_CHECK(is_fp8_byte_storage_tensor(o) && o.is_contiguous() && o.sizes() == q.sizes(),
              "MegaAtten quantized output must match q shape and be E4M3/uint8");
  TORCH_CHECK(o_sf.scalar_type() == at::kByte && o_sf.is_contiguous() && o_sf.numel() >= sf_size,
              "MegaAtten output scale storage is too small");

  bool const q8kv4 =
      is_fp8_byte_storage_tensor(q) && k_scale_tensor.defined() && v_scale_tensor.defined();
  bool const scale_layout_ok = k_scale_tensor.dim() == 4 && v_scale_tensor.dim() == 4;
  bool kv_len_ok = !plan.kv_lens_cpu.empty();
  int64_t const maximum_kv_len = kv_page_stride * page_size;
  for (int32_t kv_len : plan.kv_lens_cpu) {
    kv_len_ok &= kv_len >= 1 && kv_len <= maximum_kv_len;
  }
  TORCH_CHECK(q8kv4 && page_size == 128 && head_dim_qk == 128 && batch_size == 1 && nnz_qo == 8 &&
                  kv_len_ok && scale_layout_ok && num_qo_heads == 16 && num_kv_heads == 1 &&
                  plan.kv_block_num == 16 && kv_page_stride > 0 &&
                  kv_page_stride <= tp4_nvls::kIndexerMaxPages,
              "input does not match the fixed m3_decode_mega_atten domain");

  float const sm = sm_scale.value_or(1.0f / std::sqrt(static_cast<float>(head_dim_qk)));
  auto variant = M3DecodeVariantManager::instance().get_m3_decode_mega_atten_variant(
      wait_timeout_ns > 0, enable_placement);
  int64_t const stream = reinterpret_cast<int64_t>(at::cuda::getCurrentCUDAStream().stream());
  variant(
      torch_to_tvm(q), torch_to_tvm(k), torch_to_tvm(v), torch_to_tvm(plan.kv_segment_lens),
      torch_to_tvm(o), torch_to_tvm(o_sf), static_cast<double>(sm),
      static_cast<double>(q_scale_opt.value_or(1.0f)),
      static_cast<double>(k_scale_opt.value_or(1.0f)),
      static_cast<double>(v_scale_opt.value_or(1.0f)),
      static_cast<double>(o_scale_opt.value_or(1.0f)), torch_to_tvm(kv_indices), kv_page_stride,
      tensor_or_null(kv_block_indexes), torch_to_tvm(k_scale_tensor), torch_to_tvm(v_scale_tensor),
      torch_to_tvm(output_projection_weight), torch_to_tvm(output_projection_weight_sf),
      torch_to_tvm(plan.output_projection_barrier), tp_unicast_ptr, tp_multicast_ptr, tp_rank,
      torch_to_tvm(residual_in), torch_to_tvm(attention_residual),
      torch_to_tvm(post_attention_norm), torch_to_tvm(rms_weight), torch_to_tvm(*indexer_q),
      torch_to_tvm(*indexer_k), torch_to_tvm(*indexer_k_scale), torch_to_tvm(*indexer_page_table),
      torch_to_tvm(attention_projection.input_norm_weight),
      torch_to_tvm(attention_projection.weight), torch_to_tvm(attention_projection.weight_sf),
      torch_to_tvm(attention_projection.workspace_data),
      torch_to_tvm(attention_projection.workspace_scale),
      torch_to_tvm(attention_projection.grid_epoch),
      torch_to_tvm(attention_projection.special_partial),
      torch_to_tvm(attention_projection.special_epochs),
      torch_to_tvm(attention_projection.positions), torch_to_tvm(attention_projection.slot_mapping),
      torch_to_tvm(attention_projection.q_norm_weight),
      torch_to_tvm(attention_projection.k_norm_weight),
      torch_to_tvm(attention_projection.qi_norm_weight),
      torch_to_tvm(attention_projection.ki_norm_weight),
      torch_to_tvm(attention_projection.cos_sin_cache), attention_projection.input_norm_eps,
      attention_projection.post_norm_eps, attention_projection.ki_dequant_scale, rms_eps,
      wait_timeout_ns, stream);
  return {o, o_sf};
}

} // namespace mega_atten::m3_decode
