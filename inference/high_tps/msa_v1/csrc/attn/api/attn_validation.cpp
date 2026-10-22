#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>

#include "attn/api/validation.hpp"
#include "attn/kernel/include/m3_decode_output_projection_handoff.hpp"
#include "common/api/validation.hpp"
#include "common/kernel/include/tp4_nvls_layout.hpp"

namespace mega_atten::m3_decode {

void validate_output_projection_args(const at::Tensor &q,
                                     const at::Tensor &output_projection_weight,
                                     const at::Tensor &output_projection_weight_sf) {
  TORCH_CHECK(output_projection_weight.defined(),
              "m3_decode_megakernel requires output_projection_weight");
  TORCH_CHECK(output_projection_weight_sf.defined(),
              "m3_decode_megakernel requires output_projection_weight_sf");

  at::Tensor const &w = output_projection_weight;
  at::Tensor const &w_sf = output_projection_weight_sf;
  TORCH_CHECK(w.device() == q.device() && w_sf.device() == q.device(),
              "Output Projection tensors must be on the same device as q");
  TORCH_CHECK(is_fp8_byte_storage_tensor(w),
              "Output Projection weight must be float8_e4m3fn or uint8 storage, got ",
              w.scalar_type());
  TORCH_CHECK(w_sf.scalar_type() == torch::kUInt8,
              "Output Projection weight scale factors must be uint8, got ", w_sf.scalar_type());

  constexpr int64_t kOutputProjectionM = mega_atten::m3_decode::output_projection::kProblemM;
  constexpr int64_t kOutputProjectionK = mega_atten::m3_decode::output_projection::kProblemK;
  TORCH_CHECK(w.dim() == 2 && w.size(0) == kOutputProjectionM && w.size(1) == kOutputProjectionK,
              "Output Projection weight must have shape [", kOutputProjectionM, ", ",
              kOutputProjectionK, "], got ", w.sizes());
  TORCH_CHECK(w.stride(1) == 1, "Output Projection weight must be K-contiguous");
  int64_t const required_sf_size = compute_mxfp8_sf_size(kOutputProjectionM, kOutputProjectionK);
  TORCH_CHECK(w_sf.numel() >= required_sf_size,
              "Output Projection weight scale-factor tensor is too small: got ", w_sf.numel(),
              ", need at least ", required_sf_size);
}

void validate_attention_projection_args(const at::Tensor &q,
                                        const AttentionProjectionInfo &attention_projection) {
  if (!attention_projection.enabled()) {
    return;
  }
  constexpr int64_t kTokens = 8;
  constexpr int64_t kHidden = 6144;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kOutput = 2560;
  constexpr int64_t kScaleGroups = kHidden / 32;
  at::Device const device = q.device();

  auto require_device_contiguous = [&](const at::Tensor &tensor, char const *name) {
    TORCH_CHECK(tensor.defined(), "fused Attention Projection requires ", name);
    TORCH_CHECK(tensor.device() == device, name, " must be on the same device as q");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  };
  for (auto const &item : std::initializer_list<std::pair<at::Tensor const *, char const *>>{
           {&attention_projection.input_norm_weight, "attention_projection_input_norm_weight"},
           {&attention_projection.weight, "attention_projection_weight"},
           {&attention_projection.weight_sf, "attention_projection_weight_sf"},
           {&attention_projection.workspace_data, "attention_projection_workspace_data"},
           {&attention_projection.workspace_scale, "attention_projection_workspace_scale"},
           {&attention_projection.grid_epoch, "attention_projection_grid_epoch"},
           {&attention_projection.special_partial, "attention_projection_special_partial"},
           {&attention_projection.special_epochs, "attention_projection_special_epochs"},
           {&attention_projection.positions, "attention_projection_positions"},
           {&attention_projection.slot_mapping, "attention_projection_slot_mapping"},
           {&attention_projection.q_norm_weight, "attention_projection_q_norm_weight"},
           {&attention_projection.k_norm_weight, "attention_projection_k_norm_weight"},
           {&attention_projection.qi_norm_weight, "attention_projection_qi_norm_weight"},
           {&attention_projection.ki_norm_weight, "attention_projection_ki_norm_weight"},
           {&attention_projection.cos_sin_cache, "attention_projection_cos_sin_cache"}}) {
    require_device_contiguous(*item.first, item.second);
  }

  TORCH_CHECK(attention_projection.input_norm_weight.scalar_type() == at::kBFloat16 &&
                  attention_projection.input_norm_weight.numel() == kHidden,
              "attention_projection_input_norm_weight must be BF16 [6144]");
  TORCH_CHECK(is_fp8_byte_storage_tensor(attention_projection.weight) &&
                  attention_projection.weight.dim() == 2 &&
                  attention_projection.weight.size(0) == kOutput &&
                  attention_projection.weight.size(1) == kHidden,
              "attention_projection_weight must be E4M3/uint8 [2560, 6144]");
  TORCH_CHECK(attention_projection.weight_sf.scalar_type() == at::kByte &&
                  attention_projection.weight_sf.numel() == kOutput * kScaleGroups,
              "attention_projection_weight_sf must contain 491520 uint8 bytes");
  TORCH_CHECK(attention_projection.workspace_data.scalar_type() == at::kByte &&
                  attention_projection.workspace_data.numel() == kTokens * kHidden,
              "attention_projection_workspace_data must contain 49152 uint8 bytes");
  TORCH_CHECK(attention_projection.workspace_scale.scalar_type() == at::kByte &&
                  attention_projection.workspace_scale.numel() == kTokens * kScaleGroups,
              "attention_projection_workspace_scale must contain 1536 uint8 bytes");
  TORCH_CHECK(attention_projection.grid_epoch.scalar_type() == at::kInt &&
                  attention_projection.grid_epoch.numel() >= 32,
              "attention_projection_grid_epoch must contain at least 32 int32 values");
  TORCH_CHECK(attention_projection.special_partial.scalar_type() == at::kFloat &&
                  attention_projection.special_partial.numel() == 4 * 4 * kTokens * kHeadDim,
              "attention_projection_special_partial must contain 16384 FP32 values");
  TORCH_CHECK(attention_projection.special_epochs.scalar_type() == at::kInt &&
                  attention_projection.special_epochs.numel() >= 4 * 32,
              "attention_projection_special_epochs must contain at least 128 int32 values");
  TORCH_CHECK(attention_projection.positions.scalar_type() == at::kLong &&
                  attention_projection.positions.numel() == kTokens &&
                  attention_projection.slot_mapping.scalar_type() == at::kLong &&
                  attention_projection.slot_mapping.numel() == kTokens,
              "attention_projection positions and slot_mapping must be int64 [8]");
  for (at::Tensor const *weight :
       {&attention_projection.q_norm_weight, &attention_projection.k_norm_weight,
        &attention_projection.qi_norm_weight, &attention_projection.ki_norm_weight}) {
    TORCH_CHECK(weight->scalar_type() == at::kBFloat16 && weight->numel() == kHeadDim,
                "Attention Projection Q/K/QI/KI norm weights must be BF16 [128]");
  }
  TORCH_CHECK(attention_projection.cos_sin_cache.scalar_type() == at::kBFloat16 &&
                  attention_projection.cos_sin_cache.dim() == 2 &&
                  attention_projection.cos_sin_cache.size(1) == 64,
              "attention_projection_cos_sin_cache must be BF16 [max_position, 64]");
  TORCH_CHECK(std::isfinite(attention_projection.input_norm_eps) &&
                  attention_projection.input_norm_eps > 0.0 &&
                  std::isfinite(attention_projection.post_norm_eps) &&
                  attention_projection.post_norm_eps > 0.0 &&
                  std::isfinite(attention_projection.ki_dequant_scale) &&
                  attention_projection.ki_dequant_scale > 0.0,
              "Attention Projection epsilons and KI dequant scale must be finite and positive");
}

void validate_tp_post_attention_norm_args(const at::Tensor &q, int64_t tp_unicast_ptr,
                                          int64_t tp_multicast_ptr, const at::Tensor &residual_in,
                                          const at::Tensor &residual_out,
                                          const at::Tensor &rms_weight, double rms_eps) {
  TORCH_CHECK(tp_unicast_ptr != 0 && tp_multicast_ptr != 0,
              "TP4 Post-Attention Norm requires non-null NVLS pointers");
  TORCH_CHECK((tp_unicast_ptr & 15) == 0 && (tp_multicast_ptr & 15) == 0,
              "TP4 workspace pointers must be 16-byte aligned");
  TORCH_CHECK(residual_in.defined() && residual_out.defined() && rms_weight.defined(),
              "TP4 Post-Attention Norm requires residual_in, residual_out, and rms_weight");
  TORCH_CHECK(std::isfinite(rms_eps) && rms_eps > 0.0,
              "RMSNorm epsilon must be finite and positive");
  constexpr int64_t kTokenCount = mega_atten::m3_decode::output_projection::kProblemN;

  auto validate_matrix = [&](at::Tensor const &tensor, char const *name) {
    TORCH_CHECK(tensor.device() == q.device(), name, " must be on the same device as q");
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be BF16");
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == kTokenCount &&
                    tensor.size(1) == mega_atten::m3_decode::output_projection::kProblemM &&
                    tensor.is_contiguous(),
                name, " must be contiguous [", kTokenCount, ", ",
                mega_atten::m3_decode::output_projection::kProblemM, "]");
  };
  validate_matrix(residual_in, "residual_in");
  validate_matrix(residual_out, "residual_out");
  TORCH_CHECK(rms_weight.device() == q.device() && rms_weight.scalar_type() == at::kBFloat16 &&
                  rms_weight.dim() == 1 &&
                  rms_weight.numel() == mega_atten::m3_decode::output_projection::kProblemM &&
                  rms_weight.is_contiguous(),
              "rms_weight must be contiguous BF16 [",
              mega_atten::m3_decode::output_projection::kProblemM, "] on the same device as q");
  bool const residual_in_place = residual_in.data_ptr() == residual_out.data_ptr();
  TORCH_CHECK(residual_in_place || !tensors_overlap(residual_in, residual_out),
              "residual_in and residual_out must be identical or disjoint");
}

namespace {

constexpr int64_t kCachePageRows = 128;
constexpr int64_t kCacheDataRowBytes = 64;
constexpr int64_t kCacheScaleRowBytes = 8;
constexpr int64_t kTmaMaxStrideBytes = int64_t{1} << 40;

// Checks one page-strided cache component: [P, 1, 128, row_bytes] (or [P, 128, row_bytes] when
// allow_3d) of one-byte elements. Rows inside a page are dense; stride(0) is the page stride in
// bytes and may be any multiple of `alignment` (16, or 32 for K/V data read through the
// 16U4_ALIGN16B TMA type) that is at least the dense page size.
void validate_cache_component(at::Tensor const &tensor, at::Device device, int64_t pages,
                              int64_t row_bytes, int64_t alignment, bool fp8_storage, bool allow_3d,
                              char const *name) {
  TORCH_CHECK(tensor.defined(), "m3_decode cache requires ", name);
  TORCH_CHECK(tensor.device() == device, name, " must be on the same device as q");
  TORCH_CHECK(fp8_storage ? is_fp8_byte_storage_tensor(tensor) : tensor.scalar_type() == at::kByte,
              name, fp8_storage ? " must be E4M3/uint8 storage" : " must be uint8 storage");
  bool const four_d = tensor.dim() == 4 && tensor.size(1) == 1;
  TORCH_CHECK(four_d || (allow_3d && tensor.dim() == 3), name, " must be [P, 1, 128, ", row_bytes,
              "]", allow_3d ? " or [P, 128, " : "", allow_3d ? std::to_string(row_bytes) : "",
              allow_3d ? "]" : "", "; got ", tensor.sizes());
  int64_t const row_dim = tensor.dim() - 2;
  TORCH_CHECK(tensor.size(0) == pages && tensor.size(row_dim) == kCachePageRows &&
                  tensor.size(row_dim + 1) == row_bytes,
              name, " must have ", pages, " pages of [128, ", row_bytes, "] bytes; got ",
              tensor.sizes());
  TORCH_CHECK(tensor.stride(row_dim + 1) == 1 && tensor.stride(row_dim) == row_bytes, name,
              " rows must be dense inside a page (strides [..., ", row_bytes, ", 1]); got ",
              tensor.strides());
  int64_t const dense_page_bytes = kCachePageRows * row_bytes;
  if (pages > 1) {
    int64_t const page_stride = tensor.stride(0);
    TORCH_CHECK(page_stride >= dense_page_bytes && page_stride % alignment == 0 &&
                    page_stride < kTmaMaxStrideBytes,
                name, " page stride must be a multiple of ", alignment, " bytes in [",
                dense_page_bytes, ", 2^40); got ", page_stride);
  }
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0, name, " must be ",
              alignment, "-byte aligned");
}

} // namespace

void validate_paged_cache_args(at::Tensor const &q, at::Tensor const &k, at::Tensor const &v,
                               at::Tensor const &k_scale, at::Tensor const &v_scale,
                               at::Tensor const &indexer_k, at::Tensor const &indexer_k_scale) {
  TORCH_CHECK(k.defined() && k.dim() == 4, "k must be a 4-D [P, 1, 128, 64] uint8 cache");
  int64_t const pages = k.size(0);
  TORCH_CHECK(pages > 0 && pages <= std::numeric_limits<int>::max(),
              "k must contain between 1 and INT_MAX physical pages");
  at::Device const device = q.device();
  validate_cache_component(k, device, pages, kCacheDataRowBytes, 32, false, false, "k");
  validate_cache_component(v, device, pages, kCacheDataRowBytes, 32, false, false, "v");
  validate_cache_component(k_scale, device, pages, kCacheScaleRowBytes, 16, true, false, "k_scale");
  validate_cache_component(v_scale, device, pages, kCacheScaleRowBytes, 16, true, false, "v_scale");
  validate_cache_component(indexer_k, device, pages, kCacheDataRowBytes, 16, false, true,
                           "indexer_k");
  validate_cache_component(indexer_k_scale, device, pages, kCacheScaleRowBytes, 16, true, true,
                           "indexer_k_scale");
}

} // namespace mega_atten::m3_decode
