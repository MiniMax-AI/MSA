#pragma once

#include <torch/torch.h>
#include <tvm/ffi/function.h>

#include <optional>
#include <tuple>
#include <vector>

namespace mega_atten::m3_decode {

struct PlanInfo {
  at::Tensor kv_segment_lens;
  at::Tensor output_projection_barrier;
  int kv_block_num = 0;
  std::vector<int32_t> kv_lens_cpu;

  bool is_valid() const { return kv_segment_lens.defined(); }
};

struct AttentionProjectionInfo {
  at::Tensor input_norm_weight;
  at::Tensor weight;
  at::Tensor weight_sf;
  at::Tensor workspace_data;
  at::Tensor workspace_scale;
  at::Tensor grid_epoch;
  at::Tensor special_partial;
  at::Tensor special_epochs;
  at::Tensor positions;
  at::Tensor slot_mapping;
  at::Tensor q_norm_weight;
  at::Tensor k_norm_weight;
  at::Tensor qi_norm_weight;
  at::Tensor ki_norm_weight;
  at::Tensor cos_sin_cache;
  double input_norm_eps = 0.0;
  double post_norm_eps = 0.0;
  double ki_dequant_scale = 1.0;

  bool enabled() const { return weight.defined(); }
};

} // namespace mega_atten::m3_decode
