#pragma once

#include <torch/extension.h>

#include <string>
#include <tuple>

namespace minimax::inference::dequant {

torch::Tensor dequantize_nvfp4_to_fp8(torch::Tensor packed_nvfp4, torch::Tensor scale,
                                      torch::Tensor output);

std::tuple<torch::Tensor, torch::Tensor> dequantize_sparse_paged_nvfp4_to_fp8(
    torch::Tensor packed_k, torch::Tensor packed_v, torch::Tensor k_scale, torch::Tensor v_scale,
    torch::Tensor pair_keys, torch::Tensor output_k, torch::Tensor output_v);

std::string compiled_backend();

} // namespace minimax::inference::dequant
