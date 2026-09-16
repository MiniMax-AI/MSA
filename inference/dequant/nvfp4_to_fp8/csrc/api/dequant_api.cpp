#include "dequant_api.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "sm100/common/dequant_config.hpp"
#include "sm100/device/dequant.hpp"

namespace minimax::inference::dequant {
namespace {

using sm100::kHeadDim;
using sm100::kPackedHeadDim;
using sm100::kScaleGroups;

void check_cuda_contiguous(torch::Tensor const &tensor, char const *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(torch::Tensor const &reference, torch::Tensor const &tensor,
                       char const *name) {
  TORCH_CHECK(tensor.device() == reference.device(), name, " must be on ", reference.device());
}

void check_alignment(torch::Tensor const &tensor, uintptr_t alignment, char const *name) {
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0, name,
              " address must be aligned to ", alignment, " bytes");
}

bool ranges_overlap(torch::Tensor const &lhs, torch::Tensor const &rhs) {
  uintptr_t const lhs_begin = reinterpret_cast<uintptr_t>(lhs.data_ptr());
  uintptr_t const rhs_begin = reinterpret_cast<uintptr_t>(rhs.data_ptr());
  uintptr_t const lhs_end = lhs_begin + static_cast<uintptr_t>(lhs.numel() * lhs.element_size());
  uintptr_t const rhs_end = rhs_begin + static_cast<uintptr_t>(rhs.numel() * rhs.element_size());
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

void check_sm100_family(torch::Tensor const &reference) {
  c10::cuda::CUDAGuard const device_guard(reference.device());
  cudaDeviceProp const *properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 10 && (properties->minor == 0 || properties->minor == 3),
              "NVFP4 dequant requires an SM100 or SM103 GPU");
}

void check_dense_shapes(torch::Tensor const &packed_nvfp4, torch::Tensor const &scale,
                        torch::Tensor const &output) {
  TORCH_CHECK(packed_nvfp4.dim() >= 1 &&
                  packed_nvfp4.size(packed_nvfp4.dim() - 1) == kPackedHeadDim,
              "packed_nvfp4 must have shape [..., 64]");
  TORCH_CHECK(packed_nvfp4.numel() > 0, "packed_nvfp4 must contain at least one row");
  TORCH_CHECK(scale.dim() == packed_nvfp4.dim() && scale.size(scale.dim() - 1) == kScaleGroups,
              "scale must have shape [..., 8]");
  TORCH_CHECK(output.dim() == packed_nvfp4.dim() && output.size(output.dim() - 1) == kHeadDim,
              "out must have shape [..., 128]");
  for (int64_t dim = 0; dim + 1 < packed_nvfp4.dim(); ++dim) {
    TORCH_CHECK(scale.size(dim) == packed_nvfp4.size(dim),
                "scale leading dimensions must match packed_nvfp4");
    TORCH_CHECK(output.size(dim) == packed_nvfp4.size(dim),
                "out leading dimensions must match packed_nvfp4");
  }
}

void check_sparse_cache(torch::Tensor const &packed, torch::Tensor const &scale,
                        char const *packed_name, char const *scale_name) {
  check_cuda_contiguous(packed, packed_name);
  check_cuda_contiguous(scale, scale_name);
  TORCH_CHECK(packed.scalar_type() == at::kByte, packed_name, " must have dtype torch.uint8");
  TORCH_CHECK(scale.scalar_type() == at::kFloat8_e4m3fn, scale_name,
              " must have dtype torch.float8_e4m3fn");
  TORCH_CHECK(packed.dim() == 4 && packed.size(2) > 0 && packed.size(3) == kPackedHeadDim,
              packed_name, " must have shape [pages, kv_heads, page_size, 64]");
  TORCH_CHECK(scale.dim() == 4 && scale.size(0) == packed.size(0) &&
                  scale.size(1) == packed.size(1) && scale.size(2) == packed.size(2) &&
                  scale.size(3) == kScaleGroups,
              scale_name, " must have shape [pages, kv_heads, page_size, 8]");
}

} // namespace

torch::Tensor dequantize_nvfp4_to_fp8(torch::Tensor packed_nvfp4, torch::Tensor scale,
                                      torch::Tensor output) {
  check_cuda_contiguous(packed_nvfp4, "packed_nvfp4");
  check_cuda_contiguous(scale, "scale");
  check_cuda_contiguous(output, "out");
  check_same_device(packed_nvfp4, scale, "scale");
  check_same_device(packed_nvfp4, output, "out");
  TORCH_CHECK(packed_nvfp4.scalar_type() == at::kByte, "packed_nvfp4 must have dtype torch.uint8");
  TORCH_CHECK(scale.scalar_type() == at::kFloat8_e4m3fn,
              "scale must have dtype torch.float8_e4m3fn");
  TORCH_CHECK(output.scalar_type() == at::kFloat8_e4m3fn,
              "out must have dtype torch.float8_e4m3fn");
  check_dense_shapes(packed_nvfp4, scale, output);
  check_alignment(packed_nvfp4, 8, "packed_nvfp4");
  check_alignment(scale, 8, "scale");
  check_alignment(output, 16, "out");
  TORCH_CHECK(!ranges_overlap(output, packed_nvfp4), "out must not overlap packed_nvfp4 storage");
  TORCH_CHECK(!ranges_overlap(output, scale), "out must not overlap scale storage");

  check_sm100_family(packed_nvfp4);
  c10::cuda::CUDAGuard const device_guard(packed_nvfp4.device());
  cudaDeviceProp const *properties = at::cuda::getCurrentDeviceProperties();
  int64_t const rows = packed_nvfp4.numel() / kPackedHeadDim;
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream().stream();
  cudaError_t const status = sm100::launch_dense_nvfp4_to_fp8(
      packed_nvfp4.data_ptr<uint8_t>(), reinterpret_cast<uint8_t const *>(scale.data_ptr()),
      reinterpret_cast<uint8_t *>(output.data_ptr()), rows, properties->multiProcessorCount,
      stream);
  TORCH_CHECK(status == cudaSuccess, "NVFP4 dequant launch failed: ", cudaGetErrorString(status));
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> dequantize_sparse_paged_nvfp4_to_fp8(
    torch::Tensor packed_k, torch::Tensor packed_v, torch::Tensor k_scale, torch::Tensor v_scale,
    torch::Tensor pair_keys, torch::Tensor output_k, torch::Tensor output_v) {
  check_sparse_cache(packed_k, k_scale, "packed_k", "k_scale");
  check_sparse_cache(packed_v, v_scale, "packed_v", "v_scale");
  check_cuda_contiguous(pair_keys, "pair_keys");
  check_cuda_contiguous(output_k, "out_k");
  check_cuda_contiguous(output_v, "out_v");
  check_same_device(packed_k, packed_v, "packed_v");
  check_same_device(packed_k, k_scale, "k_scale");
  check_same_device(packed_k, v_scale, "v_scale");
  check_same_device(packed_k, pair_keys, "pair_keys");
  check_same_device(packed_k, output_k, "out_k");
  check_same_device(packed_k, output_v, "out_v");
  TORCH_CHECK(packed_v.sizes() == packed_k.sizes(), "packed_v shape must match packed_k");
  TORCH_CHECK(v_scale.sizes() == k_scale.sizes(), "v_scale shape must match k_scale");
  TORCH_CHECK(pair_keys.scalar_type() == at::kLong && pair_keys.dim() == 1 && pair_keys.numel() > 0,
              "pair_keys must be a non-empty one-dimensional torch.int64 tensor");
  TORCH_CHECK(output_k.scalar_type() == at::kFloat8_e4m3fn &&
                  output_v.scalar_type() == at::kFloat8_e4m3fn,
              "out_k and out_v must have dtype torch.float8_e4m3fn");
  int64_t const num_kv_heads = packed_k.size(1);
  int64_t const page_size = packed_k.size(2);
  int64_t const required_output_pages = pair_keys.numel() + 2 * num_kv_heads - 2;
  TORCH_CHECK(output_k.dim() == 3 && output_k.size(0) == required_output_pages &&
                  output_k.size(1) == page_size && output_k.size(2) == kHeadDim,
              "out_k must have shape [num_pairs + 2 * kv_heads - 2, page_size, 128]");
  TORCH_CHECK(output_v.dim() == 3 && output_v.size(0) == required_output_pages &&
                  output_v.size(1) == page_size && output_v.size(2) == kHeadDim,
              "out_v must have shape [num_pairs + 2 * kv_heads - 2, page_size, 128]");
  check_alignment(packed_k, 8, "packed_k");
  check_alignment(packed_v, 8, "packed_v");
  check_alignment(k_scale, 8, "k_scale");
  check_alignment(v_scale, 8, "v_scale");
  check_alignment(output_k, 16, "out_k");
  check_alignment(output_v, 16, "out_v");
  TORCH_CHECK(!ranges_overlap(output_k, packed_k) && !ranges_overlap(output_k, packed_v) &&
                  !ranges_overlap(output_k, k_scale) && !ranges_overlap(output_k, v_scale),
              "out_k must not overlap input storage");
  TORCH_CHECK(!ranges_overlap(output_v, packed_k) && !ranges_overlap(output_v, packed_v) &&
                  !ranges_overlap(output_v, k_scale) && !ranges_overlap(output_v, v_scale),
              "out_v must not overlap input storage");
  TORCH_CHECK(!ranges_overlap(output_k, output_v), "out_k and out_v must not overlap");

  check_sm100_family(packed_k);
  c10::cuda::CUDAGuard const device_guard(packed_k.device());
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream().stream();
  cudaError_t const status = sm100::launch_sparse_paged_nvfp4_to_fp8(
      packed_k.data_ptr<uint8_t>(), packed_v.data_ptr<uint8_t>(),
      reinterpret_cast<uint8_t const *>(k_scale.data_ptr()),
      reinterpret_cast<uint8_t const *>(v_scale.data_ptr()), pair_keys.data_ptr<int64_t>(),
      reinterpret_cast<uint8_t *>(output_k.data_ptr()),
      reinterpret_cast<uint8_t *>(output_v.data_ptr()), pair_keys.numel(),
      static_cast<int>(num_kv_heads), static_cast<int>(page_size), stream);
  TORCH_CHECK(status == cudaSuccess,
              "sparse NVFP4 dequant launch failed: ", cudaGetErrorString(status));
  return {output_k, output_v};
}

std::string compiled_backend() { return sm100::nvfp4_to_fp8_backend(); }

} // namespace minimax::inference::dequant
