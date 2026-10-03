#include "indexer_topk_api.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>

#include "indexer_topk_kernel.hpp"

namespace minimax::msa_v1::indexer::topk {
namespace {

constexpr int kTopK = 16;
constexpr int kMaximumColumns = 8192;

void check_same_device(torch::Tensor const &reference, torch::Tensor const &tensor,
                       char const *name) {
  TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same device as scores");
}

void check_sm100_family(torch::Tensor const &reference) {
  c10::cuda::CUDAGuard const device_guard(reference.device());
  cudaDeviceProp const *properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 10 && (properties->minor == 0 || properties->minor == 3),
              "MSA v1 indexer TopK requires an SM100-family GPU");
}

} // namespace

torch::Tensor indexer_topk_run(torch::Tensor scores, torch::Tensor lengths, torch::Tensor output,
                               bool compact_grid, bool enable_pdl) {
  TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must have dtype torch.float32");
  TORCH_CHECK(scores.dim() == 2 && scores.size(0) > 0,
              "scores must have shape [num_rows, max_cols]");
  TORCH_CHECK(scores.size(0) <= std::numeric_limits<int>::max(), "num_rows must fit in int32");
  TORCH_CHECK(scores.size(1) > 0 && scores.size(1) <= kMaximumColumns, "max_cols must be in [1, ",
              kMaximumColumns, "]");
  TORCH_CHECK(scores.stride(1) == 1, "scores must have stride(1) == 1");
  TORCH_CHECK(scores.stride(0) >= scores.size(1), "scores rows must not overlap");
  TORCH_CHECK(scores.stride(0) <= std::numeric_limits<int>::max(),
              "scores stride(0) must fit in int32");

  TORCH_CHECK(lengths.is_cuda() && lengths.is_contiguous(),
              "lengths must be a contiguous CUDA tensor");
  TORCH_CHECK(lengths.scalar_type() == at::kInt, "lengths must have dtype torch.int32");
  TORCH_CHECK(lengths.dim() == 1 && lengths.size(0) == scores.size(0),
              "lengths must have shape [num_rows]");
  check_same_device(scores, lengths, "lengths");

  TORCH_CHECK(output.is_cuda() && output.is_contiguous(), "out must be a contiguous CUDA tensor");
  TORCH_CHECK(output.scalar_type() == at::kInt, "out must have dtype torch.int32");
  TORCH_CHECK(output.dim() == 2 && output.size(0) == scores.size(0) && output.size(1) == kTopK,
              "out must have shape [num_rows, 16]");
  check_same_device(scores, output, "out");
  check_sm100_family(scores);

  c10::cuda::CUDAGuard const device_guard(scores.device());
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream().stream();
  cudaError_t const status = launch_indexer_topk(
      scores.data_ptr<float>(), lengths.data_ptr<int32_t>(), output.data_ptr<int32_t>(),
      static_cast<int>(scores.size(1)), static_cast<int>(scores.stride(0)),
      static_cast<int>(scores.size(0)), stream, compact_grid, enable_pdl);
  TORCH_CHECK(status == cudaSuccess, "indexer TopK launch failed: ", cudaGetErrorString(status));
  return output;
}

} // namespace minimax::msa_v1::indexer::topk
