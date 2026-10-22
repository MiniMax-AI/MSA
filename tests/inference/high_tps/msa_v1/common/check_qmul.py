"""Exhaustively verify four packed E2M1 lanes against FP32 multiplication."""

from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

root = Path(__file__).resolve().parents[5]
source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cute/tensor.hpp>
#include "common/kernel/include/sm100_m3_decode_fp4_transform.cuh"

__global__ void check_qmul(uint32_t* output) {
  uint32_t const index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 127u * 65536u) return;
  uint32_t const code = index & 65535u;
  uint32_t const input = (code & 15u) | ((code & 240u) << 4) |
                         ((code & 3840u) << 8) | ((code & 61440u) << 12);
  uint32_t scales = 0;
  for (int lane = 0; lane < 4; ++lane) {
    scales |= ((index / 65536u + lane * 31u) % 127u) << (8 * lane);
  }
  uint32_t result;
  cutlass::m3_decode::collective::convert_e2m1x4_to_e4m3x4_qmul4(result, input, scales);
  output[index] = result;
}

void check(torch::Tensor output) {
  check_qmul<<<(127 * 65536 + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint32_t*>(output.data_ptr()));
}
"""
module = load_inline(
    name="m3_qmul_conversion_check",
    cpp_sources="void check(torch::Tensor output);",
    cuda_sources=source,
    functions=["check"],
    extra_include_paths=[
        str(root / "inference/high_tps/msa_v1/csrc"),
        str(root / "third_party/cutlass/include"),
    ],
    extra_cuda_cflags=[
        "-std=c++20",
        "-gencode=arch=compute_100a,code=sm_100a",
        "-gencode=arch=compute_103a,code=sm_103a",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ],
    verbose=True,
)
actual = torch.empty((127, 65536, 4), device="cuda", dtype=torch.uint8)
module.check(actual)
torch.cuda.synchronize()
levels = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    device="cuda",
)
codes = torch.arange(65536, device="cuda")
scales = (
    torch.arange(127, dtype=torch.uint8, device="cuda")
    .view(torch.float8_e4m3fn)
    .float()
)
for lane in range(4):
    lane_scale = scales[(torch.arange(127, device="cuda") + lane * 31) % 127]
    expected = lane_scale[:, None] * levels[(codes >> (4 * lane)) & 15][None, :]
    expected = expected.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(
        actual.view(torch.float8_e4m3fn)[:, :, lane].float(), expected, rtol=0, atol=0
    )
print(
    "QMUL_PASS: all 65536 four-lane codes x 127 finite nonnegative scale rotations",
    flush=True,
)
