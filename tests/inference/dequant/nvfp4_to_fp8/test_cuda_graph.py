"""CUDA Graph replay tests for NVFP4-to-FP8 conversion."""

from __future__ import annotations

import pytest
import torch

from inference.dequant import dequantize_nvfp4_to_fp8
from tests.inference.dequant.nvfp4_to_fp8.reference import (
    make_exhaustive_inputs,
)

pytestmark = pytest.mark.gpu


def test_cuda_graph_replay_is_bitwise_deterministic() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
        pytest.skip("NVFP4 dequant requires SM100 or SM103")

    packed, scale = make_exhaustive_inputs(device)
    output = torch.empty((127, 128), dtype=torch.float8_e4m3fn, device=device)
    dequantize_nvfp4_to_fp8(packed, scale, out=output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = dequantize_nvfp4_to_fp8(packed, scale, out=output)
    torch.cuda.synchronize()
    baseline = result.clone()
    output.fill_(0)
    graph.replay()
    torch.cuda.synchronize()
    assert result.data_ptr() == output.data_ptr()
    assert torch.equal(output.view(torch.uint8), baseline.view(torch.uint8))


def test_capture_requires_preallocated_output() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
        pytest.skip("NVFP4 dequant requires SM100 or SM103")
    packed, scale = make_exhaustive_inputs(device)
    output = torch.empty((127, 128), dtype=torch.float8_e4m3fn, device=device)
    dequantize_nvfp4_to_fp8(packed, scale, out=output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="preallocated out"):
        with torch.cuda.graph(graph):
            dequantize_nvfp4_to_fp8(packed, scale)
