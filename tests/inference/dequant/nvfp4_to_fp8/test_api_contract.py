"""Public contract tests for NVFP4-to-FP8 conversion."""

from __future__ import annotations

import inspect

import pytest
import torch

import inference.dequant as dequant


def test_single_public_entrypoint() -> None:
    assert dequant.__all__ == [
        "SparseDequantizedPagedKvCache",
        "SparsePagedNvfp4ToFp8Wrapper",
        "dequantize_nvfp4_to_fp8",
    ]
    assert tuple(inspect.signature(dequant.dequantize_nvfp4_to_fp8).parameters) == (
        "packed_nvfp4",
        "scale",
        "out",
    )


@pytest.mark.gpu
def test_rejects_invalid_shapes_and_dtypes() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    packed = torch.zeros((2, 64), dtype=torch.uint8, device=device)
    scale = torch.ones((2, 8), dtype=torch.float8_e4m3fn, device=device)

    with pytest.raises(TypeError, match="packed_nvfp4"):
        dequant.dequantize_nvfp4_to_fp8(packed.float(), scale)
    with pytest.raises(TypeError, match="scale"):
        dequant.dequantize_nvfp4_to_fp8(packed, scale.float())
    with pytest.raises(ValueError, match="shape"):
        dequant.dequantize_nvfp4_to_fp8(packed[:, :32].contiguous(), scale)
    with pytest.raises(ValueError, match="shape"):
        dequant.dequantize_nvfp4_to_fp8(packed, scale[:1])
    with pytest.raises(ValueError, match="at least one row"):
        dequant.dequantize_nvfp4_to_fp8(
            torch.empty((0, 64), dtype=torch.uint8, device=device),
            torch.empty((0, 8), dtype=torch.float8_e4m3fn, device=device),
        )
    with pytest.raises(ValueError, match="contiguous"):
        dequant.dequantize_nvfp4_to_fp8(
            torch.zeros((64, 2), dtype=torch.uint8, device=device).t(),
            scale,
        )


@pytest.mark.gpu
def test_preallocated_output_contract() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    packed = torch.zeros((2, 64), dtype=torch.uint8, device=device)
    scale = torch.ones((2, 8), dtype=torch.float8_e4m3fn, device=device)
    output = torch.empty((2, 128), dtype=torch.float8_e4m3fn, device=device)
    result = dequant.dequantize_nvfp4_to_fp8(packed, scale, out=output)
    assert result.data_ptr() == output.data_ptr()

    with pytest.raises(TypeError, match="out"):
        dequant.dequantize_nvfp4_to_fp8(packed, scale, out=output.float())
    with pytest.raises(ValueError, match="shape"):
        dequant.dequantize_nvfp4_to_fp8(
            packed,
            scale,
            out=torch.empty((2, 64), dtype=torch.float8_e4m3fn, device=device),
        )
