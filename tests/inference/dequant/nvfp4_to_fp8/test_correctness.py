"""Bitwise correctness tests for NVFP4-to-FP8 conversion."""

from __future__ import annotations

import pytest
import torch

from inference.dequant import dequantize_nvfp4_to_fp8
from tests.inference.dequant.nvfp4_to_fp8.reference import (
    dequantize_reference,
    make_exhaustive_inputs,
)

pytestmark = pytest.mark.gpu


def _require_sm100_family() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
        pytest.skip("NVFP4 dequant requires SM100 or SM103")
    return device


def test_all_2032_e2m1_scale_combinations_are_bitwise_exact() -> None:
    device = _require_sm100_family()
    packed, scale = make_exhaustive_inputs(device)
    actual = dequantize_nvfp4_to_fp8(packed, scale)
    expected = dequantize_reference(packed, scale)
    torch.testing.assert_close(
        actual.view(torch.uint8),
        expected.view(torch.uint8),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (64,),
        (3, 64),
        (33, 64),
        (2, 1, 5, 64),
        (3, 1, 128, 64),
    ],
)
def test_random_leading_shapes_are_bitwise_exact(shape: tuple[int, ...]) -> None:
    device = _require_sm100_family()
    generator = torch.Generator(device=device).manual_seed(sum(shape))
    packed = torch.randint(
        0,
        256,
        shape,
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    scale_shape = (*shape[:-1], 8)
    scale_bits = torch.randint(
        0,
        127,
        scale_shape,
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    scale = scale_bits.view(torch.float8_e4m3fn)
    output = torch.empty(
        (*shape[:-1], 128),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    actual = dequantize_nvfp4_to_fp8(packed, scale, out=output)
    expected = dequantize_reference(packed, scale)
    assert actual.data_ptr() == output.data_ptr()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_non_default_stream_and_repeated_runs_are_deterministic() -> None:
    device = _require_sm100_family()
    packed, scale = make_exhaustive_inputs(device)
    output = torch.empty((127, 128), dtype=torch.float8_e4m3fn, device=device)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        first = dequantize_nvfp4_to_fp8(packed, scale, out=output).clone()
        dequantize_nvfp4_to_fp8(packed, scale, out=output)
    stream.synchronize()
    assert torch.equal(first.view(torch.uint8), output.view(torch.uint8))
