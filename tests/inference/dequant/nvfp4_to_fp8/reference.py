"""Independent FP32 reference for NVFP4-to-E4M3 conversion."""

from __future__ import annotations

import torch


E2M1_VALUES = (
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
)


def dequantize_reference(
    packed_nvfp4: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Decode E2M1 in FP32, multiply in FP32, then cast to E4M3."""

    low = packed_nvfp4 & 0x0F
    high = packed_nvfp4 >> 4
    codes = torch.stack((low, high), dim=-1).reshape(*packed_nvfp4.shape[:-1], 128)
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed_nvfp4.device)
    values = lut[codes.to(torch.int64)]
    expanded_scale = scale.float().repeat_interleave(16, dim=-1)
    dequantized = values * expanded_scale
    return dequantized.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def make_exhaustive_inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Cover all 16 E2M1 codes against all valid nonnegative E4M3 scales."""

    codes = torch.arange(128, dtype=torch.uint8, device=device) % 16
    packed_row = codes[0::2] | (codes[1::2] << 4)
    packed = packed_row.reshape(1, 64).expand(127, 64).clone()
    scale_bits = torch.arange(127, dtype=torch.uint8, device=device).reshape(127, 1)
    scale = scale_bits.expand(127, 8).clone().view(torch.float8_e4m3fn)
    return packed, scale


__all__ = ["E2M1_VALUES", "dequantize_reference", "make_exhaustive_inputs"]
