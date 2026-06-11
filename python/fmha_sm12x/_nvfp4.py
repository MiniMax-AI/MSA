# SPDX-License-Identifier: MIT

"""SM12x NVFP4 sparse-attention reference implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch


NVFP4_BLOCK_SIZE: Final = 16
NVFP4_FP4_MAX: Final = 6.0
NVFP4_FP8_E4M3_MAX: Final = 448.0
_HEAD_DIM: Final = 128
_FP4_VALUES: Final = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


@dataclass(frozen=True, slots=True)
class Nvfp4QuantizedTensor:
    """Packed NVFP4 tensor plus the metadata needed to dequantize it."""

    data: torch.Tensor
    scale_128x4: torch.Tensor
    global_scale: torch.Tensor
    logical_scale_shape: tuple[int, int]
    original_shape: tuple[int, ...]


def _round_up(x: int, multiple: int) -> int:
    return ((int(x) + int(multiple) - 1) // int(multiple)) * int(multiple)


def nvfp4_scale_128x4_offset(row: torch.Tensor, col: torch.Tensor, scale_cols: int) -> torch.Tensor:
    """Return flat offsets for cuBLAS/cuDNN 128x4 rowwise scale storage."""

    padded_cols = _round_up(scale_cols, 4)
    tiles_n = padded_cols // 4
    tile_m = row // 128
    tile_n = col // 4
    outer = row % 128
    inner = col % 4
    return (tile_m * tiles_n + tile_n) * 512 + (outer % 32) * 16 + (outer // 32) * 4 + inner


def swizzle_nvfp4_scale_to_128x4(scale: torch.Tensor, *, rows: int, cols: int) -> torch.Tensor:
    """Convert TE logical rowwise scales to cuBLAS/cuDNN 128x4 tiled layout."""

    if scale.ndim != 2:
        raise ValueError(f"scale must be 2D, got shape {tuple(scale.shape)}")
    rows = int(rows)
    cols = int(cols)
    padded_rows = _round_up(rows, 128)
    padded_cols = _round_up(cols, 4)
    if scale.shape[0] < rows or scale.shape[1] < cols:
        raise ValueError(
            "scale is smaller than the requested logical shape: "
            f"got {tuple(scale.shape)}, need at least {(rows, cols)}"
        )
    logical = scale[:rows, :cols].contiguous()
    if logical.shape != (padded_rows, padded_cols):
        logical = torch.nn.functional.pad(
            logical.to(torch.float32),
            (0, padded_cols - cols, 0, padded_rows - rows),
        ).to(scale.dtype)
    swizzled = torch.empty_like(logical)
    row = torch.arange(padded_rows, device=scale.device, dtype=torch.int64)[:, None]
    col = torch.arange(padded_cols, device=scale.device, dtype=torch.int64)[None, :]
    offset = nvfp4_scale_128x4_offset(row, col, padded_cols).reshape(-1)
    swizzled.reshape(-1)[offset] = logical.reshape(-1)
    return swizzled


def nvfp4_global_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """Compute TE NVFP4 tensor/global dequant scale from rowwise amax."""

    return amax.to(torch.float32) / (NVFP4_FP8_E4M3_MAX * NVFP4_FP4_MAX)


def _import_te_nvfp4_quantizer():
    try:
        from transformer_engine.pytorch.tensor import NVFP4Quantizer
    except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Transformer Engine NVFP4 quantization is unavailable. Install a "
            "Transformer Engine build with its PyTorch dependencies."
        ) from exc
    return NVFP4Quantizer


def quantize_bf16_to_nvfp4_128x4(x: torch.Tensor) -> Nvfp4QuantizedTensor:
    """Quantize a BF16/FP16 tensor to NVFP4 using Transformer Engine."""

    if not x.is_cuda:
        raise ValueError("NVFP4 quantization requires a CUDA tensor")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"x must be bf16 or fp16, got {x.dtype}")
    if x.ndim < 2:
        raise ValueError(f"x must have at least 2 dimensions, got {x.ndim}")
    if x.shape[-1] % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(f"last dimension must be divisible by {NVFP4_BLOCK_SIZE}, got {x.shape[-1]}")
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    if rows % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(f"flattened row dimension must be divisible by {NVFP4_BLOCK_SIZE}, got {rows}")

    quantizer_type = _import_te_nvfp4_quantizer()
    quantizer = quantizer_type(rowwise=True, columnwise=False)
    qx = quantizer.quantize(x.contiguous())
    meta = qx.get_metadata()
    data = meta["rowwise_data"]
    if data.dtype != torch.uint8:
        data = data.view(torch.uint8)
    scale_cols = int(x.shape[-1]) // NVFP4_BLOCK_SIZE
    scale_128x4 = swizzle_nvfp4_scale_to_128x4(meta["rowwise_scale_inv"], rows=rows, cols=scale_cols)
    return Nvfp4QuantizedTensor(
        data=data,
        scale_128x4=scale_128x4,
        global_scale=nvfp4_global_scale_from_amax(meta["amax_rowwise"]).contiguous(),
        logical_scale_shape=(rows, scale_cols),
        original_shape=tuple(int(v) for v in x.shape),
    )


def quantize_kv_bf16_to_nvfp4_128x4(
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[Nvfp4QuantizedTensor, Nvfp4QuantizedTensor]:
    """Quantize BF16/FP16 K and V tensors independently for KVFP4 attention."""

    return quantize_bf16_to_nvfp4_128x4(k), quantize_bf16_to_nvfp4_128x4(v)


def _unpack_nvfp4(data: torch.Tensor, logical_dim: int) -> torch.Tensor:
    lut = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=data.device)
    packed = data.contiguous().view(torch.uint8)
    values = torch.empty((*packed.shape[:-1], logical_dim), dtype=torch.float32, device=data.device)
    values[..., 0::2] = lut[(packed & 0x0F).long()]
    values[..., 1::2] = lut[(packed >> 4).long()]
    return values


def dequantize_nvfp4_128x4(
    data: torch.Tensor,
    scale_128x4: torch.Tensor,
    global_scale: torch.Tensor | None,
    *,
    original_shape: tuple[int, ...],
) -> torch.Tensor:
    """Dequantize packed NVFP4 data with cuBLAS/cuDNN 128x4 scales."""

    if data.dtype != torch.uint8:
        data = data.view(torch.uint8)
    logical_dim = int(original_shape[-1])
    if logical_dim % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(f"last dimension must be divisible by {NVFP4_BLOCK_SIZE}, got {logical_dim}")
    if data.shape[-1] * 2 != logical_dim:
        raise ValueError("packed data last dimension does not match original shape")
    rows = 1
    for dim in original_shape[:-1]:
        rows *= int(dim)
    scale_cols = logical_dim // NVFP4_BLOCK_SIZE
    values = _unpack_nvfp4(data, logical_dim).reshape(rows, logical_dim)
    row = torch.arange(rows, device=data.device, dtype=torch.int64)[:, None]
    col = torch.arange(scale_cols, device=data.device, dtype=torch.int64)[None, :]
    offsets = nvfp4_scale_128x4_offset(row, col, scale_cols)
    scale = scale_128x4.reshape(-1)[offsets.reshape(-1)].reshape(rows, scale_cols)
    scale_f = scale.view(torch.float8_e4m3fn).to(torch.float32).repeat_interleave(NVFP4_BLOCK_SIZE, dim=1)
    out = values * scale_f
    if global_scale is not None:
        out = out * global_scale.reshape(-1)[0].to(torch.float32)
    return out.reshape(original_shape).to(torch.bfloat16)


def dequantize_nvfp4_128x4_to_bf16(
    qx: Nvfp4QuantizedTensor,
    *,
    include_global_scale: bool = True,
) -> torch.Tensor:
    """Reference dequantization for validation of packed NVFP4 tensors."""

    global_scale = qx.global_scale if include_global_scale else None
    return dequantize_nvfp4_128x4(
        qx.data,
        qx.scale_128x4,
        global_scale,
        original_shape=qx.original_shape,
    )


def sparse_atten_nvfp4_kv_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale_128x4: torch.Tensor,
    v_scale_128x4: torch.Tensor,
    k_global_scale: torch.Tensor | None,
    v_global_scale: torch.Tensor | None,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    topK: int,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    blk_kv: int = 128,
    causal: bool = False,
    softmax_scale: float | None = None,
    lse_temperature_scale: float = 1.0,
    return_temperature_lse: bool = False,
    partial_dtype: "torch.dtype" = torch.bfloat16,
    return_softmax_lse: bool = False,
    page_table: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    schedule: object | None = None,
    usable_SM_count: int = -1,
    qk_dtype: "torch.dtype | None" = None,
    pv_dtype: "torch.dtype | None" = None,
    **_kwargs,
):
    """Run SM12x sparse attention by dequantizing packed NVFP4 K/V.

    Mirrors the SM100 ``sparse_atten_nvfp4_kv_func`` surface, including the
    temperature-scaled LSE outputs, by forwarding to ``sparse_atten_func``.
    """

    from .sparse import sparse_atten_func

    logical_shape = (*k.shape[:-1], _HEAD_DIM)
    k_bf16 = dequantize_nvfp4_128x4(k, k_scale_128x4, k_global_scale, original_shape=logical_shape)
    v_bf16 = dequantize_nvfp4_128x4(v, v_scale_128x4, v_global_scale, original_shape=logical_shape)
    return sparse_atten_func(
        q, k_bf16, v_bf16, k2q_row_ptr, k2q_q_indices, int(topK),
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=int(max_seqlen_q), max_seqlen_k=int(max_seqlen_k),
        blk_kv=int(blk_kv), causal=bool(causal), softmax_scale=softmax_scale,
        lse_temperature_scale=float(lse_temperature_scale),
        return_temperature_lse=bool(return_temperature_lse), partial_dtype=partial_dtype,
        return_softmax_lse=bool(return_softmax_lse), page_table=page_table, seqused_k=seqused_k,
        schedule=schedule, usable_SM_count=int(usable_SM_count), qk_dtype=qk_dtype, pv_dtype=pv_dtype,
    )
