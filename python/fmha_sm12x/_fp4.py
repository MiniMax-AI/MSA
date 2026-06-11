# SPDX-License-Identifier: MIT

"""SM12x FP4 indexer reference implementation."""

from __future__ import annotations

import torch

_PAGE_SIZE = 128
_PACKED_D_BYTES = 64
_HEAD_DIM = 128
_PUBLIC_SCALE_LAYOUT = "public"
_PREORDERED_MMA_SCALE_LAYOUT = "preordered_mma"
_FP4_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _ceil_div(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _scale_groups(fp4_format: str) -> int:
    match str(fp4_format).lower():
        case "mxfp4":
            return 4
        case "nvfp4":
            return 8
        case other:
            raise ValueError(f"fp4_format must be 'mxfp4' or 'nvfp4', got {other!r}")


def _require_i32_vector(tensor: torch.Tensor, *, name: str, device: torch.device) -> None:
    if tensor.device != device or tensor.dtype != torch.int32 or tensor.ndim != 1:
        raise ValueError(f"{name} must be rank-1 int32 on {device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _as_u8(tensor: torch.Tensor, *, name: str, expected_ndim: int) -> torch.Tensor:
    if tensor.ndim != expected_ndim:
        raise ValueError(f"{name} must be rank {expected_ndim}")
    if int(tensor.shape[-1]) != _PACKED_D_BYTES:
        raise ValueError(f"{name}.shape[-1] must be 64 packed bytes")
    if tensor.dtype == torch.uint8:
        return tensor.contiguous()
    return tensor.contiguous().view(torch.uint8)


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=packed.device)
    u8 = packed.to(torch.uint8)
    lo = u8 & 0x0F
    hi = u8 >> 4
    out = torch.empty((*u8.shape[:-1], _HEAD_DIM), dtype=torch.float32, device=u8.device)
    out[..., 0::2] = lut[lo.long()]
    out[..., 1::2] = lut[hi.long()]
    return out


def _restore_preordered_q_scale(scale: torch.Tensor, total_q: int, heads: int, groups: int) -> torch.Tensor:
    public = torch.empty((total_q, heads, groups), dtype=scale.dtype, device=scale.device)
    if scale.ndim == 6 and scale.shape[0] == 32:
        for row in range(total_q):
            r0 = row % 32
            r1 = (row // 32) % 4
            r2 = row // 128
            for group in range(groups):
                public[row, :, group] = scale[r0, r1, r2, group % 4, group // 4, :heads]
        return public
    if scale.ndim == 6:
        for row in range(total_q):
            for group in range(groups):
                public[row, :, group] = scale[:heads, row // 128, group // 4, row % 32, (row // 32) % 4, group % 4]
        return public
    raise ValueError("preordered q_scale must be a rank-6 MMA scale tensor")


def _restore_preordered_k_scale(scale: torch.Tensor, pages: int, heads: int, groups: int) -> torch.Tensor:
    public = torch.empty((pages, heads, _PAGE_SIZE, groups), dtype=scale.dtype, device=scale.device)
    if scale.ndim == 6 and scale.shape[0] == 32:
        for page in range(pages):
            for head in range(heads):
                scale_l = page * heads + head
                for row in range(_PAGE_SIZE):
                    for group in range(groups):
                        public[page, head, row, group] = scale[row % 32, (row // 32) % 4, 0, group % 4, group // 4, scale_l]
        return public
    if scale.ndim == 6:
        for page in range(pages):
            for head in range(heads):
                scale_l = page * heads + head
                for row in range(_PAGE_SIZE):
                    for group in range(groups):
                        public[page, head, row, group] = scale[scale_l, 0, group // 4, row % 32, (row // 32) % 4, group % 4]
        return public
    raise ValueError("preordered k_scale must be a rank-6 MMA scale tensor")


def _public_scales(scale: torch.Tensor, *, shape: tuple[int, ...], layout: str, fp4_format: str) -> torch.Tensor:
    groups = _scale_groups(fp4_format)
    if layout == _PUBLIC_SCALE_LAYOUT:
        if tuple(scale.shape) != shape:
            raise ValueError(f"scale must have shape {shape}, got {tuple(scale.shape)}")
        return scale.contiguous()
    if layout != _PREORDERED_MMA_SCALE_LAYOUT:
        raise ValueError(f"scale_layout must be 'public' or 'preordered_mma', got {layout!r}")
    if len(shape) == 3:
        return _restore_preordered_q_scale(scale, shape[0], shape[1], groups)
    return _restore_preordered_k_scale(scale, shape[0], shape[1], groups)


def _dequantize_fp4(packed: torch.Tensor, scale: torch.Tensor, *, fp4_format: str) -> torch.Tensor:
    groups = _scale_groups(fp4_format)
    logical = _unpack_fp4(packed)
    scale_f = scale.to(torch.float32).repeat_interleave(_HEAD_DIM // groups, dim=-1)
    return logical * scale_f


def fp4_indexer_block_scores(
    q_fp4: torch.Tensor,
    k_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_page_offsets: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    kv_indices: torch.Tensor,
    fp4_format: str,
    causal: bool = False,
    qo_offset: torch.Tensor | None = None,
    scale_layout: str = _PUBLIC_SCALE_LAYOUT,
) -> torch.Tensor:
    """Return dequantized FP4 QK max scores per 128-token KV page."""

    q_bytes = _as_u8(q_fp4, name="q_fp4", expected_ndim=3)
    k_bytes = _as_u8(k_fp4, name="k_fp4", expected_ndim=4)
    total_q, heads_q, _ = (int(v) for v in q_bytes.shape)
    pages, heads_k, page_size, _ = (int(v) for v in k_bytes.shape)
    if page_size != _PAGE_SIZE:
        raise ValueError(f"k_fp4 page size must be {_PAGE_SIZE}, got {page_size}")
    if heads_q % heads_k != 0:
        raise ValueError("q heads must be divisible by KV heads")
    for name, tensor in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k), ("cu_page_offsets", cu_page_offsets)):
        _require_i32_vector(tensor, name=name, device=q_fp4.device)
    if kv_indices.device != q_fp4.device or kv_indices.dtype != torch.int32 or kv_indices.ndim != 1:
        raise ValueError("kv_indices must be rank-1 int32 on q_fp4.device")
    batch = int(cu_seqlens_q.numel() - 1)
    if qo_offset is not None:
        _require_i32_vector(qo_offset, name="qo_offset", device=q_fp4.device)
        if qo_offset.shape != (batch,):
            raise ValueError("qo_offset must have shape [batch]")
    q_scales = _public_scales(q_scale, shape=(total_q, heads_q, _scale_groups(fp4_format)), layout=scale_layout, fp4_format=fp4_format)
    k_scales = _public_scales(k_scale, shape=(pages, heads_k, _PAGE_SIZE, _scale_groups(fp4_format)), layout=scale_layout, fp4_format=fp4_format)
    q = _dequantize_fp4(q_bytes, q_scales, fp4_format=fp4_format)
    k = _dequantize_fp4(k_bytes, k_scales, fp4_format=fp4_format)
    max_tiles = _ceil_div(int(max_seqlen_k), _PAGE_SIZE)
    scores = torch.full((heads_q, max_tiles, total_q), float("-inf"), dtype=torch.float32, device=q_fp4.device)
    q_cpu = cu_seqlens_q.to("cpu", dtype=torch.int64, non_blocking=False)
    k_cpu = cu_seqlens_k.to("cpu", dtype=torch.int64, non_blocking=False)
    page_cpu = cu_page_offsets.to("cpu", dtype=torch.int64, non_blocking=False)
    offset_cpu = None if qo_offset is None else qo_offset.to("cpu", dtype=torch.int64, non_blocking=False)
    h_ratio = heads_q // heads_k
    for b in range(batch):
        q_begin = int(q_cpu[b])
        q_len = int(q_cpu[b + 1] - q_cpu[b])
        kv_len = int(k_cpu[b + 1] - k_cpu[b])
        page_begin = int(page_cpu[b])
        for local_q in range(q_len):
            q_idx = q_begin + local_q
            visible_limit = (int(offset_cpu[b]) if offset_cpu is not None else kv_len - q_len) + local_q
            for tile in range(min(max_tiles, _ceil_div(kv_len, _PAGE_SIZE))):
                phys_page = int(kv_indices[page_begin + tile].item())
                valid = min(_PAGE_SIZE, kv_len - tile * _PAGE_SIZE)
                if valid <= 0:
                    continue
                positions = torch.arange(valid, device=q_fp4.device, dtype=torch.long) + tile * _PAGE_SIZE
                visible = positions <= visible_limit if causal else torch.ones_like(positions, dtype=torch.bool)
                if not bool(visible.any().item()):
                    continue
                for head in range(heads_q):
                    kv_head = head // h_ratio
                    logits = torch.matmul(k[phys_page, kv_head, :valid].float(), q[q_idx, head].float())
                    scores[head, tile, q_idx] = logits[visible].max()
    return scores
