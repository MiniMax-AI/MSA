"""Deterministic Q8KV4 tensors for real prefill inference shapes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from datas.inference.cases import InferencePrefillCase
from datas.inference.tensors import (
    cumulative_lengths,
    make_attention_topk,
    make_disjoint_page_table,
)


PAGE_SIZE = 128
HEAD_DIM = 128
Q_HEADS = 64
KV_HEADS = 4
TOPK = 16


@dataclass(frozen=True)
class RealPrefillAttentionInputs:
    """Materialized Q8KV4 attention inputs for one recorded shape."""

    q: torch.Tensor
    packed_k: torch.Tensor
    packed_v: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    topk_indices: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor


def _make_finite_e4m3(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Generate moderate finite E4M3 values without a wider temporary."""

    bits = torch.randint(
        0,
        128,
        shape,
        dtype=torch.uint8,
        generator=generator,
        device=device,
    )
    sign = torch.bitwise_left_shift(torch.bitwise_and(bits, 0x40), 1)
    bits.bitwise_and_(0x3F)
    bits.bitwise_or_(sign)
    return bits.view(torch.float8_e4m3fn)


def _make_scale(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Generate positive finite E4M3 scales directly from raw bytes."""

    bits = torch.randint(
        0x18,
        0x39,
        shape,
        dtype=torch.uint8,
        generator=generator,
        device=device,
    )
    return bits.view(torch.float8_e4m3fn)


def make_real_prefill_attention_inputs(
    case: InferencePrefillCase,
    *,
    device: torch.device,
    num_kv_heads: int = KV_HEADS,
) -> RealPrefillAttentionInputs:
    """Materialize one recorded shape using deterministic Q8KV4 payloads."""

    generator = torch.Generator(device=device).manual_seed(case.seed)
    page_table, physical_pages = make_disjoint_page_table(
        case.final_kv_lens,
        max_cols=case.max_cols,
        generator=generator,
        device=device,
    )
    q = _make_finite_e4m3(
        (case.total_q, num_kv_heads * 16, HEAD_DIM),
        generator=generator,
        device=device,
    )
    packed_shape = (physical_pages, num_kv_heads, PAGE_SIZE, HEAD_DIM // 2)
    packed_k = torch.randint(
        0,
        256,
        packed_shape,
        dtype=torch.uint8,
        generator=generator,
        device=device,
    )
    packed_v = torch.randint(
        0,
        256,
        packed_shape,
        dtype=torch.uint8,
        generator=generator,
        device=device,
    )
    scale_shape = (physical_pages, num_kv_heads, PAGE_SIZE, HEAD_DIM // 16)
    k_scale = _make_scale(scale_shape, generator=generator, device=device)
    v_scale = _make_scale(scale_shape, generator=generator, device=device)
    cu_seqlens_q = torch.tensor(
        cumulative_lengths(case.query_lens),
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_k = torch.tensor(
        cumulative_lengths(case.final_kv_lens),
        dtype=torch.int32,
        device=device,
    )
    return RealPrefillAttentionInputs(
        q=q,
        packed_k=packed_k,
        packed_v=packed_v,
        k_scale=k_scale,
        v_scale=v_scale,
        page_table=page_table,
        topk_indices=make_attention_topk(case, device=device, kv_heads=num_kv_heads),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
    )


__all__ = [
    "RealPrefillAttentionInputs",
    "cumulative_lengths",
    "make_attention_topk",
    "make_real_prefill_attention_inputs",
]
