"""Deterministic Q8KV4 decode-attention inputs for shared Indexer cases."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tests.inference.msa_v1.decode.cases import DecodeCorrectnessCase
from tests.inference.msa_v1.decode.metadata import make_decode_topk

PAGE_SIZE = 128
HEAD_DIM = 128
KV_HEADS = 4
TOPK = 16
_DISTRIBUTION_SALT = {
    "fixed": 11,
    "narrow": 23,
    "bimodal": 37,
    "long_tail": 53,
    "one_1m": 71,
}


@dataclass(frozen=True)
class DecodeAttentionInputs:
    """Materialized paged Q8KV4 inputs for one decode length vector."""

    q: torch.Tensor
    packed_k: torch.Tensor
    packed_v: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    topk_indices: torch.Tensor
    seq_lens: torch.Tensor
    q_len_per_req: int
    seed: int
    page_layout: str


def shared_workload_seed(case: DecodeCorrectnessCase) -> int:
    """Derive the payload seed without changing canonical length generation."""

    return (
        case.seed
        + case.batch_size * 1009
        + case.q_len_per_req * 131
        + _DISTRIBUTION_SALT.get(getattr(case, "distribution", "benchmark"), 97)
    )


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
    bits = torch.randint(
        0x18,
        0x39,
        shape,
        dtype=torch.uint8,
        generator=generator,
        device=device,
    )
    return bits.view(torch.float8_e4m3fn)


def make_decode_attention_inputs(
    seq_lens_cpu: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
    q_len_per_req: int = 8,
    page_layout: str = "permuted",
    num_kv_heads: int = KV_HEADS,
    num_q_heads: int | None = None,
) -> DecodeAttentionInputs:
    """Materialize one shared decode case using Q8KV4 attention payloads."""

    if seq_lens_cpu.dtype != torch.int32 or seq_lens_cpu.ndim != 1:
        raise ValueError("seq_lens_cpu must be a one-dimensional int32 tensor")
    if bool(torch.any(seq_lens_cpu < q_len_per_req)):
        raise ValueError("every sequence must include the full query chunk")
    num_q_heads = num_kv_heads * 16 if num_q_heads is None else num_q_heads
    generator = torch.Generator(device=device).manual_seed(seed)
    seq_lens = seq_lens_cpu.to(device=device)
    batch = seq_lens.numel()
    max_pages = (int(seq_lens_cpu.max()) + PAGE_SIZE - 1) // PAGE_SIZE
    if page_layout == "disjoint":
        physical_pages = batch * max_pages
    elif page_layout == "shared_prefix":
        physical_pages = max_pages
    elif page_layout == "permuted":
        physical_pages = max_pages + batch + 7
    else:
        raise ValueError(f"unsupported page layout: {page_layout}")
    q = _make_finite_e4m3(
        (batch * q_len_per_req, num_q_heads, HEAD_DIM),
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
    if page_layout == "disjoint":
        physical = torch.arange(
            physical_pages, dtype=torch.int64, device=device
        ).reshape(batch, max_pages)
        page_table = torch.stack(
            [
                physical[batch_idx][
                    torch.randperm(max_pages, generator=generator, device=device)
                ]
                for batch_idx in range(batch)
            ]
        )
    elif page_layout == "shared_prefix":
        shared = torch.randperm(
            physical_pages,
            generator=generator,
            device=device,
            dtype=torch.int64,
        )[:max_pages]
        page_table = shared.expand(batch, -1).clone()
    else:
        page_table = torch.stack(
            [
                torch.randperm(
                    physical_pages,
                    generator=generator,
                    device=device,
                    dtype=torch.int64,
                )[:max_pages]
                for _ in range(batch)
            ]
        )
    page_table = page_table.to(torch.int32)
    return DecodeAttentionInputs(
        q=q,
        packed_k=packed_k,
        packed_v=packed_v,
        k_scale=k_scale,
        v_scale=v_scale,
        page_table=page_table.contiguous(),
        topk_indices=make_decode_topk(
            seq_lens,
            q_len_per_req=q_len_per_req,
            seed=seed,
            num_kv_heads=num_kv_heads,
        ),
        seq_lens=seq_lens,
        q_len_per_req=q_len_per_req,
        seed=seed,
        page_layout=page_layout,
    )


__all__ = [
    "DecodeAttentionInputs",
    "make_decode_attention_inputs",
    "make_decode_topk",
    "shared_workload_seed",
]
