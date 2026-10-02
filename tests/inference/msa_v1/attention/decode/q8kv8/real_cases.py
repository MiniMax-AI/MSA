"""Deterministic Q8K8 paged sparse decode-attention inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tests.inference.msa_v1.decode.cases import DecodeCorrectnessCase
from tests.inference.msa_v1.decode.metadata import (
    KV_HEADS,
    PAGE_SIZE,
    TOPK,
    make_decode_topk,
)

HEAD_DIM = 128
Q_HEADS = 64
_DISTRIBUTION_SALT = {
    "boundary": 11,
    "narrow": 23,
    "bimodal": 37,
    "long_tail": 53,
}


@dataclass(frozen=True)
class DecodeAttentionInputs:
    """Materialized paged Q8K8 inputs for one decode workload."""

    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    page_table: torch.Tensor
    topk_indices: torch.Tensor
    seq_lens: torch.Tensor
    q_len_per_req: int
    seed: int
    page_layout: str


def shared_workload_seed(case: DecodeCorrectnessCase) -> int:
    """Derive a stable payload seed from one shared decode case."""

    return (
        case.seed
        + case.batch_size * 1009
        + case.q_len_per_req * 131
        + _DISTRIBUTION_SALT[case.distribution]
    )


def _make_finite_e4m3(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
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


def _make_page_table(
    batch: int,
    max_pages: int,
    physical_pages: int,
    *,
    page_layout: str,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if page_layout == "disjoint":
        physical = torch.arange(
            physical_pages, dtype=torch.int64, device=device
        ).reshape(batch, max_pages)
        rows = [
            row[torch.randperm(max_pages, generator=generator, device=device)]
            for row in physical
        ]
        return torch.stack(rows).to(torch.int32).contiguous()
    if page_layout == "shared_prefix":
        shared = torch.randperm(
            physical_pages,
            generator=generator,
            device=device,
            dtype=torch.int64,
        )[:max_pages]
        return shared.expand(batch, -1).clone().to(torch.int32).contiguous()
    if page_layout == "permuted":
        rows = [
            torch.randperm(
                physical_pages,
                generator=generator,
                device=device,
                dtype=torch.int64,
            )[:max_pages]
            for _ in range(batch)
        ]
        return torch.stack(rows).to(torch.int32).contiguous()
    raise ValueError(f"unsupported page layout: {page_layout}")


def make_decode_attention_inputs(
    seq_lens_cpu: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
    q_len_per_req: int,
    page_layout: str,
    num_q_heads: int = Q_HEADS,
    num_kv_heads: int = KV_HEADS,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> DecodeAttentionInputs:
    """Materialize one shared case without changing its sparse metadata."""

    if seq_lens_cpu.dtype != torch.int32 or seq_lens_cpu.ndim != 1:
        raise ValueError("seq_lens_cpu must be a one-dimensional int32 tensor")
    if bool(torch.any(seq_lens_cpu < q_len_per_req)):
        raise ValueError("every sequence must include the full query chunk")
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

    def make_payload(shape, *, generator, device):
        if dtype == torch.bfloat16:
            return torch.randn(
                shape, generator=generator, device=device, dtype=torch.float32
            ).to(dtype)
        if dtype != torch.float8_e4m3fn:
            raise ValueError("only BF16 and E4M3 payloads are supported")
        return _make_finite_e4m3(shape, generator=generator, device=device)

    q = make_payload(
        (batch * q_len_per_req, num_q_heads, HEAD_DIM),
        generator=generator,
        device=device,
    )
    cache_shape = (physical_pages, num_kv_heads, PAGE_SIZE, HEAD_DIM)
    k_cache = make_payload(cache_shape, generator=generator, device=device)
    v_cache = make_payload(cache_shape, generator=generator, device=device)
    page_table = _make_page_table(
        batch,
        max_pages,
        physical_pages,
        page_layout=page_layout,
        generator=generator,
        device=device,
    )
    topk_indices = make_decode_topk(
        seq_lens,
        q_len_per_req=q_len_per_req,
        seed=seed,
        num_kv_heads=num_kv_heads,
    )
    if topk_indices.shape[-1] != TOPK:
        raise AssertionError("shared TopK generator returned an invalid capacity")
    return DecodeAttentionInputs(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        topk_indices=topk_indices,
        seq_lens=seq_lens,
        q_len_per_req=q_len_per_req,
        seed=seed,
        page_layout=page_layout,
    )


__all__ = [
    "DecodeAttentionInputs",
    "make_decode_attention_inputs",
    "shared_workload_seed",
]
