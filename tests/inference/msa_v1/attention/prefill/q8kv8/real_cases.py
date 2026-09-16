"""Deterministic Q8KV8 tensors for recorded inference prefill shapes."""

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


@dataclass(frozen=True)
class RealPrefillAttentionInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    page_table: torch.Tensor
    topk_indices: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor


def make_real_prefill_attention_inputs(
    case: InferencePrefillCase,
    *,
    device: torch.device,
) -> RealPrefillAttentionInputs:
    generator = torch.Generator(device=device).manual_seed(case.seed)
    page_table, physical_pages = make_disjoint_page_table(
        case.final_kv_lens,
        max_cols=case.max_cols,
        generator=generator,
        device=device,
    )

    def random_e4m3(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            torch.randn(shape, generator=generator, device=device) * 0.25
        ).to(torch.float8_e4m3fn)

    q = random_e4m3((case.total_q, Q_HEADS, HEAD_DIM))
    cache_shape = (physical_pages, KV_HEADS, PAGE_SIZE, HEAD_DIM)
    k_cache = random_e4m3(cache_shape)
    v_cache = random_e4m3(cache_shape)
    return RealPrefillAttentionInputs(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        topk_indices=make_attention_topk(case, device=device),
        cu_seqlens_q=torch.tensor(
            cumulative_lengths(case.query_lens), dtype=torch.int32, device=device
        ),
        cu_seqlens_k=torch.tensor(
            cumulative_lengths(case.final_kv_lens),
            dtype=torch.int32,
            device=device,
        ),
    )


__all__ = ["RealPrefillAttentionInputs", "make_real_prefill_attention_inputs"]
