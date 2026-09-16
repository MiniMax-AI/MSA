"""Deterministic BF16 tensors for shared inference prefill cases."""

from dataclasses import dataclass

import torch

from datas.inference.cases import InferencePrefillCase
from datas.inference.tensors import cumulative_lengths, make_disjoint_page_table

PAGE_SIZE = 128
HEAD_DIM = 128


@dataclass(frozen=True)
class RealPrefillInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    page_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor


def make_real_prefill_inputs(
    case: InferencePrefillCase,
    *,
    num_index_heads: int,
    device: torch.device,
) -> RealPrefillInputs:
    generator = torch.Generator(device=device).manual_seed(case.seed)
    page_table, physical_pages = make_disjoint_page_table(
        case.final_kv_lens,
        max_cols=case.max_cols,
        generator=generator,
        device=device,
    )
    q = (
        torch.randn(
            (case.total_q, num_index_heads, HEAD_DIM),
            generator=generator,
            device=device,
        )
        * 0.25
    ).to(torch.bfloat16)
    k_cache = (
        torch.randn(
            (physical_pages, 1, PAGE_SIZE, HEAD_DIM),
            generator=generator,
            device=device,
        )
        * 0.25
    ).to(torch.bfloat16)
    return RealPrefillInputs(
        q=q,
        k_cache=k_cache,
        page_table=page_table,
        cu_seqlens_q=torch.tensor(
            cumulative_lengths(case.query_lens),
            dtype=torch.int32,
            device=device,
        ),
        cu_seqlens_k=torch.tensor(
            cumulative_lengths(case.final_kv_lens),
            dtype=torch.int32,
            device=device,
        ),
    )


__all__ = ["RealPrefillInputs", "make_real_prefill_inputs"]
