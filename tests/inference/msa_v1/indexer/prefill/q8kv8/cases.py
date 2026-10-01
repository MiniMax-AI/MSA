"""Deterministic tensors for real Q8KV8 prefill shapes."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from datas.inference.cases import InferencePrefillCase
from datas.inference.tensors import cumulative_lengths, make_disjoint_page_table

PAGE_SIZE = 128
HEAD_DIM = 128
SUPPORTED_CAPABILITIES = frozenset({(10, 0), (10, 3), (10, 7)})


def require_sm100_device() -> torch.device:
    """Return a supported Blackwell device or skip the GPU test."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability not in SUPPORTED_CAPABILITIES:
        pytest.skip("Q8KV8 prefill requires SM100, SM103 or SM107")
    return device


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
    device: torch.device,
    num_index_heads: int = 1,
) -> RealPrefillInputs:
    """Create finite E4M3 tensors and disjoint active page mappings."""

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
        * (torch.arange(num_index_heads, device=device).reshape(1, -1, 1) + 1)
        * 0.25
    ).to(torch.float8_e4m3fn)
    k_cache = (
        torch.randn(
            (physical_pages, 1, PAGE_SIZE, HEAD_DIM),
            generator=generator,
            device=device,
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
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
    return RealPrefillInputs(
        q=q,
        k_cache=k_cache,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
    )


__all__ = [
    "RealPrefillInputs",
    "cumulative_lengths",
    "make_real_prefill_inputs",
    "require_sm100_device",
]
