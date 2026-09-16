"""Public packed-varlen MSA v1 indexer API."""

from __future__ import annotations

from typing import Optional

import torch

from msa_v1.indexer.m3_indexer import (
    IndexerForwardWorkspace,
    IndexerSchedule,
    allocate_indexer_schedule,
    allocate_indexer_workspace,
    m3_indexer_forward,
    prepare_indexer_schedule,
)


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    fragment_indices: Optional[torch.Tensor] = None,
    schedule: Optional[IndexerSchedule] = None,
    workspace: Optional[IndexerForwardWorkspace] = None,
    lse_temperature: float = 1.0,
    deterministic: bool = False,
    use_fp16_score: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the causal packed-varlen v1 indexer.

    Each query fragment is the causal suffix of its effective KV fragment.
    ``fragment_indices[b]`` optionally redirects the physical KV start to
    ``cu_seqlens_kv[fragment_indices[b]]`` while the end remains
    ``cu_seqlens_kv[b + 1]``.
    ``lse_temperature`` rescales only the returned LSE, not TopK ranking.
    Set ``deterministic=True`` to guarantee bitwise-identical TopK indices and
    selected LSE for repeated executions with the same inputs and environment.
    Set ``use_fp16_score=True`` to store K1 scores in FP16 and rank those FP16
    values directly while keeping GEMM accumulation and LSE arithmetic in FP32.
    """

    return m3_indexer_forward(
        q,
        k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        fragment_indices=fragment_indices,
        schedule=schedule,
        workspace=workspace,
        lse_temperature=lse_temperature,
        deterministic=deterministic,
        use_fp16_score=use_fp16_score,
    )


__all__ = [
    "IndexerForwardWorkspace",
    "IndexerSchedule",
    "allocate_indexer_schedule",
    "allocate_indexer_workspace",
    "forward",
    "prepare_indexer_schedule",
]
