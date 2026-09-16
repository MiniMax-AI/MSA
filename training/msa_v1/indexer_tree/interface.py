"""Public batch-1 arbitrary-Func MSA v1 indexer API."""

from __future__ import annotations

from typing import Optional

import torch

from msa_v1.indexer_tree.m3_arbitrary_plan import (
    M3ArbitraryMaskPlan,
    compile_m3_arbitrary_mask_plan,
)
from msa_v1.indexer_tree.m3_indexer import m3_indexer_forward

IndexerPlan = M3ArbitraryMaskPlan


def compile_plan(
    arbitrary_func: torch.Tensor,
    q_len: int,
    kv_len: Optional[int] = None,
    local_block_positions: Optional[torch.Tensor] = None,
) -> IndexerPlan:
    """Compile a reusable batch-1 Func plan outside training steps.

    Plan compilation performs a deliberate device-to-host size read so it can
    allocate exact materialized plan tensors. It is an offline setup API: call
    it before CUDA Graph capture and reuse the returned plan in every training
    step. Local block positions are bound to the returned plan and default to
    the bottom-right suffix mapping. The caller owns caching; ``forward``
    itself remains asynchronous.
    """
    return compile_m3_arbitrary_mask_plan(
        arbitrary_func,
        q_len,
        kv_len,
        local_block_positions,
    )


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    plan: IndexerPlan,
    *,
    block_bases: Optional[torch.Tensor] = None,
    score_workspace: Optional[torch.Tensor] = None,
    block_sum_workspace: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    selected_lse: Optional[torch.Tensor] = None,
    lse_temperature: float = 1.0,
    deterministic: bool = False,
    use_fp16_score: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Tree TopK and return temperature-scaled LSE.

    ``block_bases`` optionally rebases flat-K block ids per query row.
    Set ``deterministic=True`` to order non-local blocks by descending score
    and ascending block id, producing bitwise-identical outputs for repeated
    executions with the same inputs and environment. Both modes place the
    local block in the final valid slot.
    Set ``use_fp16_score=True`` to store and rank K1 scores in FP16 while
    retaining FP32 GEMM accumulation and LSE arithmetic.
    """
    return m3_indexer_forward(
        q,
        k,
        plan,
        block_bases=block_bases,
        score_workspace=score_workspace,
        block_sum_workspace=block_sum_workspace,
        topk_indices=topk_indices,
        selected_lse=selected_lse,
        deterministic=deterministic,
        lse_temperature=lse_temperature,
        use_fp16_score=use_fp16_score,
    )


__all__ = ["IndexerPlan", "compile_plan", "forward"]
