"""Public MSA v1 sparse KL backward entry point."""

from __future__ import annotations

from typing import Optional

import torch

from msa_v1.attention.metadata import AttentionMetadata
from msa_v1.kl.interface import sparse_kl_bwd_cute


def backward(
    q: torch.Tensor,
    k: torch.Tensor,
    teacher_lse: torch.Tensor,
    qi: torch.Tensor,
    ki: torch.Tensor,
    indexer_lse: torch.Tensor,
    metadata: AttentionMetadata,
    *,
    softmax_scale: Optional[float] = None,
    indexer_softmax_scale: Optional[float] = None,
    loss_coeff: float = 1.0,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused true-varlen MSA v1 sparse KL backward kernel.

    Set ``deterministic=True`` to order cross-CTA dQI and dKI FP32
    accumulation with writer-rank semaphores.
    """

    return sparse_kl_bwd_cute(
        q,
        k,
        teacher_lse,
        qi,
        ki,
        indexer_lse,
        metadata,
        softmax_scale=softmax_scale,
        indexer_softmax_scale=indexer_softmax_scale,
        loss_coeff=loss_coeff,
        deterministic=deterministic,
    )


__all__ = ["backward"]
