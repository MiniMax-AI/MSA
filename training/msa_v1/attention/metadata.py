"""Reusable sparse-attention metadata for the fixed MSA v1 contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from msa_v1.attention.prepare_k2q_csr import SparseK2qCsrBuilderSm100
from msa_v1.attention.prepare_scheduler import SparseAttentionSchedule

_HEAD_KV = 4
_QHEAD_PER_KV = 16
_TOPK = 16
_KV_BLOCK_SIZE = 128


@dataclass(frozen=True)
class AttentionMetadata:
    """Per-layer metadata shared by the corresponding forward and backward."""

    topk_indices: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    fragment_indices: torch.Tensor | None
    k2q_row_ptr: torch.Tensor
    k2q_q_indices: torch.Tensor
    schedule: SparseAttentionSchedule
    kl_schedule: SparseAttentionSchedule
    total_k: int
    total_rows: int
    max_seqlen_q: int
    max_seqlen_k: int


def _validate_cu_seqlens(
    tensor: torch.Tensor, *, name: str, device: torch.device
) -> None:
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must be torch.int32")
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must be a CUDA tensor on {device}")
    if tensor.ndim != 1 or tensor.numel() < 2:
        raise ValueError(f"{name} must have shape [batch + 1]")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_fragment_indices(
    tensor: torch.Tensor | None,
    *,
    batch: int,
    device: torch.device,
) -> None:
    if tensor is None:
        return
    if tensor.dtype != torch.int32:
        raise TypeError("fragment_indices must be torch.int32")
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError("fragment_indices must be on the TopK CUDA device")
    if tuple(tensor.shape) != (batch,) or not tensor.is_contiguous():
        raise ValueError("fragment_indices must be contiguous with shape [B]")


def prepare(
    topk_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    total_k: int,
    total_rows: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    fragment_indices: torch.Tensor | None = None,
    prepare_kl_schedule: bool = True,
) -> AttentionMetadata:
    """Build layer-specific CSR and schedules from the current TopK indices.

    ``total_rows`` is the exact sum of logical K-block rows across batches,
    including fragments that share the same physical K storage.

    The returned metadata belongs to this TopK result and may be shared by its
    corresponding forward, attention backward, and KL backward. A new TopK
    result requires a new call, including during CUDA Graph replay.
    """
    if topk_indices.dtype != torch.int32:
        raise TypeError("topk_indices must be torch.int32")
    if not topk_indices.is_cuda:
        raise ValueError("topk_indices must be a CUDA tensor")
    if topk_indices.ndim != 3 or tuple(topk_indices.shape[::2]) != (_HEAD_KV, _TOPK):
        raise ValueError("topk_indices must have shape [4, total_q, 16]")
    if not topk_indices.is_contiguous():
        raise ValueError("topk_indices must be contiguous")
    _validate_cu_seqlens(cu_seqlens_q, name="cu_seqlens_q", device=topk_indices.device)
    _validate_cu_seqlens(cu_seqlens_k, name="cu_seqlens_k", device=topk_indices.device)
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same shape")
    _validate_fragment_indices(
        fragment_indices,
        batch=cu_seqlens_q.numel() - 1,
        device=topk_indices.device,
    )

    total_k = int(total_k)
    total_rows = int(total_rows)
    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_k = int(max_seqlen_k)
    if total_k < 0 or total_rows < 0:
        raise ValueError("total_k and total_rows must be non-negative")
    if max_seqlen_q < 0 or max_seqlen_k < 0:
        raise ValueError("maximum sequence lengths must be non-negative")

    builder = SparseK2qCsrBuilderSm100()
    k2q_row_ptr, k2q_q_indices, schedule, kl_schedule = builder(
        topk_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        total_k=total_k,
        blk_kv=_KV_BLOCK_SIZE,
        max_seqlen_k=max_seqlen_k,
        max_seqlen_q=max_seqlen_q,
        total_rows=total_rows,
        qhead_per_kv=_QHEAD_PER_KV,
        fragment_indices=fragment_indices,
        prepare_kl_schedule=bool(prepare_kl_schedule),
    )
    return AttentionMetadata(
        topk_indices=topk_indices,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        fragment_indices=fragment_indices,
        k2q_row_ptr=k2q_row_ptr,
        k2q_q_indices=k2q_q_indices,
        schedule=schedule,
        kl_schedule=kl_schedule,
        total_k=total_k,
        total_rows=total_rows,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )


__all__ = ["AttentionMetadata", "prepare"]
