"""Public MSA v1 sparse-attention forward API."""

from __future__ import annotations

from typing import Optional

import torch

from msa_v1.attention.interface import sparse_atten_func as _sparse_atten_func
from msa_v1.attention.metadata import AttentionMetadata


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    metadata: AttentionMetadata,
) -> None:
    if not isinstance(metadata, AttentionMetadata):
        raise TypeError("metadata must be an AttentionMetadata instance")
    if q.ndim != 3 or tuple(q.shape[1:]) != (64, 128):
        raise ValueError("q must have shape [total_q, 64, 128]")
    if k.ndim != 3 or tuple(k.shape[1:]) != (4, 128):
        raise ValueError("k must have shape [total_k, 4, 128]")
    if v.shape != k.shape:
        raise ValueError("v must have the same shape as k")
    if int(q.shape[0]) != int(metadata.topk_indices.shape[1]):
        raise ValueError("q length must match metadata.topk_indices")
    if int(k.shape[0]) != metadata.total_k:
        raise ValueError("k/v length must match metadata.total_k")
    if q.device != k.device or q.device != v.device or q.device != metadata.topk_indices.device:
        raise ValueError("q, k, v, and metadata must be on the same CUDA device")
    if not q.is_cuda:
        raise ValueError("q, k, and v must be CUDA tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must have the same dtype")
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError("q, k, and v must be BF16 or FP8 E4M3")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("q, k, and v must be contiguous")


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    metadata: AttentionMetadata,
    *,
    softmax_scale: Optional[float] = None,
    lse_temperature_scale: float = 1.0,
    return_softmax_lse: bool = False,
    return_temperature_lse: bool = False,
    partial_dtype: torch.dtype = torch.bfloat16,
    sparse_attn_p_mode: str = "",
    deterministic: bool = False,
):
    """Run fixed-shape, causal, flat-varlen MSA v1 sparse attention.

    ``deterministic`` is saved by autograd and selects the ordered backward
    path. The MSA v1 forward kernel itself is deterministic in both modes.
    """
    _validate_inputs(q, k, v, metadata)
    if sparse_attn_p_mode not in ("", "fp8"):
        raise ValueError("sparse_attn_p_mode must be '' or 'fp8'")
    if type(deterministic) is not bool:
        raise TypeError("deterministic must be a Python bool")
    if sparse_attn_p_mode and q.dtype != torch.bfloat16:
        raise ValueError("probability QAT requires BF16 q, k, and v")
    if q.dtype == torch.float8_e4m3fn and any(t.requires_grad for t in (q, k, v)):
        raise NotImplementedError(
            "native FP8 Q/K/V do not support attention training or backward"
        )

    return _sparse_atten_func(
        q,
        k,
        v,
        metadata.k2q_row_ptr,
        metadata.k2q_q_indices,
        16,
        cu_seqlens_q=metadata.cu_seqlens_q,
        cu_seqlens_k=metadata.cu_seqlens_k,
        fragment_indices=metadata.fragment_indices,
        max_seqlen_q=metadata.max_seqlen_q,
        max_seqlen_k=metadata.max_seqlen_k,
        blk_kv=128,
        causal=True,
        softmax_scale=softmax_scale,
        lse_temperature_scale=lse_temperature_scale,
        return_temperature_lse=return_temperature_lse,
        partial_dtype=partial_dtype,
        return_softmax_lse=return_softmax_lse,
        schedule=metadata.schedule,
        sparse_attn_p_mode=sparse_attn_p_mode,
        deterministic=deterministic,
    )


__all__ = ["forward"]
