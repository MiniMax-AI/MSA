"""Public explicit MSA v1 sparse-attention backward API."""

from __future__ import annotations

from typing import Optional

import torch

from msa_v1.attention.forward import _validate_inputs
from msa_v1.attention.interface import _call_sparse_bwd_csr_varlen
from msa_v1.attention.metadata import AttentionMetadata


def backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    metadata: AttentionMetadata,
    *,
    softmax_scale: Optional[float] = None,
    sparse_attn_p_mode: str = "",
    q_fp8: Optional[torch.Tensor] = None,
    k_fp8: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute GQA16 dQ, dK, and dV using metadata prepared for forward.

    For probability QAT, ``q_fp8`` and ``k_fp8`` must be the exact E4M3
    forward payloads. Logical-D recomputes QK with FP8 UMMA while the
    differentiable Q/K/V inputs remain BF16 QDQ tensors.
    Set ``deterministic=True`` to order every cross-CTA FP32 dQ/dK/dV
    accumulation with writer-rank semaphores.
    """
    _validate_inputs(q, k, v, metadata)
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v)):
        raise NotImplementedError(
            "attention backward supports only BF16 Q/K/V; native FP8 Q/K/V are unsupported"
        )
    expected_out_shape = tuple(q.shape)
    if tuple(dout.shape) != expected_out_shape or tuple(out.shape) != expected_out_shape:
        raise ValueError("dout and out must have shape [total_q, 64, 128]")
    if tuple(lse.shape) != tuple(q.shape[:2]) or lse.dtype != torch.float32:
        raise ValueError("lse must be FP32 with shape [total_q, 64]")
    if dout.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise TypeError("dout and out must be BF16")
    if any(t.device != q.device for t in (dout, out, lse)):
        raise ValueError("dout, out, and lse must be on the same device as q")
    if sparse_attn_p_mode not in ("", "fp8"):
        raise ValueError("sparse_attn_p_mode must be '' or 'fp8'")
    if type(deterministic) is not bool:
        raise TypeError("deterministic must be a Python bool")
    if sparse_attn_p_mode and q.dtype != torch.bfloat16:
        raise ValueError("probability QAT backward requires BF16 q, k, and v")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    return _call_sparse_bwd_csr_varlen(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        dout.contiguous(),
        out.contiguous(),
        lse.contiguous(),
        float(softmax_scale),
        metadata.k2q_row_ptr,
        metadata.k2q_q_indices,
        cu_seqlens_q=metadata.cu_seqlens_q,
        cu_seqlens_k=metadata.cu_seqlens_k,
        fragment_indices=metadata.fragment_indices,
        topK=16,
        blk_kv=128,
        causal=True,
        use_prepare_scheduler=True,
        scheduler_metadata=metadata.schedule.scheduler_metadata,
        work_count=metadata.schedule.work_count,
        work_capacity=metadata.schedule.work_capacity,
        dkv_owner_counts=metadata.schedule.dkv_owner_counts,
        dkv_split_indices=metadata.schedule.dkv_split_indices,
        dkv_split_count=metadata.schedule.dkv_split_count,
        k2q_qsplit_indices=(
            metadata.schedule.qsplit_indices
            if sparse_attn_p_mode == "fp8" or deterministic
            else None
        ),
        max_seqlen_q=metadata.max_seqlen_q,
        max_seqlen_k=metadata.max_seqlen_k,
        sparse_attn_p_mode=sparse_attn_p_mode,
        q_fp8=q_fp8,
        k_fp8=k_fp8,
        deterministic=deterministic,
    )


__all__ = ["backward"]
