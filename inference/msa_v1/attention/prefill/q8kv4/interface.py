"""FlashInfer-style wrapper for Q8KV4 paged sparse prefill."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from inference.msa_v1.attention.prefill._common.combine import combine
from inference.msa_v1.attention.prefill._common.metadata import (
    AttentionMetadata,
    prepare,
)
from inference.msa_v1.attention.prefill.q8kv4.jit import load_extension


_HEAD_DIM = 128
_Q_HEADS_PER_KV = 16
_PAGE_SIZE = 128
_TOPK = 16


def _check_cuda_contiguous(tensor: torch.Tensor, *, name: str) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_same_device(
    reference: torch.Tensor,
    tensor: torch.Tensor,
    *,
    name: str,
) -> None:
    if tensor.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}")


@dataclass(frozen=True)
class _PlanState:
    metadata: AttentionMetadata
    page_table: torch.Tensor
    sm_scale: float
    o_partial: torch.Tensor
    lse_partial: torch.Tensor
    out: torch.Tensor
    lse: torch.Tensor


class BatchPrefillWithPagedKVCacheWrapper:
    """Stateful Q8KV4 sparse-prefill wrapper with reusable scheduling state."""

    def __init__(self) -> None:
        self._plan_state: _PlanState | None = None

    def plan(
        self,
        topk_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        page_table: torch.Tensor,
        *,
        total_k: int,
        total_rows: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool = True,
        sm_scale: float | None = None,
    ) -> None:
        """Prepare reusable q2k-to-k2q metadata and split-combine workspace."""

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("plan() must be called outside CUDA Graph capture")
        if not causal:
            raise NotImplementedError("Q8KV4 prefill v1 supports only causal attention")
        _check_cuda_contiguous(page_table, name="page_table")
        _check_same_device(topk_indices, page_table, name="page_table")
        if page_table.dtype != torch.int32:
            raise TypeError("page_table must be torch.int32")
        batch = cu_seqlens_q.numel() - 1
        if page_table.ndim != 2 or page_table.shape[0] != batch:
            raise ValueError("page_table must have shape [batch, max_pages]")
        if page_table.shape[1] <= 0:
            raise ValueError("page_table must contain at least one logical page slot")

        metadata = prepare(
            topk_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            total_k=total_k,
            total_rows=total_rows,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            fragment_indices=None,
            prepare_kl_schedule=False,
        )
        total_q = int(topk_indices.shape[1])
        num_q_heads = int(topk_indices.shape[0]) * _Q_HEADS_PER_KV
        options = dict(device=topk_indices.device)
        scale = 1.0 / math.sqrt(_HEAD_DIM) if sm_scale is None else float(sm_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("sm_scale must be finite and positive")
        self._plan_state = _PlanState(
            metadata=metadata,
            page_table=page_table,
            sm_scale=scale,
            o_partial=torch.empty(
                (_TOPK, total_q, num_q_heads, _HEAD_DIM),
                dtype=torch.bfloat16,
                **options,
            ),
            lse_partial=torch.empty(
                (_TOPK, total_q, num_q_heads),
                dtype=torch.float32,
                **options,
            ),
            out=torch.empty(
                (total_q, num_q_heads, _HEAD_DIM),
                dtype=torch.bfloat16,
                **options,
            ),
            lse=torch.empty(
                (total_q, num_q_heads),
                dtype=torch.float32,
                **options,
            ),
        )

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        *,
        kv_cache_sf: tuple[torch.Tensor, torch.Tensor],
        out: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run one layer with the metadata and workspace cached by ``plan``."""

        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        if len(paged_kv_cache) != 2 or len(kv_cache_sf) != 2:
            raise ValueError("paged_kv_cache and kv_cache_sf must be (K, V) pairs")
        k_cache, v_cache = paged_kv_cache
        k_scale, v_scale = kv_cache_sf
        tensors = {
            "q": q,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "k_scale": k_scale,
            "v_scale": v_scale,
        }
        for name, tensor in tensors.items():
            _check_cuda_contiguous(tensor, name=name)
            _check_same_device(state.page_table, tensor, name=name)

        if q.dtype != torch.float8_e4m3fn:
            raise TypeError("q must be torch.float8_e4m3fn")
        num_q_heads = state.out.shape[1]
        num_kv_heads = num_q_heads // _Q_HEADS_PER_KV
        if q.shape != state.out.shape:
            raise ValueError(f"q must have shape {tuple(state.out.shape)}")
        for name, tensor in (("k_cache", k_cache), ("v_cache", v_cache)):
            if tensor.dtype != torch.uint8:
                raise TypeError(f"{name} must be torch.uint8")
            if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (
                num_kv_heads,
                _PAGE_SIZE,
                _HEAD_DIM // 2,
            ):
                raise ValueError(
                    f"{name} must have shape [num_pages, {num_kv_heads}, 128, 64]"
                )
        if k_cache.shape != v_cache.shape:
            raise ValueError("k_cache and v_cache must have the same shape")
        for name, tensor in (("k_scale", k_scale), ("v_scale", v_scale)):
            if tensor.dtype != torch.float8_e4m3fn:
                raise TypeError(f"{name} must be torch.float8_e4m3fn")
            if tuple(tensor.shape) != (
                k_cache.shape[0],
                num_kv_heads,
                _PAGE_SIZE,
                _HEAD_DIM // 16,
            ):
                raise ValueError(
                    f"{name} must have shape [num_pages, {num_kv_heads}, 128, 8]"
                )

        out_tensor = state.out if out is None else out
        lse_tensor = state.lse if lse is None else lse
        for name, tensor, dtype, shape in (
            ("out", out_tensor, torch.bfloat16, state.out.shape),
            ("lse", lse_tensor, torch.float32, state.lse.shape),
        ):
            _check_cuda_contiguous(tensor, name=name)
            _check_same_device(q, tensor, name=name)
            if tensor.dtype != dtype or tensor.shape != shape:
                raise ValueError(
                    f"{name} must have dtype {dtype} and shape {tuple(shape)}"
                )

        schedule = state.metadata.schedule
        if (
            schedule.scheduler_metadata is None
            or schedule.work_count is None
            or schedule.qsplit_indices is None
            or schedule.split_counts is None
        ):
            raise RuntimeError("prepare() returned an incomplete forward schedule")

        load_extension(q.device).run(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            state.page_table,
            state.metadata.cu_seqlens_q,
            state.metadata.cu_seqlens_k,
            state.metadata.k2q_row_ptr,
            schedule.qsplit_indices,
            schedule.scheduler_metadata,
            schedule.work_count,
            state.o_partial,
            state.lse_partial,
            state.sm_scale,
        )
        combine(
            state.o_partial,
            state.lse_partial,
            out_tensor,
            lse_tensor if return_lse or lse is not None else None,
            cu_seqlens=state.metadata.cu_seqlens_q,
            split_counts=schedule.split_counts,
            use_pdl=True,
            raw_partial_stats=False,
        )
        if return_lse:
            return out_tensor, lse_tensor
        return out_tensor


__all__ = ["BatchPrefillWithPagedKVCacheWrapper"]
