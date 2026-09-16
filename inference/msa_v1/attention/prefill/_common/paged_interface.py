"""Shared FlashInfer-style paged sparse-prefill wrapper."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from inference.msa_v1.attention.prefill._common.combine import (
    combine,
)
from inference.msa_v1.attention.prefill._common.metadata import (
    AttentionMetadata,
    prepare,
)
from inference.msa_v1.attention.prefill._common.atten_fwd_sm100 import run_pagekv


_HEAD_DIM = 128
_DEFAULT_NUM_Q_HEADS = 64
_PAGE_SIZE = 128
_TOPK = 16
_SUPPORTED_GQA_GROUP_SIZES = {1, 2, 4, 8, 16}


def _check_cuda_contiguous(tensor: torch.Tensor, *, name: str) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    _check_16_byte_aligned(tensor, name=name)


def _check_16_byte_aligned(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.data_ptr() % 16 != 0:
        raise ValueError(f"{name} must be 16-byte aligned")


def _check_same_device(
    reference: torch.Tensor,
    tensor: torch.Tensor,
    *,
    name: str,
) -> None:
    if tensor.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}")


def _validate_plan_inputs(
    topk_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    total_k: int,
    total_rows: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    op_name: str,
    supported_gqa_group_sizes: tuple[int, ...],
) -> None:
    _check_cuda_contiguous(topk_indices, name="topk_indices")
    if topk_indices.dtype != torch.int32:
        raise TypeError("topk_indices must be torch.int32")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    gqa_group_size = num_q_heads // num_kv_heads
    if gqa_group_size not in supported_gqa_group_sizes:
        raise ValueError(
            f"{op_name} prefill supports GQA group sizes "
            f"{list(supported_gqa_group_sizes)}, got {gqa_group_size}"
        )
    if topk_indices.ndim != 3 or tuple(topk_indices.shape[::2]) != (
        num_kv_heads,
        _TOPK,
    ):
        raise ValueError(
            f"topk_indices must have shape [{num_kv_heads}, total_q, {_TOPK}]"
        )

    for name, tensor in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
        ("page_table", page_table),
    ):
        _check_cuda_contiguous(tensor, name=name)
        _check_same_device(topk_indices, tensor, name=name)
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be torch.int32")

    if cu_seqlens_q.ndim != 1 or cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens_q must have shape [batch + 1]")
    if cu_seqlens_k.shape != cu_seqlens_q.shape:
        raise ValueError("cu_seqlens_k must match cu_seqlens_q shape")
    batch = cu_seqlens_q.numel() - 1
    if page_table.ndim != 2 or page_table.shape[0] != batch:
        raise ValueError("page_table must have shape [batch, max_pages]")
    if page_table.shape[1] <= 0:
        raise ValueError("page_table must contain at least one logical page slot")

    sizes = {
        "total_k": int(total_k),
        "total_rows": int(total_rows),
        "max_seqlen_q": int(max_seqlen_q),
        "max_seqlen_k": int(max_seqlen_k),
    }
    if any(value < 0 for value in sizes.values()):
        raise ValueError("sequence sizes must be non-negative")
    if page_table.shape[1] * _PAGE_SIZE < sizes["max_seqlen_k"]:
        raise ValueError("page_table capacity is smaller than max_seqlen_k")


@dataclass(frozen=True)
class _PlanState:
    metadata: AttentionMetadata
    page_table: torch.Tensor
    num_q_heads: int
    num_kv_heads: int
    sm_scale: float
    o_partial: torch.Tensor
    lse_partial: torch.Tensor
    out: torch.Tensor
    lse: torch.Tensor


class _BatchPrefillWithPagedKVCacheWrapperBase:
    """Shared stateful PageKV sparse-prefill wrapper implementation."""

    storage_dtype = torch.float8_e4m3fn
    op_name = "Q8KV8"
    allow_strided_kv = False
    supported_gqa_group_sizes = tuple(sorted(_SUPPORTED_GQA_GROUP_SIZES))

    def __init__(self) -> None:
        self._plan_state: _PlanState | None = None

    def plan(
        self,
        topk_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        page_table: torch.Tensor,
        *,
        num_q_heads: int = _DEFAULT_NUM_Q_HEADS,
        num_kv_heads: int | None = None,
        total_k: int,
        total_rows: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool = True,
        sm_scale: float | None = None,
    ) -> None:
        """Build reusable CSR metadata, schedule, and split workspaces."""

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("plan() must be called outside CUDA Graph capture")
        if not causal:
            raise NotImplementedError(
                f"{self.op_name} prefill supports only causal attention"
            )
        num_q_heads = int(num_q_heads)
        if num_kv_heads is None:
            if topk_indices.ndim != 3:
                raise ValueError("topk_indices must be 3D")
            num_kv_heads = int(topk_indices.shape[0])
        else:
            num_kv_heads = int(num_kv_heads)
        _validate_plan_inputs(
            topk_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            page_table,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            total_k=total_k,
            total_rows=total_rows,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            op_name=self.op_name,
            supported_gqa_group_sizes=self.supported_gqa_group_sizes,
        )
        scale = 1.0 / math.sqrt(_HEAD_DIM) if sm_scale is None else float(sm_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("sm_scale must be finite and positive")

        metadata = prepare(
            topk_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            total_k=total_k,
            total_rows=total_rows,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            qhead_per_kv=num_q_heads // num_kv_heads,
            fragment_indices=None,
            prepare_kl_schedule=False,
        )
        total_q = int(topk_indices.shape[1])
        options = {"device": topk_indices.device}
        self._plan_state = _PlanState(
            metadata=metadata,
            page_table=page_table,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
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
        out: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run attention K1 and split combine using cached plan state."""

        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        if not isinstance(paged_kv_cache, tuple) or len(paged_kv_cache) != 2:
            raise ValueError("paged_kv_cache must be a (K, V) tuple")
        k_cache, v_cache = paged_kv_cache
        _check_cuda_contiguous(q, name="q")
        for name, tensor in (("k_cache", k_cache), ("v_cache", v_cache)):
            if not tensor.is_cuda:
                raise ValueError(f"{name} must be a CUDA tensor")
            if not self.allow_strided_kv and not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            if tensor.stride(-1) != 1:
                raise ValueError(f"{name} must have unit stride in head_dim")
            _check_16_byte_aligned(tensor, name=name)
        for name, tensor in (("q", q), ("k_cache", k_cache), ("v_cache", v_cache)):
            _check_same_device(state.page_table, tensor, name=name)
            if tensor.dtype != self.storage_dtype:
                raise TypeError(f"{name} must be {self.storage_dtype}")

        expected_q_shape = (state.out.shape[0], state.num_q_heads, _HEAD_DIM)
        if tuple(q.shape) != expected_q_shape:
            raise ValueError(f"q must have shape {expected_q_shape}")
        expected_kv_tail = (state.num_kv_heads, _PAGE_SIZE, _HEAD_DIM)
        for name, tensor in (("k_cache", k_cache), ("v_cache", v_cache)):
            if tensor.ndim != 4 or tuple(tensor.shape[1:]) != expected_kv_tail:
                raise ValueError(
                    f"{name} must have shape [physical_pages, "
                    f"{state.num_kv_heads}, {_PAGE_SIZE}, {_HEAD_DIM}]"
                )
        if k_cache.shape != v_cache.shape:
            raise ValueError("k_cache and v_cache must have the same shape")

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
        if schedule.split_counts is None:
            raise RuntimeError("prepare() returned an incomplete forward schedule")
        run_pagekv(
            q,
            k_cache,
            v_cache,
            state.page_table,
            state.metadata.cu_seqlens_q,
            state.metadata.cu_seqlens_k,
            state.metadata.k2q_row_ptr,
            state.metadata.k2q_q_indices,
            schedule,
            state.o_partial,
            state.lse_partial,
            softmax_scale=state.sm_scale,
            max_seqlen_q=state.metadata.max_seqlen_q,
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
