# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""FlashInfer-style wrapper for Q8KV4 paged sparse prefill."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..decode_q8kv4.interface import interleave_v_scales
from .jit import MAX_BLOCK_SCALE_SHIFT, load_extension


_HEAD_DIM = 128
_Q_HEADS_PER_KV = 16
_PAGE_SIZE = 128
# TopK list widths the k2q CSR builder and the combine accept.
SUPPORTED_TOPK = (4, 8, 16, 32)
_KV_DATA_ROW_BYTES = _HEAD_DIM // 2
_KV_SCALE_ROW_BYTES = _HEAD_DIM // 16


def _sparse_stack():
    """Return fmha_sm100's k2q CSR builder and split combine (the CuTe-DSL sparse stack)."""
    from .. import sparse  # noqa: F401  (puts the cute/ modules on sys.path)
    from src.sm100.fwd.combine import combine

    return sparse.build_k2q_csr, combine


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


def _check_cache_view(
    tensor: torch.Tensor, *, name: str, pages: int, num_kv_heads: int, row_bytes: int
) -> None:
    """Token rows contiguous inside each head; head and page strides are free (16-byte multiples)."""
    shape = (pages, num_kv_heads, _PAGE_SIZE, row_bytes)
    if tensor.ndim != 4 or tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if tensor.stride(3) != 1 or tensor.stride(2) != row_bytes:
        raise ValueError(f"{name} must keep the 128 token rows of each head contiguous")
    if tensor.stride(1) % 16 or tensor.stride(0) % 16 or tensor.data_ptr() % 16:
        raise ValueError(f"{name} head/page strides and base must be 16-byte aligned")


def _check_global_scale(scale, *, name: str, device: torch.device) -> None:
    if scale is None:
        return
    if not isinstance(scale, torch.Tensor) or scale.numel() != 1 or scale.dtype != torch.float32:
        raise TypeError(f"{name} must be a one-element torch.float32 tensor")
    if scale.device != device:
        raise ValueError(f"{name} must be on {device}")


def _check_block_scale_shift(block_scale_shift: int) -> int:
    block_scale_shift = int(block_scale_shift)
    if not 0 <= block_scale_shift <= MAX_BLOCK_SCALE_SHIFT:
        raise ValueError(f"block_scale_shift must be in [0, {MAX_BLOCK_SCALE_SHIFT}]")
    return block_scale_shift


def _check_page_table(
    page_table: torch.Tensor, kv_indptr: torch.Tensor | None, *, batch: int
) -> None:
    """A [batch, max_pages] table (rows contiguous, any row stride) or a flat [total_pages] list
    with kv_indptr."""
    if page_table.dtype != torch.int32:
        raise TypeError("page_table must be torch.int32")
    if not page_table.is_cuda:
        raise ValueError("page_table must be a CUDA tensor")
    if page_table.ndim == 2:
        if kv_indptr is not None:
            raise ValueError("kv_indptr applies only to a flat [total_pages] page_table")
        if page_table.shape[0] != batch or page_table.shape[1] <= 0:
            raise ValueError("page_table must have shape [batch, max_pages]")
        if page_table.stride(1) != 1 or (batch > 1 and page_table.stride(0) < page_table.shape[1]):
            raise ValueError("page_table rows must be contiguous")
        return
    _check_cuda_contiguous(page_table, name="page_table")
    if page_table.ndim != 1 or page_table.numel() == 0:
        raise ValueError("page_table must be [batch, max_pages] or a flat [total_pages] list")
    if kv_indptr is None:
        raise ValueError("a flat [total_pages] page_table requires kv_indptr")
    _check_cuda_contiguous(kv_indptr, name="kv_indptr")
    _check_same_device(page_table, kv_indptr, name="kv_indptr")
    if kv_indptr.dtype != torch.int32 or tuple(kv_indptr.shape) != (batch + 1,):
        raise ValueError("kv_indptr must be torch.int32 with shape [batch + 1]")


def run_prefill(
    q: torch.Tensor,
    paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
    kv_cache_sf: tuple[torch.Tensor, torch.Tensor],
    *,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    schedule,
    sm_scale: float,
    topk: int,
    seqused_k: torch.Tensor | None = None,
    k_global_scale: torch.Tensor | None = None,
    v_global_scale: torch.Tensor | None = None,
    output_scale: float = 1.0,
    block_scale_shift: int = 0,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    o_partial: torch.Tensor | None = None,
    lse_partial: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the Q8KV4 forward over a prepared k2q CSR and schedule, then combine the splits.

    ``paged_kv_cache`` holds the packed E2M1 K/V data views ``[pages, Hkv, 128, 64]`` and
    ``kv_cache_sf`` the E4M3 block-scale views ``[pages, Hkv, 128, 8]`` (K scales linear, V
    scales in token-quad order, see :func:`interleave_v_scales`); all four may be strided views
    into vLLM's page layout. Returns ``(out, lse)`` with ``lse`` None unless ``lse`` is given.

    ``kv_indices`` maps each request's logical pages to physical pages: a ``[batch, max_pages]``
    int32 table (rows contiguous, any row stride, e.g. vLLM's block table) with ``kv_indptr``
    None, or a flat ``[total_pages]`` list with ``kv_indptr`` (``[batch + 1]``) giving each
    request's first entry.

    Values are ``E2M1(code) x E4M3(block_scale) x global_scale``: ``k_global_scale`` and
    ``v_global_scale`` are optional one-element fp32 CUDA tensors read on the device, so a
    captured graph follows their updates. ``block_scale_shift`` stages every block scale by
    ``2**-shift`` before ``code x scale`` is requantized to E4M3 (3 for TransformerEngine-style
    full-range scales, 0 when the products already fit). ``output_scale`` multiplies the output.

    ``topk`` is the width of the TopK lists the schedule was built from (one split per slot).
    ``seqused_k`` (``[batch]`` int32) replaces each request's KV length for masking and for the
    bottom-right causal alignment, as ``fmha_sm100``'s ``qo_offset`` does.
    """
    if len(paged_kv_cache) != 2 or len(kv_cache_sf) != 2:
        raise ValueError("paged_kv_cache and kv_cache_sf must be (K, V) pairs")
    k_cache, v_cache = paged_kv_cache
    k_scale, v_scale = kv_cache_sf
    if (
        schedule is None
        or schedule.scheduler_metadata is None
        or schedule.work_count is None
        or schedule.qsplit_indices is None
        or schedule.split_counts is None
    ):
        raise ValueError("run_prefill needs a forward schedule with split metadata")
    if q.dtype != torch.float8_e4m3fn:
        raise TypeError("q must be torch.float8_e4m3fn")
    _check_cuda_contiguous(q, name="q")
    _check_page_table(kv_indices, kv_indptr, batch=cu_seqlens_q.numel() - 1)
    _check_global_scale(k_global_scale, name="k_global_scale", device=q.device)
    _check_global_scale(v_global_scale, name="v_global_scale", device=q.device)
    block_scale_shift = _check_block_scale_shift(block_scale_shift)
    if topk not in SUPPORTED_TOPK:
        raise ValueError(f"topk must be one of {SUPPORTED_TOPK}, got {topk}")
    total_q, num_q_heads, _ = q.shape
    num_kv_heads = int(k_cache.shape[1]) if k_cache.ndim == 4 else -1
    pages = int(k_cache.shape[0])
    for name, tensor, row_bytes in (
        ("k_cache", k_cache, _KV_DATA_ROW_BYTES),
        ("v_cache", v_cache, _KV_DATA_ROW_BYTES),
        ("k_scale", k_scale, _KV_SCALE_ROW_BYTES),
        ("v_scale", v_scale, _KV_SCALE_ROW_BYTES),
    ):
        _check_same_device(q, tensor, name=name)
        _check_cache_view(
            tensor, name=name, pages=pages, num_kv_heads=num_kv_heads, row_bytes=row_bytes
        )
    if k_cache.dtype != torch.uint8 or v_cache.dtype != torch.uint8:
        raise TypeError("k_cache and v_cache must be torch.uint8")
    if num_q_heads != num_kv_heads * _Q_HEADS_PER_KV:
        raise ValueError(f"q must have {num_kv_heads * _Q_HEADS_PER_KV} heads (16 per KV head)")

    options = dict(device=q.device)
    if o_partial is None:
        o_partial = torch.empty(
            (topk, total_q, num_q_heads, _HEAD_DIM), dtype=torch.bfloat16, **options
        )
    if lse_partial is None:
        lse_partial = torch.empty((topk, total_q, num_q_heads), dtype=torch.float32, **options)
    if out is None:
        out = torch.empty((total_q, num_q_heads, _HEAD_DIM), dtype=torch.bfloat16, **options)

    _, combine = _sparse_stack()
    load_extension(q.device, block_scale_shift).run(
        q,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        kv_indices,
        kv_indptr,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        k2q_row_ptr,
        schedule.qsplit_indices,
        schedule.scheduler_metadata,
        schedule.work_count,
        o_partial,
        lse_partial,
        k_global_scale,
        v_global_scale,
        float(sm_scale),
        float(output_scale),
    )
    combine(
        o_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens=cu_seqlens_q,
        split_counts=schedule.split_counts,
        use_pdl=True,
    )
    return out, lse


@dataclass(frozen=True)
class _PlanState:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    kv_indices: torch.Tensor
    kv_indptr: torch.Tensor | None
    seqused_k: torch.Tensor | None
    topk: int
    k2q_row_ptr: torch.Tensor
    schedule: object
    sm_scale: float
    block_scale_shift: int
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
        kv_indptr: torch.Tensor | None = None,
        causal: bool = True,
        sm_scale: float | None = None,
        block_scale_shift: int = 0,
        seqused_k: torch.Tensor | None = None,
    ) -> None:
        """Prepare reusable q2k-to-k2q metadata and split-combine workspace.

        ``page_table`` maps each request's logical pages to physical pages, either as a
        ``[batch, max_pages]`` table (rows contiguous, any row stride) or as a flat
        ``[total_pages]`` list with ``kv_indptr`` (``[batch + 1]``) giving each request's first
        entry. ``topk_indices`` is
        ``[num_kv_heads, total_q, topk]`` with ``topk`` in ``SUPPORTED_TOPK``. See
        :func:`run_prefill` for ``block_scale_shift`` and ``seqused_k``.
        """

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("plan() must be called outside CUDA Graph capture")
        if not causal:
            raise NotImplementedError("Q8KV4 prefill v1 supports only causal attention")
        _check_same_device(topk_indices, page_table, name="page_table")
        batch = cu_seqlens_q.numel() - 1
        _check_page_table(page_table, kv_indptr, batch=batch)
        if topk_indices.dtype != torch.int32:
            raise TypeError("topk_indices must be torch.int32")
        if (
            topk_indices.ndim != 3
            or topk_indices.shape[0] <= 0
            or topk_indices.shape[2] not in SUPPORTED_TOPK
        ):
            raise ValueError(
                f"topk_indices must have shape [num_kv_heads, total_q, topk], topk in {SUPPORTED_TOPK}"
            )
        _check_cuda_contiguous(topk_indices, name="topk_indices")

        build_k2q_csr, _ = _sparse_stack()
        k2q_row_ptr, _, schedule = build_k2q_csr(
            topk_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            _PAGE_SIZE,
            total_k=int(total_k),
            max_seqlen_k=int(max_seqlen_k),
            max_seqlen_q=int(max_seqlen_q),
            total_rows=int(total_rows),
            qhead_per_kv=_Q_HEADS_PER_KV,
            return_schedule=True,
        )
        total_q = int(topk_indices.shape[1])
        topk = int(topk_indices.shape[2])
        num_q_heads = int(topk_indices.shape[0]) * _Q_HEADS_PER_KV
        if seqused_k is not None:
            _check_cuda_contiguous(seqused_k, name="seqused_k")
            if seqused_k.dtype != torch.int32 or tuple(seqused_k.shape) != (batch,):
                raise ValueError("seqused_k must be torch.int32 with shape [batch]")
        options = dict(device=topk_indices.device)
        scale = 1.0 / math.sqrt(_HEAD_DIM) if sm_scale is None else float(sm_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("sm_scale must be finite and positive")
        block_scale_shift = _check_block_scale_shift(block_scale_shift)
        self._plan_state = _PlanState(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            kv_indices=page_table,
            kv_indptr=kv_indptr,
            seqused_k=seqused_k,
            topk=topk,
            k2q_row_ptr=k2q_row_ptr,
            schedule=schedule,
            sm_scale=scale,
            block_scale_shift=block_scale_shift,
            o_partial=torch.empty(
                (topk, total_q, num_q_heads, _HEAD_DIM),
                dtype=torch.bfloat16,
                **options,
            ),
            lse_partial=torch.empty(
                (topk, total_q, num_q_heads),
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
        kv_global_scale: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
        out: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run one layer with the metadata and workspace cached by ``plan``.

        K/V data and scales may be strided views (see :func:`run_prefill`); V block scales are
        read in token-quad order. ``kv_global_scale`` is an optional ``(K, V)`` pair of
        one-element fp32 CUDA tensors.
        """

        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        if q.shape != state.out.shape:
            raise ValueError(f"q must have shape {tuple(state.out.shape)}")

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

        run_prefill(
            q,
            paged_kv_cache,
            kv_cache_sf,
            kv_indices=state.kv_indices,
            kv_indptr=state.kv_indptr,
            cu_seqlens_q=state.cu_seqlens_q,
            cu_seqlens_k=state.cu_seqlens_k,
            k2q_row_ptr=state.k2q_row_ptr,
            schedule=state.schedule,
            sm_scale=state.sm_scale,
            topk=state.topk,
            seqused_k=state.seqused_k,
            k_global_scale=None if kv_global_scale is None else kv_global_scale[0],
            v_global_scale=None if kv_global_scale is None else kv_global_scale[1],
            block_scale_shift=state.block_scale_shift,
            out=out_tensor,
            lse=lse_tensor if return_lse or lse is not None else None,
            o_partial=state.o_partial,
            lse_partial=state.lse_partial,
        )
        if return_lse:
            return out_tensor, lse_tensor
        return out_tensor


__all__ = [
    "SUPPORTED_TOPK",
    "BatchPrefillWithPagedKVCacheWrapper",
    "interleave_v_scales",
    "run_prefill",
]
