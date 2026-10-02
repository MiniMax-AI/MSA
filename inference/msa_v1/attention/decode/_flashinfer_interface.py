"""Shared FlashInfer-backed paged sparse decode attention lifecycle."""

from __future__ import annotations

import math

import torch

from inference.msa_v1.attention.decode._flashinfer import (
    FlashInferPlan,
    _check_cuda_contiguous,
    _check_cuda_tensor,
    _check_same_device,
    make_flashinfer_plan,
    run_flashinfer,
)

_HEAD_DIM = 128
_DEFAULT_NUM_Q_HEADS = 64
_DEFAULT_NUM_KV_HEADS = 4
_PAGE_SIZE = 128
_MAX_TOPK = 16


class _BatchDecodeWithPagedKVCacheWrapperBase:
    """Adapt per-query sparse metadata to the external FlashInfer backend."""

    storage_dtype: torch.dtype

    def __init__(
        self,
        workspace_buffer: torch.Tensor | None = None,
        *,
        enable_pdl: bool = True,
    ) -> None:
        if not isinstance(enable_pdl, bool):
            raise TypeError("enable_pdl must be a bool")
        self._enable_pdl = enable_pdl
        self._workspace_buffer = workspace_buffer
        self._plan_state: FlashInferPlan | None = None
        self._warmed = False

    def plan(
        self,
        topk_indices: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        q_len_per_req: int,
        num_q_heads: int = _DEFAULT_NUM_Q_HEADS,
        num_kv_heads: int = _DEFAULT_NUM_KV_HEADS,
        sm_scale: float | None = None,
    ) -> None:
        """Build query-expanded physical block tables outside Graph capture."""

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("plan() must be called outside CUDA Graph capture")
        for name, tensor in (
            ("topk_indices", topk_indices),
            ("page_table", page_table),
            ("seq_lens", seq_lens),
        ):
            _check_cuda_contiguous(tensor, name=name)
            _check_same_device(topk_indices, tensor, name=name)
            if tensor.dtype != torch.int32:
                raise TypeError(f"{name} must have dtype torch.int32")
        if page_table.ndim != 2 or page_table.shape[0] <= 0 or page_table.shape[1] <= 0:
            raise ValueError("page_table must have shape [batch, max_pages]")
        batch_size = page_table.shape[0]
        q_len_per_req = int(q_len_per_req)
        num_q_heads = int(num_q_heads)
        num_kv_heads = int(num_kv_heads)
        if q_len_per_req <= 0:
            raise ValueError("q_len_per_req must be positive")
        if num_q_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("num_q_heads and num_kv_heads must be positive")
        if num_q_heads % num_kv_heads or num_q_heads // num_kv_heads not in (8, 16):
            raise ValueError("Sparse decode requires 8 or 16 Q heads per KV head")
        if seq_lens.shape != (batch_size,):
            raise ValueError("seq_lens must have shape [batch]")
        expected_prefix = (batch_size * q_len_per_req, num_kv_heads)
        if topk_indices.ndim != 3 or tuple(topk_indices.shape[:2]) != expected_prefix:
            raise ValueError(
                "topk_indices must have shape [batch * q_len_per_req, num_kv_heads, topk]"
            )
        if topk_indices.shape[2] <= 0 or topk_indices.shape[2] > _MAX_TOPK:
            raise ValueError(f"topk capacity must be in [1, {_MAX_TOPK}]")
        scale = 1.0 / math.sqrt(_HEAD_DIM) if sm_scale is None else float(sm_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("sm_scale must be finite and positive")

        self._plan_state = make_flashinfer_plan(
            topk_indices,
            page_table,
            seq_lens,
            q_len_per_req=q_len_per_req,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            sm_scale=scale,
            workspace_buffer=self._workspace_buffer,
        )
        self._warmed = False

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run external FlashInfer with each MSA query represented as one request."""

        if self._plan_state is None:
            raise RuntimeError("plan() must be called before run()")
        state = self._plan_state
        _check_cuda_contiguous(q, name="q")
        if q.dtype != self.storage_dtype:
            raise TypeError(f"q must have dtype {self.storage_dtype}")
        if tuple(q.shape) != (state.total_q, state.num_q_heads, _HEAD_DIM):
            raise ValueError(
                f"q must have shape {(state.total_q, state.num_q_heads, _HEAD_DIM)}"
            )
        _check_same_device(state.block_tables, q, name="q")
        if not isinstance(paged_kv_cache, (tuple, list)) or len(paged_kv_cache) != 2:
            raise TypeError("paged_kv_cache must be a (k_cache, v_cache) pair")
        k_cache, v_cache = paged_kv_cache
        for name, tensor in (("k_cache", k_cache), ("v_cache", v_cache)):
            _check_cuda_tensor(tensor, name=name)
            _check_same_device(q, tensor, name=name)
            if tensor.dtype != self.storage_dtype:
                raise TypeError(f"{name} must have dtype {self.storage_dtype}")
            if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (
                state.num_kv_heads,
                _PAGE_SIZE,
                _HEAD_DIM,
            ):
                raise ValueError(
                    f"{name} must have shape [pages, {state.num_kv_heads}, "
                    f"{_PAGE_SIZE}, {_HEAD_DIM}]"
                )
            if tensor.stride(-1) != 1 or tensor.stride(-2) != _HEAD_DIM:
                raise ValueError(f"{name} token and head-dim axes must be contiguous")
            if tensor.shape[0] <= state.max_source_page:
                raise ValueError(f"{name} does not cover every planned physical page")
        if k_cache.shape != v_cache.shape or k_cache.stride() != v_cache.stride():
            raise ValueError("K and V cache shapes and strides must match")

        output = state.out if out is None else out
        _check_cuda_contiguous(output, name="out")
        _check_same_device(q, output, name="out")
        if output.dtype != torch.bfloat16:
            raise TypeError("out must have dtype torch.bfloat16")
        if output.shape != q.shape:
            raise ValueError("out shape must match q")
        if torch.cuda.is_current_stream_capturing() and not self._warmed:
            raise RuntimeError("call run() once before CUDA Graph capture")

        result = run_flashinfer(
            state, q, k_cache, v_cache, output, enable_pdl=self._enable_pdl
        )
        self._warmed = True
        return result
