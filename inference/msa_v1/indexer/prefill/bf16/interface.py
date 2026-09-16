"""FlashInfer-style BF16 paged-prefill indexer wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

from inference.msa_v1.aot_cache import compile_or_load
from inference.msa_v1.attention.prefill._common.dsl.compile_utils import (
    compile_with_timing,
)
from inference.msa_v1.indexer._common.topk_select import _topk_select
from inference.msa_v1.indexer.prefill.bf16.indexer_gemm import (
    M3_PAGED_DIRECT_SCORE_NUM_HEADS,
    M3IndexerGemmSm100,
)
from inference.msa_v1.indexer.prefill.bf16.schedule import M3IndexerScheduleSm100


_HEAD_DIM = 128
_PAGE_SIZE = 128
_TOP_K = 16
_MAXIMUM_COLUMNS = 8192
_SUPPORTED_CAPABILITIES = {(10, 0), (10, 3)}
_SCHEDULE_COMPILE_CACHE: dict[tuple[object, ...], object] = {}
_GEMM_COMPILE_CACHE: dict[tuple[object, ...], object] = {}


def _ceil_div(dividend: int, divisor: int) -> int:
    return (dividend + divisor - 1) // divisor


def _to_cute_tensor(
    tensor: torch.Tensor,
    *,
    leading_dim: int = -1,
) -> cute.Tensor:
    if leading_dim < 0:
        leading_dim += tensor.ndim
    if tensor.stride(leading_dim) != 1:
        if tensor.shape[leading_dim] != 1:
            raise ValueError("the leading tensor dimension must have unit stride")
        # PyTorch can retain an arbitrary stride for a singleton dimension while
        # still reporting the tensor as contiguous.  Use an equivalent compile
        # exemplar; the public run path continues to pass the original tensor.
        tensor = torch.empty_like(tensor, memory_format=torch.contiguous_format)
    return from_dlpack(
        tensor.detach(),
        assumed_align=16,
        enable_tvm_ffi=True,
    ).mark_layout_dynamic(leading_dim=leading_dim)


def _check_runtime(device: torch.device) -> tuple[int, int]:
    capability = torch.cuda.get_device_capability(device)
    if capability not in _SUPPORTED_CAPABILITIES:
        raise RuntimeError(
            "BF16 paged prefill indexer supports only SM100 and SM103, "
            f"got SM{capability[0]}{capability[1]}"
        )
    return capability


def _check_cuda_contiguous(tensor: torch.Tensor, *, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
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


def _num_task_slots(total_q: int, batch: int, num_index_heads: int) -> int:
    logical_q_tile = 256 // num_index_heads
    return (total_q + batch * (logical_q_tile - 1)) // logical_q_tile


def _validate_plan_inputs(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    *,
    total_q: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_index_heads: int,
) -> tuple[int, int]:
    for name, tensor in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
        ("page_table", page_table),
    ):
        _check_cuda_contiguous(tensor, name=name)
        _check_same_device(cu_seqlens_q, tensor, name=name)
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be torch.int32")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens_q must have shape [batch + 1]")
    if cu_seqlens_k.shape != cu_seqlens_q.shape:
        raise ValueError("cu_seqlens_k must match cu_seqlens_q shape")
    batch = int(cu_seqlens_q.numel() - 1)
    if page_table.ndim != 2 or page_table.shape[0] != batch:
        raise ValueError("page_table must have shape [batch, max_pages]")
    if num_index_heads not in M3_PAGED_DIRECT_SCORE_NUM_HEADS:
        raise ValueError(
            f"num_index_heads must be one of {M3_PAGED_DIRECT_SCORE_NUM_HEADS}"
        )
    total_q = int(total_q)
    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_k = int(max_seqlen_k)
    if min(total_q, max_seqlen_q, max_seqlen_k) <= 0:
        raise ValueError("total_q and maximum sequence lengths must be positive")
    max_cols = _ceil_div(max_seqlen_k, _PAGE_SIZE)
    if not 0 < max_cols <= _MAXIMUM_COLUMNS:
        raise ValueError(f"maximum page count must be in [1, {_MAXIMUM_COLUMNS}]")
    if page_table.shape[1] < max_cols:
        raise ValueError("page_table capacity is smaller than max_seqlen_k")
    return batch, max_cols


def _make_num_valid_pages(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    total_q: int,
    num_index_heads: int,
    max_cols: int,
) -> torch.Tensor:
    batch = cu_seqlens_q.numel() - 1
    q_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    batch_indices = torch.repeat_interleave(
        torch.arange(batch, dtype=torch.int32, device=cu_seqlens_q.device),
        q_lengths,
        output_size=total_q,
    )
    query_indices = torch.arange(
        total_q,
        dtype=torch.int32,
        device=cu_seqlens_q.device,
    )
    q_local = query_indices - cu_seqlens_q[batch_indices]
    seq_q = q_lengths[batch_indices]
    k_lengths = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    seq_k = k_lengths[batch_indices]
    lengths = (
        torch.div(
            seq_k - seq_q + q_local,
            _PAGE_SIZE,
            rounding_mode="floor",
        )
        + 1
    )
    lengths = lengths.clamp_(min=1, max=max_cols)
    return lengths.unsqueeze(0).expand(num_index_heads, -1).contiguous()


@dataclass(frozen=True, slots=True)
class _PlanState:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    page_table: torch.Tensor
    task_batch_idx: torch.Tensor
    task_q_local_begin: torch.Tensor
    proxy_scores: torch.Tensor
    num_valid_pages: torch.Tensor
    topk_indices: torch.Tensor
    num_task_slots: int
    total_q: int
    max_cols: int
    num_index_heads: int


def _compile_schedule(state: _PlanState) -> object:
    capability = _check_runtime(state.page_table.device)
    key = (
        "msa_v1_indexer_prefill_bf16_schedule_sm100",
        capability,
        state.num_index_heads,
    )
    compiled = _SCHEDULE_COMPILE_CACHE.get(key)
    if compiled is not None:
        return compiled
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "warm up the BF16 indexer schedule before CUDA Graph capture"
        )
    q_per_cluster = 256 // state.num_index_heads
    kernel = M3IndexerScheduleSm100(q_per_cluster=q_per_cluster)
    compiled = compile_or_load(
        key,
        lambda: compile_with_timing(
            kernel,
            _to_cute_tensor(state.cu_seqlens_q),
            _to_cute_tensor(state.cu_seqlens_k),
            _to_cute_tensor(state.task_batch_idx),
            _to_cute_tensor(state.task_q_local_begin),
            cutlass.Int32(state.cu_seqlens_q.numel() - 1),
            cutlass.Int32(state.num_task_slots),
            None,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        ),
        log_prefix="msa_v1_indexer_prefill_bf16_schedule",
    )
    _SCHEDULE_COMPILE_CACHE[key] = compiled
    return compiled


def _run_schedule(state: _PlanState) -> None:
    compiled = _compile_schedule(state)
    with torch.cuda.nvtx.range("PrefillIndexerBF16Schedule"):
        compiled(
            state.cu_seqlens_q,
            state.cu_seqlens_k,
            state.task_batch_idx,
            state.task_q_local_begin,
            state.cu_seqlens_q.numel() - 1,
            state.num_task_slots,
            None,
        )


def _compile_gemm(
    q_flat: torch.Tensor,
    k_tma_view: torch.Tensor,
    state: _PlanState,
) -> object:
    capability = _check_runtime(q_flat.device)
    key = (
        "msa_v1_indexer_prefill_bf16_gemm_sm100",
        capability,
        state.num_index_heads,
        torch.bfloat16,
        torch.float32,
    )
    compiled = _GEMM_COMPILE_CACHE.get(key)
    if compiled is not None:
        return compiled
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm up the BF16 indexer GEMM before CUDA Graph capture")
    kernel = M3IndexerGemmSm100(
        compute_capability=capability, num_index_heads=state.num_index_heads
    )
    compiled = compile_or_load(
        key,
        lambda: compile_with_timing(
            kernel,
            _to_cute_tensor(q_flat),
            _to_cute_tensor(k_tma_view, leading_dim=1),
            _to_cute_tensor(state.page_table),
            _to_cute_tensor(state.proxy_scores),
            _to_cute_tensor(state.cu_seqlens_q),
            _to_cute_tensor(state.cu_seqlens_k),
            _to_cute_tensor(state.task_batch_idx),
            _to_cute_tensor(state.task_q_local_begin),
            None,
            cutlass.Int32(state.num_task_slots),
            cutlass.Float32(1.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        ),
        log_prefix="msa_v1_indexer_prefill_bf16_gemm",
    )
    _GEMM_COMPILE_CACHE[key] = compiled
    return compiled


class BatchPrefillIndexerWithPagedKVCacheWrapper:
    """Compose BF16 paged proxy scores with forced-tail TopK selection."""

    def __init__(self) -> None:
        self._plan_state: _PlanState | None = None

    def plan(
        self,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        page_table: torch.Tensor,
        *,
        total_q: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        num_index_heads: int = 4,
    ) -> None:
        """Prepare reusable varlen tasks and graph-safe output buffers."""

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("plan() must be called outside CUDA Graph capture")
        num_index_heads = int(num_index_heads)
        batch, max_cols = _validate_plan_inputs(
            cu_seqlens_q,
            cu_seqlens_k,
            page_table,
            total_q=total_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_index_heads=num_index_heads,
        )
        total_q = int(total_q)
        num_task_slots = _num_task_slots(total_q, batch, num_index_heads)
        options = {"device": page_table.device}
        state = _PlanState(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table=page_table,
            task_batch_idx=torch.empty(num_task_slots, dtype=torch.int32, **options),
            task_q_local_begin=torch.empty(
                num_task_slots,
                dtype=torch.int32,
                **options,
            ),
            proxy_scores=torch.empty(
                (num_index_heads, total_q, max_cols),
                dtype=torch.float32,
                **options,
            ),
            num_valid_pages=_make_num_valid_pages(
                cu_seqlens_q,
                cu_seqlens_k,
                total_q=total_q,
                num_index_heads=num_index_heads,
                max_cols=max_cols,
            ),
            topk_indices=torch.empty(
                (num_index_heads, total_q, _TOP_K),
                dtype=torch.int32,
                **options,
            ),
            num_task_slots=num_task_slots,
            total_q=total_q,
            max_cols=max_cols,
            num_index_heads=num_index_heads,
        )
        self._plan_state = state
        _run_schedule(state)

    def replan(self) -> None:
        """Refresh device metadata after in-place length updates."""

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("replan() must be called outside CUDA Graph capture")
        state = self._require_plan()
        state.num_valid_pages.copy_(
            _make_num_valid_pages(
                state.cu_seqlens_q,
                state.cu_seqlens_k,
                total_q=state.total_q,
                num_index_heads=state.num_index_heads,
                max_cols=state.max_cols,
            )
        )
        _run_schedule(state)

    def run(
        self,
        q: torch.Tensor,
        paged_k_cache: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return forced-tail TopK logical page indices for one layer."""

        state = self._require_plan()
        for name, tensor in (("q", q), ("paged_k_cache", paged_k_cache)):
            _check_cuda_contiguous(tensor, name=name)
            _check_same_device(state.page_table, tensor, name=name)
            if tensor.dtype != torch.bfloat16:
                raise TypeError(f"{name} must be torch.bfloat16")
        expected_q_shape = (state.total_q, state.num_index_heads, _HEAD_DIM)
        if tuple(q.shape) != expected_q_shape:
            raise ValueError(f"q must have shape {expected_q_shape}")
        if paged_k_cache.ndim != 4 or tuple(paged_k_cache.shape[1:]) != (
            1,
            _PAGE_SIZE,
            _HEAD_DIM,
        ):
            raise ValueError(
                "paged_k_cache must have shape [physical_pages, 1, 128, 128]"
            )
        out_tensor = state.topk_indices if out is None else out
        _check_cuda_contiguous(out_tensor, name="out")
        _check_same_device(q, out_tensor, name="out")
        if (
            out_tensor.dtype != torch.int32
            or out_tensor.shape != state.topk_indices.shape
        ):
            raise ValueError(
                f"out must be torch.int32 with shape {tuple(state.topk_indices.shape)}"
            )

        q_flat = q.view(state.total_q * state.num_index_heads, _HEAD_DIM)
        k_tma_view = paged_k_cache[:, 0].permute(1, 2, 0)
        compiled = _compile_gemm(q_flat, k_tma_view, state)
        with torch.cuda.nvtx.range("PrefillIndexerBF16ProxyScore"):
            compiled(
                q_flat,
                k_tma_view,
                state.page_table,
                state.proxy_scores,
                state.cu_seqlens_q,
                state.cu_seqlens_k,
                state.task_batch_idx,
                state.task_q_local_begin,
                None,
                state.num_task_slots,
                1.0,
            )
        with torch.cuda.nvtx.range("PrefillIndexerBF16TopKSelect"):
            _topk_select(
                state.proxy_scores.view(-1, state.max_cols),
                state.num_valid_pages.view(-1),
                out=out_tensor.view(-1, _TOP_K),
            )
        return out_tensor

    def _require_plan(self) -> _PlanState:
        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        return state


__all__ = ["BatchPrefillIndexerWithPagedKVCacheWrapper"]
