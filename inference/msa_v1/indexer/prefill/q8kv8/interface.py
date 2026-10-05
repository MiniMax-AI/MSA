"""FlashInfer-style FP8 paged-prefill indexer wrapper."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import cutlass.cute as cute
import torch

from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from inference.msa_v1._build_utils import cutlass_root
from inference.msa_v1.aot_cache import compile_or_load
from inference.msa_v1.indexer._common.topk_select import _topk_select
from inference.msa_v1.indexer.prefill.q8kv8.indexer_gemm import (
    PrefillIndexerGemmSm100,
)
from inference.msa_v1.indexer.prefill.q8kv8.plan import (
    PrefillIndexerPlanBuild,
    PrefillIndexerPlanReset,
)


logger = logging.getLogger(__name__)

_TOP_K = 16
_PLAN_TASK_CAPACITY_PAGE_CHUNK = 4
_SUPPORTED_CAPABILITIES = PrefillIndexerGemmSm100.supported_compute_capabilities
_COMPILE_CACHE: dict[tuple[object, ...], object] = {}


def _ceil_div(dividend: int, divisor: int) -> int:
    return (dividend + divisor - 1) // divisor


def _to_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    leading_dim = tensor.ndim - 1
    if tensor.stride(leading_dim) != 1:
        # Degenerate singleton tensors can carry zero strides despite being
        # contiguous. Compilation only needs an equivalent layout exemplar.
        tensor = torch.empty_like(tensor, memory_format=torch.contiguous_format)
    return from_dlpack(
        tensor.detach(),
        assumed_align=16,
        enable_tvm_ffi=True,
    ).mark_layout_dynamic(leading_dim=leading_dim)


def _check_cuda_contiguous(tensor: torch.Tensor, *, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
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


def _check_runtime(device: torch.device) -> tuple[int, int]:
    os.environ.setdefault("CUTLASS_PATH", str(cutlass_root()))
    capability = torch.cuda.get_device_capability(device)
    if capability not in _SUPPORTED_CAPABILITIES:
        raise RuntimeError(
            "Q8K8 prefill indexer supports only SM100, SM103 and SM107, "
            f"got SM{capability[0]}{capability[1]}"
        )
    return capability


def _get_num_persistent_clusters(device: torch.device) -> int:
    """Return the number of resident 2-CTA clusters on the selected device."""
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    num_clusters = sm_count // PrefillIndexerGemmSm100.cta_group_size
    if num_clusters <= 0:
        raise RuntimeError("Q8K8 prefill indexer requires at least one 2-CTA cluster")
    return num_clusters


@dataclass(frozen=True, slots=True)
class _PlanState:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    page_table: torch.Tensor
    task_descriptors: torch.Tensor
    task_counts: torch.Tensor
    plan_error: torch.Tensor
    task_capacity: int
    num_candidate_q_tiles: int
    max_page_chunk: int
    total_q: int
    max_cols: int
    num_index_heads: int
    proxy_scores: torch.Tensor
    num_valid_pages: torch.Tensor
    topk_indices: torch.Tensor


def _validate_plan_inputs(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    *,
    total_q: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> tuple[int, int]:
    for name, tensor in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
        ("page_table", page_table),
    ):
        _check_cuda_contiguous(tensor, name=name)
        _check_same_device(cu_seqlens_q, tensor, name=name)
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must have dtype torch.int32")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens_q must have shape [batch + 1]")
    if cu_seqlens_k.shape != cu_seqlens_q.shape:
        raise ValueError("cu_seqlens_k must match cu_seqlens_q shape")
    batch = cu_seqlens_q.numel() - 1
    if page_table.ndim != 2 or page_table.shape[0] != batch:
        raise ValueError("page_table must have shape [batch, max_pages]")
    sizes = {
        "total_q": int(total_q),
        "max_seqlen_q": int(max_seqlen_q),
        "max_seqlen_k": int(max_seqlen_k),
    }
    if sizes["total_q"] <= 0:
        raise ValueError("total_q must be positive")
    if sizes["max_seqlen_q"] <= 0 or sizes["max_seqlen_k"] <= 0:
        raise ValueError("maximum sequence lengths must be positive")
    if sizes["max_seqlen_k"] < sizes["max_seqlen_q"]:
        raise ValueError("bottom-right causal requires max_seqlen_k >= max_seqlen_q")
    max_cols = _ceil_div(
        sizes["max_seqlen_k"],
        PrefillIndexerPlanBuild.page_size,
    )
    if page_table.shape[1] < max_cols:
        raise ValueError("page_table capacity is smaller than max_seqlen_k")
    if max_cols > 8192:
        raise ValueError("max_cols exceeds the internal TopK limit")
    return batch, max_cols


def _compile_device_plan(state: _PlanState) -> tuple[object, object]:
    capability = _check_runtime(state.page_table.device)
    reset_kernel = PrefillIndexerPlanReset()
    build_kernel = PrefillIndexerPlanBuild(state.num_index_heads)
    plan_signature = (
        capability,
        build_kernel.num_buckets,
        build_kernel.q_tile,
        build_kernel.split_page_chunk,
        build_kernel.large_page_chunk,
        build_kernel.split_q_tile_threshold,
        build_kernel.descriptor_words,
    )
    reset_key = (
        "msa_v1_indexer_prefill_q8kv8_multihead_v1_plan_reset_sm100",
        capability,
        reset_kernel.num_buckets,
    )
    build_key = (
        "msa_v1_indexer_prefill_q8kv8_multihead_v1_plan_build_sm100",
        state.num_index_heads,
        *plan_signature,
    )
    reset_compiled = _COMPILE_CACHE.get(reset_key)
    build_compiled = _COMPILE_CACHE.get(build_key)
    if reset_compiled is not None and build_compiled is not None:
        return reset_compiled, build_compiled
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm up the indexer device plan before CUDA Graph capture")

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    if reset_compiled is None:

        def _compile_reset():
            started_at = time.perf_counter()
            compiled = cute.compile(
                reset_kernel,
                _to_cute_tensor(state.task_counts),
                _to_cute_tensor(state.plan_error),
                stream,
                options="--enable-tvm-ffi",
            )
            logger.info(
                "[PrefillIndexerPlanReset] Compiled in %.1fs",
                time.perf_counter() - started_at,
            )
            return compiled

        reset_compiled = compile_or_load(
            reset_key,
            _compile_reset,
            log_prefix="msa_v1_indexer_prefill_plan_reset",
        )
        _COMPILE_CACHE[reset_key] = reset_compiled

    if build_compiled is None:

        def _compile_build():
            started_at = time.perf_counter()
            compiled = cute.compile(
                build_kernel,
                _to_cute_tensor(state.cu_seqlens_q),
                _to_cute_tensor(state.cu_seqlens_k),
                _to_cute_tensor(state.num_valid_pages),
                _to_cute_tensor(state.task_descriptors),
                _to_cute_tensor(state.task_counts),
                _to_cute_tensor(state.plan_error),
                Int32(state.num_candidate_q_tiles),
                Int32(state.task_capacity),
                Int32(state.max_page_chunk),
                stream,
                options="--enable-tvm-ffi",
            )
            logger.info(
                "[PrefillIndexerPlanBuild] Compiled in %.1fs",
                time.perf_counter() - started_at,
            )
            return compiled

        build_compiled = compile_or_load(
            build_key,
            _compile_build,
            log_prefix="msa_v1_indexer_prefill_plan_build",
        )
        _COMPILE_CACHE[build_key] = build_compiled

    return reset_compiled, build_compiled


def _run_device_plan(state: _PlanState) -> None:
    reset_compiled, build_compiled = _compile_device_plan(state)
    with torch.cuda.nvtx.range("PrefillIndexerQ8KV8Plan"):
        reset_compiled(state.task_counts, state.plan_error)
        build_compiled(
            state.cu_seqlens_q,
            state.cu_seqlens_k,
            state.num_valid_pages,
            state.task_descriptors,
            state.task_counts,
            state.plan_error,
            state.num_candidate_q_tiles,
            state.task_capacity,
            state.max_page_chunk,
        )


def _compile_kernel(
    q: torch.Tensor,
    paged_k_cache: torch.Tensor,
    state: _PlanState,
    proxy_scores: torch.Tensor,
) -> object:
    capability = _check_runtime(q.device)
    num_persistent_clusters = _get_num_persistent_clusters(q.device)
    key = (
        "msa_v1_indexer_prefill_q8kv8_multihead_v1_gemm_sm100",
        state.num_index_heads,
        capability,
        num_persistent_clusters,
        torch.float8_e4m3fn,
        torch.float32,
        PrefillIndexerGemmSm100.head_dim,
        PrefillIndexerGemmSm100.q_tile,
        PrefillIndexerGemmSm100.k_tile,
        PrefillIndexerGemmSm100.q_stages,
        PrefillIndexerGemmSm100.k_stages,
        PrefillIndexerGemmSm100.acc_stages,
        PrefillIndexerGemmSm100.num_task_buckets,
    )
    compiled = _COMPILE_CACHE.get(key)
    if compiled is not None:
        return compiled
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm up the proxy-score stage before CUDA Graph capture")

    def _do_compile():
        kernel = PrefillIndexerGemmSm100(
            compute_capability=capability,
            num_persistent_clusters=num_persistent_clusters,
            num_index_heads=state.num_index_heads,
        )
        compile_args = (
            _to_cute_tensor(q),
            _to_cute_tensor(paged_k_cache),
            _to_cute_tensor(state.page_table),
            _to_cute_tensor(proxy_scores),
            _to_cute_tensor(state.task_descriptors),
            _to_cute_tensor(state.task_counts),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        )
        started_at = time.perf_counter()
        compiled_kernel = cute.compile(
            kernel,
            *compile_args,
            options="--enable-tvm-ffi",
        )
        elapsed_seconds = time.perf_counter() - started_at
        logger.info("[PrefillIndexerGemmSm100] Compiled in %.1fs", elapsed_seconds)
        return compiled_kernel

    compiled = compile_or_load(
        key,
        _do_compile,
        log_prefix="msa_v1_indexer_prefill_gemm",
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


class _BatchPrefillProxyScoreWrapper:
    """Manage the private true-varlen paged proxy-score stage."""

    def __init__(self) -> None:
        self._plan_state: _PlanState | None = None

    @property
    def plan_state(self) -> _PlanState:
        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before accessing plan_state")
        return state

    def plan(
        self,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        page_table: torch.Tensor,
        *,
        total_q: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        num_index_heads: int = 1,
    ) -> None:
        """Bind capacity, allocate reusable outputs, and build device tasks."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("plan() must be called outside CUDA Graph capture")
        if type(num_index_heads) is not int or num_index_heads not in (1, 2, 4):
            raise ValueError("num_index_heads must be 1, 2, or 4")
        batch, max_cols = _validate_plan_inputs(
            cu_seqlens_q,
            cu_seqlens_k,
            page_table,
            total_q=total_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
        options = {"device": page_table.device}
        total_q = int(total_q)
        max_seqlen_q = int(max_seqlen_q)
        logical_q_tile = PrefillIndexerGemmSm100.q_tile // num_index_heads
        q_tile_capacity = _ceil_div(total_q, logical_q_tile) + batch - 1
        page_chunk_capacity = _ceil_div(
            max_cols,
            _PLAN_TASK_CAPACITY_PAGE_CHUNK,
        )
        task_capacity = q_tile_capacity * page_chunk_capacity
        # Cap the planner's page chunk so every Q tile splits into at most
        # clusters // q_tiles tasks: short-query requests then fill the machine
        # in one wave instead of leaving most clusters idle. Host-known upper
        # bounds only (no D2H); long-query requests keep the default chunk.
        if _check_runtime(page_table.device) == (10, 7):
            chunks_per_q_tile = max(
                1,
                _get_num_persistent_clusters(page_table.device) // q_tile_capacity,
            )
            max_page_chunk = max(
                PrefillIndexerPlanBuild.min_page_chunk,
                min(
                    PrefillIndexerPlanBuild.large_page_chunk,
                    _ceil_div(max_cols, chunks_per_q_tile),
                ),
            )
        else:
            max_page_chunk = PrefillIndexerPlanBuild.large_page_chunk
        num_candidate_q_tiles = batch * _ceil_div(
            max_seqlen_q,
            logical_q_tile,
        )
        state = _PlanState(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table=page_table,
            task_descriptors=torch.empty(
                (
                    PrefillIndexerPlanBuild.num_buckets,
                    task_capacity,
                    PrefillIndexerPlanBuild.descriptor_words,
                ),
                dtype=torch.int32,
                **options,
            ),
            task_counts=torch.empty(
                (PrefillIndexerPlanBuild.num_buckets,),
                dtype=torch.int32,
                **options,
            ),
            plan_error=torch.empty((1,), dtype=torch.int32, **options),
            task_capacity=task_capacity,
            num_candidate_q_tiles=num_candidate_q_tiles,
            max_page_chunk=max_page_chunk,
            total_q=total_q,
            max_cols=max_cols,
            num_index_heads=num_index_heads,
            proxy_scores=torch.empty(
                (num_index_heads, total_q, max_cols),
                dtype=torch.float32,
                **options,
            ),
            num_valid_pages=torch.empty(
                (num_index_heads, total_q),
                dtype=torch.int32,
                **options,
            ),
            topk_indices=torch.empty(
                (num_index_heads, total_q, _TOP_K),
                dtype=torch.int32,
                **options,
            ),
        )
        self._plan_state = state
        _run_device_plan(state)

    def replan(self) -> None:
        """Rebuild device tasks after in-place metadata updates."""
        _run_device_plan(self.plan_state)

    def run(
        self,
        q: torch.Tensor,
        paged_k_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return FP32 proxy scores and device-side valid-page counts."""
        state = self.plan_state
        for name, tensor in (("q", q), ("paged_k_cache", paged_k_cache)):
            _check_cuda_contiguous(tensor, name=name)
            _check_same_device(state.page_table, tensor, name=name)
            if tensor.dtype != torch.float8_e4m3fn:
                raise TypeError(f"{name} must have dtype torch.float8_e4m3fn")
            if tensor.data_ptr() % 16:
                raise ValueError(f"{name} must be 16-byte aligned")
        expected_q_shape = (
            state.total_q,
            state.num_index_heads,
            PrefillIndexerGemmSm100.head_dim,
        )
        if tuple(q.shape) != expected_q_shape:
            raise ValueError(f"q must have shape {expected_q_shape}")
        if paged_k_cache.ndim != 4 or tuple(paged_k_cache.shape[1:]) != (
            1,
            PrefillIndexerPlanBuild.page_size,
            PrefillIndexerGemmSm100.head_dim,
        ):
            raise ValueError(
                "paged_k_cache must have shape [physical_pages, 1, 128, 128]"
            )
        proxy_scores = state.proxy_scores
        num_valid_pages = state.num_valid_pages
        q_flat = q.view(state.total_q * state.num_index_heads, -1)
        compiled = _compile_kernel(q_flat, paged_k_cache, state, proxy_scores)
        with torch.cuda.nvtx.range("PrefillIndexerQ8KV8ProxyScore"):
            compiled(
                q_flat,
                paged_k_cache,
                state.page_table,
                proxy_scores,
                state.task_descriptors,
                state.task_counts,
            )
        return proxy_scores, num_valid_pages


class BatchPrefillIndexerWithPagedKVCacheWrapper:
    """Compose paged proxy scores with forced-tail TopK selection."""

    def __init__(self) -> None:
        self._proxy_score = _BatchPrefillProxyScoreWrapper()

    def plan(
        self,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        page_table: torch.Tensor,
        *,
        total_q: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        num_index_heads: int = 1,
    ) -> None:
        """Prepare reusable varlen metadata and internal output buffers."""

        self._proxy_score.plan(
            cu_seqlens_q,
            cu_seqlens_k,
            page_table,
            total_q=total_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_index_heads=num_index_heads,
        )

    def run(
        self,
        q: torch.Tensor,
        paged_k_cache: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return forced-tail TopK logical page indices for one layer."""

        state = self._proxy_score.plan_state
        out = state.topk_indices if out is None else out
        _check_cuda_contiguous(out, name="out")
        _check_same_device(q, out, name="out")
        if out.dtype != torch.int32 or out.shape != state.topk_indices.shape:
            raise ValueError(
                f"out must be torch.int32 with shape {tuple(state.topk_indices.shape)}"
            )
        proxy_scores, num_valid_pages = self._proxy_score.run(
            q,
            paged_k_cache,
        )
        with torch.cuda.nvtx.range("PrefillIndexerQ8KV8TopKSelect"):
            _topk_select(
                proxy_scores.view(-1, state.max_cols),
                num_valid_pages.view(-1),
                out=out.view(-1, _TOP_K),
            )
        return out

    def replan(self) -> None:
        """Rebuild device tasks after in-place metadata updates."""
        self._proxy_score.replan()


__all__ = ["BatchPrefillIndexerWithPagedKVCacheWrapper"]
