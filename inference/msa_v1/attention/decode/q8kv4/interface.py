"""FlashInfer-style wrapper for SM100 Q8KV4 paged sparse decode."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

from inference.msa_v1._build_utils import cuda_home

__all__ = ["BatchDecodeWithPagedKVCacheWrapper"]


_HEAD_DIM = 128
_DEFAULT_NUM_Q_HEADS = 64
_DEFAULT_NUM_KV_HEADS = 4
_PAGE_SIZE = 128
_TOPK = 16
_DATA_ALIGNMENT = 16


def _device_index(device) -> int:
    if isinstance(device, int):
        return device
    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        raise ValueError(f"device must be CUDA, got {device_obj}")
    return torch.cuda.current_device() if device_obj.index is None else device_obj.index


def _normalize_decode_shape(batch_size: int, q_len_per_req: int) -> tuple[int, int]:
    batch_size = int(batch_size)
    q_len_per_req = int(q_len_per_req)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if q_len_per_req <= 0:
        raise ValueError("q_len_per_req must be positive")
    return batch_size, q_len_per_req


def _check_cuda_contiguous(
    tensor: torch.Tensor, *, name: str, alignment: int = 1
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if alignment > 1 and tensor.data_ptr() % alignment:
        raise ValueError(f"{name} must have a {alignment}-byte aligned address")


def _check_same_device(
    reference: torch.Tensor,
    tensor: torch.Tensor,
    *,
    name: str,
) -> None:
    if tensor.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}")


def _select_num_kv_splits(
    batch_size: int,
    q_len_per_req: int,
    sm_count: int,
    num_kv_heads: int,
) -> int:
    """Choose split-KV from host-known CTA parallelism and fixed TopK work."""
    logical_ctas = batch_size * q_len_per_req * num_kv_heads
    # Split only while the result still fits in one wave of CTAs (one CTA per SM), as
    # FlashInfer's decode scheduler does. A TopK-16 item is short (16 pages), so a split's
    # fixed cost (separate reduction launch, per-split Q load, pipeline fill/drain) is a large
    # fraction of an item and only pays off when it removes idle SMs without adding a wave.
    if logical_ctas >= sm_count:
        return 1
    selected = 1
    for num_splits in (2, 4, 8):
        pages_per_split = (_TOPK + num_splits - 1) // num_splits
        if pages_per_split < 2 or logical_ctas * num_splits > sm_count:
            break
        selected = num_splits
    return selected


def _jit_compile_cpp_backend():
    """JIT-compile the C++ backend using torch.utils.cpp_extension.load()."""
    import tvm_ffi
    from torch.utils.cpp_extension import load

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    csrc_dir = os.path.join(pkg_dir, "csrc")
    tvm_ffi_dir = os.path.dirname(tvm_ffi.__file__)

    extra_include = [
        os.path.join(csrc_dir, "api"),
        os.path.join(tvm_ffi_dir, "include"),
    ]
    extra_include.append(str(cuda_home() / "include"))

    started_at = time.time()
    cpp = load(
        name="_minimax_msa_v1_attention_decode_q8kv4_cpp",
        sources=[
            os.path.join(csrc_dir, "api", "decode_attention_api.cpp"),
            os.path.join(csrc_dir, "api", "decode_attention_binding.cpp"),
        ],
        extra_include_paths=extra_include,
        extra_ldflags=[
            f"-L{os.path.join(tvm_ffi_dir, 'lib')}",
            "-ltvm_ffi",
            f"-Wl,-rpath,{os.path.join(tvm_ffi_dir, 'lib')}",
        ],
        extra_cflags=["-std=c++20", "-O2"],
        verbose=True,
    )
    logger.info(
        "Compiled inference.msa_v1.attention.decode.q8kv4 host API in %.1fs",
        time.time() - started_at,
    )
    return cpp


_cpp = None


def _get_cpp():
    global _cpp
    if _cpp is None:
        _cpp = _jit_compile_cpp_backend()
    return _cpp


def _make_backend_plan(
    batch_size: int,
    q_len_per_req: int,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    num_kv_splits: int,
    usable_sm_count: int,
    device: int,
    split_mode: str = "streamk",
):
    """Create the opaque C++ plan for the only supported decode domain."""
    batch_size, q_len_per_req = _normalize_decode_shape(batch_size, q_len_per_req)
    qo_segment_lens = torch.full(
        (batch_size,), q_len_per_req, dtype=torch.int32, device="cpu"
    )
    kv_segment_lens = torch.full_like(qo_segment_lens, _TOPK * _PAGE_SIZE)
    return _get_cpp().plan_decode(
        qo_segment_lens,
        kv_segment_lens,
        num_q_heads,
        num_kv_heads,
        num_kv_splits,
        _PAGE_SIZE,
        _TOPK,
        usable_sm_count,
        device,
        split_mode,
    )


def _prepare_decode_plan(
    batch_size: int,
    q_len_per_req: int,
    *,
    device,
    num_q_heads: int,
    num_kv_heads: int,
    num_kv_splits: int | None = None,
    usable_sm_count: int | None = None,
):
    """Prepare the opaque reusable schedule used by the public wrapper."""
    batch_size, q_len_per_req = _normalize_decode_shape(batch_size, q_len_per_req)

    device_idx = _device_index(device)
    physical_sm_count = torch.cuda.get_device_properties(
        device_idx
    ).multi_processor_count
    sm_count = (
        physical_sm_count
        if usable_sm_count is None or usable_sm_count <= 0
        else min(int(usable_sm_count), physical_sm_count)
    )
    selected_splits = (
        _select_num_kv_splits(batch_size, q_len_per_req, sm_count, num_kv_heads)
        if num_kv_splits is None
        else int(num_kv_splits)
    )
    if selected_splits not in (1, 2, 4, 8):
        raise ValueError("num_kv_splits must be one of 1, 2, 4, or 8")

    return _make_backend_plan(
        batch_size,
        q_len_per_req,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_kv_splits=selected_splits,
        usable_sm_count=sm_count,
        device=device_idx,
        split_mode=(
            _split_mode(
                num_q_heads // num_kv_heads,
                batch_size * q_len_per_req * num_kv_heads,
                sm_count,
            )
            if num_kv_splits is None
            else "legacy"
        ),
    )


def _split_mode(gqa_ratio: int, logical_ctas: int, sm_count: int) -> str:
    """Choose the existing direct/fixed-split or persistent stream-K schedule."""
    # N8's shorter items do not amortize persistent scheduling once the grid fills the GPU.
    # Underfilled grids retain stream-K to distribute their KV work over otherwise idle SMs.
    default_mode = (
        "legacy" if gqa_ratio == 8 and logical_ctas >= sm_count else "streamk"
    )
    mode = os.environ.get("MSA_Q8KV4_SPLIT_MODE", default_mode)
    if mode not in ("legacy", "streamk"):
        raise ValueError("MSA_Q8KV4_SPLIT_MODE must be 'legacy' or 'streamk'")
    return mode


def _run_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan_info,
    *,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
):
    return _get_cpp().run_decode(
        q,
        k,
        v,
        plan_info,
        seq_lens,
        page_table,
        topk_indices,
        k_scale.view(torch.uint8),
        v_scale.view(torch.uint8),
        out,
        sm_scale,
    )


@dataclass(frozen=True)
class _PlanState:
    backend_plan: object
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    topk_indices: torch.Tensor
    batch_size: int
    q_len_per_req: int
    num_q_heads: int
    num_kv_heads: int
    sm_scale: float
    out: torch.Tensor


class BatchDecodeWithPagedKVCacheWrapper:
    """Manage reusable metadata and workspace for Q8KV4 sparse decode."""

    def __init__(self) -> None:
        self._plan_state: _PlanState | None = None

    def plan(
        self,
        topk_indices: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        q_len_per_req: int,
        num_q_heads: int = _DEFAULT_NUM_Q_HEADS,
        num_kv_heads: int = _DEFAULT_NUM_KV_HEADS,
        num_kv_splits: int | None = None,
        usable_sm_count: int | None = None,
        sm_scale: float | None = None,
    ) -> None:
        """Prepare a reusable request plan outside CUDA Graph capture."""
        for name, tensor in (
            ("topk_indices", topk_indices),
            ("page_table", page_table),
            ("seq_lens", seq_lens),
        ):
            _check_cuda_contiguous(tensor, name=name)
            _check_same_device(topk_indices, tensor, name=name)
            if tensor.dtype != torch.int32:
                raise TypeError(f"{name} must be torch.int32")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("plan() must be called outside CUDA Graph capture")
        if page_table.ndim != 2 or page_table.shape[0] <= 0:
            raise ValueError("page_table must have shape [batch, max_pages]")
        batch_size, q_len_per_req = _normalize_decode_shape(
            page_table.shape[0], q_len_per_req
        )
        num_q_heads = int(num_q_heads)
        num_kv_heads = int(num_kv_heads)
        if num_q_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("num_q_heads and num_kv_heads must be positive")
        if num_q_heads % num_kv_heads != 0 or num_q_heads // num_kv_heads not in (
            8,
            16,
        ):
            raise ValueError("Q8KV4 sparse decode requires 8 or 16 Q heads per KV head")
        if page_table.shape[1] <= 0:
            raise ValueError("page_table must contain at least one page slot")
        if seq_lens.shape != (batch_size,):
            raise ValueError("seq_lens must have shape [batch]")
        expected_topk_shape = (
            batch_size * q_len_per_req,
            num_kv_heads,
            _TOPK,
        )
        if tuple(topk_indices.shape) != expected_topk_shape:
            raise ValueError(f"topk_indices must have shape {expected_topk_shape}")

        scale = 1.0 / math.sqrt(_HEAD_DIM) if sm_scale is None else float(sm_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("sm_scale must be finite and positive")
        from . import jit

        jit._validate_gqa_arch(num_q_heads // num_kv_heads, page_table.device)
        backend_plan = _prepare_decode_plan(
            batch_size,
            q_len_per_req,
            device=page_table.device,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            num_kv_splits=num_kv_splits,
            usable_sm_count=usable_sm_count,
        )
        self._plan_state = _PlanState(
            backend_plan=backend_plan,
            page_table=page_table,
            seq_lens=seq_lens,
            topk_indices=topk_indices,
            batch_size=batch_size,
            q_len_per_req=q_len_per_req,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            sm_scale=scale,
            out=torch.empty(
                (batch_size * q_len_per_req, num_q_heads, _HEAD_DIM),
                dtype=torch.bfloat16,
                device=page_table.device,
            ),
        )

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        *,
        kv_cache_sf: tuple[torch.Tensor, torch.Tensor],
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one layer using metadata and workspace prepared by :meth:`plan`."""
        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        if not isinstance(paged_kv_cache, tuple) or len(paged_kv_cache) != 2:
            raise ValueError("paged_kv_cache must be a (K, V) tuple")
        if not isinstance(kv_cache_sf, tuple) or len(kv_cache_sf) != 2:
            raise ValueError("kv_cache_sf must be a (K scale, V scale) tuple")
        k_cache, v_cache = paged_kv_cache
        k_scale, v_scale = kv_cache_sf
        for name, tensor in (
            ("q", q),
            ("k_cache", k_cache),
            ("v_cache", v_cache),
            ("k_scale", k_scale),
            ("v_scale", v_scale),
        ):
            _check_cuda_contiguous(tensor, name=name, alignment=_DATA_ALIGNMENT)
            _check_same_device(state.page_table, tensor, name=name)

        expected_q_shape = (
            state.batch_size * state.q_len_per_req,
            state.num_q_heads,
            _HEAD_DIM,
        )
        if q.dtype != torch.float8_e4m3fn or tuple(q.shape) != expected_q_shape:
            raise ValueError(
                f"q must be torch.float8_e4m3fn with shape {expected_q_shape}"
            )
        expected_kv_tail = (state.num_kv_heads, _PAGE_SIZE, _HEAD_DIM // 2)
        for name, tensor in (("k_cache", k_cache), ("v_cache", v_cache)):
            if tensor.dtype != torch.uint8:
                raise TypeError(f"{name} must be torch.uint8")
            if tensor.ndim != 4 or tuple(tensor.shape[1:]) != expected_kv_tail:
                raise ValueError(
                    f"{name} must have shape [physical_pages, "
                    f"{state.num_kv_heads}, {_PAGE_SIZE}, {_HEAD_DIM // 2}]"
                )
        if k_cache.shape != v_cache.shape:
            raise ValueError("K and V cache shapes must match")
        expected_scale_shape = (
            k_cache.shape[0],
            state.num_kv_heads,
            _PAGE_SIZE,
            _HEAD_DIM // 16,
        )
        for name, tensor in (("k_scale", k_scale), ("v_scale", v_scale)):
            if tensor.dtype != torch.float8_e4m3fn:
                raise TypeError(f"{name} must be torch.float8_e4m3fn")
            if tuple(tensor.shape) != expected_scale_shape:
                raise ValueError(f"{name} must have shape {expected_scale_shape}")

        out_tensor = state.out if out is None else out
        _check_cuda_contiguous(out_tensor, name="out", alignment=_DATA_ALIGNMENT)
        _check_same_device(q, out_tensor, name="out")
        if out_tensor.dtype != torch.bfloat16 or out_tensor.shape != state.out.shape:
            raise ValueError(
                f"out must be torch.bfloat16 with shape {tuple(state.out.shape)}"
            )
        return _run_backend(
            q,
            k_cache,
            v_cache,
            state.backend_plan,
            seq_lens=state.seq_lens,
            page_table=state.page_table,
            topk_indices=state.topk_indices,
            k_scale=k_scale,
            v_scale=v_scale,
            out=out_tensor,
            sm_scale=state.sm_scale,
        )
