# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""FlashInfer-style wrapper for SM100 Q8KV4 paged sparse decode."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

from ._build_utils import cuda_home
from .jit import MAX_BLOCK_SCALE_SHIFT as _MAX_BLOCK_SCALE_SHIFT
from .jit import MAX_TOPK as _MAX_TOPK

__all__ = [
    "BatchDecodeWithPagedKVCacheWrapper",
    "DecodePlan",
    "interleave_v_scales",
    "plan_decode",
    "run_decode",
]


_HEAD_DIM = 128
_DEFAULT_NUM_Q_HEADS = 64
_DEFAULT_NUM_KV_HEADS = 4
_PAGE_SIZE = 128
_DATA_ALIGNMENT = 16
_KV_DATA_ROW_BYTES = _HEAD_DIM // 2
_KV_SCALE_ROW_BYTES = _HEAD_DIM // 16
_SCALE_GROUPS = _HEAD_DIM // 16


def interleave_v_scales(v_scale: torch.Tensor) -> torch.Tensor:
    """Return linear ``[pages, Hkv, page_size, 8]`` V scales in the kernel's token-quad order.

    Same shape and dtype. Within each (page, head) block the linear layout stores byte
    ``token * 8 + group``; the kernel consumes byte ``(token // 4) * 32 + group * 4 +
    token % 4``, so every aligned 4-byte word holds one group's scales for four consecutive
    tokens (trtllm-gen's ``interleaveSfV``, the vLLM NVFP4 cache order). K scales stay linear.
    """
    if v_scale.ndim != 4 or v_scale.shape[-1] != _SCALE_GROUPS:
        raise ValueError(
            f"v_scale must have shape [pages, Hkv, page_size, {_SCALE_GROUPS}]"
        )
    pages, heads, page_size, groups = v_scale.shape
    if page_size % 4:
        raise ValueError("page_size must be a multiple of 4")
    quads = v_scale.reshape(pages, heads, page_size // 4, 4, groups)
    return quads.transpose(-1, -2).contiguous().view(v_scale.shape)


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


def _check_paged_cache(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    shape_tail: tuple[int, int, int],
    row_bytes: int,
) -> None:
    """Accept a ``[pages, Hkv, page_size, row]`` view with contiguous token rows.

    Page and head strides are free (the TMA descriptors take them in bytes) so the kernel reads
    packed pages, e.g. all heads' data blocks followed by their scale blocks, in place.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}")
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != shape_tail:
        raise ValueError(f"{name} must have shape [physical_pages, {list(shape_tail)}]")
    if tensor.stride(3) != 1 or tensor.stride(2) != row_bytes:
        raise ValueError(f"{name} must store each token's {row_bytes} bytes contiguously")
    if tensor.stride(1) % _DATA_ALIGNMENT or tensor.stride(0) % _DATA_ALIGNMENT:
        raise ValueError(
            f"{name} page and head strides must be multiples of {_DATA_ALIGNMENT} bytes"
        )
    if tensor.data_ptr() % _DATA_ALIGNMENT:
        raise ValueError(f"{name} must have a {_DATA_ALIGNMENT}-byte aligned address")


def _select_num_kv_splits(
    batch_size: int,
    q_len_per_req: int,
    sm_count: int,
    num_kv_heads: int,
    topk: int,
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
        pages_per_split = (topk + num_splits - 1) // num_splits
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
        name="_fmha_sm100_decode_q8kv4_cpp",
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
        "Compiled fmha_sm100.decode_q8kv4 host API in %.1fs",
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
    topk: int,
    split_mode: str = "streamk",
):
    """Create the opaque C++ plan for the only supported decode domain."""
    batch_size, q_len_per_req = _normalize_decode_shape(batch_size, q_len_per_req)
    qo_segment_lens = torch.full(
        (batch_size,), q_len_per_req, dtype=torch.int32, device="cpu"
    )
    kv_segment_lens = torch.full_like(qo_segment_lens, topk * _PAGE_SIZE)
    return _get_cpp().plan_decode(
        qo_segment_lens,
        kv_segment_lens,
        num_q_heads,
        num_kv_heads,
        num_kv_splits,
        _PAGE_SIZE,
        topk,
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
    topk: int,
    num_kv_splits: int | None = None,
    usable_sm_count: int | None = None,
):
    """Prepare the opaque reusable schedule used by the public wrapper."""
    from . import jit

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
        _select_num_kv_splits(batch_size, q_len_per_req, sm_count, num_kv_heads, topk)
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
        topk=topk,
        split_mode=(
            _split_mode(
                num_q_heads // num_kv_heads,
                batch_size * q_len_per_req * num_kv_heads,
                sm_count,
                jit._target_arch(device_idx),
            )
            if num_kv_splits is None
            else "legacy"
        ),
    )


def _split_mode(gqa_ratio: int, logical_ctas: int, sm_count: int, arch: str) -> str:
    """Choose the direct grid or the persistent stream-K schedule."""
    # On SM100 and SM103 the N8 kernel's short items do not amortize persistent scheduling once
    # the grid fills the GPU, so full grids take the direct grid there. SM107 keeps stream-K for
    # every shape: with its SM count the served batch sizes leave a fractional last wave whose
    # split outweighs the persistent schedule's cost. Underfilled grids keep stream-K everywhere
    # to spread their KV work over otherwise idle SMs.
    full_grid_direct = gqa_ratio == 8 and logical_ctas >= sm_count and arch != "107a"
    default_mode = "legacy" if full_grid_direct else "streamk"
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
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    topk_indices: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
    k_global_scale: torch.Tensor,
    v_global_scale: torch.Tensor,
    block_scale_shift: int,
):
    return _get_cpp().run_decode(
        q,
        k,
        v,
        plan_info,
        seq_lens,
        kv_indices,
        kv_indptr,
        topk_indices,
        k_scale.view(torch.uint8),
        v_scale.view(torch.uint8),
        out,
        sm_scale,
        k_global_scale,
        v_global_scale,
        block_scale_shift,
    )


@dataclass(frozen=True)
class DecodePlan:
    """Reusable schedule for one decode batch shape; the tensors arrive with every run."""

    backend_plan: object
    batch_size: int
    q_len_per_req: int
    num_q_heads: int
    num_kv_heads: int
    topk: int
    block_scale_shift: int
    device: torch.device
    unit_scale: torch.Tensor  # fp32 1.0 used when a run gets no global scales


def plan_decode(
    *,
    batch_size: int,
    q_len_per_req: int,
    topk: int,
    device,
    num_q_heads: int = _DEFAULT_NUM_Q_HEADS,
    num_kv_heads: int = _DEFAULT_NUM_KV_HEADS,
    num_kv_splits: int | None = None,
    usable_sm_count: int | None = None,
    block_scale_shift: int = 0,
) -> DecodePlan:
    """Build the reusable schedule outside CUDA Graph capture.

    The schedule depends only on the batch shape: ``batch_size`` requests of ``q_len_per_req``
    query tokens each, the head layout, and the TopK width (1..64). Lengths, page tables and
    TopK lists are run-time inputs of :func:`run_decode`, so one plan serves every layer of a
    step.

    ``block_scale_shift`` names the cache's block-scale convention: the kernel divides every
    E4M3 block scale by ``2 ** block_scale_shift`` before forming ``code * scale`` and folds the
    factor back into the scores and the output. Use 0 when ``code * scale`` already fits E4M3
    (products up to 448) and 3 when block scales use the full E4M3 range with a global scale
    (products up to 6 * 448, the vLLM / TransformerEngine convention). Each value is a separate
    compiled kernel.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("plan_decode() must be called outside CUDA Graph capture")
    batch_size, q_len_per_req = _normalize_decode_shape(batch_size, q_len_per_req)
    num_q_heads = int(num_q_heads)
    num_kv_heads = int(num_kv_heads)
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if num_q_heads % num_kv_heads != 0 or num_q_heads // num_kv_heads not in (8, 16):
        raise ValueError("Q8KV4 sparse decode requires 8 or 16 Q heads per KV head")
    topk = int(topk)
    if not 1 <= topk <= _MAX_TOPK:
        raise ValueError(f"topk must be in [1, {_MAX_TOPK}]")
    block_scale_shift = int(block_scale_shift)
    if not 0 <= block_scale_shift <= _MAX_BLOCK_SCALE_SHIFT:
        raise ValueError(f"block_scale_shift must be in [0, {_MAX_BLOCK_SCALE_SHIFT}]")
    device = torch.device("cuda", _device_index(device))
    from . import jit

    jit._target_arch(device)
    backend_plan = _prepare_decode_plan(
        batch_size,
        q_len_per_req,
        device=device,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        topk=topk,
        num_kv_splits=num_kv_splits,
        usable_sm_count=usable_sm_count,
    )
    return DecodePlan(
        backend_plan=backend_plan,
        batch_size=batch_size,
        q_len_per_req=q_len_per_req,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        topk=topk,
        block_scale_shift=block_scale_shift,
        device=device,
        unit_scale=torch.ones(1, dtype=torch.float32, device=device),
    )


def _check_metadata(
    plan: DecodePlan, tensor: torch.Tensor, *, name: str, shape: tuple[int, ...]
) -> None:
    _check_cuda_contiguous(tensor, name=name)
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must be torch.int32")
    if tensor.device != plan.device:
        raise ValueError(f"{name} must be on {plan.device}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")


def run_decode(
    plan: DecodePlan,
    q: torch.Tensor,
    paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
    *,
    kv_cache_sf: tuple[torch.Tensor, torch.Tensor],
    seq_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    topk_indices: torch.Tensor,
    sm_scale: float | None = None,
    kv_global_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one layer with a plan from :func:`plan_decode`; allocation-free when ``out`` is given.

    ``seq_lens`` (``[batch]``) are the requests' KV lengths, ``kv_indices`` the flat physical
    page list with ``kv_indptr`` (``[batch + 1]``) giving each request's first entry (a request
    owns at least one page), and ``topk_indices`` (``[batch * q_len_per_req, Hkv, topk]``) the
    logical page ids relative to the request; all int32 on the plan's device.

    K/V data are ``[physical_pages, Hkv, 128, 64]`` uint8 (two E2M1 per byte) and the block
    scales ``[physical_pages, Hkv, 128, 8]`` E4M3 views whose token rows are contiguous; page
    and head strides may be padded or interleave data and scales within a page. K scales are
    linear per token; V scales are in token-quad order (:func:`interleave_v_scales`).

    ``kv_global_scale`` are the per-tensor fp32 global scales of K and V as one-element CUDA
    tensors (``value = code * block_scale * global_scale``); omitted means 1.0. The kernels
    read them on the device, so a captured graph follows later updates of the tensors.
    """
    rows = plan.batch_size * plan.q_len_per_req
    _check_metadata(plan, seq_lens, name="seq_lens", shape=(plan.batch_size,))
    _check_cuda_contiguous(kv_indices, name="kv_indices")
    if kv_indices.dtype != torch.int32 or kv_indices.ndim != 1 or kv_indices.device != plan.device:
        raise ValueError(f"kv_indices must be a flat torch.int32 tensor on {plan.device}")
    _check_metadata(plan, kv_indptr, name="kv_indptr", shape=(plan.batch_size + 1,))
    _check_metadata(
        plan, topk_indices, name="topk_indices", shape=(rows, plan.num_kv_heads, plan.topk)
    )
    if not isinstance(paged_kv_cache, tuple) or len(paged_kv_cache) != 2:
        raise ValueError("paged_kv_cache must be a (K, V) tuple")
    if not isinstance(kv_cache_sf, tuple) or len(kv_cache_sf) != 2:
        raise ValueError("kv_cache_sf must be a (K scale, V scale) tuple")
    k_cache, v_cache = paged_kv_cache
    k_scale, v_scale = kv_cache_sf
    if kv_global_scale is None:
        k_global_scale = v_global_scale = plan.unit_scale
    else:
        if not isinstance(kv_global_scale, tuple) or len(kv_global_scale) != 2:
            raise ValueError("kv_global_scale must be a (K scale, V scale) tuple")
        k_global_scale, v_global_scale = kv_global_scale
        for name, tensor in (
            ("k_global_scale", k_global_scale),
            ("v_global_scale", v_global_scale),
        ):
            _check_cuda_contiguous(tensor, name=name)
            if tensor.dtype != torch.float32 or tensor.numel() != 1:
                raise ValueError(f"{name} must be a one-element torch.float32 tensor")
    _check_cuda_contiguous(q, name="q", alignment=_DATA_ALIGNMENT)
    expected_q_shape = (rows, plan.num_q_heads, _HEAD_DIM)
    if q.dtype != torch.float8_e4m3fn or tuple(q.shape) != expected_q_shape:
        raise ValueError(f"q must be torch.float8_e4m3fn with shape {expected_q_shape}")
    for name, tensor in (("k_cache", k_cache), ("v_cache", v_cache)):
        _check_paged_cache(
            tensor,
            name=name,
            dtype=torch.uint8,
            shape_tail=(plan.num_kv_heads, _PAGE_SIZE, _KV_DATA_ROW_BYTES),
            row_bytes=_KV_DATA_ROW_BYTES,
        )
    if k_cache.shape != v_cache.shape:
        raise ValueError("K and V cache shapes must match")
    for name, tensor in (("k_scale", k_scale), ("v_scale", v_scale)):
        _check_paged_cache(
            tensor,
            name=name,
            dtype=torch.float8_e4m3fn,
            shape_tail=(plan.num_kv_heads, _PAGE_SIZE, _SCALE_GROUPS),
            row_bytes=_KV_SCALE_ROW_BYTES,
        )
        if tensor.shape[0] != k_cache.shape[0]:
            raise ValueError(f"{name} must cover the same physical pages as the cache")
    for name, tensor in (
        ("q", q),
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("k_scale", k_scale),
        ("v_scale", v_scale),
        ("k_global_scale", k_global_scale),
        ("v_global_scale", v_global_scale),
    ):
        if tensor.device != plan.device:
            raise ValueError(f"{name} must be on {plan.device}")

    scale = 1.0 / math.sqrt(_HEAD_DIM) if sm_scale is None else float(sm_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("sm_scale must be finite and positive")
    if out is None:
        out = torch.empty(expected_q_shape, dtype=torch.bfloat16, device=plan.device)
    else:
        _check_cuda_contiguous(out, name="out", alignment=_DATA_ALIGNMENT)
        if (
            out.dtype != torch.bfloat16
            or tuple(out.shape) != expected_q_shape
            or out.device != plan.device
        ):
            raise ValueError(
                f"out must be torch.bfloat16 with shape {expected_q_shape} on {plan.device}"
            )
    return _run_backend(
        q,
        k_cache,
        v_cache,
        plan.backend_plan,
        seq_lens=seq_lens,
        kv_indices=kv_indices,
        kv_indptr=kv_indptr,
        topk_indices=topk_indices,
        k_scale=k_scale,
        v_scale=v_scale,
        out=out,
        sm_scale=scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        block_scale_shift=plan.block_scale_shift,
    )


@dataclass(frozen=True)
class _PlanState:
    plan: DecodePlan
    kv_indices: torch.Tensor
    kv_indptr: torch.Tensor
    seq_lens: torch.Tensor
    topk_indices: torch.Tensor
    sm_scale: float
    out: torch.Tensor


class BatchDecodeWithPagedKVCacheWrapper:
    """Hold a plan together with its request metadata for Q8KV4 sparse decode."""

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
        kv_indptr: torch.Tensor | None = None,
        block_scale_shift: int = 0,
    ) -> None:
        """Prepare a reusable request plan outside CUDA Graph capture.

        ``page_table`` maps each request's logical pages to physical pages, either as a
        ``[batch, max_pages]`` table or as a flat ``[total_pages]`` list with ``kv_indptr``
        (``[batch + 1]`` int32) giving each request's first entry; a request must own at least
        one page. ``topk_indices`` hold logical page ids relative to the request. See
        :func:`plan_decode` for ``block_scale_shift``.
        """
        for name, tensor in (
            ("topk_indices", topk_indices),
            ("page_table", page_table),
            ("seq_lens", seq_lens),
        ):
            _check_cuda_contiguous(tensor, name=name)
            _check_same_device(topk_indices, tensor, name=name)
            if tensor.dtype != torch.int32:
                raise TypeError(f"{name} must be torch.int32")
        if page_table.ndim == 2:
            if kv_indptr is not None:
                raise ValueError("kv_indptr applies only to a flat [total_pages] page_table")
            if page_table.shape[0] <= 0 or page_table.shape[1] <= 0:
                raise ValueError("page_table must have shape [batch, max_pages]")
            batch_size = int(page_table.shape[0])
            kv_indices = page_table.reshape(-1)
            kv_indptr = torch.arange(
                batch_size + 1, dtype=torch.int32, device=page_table.device
            ) * int(page_table.shape[1])
        elif page_table.ndim == 1:
            if kv_indptr is None:
                raise ValueError("a flat [total_pages] page_table requires kv_indptr")
            _check_cuda_contiguous(kv_indptr, name="kv_indptr")
            _check_same_device(topk_indices, kv_indptr, name="kv_indptr")
            if kv_indptr.dtype != torch.int32:
                raise TypeError("kv_indptr must be torch.int32")
            if kv_indptr.ndim != 1 or kv_indptr.shape[0] < 2:
                raise ValueError("kv_indptr must have shape [batch + 1]")
            batch_size = int(kv_indptr.shape[0]) - 1
            kv_indices = page_table
        else:
            raise ValueError(
                "page_table must have shape [batch, max_pages] or be a flat [total_pages] "
                "list with kv_indptr"
            )
        if topk_indices.ndim != 3 or not 1 <= topk_indices.shape[2] <= _MAX_TOPK:
            raise ValueError(
                f"topk_indices must have shape [batch * q_len_per_req, num_kv_heads, topk] "
                f"with 1 <= topk <= {_MAX_TOPK}"
            )
        scale = 1.0 / math.sqrt(_HEAD_DIM) if sm_scale is None else float(sm_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("sm_scale must be finite and positive")
        plan = plan_decode(
            batch_size=batch_size,
            q_len_per_req=q_len_per_req,
            topk=int(topk_indices.shape[2]),
            device=page_table.device,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            num_kv_splits=num_kv_splits,
            usable_sm_count=usable_sm_count,
            block_scale_shift=block_scale_shift,
        )
        rows = plan.batch_size * plan.q_len_per_req
        _check_metadata(plan, seq_lens, name="seq_lens", shape=(plan.batch_size,))
        _check_metadata(
            plan, topk_indices, name="topk_indices", shape=(rows, plan.num_kv_heads, plan.topk)
        )
        self._plan_state = _PlanState(
            plan=plan,
            kv_indices=kv_indices,
            kv_indptr=kv_indptr,
            seq_lens=seq_lens,
            topk_indices=topk_indices,
            sm_scale=scale,
            out=torch.empty(
                (rows, plan.num_q_heads, _HEAD_DIM), dtype=torch.bfloat16, device=plan.device
            ),
        )

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        *,
        kv_cache_sf: tuple[torch.Tensor, torch.Tensor],
        kv_global_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
        out: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one layer with the metadata from :meth:`plan`; see :func:`run_decode`.

        ``topk_indices`` replaces the planned list for this call (same shape), for callers whose
        selection changes per layer while the plan and page table are per step.
        """
        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        return run_decode(
            state.plan,
            q,
            paged_kv_cache,
            kv_cache_sf=kv_cache_sf,
            seq_lens=state.seq_lens,
            kv_indices=state.kv_indices,
            kv_indptr=state.kv_indptr,
            topk_indices=state.topk_indices if topk_indices is None else topk_indices,
            sm_scale=state.sm_scale,
            kv_global_scale=kv_global_scale,
            out=state.out if out is None else out,
        )
