# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Route NVFP4 sparse prefill from ``fmha_sm100`` to the Q8KV4 kernel.

``fmha_sm100_plan`` marks a sparse prefill plan (``MM-SA-Nv``) for the Q8KV4 kernel whenever the
batch fits it: page size 128, 16 Q heads per KV head, 4/8/16/32 blocks, causal, no max-score
output, an SM100/SM103 device and a CUDA 13.4+ toolkit (QMUL4). ``fmha_sm100`` then serves NVFP4
(uint8) caches with it when the call fits too (E4M3 Q); every other call keeps the CuTe-DSL
NVFP4 kernel. Both run on the same k2q CSR, schedule and split combine.
"""

from __future__ import annotations

import itertools
import math

import torch

from .q8kv4_decode_adapter import _cuda_device

PREFILL_BACKENDS = ("auto", "q8kv4", "cute_dsl")
PLAN_KEY = "q8kv4_prefill"
_PAGE_SIZE = 128
_HEAD_DIM = 128
_GQA_RATIO = 16
_SUPPORTED_ARCHES = ((10, 0), (10, 3))


def plan_options(kwargs: dict) -> str:
    """Pop the prefill routing option from ``fmha_sm100_plan``'s keyword arguments."""
    backend = kwargs.pop("prefill_backend", "auto")
    if backend not in PREFILL_BACKENDS:
        raise ValueError(f"prefill_backend must be one of {PREFILL_BACKENDS}, got {backend!r}")
    return backend


def _toolchain_blocker(device: torch.device) -> str | None:
    if torch.cuda.get_device_capability(device) not in _SUPPORTED_ARCHES:
        return f"device capability {torch.cuda.get_device_capability(device)} (kernel: SM100/SM103)"
    try:
        from .prefill_q8kv4 import jit

        jit._cuda_version()
    except RuntimeError as error:
        return str(error)
    return None


def _plan_blocker(
    *, num_qo_heads, num_kv_heads, page_size, kv_block_num, causal, output_maxscore, device,
) -> str | None:
    from .prefill_q8kv4 import SUPPORTED_TOPK

    if kv_block_num not in SUPPORTED_TOPK:
        return f"{kv_block_num} blocks (kernel: {SUPPORTED_TOPK})"
    if page_size != _PAGE_SIZE:
        return f"page size {page_size} (kernel: {_PAGE_SIZE})"
    if num_kv_heads <= 0 or num_qo_heads != num_kv_heads * _GQA_RATIO:
        return f"{num_qo_heads} Q heads over {num_kv_heads} KV heads (kernel: 16 per KV head)"
    if not causal:
        return "non-causal attention"
    if output_maxscore:
        return "max-score output"
    return _toolchain_blocker(device)


def attach_plan(
    plan,
    *,
    kv_segment_lens: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    page_size: int,
    kv_block_num: int,
    causal: bool,
    output_maxscore: bool,
    device,
    backend: str,
    kv_dtype: str | None,
    block_scale_shift: int,
) -> None:
    """Add the Q8KV4 route to a sparse prefill plan when the batch fits; raise if forced."""
    if not plan.get("MM-SA-Nv") or backend == "cute_dsl" or kv_dtype == "fp8":
        return
    device = _cuda_device(device)
    blocker = _plan_blocker(
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, page_size=page_size,
        kv_block_num=kv_block_num, causal=causal, output_maxscore=output_maxscore, device=device,
    )
    if blocker is not None:
        if backend == "q8kv4":
            raise ValueError(f"prefill_backend='q8kv4' cannot plan this batch: {blocker}")
        return
    page_counts = [(int(length) + _PAGE_SIZE - 1) // _PAGE_SIZE for length in kv_segment_lens.tolist()]
    plan[PLAN_KEY] = {
        "kv_indptr": torch.tensor(
            [0, *itertools.accumulate(page_counts)], dtype=torch.int32, device=device
        ),
        "backend": backend,
        "block_scale_shift": int(block_scale_shift),
    }


def run_blocker(
    entry: dict,
    *,
    q: torch.Tensor,
    kv_indices,
    k_global_scale,
    v_global_scale,
) -> str | None:
    """Why this call cannot run on the Q8KV4 kernel, or None when it can."""
    if kv_indices is None:
        return "no kv_indices (dense cache)"
    if q.dtype != torch.float8_e4m3fn:
        return f"q dtype {q.dtype} (kernel: float8_e4m3fn)"
    if not q.is_contiguous():
        return "non-contiguous q"
    for name, scale in (("k_scale", k_global_scale), ("v_scale", v_global_scale)):
        if scale is None:
            continue
        if not isinstance(scale, torch.Tensor) or scale.numel() != 1 or scale.dtype != torch.float32:
            return f"{name} is not a one-element float32 tensor"
    return None


def run(
    entry: dict,
    q: torch.Tensor,
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    k_sf: torch.Tensor,
    v_sf: torch.Tensor,
    *,
    plan_info: dict,
    q2k: torch.Tensor,
    kv_indices: torch.Tensor,
    seqused_k: torch.Tensor,
    k_global_scale,
    v_global_scale,
    sm_scale,
    q_scale,
    o_scale,
    out,
) -> torch.Tensor:
    """Run one layer: k2q CSR and schedule from the TopK lists, the Q8KV4 forward, the combine."""
    from .prefill_q8kv4 import run_prefill
    from .sparse import build_k2q_csr

    num_kv_heads = int(q2k.shape[0])
    topk = int(q2k.shape[2])
    common = dict(
        total_k=plan_info["total_k"],
        max_seqlen_k=plan_info["max_seqlen_k"],
        max_seqlen_q=plan_info["max_seqlen_q"],
        total_rows=plan_info["total_rows"],
        qhead_per_kv=_GQA_RATIO,
    )
    cu_seqlens_q = plan_info["cu_seqlens_q"]
    cu_seqlens_k = plan_info["cu_seqlens_k"]
    usable_sm_count = int(plan_info.get("usable_SM_count", -1))
    if usable_sm_count > 0:
        # The fused builder sizes its schedule for the whole device; build it separately.
        from src.sm100.prepare_scheduler import prepare_sparse_fwd_schedule_and_split

        k2q_row_ptr, k2q_q_indices = build_k2q_csr(
            q2k, cu_seqlens_q, cu_seqlens_k, _PAGE_SIZE, return_schedule=False, **common
        )
        schedule = prepare_sparse_fwd_schedule_and_split(
            k2q_row_ptr=k2q_row_ptr,
            k2q_q_indices=k2q_q_indices,
            k2q_qsplit_indices=torch.empty_like(k2q_q_indices),
            split_counts=torch.zeros((q.shape[0], num_kv_heads), dtype=torch.int32, device=q.device),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            total_q=int(q.shape[0]),
            max_seqlen_q=plan_info["max_seqlen_q"],
            topk=topk,
            head_kv=num_kv_heads,
            qhead_per_kv=_GQA_RATIO,
            blk_kv=_PAGE_SIZE,
            device=q.device,
            enabled=k2q_row_ptr.shape[1] > 1,
            usable_SM_count=usable_sm_count,
        )
    else:
        k2q_row_ptr, _, schedule = build_k2q_csr(
            q2k, cu_seqlens_q, cu_seqlens_k, _PAGE_SIZE, return_schedule=True, **common
        )
    scale = 1.0 / math.sqrt(_HEAD_DIM) if sm_scale is None else float(sm_scale)
    if q_scale is not None:
        scale *= float(q_scale)
    out, _ = run_prefill(
        q,
        (k_data, v_data),
        (k_sf, v_sf),
        kv_indices=kv_indices,
        kv_indptr=entry["kv_indptr"],
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        k2q_row_ptr=k2q_row_ptr,
        schedule=schedule,
        sm_scale=scale,
        topk=topk,
        seqused_k=seqused_k,
        k_global_scale=None if k_global_scale is None else k_global_scale.reshape(-1),
        v_global_scale=None if v_global_scale is None else v_global_scale.reshape(-1),
        output_scale=1.0 if o_scale is None else float(o_scale),
        block_scale_shift=entry["block_scale_shift"],
        out=out,
    )
    return out
