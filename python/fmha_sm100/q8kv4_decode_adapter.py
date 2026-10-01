# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Route NVFP4 sparse decode from ``fmha_sm100`` to the Q8KV4 kernel.

``fmha_sm100_plan`` builds a :class:`~fmha_sm100.decode_q8kv4.DecodePlan` next to the ordinary
decode plan whenever the batch fits the kernel: page size 128, 8 or 16 Q heads per KV head,
uniform query lengths, 1..64 blocks, causal, no max-score output. ``fmha_sm100`` then serves
NVFP4 (uint8) caches with it when the call fits too; every other call keeps the kv_mode 3 kernel.
"""

from __future__ import annotations

import itertools
import math

import torch

from .decode_q8kv4 import DecodePlan, plan_decode, run_decode

DECODE_BACKENDS = ("auto", "q8kv4", "kv_mode3")
KV_DTYPES = (None, "fp8", "nvfp4")
PLAN_KEY = "q8kv4"
_PAGE_SIZE = 128
_HEAD_DIM = 128
_GQA_RATIOS = (8, 16)
_MAX_BLOCKS = 64
# TransformerEngine convention: block scales use the full E4M3 range next to a global scale.
_DEFAULT_BLOCK_SCALE_SHIFT = 3


def plan_options(kwargs: dict) -> tuple[str, str | None, int]:
    """Pop the routing options from ``fmha_sm100_plan``'s keyword arguments."""
    backend = kwargs.pop("decode_backend", "auto")
    kv_dtype = kwargs.pop("kv_dtype", None)
    block_scale_shift = int(kwargs.pop("block_scale_shift", _DEFAULT_BLOCK_SCALE_SHIFT))
    if backend not in DECODE_BACKENDS:
        raise ValueError(f"decode_backend must be one of {DECODE_BACKENDS}, got {backend!r}")
    if kv_dtype not in KV_DTYPES:
        raise ValueError(f"kv_dtype must be one of {KV_DTYPES}, got {kv_dtype!r}")
    return backend, kv_dtype, block_scale_shift


def _cuda_device(device) -> torch.device:
    """``fmha_sm100_plan``'s ``device``: None, a CUDA index, a string or a ``torch.device``."""
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"device must be a CUDA device, got {device}")
    return device if device.index is not None else torch.device("cuda", torch.cuda.current_device())


def _plan_blocker(
    *, qo_lens, kv_lens, num_qo_heads, num_kv_heads, page_size, kv_block_num, causal,
    output_maxscore,
) -> str | None:
    if kv_block_num < 1:
        return "not a sparse (TopK) plan"
    if kv_block_num > _MAX_BLOCKS:
        return f"{kv_block_num} blocks exceed the kernel's {_MAX_BLOCKS}"
    if page_size != _PAGE_SIZE:
        return f"page size {page_size} (kernel: {_PAGE_SIZE})"
    if num_kv_heads <= 0 or num_qo_heads % num_kv_heads or (
        num_qo_heads // num_kv_heads not in _GQA_RATIOS
    ):
        return f"{num_qo_heads} Q heads over {num_kv_heads} KV heads (kernel: 8 or 16 per KV head)"
    if not qo_lens or min(qo_lens) != max(qo_lens):
        return "ragged query lengths"
    if min(kv_lens) < 1:
        return "a request without KV"
    if not causal:
        return "non-causal attention"
    if output_maxscore:
        return "max-score output"
    return None


def attach_plan(
    plan,
    *,
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    page_size: int,
    kv_block_num: int,
    causal: bool,
    output_maxscore: bool,
    usable_sm_count: int,
    device,
    backend: str,
    kv_dtype: str | None,
    block_scale_shift: int,
) -> None:
    """Add the Q8KV4 plan to a decode plan when the batch fits; raise if the backend was forced."""
    if plan.get("MM-SA-Nv") or backend == "kv_mode3" or kv_dtype == "fp8":
        return
    qo_lens = qo_segment_lens.tolist()
    kv_lens = kv_segment_lens.tolist()
    blocker = _plan_blocker(
        qo_lens=qo_lens, kv_lens=kv_lens, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        page_size=page_size, kv_block_num=kv_block_num, causal=causal,
        output_maxscore=output_maxscore,
    )
    if blocker is not None:
        if backend == "q8kv4":
            raise ValueError(f"decode_backend='q8kv4' cannot plan this batch: {blocker}")
        return
    device = _cuda_device(device)
    page_counts = [(length + _PAGE_SIZE - 1) // _PAGE_SIZE for length in kv_lens]
    plan[PLAN_KEY] = {
        "plan": plan_decode(
            batch_size=len(qo_lens),
            q_len_per_req=qo_lens[0],
            topk=kv_block_num,
            device=device,
            num_q_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            usable_sm_count=usable_sm_count if usable_sm_count > 0 else None,
            block_scale_shift=block_scale_shift,
        ),
        "seq_lens": torch.tensor(kv_lens, dtype=torch.int32, device=device),
        "kv_indptr": torch.tensor(
            [0, *itertools.accumulate(page_counts)], dtype=torch.int32, device=device
        ),
        "backend": backend,
    }


def run_blocker(
    entry: dict,
    *,
    q: torch.Tensor,
    kv_indices,
    kv_block_indexes,
    max_score,
    output_o: bool,
    q_offset_override,
    o_scale,
    k_global_scale,
    v_global_scale,
) -> str | None:
    """Why this call cannot run on the Q8KV4 plan, or None when it can."""
    plan: DecodePlan = entry["plan"]
    if kv_block_indexes is None:
        return "no kv_block_indexes"
    if kv_indices is None:
        return "no kv_indices (dense cache)"
    if q.dtype != torch.float8_e4m3fn:
        return f"q dtype {q.dtype} (kernel: float8_e4m3fn)"
    if max_score is not None or not output_o:
        return "max-score output"
    if q_offset_override is not None:
        return "q_offset_override"
    if o_scale not in (None, 1.0):
        return "o_scale"
    for name, scale in (("k_scale", k_global_scale), ("v_scale", v_global_scale)):
        if scale is not None and not (isinstance(scale, torch.Tensor) and scale.numel() == 1):
            return f"{name} is not a one-element tensor"
    expected = (plan.batch_size * plan.q_len_per_req, plan.num_kv_heads, plan.topk)
    if tuple(kv_block_indexes.shape) != expected:
        return f"kv_block_indexes shape {tuple(kv_block_indexes.shape)} (plan: {list(expected)})"
    return None


def run(
    entry: dict,
    q: torch.Tensor,
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    k_sf: torch.Tensor,
    v_sf: torch.Tensor,
    *,
    kv_indices: torch.Tensor,
    kv_block_indexes: torch.Tensor,
    k_global_scale,
    v_global_scale,
    sm_scale,
    q_scale,
    out,
) -> torch.Tensor:
    """Run one layer on the Q8KV4 plan with ``fmha_sm100``'s NVFP4 views and scales."""
    plan: DecodePlan = entry["plan"]
    scale = 1.0 / math.sqrt(_HEAD_DIM) if sm_scale is None else float(sm_scale)
    if q_scale is not None:
        scale *= float(q_scale)
    global_scales = None
    if k_global_scale is not None or v_global_scale is not None:
        global_scales = (
            plan.unit_scale if k_global_scale is None else k_global_scale,
            plan.unit_scale if v_global_scale is None else v_global_scale,
        )
    return run_decode(
        plan,
        q,
        (k_data, v_data),
        kv_cache_sf=(k_sf.view(torch.float8_e4m3fn), v_sf.view(torch.float8_e4m3fn)),
        seq_lens=entry["seq_lens"],
        kv_indices=kv_indices,
        kv_indptr=entry["kv_indptr"],
        topk_indices=kv_block_indexes,
        sm_scale=scale,
        kv_global_scale=global_scales,
        out=out,
    )
