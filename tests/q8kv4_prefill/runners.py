# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Shared kernel invocations for the Q8KV4 prefill tests."""

from __future__ import annotations

import torch

from fmha_sm100.prefill_q8kv4 import BatchPrefillWithPagedKVCacheWrapper

from .cases import SM_SCALE, PrefillInputs
from .conftest import run_timed


def plan_wrapper(
    inputs: PrefillInputs,
    *,
    flat: bool = False,
    topk_indices: torch.Tensor | None = None,
    sm_scale: float = SM_SCALE,
    **plan_kwargs,
) -> BatchPrefillWithPagedKVCacheWrapper:
    case = inputs.case
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    wrapper.plan(
        inputs.topk_indices if topk_indices is None else topk_indices,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        inputs.kv_indices if flat else inputs.page_table,
        kv_indptr=plan_kwargs.pop("kv_indptr", inputs.kv_indptr) if flat else None,
        total_k=sum(case.k_lens),
        total_rows=inputs.total_rows,
        max_seqlen_q=max(case.q_lens),
        max_seqlen_k=max(case.k_lens),
        sm_scale=sm_scale,
        **plan_kwargs,
    )
    return wrapper


def run_wrapper(
    inputs: PrefillInputs,
    *,
    label: str = "wrapper",
    kv=None,
    kv_sf=None,
    plan_kwargs: dict | None = None,
    **run_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plan and run the wrapper (contiguous tensors of ``inputs`` unless ``kv``/``kv_sf`` given);
    returns ``(out, lse)`` clones."""
    wrapper = plan_wrapper(inputs, **(plan_kwargs or {}))
    kv = (inputs.k_codes, inputs.v_codes) if kv is None else kv
    kv_sf = (inputs.k_scale, inputs.v_scale_kernel) if kv_sf is None else kv_sf

    def call():
        out, lse = wrapper.run(inputs.q, kv, kv_cache_sf=kv_sf, return_lse=True, **run_kwargs)
        return out.clone(), lse.clone()

    return run_timed(label, call)
