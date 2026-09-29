# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Shared kernel invocations for the Q8KV4 tests."""

from __future__ import annotations

import torch

from fmha_sm100.decode_q8kv4 import BatchDecodeWithPagedKVCacheWrapper

from .cases import KV_HEADS, SM_SCALE, DecodeInputs
from .conftest import run_timed


def run_wrapper(
    inputs: DecodeInputs,
    *,
    block_scale_shift: int = 0,
    kv_global_scale=None,
    num_kv_splits: int | None = None,
    sm_scale: float = SM_SCALE,
    k_scale: torch.Tensor | None = None,
    v_scale_kernel: torch.Tensor | None = None,
    label: str = "wrapper",
) -> torch.Tensor:
    """Plan and run the wrapper on the contiguous tensors of ``inputs``."""
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    wrapper.plan(
        inputs.topk_indices,
        inputs.page_table,
        inputs.seq_lens,
        q_len_per_req=inputs.case.q_len,
        num_q_heads=inputs.num_q_heads,
        num_kv_heads=KV_HEADS,
        num_kv_splits=num_kv_splits,
        sm_scale=sm_scale,
        block_scale_shift=block_scale_shift,
    )
    scales = (
        inputs.k_scale if k_scale is None else k_scale,
        inputs.v_scale_kernel if v_scale_kernel is None else v_scale_kernel,
    )
    return run_timed(
        label,
        lambda: wrapper.run(
            inputs.q, (inputs.k_codes, inputs.v_codes), kv_cache_sf=scales,
            kv_global_scale=kv_global_scale,
        ).clone(),
    )
