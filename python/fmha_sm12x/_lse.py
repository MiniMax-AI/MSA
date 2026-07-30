# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import torch

from ._reference import Sm12xPlan, _gather_kv, _page_offsets, _selected_positions, _validate_runtime_inputs


def _logsumexp_one(
    query: torch.Tensor,
    keys: torch.Tensor,
    positions: torch.Tensor,
    *,
    visible_limit: int,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    if positions.numel() == 0:
        return torch.tensor(float("-inf"), dtype=torch.float32, device=query.device)
    visible = positions <= int(visible_limit) if causal else torch.ones_like(positions, dtype=torch.bool)
    if not bool(visible.any().item()):
        return torch.tensor(float("-inf"), dtype=torch.float32, device=query.device)
    logits = torch.matmul(keys.index_select(0, positions).float(), query.float()) * float(sm_scale)
    return torch.logsumexp(logits.masked_fill(~visible, float("-inf")), dim=0)


def run_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: Sm12xPlan,
    *,
    kv_indices: torch.Tensor | None,
    kv_block_indexes: torch.Tensor | None,
    sm_scale: float | None,
) -> torch.Tensor:
    _validate_runtime_inputs(q, k, v, plan)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(int(q.shape[-1]))
    page_size = plan.page_size if plan.page_size > 0 else 128
    h_ratio = plan.num_qo_heads // plan.num_kv_heads
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    q_start = 0
    kv_start = 0
    page_offsets = _page_offsets(plan.kv_segment_lens, page_size)
    for batch, (qo_len, kv_len, offset) in enumerate(
        zip(plan.qo_segment_lens.tolist(), plan.kv_segment_lens.tolist(), plan.qo_offset.tolist(), strict=True)
    ):
        keys_b, _ = _gather_kv(
            k, v, kv_start=kv_start, kv_len=int(kv_len), page_begin=page_offsets[batch],
            page_size=page_size, kv_indices=kv_indices,
        )
        full_positions = torch.arange(int(kv_len), device=q.device, dtype=torch.long)
        for local_q in range(int(qo_len)):
            q_index = q_start + local_q
            visible_limit = int(offset) + local_q
            for head in range(plan.num_qo_heads):
                kv_head = head // h_ratio
                positions = full_positions
                if kv_block_indexes is not None:
                    block_head = kv_head if kv_block_indexes.shape[1] == plan.num_kv_heads else head
                    positions = _selected_positions(kv_block_indexes[q_index, block_head], int(kv_len), page_size, q.device)
                lse[q_index, head] = _logsumexp_one(
                    q[q_index, head], keys_b[:, kv_head], positions,
                    visible_limit=visible_limit, causal=plan.causal, sm_scale=float(sm_scale),
                )
        q_start += int(qo_len)
        if k.ndim == 3:
            kv_start += int(kv_len)
    return lse
