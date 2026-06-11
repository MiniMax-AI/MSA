# SPDX-License-Identifier: MIT

"""Reference-correct SM12x attention paths.

These are intentionally simple Torch implementations. They provide a safe
SM120/SM121 semantic target while the production kernels remain separate from
the SM100 tcgen05/TMEM implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True, slots=True)
class Sm12xPlan:
    qo_segment_lens: torch.Tensor
    kv_segment_lens: torch.Tensor
    qo_offset: torch.Tensor
    num_qo_heads: int
    num_kv_heads: int
    page_size: int
    output_maxscore: bool
    kv_block_num: int
    causal: bool


def _cpu_i64(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(device="cpu", dtype=torch.int64, non_blocking=False).contiguous()


def make_plan(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    qo_offset: int | torch.Tensor | None,
    page_size: int,
    output_maxscore: bool,
    kv_block_num: int,
    causal: bool,
) -> Sm12xPlan:
    qo_lens = _cpu_i64(qo_segment_lens)
    kv_lens = _cpu_i64(kv_segment_lens)
    if qo_lens.ndim != 1 or kv_lens.ndim != 1 or qo_lens.shape != kv_lens.shape:
        raise ValueError("qo_segment_lens and kv_segment_lens must be same-shape rank-1 tensors")
    if num_kv_heads == -1:
        num_kv_heads = int(num_qo_heads)
    if int(num_qo_heads) % int(num_kv_heads) != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    if qo_offset is None:
        offset = kv_lens - qo_lens
    elif isinstance(qo_offset, int):
        offset = torch.full_like(qo_lens, int(qo_offset))
    else:
        offset = _cpu_i64(qo_offset)
    if offset.shape != qo_lens.shape:
        raise ValueError("qo_offset must have shape [batch_size]")
    if bool((qo_lens < 0).any().item()) or bool((kv_lens < 0).any().item()):
        raise ValueError("qo_segment_lens and kv_segment_lens must be non-negative")
    return Sm12xPlan(
        qo_segment_lens=qo_lens,
        kv_segment_lens=kv_lens,
        qo_offset=offset,
        num_qo_heads=int(num_qo_heads),
        num_kv_heads=int(num_kv_heads),
        page_size=int(page_size),
        output_maxscore=bool(output_maxscore),
        kv_block_num=int(kv_block_num),
        causal=bool(causal),
    )


def _page_offsets(kv_lens: torch.Tensor, page_size: int) -> list[int]:
    offsets = [0]
    total = 0
    for kv_len in kv_lens.tolist():
        total += (int(kv_len) + page_size - 1) // page_size
        offsets.append(total)
    return offsets


def _gather_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kv_start: int,
    kv_len: int,
    page_begin: int,
    page_size: int,
    kv_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k.ndim == 3:
        return k[kv_start : kv_start + kv_len], v[kv_start : kv_start + kv_len]
    if k.ndim != 4 or page_size <= 0:
        raise ValueError("paged KV requires k/v shape [pages, heads, page_size, dim] and page_size > 0")
    page_count = (kv_len + page_size - 1) // page_size
    if kv_indices is None:
        physical = torch.arange(page_begin, page_begin + page_count, device=k.device)
    else:
        physical = kv_indices[page_begin : page_begin + page_count].to(torch.long)
    k_dense = k.index_select(0, physical).permute(0, 2, 1, 3).reshape(-1, k.shape[1], k.shape[3])
    v_dense = v.index_select(0, physical).permute(0, 2, 1, 3).reshape(-1, v.shape[1], v.shape[3])
    return k_dense[:kv_len], v_dense[:kv_len]


def _selected_positions(blocks: torch.Tensor, kv_len: int, page_size: int, device: torch.device) -> torch.Tensor:
    valid = blocks[(blocks >= 0) & (blocks * page_size < kv_len)].to(torch.long)
    if valid.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    starts = valid * page_size
    rel = torch.arange(page_size, device=device, dtype=torch.long)
    pos = (starts[:, None] + rel[None, :]).reshape(-1)
    return pos[pos < kv_len]


def _attend_one(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    positions: torch.Tensor,
    *,
    visible_limit: int,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    if positions.numel() == 0:
        return torch.zeros((values.shape[-1],), dtype=torch.float32, device=query.device)
    visible = positions <= int(visible_limit) if causal else torch.ones_like(positions, dtype=torch.bool)
    if not bool(visible.any().item()):
        return torch.zeros((values.shape[-1],), dtype=torch.float32, device=query.device)
    k_sel = keys.index_select(0, positions).float()
    v_sel = values.index_select(0, positions).float()
    logits = torch.matmul(k_sel, query.float()) * float(sm_scale)
    logits = logits.masked_fill(~visible, float("-inf"))
    return torch.matmul(torch.softmax(logits, dim=0), v_sel)


def _write_tile_scores(
    max_score: torch.Tensor,
    *,
    head: int,
    q_index: int,
    query: torch.Tensor,
    keys: torch.Tensor,
    positions: torch.Tensor,
    page_size: int,
    visible_limit: int,
    causal: bool,
    sm_scale: float,
) -> None:
    if positions.numel() == 0:
        return
    logits = torch.matmul(keys.index_select(0, positions).float(), query.float()) * float(sm_scale)
    visible = positions <= int(visible_limit) if causal else torch.ones_like(positions, dtype=torch.bool)
    tile_ids = torch.div(positions, page_size, rounding_mode="floor")
    for tile in torch.unique(tile_ids).tolist():
        mask = (tile_ids == int(tile)) & visible
        if bool(mask.any().item()):
            max_score[head, int(tile), q_index] = logits[mask].max()


def _validate_runtime_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, plan: Sm12xPlan) -> None:
    if q.ndim != 3:
        raise ValueError("q must have shape [total_q, num_qo_heads, head_dim]")
    if q.shape[1] != plan.num_qo_heads:
        raise ValueError("q head count does not match plan")
    if k.shape != v.shape:
        raise ValueError("k and v shapes must match")
    if k.ndim not in (3, 4):
        raise ValueError("k/v must be dense [total_k, heads, dim] or paged [pages, heads, page, dim]")
    total_q = int(plan.qo_segment_lens.sum().item())
    if int(q.shape[0]) != total_q:
        raise ValueError("q.shape[0] must equal sum(qo_segment_lens)")
    if k.ndim == 3 and int(k.shape[0]) < int(plan.kv_segment_lens.sum().item()):
        raise ValueError("k/v sequence length must cover sum(kv_segment_lens)")


def run_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: Sm12xPlan,
    *,
    kv_indices: torch.Tensor | None,
    kv_block_indexes: torch.Tensor | None,
    out: torch.Tensor | None,
    max_score: torch.Tensor | None,
    sm_scale: float | None,
    output_o: bool,
    output_maxscore: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    _validate_runtime_inputs(q, k, v, plan)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(int(q.shape[-1]))
    page_size = plan.page_size if plan.page_size > 0 else 128
    h_ratio = plan.num_qo_heads // plan.num_kv_heads
    result = out if out is not None else torch.empty(
        (q.shape[0], plan.num_qo_heads, v.shape[-1]), dtype=torch.bfloat16, device=q.device
    )
    want_o = bool(output_o)
    want_score = bool(output_maxscore or plan.output_maxscore or max_score is not None)
    score = max_score
    if want_score and score is None:
        max_tiles = max((int(x) + page_size - 1) // page_size for x in plan.kv_segment_lens.tolist())
        score = torch.full((plan.num_qo_heads, max_tiles, q.shape[0]), float("-inf"), dtype=torch.float32, device=q.device)
    elif score is not None:
        score.fill_(float("-inf"))
    q_start = 0
    kv_start = 0
    page_offsets = _page_offsets(plan.kv_segment_lens, page_size)
    for batch, (qo_len, kv_len, offset) in enumerate(
        zip(plan.qo_segment_lens.tolist(), plan.kv_segment_lens.tolist(), plan.qo_offset.tolist(), strict=True)
    ):
        keys_b, values_b = _gather_kv(
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
                if want_o:
                    result[q_index, head] = _attend_one(
                        q[q_index, head], keys_b[:, kv_head], values_b[:, kv_head], positions,
                        visible_limit=visible_limit, causal=plan.causal, sm_scale=float(sm_scale),
                    ).to(result.dtype)
                if score is not None:
                    _write_tile_scores(
                        score, head=head, q_index=q_index, query=q[q_index, head],
                        keys=keys_b[:, kv_head], positions=positions, page_size=page_size,
                        visible_limit=visible_limit, causal=plan.causal, sm_scale=float(sm_scale),
                    )
        q_start += int(qo_len)
        if k.ndim == 3:
            kv_start += int(kv_len)
    return (result if want_o else None, score)
