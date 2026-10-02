"""Vectorized FP32 reference for native-E4M3 paged sparse prefill."""

from __future__ import annotations

from bisect import bisect_right

import torch

from datas.inference.cases import InferencePrefillCase
from datas.inference.tensors import cumulative_lengths
from tests.inference.msa_v1.attention.prefill.q8kv8.real_cases import (
    RealPrefillAttentionInputs,
)

HEAD_DIM = 128
Q_HEADS = 64
KV_HEADS = 4
Q_HEADS_PER_KV = Q_HEADS // KV_HEADS
PAGE_SIZE = 128


def paged_sparse_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
    *,
    softmax_scale: float = HEAD_DIM**-0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Model paged attention, including E4M3 probability rounding when applicable."""

    output = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.full(q.shape[:-1], -torch.inf, dtype=torch.float32, device=q.device)
    token_in_page = torch.arange(PAGE_SIZE, device=q.device)
    q_offset = 0
    query_chunk = 16
    kv_heads = k_cache.shape[1]
    q_heads_per_kv = q.shape[1] // kv_heads

    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        for batch, (q_len, k_len) in enumerate(zip(q_lens, k_lens)):
            for q_start in range(0, q_len, query_chunk):
                q_end = min(q_start + query_chunk, q_len)
                q_slice = slice(q_offset + q_start, q_offset + q_end)
                query_positions = torch.arange(
                    k_len - q_len + q_start,
                    k_len - q_len + q_end,
                    device=q.device,
                )
                for kv_head in range(kv_heads):
                    head_begin = kv_head * q_heads_per_kv
                    head_end = head_begin + q_heads_per_kv
                    logical_pages = topk_indices[kv_head, q_slice].to(torch.int64)
                    valid_pages = logical_pages >= 0
                    logical_pages_safe = logical_pages.clamp_min(0)
                    physical_pages = page_table[batch].to(torch.int64)[
                        logical_pages_safe
                    ]
                    mK = k_cache[physical_pages, kv_head].float()
                    mV = v_cache[physical_pages, kv_head].float()
                    mQ = q[q_slice, head_begin:head_end].float()

                    scores = torch.einsum("qhd,qptd->qhpt", mQ, mK)
                    scores.mul_(softmax_scale)
                    key_positions = (
                        logical_pages_safe[:, :, None] * PAGE_SIZE
                        + token_in_page[None, None, :]
                    )
                    visible = (
                        valid_pages[:, :, None]
                        & (key_positions < k_len)
                        & (key_positions <= query_positions[:, None, None])
                    )
                    scores.masked_fill_(~visible[:, None], -torch.inf)
                    row_max = scores.amax(dim=-1)
                    finite_page = torch.isfinite(row_max)
                    safe_max = torch.where(finite_page, row_max, 0.0)
                    probability = torch.exp(scores - safe_max[..., None])
                    probability.masked_fill_(~visible[:, None], 0.0)
                    row_sum = probability.sum(dim=-1)
                    probability_scale = 448.0
                    probability_for_pv = (
                        (probability * probability_scale)
                        .to(torch.float8_e4m3fn)
                        .float()
                        / probability_scale
                        if q.dtype == torch.float8_e4m3fn
                        else probability
                    )
                    partial = torch.einsum("qhpt,qptd->qphd", probability_for_pv, mV)
                    partial.div_(row_sum.permute(0, 2, 1).clamp_min(1.0)[..., None])
                    if q.dtype == torch.float8_e4m3fn:
                        partial = partial.to(torch.bfloat16).float()
                    partial_lse = row_max + torch.log(row_sum.clamp_min(1.0))
                    partial_lse.masked_fill_(~finite_page, -torch.inf)

                    row_lse = torch.logsumexp(partial_lse, dim=-1)
                    weights = torch.exp(partial_lse - row_lse[..., None])
                    combined = torch.einsum("qhp,qphd->qhd", weights, partial)
                    output[q_slice, head_begin:head_end] = combined.to(
                        torch.bfloat16
                    ).float()
                    lse[q_slice, head_begin:head_end] = row_lse
            q_offset += q_len
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
    return output, lse


def sampled_real_case_reference(
    case: InferencePrefillCase,
    inputs: RealPrefillAttentionInputs,
    *,
    limit: int = 6,
    softmax_scale: float = HEAD_DIM**-0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute all KV heads for deterministic sampled query rows."""

    offsets = cumulative_lengths(case.query_lens)
    candidates = set()
    for begin, end in zip(offsets, offsets[1:]):
        candidates.update((begin, (begin + end - 1) // 2, end - 1))
    ordered = sorted(candidates)
    if len(ordered) > limit:
        indices = {
            round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)
        }
        ordered = [ordered[index] for index in sorted(indices)]

    kv_heads = torch.arange(KV_HEADS, dtype=torch.int64, device=inputs.q.device)
    tokens = torch.arange(PAGE_SIZE, dtype=torch.int64, device=inputs.q.device)
    expected_out = []
    expected_lse = []
    selected_q_heads = []
    for row in ordered:
        batch = bisect_right(offsets, row) - 1
        q_idx = row - offsets[batch]
        causal_position = case.prefix_lens[batch] + q_idx
        lane = (case.seed + row) % Q_HEADS_PER_KV
        q_heads = kv_heads * Q_HEADS_PER_KV + lane
        selected_q_heads.append(q_heads)

        logical_pages = inputs.topk_indices[kv_heads, row]
        valid_count = int(torch.count_nonzero(logical_pages[0] >= 0))
        logical_pages = logical_pages[:, :valid_count].to(torch.int64)
        physical_pages = inputs.page_table[batch, logical_pages]
        mK = inputs.k_cache[physical_pages, kv_heads[:, None]].float()
        mV = inputs.v_cache[physical_pages, kv_heads[:, None]].float()
        mQ = inputs.q[row, q_heads].float()

        scores = torch.einsum("hd,hptd->hpt", mQ, mK) * softmax_scale
        positions = logical_pages[:, :, None] * PAGE_SIZE + tokens
        visible = positions <= causal_position
        scores.masked_fill_(~visible, -torch.inf)
        row_max = scores.amax(dim=-1)
        probability = torch.exp(scores - row_max[:, :, None])
        row_sum = probability.sum(dim=-1)
        probability_scale = 448.0
        probability_fp8 = (probability * probability_scale).to(
            torch.float8_e4m3fn
        ).float() / probability_scale
        partial_out = (
            (torch.einsum("hpt,hptd->hpd", probability_fp8, mV) / row_sum[:, :, None])
            .to(torch.bfloat16)
            .float()
        )
        partial_lse = row_max + torch.log(row_sum)
        combined_lse = torch.logsumexp(partial_lse, dim=-1)
        weights = torch.exp(partial_lse - combined_lse[:, None])
        combined_out = torch.einsum("hp,hpd->hd", weights, partial_out)
        expected_out.append(combined_out.to(torch.bfloat16).float())
        expected_lse.append(combined_lse)

    return (
        torch.tensor(ordered, dtype=torch.int64, device=inputs.q.device),
        torch.stack(selected_q_heads),
        torch.stack(expected_out),
        torch.stack(expected_lse),
    )


__all__ = ["paged_sparse_attention_reference", "sampled_real_case_reference"]
