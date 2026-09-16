"""Independent FP32 reference for paged Q8KV4 sparse prefill."""

from __future__ import annotations

from bisect import bisect_right

import torch

from datas.inference.cases import InferencePrefillCase
from tests.inference.msa_v1.attention.prefill.q8kv4.real_cases import (
    RealPrefillAttentionInputs,
    cumulative_lengths,
)


E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
HEAD_DIM = 128
Q_HEADS = 64
KV_HEADS = 4
Q_HEADS_PER_KV = Q_HEADS // KV_HEADS
PAGE_SIZE = 128
TOP_K = 16
FP8_PROBABILITY_SCALE = 448.0


def sampled_rows(
    case: InferencePrefillCase,
    *,
    limit: int = 6,
) -> tuple[int, ...]:
    """Select deterministic request endpoints and interior query rows."""

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
    return tuple(ordered)


def dequantize_cache(
    packed: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Match QMUL4 E2M1-times-E4M3 with E4M3 output rounding."""

    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1)
    codes = codes.reshape(*packed.shape[:-1], HEAD_DIM).to(torch.int64)
    expanded_scale = scale.float().repeat_interleave(16, dim=-1)
    return (lut[codes] * expanded_scale).to(torch.float8_e4m3fn).float()


def paged_sparse_attention_reference(
    q: torch.Tensor,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
    *,
    softmax_scale: float = HEAD_DIM**-0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute every output element with a bounded-memory independent oracle."""

    output = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.full(
        q.shape[:-1],
        -torch.inf,
        dtype=torch.float32,
        device=q.device,
    )
    token_in_page = torch.arange(PAGE_SIZE, device=q.device)
    query_chunk = 8

    q_batch_offset = 0
    for batch, (q_len, k_len) in enumerate(zip(q_lens, k_lens)):
        for q_start in range(0, q_len, query_chunk):
            q_end = min(q_start + query_chunk, q_len)
            q_slice = slice(q_batch_offset + q_start, q_batch_offset + q_end)
            query_positions = torch.arange(
                k_len - q_len + q_start,
                k_len - q_len + q_end,
                device=q.device,
            )
            for kv_head in range(packed_k.shape[1]):
                head_begin = kv_head * Q_HEADS_PER_KV
                head_end = head_begin + Q_HEADS_PER_KV
                logical_pages = topk_indices[kv_head, q_slice].to(torch.int64)
                valid_pages = logical_pages >= 0
                logical_pages_safe = logical_pages.clamp_min(0)
                physical_pages = page_table[batch].to(torch.int64)[logical_pages_safe]
                mK = dequantize_cache(
                    packed_k[physical_pages, kv_head],
                    k_scale[physical_pages, kv_head],
                )
                mV = dequantize_cache(
                    packed_v[physical_pages, kv_head],
                    v_scale[physical_pages, kv_head],
                )
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
                probability_fp8 = (
                    (probability * FP8_PROBABILITY_SCALE).to(torch.float8_e4m3fn).float()
                    / FP8_PROBABILITY_SCALE
                )
                partial = torch.einsum("qhpt,qptd->qphd", probability_fp8, mV)
                partial.div_(row_sum.permute(0, 2, 1).clamp_min(1.0)[..., None])
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
        q_batch_offset += q_len
    return output, lse


def sampled_real_case_reference(
    case: InferencePrefillCase,
    inputs: RealPrefillAttentionInputs,
    *,
    rows: tuple[int, ...] | None = None,
    softmax_scale: float = HEAD_DIM**-0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute an FP32-world reference for sampled rows and all KV heads."""

    selected_rows = sampled_rows(case) if rows is None else rows
    offsets = cumulative_lengths(case.query_lens)
    expected_out = []
    expected_lse = []
    selected_q_heads = []
    kv_heads = torch.arange(
        inputs.packed_k.shape[1], dtype=torch.int64, device=inputs.q.device
    )
    token = torch.arange(PAGE_SIZE, dtype=torch.int64, device=inputs.q.device)

    for row in selected_rows:
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
        packed_k = inputs.packed_k[physical_pages, kv_heads[:, None]]
        packed_v = inputs.packed_v[physical_pages, kv_heads[:, None]]
        k_scale = inputs.k_scale[physical_pages, kv_heads[:, None]]
        v_scale = inputs.v_scale[physical_pages, kv_heads[:, None]]
        mK = dequantize_cache(packed_k, k_scale)
        mV = dequantize_cache(packed_v, v_scale)

        mQ = inputs.q[row, q_heads].float()
        scores = torch.einsum("hd,hptd->hpt", mQ, mK) * softmax_scale
        token_positions = logical_pages[:, :, None] * PAGE_SIZE + token
        visible = token_positions <= causal_position
        scores = scores.masked_fill(~visible, -torch.inf)
        row_max = scores.amax(dim=-1)
        probability = torch.exp(scores - row_max[:, :, None])
        row_sum = probability.sum(dim=-1)
        probability_fp8 = (
            (probability * FP8_PROBABILITY_SCALE).to(torch.float8_e4m3fn).float()
            / FP8_PROBABILITY_SCALE
        )
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

    row_tensor = torch.tensor(
        selected_rows,
        dtype=torch.int64,
        device=inputs.q.device,
    )
    return (
        row_tensor,
        torch.stack(selected_q_heads),
        torch.stack(expected_out),
        torch.stack(expected_lse),
    )


def assert_attention_topk_contract(
    case: InferencePrefillCase,
    topk_indices: torch.Tensor,
) -> None:
    """Check forced local tails, valid prefixes, bounds, and uniqueness."""

    query_positions = []
    for query_len, prefix_len in zip(
        case.query_lens,
        case.prefix_lens,
        strict=True,
    ):
        query_positions.extend(prefix_len + index for index in range(query_len))
    positions = torch.tensor(
        query_positions,
        dtype=torch.int64,
        device=topk_indices.device,
    )
    local_page = torch.div(positions, PAGE_SIZE, rounding_mode="floor")
    valid_count = (local_page + 1).clamp(max=TOP_K)
    slots = torch.arange(TOP_K, device=topk_indices.device).reshape(1, 1, -1)
    expected_valid = slots < valid_count.reshape(1, -1, 1)
    kv_heads = topk_indices.shape[0]
    assert torch.equal(topk_indices >= 0, expected_valid.expand(kv_heads, -1, -1))
    tail = topk_indices.gather(
        2,
        (valid_count - 1).reshape(1, -1, 1).expand(kv_heads, -1, -1),
    ).squeeze(-1)
    assert torch.equal(tail, local_page.to(torch.int32).expand(kv_heads, -1))

    history_valid = slots < (valid_count - 1).reshape(1, -1, 1)
    history = topk_indices[:, :, : TOP_K - 1]
    history_mask = history_valid[:, :, : TOP_K - 1].expand(kv_heads, -1, -1)
    assert bool(torch.all(history[history_mask] >= 0))
    history_limit = local_page.reshape(1, -1, 1).expand(kv_heads, -1, TOP_K - 1)
    assert bool(torch.all(history[history_mask] < history_limit[history_mask]))
    sentinel = torch.iinfo(torch.int32).max
    sorted_history = torch.where(history_mask, history, sentinel).sort(dim=-1).values
    adjacent_valid = sorted_history[:, :, 1:] != sentinel
    assert bool(
        torch.all(
            sorted_history[:, :, 1:][adjacent_valid]
            > sorted_history[:, :, :-1][adjacent_valid]
        )
    )


def qmul4_reference(
    packed_fp4: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Return raw E4M3 bytes for the standalone QMUL4 probe."""

    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed_fp4.device)
    nibbles = torch.stack(
        (
            packed_fp4[:, 0] & 0x0F,
            packed_fp4[:, 0] >> 4,
            packed_fp4[:, 1] & 0x0F,
            packed_fp4[:, 1] >> 4,
        ),
        dim=-1,
    ).to(torch.int64)
    result = (lut[nibbles] * scale.float()[:, None]).to(torch.float8_e4m3fn)
    return result.view(torch.uint8)


__all__ = [
    "E2M1_VALUES",
    "assert_attention_topk_contract",
    "dequantize_cache",
    "paged_sparse_attention_reference",
    "qmul4_reference",
    "sampled_real_case_reference",
    "sampled_rows",
]
