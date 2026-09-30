"""Independent FP32 references and whole-output structural checks."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import torch

from datas.inference.cases import InferencePrefillCase
from tests.inference.msa_v1.indexer._common.topk_select.reference import (
    assert_quantized_topk_contract,
)
from tests.inference.msa_v1.indexer.prefill.q8kv8.cases import (
    RealPrefillInputs,
    cumulative_lengths,
)

TOP_K = 16
Q_TILE = 256


def _expected_cost_bucket(q_rows: int, page_count: int) -> int:
    work = q_rows * page_count
    bucket = 0
    for threshold in (1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16):
        if work > threshold:
            bucket += 1
    return bucket


def assert_device_plan(case: InferencePrefillCase, state: Any) -> None:
    """Check every device-planned Q-tile/page rectangle exactly once."""

    num_heads = state.num_index_heads
    q_tile = Q_TILE // num_heads
    assert state.task_descriptors.data_ptr() % 16 == 0
    assert state.task_descriptors.shape == (8, state.task_capacity, 4)
    assert state.task_descriptors.stride() == (state.task_capacity * 4, 4, 1)
    assert int(state.plan_error.item()) == 0

    counts = state.task_counts.detach().cpu().tolist()
    assert len(counts) == 8
    assert all(0 <= count <= state.task_capacity for count in counts)
    descriptor_parts = [
        state.task_descriptors[bucket, :count]
        for bucket, count in enumerate(counts)
        if count
    ]
    if descriptor_parts:
        descriptors = torch.cat(descriptor_parts, dim=0).detach().cpu().tolist()
    else:
        descriptors = []

    grouped: dict[tuple[int, int], list[tuple[int, int]]] = {}
    descriptor_idx = 0
    q_offsets = cumulative_lengths(case.query_lens)
    for bucket, count in enumerate(counts):
        for _ in range(count):
            q_global, q_position, page_begin, packed = descriptors[descriptor_idx]
            descriptor_idx += 1
            batch_idx = packed & 0x1F
            q_local = q_global - q_offsets[batch_idx]
            q_rows = ((packed >> 5) & 0xFF) + 1
            page_count = ((packed >> 13) & 0xFFF) + 1
            packed_bucket = (packed >> 25) & 0x7
            assert 0 <= batch_idx < case.batch_size
            assert q_global == q_offsets[batch_idx] + q_local
            assert q_position == case.prefix_lens[batch_idx] + q_local
            assert 0 <= q_local < case.query_lens[batch_idx]
            assert q_rows == min(q_tile, case.query_lens[batch_idx] - q_local)
            assert page_begin >= 0 and page_count > 0
            assert packed_bucket == bucket
            assert bucket == _expected_cost_bucket(q_rows * num_heads, page_count)
            grouped.setdefault((batch_idx, q_local), []).append(
                (page_begin, page_begin + page_count)
            )

    expected_task_tiles = set()
    for batch_idx, (query_len, prefix_len) in enumerate(
        zip(case.query_lens, case.prefix_lens, strict=True)
    ):
        for q_local in range(0, query_len, q_tile):
            q_rows = min(q_tile, query_len - q_local)
            max_history_pages = (prefix_len + q_local + q_rows - 1) // 128
            key = (batch_idx, q_local)
            if max_history_pages == 0:
                assert key not in grouped
                continue
            expected_task_tiles.add(key)
            ranges = sorted(grouped.get(key, ()))
            assert ranges
            cursor = 0
            for page_begin, page_end in ranges:
                assert page_begin == cursor
                assert page_end > page_begin
                cursor = page_end
            assert cursor == max_history_pages
    assert set(grouped) == expected_task_tiles
    torch.testing.assert_close(
        state.num_valid_pages,
        expected_lengths(case)
        .to(device=state.num_valid_pages.device)
        .expand(num_heads, -1),
        atol=0,
        rtol=0,
    )


def expected_lengths(case: InferencePrefillCase) -> torch.Tensor:
    """Return the forced-tail-inclusive candidate count for every query row."""

    values = []
    for query_len, prefix_len in zip(
        case.query_lens,
        case.prefix_lens,
        strict=True,
    ):
        values.extend(
            (prefix_len + query_idx) // 128 + 1 for query_idx in range(query_len)
        )
    return torch.tensor(values, dtype=torch.int32)


def sampled_rows(case: InferencePrefillCase, *, limit: int = 36) -> tuple[int, ...]:
    """Select sequence endpoints plus stable interior rows."""

    offsets = cumulative_lengths(case.query_lens)
    rows = set()
    for begin, end in pairwise(offsets):
        rows.update((begin, (begin + end - 1) // 2, end - 1))
    state = case.seed
    while len(rows) < min(limit, case.total_q):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        rows.add(state % case.total_q)
    return tuple(sorted(rows)[:limit])


def assert_score_structure(
    case: InferencePrefillCase,
    scores: torch.Tensor,
    lengths: torch.Tensor,
) -> None:
    """Check every valid score was written and every suffix stayed NaN."""

    expected = expected_lengths(case).to(device=scores.device).expand_as(lengths)
    torch.testing.assert_close(lengths, expected, atol=0, rtol=0)
    columns = torch.arange(case.max_cols, device=scores.device).reshape(1, -1)
    expected_finite = columns < (expected.reshape(-1, 1) - 1)
    assert torch.equal(torch.isfinite(scores.view(-1, case.max_cols)), expected_finite)


def full_score_reference(
    case: InferencePrefillCase,
    inputs: RealPrefillInputs,
    head_index: int = 0,
) -> torch.Tensor:
    """Compute the complete proxy-score tensor in bounded row/page chunks."""

    output = torch.full(
        (case.total_q, case.max_cols),
        float("nan"),
        dtype=torch.float32,
        device=inputs.q.device,
    )
    lengths = expected_lengths(case).to(device=inputs.q.device)
    q_offsets = cumulative_lengths(case.query_lens)
    row_chunk = 64
    page_chunk = 64
    for batch_idx, (row_begin, row_end) in enumerate(pairwise(q_offsets)):
        max_history = int(lengths[row_begin:row_end].max()) - 1
        for page_begin in range(0, max_history, page_chunk):
            page_end = min(page_begin + page_chunk, max_history)
            physical_pages = inputs.page_table[batch_idx, page_begin:page_end].to(
                torch.int64
            )
            selected_k = inputs.k_cache.index_select(0, physical_pages)[:, 0].float()
            page_ids = torch.arange(
                page_begin,
                page_end,
                dtype=torch.int32,
                device=inputs.q.device,
            )
            for query_begin in range(row_begin, row_end, row_chunk):
                query_end = min(query_begin + row_chunk, row_end)
                selected_q = inputs.q[query_begin:query_end, head_index].float()
                scores = torch.einsum("qd,ptd->qpt", selected_q, selected_k).amax(
                    dim=-1
                )
                history = lengths[query_begin:query_end].reshape(-1, 1) - 1
                valid = page_ids.reshape(1, -1) < history
                destination = output[query_begin:query_end, page_begin:page_end]
                destination.copy_(torch.where(valid, scores, destination))
    return output


def assert_full_scores(
    case: InferencePrefillCase,
    inputs: RealPrefillInputs,
    scores: torch.Tensor,
) -> torch.Tensor:
    """Compare every proxy-score element against the independent reference."""

    expected = torch.stack(
        [
            full_score_reference(case, inputs, head_index)
            for head_index in range(inputs.q.shape[1])
        ]
    )
    torch.testing.assert_close(
        scores,
        expected,
        atol=2.0e-4,
        rtol=2.0e-4,
        equal_nan=True,
    )
    return expected


def assert_topk_structure(
    topk_indices: torch.Tensor,
    lengths: torch.Tensor,
) -> None:
    """Check forced tails, short rows, bounds, and uniqueness for every row."""

    topk_indices = topk_indices.reshape(-1, TOP_K)
    lengths = lengths.reshape(-1)
    slots = torch.arange(TOP_K, dtype=torch.int32, device=topk_indices.device)
    short = lengths <= TOP_K
    if bool(short.any()):
        expected_short = torch.where(
            slots.reshape(1, -1) < lengths[short].reshape(-1, 1),
            slots.reshape(1, -1),
            torch.full_like(slots.reshape(1, -1), -1),
        )
        torch.testing.assert_close(topk_indices[short], expected_short, atol=0, rtol=0)
    long = ~short
    if not bool(long.any()):
        return
    selected = topk_indices[long, : TOP_K - 1]
    long_lengths = lengths[long]
    assert torch.equal(topk_indices[long, TOP_K - 1], long_lengths - 1)
    assert bool(torch.all(selected >= 0))
    assert bool(torch.all(selected < (long_lengths - 1).reshape(-1, 1)))
    sorted_selected = selected.sort(dim=-1).values
    assert bool(torch.all(sorted_selected[:, 1:] != sorted_selected[:, :-1]))


def assert_full_topk_quality(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    topk_indices: torch.Tensor,
) -> None:
    """Apply the quantized TopK contract to every output row."""

    assert_quantized_topk_contract(
        scores.reshape(-1, scores.shape[-1])
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        lengths.reshape(-1).cpu().numpy().astype(np.int32, copy=False),
        topk_indices.reshape(-1, TOP_K).cpu().numpy().astype(np.int32, copy=False),
    )


__all__ = [
    "assert_device_plan",
    "assert_full_scores",
    "assert_full_topk_quality",
    "assert_score_structure",
    "assert_topk_structure",
    "expected_lengths",
    "full_score_reference",
    "sampled_rows",
]
