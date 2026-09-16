"""Sampled FP8-world reference and whole-output structural checks."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import torch

from datas.inference.cases import InferencePrefillCase
from tests.inference.msa_v1.indexer._common.topk_select.reference import (
    assert_quantized_topk_contract,
)
from tests.inference.msa_v1.indexer.prefill.tp4_q8kv8.cases import (
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
            assert q_rows == min(Q_TILE, case.query_lens[batch_idx] - q_local)
            assert page_begin >= 0 and page_count > 0
            assert packed_bucket == bucket
            assert bucket == _expected_cost_bucket(q_rows, page_count)
            grouped.setdefault((batch_idx, q_local), []).append(
                (page_begin, page_begin + page_count)
            )

    expected_task_tiles = set()
    for batch_idx, (query_len, prefix_len) in enumerate(
        zip(case.query_lens, case.prefix_lens, strict=True)
    ):
        for q_local in range(0, query_len, Q_TILE):
            q_rows = min(Q_TILE, query_len - q_local)
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
        expected_lengths(case).to(device=state.num_valid_pages.device),
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

    expected = expected_lengths(case).to(device=scores.device)
    torch.testing.assert_close(lengths, expected, atol=0, rtol=0)
    columns = torch.arange(case.max_cols, device=scores.device).reshape(1, -1)
    expected_finite = columns < (expected.reshape(-1, 1) - 1)
    assert torch.equal(torch.isfinite(scores), expected_finite)


def assert_sampled_scores(
    case: InferencePrefillCase,
    inputs: RealPrefillInputs,
    scores: torch.Tensor,
) -> None:
    """Compare sampled row/page pairs against FP32 CUDA-Core arithmetic."""

    lengths = expected_lengths(case)
    offsets = cumulative_lengths(case.query_lens)
    sample_row_ids = []
    sample_batch_ids = []
    sample_page_ids = []
    for row in sampled_rows(case):
        batch_idx = next(index for index, end in enumerate(offsets[1:]) if row < end)
        history = int(lengths[row]) - 1
        if history == 0:
            continue
        pages = {0, history // 2, history - 1}
        for logical_page in sorted(pages):
            sample_row_ids.append(row)
            sample_batch_ids.append(batch_idx)
            sample_page_ids.append(logical_page)
    if not sample_row_ids:
        return

    device = scores.device
    rows = torch.tensor(sample_row_ids, dtype=torch.int64, device=device)
    batches = torch.tensor(sample_batch_ids, dtype=torch.int64, device=device)
    pages = torch.tensor(sample_page_ids, dtype=torch.int64, device=device)
    physical_pages = inputs.page_table[batches, pages].to(torch.int64)
    selected_q = inputs.q[rows, 0].float()
    selected_k = inputs.k_cache[physical_pages, 0].float()
    expected = (selected_k * selected_q[:, None, :]).sum(dim=-1).amax(dim=-1)
    actual = scores[rows, pages]
    torch.testing.assert_close(actual, expected, atol=2.0e-4, rtol=2.0e-4)


def full_score_reference(
    case: InferencePrefillCase,
    inputs: RealPrefillInputs,
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
            physical_pages = inputs.page_table[
                batch_idx, page_begin:page_end
            ].to(torch.int64)
            selected_k = inputs.k_cache.index_select(0, physical_pages)[:, 0].float()
            page_ids = torch.arange(
                page_begin,
                page_end,
                dtype=torch.int32,
                device=inputs.q.device,
            )
            for query_begin in range(row_begin, row_end, row_chunk):
                query_end = min(query_begin + row_chunk, row_end)
                selected_q = inputs.q[query_begin:query_end, 0].float()
                scores = torch.einsum("qd,ptd->qpt", selected_q, selected_k).amax(
                    dim=-1
                )
                history = lengths[query_begin:query_end].reshape(-1, 1) - 1
                valid = page_ids.reshape(1, -1) < history
                destination = output[
                    query_begin:query_end, page_begin:page_end
                ]
                destination.copy_(torch.where(valid, scores, destination))
    return output


def assert_full_scores(
    case: InferencePrefillCase,
    inputs: RealPrefillInputs,
    scores: torch.Tensor,
) -> torch.Tensor:
    """Compare every proxy-score element against the independent reference."""

    expected = full_score_reference(case, inputs)
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


def assert_sampled_topk_quality(
    case: InferencePrefillCase,
    scores: torch.Tensor,
    lengths: torch.Tensor,
    topk_indices: torch.Tensor,
) -> None:
    """Apply the repository's 16-bit quantized TopK gate to sampled rows."""

    rows = torch.tensor(
        sampled_rows(case, limit=8),
        dtype=torch.int64,
        device=scores.device,
    )
    sampled_scores = scores[rows].cpu().numpy().astype(np.float32, copy=False)
    sampled_lengths = lengths[rows].cpu().numpy().astype(np.int32, copy=False)
    sampled_topk = topk_indices[rows].cpu().numpy().astype(np.int32, copy=False)
    assert_quantized_topk_contract(sampled_scores, sampled_lengths, sampled_topk)


def assert_full_topk_quality(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    topk_indices: torch.Tensor,
) -> None:
    """Apply the quantized TopK contract to every output row."""

    assert_quantized_topk_contract(
        scores.cpu().numpy().astype(np.float32, copy=False),
        lengths.cpu().numpy().astype(np.int32, copy=False),
        topk_indices.cpu().numpy().astype(np.int32, copy=False),
    )


__all__ = [
    "assert_device_plan",
    "assert_full_scores",
    "assert_full_topk_quality",
    "assert_sampled_scores",
    "assert_sampled_topk_quality",
    "assert_score_structure",
    "assert_topk_structure",
    "expected_lengths",
    "full_score_reference",
    "sampled_rows",
]
