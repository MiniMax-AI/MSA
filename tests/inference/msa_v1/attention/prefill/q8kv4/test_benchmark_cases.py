"""Tests for the shared real-inference prefill benchmark selection."""

from __future__ import annotations

import torch

from benchmarks.inference.msa_v1.attention.prefill.q8kv4.benchmark import (
    _make_topk,
)
from benchmarks.inference.msa_v1.attention.prefill.q8kv4.cases import (
    HEAD_DIM,
    PAGE_SIZE,
    Q_HEADS,
    real_prefill_cases,
)


def test_real_prefill_benchmark_suites() -> None:
    full = real_prefill_cases("full")
    smoke = real_prefill_cases("smoke")
    representative = [case for case in full if "representative" in case.tags]
    assert len(full) == 128
    assert len(smoke) == 18
    assert {case.batch for case in smoke} == set(range(1, 19))
    assert all("batch_anchor" in case.tags for case in smoke)
    assert len(representative) == 66
    assert sum(case.representative_weight or 0 for case in representative) == 14_805
    assert {case.name for case in smoke} <= {case.name for case in full}
    cases = full
    assert all(case.useful_flops > 0 for case in cases)


def test_useful_flops_matches_selected_causal_tokens() -> None:
    case = real_prefill_cases("smoke")[0]
    assert case.useful_flops == 4 * HEAD_DIM * Q_HEADS * case.selected_tokens


def test_topk_contract_over_more_than_1000_queries() -> None:
    case = next(case for case in real_prefill_cases("full") if case.total_q >= 1000)
    topk = _make_topk(case, torch.device("cuda"))
    query_positions = torch.cat(
        [
            torch.arange(query_len, dtype=torch.int64, device="cuda") + prefix_len
            for query_len, prefix_len in zip(
                case.query_lens, case.prefix_lens, strict=True
            )
        ]
    )
    local_page = torch.div(query_positions, PAGE_SIZE, rounding_mode="floor")
    valid_count = torch.clamp(local_page + 1, max=16)
    slot = torch.arange(16, dtype=torch.int64, device="cuda").reshape(1, 1, 16)
    expected_valid = slot < valid_count.reshape(1, -1, 1)

    assert torch.equal(topk >= 0, expected_valid.expand(topk.shape[0], -1, -1))
    last = topk.gather(
        2,
        (valid_count - 1).reshape(1, -1, 1).expand(4, -1, -1),
    ).squeeze(-1)
    assert torch.equal(
        last,
        local_page.to(torch.int32).expand(topk.shape[0], -1),
    )
    valid_pages = torch.where(topk >= 0, topk, torch.iinfo(torch.int32).max)
    sorted_pages = valid_pages.sort(dim=-1).values
    adjacent_valid = sorted_pages[:, :, 1:] != torch.iinfo(torch.int32).max
    sorted_deltas = sorted_pages[:, :, 1:] - sorted_pages[:, :, :-1]
    assert torch.all(
        sorted_pages[:, :, 1:][adjacent_valid]
        > sorted_pages[:, :, :-1][adjacent_valid]
    )
    history_valid_pairs = (topk[:, :, 1:-1] >= 0) & (topk[:, :, :-2] >= 0)
    assert torch.any(
        (topk[:, :, 1:-1] < topk[:, :, :-2]) & history_valid_pairs
    )
    assert torch.any(sorted_deltas[adjacent_valid] > 1)
    assert case.total_q >= 1000
