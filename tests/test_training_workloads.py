"""Static contracts for sanitized real and supplemental training workloads."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from benchmarks.training.msa_v1.cases import benchmark_manifest as v1_benchmark
from datas.training.cases import (
    MANIFEST_PATH,
    MSA_V1_SPEC,
    load_global_cases,
    load_test_selections,
    selected_rank_cases,
)


def test_real_manifest_is_sanitized_and_shape_complete() -> None:
    rows = [json.loads(line) for line in MANIFEST_PATH.read_text().splitlines()]
    assert len(rows) == 1253
    assert [row["case_id"] for row in rows] == list(range(1253))
    assert all(set(row) == {"case_id", "cu_seqlens", "metrics"} for row in rows)
    prohibited = {"source_dataset_id", "source_sample_id", "sample_id", "data_type"}
    assert all(not prohibited.intersection(row) for row in rows)
    assert all(row["cu_seqlens"][0] == 0 for row in rows)
    assert all(row["cu_seqlens"][-1] == 192 * 1024 for row in rows)


def test_full_selection_preserves_1500_calls_and_balances_ranks() -> None:
    selections = load_test_selections("full_gpu")
    assert len(selections) == 1500
    assert {item.case_id for item in selections} == set(range(1253))
    counts = Counter(item.rank for item in selections)
    assert set(counts) == set(range(16))
    assert max(counts.values()) - min(counts.values()) == 1


@pytest.mark.parametrize("sparse_spec", [MSA_V1_SPEC])
def test_real_rank_cases_use_twelve_chunks(sparse_spec) -> None:
    rank_cases = selected_rank_cases("full_gpu", sparse_spec=sparse_spec)
    assert len(rank_cases) == 1500
    assert all(len(case.chunk_ids) == 12 for _, case in rank_cases)
    assert all(case.total_q == 12 * 1024 for _, case in rank_cases)


def test_synthetic_workloads_require_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="synthetic"):
        load_global_cases("128k_cp16")
    cases = load_global_cases("128k_cp16", allow_synthetic=True)
    assert len(cases) == 128
    assert all(case.total_tokens == 128 * 1024 for case in cases)


def test_benchmark_preserves_real_selection_weights() -> None:
    cases = v1_benchmark()
    assert len(cases) == 32
    assert len({(item.rank_case.case_id, item.rank_case.rank) for item in cases}) == 32
    assert sum(item.representative_weight for item in cases) == 1500
