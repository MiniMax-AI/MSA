"""Static checks for the fixed real-inference selection."""

import hashlib
import json

import torch

from datas.inference.cases import (
    MANIFEST_FILES,
    MANIFEST_PATHS,
    load_prefill_benchmark_cases,
    load_prefill_cases,
    load_prefill_test_cases,
)
from datas.inference.tensors import make_disjoint_page_table


def test_real_prefill_manifest_and_tiers() -> None:
    all_cases = load_prefill_cases()
    smoke = load_prefill_test_cases("smoke")
    correctness = load_prefill_test_cases("full")
    msa_v1_smoke = load_prefill_test_cases("msa_v1_smoke")
    msa_v1_full = load_prefill_test_cases("msa_v1_full")
    e2e = load_prefill_test_cases("gemm_topk_e2e")
    graph = load_prefill_test_cases("cuda_graph")
    assert len(all_cases) == 10_562
    assert sum(case.count for case in all_cases) == 14_805
    assert len(correctness) == 1_024
    assert len(smoke) == 96
    assert len(msa_v1_full) == 512
    assert len(msa_v1_smoke) == 32
    assert {case.case_id for case in msa_v1_smoke} <= {
        case.case_id for case in msa_v1_full
    }
    assert {case.batch_size for case in msa_v1_smoke} == set(range(1, 19))
    assert {case.case_id for case in smoke} <= {case.case_id for case in correctness}
    assert {case.batch_size for case in smoke} == set(range(1, 19))
    assert e2e == correctness
    assert len(graph) == 16
    assert {case.batch_size for case in correctness} == set(range(1, 19))
    assert {case.case_id for case in graph} <= {case.case_id for case in correctness}


def test_real_prefill_benchmark_selection() -> None:
    cases = load_prefill_benchmark_cases()
    representative = [case for case in cases if case.is_representative]
    assert len(cases) == 128
    assert len(representative) == 66
    assert sum(case.representative_weight or 0 for case in representative) == 14_805
    assert {case.case.batch_size for case in cases} == set(range(1, 19))


def test_manifest_has_no_source_provenance_fields() -> None:
    expected_files = (
        "prefill_cases_v1/part-00000.jsonl",
        "prefill_cases_v1/part-00001.jsonl",
        "prefill_cases_v1/part-00002.jsonl",
    )
    assert MANIFEST_FILES == expected_files
    assert (
        tuple(path.relative_to(path.parents[1]).as_posix() for path in MANIFEST_PATHS)
        == expected_files
    )
    assert [len(path.read_text().splitlines()) for path in MANIFEST_PATHS] == [
        4_096,
        4_096,
        2_370,
    ]
    assert all(path.stat().st_size < 5 * 1024 * 1024 for path in MANIFEST_PATHS)

    metadata = json.loads(
        MANIFEST_PATHS[0].parents[1].joinpath("prefill_cases_v1.meta.json").read_text()
    )
    assert metadata["manifest_files"] == list(expected_files)
    assert [shard["case_count"] for shard in metadata["manifest_shards"]] == [
        4_096,
        4_096,
        2_370,
    ]
    assert [shard["sha256"] for shard in metadata["manifest_shards"]] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in MANIFEST_PATHS
    ]

    first = json.loads(MANIFEST_PATHS[0].read_text().splitlines()[0])
    assert "dp_counts" not in first
    assert set(first) == {
        "schema_version",
        "case_id",
        "batch_size",
        "query_lens",
        "prefix_lens",
        "final_kv_lens",
        "count",
        "metrics",
    }


def test_active_physical_pages_are_disjoint() -> None:
    generator = torch.Generator().manual_seed(20260831)
    page_table, physical_pages = make_disjoint_page_table(
        (1, 129, 256),
        max_cols=2,
        generator=generator,
        device=torch.device("cpu"),
    )
    active = (
        page_table[0, :1].tolist()
        + page_table[1, :2].tolist()
        + page_table[2, :2].tolist()
    )
    assert len(active) == len(set(active)) == 5
    assert physical_pages == 6
