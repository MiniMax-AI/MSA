"""Necessary traffic must credit shared K only once."""

import pytest
import torch

from benchmarks.inference.msa_v1.indexer.decode.metrics import necessary_bytes


@pytest.mark.parametrize("page_bytes", (128 * 128, 128 * 72))
@pytest.mark.parametrize("query_length", (1, 4, 8, 16))
def test_shared_k_traffic(page_bytes, query_length):
    lengths = torch.tensor([query_length, 129, 2177], dtype=torch.int32)
    local = (lengths[:, None] - query_length + torch.arange(query_length)) // 128
    shared = int(local[:, -1].sum()) * (page_bytes + 4) + 3 * 4
    per_head = (
        3 * query_length * (128 + 4 + 16 * 4)
        + 4 * int(local.sum())
        + 4 * int(local[local >= 16].sum())
    )
    assert (
        necessary_bytes(lengths, 1, page_bytes, query_length=query_length)
        == shared + per_head
    )
    assert (
        necessary_bytes(lengths, 4, page_bytes, query_length=query_length)
        == shared + 4 * per_head
    )


def test_mbu_acceptance_uses_aggregate_traffic_and_h1_latency():
    from copy import deepcopy

    from benchmarks.inference.msa_v1.indexer.decode.metrics import compare_mbu

    baseline = {
        "protocol": {"formal": True},
        "device": "GB300",
        "device_uuid": "test",
        "dsl_version": "test",
        "precision": "q8kv8",
        "num_index_heads": 1,
        "results": [
            {
                "name": str(i),
                "weight": 1.0,
                "slots": 120,
                "necessary_bytes": 100,
                "latency_us": 10.0,
                "cv": 0.01,
            }
            for i in range(28)
        ],
    }
    candidate = deepcopy(baseline)
    candidate["num_index_heads"] = 4
    for row in candidate["results"]:
        row["necessary_bytes"] = 120
        row["latency_us"] = 11
    assert compare_mbu(candidate, baseline)["passed"]
    for row in candidate["results"]:
        row["latency_us"] = 13
    assert not compare_mbu(candidate, baseline)["passed"]
    candidate = deepcopy(baseline)
    for row in candidate["results"]:
        row["latency_us"] = 10 / 0.995
    assert compare_mbu(candidate, baseline)["passed"]
    for row in candidate["results"]:
        row["latency_us"] = 10 / 0.985
    assert not compare_mbu(candidate, baseline)["passed"]
    candidate = deepcopy(baseline)
    candidate["results"][0]["latency_us"] = 10.6
    candidate["results"][1]["latency_us"] = 8.0
    assert not compare_mbu(candidate, baseline)["passed"]
    candidate["results"][0]["cv"] = 0.04
    with pytest.raises(ValueError, match="invalid timing"):
        compare_mbu(candidate, baseline)
    baseline["precision"] = "q8kv4"
    candidate = deepcopy(baseline)
    for heads in (1, 2, 4):
        candidate["num_index_heads"] = heads
        minimum_ratio = 0.99 if heads == 1 else 0.90
        for ratio in (minimum_ratio + 0.005, minimum_ratio - 0.005):
            for row in candidate["results"]:
                row["latency_us"] = 10 / ratio
            result = compare_mbu(candidate, baseline)
            assert result["minimum_weighted_effective_mbu_ratio"] == minimum_ratio
            assert result["passed"] == (ratio >= minimum_ratio)


def test_low_latency_acceptance_is_per_case_and_same_head():
    from copy import deepcopy
    from dataclasses import asdict

    from benchmarks.inference.msa_v1.decode.cases import LOW_LATENCY_CASES
    from benchmarks.inference.msa_v1.indexer.decode.metrics import compare_low_latency

    baseline = {
        "protocol": {
            "suite": "low-latency",
            "formal": True,
            "warmup_replays": 5,
            "timed_replays": 20,
            "cuda_graph_calls": 120,
        },
        "device": "GB300",
        "device_uuid": "test",
        "dsl_version": "test",
        "precision": "q8kv8",
        "num_index_heads": 4,
        "results": [
            dict(
                asdict(case),
                name=case.name,
                slots=120,
                necessary_bytes=100,
                latency_us=10.0,
                cv=0.01,
                reuse_distance_over_l2=2.1,
            )
            for case in LOW_LATENCY_CASES
        ],
    }
    assert compare_low_latency(baseline, baseline)["passed"]
    candidate = deepcopy(baseline)
    candidate["results"][0]["latency_us"] = 10.5
    assert compare_low_latency(candidate, baseline)["passed"]
    candidate["results"][0]["latency_us"] = 10.6
    candidate["results"][1]["latency_us"] = 1.0
    result = compare_low_latency(candidate, baseline)
    assert not result["passed"]
    assert result["regressed_cases"] == [LOW_LATENCY_CASES[0].name]
    for field, value in (
        ("cv", 0.04),
        ("latency_us", float("nan")),
        ("latency_us", float("inf")),
        ("reuse_distance_over_l2", 1.9),
    ):
        candidate = deepcopy(baseline)
        candidate["results"][0][field] = value
        with pytest.raises(ValueError, match="invalid timing"):
            compare_low_latency(candidate, baseline)
    candidate = deepcopy(baseline)
    candidate["results"].pop()
    with pytest.raises(ValueError, match="all 12"):
        compare_low_latency(candidate, baseline)
    candidate = deepcopy(baseline)
    candidate["num_index_heads"] = 1
    with pytest.raises(ValueError, match="num_index_heads mismatch"):
        compare_low_latency(candidate, baseline)

    candidate = deepcopy(baseline)
    candidate["protocol"]["timed_replays"] = 2
    with pytest.raises(ValueError, match="standard timing protocol"):
        compare_low_latency(candidate, candidate)
