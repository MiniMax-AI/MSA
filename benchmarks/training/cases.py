"""Shared official and supplemental training benchmark manifests."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

from datas.training.cases import (
    DEFAULT_SCENARIO,
    OFFICIAL_SCENARIOS,
    SYNTHETIC_SCENARIOS,
    CpRankCase,
    SparseSpec,
    iter_rank_cases,
    selected_benchmark_rank_cases,
)

BENCHMARK_CASE_COUNT = 32


@dataclass(frozen=True)
class MsaBenchmarkCase:
    benchmark_case_id: int
    rank_case: CpRankCase
    representative_weight: int
    stratum: str


def _synthetic_manifest(
    name: str,
    sparse_spec: SparseSpec,
) -> tuple[MsaBenchmarkCase, ...]:
    scenario = SYNTHETIC_SCENARIOS[name]
    by_rank = {rank: [] for rank in range(scenario.cp_size)}
    for case in iter_rank_cases(
        name, sparse_spec=sparse_spec, allow_synthetic=True
    ):
        by_rank[case.rank].append(case)
    result = []
    for benchmark_case_id in range(BENCHMARK_CASE_COUNT):
        rank = benchmark_case_id * scenario.cp_size // BENCHMARK_CASE_COUNT
        ordered = sorted(
            by_rank[rank],
            key=lambda case: (
                case.causal_elements,
                case.total_kv,
                case.max_seqlen_kv,
                case.case_id,
            ),
        )
        same_rank_slot = sum(
            1 for item in result if item.rank_case.rank == rank
        )
        same_rank_count = sum(
            index * scenario.cp_size // BENCHMARK_CASE_COUNT == rank
            for index in range(BENCHMARK_CASE_COUNT)
        )
        position = round(
            (same_rank_slot + 0.5) * len(ordered) / same_rank_count - 0.5
        )
        result.append(
            MsaBenchmarkCase(
                benchmark_case_id,
                ordered[position],
                representative_weight=1,
                stratum=f"synthetic_{name}_{benchmark_case_id:02d}",
            )
        )
    return tuple(result)


def benchmark_manifest(
    name: str,
    sparse_spec: SparseSpec,
    *,
    synthetic: bool = False,
) -> tuple[MsaBenchmarkCase, ...]:
    if synthetic:
        if name not in SYNTHETIC_SCENARIOS:
            raise ValueError(
                f"unknown synthetic scenario {name!r}: "
                f"expected {tuple(SYNTHETIC_SCENARIOS)}"
            )
        manifest = _synthetic_manifest(name, sparse_spec)
    else:
        if name not in OFFICIAL_SCENARIOS:
            raise ValueError(
                f"official benchmark must use {DEFAULT_SCENARIO!r}; "
                "synthetic scenarios require explicit opt-in"
            )
        manifest = tuple(
            MsaBenchmarkCase(
                selection.benchmark_case_id,
                rank_case,
                selection.representative_weight,
                selection.stratum,
            )
            for selection, rank_case in selected_benchmark_rank_cases(
                sparse_spec=sparse_spec
            )
        )
    validate_benchmark_manifest(name, manifest, synthetic=synthetic)
    return manifest


def benchmark_manifest_digest(manifest: tuple[MsaBenchmarkCase, ...]) -> str:
    digest = hashlib.sha256()
    for item in manifest:
        case = item.rank_case
        digest.update(
            f"{case.sparse_spec.name}:{item.benchmark_case_id}:{case.scenario}:"
            f"{case.case_id}:{case.rank}:{item.representative_weight}:"
            f"{item.stratum}\n".encode()
        )
    return digest.hexdigest()


def validate_benchmark_manifest(
    name: str,
    manifest: tuple[MsaBenchmarkCase, ...],
    *,
    synthetic: bool,
) -> None:
    if len(manifest) != BENCHMARK_CASE_COUNT:
        raise AssertionError(
            f"{name} benchmark manifest must contain {BENCHMARK_CASE_COUNT} cases"
        )
    if [item.benchmark_case_id for item in manifest] != list(
        range(BENCHMARK_CASE_COUNT)
    ):
        raise AssertionError(f"{name} benchmark IDs must be contiguous")
    if any(item.rank_case.scenario != name for item in manifest):
        raise AssertionError(f"{name} benchmark contains another scenario")
    if not synthetic:
        if sum(item.representative_weight for item in manifest) != 1500:
            raise AssertionError("official benchmark weights must represent 1500 calls")
        rank_counts = Counter(item.rank_case.rank for item in manifest)
        if rank_counts != Counter({rank: 2 for rank in range(16)}):
            raise AssertionError("official benchmark must select two cases per CP rank")


__all__ = [
    "BENCHMARK_CASE_COUNT",
    "MsaBenchmarkCase",
    "benchmark_manifest",
    "benchmark_manifest_digest",
    "validate_benchmark_manifest",
]
