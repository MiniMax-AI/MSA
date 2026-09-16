"""Pinned training benchmark manifest for MSA v1."""

from datas.training.cases import DEFAULT_SCENARIO, MSA_V1_SPEC
from benchmarks.training.cases import (
    BENCHMARK_CASE_COUNT,
    MsaBenchmarkCase,
    benchmark_manifest as _benchmark_manifest,
    benchmark_manifest_digest,
)

EXPECTED_BENCHMARK_DIGEST = (
    "efce23688a8b4d17acdfec0a0c9cf56915820d533af92f85d6d7e6851d9caed9"
)


def benchmark_manifest(
    name: str = DEFAULT_SCENARIO,
    *,
    synthetic: bool = False,
) -> tuple[MsaBenchmarkCase, ...]:
    return _benchmark_manifest(name, MSA_V1_SPEC, synthetic=synthetic)


__all__ = [
    "BENCHMARK_CASE_COUNT",
    "EXPECTED_BENCHMARK_DIGEST",
    "MsaBenchmarkCase",
    "benchmark_manifest",
    "benchmark_manifest_digest",
]
