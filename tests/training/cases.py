"""Shared real-workload pytest manifest helpers for MSA training kernels."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

from datas.training.cases import (
    EXPECTED_REAL_CALLS,
    CpRankCase,
    SparseSpec,
    selected_rank_cases,
)

FULL_CASE_COUNT = EXPECTED_REAL_CALLS
SMOKE_CASE_COUNT = 96


@dataclass(frozen=True)
class MsaTestCase:
    """One stable pytest item from the sanitized real-workload selection."""

    item_id: int
    rank_case: CpRankCase
    structured: bool

    @property
    def pytest_id(self) -> str:
        case = self.rank_case
        mode = "structured" if self.structured else "random"
        return (
            f"{self.item_id:04d}-{case.scenario}-g{case.case_id:04d}"
            f"-r{case.rank:02d}-{mode}"
        )


def build_manifest(tier: str, sparse_spec: SparseSpec) -> tuple[MsaTestCase, ...]:
    return tuple(
        MsaTestCase(
            item_id=selection.selection_id,
            rank_case=rank_case,
            structured=selection.structured,
        )
        for selection, rank_case in selected_rank_cases(
            tier, sparse_spec=sparse_spec
        )
    )


def manifest_digest(manifest: tuple[MsaTestCase, ...]) -> str:
    digest = hashlib.sha256()
    for item in manifest:
        case = item.rank_case
        digest.update(
            f"{case.sparse_spec.name}:{item.item_id}:{case.scenario}:"
            f"{case.case_id}:{case.rank}:{int(item.structured)}\n".encode()
        )
    return digest.hexdigest()


def validate_manifest(manifest: tuple[MsaTestCase, ...]) -> None:
    if len(manifest) != FULL_CASE_COUNT:
        raise AssertionError(f"full manifest must contain {FULL_CASE_COUNT} cases")
    if [item.item_id for item in manifest] != list(range(FULL_CASE_COUNT)):
        raise AssertionError("full manifest item IDs must be contiguous")
    if {item.rank_case.scenario for item in manifest} != {"192k_cp16"}:
        raise AssertionError("full manifest must only use the official 192K workload")
    if {item.rank_case.case_id for item in manifest} != set(range(1253)):
        raise AssertionError("full manifest must cover every unique real shape")
    rank_counts = Counter(item.rank_case.rank for item in manifest)
    if set(rank_counts) != set(range(16)):
        raise AssertionError("full manifest must cover every CP rank")
    if max(rank_counts.values()) - min(rank_counts.values()) > 1:
        raise AssertionError("full manifest CP rank counts must be balanced")
    if any(len(item.rank_case.chunk_ids) != 12 for item in manifest):
        raise AssertionError("every rank case must contain 12 chunks")
    if any(item.rank_case.total_q != 12 * 1024 for item in manifest):
        raise AssertionError("every rank case must contain 12288 Q tokens")
    if sum(item.structured for item in manifest) != 24:
        raise AssertionError("full manifest structured-case count changed")


__all__ = [
    "FULL_CASE_COUNT",
    "SMOKE_CASE_COUNT",
    "MsaTestCase",
    "build_manifest",
    "manifest_digest",
    "validate_manifest",
]
