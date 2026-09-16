"""Load normalized real-inference prefill shapes for every MSA operator."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
MANIFEST_FILES = (
    "prefill_cases_v1/part-00000.jsonl",
    "prefill_cases_v1/part-00001.jsonl",
    "prefill_cases_v1/part-00002.jsonl",
)
MANIFEST_PATHS = tuple(DATA_DIR / filename for filename in MANIFEST_FILES)
TEST_SELECTION_PATH = DATA_DIR / "prefill_test_cases_v1.json"
BENCHMARK_SELECTION_PATH = DATA_DIR / "prefill_benchmark_cases_v1.json"
CASE_FIELDS = {
    "schema_version",
    "case_id",
    "batch_size",
    "query_lens",
    "prefix_lens",
    "final_kv_lens",
    "count",
    "metrics",
}
METRIC_FIELDS = {
    "total_q",
    "max_query_len",
    "max_final_kv",
    "max_cols",
    "valid_q_tiles",
    "candidate_q_tiles",
    "active_cluster_fraction_ppm",
    "useful_flops",
    "issued_flops",
    "useful_efficiency_ppm",
    "topk_candidate_entries",
    "score_bytes",
    "q_skew_ppm",
    "current_lpt_makespan_pages",
    "ideal_lpt_makespan_pages",
    "lpt_makespan_ratio_ppm",
}


@dataclass(frozen=True)
class InferencePrefillCase:
    """One normalized true-varlen prefill shape."""

    case_id: str
    batch_size: int
    query_lens: tuple[int, ...]
    prefix_lens: tuple[int, ...]
    final_kv_lens: tuple[int, ...]
    count: int
    metrics: Mapping[str, int]

    @property
    def total_q(self) -> int:
        return int(self.metrics["total_q"])

    @property
    def max_query_len(self) -> int:
        return int(self.metrics["max_query_len"])

    @property
    def max_final_kv(self) -> int:
        return int(self.metrics["max_final_kv"])

    @property
    def max_cols(self) -> int:
        return int(self.metrics["max_cols"])

    @property
    def useful_flops(self) -> int:
        return int(self.metrics["useful_flops"])

    @property
    def seed(self) -> int:
        digest = hashlib.sha256(self.case_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


@dataclass(frozen=True)
class InferencePrefillBenchmarkCase:
    """One fixed benchmark selection entry plus its normalized shape."""

    case: InferencePrefillCase
    tags: tuple[str, ...]
    representative_weight: int | None
    stratum: str | None

    @property
    def is_representative(self) -> bool:
        return "representative" in self.tags


def _sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _parse_case(value: object) -> InferencePrefillCase:
    if not isinstance(value, dict) or set(value) != CASE_FIELDS:
        raise TypeError("each prefill manifest row must use the exact case schema")
    if value["schema_version"] != 1:
        raise ValueError("unsupported prefill case schema")
    batch_size = value["batch_size"]
    count = value["count"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    metrics = value["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != METRIC_FIELDS:
        raise TypeError("prefill metrics must use the exact metric schema")
    if any(
        isinstance(metric, bool) or not isinstance(metric, int)
        for metric in metrics.values()
    ):
        raise TypeError("prefill metrics must contain integers")
    for field in ("query_lens", "prefix_lens", "final_kv_lens"):
        items = value[field]
        if not isinstance(items, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in items
        ):
            raise TypeError(f"{field} must be an integer array")
    case = InferencePrefillCase(
        case_id=str(value["case_id"]),
        batch_size=batch_size,
        query_lens=tuple(value["query_lens"]),
        prefix_lens=tuple(value["prefix_lens"]),
        final_kv_lens=tuple(value["final_kv_lens"]),
        count=count,
        metrics=metrics,
    )
    if not (
        len(case.query_lens)
        == len(case.prefix_lens)
        == len(case.final_kv_lens)
        == case.batch_size
    ):
        raise ValueError(f"{case.case_id}: batch and length arrays disagree")
    if any(length <= 0 for length in case.query_lens):
        raise ValueError(f"{case.case_id}: query lengths must be positive")
    if any(length < 0 or length % 128 for length in case.prefix_lens):
        raise ValueError(f"{case.case_id}: prefix lengths must be 128-aligned")
    expected_final = tuple(
        prefix + query
        for prefix, query in zip(case.prefix_lens, case.query_lens, strict=True)
    )
    if case.final_kv_lens != expected_final:
        raise ValueError(f"{case.case_id}: final KV lengths are inconsistent")
    if case.total_q != sum(case.query_lens):
        raise ValueError(f"{case.case_id}: total_q is inconsistent")
    if case.max_query_len != max(case.query_lens):
        raise ValueError(f"{case.case_id}: max_query_len is inconsistent")
    if case.max_final_kv != max(case.final_kv_lens):
        raise ValueError(f"{case.case_id}: max_final_kv is inconsistent")
    if case.max_cols != (case.max_final_kv + 127) // 128:
        raise ValueError(f"{case.case_id}: max_cols is inconsistent")
    canonical = json.dumps(
        {
            "batch_size": case.batch_size,
            "prefix_lens": list(case.prefix_lens),
            "query_lens": list(case.query_lens),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    expected_case_id = f"prefill_{hashlib.sha256(canonical).hexdigest()[:20]}"
    if case.case_id != expected_case_id:
        raise ValueError(f"{case.case_id}: case ID is inconsistent")
    historical_page_rows = sum(
        query_len * (prefix_len // 128)
        + 128 * (query_len // 128) * ((query_len // 128) - 1) // 2
        + (query_len // 128) * (query_len % 128)
        for query_len, prefix_len in zip(case.query_lens, case.prefix_lens, strict=True)
    )
    if case.useful_flops != historical_page_rows * 2 * 128 * 128:
        raise ValueError(f"{case.case_id}: useful_flops is inconsistent")
    if int(case.metrics["score_bytes"]) != case.total_q * case.max_cols * 4:
        raise ValueError(f"{case.case_id}: score_bytes is inconsistent")
    return case


@cache
def load_prefill_cases() -> tuple[InferencePrefillCase, ...]:
    """Load and validate all 10,562 normalized prefill shapes."""

    cases = tuple(
        _parse_case(json.loads(line))
        for path in MANIFEST_PATHS
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )
    case_ids = {case.case_id for case in cases}
    if len(cases) != 10_562 or len(case_ids) != len(cases):
        raise ValueError("prefill manifest must contain 10,562 unique cases")
    if sum(case.count for case in cases) != 14_805:
        raise ValueError("prefill manifest count must sum to 14,805")
    return cases


@cache
def load_prefill_case_map() -> Mapping[str, InferencePrefillCase]:
    """Return the normalized cases keyed by deterministic case ID."""

    return {case.case_id: case for case in load_prefill_cases()}


def _load_selection(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported selection schema in {path}")
    if payload.get("manifest_files") != list(MANIFEST_FILES):
        raise ValueError(f"manifest shard list mismatch in {path}")
    if payload.get("manifest_sha256") != _sha256(MANIFEST_PATHS):
        raise ValueError(f"manifest SHA256 mismatch in {path}")
    return payload


@cache
def load_prefill_test_cases(
    tier: str = "full",
) -> tuple[InferencePrefillCase, ...]:
    """Resolve one fixed correctness tier to normalized cases."""

    payload = _load_selection(TEST_SELECTION_PATH)
    tiers = payload.get("tiers")
    if not isinstance(tiers, dict) or tier not in tiers:
        raise ValueError(f"unknown prefill test tier: {tier}")
    selection = tiers[tier]
    if not isinstance(selection, dict):
        raise TypeError(f"invalid prefill test tier: {tier}")
    if tier in {"static_all", "exhaustive"}:
        cases = load_prefill_cases()
    elif reference := selection.get("case_ids_ref"):
        cases = load_prefill_test_cases(str(reference))
    else:
        case_ids = selection.get("case_ids")
        if not isinstance(case_ids, list):
            raise ValueError(f"tier {tier} has no case IDs")
        case_map = load_prefill_case_map()
        try:
            cases = tuple(case_map[str(case_id)] for case_id in case_ids)
        except KeyError as exc:
            raise ValueError(f"tier {tier} references an unknown case") from exc
    if len(cases) != int(selection["case_count"]):
        raise ValueError(f"tier {tier} case count is inconsistent")
    return cases


@cache
def load_prefill_benchmark_cases(
    batch_size: int | None = None,
) -> tuple[InferencePrefillBenchmarkCase, ...]:
    """Load the fixed benchmark selection, optionally for one batch size."""

    payload = _load_selection(BENCHMARK_SELECTION_PATH)
    entries = payload.get("cases")
    if not isinstance(entries, list):
        raise TypeError("benchmark selection must contain a cases array")
    case_map = load_prefill_case_map()
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("benchmark entries must be objects")
        case_id = str(entry["case_id"])
        try:
            case = case_map[case_id]
        except KeyError as exc:
            raise ValueError(f"unknown benchmark case: {case_id}") from exc
        if int(entry["count"]) != case.count:
            raise ValueError(f"{case_id}: benchmark count is inconsistent")
        tags = tuple(str(tag) for tag in entry["tags"])
        weight = entry.get("representative_weight")
        result.append(
            InferencePrefillBenchmarkCase(
                case=case,
                tags=tags,
                representative_weight=None if weight is None else int(weight),
                stratum=None if entry.get("stratum") is None else str(entry["stratum"]),
            )
        )
    expected_count = int(payload["default_case_count"])
    if len(result) != expected_count or len(
        {item.case.case_id for item in result}
    ) != len(result):
        raise ValueError("benchmark selection must contain unique cases")
    represented = sum(
        item.representative_weight or 0 for item in result if item.is_representative
    )
    expected_weight = int(payload["representative_strata"]["represented_calls"])
    if represented != expected_weight:
        raise ValueError("representative benchmark weights are inconsistent")
    if batch_size is None:
        return tuple(result)
    filtered = tuple(item for item in result if item.case.batch_size == batch_size)
    if not filtered:
        raise ValueError(f"benchmark selection has no batch_size={batch_size} cases")
    represented = sum(
        item.representative_weight or 0 for item in filtered if item.is_representative
    )
    expected_weight = sum(
        case.count for case in load_prefill_cases() if case.batch_size == batch_size
    )
    if represented != expected_weight:
        raise ValueError(
            f"batch_size={batch_size} representative weights are inconsistent"
        )
    return filtered


__all__ = [
    "InferencePrefillBenchmarkCase",
    "InferencePrefillCase",
    "load_prefill_benchmark_cases",
    "load_prefill_case_map",
    "load_prefill_cases",
    "load_prefill_test_cases",
]
