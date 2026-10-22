#!/usr/bin/env python3
"""Generate deterministic inference prefill test and benchmark case manifests."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = DATA_DIR
DEFAULT_OUTPUT_DIR = DATA_DIR
SOURCE_PATTERN = "dspark_forward_dumps.dp*.jsonl"

SCHEMA_VERSION = 1
PAGE_SIZE = 128
HEAD_DIM = 128
Q_TILE = 256
PERSISTENT_CLUSTERS = 76
GPU_TEST_CASES = 1024
SMOKE_TEST_CASES = 96
MSA_V1_GPU_TEST_CASES = 512
MSA_V1_SMOKE_TEST_CASES = 32
CUDA_GRAPH_CASES = 16
BENCHMARK_CASES = 128

EXPECTED_RAW_PREFILL_ROWS = 10937
EXPECTED_UNIQUE_CASES = 10562
EXPECTED_DUPLICATE_ROWS = 375
EXPECTED_WEIGHTED_CALLS = 14805

MANIFEST_DIRNAME = "prefill_cases_v1"
MANIFEST_SHARD_ROWS = 4096
MAX_MANIFEST_SHARD_BYTES = 5 * 1024 * 1024
META_FILENAME = "prefill_cases_v1.meta.json"
TEST_FILENAME = "prefill_test_cases_v1.json"
BENCHMARK_FILENAME = "prefill_benchmark_cases_v1.json"


@dataclass
class MergedCase:
    batch_size: int
    query_lens: tuple[int, ...]
    prefix_lens: tuple[int, ...]
    count: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="directory containing dspark_forward_dumps.dp*.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for generated manifests",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that existing outputs are byte-for-byte reproducible",
    )
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_int(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_int_list(
    value: object,
    *,
    name: str,
    length: int,
    minimum: int,
) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must have length {length}")
    return tuple(
        _require_int(item, name=f"{name}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    )


def _load_and_merge(
    source_paths: Sequence[Path],
) -> tuple[list[MergedCase], int]:
    merged: dict[
        tuple[int, tuple[int, ...], tuple[int, ...]],
        MergedCase,
    ] = {}
    raw_prefill_rows = 0

    for source_path in source_paths:
        with source_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON at {source_path}:{line_number}"
                    ) from exc
                if record.get("batch_type") != "prefill":
                    continue
                if set(record) != {
                    "batch_type",
                    "batch_size",
                    "query_lens",
                    "kv_lens",
                    "count",
                }:
                    raise ValueError(
                        f"unexpected prefill fields at {source_path}:{line_number}"
                    )
                batch_size = _require_int(
                    record["batch_size"],
                    name="batch_size",
                    minimum=1,
                )
                query_lens = _require_int_list(
                    record["query_lens"],
                    name="query_lens",
                    length=batch_size,
                    minimum=1,
                )
                prefix_lens = _require_int_list(
                    record["kv_lens"],
                    name="kv_lens",
                    length=batch_size,
                    minimum=0,
                )
                count = _require_int(record["count"], name="count", minimum=1)
                if sum(query_lens) > 16384:
                    raise ValueError(
                        f"total query length exceeds 16384 at "
                        f"{source_path}:{line_number}"
                    )
                if any(prefix % PAGE_SIZE != 0 for prefix in prefix_lens):
                    raise ValueError(
                        f"prefix length is not page aligned at "
                        f"{source_path}:{line_number}"
                    )

                key = (batch_size, query_lens, prefix_lens)
                case = merged.get(key)
                if case is None:
                    case = MergedCase(
                        batch_size=batch_size,
                        query_lens=query_lens,
                        prefix_lens=prefix_lens,
                        count=0,
                    )
                    merged[key] = case
                case.count += count
                raw_prefill_rows += 1

    cases = sorted(
        merged.values(),
        key=lambda case: (case.batch_size, case.query_lens, case.prefix_lens),
    )
    return cases, raw_prefill_rows


def _canonical_shape(case: MergedCase) -> bytes:
    payload = {
        "batch_size": case.batch_size,
        "prefix_lens": list(case.prefix_lens),
        "query_lens": list(case.query_lens),
    }
    return json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _case_id(case: MergedCase) -> str:
    return f"prefill_{_sha256_bytes(_canonical_shape(case))[:20]}"


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _useful_page_rows(query_len: int, prefix_len: int) -> int:
    full_blocks, tail = divmod(query_len, PAGE_SIZE)
    return (
        query_len * (prefix_len // PAGE_SIZE)
        + PAGE_SIZE * full_blocks * (full_blocks - 1) // 2
        + full_blocks * tail
    )


def _issued_pages(query_len: int, prefix_len: int) -> int:
    q_tiles = _ceil_div(query_len, Q_TILE)
    return (
        q_tiles * (prefix_len // PAGE_SIZE)
        + (q_tiles - 1) ** 2
        + (query_len - 1) // PAGE_SIZE
    )


def _actual_task_costs(case: MergedCase) -> list[int]:
    costs = []
    for query_len, prefix_len in zip(case.query_lens, case.prefix_lens):
        for q_begin in range(0, query_len, Q_TILE):
            q_end = min(q_begin + Q_TILE, query_len)
            costs.append((prefix_len + q_end - 1) // PAGE_SIZE)
    return costs


def _current_lpt_loads(case: MergedCase) -> list[int]:
    max_query_len = max(case.query_lens)
    final_kv_lens = tuple(
        prefix + query for prefix, query in zip(case.prefix_lens, case.query_lens)
    )
    max_final_kv = max(final_kv_lens)
    tiles_per_batch = _ceil_div(max_query_len, Q_TILE)
    tasks = []
    for q_tile_desc in range(tiles_per_batch):
        q_tile_idx = tiles_per_batch - 1 - q_tile_desc
        q_end = min((q_tile_idx + 1) * Q_TILE, max_query_len)
        estimated_pages = max(
            (max_final_kv - max_query_len + q_end - 1) // PAGE_SIZE,
            0,
        )
        for batch_idx in range(case.batch_size):
            task_idx = q_tile_desc * case.batch_size + batch_idx
            tasks.append((-estimated_pages, task_idx))
    tasks.sort()

    estimated_heap = [(0, cluster_idx) for cluster_idx in range(PERSISTENT_CLUSTERS)]
    heapq.heapify(estimated_heap)
    assignments = [[] for _ in range(PERSISTENT_CLUSTERS)]
    for negative_cost, task_idx in tasks:
        load, cluster_idx = heapq.heappop(estimated_heap)
        assignments[cluster_idx].append(task_idx)
        heapq.heappush(estimated_heap, (load - negative_cost, cluster_idx))

    actual_loads = [0] * PERSISTENT_CLUSTERS
    for cluster_idx, task_indices in enumerate(assignments):
        for task_idx in task_indices:
            q_tile_desc, batch_idx = divmod(task_idx, case.batch_size)
            q_tile_idx = tiles_per_batch - 1 - q_tile_desc
            q_begin = q_tile_idx * Q_TILE
            query_len = case.query_lens[batch_idx]
            if q_begin >= query_len:
                continue
            q_end = min(q_begin + Q_TILE, query_len)
            actual_loads[cluster_idx] += (
                case.prefix_lens[batch_idx] + q_end - 1
            ) // PAGE_SIZE
    return actual_loads


def _ideal_lpt_loads(costs: Iterable[int]) -> list[int]:
    loads = [(0, cluster_idx) for cluster_idx in range(PERSISTENT_CLUSTERS)]
    heapq.heapify(loads)
    for cost in sorted(costs, reverse=True):
        load, cluster_idx = heapq.heappop(loads)
        heapq.heappush(loads, (load + cost, cluster_idx))
    return [load for load, _ in loads]


def _metrics(case: MergedCase) -> dict[str, int]:
    final_kv_lens = tuple(
        prefix + query for prefix, query in zip(case.prefix_lens, case.query_lens)
    )
    total_q = sum(case.query_lens)
    max_query_len = max(case.query_lens)
    max_final_kv = max(final_kv_lens)
    max_cols = _ceil_div(max_final_kv, PAGE_SIZE)
    page_rows = sum(
        _useful_page_rows(query_len, prefix_len)
        for query_len, prefix_len in zip(case.query_lens, case.prefix_lens)
    )
    issued_pages = sum(
        _issued_pages(query_len, prefix_len)
        for query_len, prefix_len in zip(case.query_lens, case.prefix_lens)
    )
    useful_flops = page_rows * 2 * PAGE_SIZE * HEAD_DIM
    issued_flops = issued_pages * 2 * Q_TILE * PAGE_SIZE * HEAD_DIM
    useful_efficiency_ppm = (
        useful_flops * 1_000_000 // issued_flops if issued_flops else 1_000_000
    )
    actual_loads = _current_lpt_loads(case)
    ideal_loads = _ideal_lpt_loads(_actual_task_costs(case))
    current_makespan = max(actual_loads)
    ideal_makespan = max(ideal_loads)
    lpt_ratio_ppm = (
        current_makespan * 1_000_000 // ideal_makespan if ideal_makespan else 1_000_000
    )
    valid_q_tiles = sum(_ceil_div(query_len, Q_TILE) for query_len in case.query_lens)
    candidate_q_tiles = case.batch_size * _ceil_div(max_query_len, Q_TILE)
    return {
        "total_q": total_q,
        "max_query_len": max_query_len,
        "max_final_kv": max_final_kv,
        "max_cols": max_cols,
        "valid_q_tiles": valid_q_tiles,
        "candidate_q_tiles": candidate_q_tiles,
        "active_cluster_fraction_ppm": min(valid_q_tiles, PERSISTENT_CLUSTERS)
        * 1_000_000
        // PERSISTENT_CLUSTERS,
        "useful_flops": useful_flops,
        "issued_flops": issued_flops,
        "useful_efficiency_ppm": useful_efficiency_ppm,
        "topk_candidate_entries": page_rows + total_q,
        "score_bytes": total_q * max_cols * 4,
        "q_skew_ppm": max_query_len * 1_000_000 // total_q,
        "current_lpt_makespan_pages": current_makespan,
        "ideal_lpt_makespan_pages": ideal_makespan,
        "lpt_makespan_ratio_ppm": lpt_ratio_ppm,
    }


def _make_case_record(case: MergedCase) -> dict[str, object]:
    final_kv_lens = [
        prefix + query for prefix, query in zip(case.prefix_lens, case.query_lens)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": _case_id(case),
        "batch_size": case.batch_size,
        "query_lens": list(case.query_lens),
        "prefix_lens": list(case.prefix_lens),
        "final_kv_lens": final_kv_lens,
        "count": case.count,
        "metrics": _metrics(case),
    }


def _select_extremes(records: Sequence[dict[str, object]]) -> set[str]:
    selected = set()
    metric_names = (
        "total_q",
        "max_query_len",
        "max_final_kv",
        "max_cols",
        "valid_q_tiles",
        "candidate_q_tiles",
        "useful_flops",
        "issued_flops",
        "useful_efficiency_ppm",
        "topk_candidate_entries",
        "score_bytes",
        "q_skew_ppm",
        "lpt_makespan_ratio_ppm",
    )
    for metric_name in metric_names:
        ordered = sorted(
            records,
            key=lambda record: (
                record["metrics"][metric_name],
                -record["count"],
                record["case_id"],
            ),
        )
        selected.add(ordered[0]["case_id"])
        selected.add(ordered[-1]["case_id"])
    return selected


def _select_gpu_test_cases(
    records: Sequence[dict[str, object]],
    *,
    case_count: int = GPU_TEST_CASES,
    salt: str = "prefill-gpu-test-v1",
) -> list[str]:
    by_id = {record["case_id"]: record for record in records}
    selected = _select_extremes(records)

    by_batch: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_batch[record["batch_size"]].append(record)
        if record["metrics"]["useful_flops"] == 0:
            selected.add(record["case_id"])
    for batch_records in by_batch.values():
        ordered = sorted(
            batch_records,
            key=lambda record: (
                record["metrics"]["total_q"],
                record["metrics"]["max_final_kv"],
                record["metrics"]["useful_flops"],
                record["case_id"],
            ),
        )
        for numerator in range(5):
            index = numerator * (len(ordered) - 1) // 4
            selected.add(ordered[index]["case_id"])

    boundary_values = (
        1,
        2,
        7,
        31,
        63,
        127,
        128,
        129,
        130,
        191,
        255,
        256,
        257,
        511,
        512,
        513,
        8192,
        16384,
    )
    for boundary in boundary_values:
        matches = [record for record in records if boundary in record["query_lens"]]
        if matches:
            best = min(
                matches, key=lambda record: (-record["count"], record["case_id"])
            )
            selected.add(best["case_id"])

    remaining = sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{salt}:{record['case_id']}".encode()
        ).hexdigest(),
    )
    for record in remaining:
        if len(selected) >= case_count:
            break
        selected.add(record["case_id"])
    if len(selected) != case_count:
        raise RuntimeError(f"expected {case_count} GPU test cases")
    return sorted(
        selected,
        key=lambda case_id: (
            by_id[case_id]["batch_size"],
            by_id[case_id]["metrics"]["total_q"],
            case_id,
        ),
    )


def _evenly_spaced(values: Sequence[str], count: int) -> list[str]:
    if len(values) < count:
        raise ValueError(f"cannot select {count} values from {len(values)}")
    if count == 1:
        return [values[len(values) // 2]]
    return [values[index * (len(values) - 1) // (count - 1)] for index in range(count)]


def _select_smoke_test_cases(
    records: Sequence[dict[str, object]],
    full_case_ids: Sequence[str],
    *,
    case_count: int = SMOKE_TEST_CASES,
    salt: str = "prefill-smoke-v1",
) -> list[str]:
    """Select a stable, multi-batch development subset of the full suite."""

    full_ids = set(full_case_ids)
    candidates = [record for record in records if record["case_id"] in full_ids]
    by_id = {record["case_id"]: record for record in candidates}
    selected = _select_extremes(candidates)
    by_batch: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in candidates:
        by_batch[record["batch_size"]].append(record)
    for batch_records in by_batch.values():
        ordered = sorted(
            batch_records,
            key=lambda record: (
                record["metrics"]["total_q"],
                record["metrics"]["max_final_kv"],
                record["case_id"],
            ),
        )
        for index in (0, len(ordered) // 2, len(ordered) - 1):
            selected.add(ordered[index]["case_id"])
    remaining = sorted(
        candidates,
        key=lambda record: hashlib.sha256(
            f"{salt}:{record['case_id']}".encode()
        ).hexdigest(),
    )
    for record in remaining:
        if len(selected) >= case_count:
            break
        selected.add(record["case_id"])
    if len(selected) != case_count:
        raise RuntimeError(f"expected {case_count} smoke test cases")
    return sorted(
        selected,
        key=lambda case_id: (
            by_id[case_id]["batch_size"],
            by_id[case_id]["metrics"]["total_q"],
            case_id,
        ),
    )


def _select_msa_v1_smoke_test_cases(
    records: Sequence[dict[str, object]],
    full_case_ids: Sequence[str],
) -> list[str]:
    """Select 32 stable cases while retaining every production batch size."""

    full_ids = set(full_case_ids)
    candidates = [record for record in records if record["case_id"] in full_ids]
    by_id = {record["case_id"]: record for record in candidates}
    by_batch: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in candidates:
        by_batch[record["batch_size"]].append(record)
    selected = {
        min(
            batch_records,
            key=lambda record: (-record["count"], record["case_id"]),
        )["case_id"]
        for batch_records in by_batch.values()
    }
    remaining = sorted(
        candidates,
        key=lambda record: hashlib.sha256(
            f"prefill-msa-v1-smoke-v1:{record['case_id']}".encode()
        ).hexdigest(),
    )
    for record in remaining:
        if len(selected) >= MSA_V1_SMOKE_TEST_CASES:
            break
        selected.add(record["case_id"])
    if len(selected) != MSA_V1_SMOKE_TEST_CASES:
        raise RuntimeError(
            f"expected {MSA_V1_SMOKE_TEST_CASES} MSA v1 smoke test cases"
        )
    return sorted(
        selected,
        key=lambda case_id: (
            by_id[case_id]["batch_size"],
            by_id[case_id]["metrics"]["total_q"],
            case_id,
        ),
    )


def _batch_bucket(batch_size: int) -> str:
    if batch_size == 1:
        return "b1"
    if batch_size == 2:
        return "b2"
    if batch_size <= 4:
        return "b3_4"
    if batch_size <= 8:
        return "b5_8"
    return "b9_18"


def _task_bucket(valid_q_tiles: int) -> str:
    if valid_q_tiles <= 3:
        return "t1_3"
    if valid_q_tiles <= 20:
        return "t4_20"
    if valid_q_tiles <= 63:
        return "t21_63"
    return "t64_plus"


def _kv_bucket(max_final_kv: int) -> str:
    if max_final_kv <= 32768:
        return "k_le_32k"
    if max_final_kv <= 65536:
        return "k_32_64k"
    if max_final_kv <= 131072:
        return "k_64_128k"
    return "k_gt_128k"


def _feature_vector(record: dict[str, object]) -> tuple[float, ...]:
    metrics = record["metrics"]
    return (
        math.log1p(metrics["total_q"]),
        math.log1p(metrics["max_final_kv"]),
        math.log1p(metrics["useful_flops"]),
        metrics["useful_efficiency_ppm"] / 1_000_000,
        math.log1p(metrics["topk_candidate_entries"]),
        math.log1p(metrics["lpt_makespan_ratio_ppm"] / 1_000_000),
    )


def _normalized_features(
    records: Sequence[dict[str, object]],
) -> dict[str, tuple[float, ...]]:
    raw = {record["case_id"]: _feature_vector(record) for record in records}
    dimensions = len(next(iter(raw.values())))
    minimums = [
        min(vector[index] for vector in raw.values()) for index in range(dimensions)
    ]
    maximums = [
        max(vector[index] for vector in raw.values()) for index in range(dimensions)
    ]
    normalized = {}
    for case_id, vector in raw.items():
        normalized[case_id] = tuple(
            (value - minimums[index]) / (maximums[index] - minimums[index])
            if maximums[index] > minimums[index]
            else 0.0
            for index, value in enumerate(vector)
        )
    return normalized


def _representative_cases(
    records: Sequence[dict[str, object]],
) -> list[tuple[dict[str, object], str, int]]:
    features = _normalized_features(records)
    strata: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        metrics = record["metrics"]
        key = (
            _batch_bucket(record["batch_size"]),
            _task_bucket(metrics["valid_q_tiles"]),
            _kv_bucket(metrics["max_final_kv"]),
        )
        strata[key].append(record)

    representatives = []
    for key in sorted(strata):
        members = strata[key]
        represented_count = sum(record["count"] for record in members)
        dimensions = len(features[members[0]["case_id"]])
        centroid = tuple(
            sum(
                features[record["case_id"]][dimension] * record["count"]
                for record in members
            )
            / represented_count
            for dimension in range(dimensions)
        )

        def distance(
            record: dict[str, object],
            *,
            center: tuple[float, ...] = centroid,
        ) -> tuple[float, int, str]:
            vector = features[record["case_id"]]
            squared_distance = sum(
                (value - center[index]) ** 2 for index, value in enumerate(vector)
            )
            return squared_distance, -record["count"], record["case_id"]

        representative = min(members, key=distance)
        representatives.append((representative, "/".join(key), represented_count))
    return representatives


def _stress_lists(
    records: Sequence[dict[str, object]],
) -> list[tuple[str, list[dict[str, object]]]]:
    def ordered(metric: str, *, reverse: bool) -> list[dict[str, object]]:
        return sorted(
            records,
            key=lambda record: (
                -record["metrics"][metric] if reverse else record["metrics"][metric],
                -record["count"],
                record["case_id"],
            ),
        )

    zero_useful = sorted(
        (record for record in records if record["metrics"]["useful_flops"] == 0),
        key=lambda record: (-record["count"], record["case_id"]),
    )
    return [
        ("stress_zero_useful_flops", zero_useful),
        ("stress_lpt_imbalance", ordered("lpt_makespan_ratio_ppm", reverse=True)),
        ("stress_tile_waste", ordered("useful_efficiency_ppm", reverse=False)),
        ("stress_max_final_kv", ordered("max_final_kv", reverse=True)),
        ("stress_score_workspace", ordered("score_bytes", reverse=True)),
        ("stress_topk_candidates", ordered("topk_candidate_entries", reverse=True)),
        ("stress_issued_flops", ordered("issued_flops", reverse=True)),
        ("stress_candidate_tasks", ordered("candidate_q_tiles", reverse=True)),
        ("stress_valid_tasks", ordered("valid_q_tiles", reverse=True)),
        ("stress_query_skew", ordered("q_skew_ppm", reverse=True)),
    ]


def _make_benchmark_selection(
    records: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    by_id = {record["case_id"]: record for record in records}
    selected_order = []
    tags: dict[str, set[str]] = defaultdict(set)
    representative_weight: dict[str, int] = {}
    stratum_by_id: dict[str, str] = {}

    for record, stratum, represented_count in _representative_cases(records):
        case_id = record["case_id"]
        selected_order.append(case_id)
        tags[case_id].add("representative")
        representative_weight[case_id] = represented_count
        stratum_by_id[case_id] = stratum

    hot_records = sorted(
        records, key=lambda record: (-record["count"], record["case_id"])
    )[:32]
    for record in hot_records:
        case_id = record["case_id"]
        tags[case_id].add("hot")
        if case_id not in selected_order:
            selected_order.append(case_id)

    by_batch: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_batch[record["batch_size"]].append(record)
    for batch_size in sorted(by_batch):
        anchor = min(
            by_batch[batch_size],
            key=lambda record: (-record["count"], record["case_id"]),
        )
        case_id = anchor["case_id"]
        tags[case_id].add("batch_anchor")
        if case_id not in selected_order:
            selected_order.append(case_id)

    stress_lists = _stress_lists(records)
    stress_rank = 0
    while len(selected_order) < BENCHMARK_CASES:
        made_progress = False
        for tag, candidates in stress_lists:
            if stress_rank >= len(candidates):
                continue
            case_id = candidates[stress_rank]["case_id"]
            tags[case_id].add(tag)
            if case_id not in selected_order:
                selected_order.append(case_id)
                made_progress = True
                if len(selected_order) == BENCHMARK_CASES:
                    break
        if not made_progress and all(
            stress_rank + 1 >= len(values) for _, values in stress_lists
        ):
            break
        stress_rank += 1

    if len(selected_order) < BENCHMARK_CASES:
        for record in sorted(
            records, key=lambda item: (-item["count"], item["case_id"])
        ):
            case_id = record["case_id"]
            if case_id in selected_order:
                continue
            tags[case_id].add("coverage_fill")
            selected_order.append(case_id)
            if len(selected_order) == BENCHMARK_CASES:
                break
    if len(selected_order) != BENCHMARK_CASES:
        raise RuntimeError(f"expected {BENCHMARK_CASES} benchmark cases")

    entries = []
    for case_id in selected_order:
        record = by_id[case_id]
        metrics = record["metrics"]
        entry = {
            "case_id": case_id,
            "tags": sorted(tags[case_id]),
            "count": record["count"],
            "batch_size": record["batch_size"],
            "total_q": metrics["total_q"],
            "max_final_kv": metrics["max_final_kv"],
            "valid_q_tiles": metrics["valid_q_tiles"],
            "useful_flops": metrics["useful_flops"],
            "topk_candidate_entries": metrics["topk_candidate_entries"],
        }
        if case_id in representative_weight:
            entry["representative_weight"] = representative_weight[case_id]
            entry["stratum"] = stratum_by_id[case_id]
        entries.append(entry)
    return entries


def _render_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _render_jsonl(records: Sequence[dict[str, object]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )


def _make_outputs(
    source_paths: Sequence[Path],
) -> dict[str, str]:
    merged_cases, raw_prefill_rows = _load_and_merge(source_paths)
    records = [_make_case_record(case) for case in merged_cases]
    case_ids = [record["case_id"] for record in records]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("case ID collision")

    duplicate_rows = raw_prefill_rows - len(records)
    weighted_calls = sum(record["count"] for record in records)
    if (
        raw_prefill_rows != EXPECTED_RAW_PREFILL_ROWS
        or len(records) != EXPECTED_UNIQUE_CASES
        or duplicate_rows != EXPECTED_DUPLICATE_ROWS
        or weighted_calls != EXPECTED_WEIGHTED_CALLS
    ):
        raise RuntimeError(
            "unexpected normalized dataset summary: "
            f"rows={raw_prefill_rows}, unique={len(records)}, "
            f"duplicates={duplicate_rows}, count={weighted_calls}"
        )

    manifest_outputs = {
        f"{MANIFEST_DIRNAME}/part-{shard_index:05d}.jsonl": _render_jsonl(
            records[shard_begin : shard_begin + MANIFEST_SHARD_ROWS]
        )
        for shard_index, shard_begin in enumerate(
            range(0, len(records), MANIFEST_SHARD_ROWS)
        )
    }
    manifest_files = list(manifest_outputs)
    oversized_shards = {
        filename: len(rendered.encode("utf-8"))
        for filename, rendered in manifest_outputs.items()
        if len(rendered.encode("utf-8")) > MAX_MANIFEST_SHARD_BYTES
    }
    if oversized_shards:
        raise RuntimeError(f"manifest shards exceed 5 MiB: {oversized_shards}")
    manifest = "".join(manifest_outputs.values())
    manifest_sha256 = _sha256_bytes(manifest.encode("utf-8"))
    meta = {
        "schema_version": SCHEMA_VERSION,
        "case_kind": "inference_prefill",
        "manifest_files": manifest_files,
        "manifest_sha256": manifest_sha256,
        "manifest_shards": [
            {
                "file": filename,
                "sha256": _sha256_bytes(rendered.encode("utf-8")),
                "case_count": rendered.count("\n"),
            }
            for filename, rendered in manifest_outputs.items()
        ],
        "normalization": {
            "batch_type": "prefill",
            "dedup_key": ["batch_size", "query_lens", "prefix_lens"],
            "prefix_semantics": "cached prefix before this prefill chunk",
            "final_kv_formula": "final_kv_lens[i] = prefix_lens[i] + query_lens[i]",
            "count_formula": "sum counts from normalized duplicate records",
        },
        "summary": {
            "raw_prefill_rows": raw_prefill_rows,
            "unique_cases": len(records),
            "duplicate_rows_merged": duplicate_rows,
            "weighted_calls": weighted_calls,
        },
        "kernel_contract": {
            "head_q_local": 1,
            "head_kv_local": 1,
            "head_dim": HEAD_DIM,
            "page_size": PAGE_SIZE,
            "q_dtype": "float8_e4m3fn",
            "k_dtype": "float8_e4m3fn",
            "accumulator_dtype": "float32",
            "score_dtype": "float32",
            "causal_alignment": "bottom_right",
        },
    }

    gpu_case_ids = _select_gpu_test_cases(records)
    smoke_case_ids = _select_smoke_test_cases(records, gpu_case_ids)
    msa_v1_gpu_case_ids = _select_gpu_test_cases(
        records,
        case_count=MSA_V1_GPU_TEST_CASES,
        salt="prefill-msa-v1-full-v1",
    )
    msa_v1_smoke_case_ids = _select_msa_v1_smoke_test_cases(
        records,
        msa_v1_gpu_case_ids,
    )
    by_id = {record["case_id"]: record for record in records}
    cuda_graph_order = sorted(
        gpu_case_ids,
        key=lambda case_id: (
            by_id[case_id]["batch_size"],
            by_id[case_id]["metrics"]["score_bytes"],
            by_id[case_id]["metrics"]["lpt_makespan_ratio_ppm"],
            case_id,
        ),
    )
    cuda_graph_case_ids = _evenly_spaced(cuda_graph_order, CUDA_GRAPH_CASES)
    test_selection = {
        "schema_version": SCHEMA_VERSION,
        "case_kind": "inference_prefill_test_selection",
        "manifest_files": manifest_files,
        "manifest_sha256": manifest_sha256,
        "tiers": {
            "static_all": {
                "case_count": len(records),
                "selection": "all manifest cases",
            },
            "smoke": {
                "case_count": len(smoke_case_ids),
                "case_ids": smoke_case_ids,
                "selection": (
                    "full-suite extrema plus min/median/max anchors for every "
                    "batch size, then a stable SHA256 fill"
                ),
            },
            "full": {
                "case_count": len(gpu_case_ids),
                "case_ids": gpu_case_ids,
                "selection": (
                    "all extrema, all zero-useful-FLOPs shapes, five quantile "
                    "anchors per batch size, query boundary anchors, then a "
                    "stable SHA256 fill"
                ),
            },
            "msa_v1_smoke": {
                "case_count": len(msa_v1_smoke_case_ids),
                "case_ids": msa_v1_smoke_case_ids,
                "selection": (
                    "one hot anchor for every batch size, then a stable "
                    "SHA256 fill from the MSA v1 full suite"
                ),
            },
            "msa_v1_full": {
                "case_count": len(msa_v1_gpu_case_ids),
                "case_ids": msa_v1_gpu_case_ids,
                "selection": (
                    "MSA v1 extrema, zero-useful-FLOPs shapes, five quantile "
                    "anchors per batch size, query boundaries, then a stable "
                    "SHA256 fill"
                ),
            },
            "gemm_topk_e2e": {
                "case_count": len(gpu_case_ids),
                "case_ids_ref": "full",
            },
            "cuda_graph": {
                "case_count": len(cuda_graph_case_ids),
                "case_ids": cuda_graph_case_ids,
            },
            "exhaustive": {
                "case_count": len(records),
                "selection": "all manifest cases; opt-in exhaustive tier",
            },
        },
    }

    benchmark_entries = _make_benchmark_selection(records)
    representative_entries = [
        entry for entry in benchmark_entries if "representative" in entry["tags"]
    ]
    represented_calls = sum(
        entry["representative_weight"] for entry in representative_entries
    )
    if represented_calls != weighted_calls:
        raise RuntimeError(
            f"representative strata cover {represented_calls}, expected {weighted_calls}"
        )
    benchmark_selection = {
        "schema_version": SCHEMA_VERSION,
        "case_kind": "inference_prefill_benchmark_selection",
        "manifest_files": manifest_files,
        "manifest_sha256": manifest_sha256,
        "default_case_count": len(benchmark_entries),
        "selection": {
            "representative": (
                "one count-weighted medoid from each occupied "
                "batch/task/max-final-KV stratum"
            ),
            "hot": "top 32 cases by merged count",
            "batch_anchor": "the highest-count case for every observed batch size",
            "stress": (
                "round-robin extrema for zero useful FLOPs, LPT imbalance, "
                "tile waste, final KV, score workspace, TopK work, issued "
                "FLOPs, candidate tasks, valid tasks, and query skew"
            ),
            "aggregate_rule": (
                "production-weighted aggregates use only representative entries "
                "and representative_weight; hot/stress-only entries are diagnostic"
            ),
        },
        "representative_strata": {
            "case_count": len(representative_entries),
            "represented_calls": represented_calls,
            "batch_buckets": ["1", "2", "3-4", "5-8", "9-18"],
            "valid_q_tile_buckets": ["1-3", "4-20", "21-63", "64+"],
            "max_final_kv_buckets": ["<=32K", "32K-64K", "64K-128K", ">128K"],
        },
        "cases": benchmark_entries,
    }
    return {
        **manifest_outputs,
        META_FILENAME: _render_json(meta),
        TEST_FILENAME: _render_json(test_selection),
        BENCHMARK_FILENAME: _render_json(benchmark_selection),
    }


def main() -> None:
    args = _parse_args()
    source_paths = sorted(args.source_dir.glob(SOURCE_PATTERN))
    if not source_paths:
        raise RuntimeError(
            f"no source files matching {SOURCE_PATTERN} in {args.source_dir}"
        )
    outputs = _make_outputs(source_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    legacy_manifest = args.output_dir / "prefill_cases_v1.jsonl"
    if legacy_manifest.exists():
        raise RuntimeError(f"remove legacy unsharded manifest: {legacy_manifest}")
    expected_shards = {
        args.output_dir / filename
        for filename in outputs
        if filename.startswith(f"{MANIFEST_DIRNAME}/")
    }
    actual_shards = set((args.output_dir / MANIFEST_DIRNAME).glob("part-*.jsonl"))
    if unexpected_shards := actual_shards - expected_shards:
        paths = ", ".join(str(path) for path in sorted(unexpected_shards))
        raise RuntimeError(f"remove stale manifest shards: {paths}")
    for filename, rendered in outputs.items():
        output_path = args.output_dir / filename
        if args.check:
            if not output_path.is_file():
                raise RuntimeError(f"missing generated file: {output_path}")
            if output_path.read_text(encoding="utf-8") != rendered:
                raise RuntimeError(f"generated file is stale: {output_path}")
            action = "verified"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered, encoding="utf-8")
            action = "wrote"
        print(f"{action} {output_path}")


if __name__ == "__main__":
    main()
