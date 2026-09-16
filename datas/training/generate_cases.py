#!/usr/bin/env python3
"""Generate sanitized training manifests from the local 192K/CP16 CSV."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parents[1]
DEFAULT_SOURCE = (
    REPO_ROOT / "datas" / "192k_cp16" / "M3.1-192k-long-0810-dist-mm.csv"
)
DEFAULT_OUTPUT_DIR = TRAINING_DIR / "real"

SCHEMA_VERSION = 1
TOTAL_TOKENS = 192 * 1024
CP_SIZE = 16
CHUNK_SIZE = 1024
CHUNKS_PER_RANK = 12
EXPECTED_CASES = 1500
EXPECTED_UNIQUE_CASES = 1253
SMOKE_CASES = 96
BENCHMARK_CASES = 32

MANIFEST_FILENAME = "cp16_192k_cases_v1.jsonl"
TEST_SELECTION_FILENAME = "cp16_192k_test_selection_v1.json"
BENCHMARK_SELECTION_FILENAME = "cp16_192k_benchmark_selection_v1.json"

METRIC_KEYS = (
    "total_tokens",
    "num_sequences",
    "min_seqlen",
    "median_seqlen",
    "max_seqlen",
    "causal_elements",
)


@dataclass(frozen=True)
class SanitizedCase:
    cu_seqlens: tuple[int, ...]
    metrics: dict[str, int]
    content_hash: str
    count: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that committed outputs are byte-for-byte reproducible",
    )
    return parser.parse_args()


def _canonical_cu_seqlens(cu_seqlens: Sequence[int]) -> bytes:
    return json.dumps(list(cu_seqlens), separators=(",", ":")).encode("utf-8")


def _shape_metrics(cu_seqlens: tuple[int, ...]) -> dict[str, int]:
    lengths = tuple(
        end - begin for begin, end in zip(cu_seqlens, cu_seqlens[1:])
    )
    ordered = sorted(lengths)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[midpoint]
    else:
        median = (ordered[midpoint - 1] + ordered[midpoint]) // 2
    return {
        "total_tokens": cu_seqlens[-1],
        "num_sequences": len(lengths),
        "min_seqlen": min(lengths),
        "median_seqlen": median,
        "max_seqlen": max(lengths),
        "causal_elements": sum(length * (length + 1) // 2 for length in lengths),
    }


def _parse_cu_seqlens(raw: str, row_number: int) -> tuple[int, ...]:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"row {row_number}: invalid cu_seqlens") from exc
    if not isinstance(value, list) or any(
        isinstance(offset, bool) or not isinstance(offset, int) for offset in value
    ):
        raise ValueError(f"row {row_number}: cu_seqlens must be an integer array")
    offsets = tuple(value)
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != TOTAL_TOKENS:
        raise ValueError(f"row {row_number}: invalid cu_seqlens endpoints")
    if any(begin >= end for begin, end in zip(offsets, offsets[1:])):
        raise ValueError(f"row {row_number}: cu_seqlens must strictly increase")
    return offsets


def _load_source(path: Path) -> list[SanitizedCase]:
    merged: dict[tuple[int, ...], int] = {}
    row_count = 0
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or "cu_seqlens" not in reader.fieldnames:
            raise ValueError("training CSV must contain cu_seqlens")
        for row_number, row in enumerate(reader, 2):
            cu_seqlens = _parse_cu_seqlens(row["cu_seqlens"], row_number)
            merged[cu_seqlens] = merged.get(cu_seqlens, 0) + 1
            row_count += 1
    if row_count != EXPECTED_CASES:
        raise ValueError(f"training CSV must contain {EXPECTED_CASES} cases")
    if len(merged) != EXPECTED_UNIQUE_CASES:
        raise ValueError(
            f"training CSV must contain {EXPECTED_UNIQUE_CASES} unique shapes"
        )
    cases = [
        SanitizedCase(
            cu_seqlens=cu_seqlens,
            metrics=_shape_metrics(cu_seqlens),
            content_hash=hashlib.sha256(
                _canonical_cu_seqlens(cu_seqlens)
            ).hexdigest(),
            count=count,
        )
        for cu_seqlens, count in merged.items()
    ]
    # Content order deliberately replaces source row/sample identifiers.
    return sorted(cases, key=lambda case: case.content_hash)


def _manifest_rows(cases: Sequence[SanitizedCase]) -> list[dict[str, object]]:
    return [
        {
            "case_id": case_id,
            "cu_seqlens": list(case.cu_seqlens),
            "metrics": {key: case.metrics[key] for key in METRIC_KEYS},
        }
        for case_id, case in enumerate(cases)
    ]


def _manifest_text(rows: Sequence[dict[str, object]]) -> str:
    return "".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows
    )


def _complexity_order(cases: Sequence[SanitizedCase]) -> list[int]:
    expanded_ids = [
        case_id
        for case_id, case in enumerate(cases)
        for _ in range(case.count)
    ]
    return sorted(
        expanded_ids,
        key=lambda case_id: (
            cases[case_id].metrics["causal_elements"],
            cases[case_id].metrics["num_sequences"],
            cases[case_id].metrics["max_seqlen"],
            case_id,
        ),
    )


def _even_quantiles(ordered_ids: Sequence[int], count: int) -> list[int]:
    if count < 2 or count > len(ordered_ids):
        raise ValueError("invalid quantile selection count")
    return [
        ordered_ids[round(index * (len(ordered_ids) - 1) / (count - 1))]
        for index in range(count)
    ]


def _selection_entry(
    selection_id: int,
    case_id: int,
    *,
    structured: bool = False,
) -> dict[str, object]:
    return {
        "selection_id": selection_id,
        "case_id": case_id,
        "rank": selection_id % CP_SIZE,
        "structured": structured,
    }


def _test_selection(
    cases: Sequence[SanitizedCase], manifest_sha256: str
) -> dict[str, object]:
    expanded_ids = [
        case_id
        for case_id, case in enumerate(cases)
        for _ in range(case.count)
    ]
    full = [
        _selection_entry(selection_id, case_id, structured=selection_id % 64 == 0)
        for selection_id, case_id in enumerate(expanded_ids)
    ]
    smoke_ids = _even_quantiles(_complexity_order(cases), SMOKE_CASES)
    smoke = [
        _selection_entry(index, case_id, structured=index % 16 == 0)
        for index, case_id in enumerate(smoke_ids)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "scenario": {
            "name": "192k_cp16",
            "total_tokens": TOTAL_TOKENS,
            "cp_size": CP_SIZE,
            "chunk_size": CHUNK_SIZE,
            "chunks_per_rank": CHUNKS_PER_RANK,
        },
        "tiers": {
            "static_all": {"case_count": len(cases)},
            "full_gpu": {"case_count": len(full), "cases": full},
            "smoke": {"case_count": len(smoke), "cases": smoke},
        },
    }


def _benchmark_selection(
    cases: Sequence[SanitizedCase], manifest_sha256: str
) -> dict[str, object]:
    ordered_ids = _complexity_order(cases)
    entries = []
    for stratum_id in range(BENCHMARK_CASES):
        begin = stratum_id * len(ordered_ids) // BENCHMARK_CASES
        end = (stratum_id + 1) * len(ordered_ids) // BENCHMARK_CASES
        stratum = ordered_ids[begin:end]
        case_id = stratum[len(stratum) // 2]
        entries.append(
            {
                "benchmark_case_id": stratum_id,
                "case_id": case_id,
                "rank": stratum_id % CP_SIZE,
                "representative_weight": len(stratum),
                "stratum": f"shape_{stratum_id:02d}",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "scenario": "192k_cp16",
        "case_count": len(entries),
        "represented_cases": sum(
            int(entry["representative_weight"]) for entry in entries
        ),
        "cases": entries,
    }


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _write_or_check(path: Path, content: str, *, check: bool) -> None:
    if check:
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"generated file is stale: {path}")
        action = "verified"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        action = "wrote"
    print(f"{action} {path}")


def main() -> None:
    args = _parse_args()
    cases = _load_source(args.source)
    rows = _manifest_rows(cases)
    manifest = _manifest_text(rows)
    manifest_sha256 = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    outputs = {
        MANIFEST_FILENAME: manifest,
        TEST_SELECTION_FILENAME: _json_text(
            _test_selection(cases, manifest_sha256)
        ),
        BENCHMARK_SELECTION_FILENAME: _json_text(
            _benchmark_selection(cases, manifest_sha256)
        ),
    }
    for filename, content in outputs.items():
        _write_or_check(args.output_dir / filename, content, check=args.check)


if __name__ == "__main__":
    main()
