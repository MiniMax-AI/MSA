"""Shared E2E performance acceptance checks for MSA v1 benchmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def compare_e2e_results(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_payload: Mapping[str, Any],
    *,
    case_key: str,
    latency_key: str,
    weight_key: str,
    maximum_case_regression: float = 0.05,
) -> dict[str, Any]:
    """Require a lower weighted mean and no case above the regression gate."""

    baseline_rows = baseline_payload.get("results")
    if not isinstance(baseline_rows, list):
        raise ValueError("baseline payload must contain a results list")
    baseline_by_case = {str(row[case_key]): row for row in baseline_rows}
    candidate_cases = {str(row[case_key]) for row in candidate_rows}
    if candidate_cases != set(baseline_by_case):
        missing = sorted(candidate_cases - set(baseline_by_case))
        extra = sorted(set(baseline_by_case) - candidate_cases)
        raise ValueError(
            f"baseline/candidate case mismatch: missing={missing}, extra={extra}"
        )

    def weight(row: Mapping[str, Any]) -> int:
        value = row.get(weight_key)
        return 0 if value is None else int(value)

    total_weight = sum(weight(row) for row in candidate_rows)
    if total_weight <= 0:
        raise ValueError("benchmark weights must sum to a positive value")
    candidate_weighted = sum(
        weight(row) * float(row[latency_key]) for row in candidate_rows
    ) / total_weight
    baseline_weighted = sum(
        weight(row)
        * float(baseline_by_case[str(row[case_key])][latency_key])
        for row in candidate_rows
    ) / total_weight

    per_case = {}
    regressed_cases = []
    for row in candidate_rows:
        case_id = str(row[case_key])
        candidate_latency = float(row[latency_key])
        baseline_latency = float(baseline_by_case[case_id][latency_key])
        regression = candidate_latency / baseline_latency - 1.0
        per_case[case_id] = {
            "baseline_latency_us": baseline_latency,
            "candidate_latency_us": candidate_latency,
            "relative_change": regression,
            "passed": regression <= maximum_case_regression,
        }
        if regression > maximum_case_regression:
            regressed_cases.append(case_id)

    weighted_passed = candidate_weighted < baseline_weighted
    passed = weighted_passed and not regressed_cases
    return {
        "passed": passed,
        "weighted_mean_strictly_improved": weighted_passed,
        "maximum_case_regression": maximum_case_regression,
        "regressed_cases": regressed_cases,
        "baseline_weighted_mean_e2e_latency_us": baseline_weighted,
        "candidate_weighted_mean_e2e_latency_us": candidate_weighted,
        "weighted_relative_change": candidate_weighted / baseline_weighted - 1.0,
        "per_case": per_case,
    }


__all__ = ["compare_e2e_results"]
