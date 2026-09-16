"""Static tests for the shared MSA v1 performance acceptance helpers."""

import pytest

from benchmarks.inference.msa_v1.acceptance import compare_e2e_results
from benchmarks.inference.msa_v1.hardware import peak_spec, roofline_metrics


def _row(case_id: str, latency_us: float, weight: int) -> dict[str, object]:
    return {
        "case": case_id,
        "e2e_latency_us": latency_us,
        "weight": weight,
    }


def test_acceptance_requires_weighted_improvement_and_per_case_gate() -> None:
    baseline = {
        "results": [
            _row("small", 10.0, 1),
            _row("large", 100.0, 9),
        ]
    }
    passing = compare_e2e_results(
        [_row("small", 10.4, 1), _row("large", 95.0, 9)],
        baseline,
        case_key="case",
        latency_key="e2e_latency_us",
        weight_key="weight",
    )
    assert passing["passed"]
    assert passing["weighted_mean_strictly_improved"]

    regressed = compare_e2e_results(
        [_row("small", 10.6, 1), _row("large", 90.0, 9)],
        baseline,
        case_key="case",
        latency_key="e2e_latency_us",
        weight_key="weight",
    )
    assert not regressed["passed"]
    assert regressed["weighted_mean_strictly_improved"]
    assert regressed["regressed_cases"] == ["small"]


def test_acceptance_rejects_case_mismatch() -> None:
    with pytest.raises(ValueError, match="case mismatch"):
        compare_e2e_results(
            [_row("candidate", 1.0, 1)],
            {"results": [_row("baseline", 1.0, 1)]},
            case_key="case",
            latency_key="e2e_latency_us",
            weight_key="weight",
        )


def test_roofline_metrics_are_diagnostic_and_dimensionally_consistent() -> None:
    spec = peak_spec("NVIDIA GB300")
    metrics = roofline_metrics(
        device_name="NVIDIA GB300",
        useful_flops=2_000_000,
        logical_bytes=1_000_000,
        latency_us=10.0,
    )
    assert spec.hbm_tb_s == 8.0
    assert metrics["roofline_bound"] == "memory"
    assert metrics["logical_mbu"] == pytest.approx(0.0125)
    assert metrics["logical_mfu"] > 0.0

    bf16_metrics = roofline_metrics(
        device_name="NVIDIA GB300",
        useful_flops=2_000_000,
        logical_bytes=1_000_000,
        latency_us=10.0,
        compute_dtype="bf16",
    )
    assert spec.dense_bf16_tflops < spec.dense_fp8_tflops
    assert bf16_metrics["peak_dense_bf16_tflops"] == spec.dense_bf16_tflops


def test_unknown_device_has_no_silent_peak_fallback() -> None:
    with pytest.raises(ValueError, match="no public MBU/MFU peak"):
        peak_spec("Unknown GPU")
