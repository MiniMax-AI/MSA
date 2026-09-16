"""Shared helpers for selecting inference correctness tiers."""

from __future__ import annotations

import os

from datas.inference.cases import load_prefill_test_cases


def active_inference_suite() -> str:
    suite = os.environ.get("MINIMAX_INFERENCE_TEST_SUITE", "smoke")
    if suite not in {"smoke", "full"}:
        raise ValueError("MINIMAX_INFERENCE_TEST_SUITE must be smoke or full")
    return suite


def selected_prefill_test_cases():
    return load_prefill_test_cases(active_inference_suite())


def selected_msa_v1_prefill_test_cases():
    """Return the MSA v1-specific 32/512 correctness selection."""

    return load_prefill_test_cases(f"msa_v1_{active_inference_suite()}")


__all__ = [
    "active_inference_suite",
    "selected_msa_v1_prefill_test_cases",
    "selected_prefill_test_cases",
]
