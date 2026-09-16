"""Real inference benchmark cases for Q8KV8 prefill attention."""

from benchmarks.inference.msa_v1.attention.prefill.cases import (
    HEAD_DIM,
    PAGE_SIZE,
    Q_HEADS,
    TOPK,
    PrefillCase,
    real_prefill_cases,
    warmup_case,
)

__all__ = [
    "HEAD_DIM",
    "PAGE_SIZE",
    "Q_HEADS",
    "TOPK",
    "PrefillCase",
    "real_prefill_cases",
    "warmup_case",
]
