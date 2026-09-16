"""Q8KV4 indexer view of the canonical MSA v1 decode workloads."""

from benchmarks.inference.msa_v1.decode.cases import (
    BATCH_SIZES,
    DEFAULT_SEED,
    FULL_CASES,
    SEQ_LENGTHS,
    SMOKE_CASES,
    DecodeBenchmarkCase,
    benchmark_cases,
    make_seq_lens,
)

STANDARD_CASES = FULL_CASES
CORE_CASES = FULL_CASES
STRESS_CASES = ()
DecodeCase = DecodeBenchmarkCase

__all__ = [
    "BATCH_SIZES",
    "CORE_CASES",
    "DEFAULT_SEED",
    "FULL_CASES",
    "SEQ_LENGTHS",
    "SMOKE_CASES",
    "STANDARD_CASES",
    "STRESS_CASES",
    "DecodeCase",
    "benchmark_cases",
    "make_seq_lens",
]
