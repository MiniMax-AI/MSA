"""Q8KV4 attention view of the canonical MSA v1 decode workloads."""

from benchmarks.inference.msa_v1.decode.cases import (
    BATCH_SIZES,
    DEFAULT_SEED,
    FULL_CASES,
    Q_LEN_PER_REQ,
    SEQ_LENGTHS,
    SMOKE_CASES,
    DecodeBenchmarkCase,
    benchmark_cases,
    make_seq_lens,
)

Q_LENGTHS = (Q_LEN_PER_REQ,)
STANDARD_CASES = FULL_CASES
DecodeAttentionCase = DecodeBenchmarkCase

__all__ = [
    "BATCH_SIZES",
    "DEFAULT_SEED",
    "FULL_CASES",
    "Q_LENGTHS",
    "SEQ_LENGTHS",
    "SMOKE_CASES",
    "STANDARD_CASES",
    "DecodeAttentionCase",
    "benchmark_cases",
    "make_seq_lens",
]
