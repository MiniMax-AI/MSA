"""Q8KV8 consumes the same canonical MSA v1 decode benchmark selection."""

from tests.inference.msa_v1.indexer.decode.tp4_q8kv4.test_benchmark_cases import (
    test_canonical_decode_benchmark_lengths_are_varlen,
    test_canonical_decode_benchmark_selection,
)

__all__ = [
    "test_canonical_decode_benchmark_lengths_are_varlen",
    "test_canonical_decode_benchmark_selection",
]
