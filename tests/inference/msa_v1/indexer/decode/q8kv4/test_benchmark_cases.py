"""Static checks for the canonical MSA v1 decode benchmark selection."""

import torch

from benchmarks.inference.msa_v1.decode.cases import (
    BATCH_SIZES,
    FULL_CASES,
    LOW_LATENCY_CASES,
    Q_LEN_PER_REQ,
    SEQ_LENGTHS,
    benchmark_cases,
    make_seq_lens,
)


def test_canonical_decode_benchmark_selection() -> None:
    assert benchmark_cases("low-latency") is LOW_LATENCY_CASES
    assert len(LOW_LATENCY_CASES) == 12
    assert {(case.batch_size, case.nominal_seq_len) for case in LOW_LATENCY_CASES} == {
        (batch, length) for batch in (1, 2, 4) for length in (1000, 4000, 8000, 32000)
    }
    assert all(case.weight == 0 for case in LOW_LATENCY_CASES)
    assert len({case.name for case in FULL_CASES + LOW_LATENCY_CASES}) == 40
    assert len(FULL_CASES) == 28
    assert sum(case.weight for case in FULL_CASES) == 10_000
    assert {(case.batch_size, case.nominal_seq_len) for case in FULL_CASES} == {
        (batch_size, seq_len) for batch_size in BATCH_SIZES for seq_len in SEQ_LENGTHS
    }
    assert {case.q_len_per_req for case in FULL_CASES} == {Q_LEN_PER_REQ}


def test_canonical_decode_benchmark_lengths_are_varlen() -> None:
    for case in FULL_CASES + LOW_LATENCY_CASES:
        first = make_seq_lens(case)
        second = make_seq_lens(case)
        assert first.dtype == torch.int32
        assert first.shape == (case.batch_size,)
        assert torch.equal(first, second)
        assert int(first.sum()) == case.batch_size * case.nominal_seq_len
        if case.batch_size == 1:
            assert first.tolist() == [case.nominal_seq_len]
        else:
            assert torch.unique(first).numel() > 1
