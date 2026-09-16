"""Static checks for the canonical MSA v1 decode benchmark selection."""

import torch

from benchmarks.inference.msa_v1.decode.cases import (
    BATCH_SIZES,
    FULL_CASES,
    Q_LEN_PER_REQ,
    SEQ_LENGTHS,
    make_seq_lens,
)


def test_canonical_decode_benchmark_selection() -> None:
    assert len(FULL_CASES) == 28
    assert sum(case.weight for case in FULL_CASES) == 10_000
    assert {(case.batch_size, case.nominal_seq_len) for case in FULL_CASES} == {
        (batch_size, seq_len)
        for batch_size in BATCH_SIZES
        for seq_len in SEQ_LENGTHS
    }
    assert {case.q_len_per_req for case in FULL_CASES} == {Q_LEN_PER_REQ}


def test_canonical_decode_benchmark_lengths_are_varlen() -> None:
    for case in FULL_CASES:
        first = make_seq_lens(case)
        second = make_seq_lens(case)
        assert first.dtype == torch.int32
        assert first.shape == (case.batch_size,)
        assert torch.equal(first, second)
        assert int(first.sum()) == case.batch_size * case.nominal_seq_len
        assert torch.unique(first).numel() > 1
