"""Static coverage checks for canonical MSA v1 decode correctness cases."""

import torch

from tests.inference.msa_v1.decode.cases import (
    FULL_CASES,
    Q_LENGTHS,
    SMOKE_CASES,
    make_seq_lens,
)


def test_decode_correctness_case_counts_and_coverage() -> None:
    assert len(FULL_CASES) == 256
    assert len(SMOKE_CASES) == 96
    assert sum(case.production for case in FULL_CASES) == 192
    assert {case.case_id for case in SMOKE_CASES} <= {
        case.case_id for case in FULL_CASES
    }
    assert {case.q_len_per_req for case in FULL_CASES} == set(Q_LENGTHS)
    assert min(case.batch_size for case in FULL_CASES) == 1
    assert max(case.batch_size for case in FULL_CASES) == 512
    assert min(case.nominal_seq_len for case in FULL_CASES) == 1_000
    assert max(case.nominal_seq_len for case in FULL_CASES) == 512 * 1024
    assert {case.page_layout for case in FULL_CASES} == {
        "disjoint",
        "permuted",
        "shared_prefix",
    }


def test_decode_correctness_lengths_are_deterministic() -> None:
    for case in FULL_CASES:
        first = make_seq_lens(case)
        second = make_seq_lens(case)
        assert first.dtype == torch.int32
        assert first.shape == (case.batch_size,)
        assert torch.equal(first, second)
        assert int(first.min()) >= case.q_len_per_req
        assert int(first.max()) <= 512 * 1024
