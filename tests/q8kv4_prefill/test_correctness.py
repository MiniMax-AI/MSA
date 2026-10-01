# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""The wrapper against the fp32 reference, and bitwise determinism across runs."""

import dataclasses
import random

import pytest
import torch

from .cases import FULL_CASES, SMOKE_CASES, make_inputs, make_topk_lists
from .reference import assert_close_to_reference, sparse_prefill_reference
from .runners import run_wrapper

CASES = [pytest.param(case, id=case.name) for case in SMOKE_CASES] + [
    pytest.param(case, id=case.name, marks=pytest.mark.full) for case in FULL_CASES
]


@pytest.mark.parametrize("case", CASES)
def test_matches_reference_and_is_deterministic(device, case):
    inputs = make_inputs(case, device)
    out, lse = run_wrapper(inputs, label=case.name)
    again, again_lse = run_wrapper(inputs, label=f"{case.name} again")
    assert torch.equal(out, again) and torch.equal(lse, again_lse), "runs must be bitwise equal"
    reference, reference_lse = sparse_prefill_reference(inputs)
    assert_close_to_reference(out, reference, label=case.name, lse=lse, reference_lse=reference_lse)


def test_seqused_k_moves_the_causal_alignment(device):
    """``seqused_k`` shorter than the cached KV (``fmha_sm100``'s ``qo_offset``) masks the tail and
    places query ``i`` of a chunk at ``seqused_k - q_len + i``; equal to the KV lengths it is a
    no-op."""
    case = SMOKE_CASES[1]
    inputs = make_inputs(case, device)
    seqused = [k_len - shrink for k_len, shrink in zip(case.k_lens, (0, 700, 129), strict=True)]
    shifted = dataclasses.replace(case, k_lens=tuple(seqused))
    inputs.topk_indices = make_topk_lists(shifted, random.Random(7)).to(device)
    seqused_k = torch.tensor(seqused, dtype=torch.int32, device=device)
    out, lse = run_wrapper(inputs, plan_kwargs={"seqused_k": seqused_k}, label="seqused_k")
    reference, reference_lse = sparse_prefill_reference(inputs, seqused_k=seqused)
    assert_close_to_reference(out, reference, label="seqused_k", lse=lse, reference_lse=reference_lse)

    inputs = make_inputs(case, device)
    full = torch.tensor(case.k_lens, dtype=torch.int32, device=device)
    explicit = run_wrapper(inputs, plan_kwargs={"seqused_k": full}, label="seqused_k = k_lens")
    assert all(map(torch.equal, explicit, run_wrapper(inputs, label="no seqused_k")))
