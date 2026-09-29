# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Q8KV4 sparse decode against the independent reference, both GQA ratios, split schedules."""

import pytest
import torch

from .cases import FULL_CASES, SMOKE_CASES, make_inputs
from .reference import PageDequantizer, assert_close_to_reference, sparse_decode_reference
from .runners import run_wrapper

CASES = [pytest.param(case, id=case.name) for case in SMOKE_CASES] + [
    pytest.param(case, id=case.name, marks=pytest.mark.full) for case in FULL_CASES
]


@pytest.mark.parametrize("gqa", (16, 8), ids=("gqa16", "gqa8"))
@pytest.mark.parametrize("case", CASES)
def test_matches_reference_and_is_deterministic(device, case, gqa):
    inputs = make_inputs(case, device, gqa=gqa)
    reference = sparse_decode_reference(
        inputs, PageDequantizer(inputs.k_codes, inputs.k_scale), PageDequantizer(inputs.v_codes, inputs.v_scale)
    )
    out = run_wrapper(inputs, label=f"{case.name} gqa{gqa}")
    assert_close_to_reference(out, reference, label=f"{case.name} gqa{gqa}")
    again = run_wrapper(inputs, label=f"{case.name} gqa{gqa} repeat")
    assert torch.equal(out, again), "the kernel must be bitwise deterministic"


@pytest.mark.parametrize("num_kv_splits", (1, 4), ids=("nosplit", "split4"))
@pytest.mark.parametrize("case", [pytest.param(SMOKE_CASES[0], id=SMOKE_CASES[0].name),
                                  pytest.param(SMOKE_CASES[2], id=SMOKE_CASES[2].name)])
def test_forced_split_schedules_match_reference(device, case, num_kv_splits):
    """The legacy fixed-split schedule (separate reduction kernel) and no split at all."""
    inputs = make_inputs(case, device)
    reference = sparse_decode_reference(
        inputs, PageDequantizer(inputs.k_codes, inputs.k_scale), PageDequantizer(inputs.v_codes, inputs.v_scale)
    )
    out = run_wrapper(inputs, num_kv_splits=num_kv_splits, label=f"{case.name} split{num_kv_splits}")
    assert_close_to_reference(out, reference, label=f"{case.name} split{num_kv_splits}")
