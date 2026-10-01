# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Per-tensor global scales and block-scale staging."""

import pytest
import torch

from .cases import SM_SCALE, SMOKE_CASES, global_scale, make_inputs, scale_bytes_times_pow2
from .conftest import run_timed
from .reference import assert_close_to_reference, sparse_prefill_reference
from .runners import plan_wrapper, run_wrapper

CASES = [pytest.param(case, id=case.name) for case in SMOKE_CASES[:2]]
K_GLOBAL, V_GLOBAL = 0.7, 1.3


def _te_style(inputs):
    """Block scales scaled by 2^8 (up to 256, products up to 1536 > 448) with the global scales
    absorbing the factor: the same values as ``inputs`` times K_GLOBAL and V_GLOBAL."""
    k_scale = scale_bytes_times_pow2(inputs.k_scale, 8)
    v_scale = scale_bytes_times_pow2(inputs.v_scale_kernel, 8)
    device = inputs.q.device
    return k_scale, v_scale, (global_scale(K_GLOBAL / 256, device), global_scale(V_GLOBAL / 256, device))


@pytest.mark.parametrize("case", CASES)
def test_unit_global_scales_and_staging_identity_are_bitwise(device, case):
    inputs = make_inputs(case, device)
    baseline = run_wrapper(inputs, label=f"{case.name} baseline")
    unit = global_scale(1.0, device)
    unit_run = run_wrapper(inputs, kv_global_scale=(unit, unit), label=f"{case.name} unit")
    assert all(map(torch.equal, unit_run, baseline))
    # Scales x8 with shift 3 recover the original scales exactly; sm_scale / 8 and a V global scale
    # of 1/8 cancel the folded factor exactly in fp32.
    staged = run_wrapper(
        inputs,
        kv_sf=(scale_bytes_times_pow2(inputs.k_scale, 3), scale_bytes_times_pow2(inputs.v_scale_kernel, 3)),
        plan_kwargs={"block_scale_shift": 3, "sm_scale": SM_SCALE / 8},
        kv_global_scale=(global_scale(1.0, device), global_scale(0.125, device)),
        label=f"{case.name} staging identity",
    )
    assert all(map(torch.equal, staged, baseline))


@pytest.mark.parametrize("case", CASES)
def test_te_style_cache_matches_reference_and_needs_staging(device, case):
    inputs = make_inputs(case, device)
    k_scale, v_scale, globals_ = _te_style(inputs)
    reference, reference_lse = sparse_prefill_reference(
        inputs,
        k_scale=scale_bytes_times_pow2(inputs.k_scale, 8),
        v_scale=scale_bytes_times_pow2(inputs.v_scale, 8),
        k_global_scale=K_GLOBAL / 256,
        v_global_scale=V_GLOBAL / 256,
        block_scale_shift=3,
    )
    out, lse = run_wrapper(inputs, kv_sf=(k_scale, v_scale), plan_kwargs={"block_scale_shift": 3},
                           kv_global_scale=globals_, label=f"{case.name} TE-style")
    assert_close_to_reference(out, reference, label=f"{case.name} TE-style shift 3", lse=lse,
                              reference_lse=reference_lse)
    unstaged, _ = run_wrapper(inputs, kv_sf=(k_scale, v_scale), kv_global_scale=globals_,
                              label=f"{case.name} TE-style unstaged")
    diff = (unstaged.float() - reference.float()).abs()
    assert int((diff > 0.05 + 0.05 * reference.float().abs()).sum()) > 0, (
        "full-range block scales must saturate without staging; the control lost its purpose"
    )


@pytest.mark.parametrize("case", CASES[:1])
def test_graph_replay_follows_updated_global_scales(device, case):
    inputs = make_inputs(case, device)
    k_scale, v_scale, (k_global, v_global) = _te_style(inputs)
    wrapper = plan_wrapper(inputs, block_scale_shift=3)
    run = lambda: wrapper.run(inputs.q, (inputs.k_codes, inputs.v_codes), kv_cache_sf=(k_scale, v_scale),
                              kv_global_scale=(k_global, v_global))
    eager_first = run_timed("eager first scales", run).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager_first)
    k_global.fill_(0.35 / 256)
    v_global.fill_(2.6 / 256)
    graph.replay()
    torch.cuda.synchronize()
    replayed = out.clone()
    assert torch.equal(replayed, run_timed("eager updated scales", run))
    assert not torch.equal(replayed, eager_first)
