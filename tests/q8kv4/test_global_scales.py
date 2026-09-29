# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Per-tensor global scales and block-scale staging."""

import pytest
import torch

from fmha_sm100.decode_q8kv4 import BatchDecodeWithPagedKVCacheWrapper

from .cases import KV_HEADS, SM_SCALE, SMOKE_CASES, global_scale, make_inputs, scale_bytes_times_pow2
from .conftest import run_timed
from .reference import PageDequantizer, assert_close_to_reference, sparse_decode_reference
from .runners import run_wrapper

CASES = [pytest.param(case, id=case.name) for case in SMOKE_CASES[:2]]
K_GLOBAL, V_GLOBAL = 0.7, 1.3


def _te_style(inputs):
    """Block scales scaled by 2^8 (up to 256, products up to 1536 > 448) with the global scales
    absorbing the factor: the same values as ``inputs`` times K_GLOBAL and V_GLOBAL."""
    k_scale = scale_bytes_times_pow2(inputs.k_scale, 8)
    v_scale = scale_bytes_times_pow2(inputs.v_scale_kernel, 8)
    return k_scale, v_scale, (global_scale(K_GLOBAL / 256, inputs.q.device), global_scale(V_GLOBAL / 256, inputs.q.device))


@pytest.mark.parametrize("case", CASES)
def test_unit_global_scales_and_staging_identity_are_bitwise(device, case):
    inputs = make_inputs(case, device)
    baseline = run_wrapper(inputs, label=f"{case.name} baseline")
    unit = global_scale(1.0, device)
    assert torch.equal(run_wrapper(inputs, kv_global_scale=(unit, unit), label=f"{case.name} unit"), baseline)
    # Scales x8 with shift 3 recover the original scales exactly; sm_scale / 8 and a V global scale
    # of 1/8 cancel the folded factor exactly in fp32.
    staged = run_wrapper(
        inputs, block_scale_shift=3, sm_scale=SM_SCALE / 8,
        k_scale=scale_bytes_times_pow2(inputs.k_scale, 3),
        v_scale_kernel=scale_bytes_times_pow2(inputs.v_scale_kernel, 3),
        kv_global_scale=(global_scale(1.0, device), global_scale(0.125, device)),
        label=f"{case.name} staging identity",
    )
    assert torch.equal(staged, baseline)


@pytest.mark.parametrize("num_kv_splits", (None, 4), ids=("auto", "split4"))
@pytest.mark.parametrize("case", CASES)
def test_te_style_cache_matches_reference_and_needs_staging(device, case, num_kv_splits):
    inputs = make_inputs(case, device)
    k_scale, v_scale, globals_ = _te_style(inputs)
    reference = sparse_decode_reference(
        inputs,
        PageDequantizer(inputs.k_codes, scale_bytes_times_pow2(inputs.k_scale, 8), global_scale=K_GLOBAL / 256, block_scale_shift=3),
        PageDequantizer(inputs.v_codes, scale_bytes_times_pow2(inputs.v_scale, 8), global_scale=V_GLOBAL / 256, block_scale_shift=3),
    )
    out = run_wrapper(inputs, block_scale_shift=3, k_scale=k_scale, v_scale_kernel=v_scale,
                      kv_global_scale=globals_, num_kv_splits=num_kv_splits, label=f"{case.name} TE-style")
    assert_close_to_reference(out, reference, label=f"{case.name} TE-style shift 3")
    if num_kv_splits is None:
        unstaged = run_wrapper(inputs, block_scale_shift=0, k_scale=k_scale, v_scale_kernel=v_scale,
                               kv_global_scale=globals_, label=f"{case.name} TE-style unstaged")
        diff = (unstaged.float() - reference.float()).abs()
        assert int((diff > 0.05 + 0.05 * reference.float().abs()).sum()) > 0, (
            "full-range block scales must saturate without staging; the control lost its purpose"
        )


@pytest.mark.parametrize("case", CASES[:1])
def test_graph_replay_follows_updated_global_scales(device, case):
    inputs = make_inputs(case, device)
    k_scale, v_scale, (k_global, v_global) = _te_style(inputs)
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    wrapper.plan(inputs.topk_indices, inputs.page_table, inputs.seq_lens, q_len_per_req=case.q_len,
                 num_q_heads=inputs.num_q_heads, num_kv_heads=KV_HEADS, block_scale_shift=3)
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
