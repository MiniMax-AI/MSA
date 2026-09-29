# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""One plan per batch shape serves changing tensors, and no batch shape triggers a recompile."""

import torch

from fmha_sm100.decode_q8kv4 import jit, plan_decode, run_decode

from .cases import KV_HEADS, SM_SCALE, SMOKE_CASES, DecodeCase, flat_page_table, make_inputs
from .conftest import run_timed
from .reference import PageDequantizer, assert_close_to_reference, sparse_decode_reference
from .runners import run_wrapper


def _run_on_plan(plan, inputs, label):
    kv_indices, kv_indptr = flat_page_table(inputs.page_table, inputs.case.seq_lens)
    return run_timed(label, lambda: run_decode(
        plan, inputs.q, (inputs.k_codes, inputs.v_codes), kv_cache_sf=(inputs.k_scale, inputs.v_scale_kernel),
        seq_lens=inputs.seq_lens, kv_indices=kv_indices, kv_indptr=kv_indptr,
        topk_indices=inputs.topk_indices, sm_scale=SM_SCALE).clone())


def test_one_plan_serves_different_lengths_pages_and_lists(device):
    base = SMOKE_CASES[1]
    plan = plan_decode(batch_size=base.batch_size, q_len_per_req=base.q_len, topk=base.topk, device=device,
                       num_q_heads=KV_HEADS * 16, num_kv_heads=KV_HEADS)
    for seed, seq_lens in ((31, base.seq_lens), (32, (300, 4_096, 12_345, 777)), (33, (1, 128, 129, 100_000))):
        case = DecodeCase(f"reuse_{seed}", seq_lens, q_len=base.q_len, topk=base.topk, seed=seed)
        inputs = make_inputs(case, device)
        reference = sparse_decode_reference(
            inputs, PageDequantizer(inputs.k_codes, inputs.k_scale), PageDequantizer(inputs.v_codes, inputs.v_scale))
        assert_close_to_reference(_run_on_plan(plan, inputs, case.name), reference, label=case.name)


def test_batch_shapes_share_compiled_variants(device):
    """Batch size, lengths and page tables never enter the JIT key."""
    for case in SMOKE_CASES:
        run_wrapper(make_inputs(case, device), label=f"warm {case.name}")
    loaded = set(jit._variant_manager._loaded)
    assert loaded, "the warm-up must have loaded at least one variant"
    for case in (DecodeCase("b3_s5000_q2", (5_000, 6_000, 7_000), q_len=2, topk=16, seed=41),
                 DecodeCase("b16_s2000_q1", (2_000,) * 16, q_len=1, topk=16, seed=42),
                 DecodeCase("b1_s200000_q8", (200_000,), q_len=8, topk=16, seed=43)):
        run_wrapper(make_inputs(case, device), label=case.name)
    assert set(jit._variant_manager._loaded) == loaded, "a new batch shape recompiled a kernel"
