# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""fmha_sm100_plan / fmha_sm100 route NVFP4 sparse prefill to the Q8KV4 kernel and keep the
CuTe-DSL NVFP4 kernel for everything else."""

import dataclasses

import pytest
import torch

from fmha_sm100.api import fmha_sm100, fmha_sm100_plan, nvfp4_head_slot_views
from fmha_sm100.q8kv4_prefill_adapter import PLAN_KEY

from .cases import HEAD_DIM, PAGE_SIZE, Q_HEADS_PER_KV, SM_SCALE, PrefillCase, global_scale, make_inputs, pack_head_slot_pages, pack_vllm_pages
from .conftest import run_timed
from .reference import assert_close_to_reference, sparse_prefill_reference
from .runners import run_wrapper

# Lists in fmha_sm100's ascending order; 32 blocks cover every page of these requests, so any
# causal offset keeps each query's own page selected.
CASE = PrefillCase("b3_api", (300, 64, 513), (300, 2_000, 3_585), topk=32, seed=31, ascending=True)


def _loosely_close(out, exact, *, label):
    """BF16-Q runs of the CuTe-DSL kernel against the exact reference: the decode tests' bound."""
    diff = (out.float() - exact.float()).abs()
    violations = int((diff > 0.05 + 0.05 * exact.float().abs()).sum())
    assert torch.isfinite(out).all() and violations == 0, (
        f"{label}: {violations} elements outside tolerance, max |diff| {diff.max().item():.4g}"
    )


def _within_fp8_noise(out, exact, *, label):
    """E4M3-Q runs of another FP8 datapath (the CuTe-DSL kernel, Q8KV4 decode) against the exact
    reference. Their roundings differ from Q8KV4 prefill's, so the bound is statistical: the
    Q8KV4 prefill datapath itself sits at mean 2.2e-3, p99.99 0.09 and max 0.28 on these cases."""
    diff = (out.float() - exact.float()).abs().flatten()
    tail = torch.kthvalue(diff, max(1, int(diff.numel() * 0.9999))).values.item()
    assert torch.isfinite(out).all(), f"{label}: non-finite output"
    assert diff.mean().item() <= 5e-3 and tail <= 0.15 and diff.max().item() <= 0.5, (
        f"{label}: mean |diff| {diff.mean().item():.3g}, p99.99 {tail:.3g}, max {diff.max().item():.3g}"
    )


class ApiHarness:
    def __init__(self, device, case=CASE):
        self.case = case
        self.inputs = make_inputs(case, device)
        self.k_packed, self.v_packed = pack_head_slot_pages(self.inputs)
        self.unit = global_scale(1.0, device)
        # fmha_sm100_plan stages block scales by 2^-3 unless told otherwise.
        self.reference, self.reference_lse = sparse_prefill_reference(self.inputs, block_scale_shift=3)
        self.exact, _ = sparse_prefill_reference(self.inputs, fp8_datapath=False)

    def plan(self, num_q_heads=None, qo_offset=None, **kwargs):
        case = self.case
        kv = torch.tensor(case.k_lens, dtype=torch.int32)
        qo = torch.tensor(case.q_lens, dtype=torch.int32)
        return fmha_sm100_plan(qo, kv, case.num_q_heads if num_q_heads is None else num_q_heads,
                               num_kv_heads=case.num_kv_heads, causal=True,
                               qo_offset=kv - qo if qo_offset is None else qo_offset,
                               page_size=PAGE_SIZE, kv_block_num=case.topk, **kwargs)

    def run(self, plan, *, q=None, out=None, kv_block_indexes=None, label="api", **kwargs):
        inputs = self.inputs
        kwargs = {"sm_scale": SM_SCALE, "k_scale": self.unit, "v_scale": self.unit, **kwargs}
        lists = inputs.kv_block_indexes if kv_block_indexes is None else kv_block_indexes

        def call():
            o, _ = fmha_sm100(inputs.q if q is None else q, self.k_packed, self.v_packed, plan,
                              kv_indices=inputs.kv_indices, kv_block_indexes=lists, out=out, **kwargs)
            return o

        return run_timed(label, call).clone()

    def wrapper(self, **plan_kwargs):
        k_data, k_sf, v_data, v_sf = nvfp4_head_slot_views(self.k_packed, self.v_packed)
        k_sf, v_sf = k_sf.view(torch.float8_e4m3fn), v_sf.view(torch.float8_e4m3fn)
        out, _ = run_wrapper(self.inputs, kv=(k_data, v_data), kv_sf=(k_sf, v_sf),
                             plan_kwargs={"flat": True, "block_scale_shift": 3, **plan_kwargs},
                             kv_global_scale=(self.unit, self.unit), label="wrapper")
        return out


@pytest.fixture(scope="module")
def harness(device):
    return ApiHarness(device)


def test_auto_backend_plans_and_runs_the_q8kv4_kernel(harness):
    plan = harness.plan()
    assert plan[3].get("MM-SA-Nv") and plan[3].get(PLAN_KEY) is not None
    for device in (torch.cuda.current_device(), f"cuda:{torch.cuda.current_device()}",
                   harness.inputs.q.device):  # every spelling of fmha_sm100_plan's device
        assert harness.plan(device=device)[3].get(PLAN_KEY) is not None
    out = harness.run(plan, label="api auto")
    assert torch.equal(out, harness.wrapper()), "the API must run the same kernel as the wrapper"
    assert_close_to_reference(out, harness.reference, label="api auto vs reference")


def test_nvfp4_cache_must_be_per_head_kv_slots(harness):
    """Sparse prefill reads only per-head K/V slot pages; the side-packed layout (every head's
    data, then every head's scales) is rejected instead of misread."""
    assert harness.case.num_kv_heads > 1
    k_side, v_side = pack_vllm_pages(harness.inputs)
    with pytest.raises(ValueError, match="per-head K/V slots"):
        fmha_sm100(harness.inputs.q, k_side, v_side, harness.plan(),
                   kv_indices=harness.inputs.kv_indices,
                   kv_block_indexes=harness.inputs.kv_block_indexes, sm_scale=SM_SCALE,
                   k_scale=harness.unit, v_scale=harness.unit)


def test_one_plan_serves_layers_with_different_topk_lists(harness):
    """vLLM plans once per step and runs every layer with its own TopK lists."""
    plan = harness.plan()
    inputs = harness.inputs
    first = inputs.kv_block_indexes
    # Layer 2 drops each list's first history page (the own page stays last).
    shifted = torch.cat([first[..., 1:], torch.full_like(first[..., :1], -1)], dim=-1)
    other = torch.where((first[..., 1:2] >= 0), shifted, first).contiguous()
    layer2 = harness.run(plan, kv_block_indexes=other, label="api layer 2")
    layer1 = harness.run(plan, label="api layer 1")
    assert torch.equal(layer1, harness.run(harness.plan(), label="api fresh plan"))
    swapped = dataclasses.replace(harness.inputs)
    swapped.topk_indices = other.permute(1, 0, 2).contiguous()
    reference, _ = sparse_prefill_reference(swapped, block_scale_shift=3)
    assert_close_to_reference(layer2, reference, label="layer 2 vs reference")
    assert not torch.equal(layer1, layer2) and torch.equal(first, inputs.kv_block_indexes)


def test_cute_dsl_backend_and_bf16_q_keep_the_cute_kernel(harness):
    plan = harness.plan(prefill_backend="cute_dsl")
    assert plan[3].get(PLAN_KEY) is None
    _within_fp8_noise(harness.run(plan, label="api cute_dsl"), harness.exact, label="cute_dsl e4m3 q")
    q_bf16 = harness.inputs.q.to(torch.bfloat16)
    cute_bf16 = harness.run(plan, q=q_bf16, label="api cute_dsl bf16 q")
    _loosely_close(cute_bf16, harness.exact, label="cute_dsl bf16 q vs exact")
    auto = harness.run(harness.plan(), q=q_bf16, label="api auto bf16 q")
    assert torch.equal(auto, cute_bf16), "BF16 Q must fall back to the CuTe-DSL kernel"


def test_unfit_batches_and_calls_fall_back_or_raise(harness):
    inputs = harness.inputs
    rows = inputs.q.shape[0]
    heads = harness.case.num_kv_heads
    q_gqa8 = inputs.q.reshape(rows, heads, Q_HEADS_PER_KV, HEAD_DIM)[:, :, :8]
    q_gqa8 = q_gqa8.reshape(rows, heads * 8, HEAD_DIM).contiguous()
    assert harness.plan(num_q_heads=heads * 8)[3].get(PLAN_KEY) is None
    with pytest.raises(ValueError, match="cannot plan this batch"):
        harness.plan(num_q_heads=heads * 8, prefill_backend="q8kv4")
    assert harness.plan(kv_dtype="fp8")[3].get(PLAN_KEY) is None
    exact_gqa8 = harness.exact.reshape(rows, heads, Q_HEADS_PER_KV, HEAD_DIM)[:, :, :8]
    _within_fp8_noise(harness.run(harness.plan(num_q_heads=heads * 8), q=q_gqa8, label="api gqa8"),
                      exact_gqa8.reshape(rows, heads * 8, HEAD_DIM), label="GQA 8 on the CuTe-DSL kernel")
    with pytest.raises(ValueError, match="cannot serve this call"):
        harness.run(harness.plan(prefill_backend="q8kv4"), q=inputs.q.to(torch.bfloat16),
                    label="api forced bf16 q")


def test_scales_and_offsets_are_served_by_the_q8kv4_kernel(harness):
    plan = harness.plan()
    baseline = harness.run(plan, label="api baseline")
    folded = harness.run(plan, sm_scale=SM_SCALE / 2, q_scale=2.0, label="api q_scale")
    assert torch.equal(folded, baseline), "q_scale folds into the softmax scale"
    scaled = harness.run(plan, o_scale=2.0, label="api o_scale")
    assert torch.allclose(scaled.float(), baseline.float() * 2.0, rtol=1e-2, atol=1e-2)
    default = torch.tensor([k - q for q, k in zip(CASE.q_lens, CASE.k_lens)], dtype=torch.int32,
                           device=harness.inputs.q.device)
    same = harness.run(plan, q_offset_override=default, label="api default offset")
    assert torch.equal(same, baseline)
    shrink = torch.tensor([0, 900, 130], dtype=torch.int32, device=default.device)
    tighter = harness.run(plan, q_offset_override=default - shrink, label="api tighter offset")
    seqused = [k - int(s) for k, s in zip(CASE.k_lens, shrink.tolist())]
    reference, _ = sparse_prefill_reference(harness.inputs, seqused_k=seqused, block_scale_shift=3)
    assert_close_to_reference(tighter, reference, label="q_offset_override vs reference")


def test_usable_sm_count_schedules_on_fewer_sms(harness):
    out = harness.run(harness.plan(usable_SM_count=24), label="api 24 SMs")
    assert_close_to_reference(out, harness.reference, label="usable_SM_count=24 vs reference")


def test_graph_replay_matches_eager(harness):
    plan = harness.plan()
    eager = harness.run(plan, label="api eager")
    out = torch.empty_like(eager)
    harness.run(plan, out=out, label="api warm")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fmha_sm100(harness.inputs.q, harness.k_packed, harness.v_packed, plan,
                   kv_indices=harness.inputs.kv_indices,
                   kv_block_indexes=harness.inputs.kv_block_indexes, out=out, sm_scale=SM_SCALE,
                   k_scale=harness.unit, v_scale=harness.unit)
    out.fill_(-7)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_mixed_decode_and_prefill_batch_routes_both_parts(device):
    """split_prefill_decode: the decode sub-plan runs Q8KV4 decode, the prefill sub-plan Q8KV4
    prefill, each against the same reference."""
    from fmha_sm100.q8kv4_decode_adapter import PLAN_KEY as DECODE_PLAN_KEY

    case = PrefillCase("b4_mixed", (4, 4, 300, 129), (1_000, 3_000, 2_300, 129), topk=16,
                       seed=41, ascending=True)
    harness = ApiHarness(device, case)
    plan = harness.plan(split_prefill_decode=True)
    has_mixed, split, _, decode, prefill = plan
    assert has_mixed and split == 2
    assert decode.get(DECODE_PLAN_KEY) is not None and prefill.get(PLAN_KEY) is not None
    out = harness.run(plan, label="api mixed")
    decode_rows = sum(case.q_lens[:split])
    _within_fp8_noise(out[:decode_rows], harness.exact[:decode_rows], label="decode part")
    assert_close_to_reference(out[decode_rows:], harness.reference[decode_rows:], label="prefill part")
