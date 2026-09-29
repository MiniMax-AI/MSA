# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""fmha_sm100_plan / fmha_sm100 route NVFP4 sparse decode to the Q8KV4 kernel and keep kv_mode 3
for everything else."""

import pytest
import torch

from fmha_sm100.api import fmha_sm100, fmha_sm100_plan
from fmha_sm100.decode_q8kv4 import BatchDecodeWithPagedKVCacheWrapper, plan_decode, run_decode
from fmha_sm100.q8kv4_decode_adapter import PLAN_KEY

from .cases import HEAD_DIM, KV_HEADS, PAGE_SIZE, SM_SCALE, SMOKE_CASES, flat_page_table, global_scale, make_inputs, pack_vllm_pages, unpack_views
from .conftest import run_timed
from .reference import PageDequantizer, assert_close_to_reference, sparse_decode_reference

CASE = SMOKE_CASES[0]


def _ascending(topk: torch.Tensor) -> torch.Tensor:
    """kv_mode 3 needs ascending lists; the Q8KV4 kernel accepts them too."""
    big = torch.iinfo(torch.int32).max
    lists = topk.clone()
    lists[lists < 0] = big
    lists, _ = lists.sort(dim=-1)
    lists[lists == big] = -1
    return lists.contiguous()


class ApiHarness:
    def __init__(self, device):
        self.inputs = make_inputs(CASE, device)
        self.inputs.topk_indices = _ascending(self.inputs.topk_indices)
        self.k_packed, self.v_packed = pack_vllm_pages(self.inputs)
        self.kv_indices, self.kv_indptr = flat_page_table(self.inputs.page_table, CASE.seq_lens)
        self.unit = global_scale(1.0, device)
        self.reference = sparse_decode_reference(
            self.inputs, PageDequantizer(self.inputs.k_codes, self.inputs.k_scale), PageDequantizer(self.inputs.v_codes, self.inputs.v_scale)
        )

    def plan(self, num_q_heads=None, **kwargs):
        inputs = self.inputs
        kv = inputs.seq_lens.cpu()
        qo = torch.full_like(kv, CASE.q_len)
        return fmha_sm100_plan(qo, kv, inputs.num_q_heads if num_q_heads is None else num_q_heads,
                               num_kv_heads=KV_HEADS, causal=True, qo_offset=kv - qo, page_size=PAGE_SIZE,
                               kv_block_num=CASE.topk, **kwargs)

    def run(self, plan, *, q=None, topk=None, out=None, label="api", **kwargs):
        inputs = self.inputs
        def call():
            o, _ = fmha_sm100(inputs.q if q is None else q, self.k_packed, self.v_packed, plan,
                              kv_indices=self.kv_indices, kv_block_indexes=inputs.topk_indices if topk is None else topk,
                              out=out, sm_scale=SM_SCALE, k_scale=self.unit, v_scale=self.unit, **kwargs)
            return o
        return run_timed(label, call).clone()

    def direct(self, shift, topk=None):
        inputs = self.inputs
        plan = plan_decode(batch_size=CASE.batch_size, q_len_per_req=CASE.q_len, topk=CASE.topk, device=inputs.q.device,
                           num_q_heads=inputs.num_q_heads, num_kv_heads=KV_HEADS, block_scale_shift=shift)
        k_data, k_sf = unpack_views(self.k_packed)
        v_data, v_sf = unpack_views(self.v_packed)
        return run_timed(f"run_decode shift{shift}", lambda: run_decode(
            plan, inputs.q, (k_data, v_data), kv_cache_sf=(k_sf, v_sf), seq_lens=inputs.seq_lens,
            kv_indices=self.kv_indices, kv_indptr=self.kv_indptr,
            topk_indices=inputs.topk_indices if topk is None else topk, sm_scale=SM_SCALE,
            kv_global_scale=(self.unit, self.unit))).clone()


@pytest.fixture(scope="module")
def harness(device):
    return ApiHarness(device)


def test_auto_backend_plans_and_runs_the_q8kv4_kernel(harness):
    plan = harness.plan()
    assert plan[3].get(PLAN_KEY) is not None
    out = harness.run(plan, label="api auto")
    assert torch.equal(out, harness.direct(3)), "the API must run the same kernel as run_decode"
    assert_close_to_reference(out, harness.reference, label="api auto vs reference")


def test_kv_mode3_still_serves_the_same_inputs(harness):
    plan = harness.plan(decode_backend="kv_mode3")
    assert plan[3].get(PLAN_KEY) is None
    assert_close_to_reference(harness.run(plan, label="api kv_mode3"), harness.reference, label="kv_mode3 vs reference")


def test_one_plan_serves_layers_with_different_topk_lists(harness):
    plan = harness.plan()
    other = harness.inputs.topk_indices.flip(0).contiguous()  # a different valid list per row
    assert torch.equal(harness.run(plan, topk=other, label="api layer 2"), harness.direct(3, topk=other))
    assert torch.equal(harness.run(plan, label="api layer 1"), harness.direct(3))


def test_graph_replay_matches_eager(harness):
    plan = harness.plan()
    eager = harness.run(plan, label="api eager")
    out = torch.empty_like(eager)
    harness.run(plan, out=out, label="api warm")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fmha_sm100(harness.inputs.q, harness.k_packed, harness.v_packed, plan, kv_indices=harness.kv_indices,
                   kv_block_indexes=harness.inputs.topk_indices, out=out, sm_scale=SM_SCALE,
                   k_scale=harness.unit, v_scale=harness.unit)
    out.fill_(-7)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_unfit_batches_and_calls_fall_back_or_raise(harness):
    inputs = harness.inputs
    rows = inputs.q.shape[0]
    q_gqa4 = inputs.q.reshape(rows, KV_HEADS, inputs.gqa, HEAD_DIM)[:, :, :4].reshape(rows, KV_HEADS * 4, HEAD_DIM).contiguous()
    plan_gqa4 = harness.plan(num_q_heads=KV_HEADS * 4)
    assert plan_gqa4[3].get(PLAN_KEY) is None
    reference_gqa4 = sparse_decode_reference(
        harness.inputs.__class__(**{**vars(inputs), "q": q_gqa4, "gqa": 4}),
        PageDequantizer(inputs.k_codes, inputs.k_scale), PageDequantizer(inputs.v_codes, inputs.v_scale))
    assert_close_to_reference(harness.run(plan_gqa4, q=q_gqa4, label="api gqa4"), reference_gqa4, label="GQA 4 on kv_mode3")
    with pytest.raises(ValueError, match="cannot plan this batch"):
        harness.plan(num_q_heads=KV_HEADS * 4, decode_backend="q8kv4")
    assert harness.plan(kv_dtype="fp8")[3].get(PLAN_KEY) is None
    assert harness.plan(output_maxscore=True)[3].get(PLAN_KEY) is None
    plan = harness.plan()
    kv_mode3 = harness.run(harness.plan(decode_backend="kv_mode3"), label="api kv_mode3 for o_scale")
    scaled = harness.run(plan, o_scale=2.0, label="api o_scale=2")
    assert torch.allclose(scaled.float(), kv_mode3.float() * 2.0, rtol=1e-2, atol=1e-2), "o_scale must land on kv_mode 3"
    with pytest.raises(ValueError, match="cannot serve this call"):
        harness.run(harness.plan(decode_backend="q8kv4"), o_scale=2.0, label="api forced o_scale")


def test_shift_zero_through_the_api_matches_the_wrapper(harness):
    inputs = harness.inputs
    plan = harness.plan(block_scale_shift=0)
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    wrapper.plan(inputs.topk_indices, harness.kv_indices, inputs.seq_lens, q_len_per_req=CASE.q_len,
                 num_q_heads=inputs.num_q_heads, num_kv_heads=KV_HEADS, kv_indptr=harness.kv_indptr)
    k_data, k_sf = unpack_views(harness.k_packed)
    v_data, v_sf = unpack_views(harness.v_packed)
    expected = run_timed("wrapper shift0", lambda: wrapper.run(inputs.q, (k_data, v_data), kv_cache_sf=(k_sf, v_sf)).clone())
    assert torch.equal(harness.run(plan, label="api shift0"), expected)
