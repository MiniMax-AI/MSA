"""CUDA Graph capture and deterministic replay on fixed real shapes."""

from __future__ import annotations

import pytest
import torch

from datas.inference.cases import load_prefill_test_cases
from inference.msa_v1.indexer.prefill.q8kv8 import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)
from tests.inference.msa_v1.indexer.prefill.q8kv8.cases import (
    make_real_prefill_inputs,
    require_sm100_device,
)
from tests.inference.msa_v1.indexer.prefill.q8kv8.reference import (
    assert_device_plan,
    assert_full_scores,
    assert_score_structure,
    assert_topk_structure,
)

pytestmark = pytest.mark.gpu
_GRAPH_CASES = load_prefill_test_cases("cuda_graph")


@pytest.mark.parametrize("case", _GRAPH_CASES, ids=lambda case: case.case_id)
@pytest.mark.parametrize("num_index_heads", (1, 2, 4))
def test_real_case_cuda_graph_replay(case, num_index_heads) -> None:
    device = require_sm100_device()

    inputs = make_real_prefill_inputs(
        case, device=device, num_index_heads=num_index_heads
    )
    wrapper = BatchPrefillIndexerWithPagedKVCacheWrapper()
    wrapper.plan(
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        inputs.page_table,
        total_q=case.total_q,
        max_seqlen_q=case.max_query_len,
        max_seqlen_k=case.max_final_kv,
        num_index_heads=num_index_heads,
    )
    state = wrapper._proxy_score.plan_state
    state.proxy_scores.fill_(float("nan"))
    wrapper.run(inputs.q, inputs.k_cache)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.replan()
        result = wrapper.run(inputs.q, inputs.k_cache)
    torch.cuda.synchronize()
    baseline_scores = state.proxy_scores.clone()
    baseline_lengths = state.num_valid_pages.clone()
    baseline_topk = result.clone()
    state.proxy_scores.fill_(float("nan"))
    state.num_valid_pages.fill_(-777)
    result.fill_(-777)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        state.proxy_scores,
        baseline_scores,
        atol=0,
        rtol=0,
        equal_nan=True,
    )
    torch.testing.assert_close(
        state.num_valid_pages,
        baseline_lengths,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(result, baseline_topk, atol=0, rtol=0)
    assert_device_plan(case, state)
    assert_score_structure(case, state.proxy_scores, state.num_valid_pages)
    assert_full_scores(case, inputs, state.proxy_scores)
    assert_topk_structure(result, state.num_valid_pages)
