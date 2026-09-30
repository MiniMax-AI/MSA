"""Proxy-score and full-indexer checks on the selected real prefill tier."""

from __future__ import annotations

import pytest
import torch

from inference.msa_v1.indexer.prefill.q8kv8 import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)
from tests.inference.cases import (
    active_inference_suite,
    selected_msa_v1_prefill_test_cases,
)
from tests.inference.msa_v1.indexer.prefill.q8kv8.cases import (
    make_real_prefill_inputs,
    require_sm100_device,
)
from tests.inference.msa_v1.indexer.prefill.q8kv8.reference import (
    assert_device_plan,
    assert_full_scores,
    assert_full_topk_quality,
    assert_score_structure,
    assert_topk_structure,
)
from tests.inference.msa_v1.indexer.prefill.q8kv8.runtime import run_checked

pytestmark = pytest.mark.gpu
_CASES = selected_msa_v1_prefill_test_cases()
_SHARD_COUNT = min(64, len(_CASES))
_SUITE = active_inference_suite()


@pytest.mark.parametrize("shard_index", range(_SHARD_COUNT))
@pytest.mark.parametrize("num_index_heads", (1, 2, 4))
def test_real_varlen_gemm_and_topk_cases(
    shard_index: int, num_index_heads: int
) -> None:
    """Check all selected cases in bounded shards so one test stays under 30s."""

    device = require_sm100_device()
    shard = _CASES[shard_index::_SHARD_COUNT]
    for case in shard:
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
        assert_device_plan(case, state)
        state.proxy_scores.fill_(float("nan"))
        state.topk_indices.fill_(-777)
        result = run_checked(wrapper, inputs.q, inputs.k_cache)

        assert result.data_ptr() == state.topk_indices.data_ptr()
        assert_score_structure(
            case,
            state.proxy_scores,
            state.num_valid_pages,
        )
        expected_scores = assert_full_scores(case, inputs, state.proxy_scores)
        assert_topk_structure(state.topk_indices, state.num_valid_pages)
        assert_full_topk_quality(
            expected_scores,
            state.num_valid_pages,
            state.topk_indices,
        )
        if _SUITE == "full":
            baseline = result.clone()
            for _ in range(2):
                repeated = run_checked(wrapper, inputs.q, inputs.k_cache)
                torch.cuda.synchronize()
                assert torch.equal(baseline, repeated), case.case_id
