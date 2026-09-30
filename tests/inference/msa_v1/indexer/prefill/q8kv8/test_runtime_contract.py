"""Multi-head isolation, request ownership, and compile-cache reuse."""

import gc
import weakref

import pytest
import torch

from inference.msa_v1.indexer.prefill.q8kv8 import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)
from inference.msa_v1.indexer.prefill.q8kv8.interface import _COMPILE_CACHE
from tests.inference.msa_v1.indexer.prefill.q8kv8.cases import (
    make_real_prefill_inputs,
    require_sm100_device,
)
from tests.inference.msa_v1.indexer.prefill.q8kv8.reference import (
    assert_full_scores,
    assert_full_topk_quality,
)
from tests.inference.msa_v1.indexer.prefill.q8kv8.runtime import run_checked
from tests.inference.msa_v1.indexer.prefill.q8kv8.test_plan_reuse import _case

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("num_index_heads", (1, 2, 4))
def test_multihead_requests_reuse_code_without_sharing_workspace(num_index_heads):
    device = require_sm100_device()
    cases = (
        _case("first", (65, 129), (2048, 2176)),
        _case("second", (1, 127, 257), (2560, 2304, 2048)),
    )
    owners = []
    cache_keys = None
    for case in cases:
        inputs = make_real_prefill_inputs(
            case, device=device, num_index_heads=num_index_heads
        )
        # Share a historical prefix while retaining unordered physical pages.
        inputs.page_table[1, :3].copy_(inputs.page_table[0, :3])
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
        out = torch.empty_like(state.topk_indices)
        result = run_checked(wrapper, inputs.q, inputs.k_cache, out=out)
        assert result is out
        expected = assert_full_scores(case, inputs, state.proxy_scores)
        assert_full_topk_quality(expected, state.num_valid_pages, out)
        if cache_keys is not None:
            assert set(_COMPILE_CACHE) == cache_keys
        cache_keys = set(_COMPILE_CACHE)
        with pytest.raises(ValueError, match="shape"):
            wrapper.run(inputs.q, inputs.k_cache, out=out.view(-1, 16))
        changed_q = inputs.q.float()
        changed_q[:, 0] *= -1.5
        changed_q = changed_q.to(inputs.q.dtype)
        before = result.clone()
        after = run_checked(wrapper, changed_q, inputs.k_cache)
        assert torch.equal(before[1:], after[1:])
        owners.append((wrapper, inputs, after.clone(), changed_q))

    assert (
        owners[0][0]._proxy_score.plan_state.proxy_scores.data_ptr()
        != owners[1][0]._proxy_score.plan_state.proxy_scores.data_ptr()
    )
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream, (owner, payload, _, changed) in zip(streams, owners, strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            owner.run(changed, payload.k_cache)
    torch.cuda.synchronize()
    for owner, _, expected_output, _ in owners:
        assert torch.equal(owner._proxy_score.plan_state.topk_indices, expected_output)

    references = [
        weakref.ref(owner._proxy_score.plan_state.proxy_scores) for owner, *_ in owners
    ]
    del owner, wrapper, state, owners, inputs, result, after, payload
    gc.collect()
    assert all(reference() is None for reference in references)
