"""Multi-head isolation, request ownership, and compile-cache reuse."""

import gc
import weakref
from dataclasses import replace

import pytest
import torch

from inference.msa_v1 import aot_cache
from inference.msa_v1.indexer.prefill.q8kv8 import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)
from inference.msa_v1.indexer.prefill.q8kv8.interface import _COMPILE_CACHE
from tests.inference.msa_v1.attention.decode.runtime import run_cuda
from tests.inference.msa_v1.cache_layout import CACHE_LAYOUTS, cache_view
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


@pytest.mark.parametrize("num_index_heads", (1, 4))
@pytest.mark.parametrize("final_length", (1048576, 1048584, 2097152))
def test_long_context(num_index_heads, final_length):
    case = _case(
        "long_context",
        (8, 3, 17),
        (final_length - 8, 126, final_length // 2 - 17),
    )
    inputs = make_real_prefill_inputs(
        case, device=require_sm100_device(), num_index_heads=num_index_heads
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
    baseline = None
    for _ in range(3):
        state.proxy_scores.fill_(float("nan"))
        output = run_checked(wrapper, inputs.q, inputs.k_cache)
        if baseline is None:
            expected = assert_full_scores(case, inputs, state.proxy_scores)
            assert_full_topk_quality(expected, state.num_valid_pages, output)
            baseline = output.clone()
        else:
            assert torch.equal(baseline, output)


@pytest.mark.parametrize("num_index_heads", (1, 2, 4))
@pytest.mark.parametrize(
    "layout", (*CACHE_LAYOUTS, "singleton_zero", "singleton_unaligned")
)
def test_multihead_requests_reuse_code_without_sharing_workspace(
    num_index_heads, layout, monkeypatch
):
    device = require_sm100_device()
    cases = (
        _case("first", (65, 129), (2048, 2176)),
        _case("second", (1, 127, 257), (2560, 2304, 2048)),
    )
    owners = []
    cache_keys = None
    singleton_first = layout.startswith("singleton")
    if singleton_first:
        monkeypatch.setattr(aot_cache, "_AOT_DISABLE", True)
        _COMPILE_CACHE.clear()
    for case_index, case in enumerate(cases):
        inputs = make_real_prefill_inputs(
            case, device=device, num_index_heads=num_index_heads
        )
        # Share a historical prefix while retaining unordered physical pages.
        contiguous_cache = inputs.k_cache
        if singleton_first and case_index == 0:
            inputs = replace(inputs, k_cache=cache_view(contiguous_cache[:1], layout))
            inputs.page_table.zero_()
        else:
            current_layout = "combined" if singleton_first else layout
            inputs = replace(
                inputs, k_cache=cache_view(contiguous_cache, current_layout)
            )
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
        initial_cache = inputs.k_cache if singleton_first else contiguous_cache
        run_checked(wrapper, inputs.q, initial_cache, out=out)
        contiguous_keys = set(_COMPILE_CACHE)
        state.proxy_scores.fill_(float("nan"))
        result = run_checked(wrapper, inputs.q, inputs.k_cache, out=out)
        assert set(_COMPILE_CACHE) == contiguous_keys
        assert result is out
        expected = assert_full_scores(case, inputs, state.proxy_scores)
        assert_full_topk_quality(expected, state.num_valid_pages, out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper.run(inputs.q, inputs.k_cache, out=out)
        state.proxy_scores.fill_(float("nan"))
        out.fill_(-123)
        run_cuda("prefill indexer graph", graph.replay)
        expected = assert_full_scores(case, inputs, state.proxy_scores)
        assert_full_topk_quality(expected, state.num_valid_pages, out)
        graph.reset()
        del graph
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
