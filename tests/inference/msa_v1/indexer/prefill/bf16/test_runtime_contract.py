"""BF16 prefill metadata, graph, ownership, and specialization contracts."""

import gc
import weakref
from functools import partial

import pytest
import torch

from inference.msa_v1.indexer._common.topk_select.build import load_extension
from inference.msa_v1.indexer.prefill.bf16 import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)
from inference.msa_v1.indexer.prefill.bf16.interface import (
    _GEMM_COMPILE_CACHE,
    _SCHEDULE_COMPILE_CACHE,
    _compile_gemm,
)
from tests.inference.msa_v1.attention.decode.runtime import run_cuda
from tests.inference.msa_v1.indexer.prefill.bf16.real_cases import (
    make_real_prefill_inputs,
)
from tests.inference.msa_v1.indexer.prefill.q8kv8.cases import RealPrefillInputs
from tests.inference.msa_v1.indexer.prefill.q8kv8.reference import (
    assert_full_topk_quality,
    assert_topk_structure,
    expected_lengths,
    full_score_reference,
)
from tests.inference.msa_v1.indexer.prefill.q8kv8.test_plan_reuse import _case

pytestmark = pytest.mark.gpu


def _check(case, inputs, scores, out):
    lengths = expected_lengths(case).to(inputs.q.device)
    scores = scores[..., : case.max_cols]
    mask = torch.arange(scores.shape[-1], device=scores.device)[None, :] < (
        lengths[:, None] - 1
    )
    for head in range(inputs.q.shape[1]):
        reference_inputs = RealPrefillInputs(
            q=inputs.q[:, head : head + 1],
            k_cache=inputs.k_cache,
            page_table=inputs.page_table,
            cu_seqlens_q=inputs.cu_seqlens_q,
            cu_seqlens_k=inputs.cu_seqlens_k,
        )
        expected = full_score_reference(case, reference_inputs)
        assert torch.isfinite(scores[head][mask]).all()
        torch.testing.assert_close(
            scores[head][mask], expected[mask] / 128**0.5, atol=2e-4, rtol=2e-4
        )
        assert_topk_structure(out[head], lengths)
        assert_full_topk_quality(expected, lengths, out[head])


@pytest.mark.parametrize("heads", (1, 2, 4))
def test_graph_replan_and_independent_owners(heads):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    cases = (
        _case("ownership_first", (65, 129), (2048, 2176)),
        _case("ownership_second", (1, 127, 257), (2560, 2304, 2048)),
    )
    owners = []
    cache_keys = None
    load_extension()
    for case in cases:
        inputs = make_real_prefill_inputs(
            case, num_index_heads=heads, device=torch.device("cuda")
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
            num_index_heads=heads,
        )
        state = wrapper._plan_state
        _compile_gemm(
            inputs.q.view(-1, 128), inputs.k_cache[:, 0].permute(1, 2, 0), state
        )
        keys = (set(_GEMM_COMPILE_CACHE), set(_SCHEDULE_COMPILE_CACHE))
        if cache_keys is not None:
            assert keys == cache_keys
        cache_keys = keys
        out = torch.empty_like(state.topk_indices)
        assert (
            run_cuda(
                "BF16 prefill", partial(wrapper.run, inputs.q, inputs.k_cache, out=out)
            )
            is out
        )
        _check(case, inputs, state.proxy_scores, out)
        before = out.clone()
        inputs.q[:, 0].copy_((-1.5 * inputs.q[:, 0].float()).bfloat16())
        run_cuda(
            "head isolation", partial(wrapper.run, inputs.q, inputs.k_cache, out=out)
        )
        assert torch.equal(before[1:], out[1:])
        _check(case, inputs, state.proxy_scores, out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper.run(inputs.q, inputs.k_cache, out=out)
        owners.append((wrapper, inputs, out, graph))
    assert (
        owners[0][0]._plan_state.proxy_scores.data_ptr()
        != owners[1][0]._plan_state.proxy_scores.data_ptr()
    )
    streams = [torch.cuda.Stream() for _ in owners]
    for generation in range(2):
        for case, (wrapper, inputs, out, graph) in zip(cases, owners, strict=True):
            if generation:
                # Change historical mapping in place; graph bindings remain valid.
                for request, length in enumerate(case.final_kv_lens):
                    pages = (length + 127) // 128
                    row = inputs.page_table[request, :pages]
                    row.copy_(row.roll(1))
                inputs.cu_seqlens_k[1:].sub_(
                    torch.arange(
                        1, len(case.final_kv_lens) + 1, device="cuda", dtype=torch.int32
                    )
                )
                wrapper.replan()
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        for _ in range(3):
            for stream, (wrapper, inputs, out, graph) in zip(
                streams, owners, strict=True
            ):
                with torch.cuda.stream(stream):
                    graph.replay()
            run_cuda("concurrent graph completion", lambda: None)
        for case, (wrapper, inputs, out, graph) in zip(cases, owners, strict=True):
            current = (
                case
                if generation == 0
                else _case(
                    case.case_id,
                    case.query_lens,
                    tuple(v - 1 for v in case.prefix_lens),
                )
            )
            _check(current, inputs, wrapper._plan_state.proxy_scores, out)
    references = [weakref.ref(owner[0]._plan_state.proxy_scores) for owner in owners]
    for wrapper, inputs, out, graph in owners:
        graph.reset()
    del wrapper, state, graph, owners
    gc.collect()
    assert all(reference() is None for reference in references)
