"""BF16 sparse decode with unordered pages, per-query causality, and PDL."""

import gc
import weakref
from functools import partial

import pytest
import torch

from inference.msa_v1.attention.decode.bf16 import BatchDecodeWithPagedKVCacheWrapper
from tests.inference.cases import active_inference_suite
from tests.inference.msa_v1.attention.decode.q8kv8.real_cases import (
    make_decode_attention_inputs,
)
from tests.inference.msa_v1.attention.decode.q8kv8.reference import (
    assert_decode_topk_contract,
    decode_attention_reference,
)
from tests.inference.msa_v1.attention.decode.q8kv8.runtime import compile_module
from tests.inference.msa_v1.attention.decode.runtime import run_cuda
from tests.inference.msa_v1.decode.cases import correctness_cases, make_seq_lens

pytestmark = pytest.mark.gpu
_CASES = correctness_cases(active_inference_suite())
_SHARDS = tuple(_CASES[start : start + 8] for start in range(0, len(_CASES), 8))


@pytest.mark.parametrize("query_length", range(1, 17))
@pytest.mark.parametrize("heads", (1, 2, 4))
@pytest.mark.parametrize("enable_pdl", (False, True))
@pytest.mark.parametrize("gqa_ratio", (8, 16), ids=("gqa8", "gqa16"))
def test_unordered_pages_and_graph(query_length, heads, enable_pdl, gqa_ratio):
    compile_module()
    inputs = make_decode_attention_inputs(
        torch.tensor([query_length, 129, 2053, 4097], dtype=torch.int32),
        seed=1701,
        device=torch.device("cuda"),
        q_len_per_req=query_length,
        page_layout="permuted",
        num_q_heads=heads * gqa_ratio,
        num_kv_heads=heads,
        dtype=torch.bfloat16,
    )
    assert_decode_topk_contract(inputs)
    # Assert that this test actually exercises unordered historical pages.
    history = inputs.topk_indices[-1, 0, :15]
    assert bool(torch.any(history[1:] < history[:-1]))
    _check_inputs(inputs, enable_pdl)


def _check_inputs(inputs, enable_pdl):
    assert_decode_topk_contract(inputs)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(enable_pdl=enable_pdl)
    wrapper.plan(
        inputs.topk_indices,
        inputs.page_table,
        inputs.seq_lens,
        q_len_per_req=inputs.q_len_per_req,
        num_q_heads=inputs.q.shape[1],
        num_kv_heads=inputs.k_cache.shape[1],
    )
    output = torch.empty_like(inputs.q)
    actual = run_cuda(
        "BF16 attention",
        lambda: wrapper.run(inputs.q, (inputs.k_cache, inputs.v_cache), out=output),
    )
    assert actual.data_ptr() == output.data_ptr()
    expected = decode_attention_reference(inputs)
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual.float(), expected.float(), atol=3e-2, rtol=3e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(inputs.q, (inputs.k_cache, inputs.v_cache), out=output)
    for _ in range(3):
        run_cuda("BF16 attention Graph replay", graph.replay)
        assert bool(torch.isfinite(output).all())
        torch.testing.assert_close(
            output.float(), expected.float(), atol=3e-2, rtol=3e-2
        )


@pytest.mark.parametrize("heads", (1, 2, 4))
@pytest.mark.parametrize("shard", _SHARDS, ids=lambda cases: cases[0].case_id)
def test_shared_cases(heads, shard):
    compile_module()
    for case in shard:
        inputs = make_decode_attention_inputs(
            make_seq_lens(case),
            seed=case.seed,
            device=torch.device("cuda"),
            q_len_per_req=case.q_len_per_req,
            page_layout=case.page_layout,
            num_q_heads=heads * 8,
            num_kv_heads=heads,
            dtype=torch.bfloat16,
        )
        _check_inputs(inputs, enable_pdl=True)


@pytest.mark.parametrize("enable_pdl", (False, True))
def test_metadata_update_and_stream_isolation(enable_pdl):
    compile_module()
    wrappers = [
        BatchDecodeWithPagedKVCacheWrapper(enable_pdl=enable_pdl) for _ in range(2)
    ]
    streams = [torch.cuda.Stream() for _ in wrappers]
    workspace_refs = []
    for step in range(2):
        inputs = [
            make_decode_attention_inputs(
                torch.tensor([4, 129 + step * 128, 2177], dtype=torch.int32),
                seed=1701 + step * 2 + index,
                device=torch.device("cuda"),
                q_len_per_req=4,
                page_layout="permuted",
                num_q_heads=16,
                num_kv_heads=2,
                dtype=torch.bfloat16,
            )
            for index in range(2)
        ]
        expected = [decode_attention_reference(item) for item in inputs]
        outputs = [torch.empty_like(item.q) for item in inputs]
        graphs = []
        for wrapper, stream, item, output in zip(wrappers, streams, inputs, outputs):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                wrapper.plan(
                    item.topk_indices,
                    item.page_table,
                    item.seq_lens,
                    q_len_per_req=4,
                    num_q_heads=16,
                    num_kv_heads=2,
                )
                run_cuda(
                    "BF16 lifecycle warmup",
                    partial(
                        wrapper.run, item.q, (item.k_cache, item.v_cache), out=output
                    ),
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    wrapper.run(item.q, (item.k_cache, item.v_cache), out=output)
                graphs.append(graph)
        assert (
            wrappers[0]._plan_state.workspace.data_ptr()
            != wrappers[1]._plan_state.workspace.data_ptr()
        )
        workspace_refs.extend(
            weakref.ref(item._plan_state.workspace) for item in wrappers
        )

        def replay_independent_streams(active_streams, active_graphs):
            for stream, graph in zip(active_streams, active_graphs):
                with torch.cuda.stream(stream):
                    graph.replay()

        for _ in range(3):
            run_cuda(
                "BF16 concurrent Graph replay",
                partial(replay_independent_streams, streams, graphs),
            )
            for output, reference in zip(outputs, expected):
                torch.testing.assert_close(
                    output.float(), reference.float(), atol=3e-2, rtol=3e-2
                )
        # Retire each Graph before replacing its wrapper's metadata on the next step.
        for graph in graphs:
            graph.reset()
        del graph, graphs
    del wrapper, wrappers
    gc.collect()
    assert all(reference() is None for reference in workspace_refs)
