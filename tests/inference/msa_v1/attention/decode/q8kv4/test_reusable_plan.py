"""Regression for split workspaces reused after request replacement or padding."""

from __future__ import annotations

import pytest
import torch

from inference.msa_v1.attention.decode.q8kv4 import (
    BatchDecodeWithPagedKVCacheWrapper,
    jit,
)
from tests.inference.msa_v1.attention.decode.q8kv4.real_cases import (
    make_decode_attention_inputs,
)
from tests.inference.msa_v1.attention.decode.q8kv4.reference import (
    decode_attention_reference,
)
from tests.inference.msa_v1.attention.decode.q8kv4.runtime import compile_modules
from tests.inference.msa_v1.attention.decode.runtime import run_cuda
from tests.inference.msa_v1.decode.metadata import make_decode_topk

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    "num_kv_heads,query_length,splits",
    [(1, 1, 1), (1, 8, 2), (2, 8, 4), (4, 8, 4), (1, 16, 8)],
)
@pytest.mark.parametrize("graphed", (False, True), ids=("eager", "graph"))
@pytest.mark.parametrize("gqa_ratio", (8, 16), ids=("gqa8", "gqa16"))
def test_request_replacement_reuses_plan(
    num_kv_heads, query_length, splits, graphed, gqa_ratio
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in {
        (10, 0),
        (10, 3),
        (10, 7),
    }:
        pytest.skip("Q8KV4 regression requires SM100, SM103, or SM107")
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = make_decode_attention_inputs(
        torch.tensor([4097, 4097, 4097], dtype=torch.int32),
        seed=1701,
        device=torch.device("cuda"),
        q_len_per_req=query_length,
        num_kv_heads=num_kv_heads,
        num_q_heads=num_kv_heads * gqa_ratio,
    )
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    wrapper.plan(
        inputs.topk_indices,
        inputs.page_table,
        inputs.seq_lens,
        q_len_per_req=query_length,
        num_q_heads=num_kv_heads * gqa_ratio,
        num_kv_heads=num_kv_heads,
        num_kv_splits=splits,
    )
    output = torch.empty_like(inputs.q, dtype=torch.bfloat16)

    def run():
        return wrapper.run(
            inputs.q,
            (inputs.packed_k, inputs.packed_v),
            kv_cache_sf=(inputs.k_scale, inputs.v_scale),
            out=output,
        )

    compile_modules(gqa_ratio, inputs.q.device)
    run_cuda("metadata warmup", run)
    if graphed:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        invoke = graph.replay
    else:
        invoke = run

    # Capacity is constant; these transitions cross page and split boundaries.
    for step, lengths in enumerate(
        (
            [129, 513, 1024],
            [0, 0, 0],
            [1, 128, 257],
            [2048, 2049, 4096],
        )
    ):
        inputs.seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        topk = make_decode_topk(
            inputs.seq_lens.clamp_min(query_length),
            q_len_per_req=query_length,
            seed=1702 + step,
        )[:, :num_kv_heads].contiguous()
        position = inputs.seq_lens.repeat_interleave(query_length) - query_length
        position = position + torch.arange(query_length, device="cuda").repeat(3)
        topk.masked_fill_(position[:, None, None] < 0, -1)
        inputs.topk_indices.copy_(topk)
        inputs.page_table.copy_(inputs.page_table.roll(1, dims=1))
        inputs.packed_v.bitwise_xor_(17)
        inputs.k_scale.copy_((inputs.k_scale.float() * 0.875).to(torch.float8_e4m3fn))
        output.fill_(-777)
        run_cuda(f"metadata step={step} splits={splits} graph={graphed}", invoke)
        expected = decode_attention_reference(inputs)
        assert torch.isfinite(output).all(), (step, splits, graphed)
        torch.testing.assert_close(
            output.float(), expected.float(), atol=0.05, rtol=0.05
        )
        torch.testing.assert_close(
            output[position < 0], expected[position < 0], atol=0, rtol=0
        )
        previous = output.clone()
        output.fill_(-777)
        run_cuda("metadata repeated execution", invoke)
        torch.testing.assert_close(output, previous, atol=0, rtol=0)


@pytest.mark.parametrize("splits", (None, 4), ids=("automatic", "fixed"))
@pytest.mark.parametrize("gqa_ratio", (8, 16), ids=("gqa8", "gqa16"))
@pytest.mark.parametrize("batch", (1, 20), ids=("underfilled", "multiple_waves"))
def test_independent_plans_release_workspace_and_run_concurrently(
    splits, gqa_ratio, batch, monkeypatch
):
    """Exercise both underfilled and full-wave grids using separately captured requests."""
    import gc

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in {
        (10, 0),
        (10, 3),
        (10, 7),
    }:
        pytest.skip("Q8KV4 regression requires SM100, SM103, or SM107")
    device = torch.device("cuda")
    inputs = [
        make_decode_attention_inputs(
            torch.full((batch,), 4097, dtype=torch.int32),
            seed=1701 + index,
            device=device,
            q_len_per_req=8,
            num_q_heads=gqa_ratio * 4,
        )
        for index in range(2)
    ]
    expected = [decode_attention_reference(item) for item in inputs]
    outputs = [torch.empty_like(item.q, dtype=torch.bfloat16) for item in inputs]
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    compile_modules(gqa_ratio, device)

    def exercise():
        wrappers = [BatchDecodeWithPagedKVCacheWrapper() for _ in inputs]
        streams = [torch.cuda.Stream() for _ in inputs]
        graphs = [torch.cuda.CUDAGraph() for _ in inputs]
        for index, (wrapper, item) in enumerate(zip(wrappers, inputs)):
            wrapper.plan(
                item.topk_indices,
                item.page_table,
                item.seq_lens,
                q_len_per_req=8,
                num_kv_splits=splits,
                usable_sm_count=148,
                num_q_heads=gqa_ratio * 4,
            )
            run_cuda(
                "concurrent plan warmup",
                lambda wrapper=wrapper, item=item, output=outputs[index]: wrapper.run(
                    item.q,
                    (item.packed_k, item.packed_v),
                    kv_cache_sf=(item.k_scale, item.v_scale),
                    out=output,
                ),
            )
        torch.cuda.synchronize()
        for index, (wrapper, item) in enumerate(zip(wrappers, inputs)):
            with torch.cuda.graph(graphs[index], stream=streams[index]):
                wrapper.run(
                    item.q,
                    (item.packed_k, item.packed_v),
                    kv_cache_sf=(item.k_scale, item.v_scale),
                    out=outputs[index],
                )

        def replay():
            for stream, graph in zip(streams, graphs):
                with torch.cuda.stream(stream):
                    graph.replay()

        previous = None
        for _ in range(5):
            for output in outputs:
                output.fill_(-777)
            run_cuda("concurrent graphs", replay)
            for actual, reference in zip(outputs, expected):
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(
                    actual.float(), reference.float(), atol=0.05, rtol=0.05
                )
            if previous is not None:
                assert all(torch.equal(a, b) for a, b in zip(outputs, previous))
            previous = [output.clone() for output in outputs]

    exercise()
    gc.collect()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated_before

    # Both algorithms are static specializations; request sizes and values are not.
    compile_modules(gqa_ratio, device)

    def unexpected_build(*args, **kwargs):
        pytest.fail("Changing runtime shapes or metadata triggered compilation")

    monkeypatch.setattr(jit, "_run_ninja", unexpected_build)

    def run_shape(batch, query_length):
        lengths = torch.arange(batch, dtype=torch.int32) * 37 + query_length
        item = make_decode_attention_inputs(
            lengths,
            seed=1701 + batch,
            device=device,
            q_len_per_req=query_length,
            page_layout="permuted",
            num_q_heads=gqa_ratio * 4,
        )
        wrapper = BatchDecodeWithPagedKVCacheWrapper()
        wrapper.plan(
            item.topk_indices,
            item.page_table,
            item.seq_lens,
            q_len_per_req=query_length,
            num_kv_splits=splits,
            num_q_heads=gqa_ratio * 4,
        )
        for step in range(2):
            if step:
                item.seq_lens.sub_(query_length - 1)
                topk = make_decode_topk(
                    item.seq_lens.clamp_min(query_length),
                    q_len_per_req=query_length,
                    seed=1703 + batch,
                )
                position = item.seq_lens.repeat_interleave(query_length) - query_length
                position += torch.arange(query_length, device=device).repeat(batch)
                topk.masked_fill_(position[:, None, None] < 0, -1)
                item.topk_indices.copy_(topk)
            actual = run_cuda(
                "reused compilation",
                lambda: wrapper.run(
                    item.q,
                    (item.packed_k, item.packed_v),
                    kv_cache_sf=(item.k_scale, item.v_scale),
                ),
            )
            expected = decode_attention_reference(item)
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(
                actual.float(), expected.float(), atol=0.05, rtol=0.05
            )

    for runtime_batch, query_length in ((1, 1), (5, 3), (32, 8), (128, 16)):
        run_shape(runtime_batch, query_length)
        gc.collect()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated_before
