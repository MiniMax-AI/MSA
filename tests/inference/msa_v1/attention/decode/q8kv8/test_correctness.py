"""Correctness and CUDA Graph checks for the external FlashInfer Q8K8 adapter."""

from __future__ import annotations

import gc
import weakref
from dataclasses import replace

import pytest
import torch

from inference.msa_v1.attention.decode import bf16
from inference.msa_v1.attention.decode.q8kv8 import (
    BatchDecodeWithPagedKVCacheWrapper,
)
from tests.inference.msa_v1.attention.decode.q8kv8.real_cases import (
    make_decode_attention_inputs,
)
from tests.inference.msa_v1.attention.decode.q8kv8.reference import (
    decode_attention_reference,
)
from tests.inference.msa_v1.attention.decode.q8kv8.runtime import (
    compile_module,
    run_decode,
)
from tests.inference.msa_v1.attention.decode.runtime import run_cuda
from tests.inference.msa_v1.cache_layout import CACHE_LAYOUTS, cache_view

pytestmark = pytest.mark.gpu


def _require_backend() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
        pytest.skip("FlashInfer Q8K8 block-sparse decode requires SM100 or SM103")
    compile_module()
    return device


def _make_inputs(
    device: torch.device,
    *,
    gqa_ratio=16,
    num_kv_heads=4,
    lengths=(385, 257),
    q_len=4,
    seed=1701,
):
    return make_decode_attention_inputs(
        torch.tensor(lengths, dtype=torch.int32),
        seed=seed,
        device=device,
        q_len_per_req=q_len,
        page_layout="permuted",
        num_q_heads=gqa_ratio * num_kv_heads,
        num_kv_heads=num_kv_heads,
    )


def _assert_close(
    actual: torch.Tensor, expected: torch.Tensor, *, tolerance=0.05
) -> None:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    assert bool(torch.isfinite(actual_fp32).all())
    assert bool(torch.isfinite(expected_fp32).all())
    cosine = torch.nn.functional.cosine_similarity(
        actual_fp32.flatten(), expected_fp32.flatten(), dim=0
    )
    assert float(cosine) >= 0.999
    torch.testing.assert_close(
        actual_fp32, expected_fp32, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("gqa_ratio", (8, 16), ids=("gqa8", "gqa16"))
@pytest.mark.parametrize("num_kv_heads", (1, 2, 4))
@pytest.mark.parametrize("layout", CACHE_LAYOUTS)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float8_e4m3fn))
def test_full_output_matches_independent_reference(
    gqa_ratio, num_kv_heads, layout, dtype
) -> None:
    device = _require_backend()
    inputs = _make_inputs(device, gqa_ratio=gqa_ratio, num_kv_heads=num_kv_heads)
    inputs = replace(
        inputs,
        q=inputs.q.to(dtype),
        k_cache=inputs.k_cache.to(dtype),
        v_cache=inputs.v_cache.to(dtype),
    )
    wrapper = (
        bf16.BatchDecodeWithPagedKVCacheWrapper()
        if dtype == torch.bfloat16
        else BatchDecodeWithPagedKVCacheWrapper()
    )
    wrapper.plan(
        inputs.topk_indices,
        inputs.page_table,
        inputs.seq_lens,
        q_len_per_req=inputs.q_len_per_req,
        num_q_heads=gqa_ratio * num_kv_heads,
        num_kv_heads=num_kv_heads,
    )
    from flashinfer.decode import get_trtllm_gen_fmha_module

    run_decode(wrapper, inputs)
    misses = get_trtllm_gen_fmha_module.cache_info().misses
    with pytest.raises(ValueError, match="shapes and strides must match"):
        wrapper.run(inputs.q, (cache_view(inputs.k_cache, "token"), inputs.v_cache))
    inputs = replace(
        inputs,
        k_cache=cache_view(inputs.k_cache, layout),
        v_cache=cache_view(inputs.v_cache, layout),
    )
    actual = run_decode(wrapper, inputs)
    assert get_trtllm_gen_fmha_module.cache_info().misses == misses
    tolerance = 0.03 if dtype == torch.bfloat16 else 0.05
    _assert_close(actual, decode_attention_reference(inputs), tolerance=tolerance)
    out = torch.empty_like(actual)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(inputs.q, (inputs.k_cache, inputs.v_cache), out=out)
    out.fill_(float("nan"))
    run_cuda("cache layout graph replay", graph.replay)
    _assert_close(out, decode_attention_reference(inputs), tolerance=tolerance)
    inputs.q.zero_()
    _assert_close(
        run_decode(wrapper, inputs),
        decode_attention_reference(inputs),
        tolerance=tolerance,
    )
    if dtype == torch.bfloat16:
        inputs.k_cache.fill_(448.0)
        inputs.v_cache.fill_(448.0)
    else:
        inputs.k_cache.view(torch.uint8).fill_(0x7E)
        inputs.v_cache.view(torch.uint8).fill_(0x7E)
    _assert_close(
        run_decode(wrapper, inputs),
        decode_attention_reference(inputs),
        tolerance=tolerance,
    )


@pytest.mark.parametrize("gqa_ratio", (8, 16), ids=("gqa8", "gqa16"))
def test_cuda_graph_replay_is_bitwise_deterministic(gqa_ratio) -> None:
    device = _require_backend()
    inputs = _make_inputs(device, gqa_ratio=gqa_ratio)
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    wrapper.plan(
        inputs.topk_indices,
        inputs.page_table,
        inputs.seq_lens,
        q_len_per_req=inputs.q_len_per_req,
        num_q_heads=gqa_ratio * 4,
    )
    out = torch.empty_like(inputs.q, dtype=torch.bfloat16)
    run_decode(wrapper, inputs, out=out)
    expected = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(inputs.q, (inputs.k_cache, inputs.v_cache), out=out)
    out.zero_()
    run_cuda("FlashInfer graph replay", graph.replay)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("gqa_ratio", (8, 16), ids=("gqa8", "gqa16"))
def test_replan_reuses_compilation_and_releases_workspace(gqa_ratio):
    device = _require_backend()
    from flashinfer.decode import get_trtllm_gen_fmha_module

    wrapper = BatchDecodeWithPagedKVCacheWrapper()

    def exercise(wrapper, lengths, q_len):
        inputs = _make_inputs(device, gqa_ratio=gqa_ratio, lengths=lengths, q_len=q_len)
        wrapper.plan(
            inputs.topk_indices,
            inputs.page_table,
            inputs.seq_lens,
            q_len_per_req=q_len,
            num_q_heads=gqa_ratio * 4,
        )
        actual = run_decode(wrapper, inputs)
        _assert_close(actual, decode_attention_reference(inputs))
        torch.cuda.synchronize()

    exercise(wrapper, (385, 257), 4)
    misses = get_trtllm_gen_fmha_module.cache_info().misses
    allocated = torch.cuda.memory_allocated()
    old_workspace = weakref.ref(wrapper._plan_state.workspace)
    for lengths, q_len in (((129,), 1), ((258, 385, 513), 8), ((257, 519), 16)):
        exercise(wrapper, lengths, q_len)
    assert old_workspace() is None
    assert get_trtllm_gen_fmha_module.cache_info().misses == misses
    assert torch.cuda.memory_allocated() < allocated + 16 * 1024 * 1024
    refs = [
        weakref.ref(getattr(wrapper._plan_state, name))
        for name in ("workspace", "counter", "out", "block_tables", "sparse_seq_lens")
    ]
    live_allocation = torch.cuda.memory_allocated()
    workspace_bytes = wrapper._plan_state.workspace.numel()
    del wrapper
    gc.collect()
    torch.cuda.synchronize()
    assert all(ref() is None for ref in refs)
    assert torch.cuda.memory_allocated() <= live_allocation - workspace_bytes


@pytest.mark.parametrize("gqa_ratio", (8, 16), ids=("gqa8", "gqa16"))
def test_concurrent_graphs_have_independent_workspace(gqa_ratio):
    device = _require_backend()
    inputs = [
        _make_inputs(device, gqa_ratio=gqa_ratio, seed=1701 + i) for i in range(2)
    ]
    wrappers = [BatchDecodeWithPagedKVCacheWrapper() for _ in inputs]
    streams = [torch.cuda.Stream() for _ in inputs]
    graphs = [torch.cuda.CUDAGraph() for _ in inputs]
    outputs = [torch.empty_like(item.q, dtype=torch.bfloat16) for item in inputs]
    references = [decode_attention_reference(item) for item in inputs]
    for wrapper, item, out in zip(wrappers, inputs, outputs):
        wrapper.plan(
            item.topk_indices,
            item.page_table,
            item.seq_lens,
            q_len_per_req=item.q_len_per_req,
            num_q_heads=gqa_ratio * 4,
        )
        run_decode(wrapper, item, out=out)
    for name in ("workspace", "counter", "out"):
        assert (
            getattr(wrappers[0]._plan_state, name).data_ptr()
            != getattr(wrappers[1]._plan_state, name).data_ptr()
        )
    torch.cuda.synchronize()
    for graph, stream, wrapper, item, out in zip(
        graphs, streams, wrappers, inputs, outputs
    ):
        with torch.cuda.graph(graph, stream=stream):
            wrapper.run(item.q, (item.k_cache, item.v_cache), out=out)

    def replay():
        for stream, graph in zip(streams, graphs):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()

    previous = None
    for _ in range(3):
        for out in outputs:
            out.fill_(-777)
        run_cuda("FlashInfer concurrent graphs", replay)
        for actual, expected in zip(outputs, references):
            _assert_close(actual, expected)
        if previous is not None:
            assert all(torch.equal(a, b) for a, b in zip(outputs, previous))
        previous = [out.clone() for out in outputs]
