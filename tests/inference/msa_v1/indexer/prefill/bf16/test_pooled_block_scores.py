"""Pooled prefill scores, complete-group visibility, and mutable graph metadata."""

import dataclasses
import gc
import weakref
from functools import partial

import pytest
import torch

from datas.inference.tensors import cumulative_lengths
from inference.msa_v1.indexer.prefill.bf16.interface import (
    _GEMM_COMPILE_CACHE,
    _SCHEDULE_COMPILE_CACHE,
    BatchPrefillBlockScoreWrapper,
    _compile_gemm,
)
from tests.inference.cases import (
    active_inference_suite,
    selected_msa_v1_prefill_test_cases,
)
from tests.inference.msa_v1.attention.decode.runtime import run_cuda
from tests.inference.msa_v1.cache_layout import CACHE_LAYOUTS, cache_view
from tests.inference.msa_v1.indexer.prefill.bf16.real_cases import (
    make_real_prefill_inputs,
)
from tests.inference.msa_v1.indexer.prefill.q8kv8.test_plan_reuse import _case

pytestmark = pytest.mark.gpu
_CASES = selected_msa_v1_prefill_test_cases()
_SHARDS = min(64, len(_CASES))


def _inputs(case, heads, factor, page_size=128, layout="contiguous"):
    inputs = make_real_prefill_inputs(
        case,
        num_index_heads=heads,
        device=torch.device("cuda"),
        page_size=page_size,
        rows_per_page=page_size // factor,
    )
    for request, length in enumerate(case.final_kv_lens):
        inputs.page_table[request, (length + page_size - 1) // page_size :].fill_(-1)
    return dataclasses.replace(inputs, k_cache=cache_view(inputs.k_cache, layout))


def _plan(inputs, case, factor, page_size=128):
    wrapper = BatchPrefillBlockScoreWrapper()
    wrapper.plan(
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        inputs.page_table,
        total_q=case.total_q,
        max_seqlen_q=case.max_query_len,
        max_seqlen_k=case.max_final_kv,
        num_index_heads=inputs.q.shape[1],
        pooling_factor=factor,
        page_size=page_size,
    )
    _compile_gemm(
        inputs.q.view(-1, 128),
        inputs.k_cache[:, 0].permute(1, 2, 0),
        wrapper._plan_state,
        output=torch.empty(
            (inputs.q.shape[1], case.total_q, case.max_cols),
            dtype=torch.float32,
            device=inputs.q.device,
        ),
    )
    return wrapper


def _reference(case, inputs, factor, page_size=128):
    """Expand each stored key per token and mask by its completion position."""
    result = torch.full(
        (inputs.q.shape[1], case.total_q, case.max_cols),
        -torch.inf,
        dtype=torch.float32,
        device="cuda",
    )
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        q_begin = 0
        for request, (q_len, kv_len) in enumerate(
            zip(case.query_lens, case.final_kv_lens, strict=True)
        ):
            if not q_len:
                continue
            physical = inputs.page_table[
                request, : (kv_len + page_size - 1) // page_size
            ]
            keys = inputs.k_cache[physical.long(), 0].reshape(-1, 128)
            expanded = keys.repeat_interleave(factor, dim=0)
            blocks = (kv_len + 127) // 128
            if expanded.shape[0] < blocks * 128:
                expanded = torch.cat(
                    (expanded, torch.zeros_like(expanded[: 128 - page_size]))
                )
            expanded = expanded[: blocks * 128].float().view(blocks, 128, 128)
            for query_start in range(0, q_len, 64):
                query_end = min(query_start + 64, q_len)
                queries = inputs.q[q_begin + query_start : q_begin + query_end].float()
                visible = (
                    kv_len
                    - q_len
                    + torch.arange(query_start, query_end, device="cuda")
                    + 1
                )
                for block_start in range(0, blocks, 64):
                    block_end = min(block_start + 64, blocks)
                    dots = torch.einsum(
                        "qhd,btd->hqbt", queries, expanded[block_start:block_end]
                    )
                    positions = torch.arange(
                        block_start * 128, block_end * 128, device="cuda"
                    ).view(block_end - block_start, 128)
                    completion = (positions // factor + 1) * factor
                    dots.masked_fill_(
                        completion[None, None] > visible[None, :, None, None],
                        -torch.inf,
                    )
                    result[
                        :,
                        q_begin + query_start : q_begin + query_end,
                        block_start:block_end,
                    ] = dots.amax(-1)
            q_begin += q_len
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return result


def _check(case, inputs, output, factor, page_size=128):
    expected = _reference(case, inputs, factor, page_size)
    assert torch.equal(torch.isneginf(output), torch.isneginf(expected))
    assert not torch.isnan(output).any()
    torch.testing.assert_close(output, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("heads", (1, 2, 4))
@pytest.mark.parametrize("page_size", (64, 128))
@pytest.mark.parametrize("shard", range(_SHARDS))
def test_pooled_shared_varlen_cases(heads, page_size, shard):
    for case in _CASES[shard::_SHARDS]:
        inputs = _inputs(case, heads, 4, page_size)
        wrapper = _plan(inputs, case, 4, page_size)
        output = run_cuda(case.case_id, partial(wrapper.run, inputs.q, inputs.k_cache))
        _check(case, inputs, output, 4, page_size)
        if active_inference_suite() == "full":
            baseline = output.clone()
            for _ in range(2):
                repeated = run_cuda(
                    case.case_id, partial(wrapper.run, inputs.q, inputs.k_cache)
                )
                assert torch.equal(baseline, repeated)


@pytest.mark.parametrize("heads", (1, 2, 4))
@pytest.mark.parametrize("factor", (4, 2, 1))
@pytest.mark.parametrize("page_size", (64, 128))
def test_partial_groups_blocks_and_mutable_graph(heads, factor, page_size):
    # Irregular prefixes cross group, physical page, and logical block boundaries.
    cases = (
        _case("pooled_boundaries", (133, 4, 1), (0, 125, 131)),
        _case("pooled_shrink", (5, 0, 129), (2, 1, 0)),
        _case("pooled_empty", (0, 0, 0), (2, 1, 0)),
    )
    inputs = _inputs(cases[0], heads, factor, page_size, "page")
    wrapper = _plan(inputs, cases[0], factor, page_size)
    out = torch.empty(
        (heads, cases[0].total_q, cases[0].max_cols),
        dtype=torch.float32,
        device="cuda",
    )
    run_cuda("pooled warmup", partial(wrapper.run, inputs.q, inputs.k_cache, out=out))
    _check(cases[0], inputs, out, factor, page_size)
    before_keys = (set(_GEMM_COMPILE_CACHE), set(_SCHEDULE_COMPILE_CACHE))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(inputs.q, inputs.k_cache, out=out)
    for case in cases:
        inputs.cu_seqlens_q.copy_(
            torch.tensor(
                cumulative_lengths(case.query_lens), dtype=torch.int32, device="cuda"
            )
        )
        inputs.cu_seqlens_k.copy_(
            torch.tensor(
                cumulative_lengths(case.final_kv_lens), dtype=torch.int32, device="cuda"
            )
        )
        for request, length in enumerate(case.final_kv_lens):
            inputs.page_table[request, (length + page_size - 1) // page_size :].fill_(
                -1
            )
        out.fill_(float("nan"))
        run_cuda("pooled graph replay", graph.replay)
        # Planned Q/output capacity can exceed the new packed query count.
        expected = torch.full_like(out, -torch.inf)
        current = _reference(case, inputs, factor, page_size)
        expected[:, : case.total_q, : case.max_cols] = current
        torch.testing.assert_close(out, expected, atol=2e-4, rtol=2e-4)
        assert torch.equal(torch.isneginf(out), torch.isneginf(expected))
    assert before_keys == (set(_GEMM_COMPILE_CACHE), set(_SCHEDULE_COMPILE_CACHE))


@pytest.mark.parametrize("layout", CACHE_LAYOUTS)
@pytest.mark.parametrize("page_size", (64, 128))
def test_pooled_layouts_and_independent_owners(layout, page_size):
    case = _case("pooled_layout", (65, 3), (1025, 127))
    inputs = _inputs(case, 2, 4, page_size, layout=layout)
    owners = [_plan(inputs, case, 4, page_size) for _ in range(2)]
    streams = [torch.cuda.Stream() for _ in owners]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for stream, wrapper in zip(streams, owners, strict=True):
        with torch.cuda.stream(stream):
            wrapper.run(inputs.q, inputs.k_cache)
    run_cuda("pooled concurrent completion", lambda: None)
    references = []
    for wrapper in owners:
        _check(case, inputs, wrapper._plan_state.default_out, 4, page_size)
        references.append(weakref.ref(wrapper._plan_state.default_out))
    assert (
        owners[0]._plan_state.default_out.data_ptr()
        != owners[1]._plan_state.default_out.data_ptr()
    )
    del wrapper, owners
    gc.collect()
    assert all(ref() is None for ref in references)


@pytest.mark.parametrize("factor,page_size", ((8, 128), (4, 32)))
def test_reject_inconsistent_pooled_geometry(factor, page_size):
    metadata = torch.tensor((0, 1), dtype=torch.int32, device="cuda")
    pages = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="pooling_factor|page_size"):
        BatchPrefillBlockScoreWrapper().plan(
            metadata,
            metadata,
            pages,
            total_q=1,
            max_seqlen_q=1,
            max_seqlen_k=1,
            pooling_factor=factor,
            page_size=page_size,
        )


@pytest.mark.parametrize("heads", (1, 4))
@pytest.mark.parametrize("factor", (4, 2, 1))
@pytest.mark.parametrize("page_size", (64, 128))
def test_pooled_split_and_long_context(heads, factor, page_size):
    cases = (
        _case("pooled_split", (5, 2, 3), (65533, 65665, 1048573)),
        # The planned width itself must leave a partial final grid split.
        _case("pooled_grid_tail", (5, 2, 3), (65533, 65665, 65537)),
    )
    for case in cases:
        inputs = _inputs(case, heads, factor, page_size)
        wrapper = _plan(inputs, case, factor, page_size)
        output_shape = (heads, case.total_q, case.max_cols)
        output_elements = heads * case.total_q * case.max_cols
        storage = torch.full((output_elements + 8,), 12345.0, device="cuda")
        out = storage[:output_elements].view(output_shape)
        output = run_cuda(
            "pooled split boundary",
            partial(wrapper.run, inputs.q, inputs.k_cache, out=out),
        )
        _check(case, inputs, output, factor, page_size)
        assert torch.all(storage[output_elements:] == 12345.0)


@pytest.mark.parametrize("factor", (4, 2, 1))
@pytest.mark.parametrize("page_size", (64, 128))
@pytest.mark.parametrize("key_value", (0.0, -0.0, -1.0))
def test_pooled_zero_and_negative_scores(factor, page_size, key_value):
    case = _case("pooled_zero_negative", (3, 2), (0, 127))
    inputs = _inputs(case, 2, factor, page_size)
    inputs.q.fill_(1.0)
    inputs.k_cache.fill_(key_value)
    wrapper = _plan(inputs, case, factor, page_size)
    output = run_cuda(
        "pooled zero/negative", partial(wrapper.run, inputs.q, inputs.k_cache)
    )
    _check(case, inputs, output, factor, page_size)
    baseline = output.clone()
    repeated = run_cuda(
        "pooled zero/negative repeat", partial(wrapper.run, inputs.q, inputs.k_cache)
    )
    assert torch.equal(baseline.view(torch.int32), repeated.view(torch.int32))
    if key_value < 0:
        assert torch.all(output[torch.isfinite(output)] == -128)


@pytest.mark.parametrize("factor", (4, 2, 1))
def test_pooled_maximum_context(factor):
    case = _case("pooled_upper_bound", (1,), (2097151,))
    inputs = _inputs(case, 2, factor, 64)
    wrapper = _plan(inputs, case, factor, 64)
    output = run_cuda(
        "pooled maximum context", partial(wrapper.run, inputs.q, inputs.k_cache)
    )
    _check(case, inputs, output, factor, 64)


def test_pooled_explicit_out_does_not_allocate_default():
    case = _case("pooled_lazy_out", (3,), (127,))
    inputs = _inputs(case, 2, 4)
    wrapper = _plan(inputs, case, 4)
    assert wrapper._plan_state.default_out is None
    out = torch.empty((2, case.total_q, case.max_cols), device="cuda")
    result = run_cuda(
        "pooled explicit out", partial(wrapper.run, inputs.q, inputs.k_cache, out=out)
    )
    assert result is out
    assert wrapper._plan_state.default_out is None
    _check(case, inputs, out, 4)
    default = run_cuda(
        "pooled lazy output", partial(wrapper.run, inputs.q, inputs.k_cache)
    )
    assert wrapper._plan_state.default_out is default
    assert default.data_ptr() != out.data_ptr()
    assert torch.equal(default.view(torch.int32), out.view(torch.int32))


@pytest.mark.parametrize(
    "changes,message",
    (
        ({"num_index_heads": 3}, "num_index_heads"),
        ({"total_q": 32768, "max_seqlen_k": 2097152}, "fewer than"),
        ({"max_seqlen_k": 129, "page_size": 64}, "page_table capacity"),
        ({"max_seqlen_k": 2097153}, "maximum page count"),
    ),
)
def test_pooled_plan_rejects_invalid_capacity(changes, message):
    metadata = torch.tensor((0, 1), dtype=torch.int32, device="cuda")
    pages = torch.zeros((1, 2), dtype=torch.int32, device="cuda")
    kwargs = dict(total_q=1, max_seqlen_q=1, max_seqlen_k=1)
    kwargs.update(changes)
    if changes.get("total_q") == 32768:
        pages = torch.zeros((1, 16384), dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match=message):
        BatchPrefillBlockScoreWrapper().plan(metadata, metadata, pages, **kwargs)


def test_pooled_run_requires_plan():
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called"):
        BatchPrefillBlockScoreWrapper().run(torch.empty(0), torch.empty(0))


@pytest.mark.parametrize("invalid", ("cache", "out_dtype", "out_shape"))
def test_pooled_run_rejects_invalid_layout(invalid):
    case = _case("pooled_invalid_layout", (3,), (127,))
    inputs = _inputs(case, 2, 4, 64)
    wrapper = _plan(inputs, case, 4, 64)
    cache = inputs.k_cache
    out = torch.empty((2, case.total_q, case.max_cols), device="cuda")
    if invalid == "cache":
        cache = torch.empty((1, 1, 32, 128), device="cuda", dtype=torch.bfloat16)
    elif invalid == "out_dtype":
        out = out.bfloat16()
    else:
        out = out[:, :1].contiguous()
    with pytest.raises(ValueError, match="shape|torch.float32"):
        wrapper.run(inputs.q, cache, out=out)


def test_pooled_plan_is_rejected_during_capture():
    case = _case("pooled_capture_plan", (1,), (127,))
    inputs = _inputs(case, 1, 4)
    wrapper = _plan(inputs, case, 4)
    run_cuda("pooled capture warmup", partial(wrapper.run, inputs.q, inputs.k_cache))
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="outside CUDA Graph capture"),
        torch.cuda.graph(graph),
    ):
        wrapper.plan(
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            inputs.page_table,
            total_q=1,
            max_seqlen_q=1,
            max_seqlen_k=128,
        )


def test_pooled_default_output_requires_warmup_before_capture():
    case = _case("pooled_capture_default", (1,), (127,))
    inputs = _inputs(case, 1, 4)
    wrapper = _plan(inputs, case, 4)
    out = torch.empty((1, 1, 1), device="cuda")
    run_cuda(
        "pooled explicit capture warmup",
        partial(wrapper.run, inputs.q, inputs.k_cache, out=out),
    )
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="provide out or warm up"),
        torch.cuda.graph(graph),
    ):
        wrapper.run(inputs.q, inputs.k_cache)
    assert wrapper._plan_state.default_out is None
