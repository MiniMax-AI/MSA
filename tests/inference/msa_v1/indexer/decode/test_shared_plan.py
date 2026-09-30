"""Shared metadata generation, replay, and specialization contracts."""

import gc
import importlib
import logging
import time
import weakref
from itertools import pairwise

import pytest
import torch

from inference.msa_v1.indexer.decode import BatchDecodeIndexerPlan
from inference.msa_v1.indexer.decode.plan import _COMPILE_CACHE
from tests.inference.msa_v1.indexer._common.topk_select.reference import (
    assert_quantized_topk_contract,
)
from tests.inference.msa_v1.indexer.decode.test_multihead_runtime import _synchronize

logger = logging.getLogger(__name__)


def _check_plan(plan, lengths):
    batch = len(lengths)
    workers = plan.num_workers
    history = [(length - 1) // 128 for length in lengths]
    prefix = [0]
    for count in history:
        prefix.append(prefix[-1] + count)
    quotient, remainder = divmod(prefix[-1], workers)
    boundaries = [i * quotient + min(i, remainder) for i in range(workers + 1)]
    requests, pages, segments = [], [], []
    for begin, end in pairwise(boundaries):
        request = 0
        while request < batch and prefix[request + 1] <= begin:
            request += 1
        requests.append(request)
        pages.append(begin - prefix[request])
        request_end = prefix[request + 1] if request < batch else begin
        segments.append(min(end, request_end, begin + 16) - begin)
    expected = torch.tensor(
        prefix + boundaries + requests + pages + segments, dtype=torch.int32
    )
    torch.testing.assert_close(plan._scheduler.cpu(), expected)
    torch.testing.assert_close(
        plan._seq_lens.cpu(), torch.tensor(lengths, dtype=torch.int32)
    )
    valid = [
        (length - plan.query_length + token) // 128 + 1
        for length in lengths
        for token in range(plan.query_length)
    ]
    expected_valid = torch.tensor([valid] * plan.num_index_heads, dtype=torch.int32)
    torch.testing.assert_close(plan._num_valid_pages.cpu(), expected_valid)


@pytest.mark.gpu
def test_shared_plan_replay_and_cache():
    """Every supported Q and H reuses code across changing request geometry."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    source = torch.tensor([128], dtype=torch.int32, device="cuda")
    for field, invalid_values in (
        ("query_length", (0, 17, True, 1.5)),
        ("num_index_heads", (0, 3, True)),
        ("max_pages", (0, 8193, True)),
    ):
        for value in invalid_values:
            options = {"max_pages": 1, field: value}
            with pytest.raises(ValueError, match=field):
                BatchDecodeIndexerPlan(source, **options)
    required = BatchDecodeIndexerPlan.workspace_size(1, device=source.device)
    with pytest.raises(ValueError, match="too small"):
        BatchDecodeIndexerPlan(
            source,
            max_pages=1,
            workspace_buffer=torch.empty(
                required - 1, dtype=torch.uint8, device="cuda"
            ),
        )
    cache_keys = None
    for query_length in range(1, 17):
        for heads in (1, 2, 4):
            for batch in (1, 7, 32, 129, 257, 513):
                lengths = [max(query_length, (i * 193) % 4097) for i in range(batch)]
                source = torch.tensor(lengths, dtype=torch.int32, device="cuda")
                plan = BatchDecodeIndexerPlan(
                    source,
                    max_pages=33,
                    num_index_heads=heads,
                    query_length=query_length,
                )
                with pytest.raises(RuntimeError, match="update"):
                    plan._require_updated()
                _synchronize()
                start = time.perf_counter()
                plan.update()
                _synchronize()
                logger.info(
                    "Ran decode plan in %.3fms", (time.perf_counter() - start) * 1000
                )
                _check_plan(plan, lengths)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    plan.update()
                # Update bound values, keeping captured source and destination addresses stable.
                lengths.reverse()
                source.copy_(torch.tensor(lengths, dtype=torch.int32))
                graph.replay()
                _synchronize()
                _check_plan(plan, lengths)
                current_keys = set(_COMPILE_CACHE)
                if cache_keys is not None:
                    assert current_keys == cache_keys
                cache_keys = current_keys


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ("q8kv8", "q8kv4"))
@pytest.mark.parametrize("heads", (1, 2, 4))
def test_shared_plan_query_lengths(kind, heads):
    """Check all rows, padding, local-page boundaries and query-length cache reuse."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    api = importlib.import_module(f"inference.msa_v1.indexer.decode.{kind}.interface")
    ref = importlib.import_module(
        f"tests.inference.msa_v1.indexer.decode.{kind}.reference"
    )
    cache_keys = None
    previous_columns = None
    for query_length in range(1, 17):
        lengths = torch.tensor([query_length, 128, 129, 257, 4096], dtype=torch.int32)
        values = ref.make_inputs(
            5,
            32,
            lengths,
            seed=1701,
            device=torch.device("cuda"),
            num_index_heads=heads,
            query_length=query_length,
        )
        q, k, *tail = values
        scale, table, source = tail if kind == "q8kv4" else (None, *tail)
        kwargs = {"k_scale": scale} if scale is not None else {}
        plan = BatchDecodeIndexerPlan(
            source, max_pages=32, num_index_heads=heads, query_length=query_length
        )
        wrapper = api.BatchDecodeIndexerWithPagedKVCacheWrapper()
        wrapper.plan(
            table,
            source,
            num_index_heads=heads,
            query_length=query_length,
            shared_plan=plan,
        )
        plan.update()
        wrapper.run(q, k, **kwargs)
        _synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan.update()
            output = wrapper.run(q, k, **kwargs)
        for replay in range(2):
            if replay:
                source.copy_(source.flip(0))
            wrapper._proxy_scores.fill_(float("nan"))
            graph.replay()
            _synchronize()
            expected = ref.indexer_gemm_reference(*values)
            local = (
                source[:, None]
                - query_length
                + torch.arange(query_length, device="cuda")
            ) // 128
            mask = torch.arange(32, device="cuda").view(1, 1, -1) < local.reshape(
                1, -1, 1
            )
            actual = wrapper._proxy_scores.masked_select(mask)
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(
                actual, expected.masked_select(mask), atol=1e-4, rtol=1e-4
            )
            assert_quantized_topk_contract(
                expected.reshape(-1, 32).cpu().numpy(),
                (local + 1).reshape(-1).repeat(heads).cpu().numpy(),
                output.reshape(-1, 16).cpu().numpy(),
            )
        if kind == "q8kv8":
            current_keys = set(api._COMPILE_CACHE)
        else:
            jit = importlib.import_module("inference.msa_v1.indexer.decode.q8kv4.jit")
            current_keys = jit._load_extension_for_arch.cache_info().misses
        granularity = 8 if kind == "q8kv8" else 16
        query_columns = (
            (query_length * heads + granularity - 1) // granularity
        ) * granularity
        if previous_columns == query_columns:
            assert current_keys == cache_keys
        cache_keys = current_keys
        previous_columns = query_columns


@pytest.mark.gpu
def test_shared_plan_cross_precision_lifetime():
    """Two precision consumers own outputs while sharing only read-only metadata."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    source = torch.tensor([129, 257, 4096], dtype=torch.int32, device="cuda")
    plan = BatchDecodeIndexerPlan(
        source, max_pages=32, num_index_heads=4, query_length=16
    )
    workspace_ref = weakref.ref(plan._workspace)
    consumers = []
    for kind in ("q8kv8", "q8kv4"):
        api = importlib.import_module(
            f"inference.msa_v1.indexer.decode.{kind}.interface"
        )
        ref = importlib.import_module(
            f"tests.inference.msa_v1.indexer.decode.{kind}.reference"
        )
        values = ref.make_inputs(
            3,
            32,
            source,
            seed=1701,
            device=torch.device("cuda"),
            num_index_heads=4,
            query_length=16,
        )
        q, k, *tail = values
        scale, table, _ = tail if kind == "q8kv4" else (None, *tail)
        kwargs = {"k_scale": scale} if scale is not None else {}
        wrapper = api.BatchDecodeIndexerWithPagedKVCacheWrapper()
        wrapper.plan(
            table, source, num_index_heads=4, query_length=16, shared_plan=plan
        )
        consumers.append((wrapper, q, k, kwargs))
    first, second = (consumer[0] for consumer in consumers)
    assert first._proxy_scores.data_ptr() != second._proxy_scores.data_ptr()
    assert first._num_valid_pages.data_ptr() == second._num_valid_pages.data_ptr()
    plan.update()
    for wrapper, q, k, kwargs in consumers:
        wrapper.run(q, k, **kwargs)
    _synchronize()
    expected = [consumer[0]._topk_indices.clone() for consumer in consumers]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream, (wrapper, q, k, kwargs) in zip(streams, consumers, strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            wrapper.run(q, k, **kwargs)
    for stream in streams:
        _synchronize(stream)
    for original, (wrapper, q, k, kwargs) in zip(expected, consumers, strict=True):
        assert torch.equal(original, wrapper._topk_indices)
        kind = "q8kv4" if kwargs else "q8kv8"
        ref = importlib.import_module(
            f"tests.inference.msa_v1.indexer.decode.{kind}.reference"
        )
        arguments = (q, k, kwargs["k_scale"]) if kwargs else (q, k)
        scores = ref.indexer_gemm_reference(
            *arguments, wrapper._proxy_score._page_table, source
        )
        local = (source[:, None] - 16 + torch.arange(16, device="cuda")) // 128
        mask = torch.arange(32, device="cuda").view(1, 1, -1) < local.reshape(1, -1, 1)
        torch.testing.assert_close(
            wrapper._proxy_scores.masked_select(mask),
            scores.masked_select(mask),
            atol=1e-4,
            rtol=1e-4,
        )
    del wrapper, first, second, plan
    consumers.clear()
    gc.collect()
    assert workspace_ref() is None
