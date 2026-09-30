"""Numerical and boundary correctness tests."""

from __future__ import annotations

import logging
import os
import threading
import time
from functools import partial

import pytest
import torch

from inference.msa_v1.indexer.decode.q8kv4.interface import (
    BatchDecodeIndexerWithPagedKVCacheWrapper,
    _BatchDecodeProxyScoreWrapper,
)
from tests.inference.cases import active_inference_suite
from tests.inference.msa_v1.decode.cases import correctness_cases, make_seq_lens
from tests.inference.msa_v1.indexer._common.topk_select.reference import (
    assert_quantized_topk_contract,
)
from tests.inference.msa_v1.indexer.decode.q8kv4.cases import boundary_lengths
from tests.inference.msa_v1.indexer.decode.q8kv4.reference import (
    indexer_gemm_reference,
    make_inputs,
)

logger = logging.getLogger(__name__)
_SUITE = active_inference_suite()
_CASES = correctness_cases(_SUITE)
_SHARD_SIZE = 8
_CASE_SHARDS = tuple(
    _CASES[offset : offset + _SHARD_SIZE]
    for offset in range(0, len(_CASES), _SHARD_SIZE)
)
_POISON = 123.0


def _assert_scores(
    actual: torch.Tensor,
    expected: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    query_length = actual.shape[1] // seq_lens.numel()
    max_pages = actual.shape[-1]
    pages = torch.arange(max_pages, device=actual.device).reshape(1, 1, -1)
    query_positions = (
        seq_lens.reshape(-1, 1)
        - query_length
        + torch.arange(query_length, dtype=torch.int32, device=actual.device).reshape(
            1, query_length
        )
    )
    local_pages = torch.div(query_positions, 128, rounding_mode="floor").reshape(
        -1, query_length, 1
    )
    scored_mask = pages < local_pages.reshape(1, -1, 1)
    unwritten_mask = pages >= local_pages.reshape(1, -1, 1)
    torch.testing.assert_close(
        actual.masked_select(scored_mask),
        expected.masked_select(scored_mask),
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.all(actual.masked_select(unwritten_mask) == _POISON)
    return local_pages.reshape(-1).add(1).contiguous()


def _timed_run(label: str, run):
    watchdog = threading.Timer(30.0, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = run()
        torch.cuda.synchronize()
    finally:
        watchdog.cancel()
    elapsed = time.perf_counter() - start
    logger.info("%s ran in %.3fms", label, elapsed * 1e3)
    assert elapsed < 30.0, f"{label} exceeded the 30-second deadlock threshold"
    return result


@pytest.mark.parametrize(
    "case_shard",
    _CASE_SHARDS,
    ids=lambda shard: f"{shard[0].case_id}-{shard[-1].case_id}",
)
@pytest.mark.parametrize("num_index_heads", (1, 2, 4))
def test_selected_sequence_cases(case_shard, num_index_heads) -> None:
    """Run the shared 96/256-case decode correctness suite."""

    device = torch.device("cuda")
    for case in case_shard:
        lengths = make_seq_lens(case)
        batch = case.batch_size
        query_length = case.q_len_per_req
        max_pages = (int(lengths.max()) + 127) // 128
        seed = case.seed ^ int(case.case_id[-8:], 16)
        inputs = make_inputs(
            batch,
            max_pages,
            lengths,
            seed=seed,
            device=device,
            num_index_heads=num_index_heads,
            query_length=query_length,
        )
        output = torch.full(
            (num_index_heads, batch * query_length, max_pages),
            _POISON,
            dtype=torch.float32,
            device=device,
        )
        q, packed_k, k_scale, page_table, seq_lens = inputs
        wrapper = _BatchDecodeProxyScoreWrapper()
        wrapper.plan(
            page_table,
            seq_lens,
            num_index_heads=num_index_heads,
            query_length=query_length,
        )
        actual = _timed_run(
            f"{case.case_id}:proxy",
            partial(wrapper.run, q, packed_k, k_scale, out=output),
        )
        expected = indexer_gemm_reference(*inputs)
        num_valid_pages = _assert_scores(actual, expected, seq_lens)

        topk_wrapper = BatchDecodeIndexerWithPagedKVCacheWrapper()
        topk_wrapper.plan(
            page_table,
            seq_lens,
            num_index_heads=num_index_heads,
            query_length=query_length,
        )
        from inference.msa_v1.indexer._common.topk_select.build import load_extension

        load_extension()
        topk_indices = _timed_run(
            f"{case.case_id}:topk",
            partial(topk_wrapper.run, q, packed_k, k_scale=k_scale),
        )
        assert_quantized_topk_contract(
            expected.view(-1, max_pages).cpu().numpy(),
            num_valid_pages.repeat(num_index_heads).cpu().numpy(),
            topk_indices.reshape(-1, 16).cpu().numpy(),
        )
        if _SUITE == "full" and case.deterministic:
            topk_indices = topk_indices.clone()
            for repeat in range(2):
                repeated = _timed_run(
                    f"{case.case_id}:topk-repeat-{repeat + 2}",
                    partial(topk_wrapper.run, q, packed_k, k_scale=k_scale),
                )
                assert torch.equal(topk_indices, repeated)


@pytest.mark.parametrize("batch", (129, 257, 1025))
def test_scheduler_supports_batches_beyond_single_cta_scan(batch: int) -> None:
    """Cover runtime batch sizes on both sides of the 1024-thread scan."""

    device = torch.device("cuda")
    max_pages = 2
    seed = 101 + batch
    lengths = boundary_lengths(batch, max_pages, seed)
    inputs = make_inputs(batch, max_pages, lengths, seed=seed, device=device)
    query_length = inputs[0].shape[1]
    output = torch.full(
        (1, batch * query_length, max_pages), 123.0, dtype=torch.float32, device=device
    )
    q, packed_k, k_scale, page_table, seq_lens = inputs
    wrapper = _BatchDecodeProxyScoreWrapper()
    wrapper.plan(page_table, seq_lens)
    actual = wrapper.run(q, packed_k, k_scale, out=output)
    expected = indexer_gemm_reference(*inputs)
    torch.cuda.synchronize()
    pages = torch.arange(max_pages, device=device).reshape(1, 1, -1)
    query_positions = (
        inputs[-1].reshape(-1, 1)
        - query_length
        + torch.arange(query_length, dtype=torch.int32, device=device).reshape(
            1, query_length
        )
    )
    local_pages = torch.div(query_positions, 128, rounding_mode="floor").reshape(
        1, -1, 1
    )
    scored_mask = pages < local_pages.reshape(1, -1, 1)
    unwritten_mask = pages >= local_pages.reshape(1, -1, 1)
    torch.testing.assert_close(
        actual.masked_select(scored_mask),
        expected.masked_select(scored_mask),
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.all(actual.masked_select(unwritten_mask) == 123.0)


def test_1m_length_and_unwritten_suffix() -> None:
    """Exercise the maximum page-table width without materializing a huge oracle."""

    device = torch.device("cuda")
    batch = 32
    max_pages = 8192
    lengths = torch.tensor(
        [1048576, 524289, 100001, 129] + [8 + index for index in range(batch - 4)],
        dtype=torch.int32,
    )
    inputs = make_inputs(batch, max_pages, lengths, seed=91, device=device)
    output = torch.full(
        (1, batch * 8, max_pages), 123.0, dtype=torch.float32, device=device
    )
    q, packed_k, k_scale, page_table, seq_lens = inputs
    wrapper = _BatchDecodeProxyScoreWrapper()
    wrapper.plan(page_table, seq_lens)
    actual = wrapper.run(q, packed_k, k_scale, out=output)
    torch.cuda.synchronize()
    pages = torch.arange(max_pages, device=device).reshape(1, 1, -1)
    query_positions = (
        inputs[-1].reshape(-1, 1)
        - 8
        + torch.arange(8, dtype=torch.int32, device=device).reshape(1, 8)
    )
    local_pages = torch.div(query_positions, 128, rounding_mode="floor").reshape(
        1, -1, 1
    )
    assert torch.all(actual.masked_select(pages >= local_pages) == 123.0)
    assert torch.all(torch.isfinite(actual.masked_select(pages < local_pages)))


@pytest.mark.parametrize("batch", (32, 129))
@pytest.mark.parametrize("num_index_heads", (1, 2, 4))
def test_plan_reuse_across_layers(batch: int, num_index_heads: int) -> None:
    """Reuse one scheduler plan with independently generated layer caches."""

    device = torch.device("cuda")
    max_pages = 3
    lengths = boundary_lengths(batch, max_pages, seed=211 + batch)
    layer_0 = make_inputs(
        batch,
        max_pages,
        lengths,
        seed=301,
        device=device,
        num_index_heads=num_index_heads,
    )
    layer_1 = make_inputs(
        batch,
        max_pages,
        lengths,
        seed=302,
        device=device,
        num_index_heads=num_index_heads,
    )
    # Exercise satfinite E4M3 dequantization with both signs and varying scales.
    layer_1[2].copy_((layer_1[2].float() * 1024).clamp(max=448).to(layer_1[2].dtype))
    page_table, seq_lens = layer_0[3:]
    wrapper = _BatchDecodeProxyScoreWrapper()
    wrapper.plan(page_table, seq_lens, num_index_heads=num_index_heads)

    for q, packed_k, k_scale in (layer_0[:3], layer_1[:3]):
        output = torch.full(
            (num_index_heads, batch * 8, max_pages),
            123.0,
            dtype=torch.float32,
            device=device,
        )
        actual = wrapper.run(q, packed_k, k_scale, out=output)
        expected = indexer_gemm_reference(q, packed_k, k_scale, page_table, seq_lens)
        query_positions = (
            seq_lens.reshape(-1, 1)
            - 8
            + torch.arange(8, dtype=torch.int32, device=device).reshape(1, 8)
        )
        local_pages = torch.div(query_positions, 128, rounding_mode="floor").reshape(
            1, -1, 1
        )
        pages = torch.arange(max_pages, device=device).reshape(1, 1, -1)
        torch.testing.assert_close(
            actual.masked_select(pages < local_pages),
            expected.masked_select(pages < local_pages),
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.all(actual.masked_select(pages >= local_pages) == 123.0)


def test_replan_updates_metadata() -> None:
    device = torch.device("cuda")
    batch = 129
    max_pages = 3
    first = make_inputs(
        batch,
        max_pages,
        boundary_lengths(batch, max_pages, seed=401),
        seed=402,
        device=device,
    )
    second = make_inputs(
        batch,
        max_pages,
        boundary_lengths(batch, max_pages, seed=403),
        seed=404,
        device=device,
    )
    wrapper = _BatchDecodeProxyScoreWrapper()
    for inputs in (first, second):
        q, packed_k, k_scale, page_table, seq_lens = inputs
        wrapper.plan(page_table, seq_lens)
        output = torch.full(
            (1, batch * 8, max_pages), 123.0, dtype=torch.float32, device=device
        )
        actual = wrapper.run(q, packed_k, k_scale, out=output)
        expected = indexer_gemm_reference(*inputs)
        query_positions = (
            seq_lens.reshape(-1, 1)
            - 8
            + torch.arange(8, dtype=torch.int32, device=device).reshape(1, 8)
        )
        local_pages = torch.div(query_positions, 128, rounding_mode="floor").reshape(
            1, -1, 1
        )
        pages = torch.arange(max_pages, device=device).reshape(1, 1, -1)
        torch.testing.assert_close(
            actual.masked_select(pages < local_pages),
            expected.masked_select(pages < local_pages),
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.all(actual.masked_select(pages >= local_pages) == 123.0)
