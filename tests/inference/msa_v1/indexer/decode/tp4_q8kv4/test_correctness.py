"""Numerical and boundary correctness tests."""

from __future__ import annotations

import logging
import time

import pytest
import torch

from inference.msa_v1.indexer.decode.tp4_q8kv4.interface import (
    BatchDecodeIndexerWithPagedKVCacheWrapper,
    _BatchDecodeProxyScoreWrapper,
)
from tests.inference.msa_v1.indexer._common.topk_select.reference import (
    assert_quantized_topk_contract,
)
from tests.inference.msa_v1.indexer.decode.tp4_q8kv4.cases import boundary_lengths
from tests.inference.msa_v1.indexer.decode.tp4_q8kv4.reference import (
    indexer_gemm_reference,
    make_inputs,
)
from tests.inference.msa_v1.decode.cases import correctness_cases, make_seq_lens
from tests.inference.cases import active_inference_suite

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
    max_pages = actual.shape[-1]
    pages = torch.arange(max_pages, device=actual.device).reshape(1, 1, -1)
    query_positions = seq_lens.reshape(-1, 1) - 8 + torch.arange(
        8, dtype=torch.int32, device=actual.device
    ).reshape(1, 8)
    local_pages = torch.div(query_positions, 128, rounding_mode="floor").reshape(
        -1, 8, 1
    )
    scored_mask = pages < local_pages
    unwritten_mask = pages >= local_pages
    torch.testing.assert_close(
        actual.masked_select(scored_mask),
        expected.masked_select(scored_mask),
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.all(actual.masked_select(unwritten_mask) == _POISON)
    return local_pages.reshape(-1).add(1).contiguous()


def _timed_run(label: str, run):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    logger.info("%s ran in %.3fms", label, elapsed * 1e3)
    assert elapsed < 30.0, f"{label} exceeded the 30-second deadlock threshold"
    return result


@pytest.mark.parametrize(
    "case_shard",
    _CASE_SHARDS,
    ids=lambda shard: f"{shard[0].case_id}-{shard[-1].case_id}",
)
def test_selected_sequence_cases(case_shard) -> None:
    """Run the shared 96/256-case decode correctness suite."""

    device = torch.device("cuda")
    for case in case_shard:
        lengths = make_seq_lens(case)
        batch = case.batch_size
        max_pages = (int(lengths.max()) + 127) // 128
        seed = case.seed ^ int(case.case_id[-8:], 16)
        inputs = make_inputs(batch, max_pages, lengths, seed=seed, device=device)
        output = torch.full(
            (batch, 8, max_pages), _POISON, dtype=torch.float32, device=device
        )
        q, packed_k, k_scale, page_table, seq_lens = inputs
        wrapper = _BatchDecodeProxyScoreWrapper()
        wrapper.plan(page_table, seq_lens)
        actual = _timed_run(
            f"{case.case_id}:proxy",
            lambda: wrapper.run(q, packed_k, k_scale, out=output),
        )
        expected = indexer_gemm_reference(*inputs)
        num_valid_pages = _assert_scores(actual, expected, seq_lens)

        topk_wrapper = BatchDecodeIndexerWithPagedKVCacheWrapper()
        topk_wrapper.plan(page_table, seq_lens)
        topk_indices = _timed_run(
            f"{case.case_id}:topk",
            lambda: topk_wrapper.run(q, packed_k, k_scale=k_scale),
        )
        assert_quantized_topk_contract(
            actual.view(batch * 8, max_pages).cpu().numpy(),
            num_valid_pages.cpu().numpy(),
            topk_indices.cpu().numpy(),
        )
        if _SUITE == "full" and case.deterministic:
            for repeat in range(2):
                repeated = _timed_run(
                    f"{case.case_id}:topk-repeat-{repeat + 2}",
                    lambda: topk_wrapper.run(q, packed_k, k_scale=k_scale),
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
    output = torch.full(
        (batch, 8, max_pages), 123.0, dtype=torch.float32, device=device
    )
    q, packed_k, k_scale, page_table, seq_lens = inputs
    wrapper = _BatchDecodeProxyScoreWrapper()
    wrapper.plan(page_table, seq_lens)
    actual = wrapper.run(q, packed_k, k_scale, out=output)
    expected = indexer_gemm_reference(*inputs)
    torch.cuda.synchronize()
    pages = torch.arange(max_pages, device=device).reshape(1, 1, -1)
    query_positions = inputs[-1].reshape(-1, 1) - 8 + torch.arange(
        8, dtype=torch.int32, device=device
    ).reshape(1, 8)
    local_pages = torch.div(
        query_positions, 128, rounding_mode="floor"
    ).reshape(-1, 8, 1)
    scored_mask = pages < local_pages
    unwritten_mask = pages >= local_pages
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
        [1048576, 524289, 100001, 129]
        + [8 + index for index in range(batch - 4)],
        dtype=torch.int32,
    )
    inputs = make_inputs(batch, max_pages, lengths, seed=91, device=device)
    output = torch.full(
        (batch, 8, max_pages), 123.0, dtype=torch.float32, device=device
    )
    q, packed_k, k_scale, page_table, seq_lens = inputs
    wrapper = _BatchDecodeProxyScoreWrapper()
    wrapper.plan(page_table, seq_lens)
    actual = wrapper.run(q, packed_k, k_scale, out=output)
    torch.cuda.synchronize()
    pages = torch.arange(max_pages, device=device).reshape(1, 1, -1)
    query_positions = inputs[-1].reshape(-1, 1) - 8 + torch.arange(
        8, dtype=torch.int32, device=device
    ).reshape(1, 8)
    local_pages = torch.div(
        query_positions, 128, rounding_mode="floor"
    ).reshape(-1, 8, 1)
    assert torch.all(actual.masked_select(pages >= local_pages) == 123.0)
    assert torch.all(torch.isfinite(actual.masked_select(pages < local_pages)))


@pytest.mark.parametrize("batch", (32, 129))
def test_plan_reuse_across_layers(batch: int) -> None:
    """Reuse one scheduler plan with independently generated layer caches."""

    device = torch.device("cuda")
    max_pages = 3
    lengths = boundary_lengths(batch, max_pages, seed=211 + batch)
    layer_0 = make_inputs(batch, max_pages, lengths, seed=301, device=device)
    layer_1 = make_inputs(batch, max_pages, lengths, seed=302, device=device)
    page_table, seq_lens = layer_0[3:]
    wrapper = _BatchDecodeProxyScoreWrapper()
    wrapper.plan(page_table, seq_lens)

    for q, packed_k, k_scale in (layer_0[:3], layer_1[:3]):
        output = torch.full(
            (batch, 8, max_pages), 123.0, dtype=torch.float32, device=device
        )
        actual = wrapper.run(q, packed_k, k_scale, out=output)
        expected = indexer_gemm_reference(
            q, packed_k, k_scale, page_table, seq_lens
        )
        query_positions = seq_lens.reshape(-1, 1) - 8 + torch.arange(
            8, dtype=torch.int32, device=device
        ).reshape(1, 8)
        local_pages = torch.div(
            query_positions, 128, rounding_mode="floor"
        ).reshape(-1, 8, 1)
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
            (batch, 8, max_pages), 123.0, dtype=torch.float32, device=device
        )
        actual = wrapper.run(q, packed_k, k_scale, out=output)
        expected = indexer_gemm_reference(*inputs)
        query_positions = seq_lens.reshape(-1, 1) - 8 + torch.arange(
            8, dtype=torch.int32, device=device
        ).reshape(1, 8)
        local_pages = torch.div(
            query_positions, 128, rounding_mode="floor"
        ).reshape(-1, 8, 1)
        pages = torch.arange(max_pages, device=device).reshape(1, 1, -1)
        torch.testing.assert_close(
            actual.masked_select(pages < local_pages),
            expected.masked_select(pages < local_pages),
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.all(actual.masked_select(pages >= local_pages) == 123.0)
