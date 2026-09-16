"""Numerical, boundary, and metadata-reuse correctness tests."""

from __future__ import annotations

import logging
import time

import pytest
import torch

from inference.msa_v1.indexer.decode.tp4_q8kv8.interface import (
    BatchDecodeIndexerWithPagedKVCacheWrapper,
    _BatchDecodeProxyScoreWrapper,
)
from tests.inference.msa_v1.indexer._common.topk_select.reference import (
    assert_quantized_topk_contract,
)
from tests.inference.msa_v1.indexer.decode.tp4_q8kv8.cases import (
    boundary_lengths,
)
from tests.inference.msa_v1.indexer.decode.tp4_q8kv8.reference import (
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
) -> None:
    max_pages = actual.shape[-1]
    pages = torch.arange(max_pages, device=actual.device).reshape(1, 1, -1)
    query_positions = (
        seq_lens.reshape(-1, 1)
        - 8
        + torch.arange(
            8,
            dtype=torch.int32,
            device=actual.device,
        ).reshape(1, 8)
    )
    local_pages = torch.div(
        query_positions,
        128,
        rounding_mode="floor",
    ).reshape(-1, 8, 1)
    scored_mask = pages < local_pages
    unwritten_mask = pages >= local_pages
    torch.testing.assert_close(
        actual.masked_select(scored_mask),
        expected.masked_select(scored_mask),
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.all(actual.masked_select(unwritten_mask) == _POISON)


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
        seq_lens_cpu = make_seq_lens(case)
        batch = case.batch_size
        max_pages = (int(seq_lens_cpu.max()) + 127) // 128
        seed = case.seed ^ int(case.case_id[-8:], 16)
        q, k_cache, page_table, seq_lens = make_inputs(
            batch,
            max_pages,
            seq_lens_cpu,
            seed=seed,
            device=device,
        )
        output = torch.full(
            (batch, 8, max_pages),
            123.0,
            dtype=torch.float32,
            device=device,
        )
        wrapper = _BatchDecodeProxyScoreWrapper()
        wrapper.plan(page_table, seq_lens)
        actual = _timed_run(
            f"{case.case_id}:proxy",
            lambda: wrapper.run(q, k_cache, out=output),
        )
        expected = indexer_gemm_reference(q, k_cache, page_table, seq_lens)
        _assert_scores(actual, expected, seq_lens)

        topk_wrapper = BatchDecodeIndexerWithPagedKVCacheWrapper()
        topk_wrapper.plan(page_table, seq_lens)
        topk_indices = _timed_run(
            f"{case.case_id}:topk",
            lambda: topk_wrapper.run(q, k_cache),
        )
        query_positions = (
            seq_lens.reshape(-1, 1)
            - 8
            + torch.arange(8, dtype=torch.int32, device=device).reshape(1, 8)
        )
        num_valid_pages = torch.div(
            query_positions,
            128,
            rounding_mode="floor",
        ).add_(1).reshape(-1).contiguous()
        torch.cuda.synchronize()
        assert_quantized_topk_contract(
            actual.view(batch * 8, max_pages).cpu().numpy(),
            num_valid_pages.cpu().numpy(),
            topk_indices.cpu().numpy(),
        )
        if _SUITE == "full" and case.deterministic:
            for repeat in range(2):
                repeated = _timed_run(
                    f"{case.case_id}:topk-repeat-{repeat + 2}",
                    lambda: topk_wrapper.run(q, k_cache),
                )
                assert torch.equal(topk_indices, repeated)


def test_1m_length_and_unwritten_suffix() -> None:
    """Exercise the Q8KV4 maximum page-table width and output contract."""

    device = torch.device("cuda")
    batch = 32
    max_pages = 8192
    lengths = torch.tensor(
        [1048576, 524289, 100001, 129]
        + [8 + index for index in range(batch - 4)],
        dtype=torch.int32,
    )
    q, k_cache, page_table, seq_lens = make_inputs(
        batch,
        max_pages,
        lengths,
        seed=91,
        device=device,
    )
    output = torch.full(
        (batch, 8, max_pages),
        123.0,
        dtype=torch.float32,
        device=device,
    )
    wrapper = _BatchDecodeProxyScoreWrapper()
    wrapper.plan(page_table, seq_lens)
    actual = wrapper.run(q, k_cache, out=output)
    torch.cuda.synchronize()
    pages = torch.arange(max_pages, device=device).reshape(1, 1, -1)
    query_positions = (
        seq_lens.reshape(-1, 1)
        - 8
        + torch.arange(8, dtype=torch.int32, device=device).reshape(1, 8)
    )
    local_pages = torch.div(
        query_positions,
        128,
        rounding_mode="floor",
    ).reshape(-1, 8, 1)
    assert torch.all(actual.masked_select(pages >= local_pages) == 123.0)
    assert torch.all(torch.isfinite(actual.masked_select(pages < local_pages)))


@pytest.mark.parametrize("page_table_kind", ("interval", "reverse"))
@pytest.mark.parametrize("value_kind", ("random", "zero", "extreme", "negative"))
def test_page_table_and_e4m3_value_patterns(
    page_table_kind: str,
    value_kind: str,
) -> None:
    """Cover non-permutation mappings and representative E4M3 patterns."""

    device = torch.device("cuda")
    batch = 2
    max_pages = 4
    q, k_cache, _, seq_lens = make_inputs(
        batch,
        max_pages,
        torch.tensor([512, 385], dtype=torch.int32),
        seed=701,
        device=device,
    )
    if page_table_kind == "interval":
        page_table = torch.tensor(
            [[0, 2, 4, 6], [1, 3, 5, 0]],
            dtype=torch.int32,
            device=device,
        )
    else:
        page_table = torch.tensor(
            [[6, 5, 4, 3], [3, 2, 1, 0]],
            dtype=torch.int32,
            device=device,
        )

    if value_kind == "zero":
        q.zero_()
        k_cache.zero_()
    elif value_kind == "extreme":
        max_e4m3 = torch.finfo(torch.float8_e4m3fn).max
        q.fill_(max_e4m3)
        k_cache.fill_(max_e4m3)
    elif value_kind == "negative":
        q.fill_(-0.5)
        k_cache.fill_(0.5)

    output = torch.full(
        (batch, 8, max_pages),
        123.0,
        dtype=torch.float32,
        device=device,
    )
    wrapper = _BatchDecodeProxyScoreWrapper()
    wrapper.plan(page_table, seq_lens)
    actual = wrapper.run(q, k_cache, out=output)
    expected = indexer_gemm_reference(q, k_cache, page_table, seq_lens)
    torch.cuda.synchronize()
    _assert_scores(actual, expected, seq_lens)


@pytest.mark.parametrize("batch", (129, 257, 1025))
def test_supports_large_batches(batch: int) -> None:
    """Cover batch sizes around and beyond a 1024-thread metadata scan."""

    device = torch.device("cuda")
    max_pages = 2
    seed = 101 + batch
    seq_lens_cpu = boundary_lengths(batch, max_pages, seed)
    q, k_cache, page_table, seq_lens = make_inputs(
        batch,
        max_pages,
        seq_lens_cpu,
        seed=seed,
        device=device,
    )
    output = torch.full(
        (batch, 8, max_pages),
        123.0,
        dtype=torch.float32,
        device=device,
    )
    wrapper = _BatchDecodeProxyScoreWrapper()
    wrapper.plan(page_table, seq_lens)
    actual = wrapper.run(q, k_cache, out=output)
    expected = indexer_gemm_reference(q, k_cache, page_table, seq_lens)
    torch.cuda.synchronize()
    _assert_scores(actual, expected, seq_lens)


@pytest.mark.parametrize("batch", (32, 64, 128, 129))
def test_plan_reuse_across_layers(batch: int) -> None:
    """Reuse one metadata plan with independent Q/K layer inputs."""

    device = torch.device("cuda")
    max_pages = 3
    lengths = boundary_lengths(batch, max_pages, seed=211 + batch)
    layer_0 = make_inputs(
        batch,
        max_pages,
        lengths,
        seed=301,
        device=device,
    )
    layer_1 = make_inputs(
        batch,
        max_pages,
        lengths,
        seed=302,
        device=device,
    )
    page_table, seq_lens = layer_0[2:]
    wrapper = _BatchDecodeProxyScoreWrapper()
    wrapper.plan(page_table, seq_lens)

    for q, k_cache in (layer_0[:2], layer_1[:2]):
        output = torch.full(
            (batch, 8, max_pages),
            123.0,
            dtype=torch.float32,
            device=device,
        )
        actual = wrapper.run(q, k_cache, out=output)
        expected = indexer_gemm_reference(
            q,
            k_cache,
            page_table,
            seq_lens,
        )
        _assert_scores(actual, expected, seq_lens)
