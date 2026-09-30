"""Device-plan reuse checks for layer payloads and in-place metadata updates."""

from __future__ import annotations

import pytest
import torch

from datas.inference.cases import InferencePrefillCase
from inference.msa_v1.indexer.prefill.q8kv8 import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)
from tests.inference.msa_v1.indexer.prefill.q8kv8.cases import (
    HEAD_DIM,
    PAGE_SIZE,
    cumulative_lengths,
    require_sm100_device,
)
from tests.inference.msa_v1.indexer.prefill.q8kv8.reference import (
    assert_device_plan,
    assert_topk_structure,
    expected_lengths,
)

pytestmark = pytest.mark.gpu


def _case(
    case_id: str,
    query_lens: tuple[int, ...],
    prefix_lens: tuple[int, ...],
) -> InferencePrefillCase:
    final_kv_lens = tuple(
        query + prefix for query, prefix in zip(query_lens, prefix_lens, strict=True)
    )
    total_q = sum(query_lens)
    max_final_kv = max(final_kv_lens)
    return InferencePrefillCase(
        case_id=case_id,
        batch_size=len(query_lens),
        query_lens=query_lens,
        prefix_lens=prefix_lens,
        final_kv_lens=final_kv_lens,
        count=1,
        metrics={
            "total_q": total_q,
            "max_query_len": max(query_lens),
            "max_final_kv": max_final_kv,
            "max_cols": (max_final_kv + PAGE_SIZE - 1) // PAGE_SIZE,
            "useful_flops": 0,
        },
    )


def _device_cumulative(lengths: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        cumulative_lengths(lengths),
        dtype=torch.int32,
        device=device,
    )


def _assert_score_capacity_domain(
    case: InferencePrefillCase,
    scores: torch.Tensor,
) -> None:
    lengths = expected_lengths(case).to(device=scores.device)
    columns = torch.arange(scores.shape[-1], device=scores.device).reshape(1, -1)
    expected_finite = columns < (lengths.reshape(-1, 1) - 1)
    assert torch.equal(torch.isfinite(scores), expected_finite.expand_as(scores))


@pytest.mark.parametrize("num_index_heads", (1, 2, 4))
def test_plan_reuses_capacity_across_layers_and_metadata_updates(
    num_index_heads,
) -> None:
    device = require_sm100_device()

    first = _case(
        "reuse_first",
        (128, 128, 128, 128),
        (0, 128, 256, 384),
    )
    second = _case(
        "reuse_second",
        (64, 192, 96, 160),
        (512, 0, 256, 128),
    )
    total_q = first.total_q
    assert second.total_q == total_q
    max_seqlen_q = 256
    max_seqlen_k = 1024
    max_cols = max_seqlen_k // PAGE_SIZE
    page_ids = torch.arange(max_cols - 1, -1, -1, dtype=torch.int32, device=device)
    page_table = torch.stack(
        [(page_ids + batch_idx) % max_cols for batch_idx in range(first.batch_size)]
    ).contiguous()

    wrapper = BatchPrefillIndexerWithPagedKVCacheWrapper()
    wrapper.plan(
        _device_cumulative(first.query_lens, device),
        _device_cumulative(first.final_kv_lens, device),
        page_table,
        total_q=total_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_index_heads=num_index_heads,
    )
    state = wrapper._proxy_score.plan_state
    k_cache = torch.ones(
        (max_cols, 1, PAGE_SIZE, HEAD_DIM),
        dtype=torch.float8_e4m3fn,
        device=device,
    )

    state.proxy_scores.fill_(float("nan"))
    q_ones = torch.ones(
        (total_q, num_index_heads, HEAD_DIM),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    wrapper.run(q_ones, k_cache)
    torch.cuda.synchronize()
    assert_device_plan(first, state)
    _assert_score_capacity_domain(first, state.proxy_scores)
    first_valid = torch.isfinite(state.proxy_scores)
    assert bool(torch.all(state.proxy_scores[first_valid] == float(HEAD_DIM)))

    state.proxy_scores.fill_(float("nan"))
    q_zeros = torch.zeros_like(q_ones)
    wrapper.run(q_zeros, k_cache)
    torch.cuda.synchronize()
    _assert_score_capacity_domain(first, state.proxy_scores)
    assert bool(torch.all(state.proxy_scores[first_valid] == 0.0))

    state.cu_seqlens_q.copy_(_device_cumulative(second.query_lens, device))
    state.cu_seqlens_k.copy_(_device_cumulative(second.final_kv_lens, device))
    state.proxy_scores.fill_(float("nan"))
    state.topk_indices.fill_(-777)
    wrapper.replan()
    result = wrapper.run(q_zeros, k_cache)
    torch.cuda.synchronize()

    assert result.data_ptr() == state.topk_indices.data_ptr()
    assert_device_plan(second, state)
    _assert_score_capacity_domain(second, state.proxy_scores)
    second_valid = torch.isfinite(state.proxy_scores)
    assert bool(torch.all(state.proxy_scores[second_valid] == 0.0))
    assert_topk_structure(state.topk_indices, state.num_valid_pages)
