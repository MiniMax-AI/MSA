"""Exhaustive structural execution checks for all 10,562 prefill shapes."""

from __future__ import annotations

import pytest
import torch

from datas.inference.cases import (
    InferencePrefillCase,
    load_prefill_test_cases,
)
from inference.msa_v1.indexer.prefill.q8kv8 import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)
from tests.inference.msa_v1.indexer.prefill.q8kv8.cases import (
    HEAD_DIM,
    PAGE_SIZE,
    RealPrefillInputs,
    cumulative_lengths,
    require_sm100_device,
)
from tests.inference.msa_v1.indexer.prefill.q8kv8.reference import (
    assert_device_plan,
    assert_topk_structure,
    expected_lengths,
    sampled_rows,
)

pytestmark = pytest.mark.gpu
_SHARD_COUNT = 512
_CASES = load_prefill_test_cases("exhaustive")


def _make_structural_inputs(
    case: InferencePrefillCase,
    *,
    device: torch.device,
) -> RealPrefillInputs:
    """Create zero payloads with a non-identity physical page mapping."""

    q = torch.zeros(
        (case.total_q, 1, HEAD_DIM),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    k_cache = torch.zeros(
        (case.max_cols, 1, PAGE_SIZE, HEAD_DIM),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    logical_pages = torch.arange(case.max_cols, dtype=torch.int32, device=device)
    page_table = torch.stack(
        [
            (case.max_cols - 1 - logical_pages + batch_idx * 17) % case.max_cols
            for batch_idx in range(case.batch_size)
        ]
    ).contiguous()
    return RealPrefillInputs(
        q=q,
        k_cache=k_cache,
        page_table=page_table,
        cu_seqlens_q=torch.tensor(
            cumulative_lengths(case.query_lens),
            dtype=torch.int32,
            device=device,
        ),
        cu_seqlens_k=torch.tensor(
            cumulative_lengths(case.final_kv_lens),
            dtype=torch.int32,
            device=device,
        ),
    )


def _assert_sampled_zero_score_domain(
    case: InferencePrefillCase,
    scores: torch.Tensor,
) -> None:
    rows = torch.tensor(
        sampled_rows(case, limit=8),
        dtype=torch.int64,
        device=scores.device,
    )
    sampled = scores.index_select(0, rows)
    lengths = expected_lengths(case).to(device=scores.device).index_select(0, rows)
    columns = torch.arange(case.max_cols, device=scores.device).reshape(1, -1)
    expected_finite = columns < (lengths.reshape(-1, 1) - 1)
    assert torch.equal(torch.isfinite(sampled), expected_finite)
    assert bool(torch.all(sampled[expected_finite] == 0.0))


@pytest.mark.parametrize("shard_index", range(_SHARD_COUNT))
def test_all_10562_real_varlen_cases_execute_structurally(shard_index: int) -> None:
    """Execute every real shape while keeping each test shard below 30 seconds."""

    device = require_sm100_device()
    for case in _CASES[shard_index::_SHARD_COUNT]:
        inputs = _make_structural_inputs(case, device=device)
        wrapper = BatchPrefillIndexerWithPagedKVCacheWrapper()
        wrapper.plan(
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            inputs.page_table,
            total_q=case.total_q,
            max_seqlen_q=case.max_query_len,
            max_seqlen_k=case.max_final_kv,
        )
        state = wrapper._proxy_score.plan_state
        assert_device_plan(case, state)
        state.proxy_scores.fill_(float("nan"))
        state.topk_indices.fill_(-777)
        result = wrapper.run(inputs.q, inputs.k_cache)
        torch.cuda.synchronize()

        assert result.data_ptr() == state.topk_indices.data_ptr()
        _assert_sampled_zero_score_domain(case, state.proxy_scores[0])
        assert_topk_structure(state.topk_indices, state.num_valid_pages)
