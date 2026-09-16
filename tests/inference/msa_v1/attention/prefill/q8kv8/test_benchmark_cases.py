"""Tests for Q8KV8 real-varlen benchmark tensor construction."""

from __future__ import annotations

import pytest
import torch

from benchmarks.inference.msa_v1.attention.prefill.q8kv8.benchmark import (
    _make_case_tensors,
)
from benchmarks.inference.msa_v1.attention.prefill.q8kv8.cases import (
    PAGE_SIZE,
    real_prefill_cases,
    warmup_case,
)

pytestmark = pytest.mark.gpu


def test_real_benchmark_suites_match_q8kv4_selection() -> None:
    full = real_prefill_cases("full")
    smoke = real_prefill_cases("smoke")
    assert len(full) == 128
    assert len(smoke) == 18
    assert {case.batch for case in smoke} == set(range(1, 19))


def test_varlen_page_cache_and_topk_contract() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    case = warmup_case()
    tensors = _make_case_tensors(case, device)

    active_ids = []
    for batch_idx, final_kv_len in enumerate(case.final_kv_lens):
        page_count = (final_kv_len + PAGE_SIZE - 1) // PAGE_SIZE
        physical_ids = tensors["page_table"][batch_idx, :page_count].long()
        active_ids.extend(physical_ids.tolist())
    assert len(active_ids) == len(set(active_ids))
    assert bool(torch.isfinite(tensors["k_cache"].float()).all())

    query_positions = torch.cat(
        [
            torch.arange(query_len, dtype=torch.int64, device=device) + prefix_len
            for query_len, prefix_len in zip(
                case.query_lens, case.prefix_lens, strict=True
            )
        ]
    )
    local_page = torch.div(query_positions, PAGE_SIZE, rounding_mode="floor")
    valid_count = (local_page + 1).clamp(max=16)
    topk = tensors["topk"]
    assert torch.equal((topk >= 0).sum(dim=-1), valid_count.unsqueeze(0).expand(4, -1))
    tail = topk.gather(
        2,
        (valid_count - 1).reshape(1, -1, 1).expand(4, -1, -1),
    ).squeeze(-1)
    assert torch.equal(tail, local_page.to(torch.int32).expand(4, -1))
