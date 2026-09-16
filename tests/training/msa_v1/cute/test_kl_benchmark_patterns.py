"""Contract tests for the KL pathology benchmark TopK generators."""

from __future__ import annotations

import pytest
import torch

from tests.training.msa_v1.cute.topk_patterns import (
    BLOCK_K,
    DEFAULT_PATTERNS,
    KV_HEADS,
    TOPK,
    build_topk,
)


QUERY_COUNT = 4096


@pytest.mark.parametrize("pattern", DEFAULT_PATTERNS)
def test_kl_pathology_topk_contract(pattern: str, cuda_device) -> None:
    visible = (
        torch.arange(QUERY_COUNT, device=cuda_device, dtype=torch.int64) // BLOCK_K
        + 256
    )
    topk = build_topk(pattern, visible, seed=1234)

    assert topk.shape == (KV_HEADS, QUERY_COUNT, TOPK)
    assert topk.dtype == torch.int32
    assert topk.is_contiguous()
    sorted_values = topk.sort(dim=-1).values
    assert not torch.any(sorted_values[..., 1:] == sorted_values[..., :-1])
    assert torch.all(topk >= 0)
    assert torch.all(topk < visible.reshape(1, -1, 1))

    if pattern.startswith("b0p99_local_"):
        local = visible - 1
        assert torch.all((topk == local.reshape(1, -1, 1)).any(dim=-1))
        block0_rate = (topk == 0).any(dim=-1).float().mean()
        assert abs(float(block0_rate) - 0.99) <= 1.0 / QUERY_COUNT
