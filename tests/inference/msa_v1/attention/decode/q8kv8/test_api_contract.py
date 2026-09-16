"""Public API contract tests for the external FlashInfer Q8K8 adapter."""

from __future__ import annotations

import inspect

import pytest
import torch

from inference.msa_v1.attention.decode import q8kv8
from inference.msa_v1.attention.decode.q8kv8 import (
    BatchDecodeWithPagedKVCacheWrapper,
)


def test_public_package_exports_only_wrapper() -> None:
    assert q8kv8.__all__ == ["BatchDecodeWithPagedKVCacheWrapper"]


def test_plan_and_run_signatures_use_canonical_names() -> None:
    assert tuple(
        inspect.signature(BatchDecodeWithPagedKVCacheWrapper.plan).parameters
    ) == (
        "self",
        "topk_indices",
        "page_table",
        "seq_lens",
        "q_len_per_req",
        "num_q_heads",
        "num_kv_heads",
        "sm_scale",
    )
    assert tuple(
        inspect.signature(BatchDecodeWithPagedKVCacheWrapper.run).parameters
    ) == ("self", "q", "paged_kv_cache", "out")


def test_run_requires_plan_without_importing_flashinfer() -> None:
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called"):
        wrapper.run(torch.empty(0), (torch.empty(0), torch.empty(0)))
