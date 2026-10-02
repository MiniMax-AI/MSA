"""Public API contract tests for the external FlashInfer adapters."""

from __future__ import annotations

import inspect

import pytest
import torch

from inference.msa_v1.attention.decode import bf16, q8kv8


@pytest.mark.parametrize("package", (bf16, q8kv8))
def test_public_package_exports_only_wrapper(package) -> None:
    assert package.__all__ == ["BatchDecodeWithPagedKVCacheWrapper"]


@pytest.mark.parametrize("package", (bf16, q8kv8))
def test_plan_and_run_signatures_use_canonical_names(package) -> None:
    wrapper = package.BatchDecodeWithPagedKVCacheWrapper
    assert tuple(inspect.signature(wrapper.plan).parameters) == (
        "self",
        "topk_indices",
        "page_table",
        "seq_lens",
        "q_len_per_req",
        "num_q_heads",
        "num_kv_heads",
        "sm_scale",
    )
    assert tuple(inspect.signature(wrapper.run).parameters) == (
        "self",
        "q",
        "paged_kv_cache",
        "out",
    )


@pytest.mark.parametrize("package", (bf16, q8kv8))
def test_run_requires_plan_without_importing_flashinfer(package) -> None:
    for enable_pdl in (False, True):
        wrapper = package.BatchDecodeWithPagedKVCacheWrapper(enable_pdl=enable_pdl)
        with pytest.raises(RuntimeError, match=r"plan\(\) must be called"):
            wrapper.run(torch.empty(0), (torch.empty(0), torch.empty(0)))
    with pytest.raises(TypeError, match="enable_pdl must be a bool"):
        package.BatchDecodeWithPagedKVCacheWrapper(enable_pdl=1)
