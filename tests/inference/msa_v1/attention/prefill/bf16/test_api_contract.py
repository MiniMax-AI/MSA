"""Public API and layout contract tests for BF16 paged sparse prefill."""

import inspect

import pytest
import torch

from inference.msa_v1.attention.prefill import bf16 as bf16_package
from inference.msa_v1.attention.prefill.bf16 import (
    BatchPrefillWithPagedKVCacheWrapper,
)

pytestmark = pytest.mark.gpu


def _single_page_inputs() -> tuple[torch.Tensor, ...]:
    topk = torch.full((4, 1, 16), -1, dtype=torch.int32, device="cuda")
    topk[:, :, 0] = 0
    cu_seqlens = torch.tensor((0, 1), dtype=torch.int32, device="cuda")
    page_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    q = torch.zeros((1, 64, 128), dtype=torch.bfloat16, device="cuda")
    slots = torch.zeros((128, 4, 128), dtype=torch.bfloat16, device="cuda")
    k_cache = slots.view(1, 128, 4, 128).permute(0, 2, 1, 3)
    v_cache = torch.ones_like(slots).view(1, 128, 4, 128).permute(0, 2, 1, 3)
    return topk, cu_seqlens, page_table, q, k_cache, v_cache


def test_public_package_exports_only_wrapper() -> None:
    assert bf16_package.__all__ == ["BatchPrefillWithPagedKVCacheWrapper"]


def test_wrapper_uses_canonical_plan_run_contract() -> None:
    plan_parameters = inspect.signature(
        BatchPrefillWithPagedKVCacheWrapper.plan
    ).parameters
    run_parameters = inspect.signature(
        BatchPrefillWithPagedKVCacheWrapper.run
    ).parameters
    assert "sm_scale" in plan_parameters
    assert "softmax_scale" not in plan_parameters
    assert "seqused_k" not in plan_parameters
    assert "paged_kv_cache" in run_parameters


def test_strided_paged_kv_runs_without_materialization() -> None:
    topk, cu_seqlens, page_table, q, k_cache, v_cache = _single_page_inputs()
    assert not k_cache.is_contiguous() and not v_cache.is_contiguous()
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    wrapper.plan(
        topk,
        cu_seqlens,
        cu_seqlens,
        page_table,
        total_k=1,
        total_rows=1,
        max_seqlen_q=1,
        max_seqlen_k=1,
    )
    out, lse = wrapper.run(q, (k_cache, v_cache), return_lse=True)
    torch.cuda.synchronize()
    assert torch.all(out == 1)
    assert torch.all(lse == 0)


def test_plan_supports_tp4_head_configuration() -> None:
    topk = torch.full((1, 1, 16), -1, dtype=torch.int32, device="cuda")
    topk[:, :, 0] = 0
    cu_seqlens = torch.tensor((0, 1), dtype=torch.int32, device="cuda")
    page_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    wrapper.plan(
        topk,
        cu_seqlens,
        cu_seqlens,
        page_table,
        num_q_heads=16,
        num_kv_heads=1,
        total_k=1,
        total_rows=1,
        max_seqlen_q=1,
        max_seqlen_k=1,
    )
    q = torch.zeros((1, 16, 128), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.zeros((1, 1, 128, 128), dtype=torch.bfloat16, device="cuda")
    v_cache = torch.ones_like(k_cache)
    out = wrapper.run(q, (k_cache, v_cache))
    torch.cuda.synchronize()
    assert out.shape == (1, 16, 128)
    assert torch.all(out == 1)


def test_plan_supports_only_serving_gqa_ratios() -> None:
    topk = torch.full((1, 1, 16), -1, dtype=torch.int32, device="cuda")
    topk[:, :, 0] = 0
    cu_seqlens = torch.tensor((0, 1), dtype=torch.int32, device="cuda")
    page_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    wrapper.plan(
        topk,
        cu_seqlens,
        cu_seqlens,
        page_table,
        num_q_heads=8,
        num_kv_heads=1,
        total_k=1,
        total_rows=1,
        max_seqlen_q=1,
        max_seqlen_k=1,
    )
    q = torch.zeros((1, 8, 128), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.zeros((1, 1, 128, 128), dtype=torch.bfloat16, device="cuda")
    v_cache = torch.ones_like(k_cache)
    out = wrapper.run(q, (k_cache, v_cache))
    torch.cuda.synchronize()
    assert torch.all(out == 1)
    with pytest.raises(ValueError, match="GQA group sizes"):
        wrapper.plan(
            topk,
            cu_seqlens,
            cu_seqlens,
            page_table,
            num_q_heads=4,
            num_kv_heads=1,
            total_k=1,
            total_rows=1,
            max_seqlen_q=1,
            max_seqlen_k=1,
        )


def test_run_rejects_misaligned_q() -> None:
    topk, cu_seqlens, page_table, _, k_cache, v_cache = _single_page_inputs()
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    wrapper.plan(
        topk,
        cu_seqlens,
        cu_seqlens,
        page_table,
        total_k=1,
        total_rows=1,
        max_seqlen_q=1,
        max_seqlen_k=1,
    )
    storage = torch.empty(1 + 64 * 128, dtype=torch.bfloat16, device="cuda")
    misaligned_q = storage[1:].view(1, 64, 128)
    assert misaligned_q.is_contiguous()
    with pytest.raises(ValueError, match="16-byte aligned"):
        wrapper.run(misaligned_q, (k_cache, v_cache))
