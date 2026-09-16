"""Public API and CUDA Graph tests for Q8KV8 paged sparse prefill."""

from __future__ import annotations

import inspect

import pytest
import torch

from inference.msa_v1.attention.prefill import q8kv8 as q8kv8_package
from inference.msa_v1.attention.prefill.q8kv8 import (
    BatchPrefillWithPagedKVCacheWrapper,
)


def _make_single_page_inputs() -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    topk = torch.full((4, 1, 16), -1, dtype=torch.int32, device=device)
    topk[:, 0, 0] = 0
    cu_seqlens = torch.tensor((0, 1), dtype=torch.int32, device=device)
    page_table = torch.tensor(((0,),), dtype=torch.int32, device=device)
    q = torch.zeros((1, 64, 128), dtype=torch.float8_e4m3fn, device=device)
    k_cache = torch.zeros((1, 4, 128, 128), dtype=torch.float8_e4m3fn, device=device)
    v_cache = torch.ones_like(k_cache)
    return topk, cu_seqlens, page_table, q, k_cache, v_cache


def _plan_single_page(
    wrapper: BatchPrefillWithPagedKVCacheWrapper,
    topk: torch.Tensor,
    cu_seqlens: torch.Tensor,
    page_table: torch.Tensor,
) -> None:
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


def test_public_package_exports_only_wrapper() -> None:
    assert q8kv8_package.__all__ == ["BatchPrefillWithPagedKVCacheWrapper"]


def test_run_signature_has_no_scale_metadata() -> None:
    signature = inspect.signature(BatchPrefillWithPagedKVCacheWrapper.run)
    assert "kv_cache_sf" not in signature.parameters
    assert "kv_lengths" not in signature.parameters
    assert "seqused_k" not in signature.parameters


def test_plan_uses_flashinfer_scale_name() -> None:
    signature = inspect.signature(BatchPrefillWithPagedKVCacheWrapper.plan)
    assert "sm_scale" in signature.parameters
    assert "softmax_scale" not in signature.parameters


def test_run_requires_plan() -> None:
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called"):
        wrapper.run(torch.empty(0), (torch.empty(0), torch.empty(0)))


def test_plan_rejects_noncausal_mode() -> None:
    topk, cu_seqlens, page_table, *_ = _make_single_page_inputs()
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    with pytest.raises(NotImplementedError, match="only causal"):
        wrapper.plan(
            topk,
            cu_seqlens,
            cu_seqlens,
            page_table,
            total_k=1,
            total_rows=1,
            max_seqlen_q=1,
            max_seqlen_k=1,
            causal=False,
        )


def test_plan_validates_metadata_contract_before_prepare() -> None:
    topk, cu_seqlens, page_table, *_ = _make_single_page_inputs()
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    with pytest.raises(TypeError, match="topk_indices must be torch.int32"):
        wrapper.plan(
            topk.to(torch.int64),
            cu_seqlens,
            cu_seqlens,
            page_table,
            total_k=1,
            total_rows=1,
            max_seqlen_q=1,
            max_seqlen_k=1,
        )
    with pytest.raises(ValueError, match="page_table capacity"):
        wrapper.plan(
            topk,
            cu_seqlens,
            cu_seqlens,
            page_table,
            total_k=129,
            total_rows=2,
            max_seqlen_q=1,
            max_seqlen_k=129,
        )


def test_run_validates_dtype_shape_device_and_contiguity() -> None:
    topk, cu_seqlens, page_table, q, k_cache, v_cache = _make_single_page_inputs()
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    _plan_single_page(wrapper, topk, cu_seqlens, page_table)

    with pytest.raises(TypeError, match="q must be torch.float8_e4m3fn"):
        wrapper.run(q.to(torch.bfloat16), (k_cache, v_cache))
    with pytest.raises(ValueError, match="q must have shape"):
        wrapper.run(q[:, :63], (k_cache, v_cache))
    q_noncontiguous = torch.empty(
        (1, 128, 64), dtype=torch.float8_e4m3fn, device="cuda"
    ).transpose(1, 2)
    with pytest.raises(ValueError, match="q must be contiguous"):
        wrapper.run(q_noncontiguous, (k_cache, v_cache))
    with pytest.raises(ValueError, match="k_cache must have shape"):
        wrapper.run(q, (k_cache[:, :, :, :64].contiguous(), v_cache))
    with pytest.raises(TypeError, match="k_cache must be torch.float8_e4m3fn"):
        wrapper.run(q, (k_cache.to(torch.bfloat16), v_cache))
    if torch.cuda.device_count() > 1:
        k_other_device = torch.empty_like(k_cache, device="cuda:1")
        with pytest.raises(ValueError, match="k_cache must be on"):
            wrapper.run(q, (k_other_device, v_cache))


def test_plan_is_rejected_during_cuda_graph_capture() -> None:
    topk, cu_seqlens, page_table, *_ = _make_single_page_inputs()
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.warns(UserWarning, match="CUDA Graph is empty"),
        pytest.raises(RuntimeError, match=r"plan\(\) must be called outside"),
    ):
        with torch.cuda.graph(graph):
            _plan_single_page(wrapper, topk, cu_seqlens, page_table)


def test_run_uses_preallocated_output_and_lse() -> None:
    topk, cu_seqlens, page_table, q, k_cache, v_cache = _make_single_page_inputs()
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    _plan_single_page(wrapper, topk, cu_seqlens, page_table)
    out = torch.empty((1, 64, 128), dtype=torch.bfloat16, device="cuda")
    lse = torch.empty((1, 64), dtype=torch.float32, device="cuda")
    actual_out, actual_lse = wrapper.run(
        q,
        (k_cache, v_cache),
        out=out,
        lse=lse,
        return_lse=True,
    )
    torch.cuda.synchronize()
    assert actual_out.data_ptr() == out.data_ptr()
    assert actual_lse.data_ptr() == lse.data_ptr()
    assert torch.all(out == 1)
    assert torch.all(lse == 0)


@pytest.mark.parametrize(
    ("num_q_heads", "num_kv_heads"),
    ((32, 2), (16, 1), (8, 1)),
)
def test_run_supports_tensor_parallel_head_shapes(
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    device = torch.device("cuda")
    topk = torch.full((num_kv_heads, 1, 16), -1, dtype=torch.int32, device=device)
    topk[:, 0, 0] = 0
    cu_seqlens = torch.tensor((0, 1), dtype=torch.int32, device=device)
    page_table = torch.tensor(((0,),), dtype=torch.int32, device=device)
    q = torch.zeros((1, num_q_heads, 128), dtype=torch.float8_e4m3fn, device=device)
    k_cache = torch.zeros(
        (1, num_kv_heads, 128, 128),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    v_cache = torch.ones_like(k_cache)
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    wrapper.plan(
        topk,
        cu_seqlens,
        cu_seqlens,
        page_table,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        total_k=1,
        total_rows=1,
        max_seqlen_q=1,
        max_seqlen_k=1,
    )
    output = wrapper.run(q, (k_cache, v_cache))
    torch.cuda.synchronize()
    assert output.shape == (1, num_q_heads, 128)
    assert torch.all(output == 1)


def test_cuda_graph_capture_and_replay() -> None:
    topk, cu_seqlens, page_table, q, k_cache, v_cache = _make_single_page_inputs()
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    _plan_single_page(wrapper, topk, cu_seqlens, page_table)
    out = torch.empty((1, 64, 128), dtype=torch.bfloat16, device="cuda")
    lse = torch.empty((1, 64), dtype=torch.float32, device="cuda")
    wrapper.run(q, (k_cache, v_cache), out=out, lse=lse, return_lse=True)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out, captured_lse = wrapper.run(
            q,
            (k_cache, v_cache),
            out=out,
            lse=lse,
            return_lse=True,
        )
    assert captured_out.data_ptr() == out.data_ptr()
    assert captured_lse.data_ptr() == lse.data_ptr()

    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(out == 1)
    assert torch.all(lse == 0)
