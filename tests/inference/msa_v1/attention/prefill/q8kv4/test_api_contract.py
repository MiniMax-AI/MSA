"""Public API and CUDA Graph contract tests for Q8KV4 paged prefill."""

from __future__ import annotations

import inspect

import pytest
import torch

from inference.msa_v1.attention.prefill import q8kv4 as q8kv4_package
from inference.msa_v1.attention.prefill._common import metadata
from inference.msa_v1.attention.prefill._common.prepare_k2q_csr import (
    SparseK2qCsrBuilderSm100,
)
from inference.msa_v1.attention.prefill.q8kv4 import (
    BatchPrefillWithPagedKVCacheWrapper,
    jit,
)


def _make_single_page_inputs(
    num_kv_heads: int = 4,
    batch: int = 1,
    length: int = 1,
) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    topk = torch.full(
        (num_kv_heads, batch * length, 16), -1, dtype=torch.int32, device=device
    )
    topk[:, :, 0] = 0
    cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=device) * length
    page_table = torch.arange(batch, dtype=torch.int32, device=device).reshape(batch, 1)
    q = torch.zeros(
        (batch * length, num_kv_heads * 16, 128),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    packed_k = torch.zeros(
        (batch, num_kv_heads, 128, 64), dtype=torch.uint8, device=device
    )
    packed_v = torch.full_like(packed_k, 0x22)
    scale = torch.ones(
        (batch, num_kv_heads, 128, 8), dtype=torch.float8_e4m3fn, device=device
    )
    return topk, cu_seqlens, page_table, q, packed_k, packed_v, scale


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
        total_k=topk.shape[1],
        total_rows=page_table.shape[0],
        max_seqlen_q=topk.shape[1] // page_table.shape[0],
        max_seqlen_k=topk.shape[1] // page_table.shape[0],
    )


def test_public_package_exports_only_wrapper() -> None:
    assert q8kv4_package.__all__ == ["BatchPrefillWithPagedKVCacheWrapper"]


def test_prepare_does_not_retain_layout_or_workspace_caches() -> None:
    assert "_BUILDER" not in vars(metadata)
    assert vars(SparseK2qCsrBuilderSm100()) == {}


def test_plan_uses_flashinfer_scale_name() -> None:
    signature = inspect.signature(BatchPrefillWithPagedKVCacheWrapper.plan)
    assert "sm_scale" in signature.parameters
    assert "softmax_scale" not in signature.parameters


def test_run_requires_plan() -> None:
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called"):
        wrapper.run(  # type: ignore[arg-type]
            torch.empty(0),
            (torch.empty(0), torch.empty(0)),
            kv_cache_sf=(torch.empty(0), torch.empty(0)),
        )


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


@pytest.mark.parametrize("num_kv_heads", (4, 1), ids=("tp1", "tp4"))
def test_run_uses_preallocated_output_and_lse(num_kv_heads: int) -> None:
    topk, cu_seqlens, page_table, q, packed_k, packed_v, scale = (
        _make_single_page_inputs(num_kv_heads)
    )
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    _plan_single_page(wrapper, topk, cu_seqlens, page_table)
    out = torch.empty(q.shape, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device="cuda")
    actual_out, actual_lse = wrapper.run(
        q,
        (packed_k, packed_v),
        kv_cache_sf=(scale, scale),
        out=out,
        lse=lse,
        return_lse=True,
    )
    torch.cuda.synchronize()
    assert actual_out.data_ptr() == out.data_ptr()
    assert actual_lse.data_ptr() == lse.data_ptr()
    assert torch.all(out == 1)
    assert torch.all(lse == 0)


@pytest.mark.parametrize("num_kv_heads", (4, 1), ids=("tp1", "tp4"))
def test_cuda_graph_capture_and_replay(num_kv_heads: int) -> None:
    topk, cu_seqlens, page_table, q, packed_k, packed_v, scale = (
        _make_single_page_inputs(num_kv_heads)
    )
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    _plan_single_page(wrapper, topk, cu_seqlens, page_table)
    out = torch.empty(q.shape, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device="cuda")

    wrapper.run(
        q,
        (packed_k, packed_v),
        kv_cache_sf=(scale, scale),
        out=out,
        lse=lse,
        return_lse=True,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out, captured_lse = wrapper.run(
            q,
            (packed_k, packed_v),
            kv_cache_sf=(scale, scale),
            out=out,
            lse=lse,
            return_lse=True,
        )
    assert captured_out.data_ptr() == out.data_ptr()
    assert captured_lse.data_ptr() == lse.data_ptr()

    graph.replay()
    torch.cuda.synchronize()
    first_out = out.clone()
    first_lse = lse.clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, first_out, atol=0, rtol=0)
    torch.testing.assert_close(lse, first_lse, atol=0, rtol=0)


def test_plan_is_rejected_during_capture() -> None:
    topk, cu_seqlens, page_table, q, packed_k, packed_v, scale = (
        _make_single_page_inputs()
    )
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    _plan_single_page(wrapper, topk, cu_seqlens, page_table)
    out = torch.empty((1, 64, 128), dtype=torch.bfloat16, device="cuda")
    wrapper.run(
        q,
        (packed_k, packed_v),
        kv_cache_sf=(scale, scale),
        out=out,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with (
        pytest.warns(UserWarning, match="The CUDA Graph is empty"),
        pytest.raises(RuntimeError, match=r"plan\(\) must be called outside"),
    ):
        with torch.cuda.graph(graph):
            _plan_single_page(wrapper, topk, cu_seqlens, page_table)


def test_jit_spec_contains_only_compile_time_configuration() -> None:
    spec = jit.gen_jit_spec()

    assert tuple(spec.__dataclass_fields__) == (
        "target_arch",
        "variant_name",
        "q_heads_per_kv",
        "head_dim",
        "page_size",
        "topk",
        "q_stages",
        "score_stages",
    )
    assert spec.variant_name == "prefill_attention_q8kv4"
    assert spec.topk == 16


def test_runtime_heads_and_shapes_reuse_extension() -> None:
    device = torch.device("cuda")
    extension = jit.load_extension(device)
    cache_before = jit._load_extension_for_arch.cache_info()
    for num_kv_heads, batch, length in ((4, 1, 1), (1, 2, 3), (4, 1, 8)):
        topk, cu_seqlens, page_table, q, packed_k, packed_v, scale = (
            _make_single_page_inputs(num_kv_heads, batch, length)
        )
        wrapper = BatchPrefillWithPagedKVCacheWrapper()
        _plan_single_page(wrapper, topk, cu_seqlens, page_table)
        actual = wrapper.run(q, (packed_k, packed_v), kv_cache_sf=(scale, scale))
        torch.testing.assert_close(actual, torch.ones_like(actual), atol=0, rtol=0)
        assert jit.load_extension(device) is extension
    assert jit._load_extension_for_arch.cache_info().misses == cache_before.misses


def test_tp4_plan_rejects_mismatched_heads() -> None:
    topk, cu_seqlens, page_table, q, packed_k, packed_v, scale = (
        _make_single_page_inputs(1)
    )
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    _plan_single_page(wrapper, topk, cu_seqlens, page_table)
    with pytest.raises(ValueError, match="q must have shape"):
        wrapper.run(q.repeat(1, 4, 1), (packed_k, packed_v), kv_cache_sf=(scale, scale))
    with pytest.raises(ValueError, match="k_cache must have shape"):
        wrapper.run(
            q, (packed_k.repeat(1, 4, 1, 1), packed_v), kv_cache_sf=(scale, scale)
        )


def test_lazy_jit_uses_tensor_device_over_offline_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:5")
    monkeypatch.setenv("MM_SPARSE_TARGET_ARCH", "100a")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda requested: (10, 3) if requested == device else (10, 0),
    )
    jit._target_arch.cache_clear()
    try:
        assert jit._target_arch(device) == "103a"
    finally:
        jit._target_arch.cache_clear()
