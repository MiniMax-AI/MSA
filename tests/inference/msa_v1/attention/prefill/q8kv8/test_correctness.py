"""Correctness coverage for native-E4M3 paged sparse prefill."""

from __future__ import annotations

from functools import partial
from itertools import pairwise

import pytest
import torch

from inference.msa_v1 import aot_cache
from inference.msa_v1.attention.prefill import bf16
from inference.msa_v1.attention.prefill._common import atten_fwd_sm100
from inference.msa_v1.attention.prefill._common.atten_fwd_sm100 import _COMPILE_CACHE
from inference.msa_v1.attention.prefill.q8kv8 import (
    BatchPrefillWithPagedKVCacheWrapper,
)
from tests.inference.msa_v1.attention.decode.runtime import run_cuda
from tests.inference.msa_v1.attention.prefill.q8kv8.reference import (
    paged_sparse_attention_reference,
)
from tests.inference.msa_v1.attention.prefill.q8kv8.utils import (
    make_cu_seqlens,
    make_shuffled_page_table,
    make_topk,
)
from tests.inference.msa_v1.cache_layout import CACHE_LAYOUTS, cache_view


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float8_e4m3fn))
@pytest.mark.parametrize(
    "first_layout", ("contiguous", "singleton_zero", "singleton_unaligned")
)
def test_singleton_cache_first_compile_and_graph_reuse(
    dtype, first_layout, monkeypatch
):
    monkeypatch.setattr(aot_cache, "_AOT_DISABLE", True)
    _COMPILE_CACHE.clear()
    original_compile = atten_fwd_sm100._compile_kernel

    def checked_compile(*args, **kwargs):
        compiled = original_compile(*args, **kwargs)
        if torch.cuda.is_current_stream_capturing():
            return compiled
        return lambda *values: run_cuda("prefill singleton", partial(compiled, *values))

    monkeypatch.setattr(atten_fwd_sm100, "_compile_kernel", checked_compile)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(8127)
    q_lens, k_lens = (3, 5), (262, 268)
    cu_q, cu_k = make_cu_seqlens(q_lens, device), make_cu_seqlens(k_lens, device)
    base_q = (torch.randn(8, 64, 128, generator=generator, device=device) * 0.25).to(
        dtype
    )
    base_k = (
        torch.randn(6, 4, 128, 128, generator=generator, device=device) * 0.25
    ).to(dtype)
    base_v = (
        torch.randn(6, 4, 128, 128, generator=generator, device=device) * 0.25
    ).to(dtype)
    original_table = torch.tensor(
        [[5, 0, 3], [2, 4, 1]], dtype=torch.int32, device=device
    )
    wrapper = (
        bf16.BatchPrefillWithPagedKVCacheWrapper()
        if dtype == torch.bfloat16
        else BatchPrefillWithPagedKVCacheWrapper()
    )
    keys = None
    layouts = (
        first_layout,
        "contiguous",
        "singleton_zero",
        "singleton_unaligned",
        "combined",
    )
    for layout in layouts:
        singleton = layout.startswith("singleton")
        heads = 1 if singleton else 4
        source_k = base_k[:1, :1] if singleton else base_k
        source_v = base_v[:1, :1] if singleton else base_v
        k_cache, v_cache = cache_view(source_k, layout), cache_view(source_v, layout)
        q = base_q[:, : heads * 16].clone()
        table = torch.zeros_like(original_table) if singleton else original_table
        topk = torch.full((heads, 8, 16), -1, dtype=torch.int32, device=device)
        topk[:, :, :3] = torch.tensor([1, 0, 2], dtype=torch.int32, device=device)
        wrapper.plan(
            topk,
            cu_q,
            cu_k,
            table,
            num_q_heads=heads * 16,
            num_kv_heads=heads,
            total_k=sum(k_lens),
            total_rows=6,
            max_seqlen_q=5,
            max_seqlen_k=268,
        )
        out, lse = wrapper.run(q, (k_cache, v_cache), return_lse=True)
        expected, expected_lse = paged_sparse_attention_reference(
            q, k_cache, v_cache, table, topk, q_lens, k_lens
        )
        tolerance = 1e-3 if dtype == torch.bfloat16 else 3e-3
        output_rtol = 3e-2 if dtype == torch.bfloat16 else 1e-1
        torch.testing.assert_close(out.float(), expected, atol=3e-2, rtol=output_rtol)
        torch.testing.assert_close(lse, expected_lse, atol=tolerance, rtol=tolerance)
        assert torch.isfinite(out).all() and torch.isfinite(lse).all()
        if keys is not None:
            assert set(_COMPILE_CACHE) == keys
        keys = set(_COMPILE_CACHE)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper.run(q, (k_cache, v_cache), out=out, lse=lse, return_lse=True)
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        run_cuda("prefill singleton graph", graph.replay)
        torch.testing.assert_close(out.float(), expected, atol=3e-2, rtol=output_rtol)
        torch.testing.assert_close(lse, expected_lse, atol=tolerance, rtol=tolerance)
        assert set(_COMPILE_CACHE) == keys


@pytest.mark.parametrize("layout", CACHE_LAYOUTS)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float8_e4m3fn))
def test_varlen_chunk_prefill_matches_fp32_reference(layout, dtype) -> None:
    """Cover more than 1000 row/head/page/causal boundary samples."""

    device = torch.device("cuda")
    q_lens = (17, 19, 23, 29)
    k_lens = (2055, 2307, 1931, 2693)
    total_q = sum(q_lens)
    max_pages = (max(k_lens) + 127) // 128
    assert total_q * 64 >= 1000

    generator = torch.Generator(device=device).manual_seed(20260829)
    q = (
        torch.randn(
            (total_q, 64, 128),
            generator=generator,
            device=device,
        )
        * 0.25
    ).to(dtype)
    page_table = make_shuffled_page_table(
        len(q_lens),
        max_pages,
        device=device,
        seed=20260830,
    )
    physical_pages = page_table.numel()
    k_cache = (
        torch.randn(
            (physical_pages, 4, 128, 128),
            generator=generator,
            device=device,
        )
        * 0.25
    ).to(dtype)
    v_cache = (
        torch.randn(
            k_cache.shape,
            generator=generator,
            device=device,
        )
        * 0.25
    ).to(dtype)
    topk = make_topk(q_lens, k_lens, device=device)
    cu_seqlens_q = make_cu_seqlens(q_lens, device)
    cu_seqlens_k = make_cu_seqlens(k_lens, device)
    assert not torch.equal(
        page_table.cpu(),
        torch.arange(page_table.numel(), dtype=torch.int32).reshape_as(page_table),
    )
    topk_host = topk.cpu()
    q_offset = 0
    saw_unordered_history = False
    saw_noncontiguous_history = False
    for q_len, k_len in zip(q_lens, k_lens):
        for q_idx in range(q_len):
            current_page = (k_len - q_len + q_idx) // 128
            for kv_head in range(4):
                row = topk_host[kv_head, q_offset + q_idx]
                valid = row[row >= 0].tolist()
                assert valid[-1] == current_page
                assert torch.all(row[len(valid) :] == -1)
                history = valid[:-1]
                saw_unordered_history |= history != sorted(history)
                saw_noncontiguous_history |= any(
                    right - left != 1 for left, right in pairwise(history)
                )
        q_offset += q_len
    assert saw_unordered_history
    assert saw_noncontiguous_history

    wrapper = (
        bf16.BatchPrefillWithPagedKVCacheWrapper()
        if dtype == torch.bfloat16
        else BatchPrefillWithPagedKVCacheWrapper()
    )
    wrapper.plan(
        topk,
        cu_seqlens_q,
        cu_seqlens_k,
        page_table,
        total_k=sum(k_lens),
        total_rows=sum((length + 127) // 128 for length in k_lens),
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
    )
    wrapper.run(q, (k_cache, v_cache), return_lse=True)
    cache_keys = set(_COMPILE_CACHE)
    k_cache, v_cache = cache_view(k_cache, layout), cache_view(v_cache, layout)
    actual, actual_lse = wrapper.run(
        q,
        (k_cache, v_cache),
        return_lse=True,
    )
    assert set(_COMPILE_CACHE) == cache_keys
    eager_out, eager_lse = actual.clone(), actual_lse.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, (k_cache, v_cache), out=actual, lse=actual_lse, return_lse=True)
    actual.fill_(float("nan"))
    actual_lse.fill_(float("nan"))
    graph.replay()
    expected, expected_lse = paged_sparse_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        topk,
        q_lens,
        k_lens,
    )
    torch.cuda.synchronize()

    actual_fp32 = actual.float()
    torch.testing.assert_close(
        actual_fp32, expected, atol=3e-2, rtol=3e-2 if dtype == torch.bfloat16 else 1e-1
    )
    lse_tolerance = 1e-3 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(
        eager_out.float(),
        expected,
        atol=3e-2,
        rtol=3e-2 if dtype == torch.bfloat16 else 1e-1,
    )
    torch.testing.assert_close(
        eager_lse, expected_lse, atol=lse_tolerance, rtol=lse_tolerance
    )
    torch.testing.assert_close(
        actual_lse, expected_lse, atol=lse_tolerance, rtol=lse_tolerance
    )
    relative_l2 = torch.linalg.vector_norm(
        actual_fp32 - expected
    ) / torch.linalg.vector_norm(expected)
    assert float(relative_l2) < 5e-2


def test_partial_page_bottom_right_causal_mask() -> None:
    """A one-token sequence may read only the first value of its partial page."""

    device = torch.device("cuda")
    topk = torch.full((4, 1, 16), -1, dtype=torch.int32, device=device)
    topk[:, 0, 0] = 0
    cu_seqlens = torch.tensor((0, 1), dtype=torch.int32, device=device)
    page_table = torch.tensor(((0,),), dtype=torch.int32, device=device)
    q = torch.zeros((1, 64, 128), dtype=torch.float8_e4m3fn, device=device)
    k_cache = torch.zeros((1, 4, 128, 128), dtype=torch.float8_e4m3fn, device=device)
    v_cache = torch.zeros_like(k_cache)
    v_cache[:, :, 0] = 1
    v_cache[:, :, 1:] = 8

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
    output, lse = wrapper.run(q, (k_cache, v_cache), return_lse=True)
    torch.cuda.synchronize()
    assert torch.all(output == 1)
    assert torch.all(lse == 0)
