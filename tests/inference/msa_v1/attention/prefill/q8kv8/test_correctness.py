"""Correctness coverage for native-E4M3 paged sparse prefill."""

from __future__ import annotations

import torch

from inference.msa_v1.attention.prefill.q8kv8 import (
    BatchPrefillWithPagedKVCacheWrapper,
)
from tests.inference.msa_v1.attention.prefill.q8kv8.reference import (
    paged_sparse_attention_reference,
)
from tests.inference.msa_v1.attention.prefill.q8kv8.utils import (
    make_cu_seqlens,
    make_shuffled_page_table,
    make_topk,
)


def test_varlen_chunk_prefill_matches_fp32_reference() -> None:
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
    ).to(torch.float8_e4m3fn)
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
    ).to(torch.float8_e4m3fn)
    v_cache = (
        torch.randn(
            k_cache.shape,
            generator=generator,
            device=device,
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
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
                    right - left != 1 for left, right in zip(history, history[1:])
                )
        q_offset += q_len
    assert saw_unordered_history
    assert saw_noncontiguous_history

    wrapper = BatchPrefillWithPagedKVCacheWrapper()
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
    actual, actual_lse = wrapper.run(
        q,
        (k_cache, v_cache),
        return_lse=True,
    )
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
    torch.testing.assert_close(actual_fp32, expected, atol=3e-2, rtol=1e-1)
    torch.testing.assert_close(actual_lse, expected_lse, atol=3e-3, rtol=3e-3)
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
