"""Correctness coverage for Q8KV4 paged sparse prefill."""

from __future__ import annotations

import logging
import os
import threading
import time

import pytest
import torch

from inference.msa_v1.attention.prefill.q8kv4 import (
    BatchPrefillWithPagedKVCacheWrapper,
)
from tests.inference.msa_v1.attention.prefill.q8kv4.reference import (
    paged_sparse_attention_reference,
)


def _make_topk(
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
    *,
    device: torch.device,
    num_kv_heads: int = 4,
) -> torch.Tensor:
    total_q = sum(q_lens)
    topk = torch.full((num_kv_heads, total_q, 16), -1, dtype=torch.int32, device=device)
    q_offset = 0
    for batch, (q_len, k_len) in enumerate(zip(q_lens, k_lens)):
        del batch
        for q_idx in range(q_len):
            local_page = (k_len - q_len + q_idx) // 128
            pages = list(range(local_page))
            for head in range(num_kv_heads):
                shift = (head + q_idx) % max(len(pages), 1)
                permuted = pages[shift:] + pages[:shift]
                if (head + q_idx) & 1:
                    permuted.reverse()
                permuted.append(local_page)
                topk[head, q_offset + q_idx, : len(permuted)] = torch.tensor(
                    permuted,
                    dtype=torch.int32,
                    device=device,
                )
        q_offset += q_len
    return topk


@pytest.mark.parametrize("num_kv_heads", (4, 1), ids=("tp1", "tp4"))
def test_paged_chunk_prefill_matches_reference(num_kv_heads: int) -> None:
    """Cover both head counts with varlen, partial, and unordered history pages."""

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260829)
    q_lens = (8, 9)
    k_lens = (131, 260)
    total_q = sum(q_lens)
    num_q_heads = num_kv_heads * 16
    physical_pages = 7

    q = (
        torch.randn(
            (total_q, num_q_heads, 128),
            generator=generator,
            device=device,
        )
        * 0.2
    ).to(torch.float8_e4m3fn)
    packed_k = torch.randint(
        0,
        256,
        (physical_pages, num_kv_heads, 128, 64),
        dtype=torch.uint8,
        generator=generator,
        device=device,
    )
    packed_v = torch.randint(
        0,
        256,
        packed_k.shape,
        dtype=torch.uint8,
        generator=generator,
        device=device,
    )
    k_scale = (
        torch.rand(
            (physical_pages, num_kv_heads, 128, 8),
            generator=generator,
            device=device,
        )
        * 0.125
        + 0.0625
    ).to(torch.float8_e4m3fn)
    v_scale = (
        torch.rand(
            k_scale.shape,
            generator=generator,
            device=device,
        )
        * 0.125
        + 0.0625
    ).to(torch.float8_e4m3fn)
    page_table = torch.tensor(
        ((5, 1, 6), (4, 0, 3)),
        dtype=torch.int32,
        device=device,
    )
    topk = _make_topk(q_lens, k_lens, device=device, num_kv_heads=num_kv_heads)
    cu_seqlens_q = torch.tensor(
        (0, q_lens[0], total_q),
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_k = torch.tensor(
        (0, k_lens[0], sum(k_lens)),
        dtype=torch.int32,
        device=device,
    )

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
        (packed_k, packed_v),
        kv_cache_sf=(k_scale, v_scale),
        return_lse=True,
    )
    torch.cuda.synchronize()
    first_out, first_lse = actual.clone(), actual_lse.clone()
    for _ in range(2):
        torch.cuda.synchronize()
        started = time.perf_counter()
        watchdog = threading.Timer(30.0, lambda: os._exit(124))
        watchdog.start()
        try:
            actual, actual_lse = wrapper.run(
                q,
                (packed_k, packed_v),
                kv_cache_sf=(k_scale, v_scale),
                return_lse=True,
            )
            torch.cuda.synchronize()
        finally:
            watchdog.cancel()
        elapsed = time.perf_counter() - started
        logging.getLogger(__name__).info("Q8KV4 repeat ran in %.3fms", elapsed * 1e3)
        assert elapsed < 30.0
        assert torch.equal(actual, first_out)
        assert torch.equal(actual_lse, first_lse)
    expected, expected_lse = paged_sparse_attention_reference(
        q,
        packed_k,
        packed_v,
        k_scale,
        v_scale,
        page_table,
        topk,
        q_lens,
        k_lens,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual.float(),
        expected,
        atol=2e-3,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        actual_lse,
        expected_lse,
        atol=2e-5,
        rtol=2e-5,
    )


def test_single_token_reads_only_causal_prefix() -> None:
    """A one-token chunk at KV length one must return the first V token."""

    device = torch.device("cuda")
    topk = torch.full((4, 1, 16), -1, dtype=torch.int32, device=device)
    topk[:, :, 0] = 0
    cu_seqlens = torch.tensor((0, 1), dtype=torch.int32, device=device)
    page_table = torch.tensor(((0,),), dtype=torch.int32, device=device)
    q = torch.zeros((1, 64, 128), dtype=torch.float8_e4m3fn, device=device)
    packed_k = torch.zeros((1, 4, 128, 64), dtype=torch.uint8, device=device)
    packed_v = torch.full_like(packed_k, 0x22)
    scale = torch.ones((1, 4, 128, 8), dtype=torch.float8_e4m3fn, device=device)

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
    output, lse = wrapper.run(
        q,
        (packed_k, packed_v),
        kv_cache_sf=(scale, scale),
        return_lse=True,
    )
    torch.cuda.synchronize()
    assert torch.all(output == 1)
    assert torch.all(lse == 0)
