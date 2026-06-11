# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import pytest
import torch


def _need_triton_sm12x():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, _ = torch.cuda.get_device_capability()
    if major != 12:
        pytest.skip("SM120/SM121 CUDA device is required")
    from fmha_sm12x._triton_sparse import triton_available

    if not triton_available():
        pytest.skip("Triton is required")


def _reference(q, k, v, block_indexes, *, cu_q, cu_k, num_kv_heads, page_size, causal, sm_scale, want_temp, temp):
    from fmha_sm12x._lse import run_lse
    from fmha_sm12x.api import fmha_sm12x, fmha_sm12x_plan

    qo_lens = cu_q[1:] - cu_q[:-1]
    kv_lens = cu_k[1:] - cu_k[:-1]
    plan = fmha_sm12x_plan(
        qo_lens, kv_lens, int(q.shape[1]), int(num_kv_heads), page_size=int(page_size), causal=bool(causal)
    )
    out, _ = fmha_sm12x(q, k, v, plan, kv_indices=None, kv_block_indexes=block_indexes, sm_scale=sm_scale)
    lse = run_lse(q, k, v, plan, kv_indices=None, kv_block_indexes=block_indexes, sm_scale=sm_scale)
    tlse = None
    if want_temp:
        tlse = run_lse(q, k, v, plan, kv_indices=None, kv_block_indexes=block_indexes, sm_scale=sm_scale / temp)
    return out, lse, tlse


@pytest.mark.parametrize(
    "batches,h_ratio,num_kv_heads,topk,page_size,causal,temp",
    [
        ([(2, 256), (3, 384)], 16, 1, 2, 128, True, 1.0),
        ([(4, 128)], 4, 2, 4, 64, True, 2.0),
        ([(1, 64), (2, 96)], 1, 3, 2, 16, False, 1.0),
        ([(3, 200)], 8, 1, 1, 128, True, 4.0),
        ([(2, 130), (2, 70)], 2, 2, 4, 64, False, 1.0),
    ],
)
def test_triton_sparse_matches_reference(batches, h_ratio, num_kv_heads, topk, page_size, causal, temp):
    _need_triton_sm12x()
    from fmha_sm12x._triton_sparse import triton_sparse_atten_dense

    device = torch.device("cuda")
    torch.manual_seed(1234 + topk + page_size)
    num_qo_heads = num_kv_heads * h_ratio
    head_dim = 128

    qo_lens = [b[0] for b in batches]
    kv_lens = [b[1] for b in batches]
    cu_q = torch.tensor([0, *torch.tensor(qo_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, *torch.tensor(kv_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    total_q = int(cu_q[-1].item())
    total_k = int(cu_k[-1].item())

    q = torch.randn((total_q, num_qo_heads, head_dim), device=device, dtype=torch.bfloat16) * 0.3
    k = torch.randn((total_k, num_kv_heads, head_dim), device=device, dtype=torch.bfloat16) * 0.3
    v = torch.randn((total_k, num_kv_heads, head_dim), device=device, dtype=torch.bfloat16) * 0.3
    sm_scale = 1.0 / math.sqrt(head_dim)

    # Build per-(query, kv_head) block selections within each batch's KV, with
    # some -1 padding and an occasional fully-padded (no-block) query.
    block_indexes = torch.full((total_q, num_kv_heads, topk), -1, dtype=torch.int32, device=device)
    rng = torch.Generator(device="cpu").manual_seed(7)
    qo_cpu = cu_q.cpu().tolist()
    for b in range(len(batches)):
        n_blocks = (kv_lens[b] + page_size - 1) // page_size
        for qi in range(qo_cpu[b], qo_cpu[b + 1]):
            for kh in range(num_kv_heads):
                if torch.rand(1, generator=rng).item() < 0.1:
                    continue  # leave fully padded
                n_sel = int(torch.randint(1, topk + 1, (1,), generator=rng).item())
                perm = torch.randperm(n_blocks, generator=rng)[:n_sel]
                block_indexes[qi, kh, : perm.numel()] = perm.to(torch.int32).to(device)

    ref_out, ref_lse, ref_tlse = _reference(
        q, k, v, block_indexes, cu_q=cu_q, cu_k=cu_k, num_kv_heads=num_kv_heads,
        page_size=page_size, causal=causal, sm_scale=sm_scale, want_temp=(temp != 1.0), temp=temp,
    )
    out, lse, tlse = triton_sparse_atten_dense(
        q, k, v, block_indexes, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        num_kv_heads=num_kv_heads, page_size=page_size, causal=causal, sm_scale=sm_scale,
        return_lse=True, lse_temperature_scale=temp, return_temperature_lse=(temp != 1.0),
    )

    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)

    # Compare LSE only where the reference has finite mass (fully-padded
    # queries are -inf in both and would fail allclose on inf arithmetic).
    finite = torch.isfinite(ref_lse)
    torch.testing.assert_close(lse[finite], ref_lse[finite], atol=2e-2, rtol=2e-2)
    assert bool((torch.isfinite(lse) == finite).all().item())
    if temp != 1.0:
        finite_t = torch.isfinite(ref_tlse)
        torch.testing.assert_close(tlse[finite_t], ref_tlse[finite_t], atol=2e-2, rtol=2e-2)


def test_triton_sparse_duplicate_and_fully_masked_blocks():
    # Lock two edge invariants against the reference: a block id duplicated in
    # a topk row (double-counted identically by both paths) and a selected
    # block entirely beyond an early query's causal window (no visible position
    # -> zero output, -inf LSE). qo_len == kv_len makes query 0's causal limit
    # 0, so any block it selects past block 0 is fully masked.
    _need_triton_sm12x()
    from fmha_sm12x._triton_sparse import triton_sparse_atten_dense

    device = torch.device("cuda")
    torch.manual_seed(99)
    head_dim, page_size, num_kv_heads, h_ratio = 64, 16, 1, 2
    num_qo_heads = num_kv_heads * h_ratio
    seq = 48  # qo_len == kv_len == 48 -> 3 blocks of 16
    cu_q = torch.tensor([0, seq], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, seq], dtype=torch.int32, device=device)
    q = torch.randn((seq, num_qo_heads, head_dim), device=device, dtype=torch.bfloat16) * 0.3
    k = torch.randn((seq, num_kv_heads, head_dim), device=device, dtype=torch.bfloat16) * 0.3
    v = torch.randn((seq, num_kv_heads, head_dim), device=device, dtype=torch.bfloat16) * 0.3
    sm_scale = 1.0 / math.sqrt(head_dim)

    block_indexes = torch.full((seq, num_kv_heads, 3), -1, dtype=torch.int32, device=device)
    block_indexes[0, 0, 0] = 2  # query 0 causal limit 0: block 2 fully masked
    block_indexes[20, 0, 0] = 0
    block_indexes[20, 0, 1] = 0  # duplicate, finite mass
    block_indexes[40, 0, 0] = 1
    block_indexes[40, 0, 1] = 2

    ref_out, ref_lse, _ = _reference(
        q, k, v, block_indexes, cu_q=cu_q, cu_k=cu_k, num_kv_heads=num_kv_heads,
        page_size=page_size, causal=True, sm_scale=sm_scale, want_temp=False, temp=1.0,
    )
    out, lse, _ = triton_sparse_atten_dense(
        q, k, v, block_indexes, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        num_kv_heads=num_kv_heads, page_size=page_size, causal=True, sm_scale=sm_scale,
        return_lse=True, out_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    # query 0 selects only a fully-masked block: zero output, -inf LSE.
    assert bool((out[0] == 0).all().item())
    assert bool(torch.isneginf(lse[0]).all().item())
    finite = torch.isfinite(ref_lse)
    torch.testing.assert_close(lse[finite], ref_lse[finite], atol=2e-2, rtol=2e-2)
    assert bool((torch.isfinite(lse) == finite).all().item())
