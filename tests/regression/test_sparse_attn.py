# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Correctness tests for Sparse Attention Mode (SM100 FMHA).

Uses PyTorch as reference: gather selected KV blocks, run dense attention.
Covers varlen, shuffled pages, different page sizes, edge cases.
"""
import math
import random
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from fmha_sm100.sparse_fmha_adapter import sparse_fmha as fmha_sm100, sparse_fmha_plan as fmha_sm100_plan
from fmha_sm100.api import (
    fmha_sm100 as dispatch_fmha_sm100,
    fmha_sm100_plan as dispatch_fmha_sm100_plan,
)

failed_cases = []


def check(name, o, o_ref, threshold=0.9999):
    # Where ref is NaN, kernel may output NaN or 0 — both are acceptable.
    # Only compare positions where ref is NOT NaN.
    valid = ~o_ref.isnan()
    if not valid.any():
        print(f"  [PASS] {name}: all NaN in ref")
        return True
    o_valid = o[valid].float()
    o_ref_valid = o_ref[valid].float()
    if o_valid.isnan().any() or o_valid.isinf().any():
        print(f"  [FAIL] {name}: kernel has NaN/inf where ref is finite (nan={o_valid.isnan().sum().item()}, inf={o_valid.isinf().sum().item()})")
        failed_cases.append(f"{name}: kernel NaN/inf on finite ref positions")
        return False
    cos_sim = torch.nn.functional.cosine_similarity(
        o_valid.reshape(-1), o_ref_valid.reshape(-1), dim=0
    ).item()
    max_diff = (o_valid - o_ref_valid).abs().max().item()
    passed = cos_sim > threshold
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4f}")
    if not passed:
        failed_cases.append(f"{name}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4f}")
    return passed


def sparse_ref(q_flat, k_pages, v_pages, qo_lens, kv_page_indptr,
               per_batch_blocks, num_qo_heads, num_kv_heads, page_size,
               qo_offsets, head_dim, device):
    """Reference: gather selected KV blocks, causal mask with original positions."""
    h_r = num_qo_heads // num_kv_heads
    ref_parts = []
    q_pos = 0
    for b in range(len(qo_lens)):
        ql = qo_lens[b]
        q_b = q_flat[q_pos:q_pos + ql]
        page_off = kv_page_indptr[b]
        heads_out = []
        for h in range(num_qo_heads):
            kv_h = h // h_r
            blocks = per_batch_blocks[b][kv_h]
            k_g = torch.cat([k_pages[page_off + blk, kv_h] for blk in blocks], dim=0)
            v_g = torch.cat([v_pages[page_off + blk, kv_h] for blk in blocks], dim=0)

            scores = torch.matmul(q_b[:, h].float(), k_g.float().T) / math.sqrt(head_dim)
            qi = torch.arange(ql, device=device).unsqueeze(1) + qo_offsets[b]
            kv_pos = []
            for blk in blocks:
                kv_pos.extend(range(blk * page_size, (blk + 1) * page_size))
            ki = torch.tensor(kv_pos, device=device, dtype=torch.int64).unsqueeze(0)
            scores.masked_fill_(qi < ki, float("-inf"))

            heads_out.append(torch.matmul(torch.softmax(scores, dim=-1), v_g.float()))
        ref_parts.append(torch.stack(heads_out, dim=1))
        q_pos += ql
    return torch.cat(ref_parts, dim=0).to(torch.bfloat16)


def run_sparse_flashinfer(q, k_pages, v_pages, qo_lens_list, original_kv_lens_list,
                          qo_offsets_list, kv_indices, pages_per_batch, kv_block_indexes,
                          kv_block_num, num_qo_heads, page_size, head_dim, device,
                          num_kv_splits=-1, dtype=torch.bfloat16,
                          public_page_table_mode=None):
    """Run sparse attention through the adapter or public dispatcher."""
    num_kv_heads = k_pages.shape[1]
    batch_size = len(qo_lens_list)
    qo_segment_lens = torch.tensor(qo_lens_list, dtype=torch.int32)
    kv_segment_lens = torch.tensor(original_kv_lens_list, dtype=torch.int32)
    qo_offset_tensor = torch.tensor(qo_offsets_list, dtype=torch.int32)
    plan_fn = (
        dispatch_fmha_sm100_plan
        if public_page_table_mode is not None
        else fmha_sm100_plan
    )
    plan_info = plan_fn(qo_segment_lens, kv_segment_lens,
        num_qo_heads, qo_offset=qo_offset_tensor,
        page_size=page_size,
        num_kv_splits=num_kv_splits, kv_block_num=kv_block_num,
        num_kv_heads=num_kv_heads,
    )
    torch.cuda.synchronize()
    page_table = None
    if public_page_table_mode is not None:
        page_table = torch.zeros(
            batch_size, max(pages_per_batch), dtype=torch.int32, device=device
        )
        offset = 0
        for batch_idx, num_pages in enumerate(pages_per_batch):
            page_table[batch_idx, :num_pages] = kv_indices[offset:offset + num_pages]
            offset += num_pages

        has_mixed, split, _, _, _ = plan_info
        if public_page_table_mode == "precedence":
            assert has_mixed, "expected public mixed dispatch"
            prefill_page_table = page_table[split:]
            assert prefill_page_table.data_ptr() % 16 != 0

            kv_page_split = sum(pages_per_batch[:split])
            prefill_kv_indices = kv_indices[kv_page_split:]
            rolled_kv_indices = prefill_kv_indices.roll(1)
            assert not torch.equal(rolled_kv_indices, prefill_kv_indices)
            kv_indices = kv_indices.clone()
            kv_indices[kv_page_split:] = rolled_kv_indices
        else:
            assert not has_mixed, "expected non-mixed sparse dispatch"
            kv_indices = None

    run_fn = (
        dispatch_fmha_sm100
        if public_page_table_mode is not None
        else fmha_sm100
    )

    out, _ = run_fn(
        q, k_pages, v_pages,
        plan_info=plan_info, sm_scale=1.0 / math.sqrt(head_dim), 
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
        check_input_valid=True,
        page_table=page_table,
    )
    torch.cuda.synchronize()
    return out


def _run_sparse_varlen(name, seed, batch_size, num_kv_heads, num_qo_heads,
                       page_size=128, head_dim=128, shuffle_pages=False,
                       qo_lens=None, original_kv_lens=None, qo_offsets=None,
                       sparse_block_counts=None, max_sparse_blocks=16,
                       num_kv_splits=-1, dtype=torch.bfloat16,
                       public_page_table_mode=None):
    """Generic sparse attention test with full control over parameters."""
    torch.manual_seed(seed)
    random.seed(seed)
    dev = torch.device("cuda")
    # dtype = torch.bfloat16

    # Generate per-batch configs if not provided
    if qo_lens is None:
        qo_lens = [random.randint(1, 16) for _ in range(batch_size)]
    if original_kv_lens is None:
        original_kv_lens = [random.randint(32, 64) * page_size for _ in range(batch_size)]

    pages_per_batch = [kv // page_size for kv in original_kv_lens]
    total_pages = sum(pages_per_batch)

    # Sparse block selection per (batch, kv_head)
    per_batch_blocks = []
    if sparse_block_counts is None:
        sparse_block_counts = [min(max_sparse_blocks, pp) for pp in pages_per_batch]
    for b in range(batch_size):
        heads_blocks = []
        for h in range(num_kv_heads):
            n = min(sparse_block_counts[b], pages_per_batch[b])
            heads_blocks.append(sorted(random.sample(range(pages_per_batch[b]), n)))
        per_batch_blocks.append(heads_blocks)

    kv_block_num = max(max_sparse_blocks, max(len(hb) for bb in per_batch_blocks for hb in bb))

    # qo_offsets: ensure ALL Q rows can see at least some blocks on EVERY head
    if qo_offsets is None:
        qo_offsets = []
        for b in range(batch_size):
            max_min_blk = max(min(hb) for hb in per_batch_blocks[b])
            max_offset = original_kv_lens[b] - qo_lens[b]
            min_offset = max(0, max_min_blk * page_size - (qo_lens[b] - 1))
            min_offset = min(min_offset, max_offset)
            qo_offsets.append(random.randint(min_offset, max(min_offset, max_offset)))

    # Build KV pages
    k_pages_bf16 = torch.randn(total_pages, num_kv_heads, page_size, head_dim, device=dev, dtype=torch.bfloat16)
    v_pages_bf16 = torch.randn(total_pages, num_kv_heads, page_size, head_dim, device=dev, dtype=torch.bfloat16)
    if dtype == torch.float8_e4m3fn:
        k_pages = k_pages_bf16.to(dtype)
        v_pages = v_pages_bf16.to(dtype)
        k_pages_ref = k_pages.to(torch.bfloat16)
        v_pages_ref = v_pages.to(torch.bfloat16)
    else:
        k_pages = k_pages_bf16
        v_pages = v_pages_bf16
        k_pages_ref = k_pages_bf16
        v_pages_ref = v_pages_bf16

    # Page table (optionally shuffled)
    if shuffle_pages:
        perm = torch.randperm(total_pages, device=dev, dtype=torch.int64)
        k_shuffled = torch.empty_like(k_pages)
        v_shuffled = torch.empty_like(v_pages)
        k_shuffled[perm] = k_pages
        v_shuffled[perm] = v_pages
        kv_indices = perm.to(torch.int32)
        k_pages_for_kernel = k_shuffled
        v_pages_for_kernel = v_shuffled
    else:
        kv_indices = torch.arange(total_pages, device=dev, dtype=torch.int32)
        k_pages_for_kernel = k_pages
        v_pages_for_kernel = v_pages

    kv_page_indptr = [0]
    for pp in pages_per_batch:
        kv_page_indptr.append(kv_page_indptr[-1] + pp)

    # Q
    total_qo = sum(qo_lens)
    q_bf16 = torch.randn(total_qo, num_qo_heads, head_dim, device=dev, dtype=torch.bfloat16)
    if dtype == torch.float8_e4m3fn:
        q = q_bf16.to(dtype)
        q_ref = q.to(torch.bfloat16)
    else:
        q = q_bf16
        q_ref = q_bf16

    # kv_block_indexes [total_qo, H_kv, kv_block_num], pad -1
    # Each Q token gets its batch's block list (same blocks for all tokens in a batch for now)
    kv_block_indexes = torch.full((total_qo, num_kv_heads, kv_block_num), -1, device=dev, dtype=torch.int32)
    q_pos = 0
    for b in range(batch_size):
        for h in range(num_kv_heads):
            blocks = per_batch_blocks[b][h]
            kv_block_indexes[q_pos:q_pos + qo_lens[b], h, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        q_pos += qo_lens[b]

    # Reference (uses quantized data for fp8 to isolate kernel accuracy)
    o_ref = sparse_ref(q_ref, k_pages_ref, v_pages_ref, qo_lens, kv_page_indptr,
                       per_batch_blocks, num_qo_heads, num_kv_heads, page_size,
                       qo_offsets, head_dim, dev)

    # FlashInfer
    out = run_sparse_flashinfer(
        q, k_pages_for_kernel, v_pages_for_kernel,
        qo_lens, original_kv_lens, qo_offsets,
        kv_indices, pages_per_batch, kv_block_indexes, kv_block_num,
        num_qo_heads, page_size, head_dim, dev,
        num_kv_splits=num_kv_splits, dtype=dtype,
        public_page_table_mode=public_page_table_mode,
    )

    threshold = 0.9999 if dtype == torch.bfloat16 else 0.999
    return check(name, out, o_ref, threshold)


if __name__ == "__main__":

    all_pass = True
    dtypes = [torch.bfloat16, torch.float8_e4m3fn]

    print("=== 1. Basic varlen + random offsets ===")
    for dt in dtypes:
        for seed in range(3):
            for B, hk, hq, desc in [(2,4,4,"MHA"), (4,2,32,"GQA16"), (1,1,1,"1head")]:
                all_pass &= _run_sparse_varlen(f"{desc} s={seed} {dt}", seed*100+B, B, hk, hq, dtype=dt)

    print("\n=== 2. Shuffled page table ===")
    for dt in dtypes:
        for seed in range(3):
            all_pass &= _run_sparse_varlen(f"shuffle s={seed} {dt}", seed, 3, 4, 16, shuffle_pages=True, dtype=dt)

    print("\n=== 3. Heavy padding (most blocks -1) ===")
    for dt in dtypes:
        for seed in range(3):
            all_pass &= _run_sparse_varlen(
                f"heavy_pad s={seed} {dt}", seed, 4, 4, 4,
                sparse_block_counts=[2, 1, 3, 1], max_sparse_blocks=16, dtype=dt,
            )

    print("\n=== 4. Q near sequence start (many above-diagonal skips) ===")
    for dt in dtypes:
        for seed in range(2):
            all_pass &= _run_sparse_varlen(
                f"q_start s={seed} {dt}", seed, 2, 4, 4,
                qo_lens=[256, 256],
                original_kv_lens=[8192, 4096],
                qo_offsets=[256, 128],
                sparse_block_counts=[8, 8], dtype=dt,
            )

    print("\n=== 5. Q at sequence end (all unmasked) ===")
    for dt in dtypes:
        for seed in range(2):
            all_pass &= _run_sparse_varlen(
                f"q_end s={seed} {dt}", seed, 2, 4, 4,
                qo_lens=[256, 512],
                original_kv_lens=[8192, 4096],
                qo_offsets=[8192 - 256, 4096 - 512], dtype=dt,
            )

    print("\n=== 6. Large scale ===")
    for dt in dtypes:
        all_pass &= _run_sparse_varlen(
            f"large {dt}", 42, 4, 8, 32,
            qo_lens=[256]*4,
            original_kv_lens=[16384]*4,
            max_sparse_blocks=32, dtype=dt,
        )

    print("\n=== 7. Single block per batch ===")
    for dt in dtypes:
        all_pass &= _run_sparse_varlen(
            f"single_block {dt}", 42, 4, 4, 4,
            sparse_block_counts=[1, 1, 1, 1], dtype=dt,
        )

    print("\n=== 8. Mixed block counts across batches ===")
    for dt in dtypes:
        all_pass &= _run_sparse_varlen(
            f"mixed_counts {dt}", 42, 4, 4, 16,
            sparse_block_counts=[2, 8, 1, 15], dtype=dt,
        )

    print("\n=== 9. SplitKV (auto split) ===")
    for dt in dtypes:
        for seed in range(3):
            for B, hk, hq, desc in [(2,4,4,"MHA"), (3,4,16,"GQA")]:
                all_pass &= _run_sparse_varlen(f"{desc} s={seed} {dt}", seed*100+B, B, hk, hq, dtype=dt)

    print("\n=== 10. SplitKV (forced 2 splits) ===")
    for dt in dtypes:
        for seed in range(2):
            all_pass &= _run_sparse_varlen(
                f"split2 s={seed} {dt}", seed, 2, 4, 16,
                qo_lens=[256, 512],
                original_kv_lens=[8192, 4096],
                num_kv_splits=2, dtype=dt,
            )

    print("\n=== 11. SplitKV (forced 4 splits, near tile limit) ===")
    for dt in dtypes:
        all_pass &= _run_sparse_varlen(
            f"split4 {dt}", 42, 4, 8, 32,
            qo_lens=[128]*4,
            original_kv_lens=[16384]*4,
            max_sparse_blocks=32,
            num_kv_splits=4, dtype=dt,
        )

    print("\n=== 12. Decode (qo_len=1 per batch) ===")
    for dt in dtypes:
        for seed in range(2):
            all_pass &= _run_sparse_varlen(
                f"decode s={seed} {dt}", seed, 4, 4, 16,
                qo_lens=[1]*4,
                original_kv_lens=[2048, 4096, 1024, 8192], dtype=dt,
            )

    print("\n=== 13. Odd qo_lens ===")
    for dt in dtypes:
        all_pass &= _run_sparse_varlen(
            f"odd_q {dt}", 42, 3, 4, 4,
            qo_lens=[1, 7, 33],
            original_kv_lens=[1024, 2048, 4096], dtype=dt,
        )

    print("\n=== 14. Per-token different blocks ===")
    # Each token in a batch selects different sparse blocks
    torch.manual_seed(99); random.seed(99)
    dev = torch.device("cuda")
    B, hk, hq, ps, hd = 1, 4, 4, 128, 128
    qo_len, kv_len = 8, 2048
    pages = kv_len // ps
    kbn = 8
    k_pages = torch.randn(pages, hk, ps, hd, device=dev, dtype=torch.bfloat16)
    v_pages = torch.randn(pages, hk, ps, hd, device=dev, dtype=torch.bfloat16)
    q = torch.randn(qo_len, hq, hd, device=dev, dtype=torch.bfloat16)
    ki_t = torch.arange(pages, device=dev, dtype=torch.int32)
    # Each token picks a DIFFERENT random subset of 8 blocks
    kbi_pt = torch.full((qo_len, hk, kbn), -1, device=dev, dtype=torch.int32)
    per_token_blocks = []
    for t in range(qo_len):
        blocks = sorted(random.sample(range(pages), kbn))
        per_token_blocks.append(blocks)
        kbi_pt[t, :, :] = torch.tensor(blocks, dtype=torch.int32)
    qo_off = kv_len - qo_len
    out_pt = run_sparse_flashinfer(
        q, k_pages, v_pages, [qo_len], [kv_len], [qo_off],
        ki_t, [pages], kbi_pt, kbn, hq, ps, hd, dev,
    )
    # Reference per-token
    ref_parts = []
    for t in range(qo_len):
        q_t = q[t:t+1]
        blocks = per_token_blocks[t]
        for h in range(hq):
            kv_h = h // (hq // hk)
            k_g = torch.cat([k_pages[blk, kv_h] for blk in blocks], dim=0)
            v_g = torch.cat([v_pages[blk, kv_h] for blk in blocks], dim=0)
            scores = (q_t[0, h].float() @ k_g.float().T) / math.sqrt(hd)
            qi_pos = qo_off + t
            kv_pos = torch.tensor([blk * ps + j for blk in blocks for j in range(ps)], device=dev)
            scores[kv_pos > qi_pos] = float("-inf")
            ref_parts.append(torch.softmax(scores, -1) @ v_g.float())
    o_ref_pt = torch.stack([torch.stack(ref_parts[t*hq:(t+1)*hq], dim=0) for t in range(qo_len)]).to(torch.bfloat16)
    all_pass &= check("per_token_diff_blocks", out_pt, o_ref_pt)

    print("\n=== 15. Extreme GQA (h_q=64, h_k=1) ===")
    for dt in dtypes:
        all_pass &= _run_sparse_varlen(f"extreme_gqa {dt}", 42, 2, 1, 16, qo_lens=[4, 8], original_kv_lens=[1024, 2048], dtype=dt)

    print("\n=== 16. Large batch + small seq ===")
    for dt in dtypes:
        all_pass &= _run_sparse_varlen(f"large_batch {dt}", 42, 8, 4, 4, qo_lens=[1]*8, original_kv_lens=[512]*8, dtype=dt)

    print("\n=== 17. Direct page table ===")
    for dt in dtypes:
        all_pass &= _run_sparse_varlen(
            f"public_mixed_unaligned_page_table {dt}", 42, 3, 1, 16,
            qo_lens=[8, 17, 33],
            original_kv_lens=[1024, 2176, 4224],
            shuffle_pages=True,
            max_sparse_blocks=32,
            dtype=dt,
            public_page_table_mode="precedence",
        )

    all_pass &= _run_sparse_varlen(
        "public_sparse_page_table_only", 43, 1, 1, 16,
        qo_lens=[64],
        original_kv_lens=[4096],
        shuffle_pages=True,
        max_sparse_blocks=32,
        public_page_table_mode="only",
    )

    print()
    total = len(failed_cases)
    if all_pass:
        print("All tests PASSED!")
    else:
        print(f"{total} test(s) FAILED:")
        for f in failed_cases:
            print(f"  {f}")
        exit(1)
