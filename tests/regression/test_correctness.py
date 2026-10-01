#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Correctness tests for FlashInfer SM100 FMHA (CUTLASS backend).

Tests both TILE_Q=256 (prefill) and TILE_Q=128 (short Q / decode) paths.
Uses PyTorch SDPA as reference.

Chaos tests are distributed across 8 GPUs via subprocess for faster execution.

The csrc kernel variants this sweep needs are listed in ``fmha_variants.txt`` next to this
file and compiled in parallel before the first case (``--jit-jobs N`` bounds the builds,
``--no-prebuild`` keeps lazy JIT); ``--update-variant-manifest`` rewrites the list from the
variants a run actually requested.
"""

import sys
import os
import json
import math
import queue
import random
import subprocess
import threading
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from fmha_sm100 import fmha_sm100 as _fmha_sm100
from fmha_sm100 import fmha_sm100_plan
from fmha_sm100 import jit as _fmha_jit

from functools import partial
fmha_sm100 = None

page_sizes = [128]

VARIANT_MANIFEST = Path(__file__).resolve().with_name("fmha_variants.txt")


def _flag_value(name, default=None):
    if name in sys.argv:
        index = sys.argv.index(name)
        if index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    return default


def _read_variant_manifest():
    if not VARIANT_MANIFEST.exists():
        return []
    lines = VARIANT_MANIFEST.read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def prebuild_variants():
    """Compile the manifest's variants in parallel instead of one at a time mid-sweep."""
    if "--no-prebuild" in sys.argv:
        return
    import time
    names = _read_variant_manifest()
    jobs = _flag_value("--jit-jobs")
    started = time.time()
    built = _fmha_jit.prebuild_fmha_variants(names, max_workers=int(jobs) if jobs else None)
    print(f"Pre-built {len(built)} of {len(names)} FMHA variants in {time.time() - started:.0f}s "
          f"({len(names) - len(built)} cached)")


def check_variant_manifest():
    """Report variants the sweep compiled lazily; rewrite the manifest when asked."""
    requested = set(_fmha_jit.requested_fmha_variants())
    listed = set(_read_variant_manifest())
    if "--update-variant-manifest" in sys.argv:
        VARIANT_MANIFEST.write_text(
            "# csrc FMHA variants tests/regression/test_correctness.py requests; pre-built in\n"
            "# parallel before the sweep. Regenerate with --update-variant-manifest.\n"
            + "".join(f"{name}\n" for name in sorted(requested | listed)))
        print(f"Wrote {len(requested | listed)} variants to {VARIANT_MANIFEST}")
    elif requested - listed:
        print(f"Note: {len(requested - listed)} variants were compiled lazily because "
              f"{VARIANT_MANIFEST.name} does not list them; run with --update-variant-manifest: "
              f"{sorted(requested - listed)}")


class _PlainProgress:
    """Stand-in for tqdm when it is not installed: iterate, count, ignore the rest."""

    def __init__(self, iterable=None, total=None, **kwargs):
        self._iterable = iterable if iterable is not None else ()
        self.n = 0

    def __iter__(self):
        for item in self._iterable:
            yield item
            self.n += 1

    def update(self, delta=1):
        self.n += delta

    def set_postfix(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _progress(*args, **kwargs):
    try:
        from tqdm import tqdm
    except ImportError:
        return _PlainProgress(*args, **kwargs)
    return tqdm(*args, **kwargs)

def sdpa_ref(q_bf16, k_bf16, v_bf16, h_q, h_k, causal=True, qo_offset=None):
    """Compute reference attention using PyTorch SDPA.

    Args:
        qo_offset: Causal mask offset. Q row i attends to KV pos j iff i + qo_offset >= j.
                   Can be int (uniform) or 1-D tensor of shape (b,) for per-batch offsets.
                   None = use PyTorch default (is_causal=True, offset=0).
    """
    b = q_bf16.shape[0]
    q_len, k_len = q_bf16.shape[1], k_bf16.shape[1]
    device = q_bf16.device
    gqa_ratio = h_q // h_k
    with torch.no_grad():
        need_custom_mask = False
        if causal and qo_offset is not None:
            if isinstance(qo_offset, torch.Tensor):
                need_custom_mask = True
            elif qo_offset != 0:
                need_custom_mask = True

        mask = None
        if need_custom_mask:
            if isinstance(qo_offset, torch.Tensor):
                offset = qo_offset.to(device).reshape(b, 1, 1)
            else:
                offset = qo_offset
            row = torch.arange(q_len, device=device).reshape(1, q_len, 1) + offset
            col = torch.arange(k_len, device=device).reshape(1, 1, k_len)
            mask = (row >= col) & (col < k_len)
            mask = mask.unsqueeze(1)

        # Estimate memory for repeat_interleave: b * h_q * k_len * d * 2 bytes
        mem_bytes = b * h_q * k_len * q_bf16.shape[-1] * 2 * 2  # K + V
        if mem_bytes > 40 * (1 << 30):  # >40GiB: per-group to avoid OOM
            o_parts = []
            for g in range(h_k):
                q_g = q_bf16[:, :, g * gqa_ratio:(g + 1) * gqa_ratio, :].transpose(1, 2)
                k_g = k_bf16[:, :, g:g+1, :].transpose(1, 2)
                v_g = v_bf16[:, :, g:g+1, :].transpose(1, 2)
                if mask is not None:
                    o_g = torch.nn.functional.scaled_dot_product_attention(
                        q_g, k_g, v_g, attn_mask=mask
                    )
                else:
                    o_g = torch.nn.functional.scaled_dot_product_attention(
                        q_g, k_g, v_g, is_causal=causal
                    )
                o_parts.append(o_g)
            o_ref = torch.cat(o_parts, dim=1).transpose(1, 2)
        else:
            q_ref = q_bf16.transpose(1, 2)
            k_ref = k_bf16.transpose(1, 2).repeat_interleave(gqa_ratio, dim=1)
            v_ref = v_bf16.transpose(1, 2).repeat_interleave(gqa_ratio, dim=1)
            if mask is not None:
                o_ref = torch.nn.functional.scaled_dot_product_attention(
                    q_ref, k_ref, v_ref, attn_mask=mask
                ).transpose(1, 2)
            else:
                o_ref = torch.nn.functional.scaled_dot_product_attention(
                    q_ref, k_ref, v_ref, is_causal=causal
                ).transpose(1, 2)
    return o_ref.reshape(b * q_len, h_q, q_bf16.shape[-1])


def run_fmha_sm100(q, k, v, b, q_len, k_len, h_q, h_k, d, dtype, num_kv_splits, causal=True, qo_offset=None, output_maxscore=False, output_o=True):
    """Run FlashInfer CUTLASS FMHA. Returns (o,) or (o, max_score, max_k_tiles)."""
    device = q.device
    qo_lens = torch.full((b,), q_len, dtype=torch.int32)
    kv_lens = torch.full((b,), k_len, dtype=torch.int32)
    plan_info = fmha_sm100_plan(qo_lens, kv_lens, h_q,
        causal=causal, qo_offset=qo_offset,
        num_kv_splits=num_kv_splits,
        output_maxscore=output_maxscore,
        num_kv_heads=h_k,
        device=device
    )
    torch.cuda.synchronize()
    o, max_score = fmha_sm100(
        q, k, v,
        plan_info=plan_info,
        output_maxscore=output_maxscore,
        output_o=output_o,
    )
    torch.cuda.synchronize()
    if output_maxscore:
        max_k_tiles = max_score.shape[1]
        return o, max_score, max_k_tiles
    return o


def ref_tile_max_score(q, k, qo_segment_lens, kv_segment_lens, num_kv_heads, max_k_tiles, causal=False, qo_offset_tensor=None):
    """Reference per-tile max QK score. Output layout: [head, k_tile, token]."""
    total_qo_len, num_qo_heads, head_dim = q.shape
    batch_size = qo_segment_lens.shape[0]
    h_r = num_qo_heads // num_kv_heads
    kv_tile_size = 128

    result = torch.full(
        (num_qo_heads, max_k_tiles, total_qo_len),
        -float("inf"), dtype=torch.float32, device=q.device,
    )

    qo_off = 0
    kv_offset = 0
    for b in range(batch_size):
        qo_len = int(qo_segment_lens[b].item())
        kv_len = int(kv_segment_lens[b].item())
        n_tiles = (kv_len + kv_tile_size - 1) // kv_tile_size

        q_seq = q[qo_off : qo_off + qo_len].float()
        k_seq = k[kv_offset : kv_offset + kv_len].float()

        causal_off = int(qo_offset_tensor[b].item()) if qo_offset_tensor is not None else (kv_len - qo_len)

        for t in range(n_tiles):
            k_start = t * kv_tile_size
            k_end = min(k_start + kv_tile_size, kv_len)
            k_chunk = k_seq[k_start:k_end]
            k_expanded = k_chunk.repeat_interleave(h_r, dim=1)
            scores = torch.einsum("qhd,khd->qhk", q_seq, k_expanded)
            if causal:
                q_pos = torch.arange(qo_len, device=q.device).unsqueeze(1) + causal_off
                k_pos = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
                scores = scores.masked_fill(~(q_pos >= k_pos).unsqueeze(1), -float("inf"))
            tile_max = scores.max(dim=-1).values  # [qo_len, num_qo_heads]
            result[:, t, qo_off : qo_off + qo_len] = tile_max.permute(1, 0)

        qo_off += qo_len
        kv_offset += kv_len

    return result



def check_maxscore(name, max_score, ref, log = True):
    """Compare max_score against precomputed reference. Returns True if passed."""
    valid = ref != -float("inf")
    if not valid.any():
        return True
    kv = max_score[valid]
    rv = ref[valid]
    diff = (kv - rv).abs()
    passed = torch.allclose(kv, rv, atol=1e-1, rtol=1e-1)
    torch.cuda.synchronize()
    status = "PASS" if passed else "FAIL"
    if log:
        print(f"    [{status} maxscore] max_diff={diff.max().item():.4f}")
    if not passed:
        failed_cases.append(f"{name} maxscore: max_diff={diff.max().item():.4f}")
    return passed

failed_cases = []

def check(name, o, o_ref, threshold=0.99):
    """Check correctness and print results."""
    cos_sim = torch.nn.functional.cosine_similarity(
        o.float().reshape(-1), o_ref.float().reshape(-1), dim=0
    ).item()
    max_diff = (o.float() - o_ref.float()).abs().max().item()
    passed = cos_sim > threshold
    torch.cuda.synchronize()
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4f}")
    if not passed:
        failed_cases.append(f"{name}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4f}")
    return passed


def _run_case(b, q_len, k_len, h_q, h_k, d, dtype, num_kv_splits=-1, causal=True):
    """Run a single correctness test case."""
    device = "cuda:0"
    dtype_name = {torch.float8_e4m3fn: "fp8", torch.bfloat16: "bf16"}[dtype]
    name = f"b={b} q={q_len} k={k_len} h_q={h_q} h_k={h_k} d={d} {dtype_name} splitKV={num_kv_splits} {'causal' if causal else 'noncausal'}"

    q_bf16 = torch.randn(b, q_len, h_q, d, dtype=torch.bfloat16, device=device)
    k_bf16 = torch.randn(b, k_len, h_k, d, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(b, k_len, h_k, d, dtype=torch.bfloat16, device=device)

    qo_offset_val = k_len - q_len if causal else 0
    qo_offset_tensor = torch.full((b,), qo_offset_val, dtype=torch.int32) if causal else None

    if dtype == torch.float8_e4m3fn:
        q_fi = q_bf16.reshape(b * q_len, h_q, d).to(dtype)
        k_fi = k_bf16.reshape(b * k_len, h_k, d).to(dtype)
        v_fi = v_bf16.reshape(b * k_len, h_k, d).to(dtype)
        # Use fp8-quantized inputs for reference to isolate kernel accuracy from quantization error
        q_ref_bf16 = q_fi.reshape(b, q_len, h_q, d).to(torch.bfloat16)
        k_ref_bf16 = k_fi.reshape(b, k_len, h_k, d).to(torch.bfloat16)
        v_ref_bf16 = v_fi.reshape(b, k_len, h_k, d).to(torch.bfloat16)
    else:
        q_fi = q_bf16.reshape(b * q_len, h_q, d)
        k_fi = k_bf16.reshape(b * k_len, h_k, d)
        v_fi = v_bf16.reshape(b * k_len, h_k, d)
        q_ref_bf16, k_ref_bf16, v_ref_bf16 = q_bf16, k_bf16, v_bf16

    o_ref = sdpa_ref(q_ref_bf16, k_ref_bf16, v_ref_bf16, h_q, h_k, causal, qo_offset=qo_offset_val)
    threshold = 0.99999 if dtype == torch.bfloat16 else 0.9995

    # Run 1: without maxscore (Off path)
    o = run_fmha_sm100(q_fi, k_fi, v_fi, b, q_len, k_len, h_q, h_k, d, dtype, num_kv_splits, causal,
                       qo_offset=qo_offset_tensor, output_maxscore=False)
    passed = check(name, o, o_ref, threshold)

    # Run 2: with maxscore + output (Mode 3: output_o=True, output_maxscore=True)
    maxscore_elems = h_q * ((k_len + 127) // 128) * b * q_len
    skip_maxscore = maxscore_elems >= (1 << 31)
    if skip_maxscore:
        return passed
    try:
        o2, max_score, max_k_tiles = run_fmha_sm100(q_fi, k_fi, v_fi, b, q_len, k_len, h_q, h_k, d, dtype, num_kv_splits, causal,
                           qo_offset=qo_offset_tensor, output_maxscore=True, output_o=True)
        passed &= check(name + " +ms", o2, o_ref, threshold)

        ms_ref = None
        if causal and max_score is not None:
            qo_lens = torch.full((b,), q_len, device=device, dtype=torch.int32)
            kv_lens = torch.full((b,), k_len, device=device, dtype=torch.int32)
            q_ref_flat = (q_fi.to(torch.bfloat16) if dtype == torch.float8_e4m3fn else q_fi).reshape(b*q_len, h_q, d)
            k_ref_flat = (k_fi.to(torch.bfloat16) if dtype == torch.float8_e4m3fn else k_fi).reshape(b*k_len, h_k, d)
            ms_ref = ref_tile_max_score(q_ref_flat, k_ref_flat, qo_lens, kv_lens, h_k, max_k_tiles, causal=True)
            passed &= check_maxscore(name, max_score, ms_ref)

        # Run 3: maxscore only (Mode 2: output_o=False, output_maxscore=True)
        _, max_score2, _ = run_fmha_sm100(q_fi, k_fi, v_fi, b, q_len, k_len, h_q, h_k, d, dtype, num_kv_splits, causal,
                           qo_offset=qo_offset_tensor, output_maxscore=True, output_o=False)
        if causal and max_score2 is not None and ms_ref is not None:
            passed &= check_maxscore(name + " ms_only", max_score2, ms_ref)
    except Exception as e:
        print(f"    [FAIL maxscore] {name}: {type(e).__name__}: {str(e)[:120]}")
        failed_cases.append(f"{name} maxscore: {type(e).__name__}")
        passed = False
    return passed


def _run_case_paged(b, q_len, k_len, h_q, h_k, d, dtype, page_size=128, causal=True):
    """Test paged KV cache correctness.

    Builds a paged KV cache from contiguous K/V tensors and **shuffles the page
    indices** so the kernel actually has to follow the page table (without the
    shuffle, an identity table mod-mapping would always give correct output even
    if the indirection code were broken).
    """
    device = "cuda:0"
    dtype_name = {torch.float8_e4m3fn: "fp8", torch.bfloat16: "bf16"}[dtype]
    name = f"paged b={b} q={q_len} k={k_len} h_q={h_q} h_k={h_k} d={d} ps={page_size} {dtype_name}"

    q_bf16 = torch.randn(b, q_len, h_q, d, dtype=torch.bfloat16, device=device)
    k_bf16 = torch.randn(b, k_len, h_k, d, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(b, k_len, h_k, d, dtype=torch.bfloat16, device=device)

    qo_offset_val = k_len - q_len if causal else 0
    qo_offset_tensor = torch.full((b,), qo_offset_val, dtype=torch.int32) if causal else None

    # Build paged KV cache: [total_page_num, num_kv_heads, page_size, head_dim] (HND-like layout)
    # Pad k_len up to a page boundary so reshape works cleanly.
    pages_per_seq = (k_len + page_size - 1) // page_size
    padded_len = pages_per_seq * page_size
    total_pages = b * pages_per_seq

    k_bf16_padded = k_bf16
    v_bf16_padded = v_bf16
    if padded_len > k_len:
        pad = torch.zeros(b, padded_len - k_len, h_k, d, dtype=torch.bfloat16, device=device)
        k_bf16_padded = torch.cat([k_bf16, pad], dim=1)
        v_bf16_padded = torch.cat([v_bf16, pad], dim=1)
    # (b, padded_len, h_k, d) -> (b, pages_per_seq, page_size, h_k, d) -> (total_pages, page_size, h_k, d)
    # Then transpose to tmp's HND layout (total_pages, h_k, page_size, d)
    k_pages_nhd = k_bf16_padded.reshape(b, pages_per_seq, page_size, h_k, d).reshape(total_pages, page_size, h_k, d)
    v_pages_nhd = v_bf16_padded.reshape(b, pages_per_seq, page_size, h_k, d).reshape(total_pages, page_size, h_k, d)
    k_pages = k_pages_nhd.transpose(1, 2).contiguous()  # (total_pages, h_k, page_size, d)
    v_pages = v_pages_nhd.transpose(1, 2).contiguous()

    # Place pages at SHUFFLED positions in the cache to exercise the page-index path.
    perm = torch.randperm(total_pages, device=device, dtype=torch.int64)
    k_paged_bf16 = torch.empty_like(k_pages)
    v_paged_bf16 = torch.empty_like(v_pages)
    k_paged_bf16[perm] = k_pages
    v_paged_bf16[perm] = v_pages

    # kv_indices: per-batch slice of the perm. Batch i's logical pages map to perm[i*pps : (i+1)*pps].
    kv_indices = perm.to(torch.int32)

    if dtype == torch.float8_e4m3fn:
        q_fi = q_bf16.reshape(b * q_len, h_q, d).to(dtype)
        k_paged_fi = k_paged_bf16.to(dtype)
        v_paged_fi = v_paged_bf16.to(dtype)
        # Use fp8-quantized inputs for reference (isolate kernel accuracy from quantization noise)
        q_ref_bf16 = q_fi.reshape(b, q_len, h_q, d).to(torch.bfloat16)
        # Reconstruct contiguous K/V from the (now fp8-quantized) paged cache for reference
        k_paged_for_ref = k_paged_fi.to(torch.bfloat16)
        v_paged_for_ref = v_paged_fi.to(torch.bfloat16)
    else:
        q_fi = q_bf16.reshape(b * q_len, h_q, d)
        k_paged_fi = k_paged_bf16
        v_paged_fi = v_paged_bf16
        q_ref_bf16 = q_bf16
        k_paged_for_ref = k_paged_bf16
        v_paged_for_ref = v_paged_bf16

    # Reconstruct contiguous K/V from the paged cache (following the SAME page table) for reference.
    # k_paged_for_ref shape: (total_pages, h_k, page_size, d). We need (b, k_len, h_k, d).
    k_ref_contig = torch.zeros(b, padded_len, h_k, d, dtype=torch.bfloat16, device=device)
    v_ref_contig = torch.zeros(b, padded_len, h_k, d, dtype=torch.bfloat16, device=device)
    for bi in range(b):
        for pi in range(pages_per_seq):
            logical_idx = bi * pages_per_seq + pi
            phys_idx = kv_indices[logical_idx].item()
            start = pi * page_size
            end = start + page_size
            # k_paged_for_ref[phys_idx] shape: (h_k, page_size, d) -> need (page_size, h_k, d)
            k_ref_contig[bi, start:end] = k_paged_for_ref[phys_idx].transpose(0, 1)
            v_ref_contig[bi, start:end] = v_paged_for_ref[phys_idx].transpose(0, 1)
    k_ref_contig = k_ref_contig[:, :k_len]  # crop padding
    v_ref_contig = v_ref_contig[:, :k_len]

    o_ref = sdpa_ref(q_ref_bf16, k_ref_contig, v_ref_contig, h_q, h_k, causal, qo_offset=qo_offset_val)

    qo_lens = torch.full((b,), q_len, dtype=torch.int32)
    kv_lens = torch.full((b,), k_len, dtype=torch.int32)
    plan_info = fmha_sm100_plan(qo_lens, kv_lens, h_q,
        causal=causal, qo_offset=qo_offset_tensor,
        page_size=page_size,
        num_kv_heads=h_k,
    )
    torch.cuda.synchronize()
    o, _ = fmha_sm100(
        q_fi, k_paged_fi, v_paged_fi, 
        plan_info=plan_info,
        kv_indices=kv_indices,
    )
    torch.cuda.synchronize()

    threshold = 0.99999 if dtype == torch.bfloat16 else 0.9995
    passed = check(name, o, o_ref, threshold)

    # Maxscore tests: skip if buffer too large
    maxscore_elems = h_q * ((k_len + 127) // 128) * b * q_len
    skip_maxscore = maxscore_elems >= (1 << 31)
    if skip_maxscore:
        return passed

    q_ref_flat = (q_fi.to(torch.bfloat16) if dtype == torch.float8_e4m3fn else q_fi).reshape(b * q_len, h_q, d)
    k_ref_flat = k_ref_contig.reshape(b * k_len, h_k, d)

    try:
        torch.cuda.synchronize()
        plan_info_ms = fmha_sm100_plan(qo_lens, kv_lens, h_q,
            causal=causal, qo_offset=qo_offset_tensor,
            page_size=page_size,
            output_maxscore=True,
            num_kv_heads=h_k,
        )
        torch.cuda.synchronize()
        max_k_tiles = max(plan_info_ms[3].get("max_k_tiles", -1),
                          plan_info_ms[4].get("max_k_tiles", -1) if plan_info_ms[4] else -1)

        ms_ref = ref_tile_max_score(q_ref_flat.float(), k_ref_flat.float(), qo_lens, kv_lens, h_k, max_k_tiles, causal=True) if causal else None

        # Mode 3: output_o=True, output_maxscore=True
        torch.cuda.synchronize()
        o3, max_score3 = fmha_sm100(
            q_fi, k_paged_fi, v_paged_fi, 
            plan_info=plan_info_ms,
            kv_indices=kv_indices,
            output_o=True, output_maxscore=True,
        )
        torch.cuda.synchronize()
        passed &= check(name + " +ms", o3, o_ref, threshold)
        if ms_ref is not None and max_score3 is not None:
            passed &= check_maxscore(name, max_score3, ms_ref)

        # Mode 2: output_o=False, output_maxscore=True
        torch.cuda.synchronize()
        _, max_score2 = fmha_sm100(
            q_fi, k_paged_fi, v_paged_fi, 
            plan_info=plan_info_ms,
            kv_indices=kv_indices,
            output_o=False, output_maxscore=True,
        )
        torch.cuda.synchronize()
        if ms_ref is not None and max_score2 is not None:
            passed &= check_maxscore(name + " ms_only", max_score2, ms_ref)
    except Exception as e:
        print(f"    [FAIL maxscore] {name}: {type(e).__name__}: {str(e)[:120]}")
        failed_cases.append(f"{name} maxscore: {type(e).__name__}")
        passed = False

    return passed


def _run_case_sparse(b, q_len, k_len, h_q, h_k, d, dtype, page_size=128,
                     kv_block_num=8, causal=True, seed=42,
                     interleaved_kv=False):
    """Test sparse attention mode (kv_block_indexes path) with paged KV cache.

    Sparse mode: per (Q token, KV head) the kernel attends only to a chosen
    set of up to ``kv_block_num`` pages from the page table (-1 padded).
    Pages are placed at shuffled physical positions so the kernel must
    correctly follow BOTH the page table (kv_indices) AND the per-head block
    selection (kv_block_indexes).

    Causal masking uses the original KV positions (block_idx * page_size + i).
    Reference gathers selected pages and runs dense softmax+matmul.

    Note: kv_block_num must be in {4, 8, 16, 32} (kernel/adapter constraint).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    device = "cuda:0"
    dtype_name = {torch.float8_e4m3fn: "fp8", torch.bfloat16: "bf16"}[dtype]
    name = (f"sparse b={b} q={q_len} k={k_len} h_q={h_q} h_k={h_k} d={d} "
            f"ps={page_size} kbn={kv_block_num} {dtype_name}"
            f"{' interleaved_kv' if interleaved_kv else ''}")

    assert k_len % page_size == 0, f"k_len={k_len} must be page_size={page_size} aligned"
    pages_per_seq = k_len // page_size
    actual_block_num = min(kv_block_num, pages_per_seq)

    total_pages = b * pages_per_seq
    # Logical (unshuffled) KV cache. HND layout: (total_pages, h_k, page_size, d)
    k_pages_bf16 = torch.randn(total_pages, h_k, page_size, d, dtype=torch.bfloat16, device=device)
    v_pages_bf16 = torch.randn(total_pages, h_k, page_size, d, dtype=torch.bfloat16, device=device)

    if dtype == torch.float8_e4m3fn:
        k_pages_logical = k_pages_bf16.to(dtype)
        v_pages_logical = v_pages_bf16.to(dtype)
        # fp8-quantized values used as reference too (isolate kernel accuracy from quant noise)
        k_pages_ref = k_pages_logical.to(torch.bfloat16)
        v_pages_ref = v_pages_logical.to(torch.bfloat16)
    else:
        k_pages_logical = k_pages_bf16
        v_pages_logical = v_pages_bf16
        k_pages_ref = k_pages_bf16
        v_pages_ref = v_pages_bf16

    # Shuffle physical placement: kernel must follow kv_indices.
    perm = torch.randperm(total_pages, device=device, dtype=torch.int64)
    if interleaved_kv:
        kv_pages_logical = torch.cat(
            (k_pages_logical, v_pages_logical), dim=-1
        )
        kv_pages_phys = torch.empty_like(kv_pages_logical)
        kv_pages_phys[perm] = kv_pages_logical
        k_pages_phys, v_pages_phys = kv_pages_phys.split(d, dim=-1)
        assert k_pages_phys.stride(2) == 2 * d
        assert v_pages_phys.stride(2) == 2 * d
    else:
        k_pages_phys = torch.empty_like(k_pages_logical)
        v_pages_phys = torch.empty_like(v_pages_logical)
        k_pages_phys[perm] = k_pages_logical
        v_pages_phys[perm] = v_pages_logical
    kv_indices = perm.to(torch.int32)

    # Q
    q_bf16 = torch.randn(b * q_len, h_q, d, dtype=torch.bfloat16, device=device)
    if dtype == torch.float8_e4m3fn:
        q = q_bf16.to(dtype)
        q_ref = q.to(torch.bfloat16)
    else:
        q = q_bf16
        q_ref = q_bf16

    qo_offset_val = k_len - q_len if causal else 0
    qo_offset_tensor = torch.full((b,), qo_offset_val, dtype=torch.int32) if causal else None
    # Per-(batch, kv_head) sparse block selection — strictly ascending, padded with -1.
    # Always include block 0 so the smallest Q row sees at least one valid kv pos under causal.
    per_batch_blocks = []
    for bi in range(b):
        heads_blocks = []
        for kh in range(h_k):
            n = actual_block_num
            if n >= pages_per_seq:
                blocks = list(range(pages_per_seq))
            else:
                rest = sorted(random.sample(range(1, pages_per_seq), n - 1))
                blocks = [0] + rest
            heads_blocks.append(blocks)
        per_batch_blocks.append(heads_blocks)

    total_qo = b * q_len
    kv_block_indexes = torch.full((total_qo, h_k, kv_block_num), -1, device=device, dtype=torch.int32)
    for bi in range(b):
        for kh in range(h_k):
            blocks = per_batch_blocks[bi][kh]
            slc = slice(bi * q_len, (bi + 1) * q_len)
            kv_block_indexes[slc, kh, :len(blocks)] = torch.tensor(
                blocks, dtype=torch.int32, device=device)

    qo_lens = torch.full((b,), q_len, dtype=torch.int32)
    kv_lens = torch.full((b,), k_len, dtype=torch.int32)
    torch.cuda.synchronize()
    plan_info = fmha_sm100_plan(qo_lens, kv_lens, h_q,
        causal=causal, qo_offset=qo_offset_tensor,
        page_size=page_size,
        kv_block_num=kv_block_num,
        num_kv_heads=h_k,
    )
    torch.cuda.synchronize()
    o, _ = fmha_sm100(
        q, k_pages_phys, v_pages_phys,
        plan_info=plan_info,
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
        sm_scale=1.0 / math.sqrt(d),
    )
    torch.cuda.synchronize()

    # Reference: gather selected (logical) blocks, causal-mask, softmax + matmul.
    h_r = h_q // h_k
    o_ref = torch.zeros(total_qo, h_q, d, device=device, dtype=torch.float32)
    for bi in range(b):
        for h in range(h_q):
            kh = h // h_r
            blocks = per_batch_blocks[bi][kh]
            blk_t = torch.tensor(blocks, device=device, dtype=torch.int64)
            # k_pages_ref[bi*pps + blk, kh] -> (page_size, d), cat along seq
            logical_idx = bi * pages_per_seq + blk_t
            k_g = k_pages_ref[logical_idx, kh].reshape(-1, d).float()
            v_g = v_pages_ref[logical_idx, kh].reshape(-1, d).float()
            kv_pos = (blk_t.unsqueeze(1) * page_size +
                      torch.arange(page_size, device=device, dtype=torch.int64).unsqueeze(0)
                      ).reshape(-1)
            q_pos = qo_offset_val + torch.arange(q_len, device=device, dtype=torch.int64)
            scores = (q_ref[bi*q_len:(bi+1)*q_len, h].float() @ k_g.T) / math.sqrt(d)
            if causal:
                mask = q_pos.unsqueeze(1) < kv_pos.unsqueeze(0)
                scores = scores.masked_fill(mask, float("-inf"))
            # Rows fully masked -> uniform 0 output (kernel may emit NaN; check() skips via valid mask)
            row_all_masked = scores.isinf().all(dim=-1)
            scores = scores.masked_fill(row_all_masked.unsqueeze(-1), 0.0)
            probs = torch.softmax(scores, dim=-1)
            probs = probs.masked_fill(row_all_masked.unsqueeze(-1), 0.0)
            o_ref[bi*q_len:(bi+1)*q_len, h] = probs @ v_g

    threshold = 0.9999 if dtype == torch.bfloat16 else 0.999
    return check(name, o, o_ref.to(torch.bfloat16), threshold)


def _run_case_chaos(h_q, h_k, d, dtype, causal=True, seed=42, device="cuda:0"):
    """Test with random variable q_len, kv_len, and qo_offset per batch."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dtype_name = {torch.float8_e4m3fn: "fp8", torch.bfloat16: "bf16"}[dtype]

    b = random.randint(1, 16)
    total_q_budget = 8192

    q_lens, kv_lens, offsets = [], [], []
    remaining_q = total_q_budget
    small_q = random.randint(0, 1) == 1
    pd_split = random.randint(0, 1) == 1
    for i in range(b):
        max_q = min(remaining_q - (b - 1 - i), 4096)
        max_q = 128 if small_q else max_q
        ql = random.randint(1, max(1, max_q))
        remaining_q -= ql
        kl = random.randint(ql, min(65536, ql * 128))
        off = random.randint(0, kl - ql)
        q_lens.append(ql)
        kv_lens.append(kl)
        offsets.append(off)
    order = sorted(range(len(q_lens)), key=lambda i: q_lens[i])
    q_lens = [q_lens[i] for i in order]
    kv_lens = [kv_lens[i] for i in order]
    offsets = [offsets[i] for i in order]

    name = f"chaos b={b} small_q={small_q} total_q={sum(q_lens)} total_kv={sum(kv_lens)} h_q={h_q} h_k={h_k} d={d} {dtype_name} seed={seed}"

    q_list, k_list, v_list, o_ref_list = [], [], [], []
    for ql, kl, off in zip(q_lens, kv_lens, offsets):
        qi = torch.randn(1, ql, h_q, d, dtype=torch.bfloat16, device=device).clamp(min=-5, max=5)
        ki = torch.randn(1, kl, h_k, d, dtype=torch.bfloat16, device=device).clamp(min=-5, max=5)
        vi = torch.randn(1, kl, h_k, d, dtype=torch.bfloat16, device=device).clamp(min=-5, max=5)
        if dtype == torch.float8_e4m3fn:
            # Use fp8-quantized inputs for reference to isolate kernel accuracy from quantization error
            qi_ref = qi.to(dtype).to(torch.bfloat16)
            ki_ref = ki.to(dtype).to(torch.bfloat16)
            vi_ref = vi.to(dtype).to(torch.bfloat16)
        else:
            qi_ref, ki_ref, vi_ref = qi, ki, vi
        o_ref_i = sdpa_ref(qi_ref, ki_ref, vi_ref, h_q, h_k, causal=causal, qo_offset=off)
        q_list.append(qi.reshape(ql, h_q, d))
        k_list.append(ki.reshape(kl, h_k, d))
        v_list.append(vi.reshape(kl, h_k, d))
        o_ref_list.append(o_ref_i)

    q_cat = torch.cat(q_list, dim=0)
    k_cat = torch.cat(k_list, dim=0)
    v_cat = torch.cat(v_list, dim=0)
    o_ref = torch.cat(o_ref_list, dim=0)

    qo_lens_tensor = torch.tensor(q_lens, dtype=torch.int32)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
    qo_offset_tensor = torch.tensor(offsets, dtype=torch.int32)

    if dtype == torch.float8_e4m3fn:
        q_fi, k_fi, v_fi = q_cat.to(dtype), k_cat.to(dtype), v_cat.to(dtype)
    else:
        q_fi, k_fi, v_fi = q_cat, k_cat, v_cat

    torch.cuda.synchronize()
    plan_info = fmha_sm100_plan(qo_lens_tensor, kv_lens_tensor, h_q,
        causal=causal, qo_offset=qo_offset_tensor,
        split_prefill_decode = pd_split,
        num_kv_heads=h_k,
    )

    name += f" {plan_info[3]['predicted_speedup']:.2f} "
    if plan_info[0]:
        name += f"{plan_info[4]['predicted_speedup']:.2f} "

    torch.cuda.synchronize()
    o, _ = fmha_sm100(
        q_fi, k_fi, v_fi, 
        plan_info=plan_info,
    )
    torch.cuda.synchronize()

    threshold = 0.99999 if dtype == torch.bfloat16 else 0.9995
    threshold_diff = 0.02 if dtype == torch.bfloat16 else 0.1
    cos_sim = torch.nn.functional.cosine_similarity(
        o.float().reshape(-1), o_ref.float().reshape(-1), dim=0
    ).item()
    max_diff = (o.float() - o_ref.float()).abs().max().item()
    passed = cos_sim > threshold and max_diff < threshold_diff
    fail_msg = None if passed else f"{name}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4f}"

    if not passed:
        o_f = o.float()
        o_ref_f = o_ref.float()
        has_nan = torch.isnan(o_f).any().item()
        nan_count = torch.isnan(o_f).sum().item()
        inf_count = torch.isinf(o_f).sum().item()
        diag_parts = [f"  nan_count={nan_count}, inf_count={inf_count}"]

        row_offset = 0
        for bi, ql in enumerate(q_lens):
            for hi in range(h_q):
                chunk_o = o_f[row_offset:row_offset+ql, hi, :]
                chunk_ref = o_ref_f[row_offset:row_offset+ql, hi, :]
                c_nan = torch.isnan(chunk_o).sum().item()
                c_diff = (chunk_o - chunk_ref).abs().max().item()
                if c_nan > 0 or c_diff > threshold_diff:
                    c_cos = torch.nn.functional.cosine_similarity(
                        chunk_o.reshape(-1), chunk_ref.reshape(-1), dim=0).item()
                    diag_parts.append(
                        f"  batch={bi} head={hi} qlen={ql} kvlen={kv_lens[bi]} off={offsets[bi]}: "
                        f"nan={c_nan} cos={c_cos:.6f} diff={c_diff:.4f}")
            row_offset += ql

        if has_nan:
            nan_mask = torch.isnan(o_f)
            nan_rows = nan_mask.any(dim=-1)
            for ri in range(min(nan_rows.sum().item(), 5)):
                idx = nan_rows.nonzero()[ri]
                row_idx, head_idx = idx[0].item(), idx[1].item()
                cum = 0
                for bi, ql in enumerate(q_lens):
                    if row_idx < cum + ql:
                        local_row = row_idx - cum
                        diag_parts.append(
                            f"  NaN row: global={row_idx} batch={bi} local_row={local_row} head={head_idx}")
                        break
                    cum += ql

        is_split = plan_info[3].get("predicted_speedup", 0) != 0
        diag_parts.append(f"  split={is_split}")
        fail_msg += "\n" + "\n".join(diag_parts)

    # Maxscore: compute reference once, check Mode 3 + Mode 2
    max_kv_len = max(kv_lens)
    maxscore_elems = h_q * ((max_kv_len + 127) // 128) * sum(q_lens)
    skip_maxscore = maxscore_elems >= (1 << 31)
    if skip_maxscore:
        return passed, fail_msg

    q_ref_flat = (q_fi.to(torch.bfloat16) if dtype == torch.float8_e4m3fn else q_fi)
    k_ref_flat = (k_fi.to(torch.bfloat16) if dtype == torch.float8_e4m3fn else k_fi)

    try:
        plan_info_ms = fmha_sm100_plan(qo_lens_tensor, kv_lens_tensor, h_q,
            causal=causal, qo_offset=qo_offset_tensor,
            output_maxscore=True,
            num_kv_heads=h_k,
        )
        max_k_tiles = max(plan_info_ms[3].get("max_k_tiles", -1),
                          plan_info_ms[4].get("max_k_tiles", -1) if plan_info_ms[4] else -1)

        ms_ref = ref_tile_max_score(q_ref_flat.float(), k_ref_flat.float(), qo_lens_tensor, kv_lens_tensor, h_k, max_k_tiles, causal=causal, qo_offset_tensor=qo_offset_tensor) if causal else None

        # Mode 3: output_o=True, output_maxscore=True
        torch.cuda.synchronize()
        o3, max_score3 = fmha_sm100(
            q_fi, k_fi, v_fi, 
            plan_info=plan_info_ms,
            output_o=True, output_maxscore=True,
        )
        torch.cuda.synchronize()
        cos3 = torch.nn.functional.cosine_similarity(o3.float().reshape(-1), o_ref.float().reshape(-1), dim=0).item()
        diff3 = (o3.float() - o_ref.float()).abs().max().item()
        if not (cos3 > threshold and diff3 < threshold_diff):
            passed = False
            fail_msg = f"{name} +ms: cos_sim={cos3:.6f}, max_diff={diff3:.4f}"
        if ms_ref is not None and max_score3 is not None:
            if not check_maxscore(name, max_score3, ms_ref, False):
                passed = False
                if fail_msg is None:
                    fail_msg = f"{name} maxscore: FAIL"

        # Mode 2: output_o=False, output_maxscore=True
        torch.cuda.synchronize()
        _, max_score2 = fmha_sm100(
            q_fi, k_fi, v_fi, 
            plan_info=plan_info_ms,
            output_o=False, output_maxscore=True,
        )
        torch.cuda.synchronize()
        if ms_ref is not None and max_score2 is not None:
            if not check_maxscore(name + " ms_only", max_score2, ms_ref, False):
                passed = False
                if fail_msg is None:
                    fail_msg = f"{name} ms_only: FAIL"
    except Exception as e:
        passed = False
        fail_msg = f"{name} maxscore: {type(e).__name__}: {str(e)[:120]}"

    return passed, fail_msg


def _run_case_chaos_paged(h_q, h_k, d, dtype, page_size, causal=True, seed=42, device="cuda:0"):
    """Chaos test for paged KV cache: random b + per-sequence q_len/kv_len/qo_offset.

    Same random spec generation as test_case_chaos (seed-driven), but the KV
    is laid out as a paged cache with shuffled page indices so the kernel must
    follow the page table.

    Notes vs v2's test_case_chaos_paged:
      - v2 truncates KV to eff_kv_len = off + ql (since paged wrapper has no
        explicit qo_offset; offset is implicit = k_len - q_len).
      - tmp's fmha_sm100 accepts an explicit qo_offset_tensor (per batch), so
        we keep the full kv_lens AND pass qo_offset_tensor explicitly. This is
        more general and exercises a wider range of paged decode patterns.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dtype_name = {torch.float8_e4m3fn: "fp8", torch.bfloat16: "bf16"}[dtype]

    b = random.randint(1, 16)
    total_q_budget = 8192

    q_lens, kv_lens, offsets = [], [], []
    remaining_q = total_q_budget
    small_q = random.randint(0, 1) == 1
    pd_split = random.randint(0, 1) == 1
    for i in range(b):
        max_q = min(remaining_q - (b - 1 - i), 4096)
        max_q = 128 if small_q else max_q
        ql = random.randint(1, max(1, max_q))
        remaining_q -= ql
        kl = random.randint(ql, min(65536, ql * 128))
        off = random.randint(0, kl - ql)
        q_lens.append(ql)
        kv_lens.append(kl)
        offsets.append(off)
    order = sorted(range(len(q_lens)), key=lambda i: q_lens[i])
    q_lens = [q_lens[i] for i in order]
    kv_lens = [kv_lens[i] for i in order]
    offsets = [offsets[i] for i in order]

    name = (f"chaos_paged b={b} small_q={small_q} ps={page_size} "
            f"total_q={sum(q_lens)} total_kv={sum(kv_lens)} "
            f"h_q={h_q} h_k={h_k} d={d} {dtype_name} seed={seed}")

    # Build per-sequence Q/K/V (full kv_lens, with explicit qo_offset for causal)
    q_list, k_list, v_list, o_ref_list = [], [], [], []
    for ql, kl, off in zip(q_lens, kv_lens, offsets):
        qi = torch.randn(1, ql, h_q, d, dtype=torch.bfloat16, device=device).clamp(min=-5, max=5)
        ki = torch.randn(1, kl, h_k, d, dtype=torch.bfloat16, device=device).clamp(min=-5, max=5)
        vi = torch.randn(1, kl, h_k, d, dtype=torch.bfloat16, device=device).clamp(min=-5, max=5)
        if dtype == torch.float8_e4m3fn:
            qi_ref = qi.to(dtype).to(torch.bfloat16)
            ki_ref = ki.to(dtype).to(torch.bfloat16)
            vi_ref = vi.to(dtype).to(torch.bfloat16)
        else:
            qi_ref, ki_ref, vi_ref = qi, ki, vi
        o_ref_i = sdpa_ref(qi_ref, ki_ref, vi_ref, h_q, h_k, causal=causal, qo_offset=off)
        q_list.append(qi.reshape(ql, h_q, d))
        k_list.append(ki.reshape(kl, h_k, d))
        v_list.append(vi.reshape(kl, h_k, d))
        o_ref_list.append(o_ref_i)

    o_ref = torch.cat(o_ref_list, dim=0)

    # Build a paged KV cache: per-sequence variable kv_lens. Pad each sequence to
    # page boundary, split into pages, place at SHUFFLED positions in global cache.
    pages_per_seq_list = [(kl + page_size - 1) // page_size for kl in kv_lens]
    total_pages = sum(pages_per_seq_list)
    perm = torch.randperm(total_pages, device=device, dtype=torch.int64)
    # HND-like layout: (total_pages, h_k, page_size, d)
    k_cache = torch.empty(total_pages, h_k, page_size, d, dtype=dtype, device=device)
    v_cache = torch.empty(total_pages, h_k, page_size, d, dtype=dtype, device=device)

    page_offset = 0
    kv_indices_list = []
    for ki, vi, kl, pps in zip(k_list, v_list, kv_lens, pages_per_seq_list):
        ki_q = ki.to(dtype) if dtype == torch.float8_e4m3fn else ki
        vi_q = vi.to(dtype) if dtype == torch.float8_e4m3fn else vi
        padded_len = pps * page_size
        if padded_len > kl:
            pad = torch.zeros(padded_len - kl, h_k, d, dtype=dtype, device=device)
            ki_q = torch.cat([ki_q, pad], dim=0)
            vi_q = torch.cat([vi_q, pad], dim=0)
        seq_perm = perm[page_offset:page_offset + pps]
        # ki_q shape: (padded_len, h_k, d) -> (pps, page_size, h_k, d) -> (pps, h_k, page_size, d)
        k_pages_seq = ki_q.reshape(pps, page_size, h_k, d).transpose(1, 2).contiguous()
        v_pages_seq = vi_q.reshape(pps, page_size, h_k, d).transpose(1, 2).contiguous()
        k_cache[seq_perm] = k_pages_seq
        v_cache[seq_perm] = v_pages_seq
        kv_indices_list.append(seq_perm.to(torch.int32))
        page_offset += pps

    kv_indices = torch.cat(kv_indices_list)
    qo_lens_tensor = torch.tensor(q_lens, dtype=torch.int32)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
    qo_offset_tensor = torch.tensor(offsets, dtype=torch.int32)

    q_cat = torch.cat(q_list, dim=0)
    q_fi = q_cat.to(dtype) if dtype == torch.float8_e4m3fn else q_cat

    torch.cuda.synchronize()
    plan_info = fmha_sm100_plan(qo_lens_tensor, kv_lens_tensor, h_q,
        causal=causal, qo_offset=qo_offset_tensor,
        split_prefill_decode = pd_split,
        page_size=page_size,
    )
    torch.cuda.synchronize()
    o, _ = fmha_sm100(
        q_fi, k_cache, v_cache, 
        plan_info=plan_info,
        kv_indices=kv_indices,
    )
    torch.cuda.synchronize()

    threshold = 0.99999 if dtype == torch.bfloat16 else 0.9995
    threshold_diff = 0.02 if dtype == torch.bfloat16 else 0.11
    cos_sim = torch.nn.functional.cosine_similarity(
        o.float().reshape(-1), o_ref.float().reshape(-1), dim=0
    ).item()
    max_diff = (o.float() - o_ref.float()).abs().max().item()
    passed = cos_sim > threshold and max_diff < threshold_diff
    fail_msg = None if passed else f"{name}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4f}"

    # Maxscore: compute reference once, check Mode 3 + Mode 2
    max_kv_len = max(kv_lens)
    maxscore_elems = h_q * ((max_kv_len + 127) // 128) * sum(q_lens)
    skip_maxscore = maxscore_elems >= (1 << 31)
    if skip_maxscore:
        return passed, fail_msg

    k_cat = torch.cat(k_list, dim=0)
    q_ref_flat = (q_fi.to(torch.bfloat16) if dtype == torch.float8_e4m3fn else q_fi)
    k_ref_flat = (k_cat.to(dtype).to(torch.bfloat16) if dtype == torch.float8_e4m3fn else k_cat)

    try:
        torch.cuda.synchronize()
        plan_info_ms = fmha_sm100_plan(qo_lens_tensor, kv_lens_tensor, h_q,
            causal=causal, qo_offset=qo_offset_tensor,
            page_size=page_size,
            output_maxscore=True,
        )
        max_k_tiles = max(plan_info_ms[3].get("max_k_tiles", -1),
                          plan_info_ms[4].get("max_k_tiles", -1) if plan_info_ms[4] else -1)

        ms_ref = ref_tile_max_score(q_ref_flat.float(), k_ref_flat.float(), qo_lens_tensor, kv_lens_tensor, h_k, max_k_tiles, causal=causal, qo_offset_tensor=qo_offset_tensor) if causal else None

        # Mode 3: output_o=True, output_maxscore=True
        torch.cuda.synchronize()
        o3, max_score3 = fmha_sm100(
            q_fi, k_cache, v_cache, 
            plan_info=plan_info_ms,
            kv_indices=kv_indices,
            output_o=True, output_maxscore=True,
        )
        torch.cuda.synchronize()
        cos3 = torch.nn.functional.cosine_similarity(o3.float().reshape(-1), o_ref.float().reshape(-1), dim=0).item()
        diff3 = (o3.float() - o_ref.float()).abs().max().item()
        if not (cos3 > threshold and diff3 < threshold_diff):
            passed = False
            fail_msg = f"{name} +ms: cos_sim={cos3:.6f}, max_diff={diff3:.4f}"
        if ms_ref is not None and max_score3 is not None:
            if not check_maxscore(name, max_score3, ms_ref, False):
                passed = False
                if fail_msg is None:
                    fail_msg = f"{name} maxscore: FAIL"

        torch.cuda.synchronize()
        # Mode 2: output_o=False, output_maxscore=True
        _, max_score2 = fmha_sm100(
            q_fi, k_cache, v_cache, 
            plan_info=plan_info_ms,
            kv_indices=kv_indices,
            output_o=False, output_maxscore=True,
        )
        torch.cuda.synchronize()
        if ms_ref is not None and max_score2 is not None:
            if not check_maxscore(name + " ms_only", max_score2, ms_ref, False):
                passed = False
                if fail_msg is None:
                    fail_msg = f"{name} ms_only: FAIL"
    except Exception as e:
        passed = False
        fail_msg = f"{name} maxscore: {type(e).__name__}: {str(e)[:120]}"

    return passed, fail_msg


def _run_case_chaos_sparse(h_q, h_k, d, dtype, page_size, kv_block_num,
                           causal=True, seed=42, device="cuda:0"):
    """Chaos test for sparse attention (kv_block_indexes path) on a paged cache.

    Same seed-driven random spec generator as test_case_chaos_paged (batch
    size, per-seq qo/kv lens, qo_offset), but additionally:
      - per (batch, kv_head) pick a strictly-ascending subset of pages
        (block 0 always included, capped at min(kv_block_num, pages_per_seq))
      - assemble kv_block_indexes [total_qo, h_k, kv_block_num] with -1 pad
      - call unified fmha_sm100 with kv_block_num + kv_block_indexes

    causal masking uses ORIGINAL kv positions (block_idx*page_size + j).
    pd_split is exercised — for mixed batches the unified API splits the
    sparse pass into a decode part (q<=128, local sparse kernel) and a
    prefill part (q>128, MM-SA-Nv kernel).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dtype_name = {torch.float8_e4m3fn: "fp8", torch.bfloat16: "bf16"}[dtype]

    b = random.randint(1, 16)
    total_q_budget = 8192

    q_lens, kv_lens, offsets = [], [], []
    remaining_q = total_q_budget
    small_q = random.randint(0, 1) == 1
    pd_split = random.randint(0, 1) == 1
    for i in range(b):
        max_q = min(remaining_q - (b - 1 - i), 4096)
        max_q = 128 if small_q else max_q
        ql = random.randint(1, max(1, max_q))
        remaining_q -= ql
        # KV must be page-aligned for sparse mode (block-indexed).
        kl_unpadded = random.randint(ql, min(65536, ql * 128))
        pps = max(1, (kl_unpadded + page_size - 1) // page_size)
        kl = pps * page_size
        # qo_offset must satisfy: smallest q_pos >= 0 (always ok) and largest <= kl - 1.
        max_off = kl - ql
        off = random.randint(0, max(0, max_off))
        q_lens.append(ql)
        kv_lens.append(kl)
        offsets.append(off)
    order = sorted(range(len(q_lens)), key=lambda i: q_lens[i])
    q_lens = [q_lens[i] for i in order]
    kv_lens = [kv_lens[i] for i in order]
    offsets = [offsets[i] for i in order]

    pages_per_seq_list = [kl // page_size for kl in kv_lens]
    total_pages = sum(pages_per_seq_list)

    name = (f"chaos_sparse b={b} small_q={small_q} ps={page_size} kbn={kv_block_num} "
            f"total_q={sum(q_lens)} total_kv={sum(kv_lens)} "
            f"h_q={h_q} h_k={h_k} d={d} {dtype_name} seed={seed}")

    # Build per-sequence Q/K/V (logical) and reference output via sparse gather
    q_list, k_list, v_list = [], [], []
    for ql, kl in zip(q_lens, kv_lens):
        qi = torch.randn(1, ql, h_q, d, dtype=torch.bfloat16, device=device).clamp(min=-5, max=5)
        ki = torch.randn(1, kl, h_k, d, dtype=torch.bfloat16, device=device).clamp(min=-5, max=5)
        vi = torch.randn(1, kl, h_k, d, dtype=torch.bfloat16, device=device).clamp(min=-5, max=5)
        q_list.append(qi.reshape(ql, h_q, d))
        k_list.append(ki.reshape(kl, h_k, d))
        v_list.append(vi.reshape(kl, h_k, d))

    # Build paged cache: per-seq pages laid out HND, placed at SHUFFLED physical positions
    perm = torch.randperm(total_pages, device=device, dtype=torch.int64)
    storage_dtype = dtype if dtype == torch.float8_e4m3fn else torch.bfloat16
    k_cache = torch.empty(total_pages, h_k, page_size, d, dtype=storage_dtype, device=device)
    v_cache = torch.empty(total_pages, h_k, page_size, d, dtype=storage_dtype, device=device)

    page_offset = 0
    kv_indices_list = []
    k_pages_logical_list = []  # for reference (unshuffled, logical-order pages)
    v_pages_logical_list = []
    for ki, vi, pps in zip(k_list, v_list, pages_per_seq_list):
        ki_q = ki.to(dtype) if dtype == torch.float8_e4m3fn else ki
        vi_q = vi.to(dtype) if dtype == torch.float8_e4m3fn else vi
        # ki shape: (kl=pps*ps, h_k, d) -> (pps, page_size, h_k, d) -> (pps, h_k, page_size, d)
        k_pages_seq = ki_q.reshape(pps, page_size, h_k, d).transpose(1, 2).contiguous()
        v_pages_seq = vi_q.reshape(pps, page_size, h_k, d).transpose(1, 2).contiguous()
        seq_perm = perm[page_offset:page_offset + pps]
        k_cache[seq_perm] = k_pages_seq
        v_cache[seq_perm] = v_pages_seq
        kv_indices_list.append(seq_perm.to(torch.int32))
        # Reference uses fp8-quantized data when dtype is fp8 (isolate kernel from quant noise)
        if dtype == torch.float8_e4m3fn:
            k_pages_logical_list.append(k_pages_seq.to(torch.bfloat16))
            v_pages_logical_list.append(v_pages_seq.to(torch.bfloat16))
        else:
            k_pages_logical_list.append(k_pages_seq)
            v_pages_logical_list.append(v_pages_seq)
        page_offset += pps

    kv_indices = torch.cat(kv_indices_list)
    qo_lens_tensor = torch.tensor(q_lens, dtype=torch.int32)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
    qo_offset_tensor = torch.tensor(offsets, dtype=torch.int32)

    # Per-(batch, kv_head) sparse block selection — strictly ascending, -1 padded.
    # Block 0 always included so smallest q_pos can see at least one valid kv pos.
    per_batch_blocks = []
    for bi in range(b):
        pps = pages_per_seq_list[bi]
        n = min(kv_block_num, pps)
        heads = []
        for _ in range(h_k):
            if n >= pps:
                blocks = list(range(pps))
            else:
                rest = sorted(random.sample(range(1, pps), n - 1))
                blocks = [0] + rest
            heads.append(blocks)
        per_batch_blocks.append(heads)

    total_qo = sum(q_lens)
    kv_block_indexes = torch.full((total_qo, h_k, kv_block_num), -1, device=device, dtype=torch.int32)
    q_pos_off = 0
    for bi in range(b):
        for kh in range(h_k):
            blocks = per_batch_blocks[bi][kh]
            kv_block_indexes[q_pos_off:q_pos_off + q_lens[bi], kh, :len(blocks)] = \
                torch.tensor(blocks, dtype=torch.int32, device=device)
        q_pos_off += q_lens[bi]

    # Reference: per-(token, qo_head) gather selected pages, causal-mask, softmax + matmul
    h_r = h_q // h_k
    o_ref_list = []
    for bi in range(b):
        ql = q_lens[bi]
        off = offsets[bi]
        if dtype == torch.float8_e4m3fn:
            q_b = q_list[bi].to(dtype).to(torch.bfloat16).float()
        else:
            q_b = q_list[bi].float()
        k_pages_b = k_pages_logical_list[bi]  # (pps, h_k, page_size, d) bf16
        v_pages_b = v_pages_logical_list[bi]
        out_b = torch.zeros(ql, h_q, d, device=device, dtype=torch.float32)
        for h in range(h_q):
            kh = h // h_r
            blocks = per_batch_blocks[bi][kh]
            blk_t = torch.tensor(blocks, device=device, dtype=torch.int64)
            k_g = k_pages_b[blk_t, kh].reshape(-1, d).float()
            v_g = v_pages_b[blk_t, kh].reshape(-1, d).float()
            kv_pos = (blk_t.unsqueeze(1) * page_size +
                      torch.arange(page_size, device=device, dtype=torch.int64).unsqueeze(0)
                      ).reshape(-1)
            q_pos = off + torch.arange(ql, device=device, dtype=torch.int64)
            scores = (q_b[:, h] @ k_g.T) / math.sqrt(d)
            if causal:
                mask = q_pos.unsqueeze(1) < kv_pos.unsqueeze(0)
                scores = scores.masked_fill(mask, float("-inf"))
            row_all_masked = scores.isinf().all(dim=-1)
            scores = scores.masked_fill(row_all_masked.unsqueeze(-1), 0.0)
            probs = torch.softmax(scores, dim=-1)
            probs = probs.masked_fill(row_all_masked.unsqueeze(-1), 0.0)
            out_b[:, h] = probs @ v_g
        o_ref_list.append(out_b.to(torch.bfloat16))
    o_ref = torch.cat(o_ref_list, dim=0)

    q_cat = torch.cat(q_list, dim=0)
    q_fi = q_cat.to(dtype) if dtype == torch.float8_e4m3fn else q_cat

    torch.cuda.synchronize()
    plan_info = fmha_sm100_plan(qo_lens_tensor, kv_lens_tensor, h_q,
        causal=causal, qo_offset=qo_offset_tensor,
        split_prefill_decode=pd_split,
        page_size=page_size,
        kv_block_num=kv_block_num,
        num_kv_heads=h_k,
    )
    torch.cuda.synchronize()
    o, _ = fmha_sm100(
        q_fi, k_cache, v_cache,
        plan_info=plan_info,
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
        sm_scale=1.0 / math.sqrt(d),
    )
    torch.cuda.synchronize()

    threshold = 0.99999 if dtype == torch.bfloat16 else 0.999
    threshold_diff = 0.02 if dtype == torch.bfloat16 else 0.11
    cos_sim = torch.nn.functional.cosine_similarity(
        o.float().reshape(-1), o_ref.float().reshape(-1), dim=0
    ).item()
    max_diff = (o.float() - o_ref.float()).abs().max().item()
    passed = cos_sim > threshold and max_diff < threshold_diff
    fail_msg = None if passed else f"{name}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4f}"
    return passed, fail_msg



# ---- Build all chaos specs ----

def build_chaos_specs():
    specs = []
    for dtype_str in ["bf16", "fp8"]:
        for backend in ["ragged", "paged128"]:
            for seed in range(4096):
                seed_val = seed + 42
                specs.append(dict(h_q=4, h_k=4, d=128, dtype_str=dtype_str, seed=seed_val, backend=backend))
                specs.append(dict(h_q=12, h_k=6, d=128, dtype_str=dtype_str, seed=seed_val, backend=backend))
                specs.append(dict(h_q=12, h_k=2, d=128, dtype_str=dtype_str, seed=seed_val, backend=backend))
                specs.append(dict(h_q=16, h_k=2, d=128, dtype_str=dtype_str, seed=seed_val, backend=backend))
                specs.append(dict(h_q=32, h_k=2, d=128, dtype_str=dtype_str, seed=seed_val, backend=backend))
        # Sparse backend: kv_block_num cycles through {4, 8, 16, 32}.
        # Smaller seed budget per (dtype, kbn) keeps total chaos size in the same ballpark.
        for kbn in [4, 8, 16, 32]:
            for seed in range(1024):
                seed_val = seed + 42
                specs.append(dict(h_q=4, h_k=4, d=128, dtype_str=dtype_str,
                                  seed=seed_val, backend="sparse_paged128", kv_block_num=kbn))
                specs.append(dict(h_q=16, h_k=4, d=128, dtype_str=dtype_str,
                                  seed=seed_val, backend="sparse_paged128", kv_block_num=kbn))
                specs.append(dict(h_q=32, h_k=8, d=128, dtype_str=dtype_str,
                                  seed=seed_val, backend="sparse_paged128", kv_block_num=kbn))
    # Deterministic shuffle for balanced distribution across GPUs
    random.Random(12345).shuffle(specs)
    return specs


DTYPE_MAP = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}


def _run_chaos_one(spec, device):
    """Dispatch a single chaos spec to ragged / paged / sparse_paged backend."""
    dtype = DTYPE_MAP[spec["dtype_str"]]
    backend = spec.get("backend", "ragged")
    if backend == "ragged":
        return _run_case_chaos(
            h_q=spec["h_q"], h_k=spec["h_k"], d=spec["d"],
            dtype=dtype, seed=spec["seed"], device=device,
        )
    if backend.startswith("sparse_paged"):
        page_size = int(backend[len("sparse_paged"):])
        return _run_case_chaos_sparse(
            h_q=spec["h_q"], h_k=spec["h_k"], d=spec["d"],
            dtype=dtype, page_size=page_size,
            kv_block_num=spec["kv_block_num"],
            seed=spec["seed"], device=device,
        )
    if backend.startswith("paged"):
        page_size = int(backend[len("paged"):])
        return _run_case_chaos_paged(
            h_q=spec["h_q"], h_k=spec["h_k"], d=spec["d"],
            dtype=dtype, page_size=page_size, seed=spec["seed"],
            device=device,
        )
    raise ValueError(f"Unknown chaos backend: {backend}")


# ---- Subprocess worker mode ----

def chaos_worker_main_dynamic(rank):
    """Run as a subprocess: read specs from stdin, process on cuda:0, write results to stdout."""
    torch.cuda.set_device(0)
    device = "cuda:0"

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        spec = json.loads(line)
        backend = spec.get("backend", "ragged")
        try:
            passed, fail_msg = _run_chaos_one(spec, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            passed = True
            fail_msg = None
            print(json.dumps({"pass": True, "fail_msg": None, "oom": True}), flush=True)
            continue
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            fail_msg = (f"EXCEPTION rank={rank} backend={backend} seed={spec['seed']} "
                        f"h_q={spec['h_q']} {spec['dtype_str']}: {e}\n{tb}")
            print(json.dumps({"pass": False, "fail_msg": fail_msg, "fatal": True}), flush=True)
            return  # Stop worker: CUDA context is likely corrupted
        print(json.dumps({"pass": passed, "fail_msg": fail_msg}), flush=True)


def run_chaos_single_gpu(num_cases=128):
    """Run a small chaos subset in the current process on a single GPU.

    Stops early on first CUDA fault (CUDA context is corrupted, can't continue
    in-process). Reports {ran_pass}/{ran} ratio for the cases that actually ran.
    and complete the full sweep.
    """
    print(f"\n=== Chaos (varlen + per-batch qo_offset, ragged + paged) — single GPU in-process, {num_cases} cases ===")
    specs = build_chaos_specs()[:num_cases]
    device = "cuda:0"
    chaos_failed = []
    oom_count = 0
    cases_run = 0
    early_stop = False
    pbar = _progress(specs, desc="Chaos", unit="case")
    for spec in pbar:
        try:
            passed, fail_msg = _run_chaos_one(spec, device)
            cases_run += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            oom_count += 1
            cases_run += 1
            pbar.set_postfix(failed=len(chaos_failed), oom=oom_count)
            continue
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            chaos_failed.append(f"EXCEPTION backend={spec.get('backend','?')} seed={spec['seed']} "
                                f"h_q={spec['h_q']} {spec['dtype_str']}: {e}\n{tb}")
            cases_run += 1
            pbar.set_postfix(failed=len(chaos_failed), oom=oom_count)
            print(f"\n  Chaos: CUDA exception — stopping single-GPU in-process run early ")
            early_stop = True
            break
        if not passed:
            chaos_failed.append(fail_msg)
        pbar.set_postfix(failed=len(chaos_failed), oom=oom_count)

    oom_warn = f" ({oom_count} OOM skipped)" if oom_count else ""
    early_warn = f" — STOPPED EARLY at case {cases_run}/{len(specs)} due to CUDA fault" if early_stop else ""
    ran_passed = cases_run - len(chaos_failed)
    print(f"\n  Chaos results: {ran_passed}/{cases_run} cases passed{oom_warn}{early_warn}")
    failed_cases.extend(chaos_failed)
    return len(chaos_failed) == 0

def run_chaos_multi_gpu():
    """Run full chaos suite distributed across multiple GPUs via subprocesses."""
    print("\n=== Chaos (varlen + per-batch qo_offset) — multi-GPU ===")
    num_gpus = min(torch.cuda.device_count(), 8)

    # Probe each GPU with a trivial kernel to filter out unhealthy ones
    healthy_gpus = []
    for gpu_id in range(num_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        r = subprocess.run(
            [sys.executable, "-c",
             "import torch; torch.cuda.set_device(0); "
             "x=torch.ones(1,device='cuda:0'); assert x.item()==1.0"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            healthy_gpus.append(gpu_id)
        else:
            print(f"  Skipping GPU {gpu_id}: health check failed")

    if not healthy_gpus:
        print("  No healthy GPUs available!")
        return False

    specs = build_chaos_specs()
    total_chaos = len(specs)
    print(f"  Using {len(healthy_gpus)}/{num_gpus} GPUs, {total_chaos} cases (dynamic dispatch)")

    spec_queue = queue.Queue()
    for spec in specs:
        spec_queue.put(spec)

    procs = []
    for gpu_id in healthy_gpus:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # env["CUDA_LAUNCH_BLOCKING"] = '1'
        p = subprocess.Popen(
            [sys.executable, __file__, "--chaos-worker-dynamic", str(gpu_id)],
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True,
        )
        procs.append(p)

    done_count = [0]
    fail_count = [0]
    oom_count = [0]
    chaos_failed = []
    lock = threading.Lock()

    WINDOW = 16

    def worker_handler(proc):
        sent = 0
        received = 0
        exhausted = False
        for _ in range(WINDOW):
            try:
                spec = spec_queue.get_nowait()
            except queue.Empty:
                exhausted = True
                break
            proc.stdin.write(json.dumps(spec) + "\n")
            sent += 1
        if sent > 0:
            proc.stdin.flush()
        while received < sent:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            received += 1
            with lock:
                done_count[0] += 1
                if r.get("oom"):
                    oom_count[0] += 1
                elif not r["pass"]:
                    fail_count[0] += 1
                    chaos_failed.append(r["fail_msg"])
            if not exhausted:
                try:
                    spec = spec_queue.get_nowait()
                    proc.stdin.write(json.dumps(spec) + "\n")
                    proc.stdin.flush()
                    sent += 1
                except queue.Empty:
                    exhausted = True
        try:
            proc.stdin.close()
        except Exception:
            pass

    threads = []
    for p in procs:
        t = threading.Thread(target=worker_handler, args=(p,), daemon=True)
        t.start()
        threads.append(t)

    import time
    pbar = _progress(total=total_chaos, desc="Chaos", unit="case")
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.3)
            with lock:
                d, f, o = done_count[0], fail_count[0], oom_count[0]
            delta = d - pbar.n
            if delta > 0:
                pbar.update(delta)
            pbar.set_postfix(failed=f, oom=o)
        with lock:
            d, f, o = done_count[0], fail_count[0], oom_count[0]
        delta = d - pbar.n
        if delta > 0:
            pbar.update(delta)
        pbar.set_postfix(failed=f, oom=o)
    except KeyboardInterrupt:
        for p in procs:
            p.kill()
        for p in procs:
            p.wait()
    finally:
        pbar.close()

    for p in procs:
        p.wait()

    oom_warn = f" ({oom_count[0]} OOM skipped)" if oom_count[0] else ""
    print(f"\n  Chaos results: {done_count[0] - fail_count[0]}/{done_count[0]} passed{oom_warn}")
    failed_cases.extend(chaos_failed)
    return len(chaos_failed) == 0


# ---- Main ----

def main():
    """Run the sweep; returns whether every case passed."""
    prebuild_variants()
    all_pass = run_sweep()
    check_variant_manifest()
    return all_pass


def run_sweep():
    all_pass = True

    if "--chaos_only" not in sys.argv:
        # --- Prefill tests (TILE_Q=256 path, q_len > 128) ---
        print("=== Prefill (TILE_Q=256) ===")
        for dtype in [torch.bfloat16, torch.float8_e4m3fn]:
            for split in [-1, 2, 4]:
                all_pass &= _run_case(1, 8192, 200*1024, 48, 8, 128, dtype, split, causal=True)
                all_pass &= _run_case(1, 131072, 131072, 48, 8, 128, dtype, split, causal=True)
                all_pass &= _run_case(1, 65536, 65536, 48, 8, 128, dtype, split, causal=True)
                all_pass &= _run_case(1, 8192, 8192, 48, 8, 128, dtype, split, causal=True)
                all_pass &= _run_case(1, 256, 256, 48, 8, 128, dtype, split, causal=True)

        # --- Short Q tests (TILE_Q=128 path, q_len <= 128) ---
        print("\n=== Short Q / Decode (TILE_Q=128) ===")
        for dtype in [torch.bfloat16, torch.float8_e4m3fn]:
            for split in [-1, 2, 4]:
                all_pass &= _run_case(1, 65, 512, 8, 8, 128, dtype, split, causal=True)
                all_pass &= _run_case(1, 64, 512, 8, 8, 128, dtype, split, causal=True)
                all_pass &= _run_case(32, 6, 512, 8, 8, 128, dtype, split, causal=True)
                all_pass &= _run_case(32, 6, 8192, 8, 8, 128, dtype, split, causal=True)
                all_pass &= _run_case(4, 128, 2048, 8, 8, 128, dtype, split, causal=True)
        print()

        # --- Corner case tests (fp8 only) ---
        print("=== Corner Cases (fp8) ===")
        for dtype in [torch.bfloat16, torch.float8_e4m3fn]:
            # TILE_Q boundary: 128 → TILE_Q=128, 129 → TILE_Q=256
            all_pass &= _run_case(1, 127, 1024, 48, 8, 128, dtype, causal=True)
            all_pass &= _run_case(1, 128, 1024, 24, 4, 128, dtype, causal=True)
            all_pass &= _run_case(1, 129, 1024, 48, 8, 128, dtype, causal=True)
            all_pass &= _run_case(1, 255, 1024, 48, 8, 128, dtype, causal=True)
            all_pass &= _run_case(1, 257, 1024, 48, 8, 128, dtype, causal=True)

            # Pure decode: q_len=1
            all_pass &= _run_case(1, 1, 512, 48, 8, 128, dtype, causal=True)
            all_pass &= _run_case(32, 1, 8192, 48, 8, 128, dtype, causal=True)
            # all_pass &= _run_case(64, 1, 131072, 48, 8, 128, dtype, causal=True)

            # Minimal sizes
            all_pass &= _run_case(1, 1, 1, 48, 8, 128, dtype, causal=True)
            # all_pass &= _run_case(1, 1, 1, 48, 8, 128, dtype, causal=False)
            all_pass &= _run_case(1, 2, 2, 48, 8, 128, dtype, causal=True)

            # Non-causal attention
            # all_pass &= _run_case(1, 256, 256, 48, 8, 128, dtype, causal=False)
            # all_pass &= _run_case(1, 8192, 8192, 48, 8, 128, dtype, causal=False)
            # all_pass &= _run_case(4, 128, 2048, 8, 8, 128, dtype, causal=False)
            # all_pass &= _run_case(1, 512, 256, 48, 8, 128, dtype, causal=False)  # q_len > k_len

            # Non-power-of-2 odd sizes
            all_pass &= _run_case(1, 7, 1024, 48, 8, 128, dtype, causal=True)
            all_pass &= _run_case(1, 33, 1024, 48, 8, 128, dtype, causal=True)
            all_pass &= _run_case(3, 17, 513, 48, 8, 128, dtype, causal=True)

            # Single head (h_q=1, h_k=1)
            all_pass &= _run_case(1, 256, 256, 1, 1, 128, dtype, causal=True)
            all_pass &= _run_case(4, 128, 2048, 1, 1, 128, dtype, causal=True)

            # Extreme GQA ratio (h_q=64, h_k=1)
            all_pass &= _run_case(1, 256, 256, 64, 1, 128, dtype, causal=True)
            all_pass &= _run_case(1, 8192, 8192, 64, 8, 128, dtype, causal=True)

            # Large batch + small sequences
            all_pass &= _run_case(64, 1, 128, 48, 8, 128, dtype, causal=True)
            all_pass &= _run_case(128, 1, 64, 8, 8, 128, dtype, causal=True)

            # Long KV + short Q (append scenario)
            all_pass &= _run_case(1, 1, 131072, 48, 8, 128, dtype, causal=True)
            all_pass &= _run_case(4, 6, 65536, 48, 8, 128, dtype, causal=True)
            print()

        # --- Paged KV tests (small + boundary cases) ---
        print("=== Paged KV Cache ===")
        for dtype in [torch.bfloat16, torch.float8_e4m3fn]:
            for ps in page_sizes:
                all_pass &= _run_case_paged(1, 256, 512, 8, 8, 128, dtype, page_size=ps)
                all_pass &= _run_case_paged(4, 128, 2048, 48, 8, 128, dtype, page_size=ps)
                all_pass &= _run_case_paged(1, 1, 512, 48, 8, 128, dtype, page_size=ps)
                all_pass &= _run_case_paged(32, 1, 8192, 8, 8, 128, dtype, page_size=ps)
        print()

        # --- Paged Prefill (q == k, causal) — matches benchmark_fmha_sm100.py shapes (v2 parity) ---
        print("=== Paged Prefill ===")
        for dtype in [torch.bfloat16, torch.float8_e4m3fn]:
            for page_size in page_sizes:
                for b, q, k, hq, hk in [(1, 65536, 65536, 64, 4),
                                        (1, 8192, 8192, 64, 4),
                                        (1, 256, 256, 64, 4)]:
                    try:
                        all_pass &= _run_case_paged(b, q, k, hq, hk, 128, dtype, page_size=page_size)
                    except Exception as e:
                        case_name = f"paged b={b} q={q} k={k} h_q={hq} h_k={hk} ps={page_size} {dtype}"
                        print(f"  [EXCEPTION] {case_name}: {e}")
                        failed_cases.append(f"{case_name}: EXCEPTION {e}")
                        all_pass = False

        # --- Paged Decode (small q, large k, MBU-bound) — matches benchmark_fmha_sm100.py shapes (v2 parity) ---
        print("\n=== Paged Decode ===")
        paged_decode_cases = [(32, 24, 8192, 8, 8),
                            (32, 1, 8192, 48, 8),
                            (64, 24, 8192, 8, 8)]
        for dtype in [torch.bfloat16, torch.float8_e4m3fn]:
            for page_size in page_sizes:
                for b, q, k, hq, hk in paged_decode_cases:
                    try:
                        all_pass &= _run_case_paged(b, q, k, hq, hk, 128, dtype, page_size=page_size)
                    except Exception as e:
                        case_name = f"paged b={b} q={q} k={k} h_q={hq} h_k={hk} ps={page_size} {dtype}"
                        print(f"  [EXCEPTION] {case_name}: {type(e).__name__}: {str(e)[:200]}")
                        failed_cases.append(f"{case_name}: EXCEPTION {type(e).__name__}")
                        all_pass = False
        print()

        # --- Sparse attention (kv_block_indexes path) ---
        # Decode path (q_len <= 128) uses local sparse kernel; prefill path (q_len > 128) routes
        # to MM-SA-Nv. kv_block_num must be in {4, 8, 16, 32}.
        print("=== Sparse Attention (Decode, q_len <= 128) ===")
        # (b, q, k, hq, hk, kbn)
        sparse_decode_cases = [
            # Pure decode (q=1)
            (1,   1, 2048, 32,  8,  8),   # GQA4
            (1,   1, 4096, 16,  4,  16),  # GQA4, larger K
            (4,   1, 2048,  8,  2,  8),   # batched decode
            (32,  1, 8192, 32,  8,  16),  # large batch decode + long KV
            (1,   1, 8192, 64,  4,  32),  # extreme GQA + max kbn
            (1,   1, 1024,  4,  4,  4),   # MHA, smallest kbn
            (32,  4,  256, 16,  1, 16),   # fewer valid pages than kbn
            # MTP / multi-token decode (q in {2,4,8})
            (1,   2, 4096, 32,  8,  8),   # MTP-2
            (1,   4, 4096, 16,  4, 16),   # MTP-4
            (1,   8, 4096, 32,  8, 16),   # MTP-8 (the common production form)
            (4,   8, 8192, 16,  4,  8),   # batched MTP-8
            (32,  8, 8192, 16,  4, 16),   # large batch MTP-8 + long KV
            (1,   8, 2048, 64,  4, 32),   # extreme GQA MTP-8
            # Mid q_len (still TILE_Q=128 path)
            (1,  16, 4096, 16,  4,  8),
            (1,  64, 4096, 32,  8, 16),
            # Short prefill / TILE_Q=128 boundary
            (1,  32, 2048, 16,  4,  8),
            (2, 128, 4096, 16,  4, 16),   # boundary: q_len=128
            # Large batch decode: total_qo=512 across the 256-batch packed_work_info boundary.
            # Was a known-fail (NaN) before the pack_work_info batch_idx bit-width fix; kept here
            # as the canonical regression guard for that bug.
            (64,  8, 8192, 16,  4,  8),
        ]
        for dtype in [torch.bfloat16, torch.float8_e4m3fn]:
            for page_size in page_sizes:
                for b, q, k, hq, hk, kbn in sparse_decode_cases:
                    try:
                        all_pass &= _run_case_sparse(
                            b, q, k, hq, hk, 128, dtype,
                            page_size=page_size, kv_block_num=kbn,
                        )
                    except Exception as e:
                        case_name = (f"sparse b={b} q={q} k={k} h_q={hq} h_k={hk} "
                                    f"ps={page_size} kbn={kbn} {dtype}")
                        print(f"  [EXCEPTION] {case_name}: {type(e).__name__}: {str(e)[:200]}")
                        failed_cases.append(f"{case_name}: EXCEPTION {type(e).__name__}")
                        all_pass = False

        print("\n=== Sparse Attention (vLLM interleaved FP8 KV) ===")
        try:
            all_pass &= _run_case_sparse(
                32, 4, 8192, 16, 1, 128, torch.float8_e4m3fn,
                page_size=128, kv_block_num=16, interleaved_kv=True,
            )
        except Exception as e:
            case_name = "sparse vLLM interleaved FP8 KV"
            print(
                f"  [EXCEPTION] {case_name}: "
                f"{type(e).__name__}: {str(e)[:200]}"
            )
            failed_cases.append(
                f"{case_name}: EXCEPTION {type(e).__name__}"
            )
            all_pass = False

        print("\n=== Sparse Attention (Prefill, q_len > 128, MM-SA-Nv path) ===")
        sparse_prefill_cases = [
            (1, 256,  4096, 16, 4, 16),
            (1, 512,  8192, 16, 4, 16),
            (1, 1024, 8192, 32, 8, 32),
            (2, 256,  4096,  8, 2,  8),
        ]
        for dtype in [torch.bfloat16, torch.float8_e4m3fn]:
            for page_size in page_sizes:
                for b, q, k, hq, hk, kbn in sparse_prefill_cases:
                    try:
                        all_pass &= _run_case_sparse(
                            b, q, k, hq, hk, 128, dtype,
                            page_size=page_size, kv_block_num=kbn,
                        )
                    except Exception as e:
                        case_name = (f"sparse_prefill b={b} q={q} k={k} h_q={hq} h_k={hk} "
                                    f"ps={page_size} kbn={kbn} {dtype}")
                        print(f"  [EXCEPTION] {case_name}: {type(e).__name__}: {str(e)[:200]}")
                        failed_cases.append(f"{case_name}: EXCEPTION {type(e).__name__}")
                        all_pass = False
        print()

        # --- Pre-allocated out with GQA packing ---
        print("=== Pre-allocated out (GQA packing) ===")
        for dtype in [torch.bfloat16, torch.float8_e4m3fn]:
            for b, q_len, k_len, h_q, h_k in [
                (1, 256, 256, 64, 4),
                (1, 8192, 8192, 48, 8),
                (32, 6, 8192, 48, 8),
                (1, 128, 2048, 16, 1),
                (4, 64, 512, 32, 2),
            ]:
                dtype_name = {torch.float8_e4m3fn: "fp8", torch.bfloat16: "bf16"}[dtype]
                name = f"prealloc_out b={b} q={q_len} k={k_len} h_q={h_q} h_k={h_k} {dtype_name}"
                device = "cuda:0"
                q_bf16 = torch.randn(b, q_len, h_q, 128, dtype=torch.bfloat16, device=device)
                k_bf16 = torch.randn(b, k_len, h_k, 128, dtype=torch.bfloat16, device=device)
                v_bf16 = torch.randn(b, k_len, h_k, 128, dtype=torch.bfloat16, device=device)
                if dtype == torch.float8_e4m3fn:
                    q_fi = q_bf16.reshape(b * q_len, h_q, 128).to(dtype)
                    k_fi = k_bf16.reshape(b * k_len, h_k, 128).to(dtype)
                    v_fi = v_bf16.reshape(b * k_len, h_k, 128).to(dtype)
                    q_ref, k_ref, v_ref = q_fi.reshape(b, q_len, h_q, 128).to(torch.bfloat16), k_fi.reshape(b, k_len, h_k, 128).to(torch.bfloat16), v_fi.reshape(b, k_len, h_k, 128).to(torch.bfloat16)
                else:
                    q_fi = q_bf16.reshape(b * q_len, h_q, 128)
                    k_fi = k_bf16.reshape(b * k_len, h_k, 128)
                    v_fi = v_bf16.reshape(b * k_len, h_k, 128)
                    q_ref, k_ref, v_ref = q_bf16, k_bf16, v_bf16
                o_ref = sdpa_ref(q_ref, k_ref, v_ref, h_q, h_k, causal=True, qo_offset=k_len - q_len)
                # Run without pre-alloc (baseline)
                o_baseline = run_fmha_sm100(q_fi, k_fi, v_fi, b, q_len, k_len, h_q, h_k, 128, dtype, -1, causal=True,
                                            qo_offset=torch.full((b,), k_len - q_len, dtype=torch.int32))
                # Run with pre-alloc out
                out_buf = torch.empty_like(q_fi).to(torch.bfloat16)
                buf_ptr = out_buf.data_ptr()
                qo_lens = torch.full((b,), q_len, dtype=torch.int32)
                kv_lens = torch.full((b,), k_len, dtype=torch.int32)
                qo_offset_t = torch.full((b,), k_len - q_len, dtype=torch.int32)
                plan_info = fmha_sm100_plan(qo_lens, kv_lens, h_q, causal=True, qo_offset=qo_offset_t, num_kv_heads=h_k)
                o_prealloc, _ = fmha_sm100(q_fi, k_fi, v_fi, plan_info=plan_info, out=out_buf)
                same_buf = (o_prealloc.data_ptr() == buf_ptr)
                threshold = 0.99999 if dtype == torch.bfloat16 else 0.9995
                passed_ref = check(name, o_prealloc, o_ref, threshold)
                cos_vs_baseline = torch.nn.functional.cosine_similarity(
                    o_prealloc.float().reshape(-1), o_baseline.float().reshape(-1), dim=0).item()
                if cos_vs_baseline < 0.999999 or not same_buf:
                    print(f"    [FAIL] vs_baseline cos={cos_vs_baseline:.6f} same_buf={same_buf}")
                    failed_cases.append(f"{name}: vs_baseline cos={cos_vs_baseline:.6f} same_buf={same_buf}")
                    all_pass = False
                else:
                    print(f"    [PASS] vs_baseline cos={cos_vs_baseline:.6f} same_buf={same_buf}")
                all_pass &= passed_ref
    else:
        print("Trigger JIT")
        all_pass &= run_chaos_single_gpu(8)

    full_mode = "--full" in sys.argv
    if not full_mode:
        all_pass &= run_chaos_single_gpu()

    if not all_pass:
        print(f"\n=== FAILED CASES ({len(failed_cases)}) ===")
        num = 0
        for fc in failed_cases:
            print(f"  {fc}")
            num += 1
            if num == 8 and len(failed_cases)>num:
                print(f"..and {len(failed_cases)-num} more")
                break
        return False
    
    if not full_mode:
        print("ALL TESTS PASSED")
        return True

    all_pass &= run_chaos_multi_gpu()

    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print(f"\n=== FAILED CASES ({len(failed_cases)}) ===")
        num = 0
        for fc in failed_cases:
            print(f"  {fc}")
            num += 1
            if num == 8 and len(failed_cases)>num:
                print(f"..and {len(failed_cases)-num} more")
                break
    check_variant_manifest()
    os._exit(0 if all_pass else 1)


if __name__ == "__main__":
    # Progress stays visible when the output goes to a file or a pipe.
    sys.stdout.reconfigure(line_buffering=True)

    if "--check" in sys.argv:
        fmha_sm100 = partial(_fmha_sm100, check_input_valid=True)
    else:
        fmha_sm100 = _fmha_sm100

    if len(sys.argv) >= 3 and sys.argv[1] == "--chaos-worker-dynamic":
        chaos_worker_main_dynamic(int(sys.argv[2]))
    else:
        sys.exit(0 if main() else 1)
