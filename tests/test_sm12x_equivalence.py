# SPDX-License-Identifier: MIT

"""SM12x equivalents of the SM100 behavioural tests.

Each test mirrors a behaviour the SM100 suite checks (proxy-KV pipeline,
onlyscore output, q-offset override, paged sparse attention, the FP4 indexer,
and paged decode) against an independent Torch oracle, using only the
``fmha_sm12x`` public surface. FP8/tcgen05-specific SM100 tests have no SM12x
analog and are intentionally absent.
"""

from __future__ import annotations

import math

import pytest
import torch


def _need_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _sparse_ref_dense(q, k, v, block_ids, cu_q, cu_k, *, page_size, sm_scale, causal):
    """Independent oracle: per-(token, kv_head) block-sparse attention.

    ``block_ids`` is ``[total_q, Hkv, topk]`` (batch-local block ids, -1 pad),
    shared across each GQA group. Returns float32 ``[total_q, Hq, D]``.
    """

    total_q, num_qo_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    h_ratio = num_qo_heads // num_kv_heads
    out = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
    cuq = cu_q.tolist()
    cuk = cu_k.tolist()
    for b in range(len(cuq) - 1):
        q0, q1, k0, k1 = cuq[b], cuq[b + 1], cuk[b], cuk[b + 1]
        kv_len, qo_len = k1 - k0, q1 - q0
        for lq in range(qo_len):
            qi = q0 + lq
            vis = (kv_len - qo_len) + lq
            for h in range(num_qo_heads):
                kvh = h // h_ratio
                pos = []
                for bid in block_ids[qi, kvh].tolist():
                    if bid < 0 or bid * page_size >= kv_len:
                        continue
                    for p in range(bid * page_size, min(bid * page_size + page_size, kv_len)):
                        if (not causal) or p <= vis:
                            pos.append(p)
                if not pos:
                    continue
                idx = torch.tensor(pos, device=q.device, dtype=torch.long)
                k_sel = k[k0:k1][idx, kvh].float()
                v_sel = v[k0:k1][idx, kvh].float()
                logits = (k_sel @ q[qi, h].float()) * sm_scale
                out[qi, h] = torch.softmax(logits, dim=0) @ v_sel
    return out


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()


def test_proxy_kv_e2e_pipeline_matches_reference():
    # max_score (dense proxy) -> sparse_topk_select -> CSR -> sparse_atten_func,
    # all via fmha_sm12x, vs a Torch oracle over the kernel-selected blocks.
    _need_cuda()
    from fmha_sm12x import (
        build_k2q_csr,
        fmha_sm12x,
        fmha_sm12x_plan,
        sparse_atten_func,
        sparse_topk_select,
    )

    torch.manual_seed(0)
    dev = torch.device("cuda")
    total_q, num_kv_heads, h_ratio, head_dim, page_size, topk = 4, 2, 2, 128, 128, 16
    num_qo_heads = num_kv_heads * h_ratio
    n_pages = 20
    kv_len = n_pages * page_size
    cu_q = torch.tensor([0, total_q], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device=dev)

    # Proxy cache (num_qo_heads_proxy == num_kv_heads_real, MQA-compressed).
    proxy_q = torch.randn(total_q, num_kv_heads, head_dim, device=dev, dtype=torch.bfloat16) * 0.3
    proxy_k = torch.randn(kv_len, 1, head_dim, device=dev, dtype=torch.bfloat16) * 0.3
    proxy_v = torch.randn_like(proxy_k)
    sm_scale = 1.0 / math.sqrt(head_dim)

    proxy_plan = fmha_sm12x_plan(
        cu_q[1:] - cu_q[:-1], cu_k[1:] - cu_k[:-1], num_kv_heads, 1,
        page_size=page_size, output_maxscore=True, causal=True,
    )
    _, max_score = fmha_sm12x(
        proxy_q, proxy_k, proxy_v, proxy_plan, sm_scale=sm_scale,
        output_o=False, output_maxscore=True,
    )
    assert max_score is not None and max_score.shape[0] == num_kv_heads

    block_ids = sparse_topk_select(max_score.contiguous(), topk, num_valid_pages=n_pages)
    assert block_ids.shape == (total_q, num_kv_heads, topk)

    # Real (GQA) cache, attended sparsely with the selected blocks.
    real_k = torch.randn(kv_len, num_kv_heads, head_dim, device=dev, dtype=torch.bfloat16) * 0.3
    real_v = torch.randn_like(real_k)
    real_q = torch.randn(total_q, num_qo_heads, head_dim, device=dev, dtype=torch.bfloat16) * 0.3
    q2k = block_ids.permute(1, 0, 2).contiguous()
    row_ptr, q_idx = build_k2q_csr(q2k, cu_q, cu_k, page_size)
    out = sparse_atten_func(
        real_q, real_k, real_v, row_ptr, q_idx, topk,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=total_q, max_seqlen_k=kv_len,
        blk_kv=page_size, causal=True, softmax_scale=sm_scale,
    )
    assert out.shape == (total_q, num_qo_heads, head_dim) and not out.isnan().any()

    ref = _sparse_ref_dense(
        real_q, real_k, real_v, block_ids, cu_q, cu_k,
        page_size=page_size, sm_scale=sm_scale, causal=True,
    )
    assert _cos(out, ref) > 0.999


def test_onlyscore_matches_full_run_and_skips_output():
    # output_maxscore path: score is independent of output_o, O is skipped when
    # output_o=False, and the score equals a Torch per-tile max-logit reference.
    _need_cuda()
    from fmha_sm12x import fmha_sm12x, fmha_sm12x_plan

    torch.manual_seed(1)
    dev = torch.device("cuda")
    total_q, num_heads, head_dim, page_size = 3, 2, 128, 128
    kv_len = 3 * page_size
    cu_q = torch.tensor([0, total_q], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device=dev)
    q = torch.randn(total_q, num_heads, head_dim, device=dev, dtype=torch.bfloat16) * 0.3
    k = torch.randn(kv_len, num_heads, head_dim, device=dev, dtype=torch.bfloat16) * 0.3
    v = torch.randn_like(k)
    sm_scale = 1.0 / math.sqrt(head_dim)

    plan = fmha_sm12x_plan(
        cu_q[1:] - cu_q[:-1], cu_k[1:] - cu_k[:-1], num_heads, num_heads,
        page_size=page_size, output_maxscore=True, causal=True,
    )
    o_full, score_full = fmha_sm12x(q, k, v, plan, sm_scale=sm_scale, output_o=True, output_maxscore=True)
    o_none, score_only = fmha_sm12x(q, k, v, plan, sm_scale=sm_scale, output_o=False, output_maxscore=True)

    assert o_full is not None and o_none is None
    torch.testing.assert_close(score_only, score_full, atol=0, rtol=0)

    # Torch reference: per 128-token tile, max causal logit.
    n_tiles = kv_len // page_size
    ref = torch.full((num_heads, n_tiles, total_q), float("-inf"), device=dev, dtype=torch.float32)
    for qi in range(total_q):
        vis = (kv_len - total_q) + qi
        for h in range(num_heads):
            logits = (k[:, h].float() @ q[qi, h].float()) * sm_scale
            for t in range(n_tiles):
                seg = logits[t * page_size : (t + 1) * page_size]
                mask = torch.arange(t * page_size, (t + 1) * page_size, device=dev) <= vis
                if mask.any():
                    ref[h, t, qi] = seg[mask].max()
    finite = torch.isfinite(ref)
    torch.testing.assert_close(score_full[finite], ref[finite], atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("override", ["int", "tensor", "none"])
def test_q_offset_override_matches_explicit_plan(override):
    # fmha_sm12x(q_offset_override=X) must equal a plan built with qo_offset=X.
    _need_cuda()
    from fmha_sm12x import fmha_sm12x, fmha_sm12x_plan

    torch.manual_seed(2)
    dev = torch.device("cuda")
    total_q, num_heads, head_dim = 4, 2, 16
    kv_len = 12
    cu_q = torch.tensor([0, total_q], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device=dev)
    q = torch.randn(total_q, num_heads, head_dim, device=dev, dtype=torch.bfloat16)
    k = torch.randn(kv_len, num_heads, head_dim, device=dev, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    qo_lens = cu_q[1:] - cu_q[:-1]
    kv_lens = cu_k[1:] - cu_k[:-1]
    sm_scale = 1.0 / math.sqrt(head_dim)

    if override == "int":
        ov = 5
        ref_plan = fmha_sm12x_plan(qo_lens, kv_lens, num_heads, num_heads, qo_offset=5, causal=True)
    elif override == "tensor":
        ov = torch.tensor([6], dtype=torch.int32)
        ref_plan = fmha_sm12x_plan(qo_lens, kv_lens, num_heads, num_heads, qo_offset=ov, causal=True)
    else:
        ov = None
        ref_plan = fmha_sm12x_plan(qo_lens, kv_lens, num_heads, num_heads, causal=True)

    base_plan = fmha_sm12x_plan(qo_lens, kv_lens, num_heads, num_heads, causal=True)
    out_ov, _ = fmha_sm12x(q, k, v, base_plan, q_offset_override=ov, sm_scale=sm_scale)
    out_ref, _ = fmha_sm12x(q, k, v, ref_plan, sm_scale=sm_scale)
    torch.testing.assert_close(out_ov, out_ref, atol=0, rtol=0)


def test_paged_sparse_attention_matches_dense():
    # sparse_atten_func paged-KV reference path must equal the dense path for
    # the same logical KV (the Triton fast path is dense-only, so this exercises
    # the paged Torch fallback).
    _need_cuda()
    from fmha_sm12x import build_k2q_csr, sparse_atten_func

    torch.manual_seed(3)
    dev = torch.device("cuda")
    total_q, num_kv_heads, h_ratio, head_dim, page_size, topk = 2, 1, 2, 128, 128, 4
    num_qo_heads = num_kv_heads * h_ratio
    n_pages = 4
    kv_len = n_pages * page_size
    cu_q = torch.tensor([0, total_q], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device=dev)
    q = torch.randn(total_q, num_qo_heads, head_dim, device=dev, dtype=torch.bfloat16) * 0.3
    k_dense = torch.randn(kv_len, num_kv_heads, head_dim, device=dev, dtype=torch.bfloat16) * 0.3
    v_dense = torch.randn_like(k_dense)
    sm_scale = 1.0 / math.sqrt(head_dim)

    # Same logical KV, paged as [pages, Hkv, page_size, D] with an identity table.
    k_paged = k_dense.reshape(n_pages, page_size, num_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    v_paged = v_dense.reshape(n_pages, page_size, num_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    page_table = torch.arange(n_pages, device=dev, dtype=torch.int32).reshape(1, n_pages)

    q2k = torch.full((num_kv_heads, total_q, topk), -1, dtype=torch.int32, device=dev)
    q2k[0, :, 0] = 0
    q2k[0, :, 1] = 2
    row_ptr, q_idx = build_k2q_csr(q2k, cu_q, cu_k, page_size)
    common = dict(
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=total_q, max_seqlen_k=kv_len,
        blk_kv=page_size, causal=True, softmax_scale=sm_scale,
    )
    out_dense = sparse_atten_func(q, k_dense, v_dense, row_ptr, q_idx, topk, **common)
    out_paged = sparse_atten_func(q, k_paged, v_paged, row_ptr, q_idx, topk, page_table=page_table, **common)
    torch.testing.assert_close(out_paged.float(), out_dense.float(), atol=2e-2, rtol=2e-2)


def test_fp4_indexer_block_scores_matches_dequant_reference():
    # fp4_indexer_block_scores (public scales) must equal a Torch dequant + per
    # 128-tile max-logit reference.
    _need_cuda()
    from fmha_sm12x import fp4_indexer_block_scores
    from fmha_sm12x._fp4 import _FP4_VALUES

    dev = torch.device("cuda")
    groups = 8  # nvfp4 scale groups per 128-dim row
    nibble = 2  # -> fp4 value 1.0
    q_fp4 = torch.full((1, 1, 64), nibble | (nibble << 4), dtype=torch.uint8, device=dev)
    k_fp4 = torch.full((1, 1, 128, 64), nibble | (nibble << 4), dtype=torch.uint8, device=dev)
    q_scale = torch.full((1, 1, groups), 1.0, device=dev).to(torch.float8_e4m3fn)
    k_scale = torch.full((1, 1, 128, groups), 1.0, device=dev).to(torch.float8_e4m3fn)
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, 128], dtype=torch.int32, device=dev)
    page_offsets = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    kv_indices = torch.tensor([0], dtype=torch.int32, device=dev)

    scores = fp4_indexer_block_scores(
        q_fp4, k_fp4, q_scale, k_scale, cu_q, cu_k, page_offsets,
        max_seqlen_q=1, max_seqlen_k=128, kv_indices=kv_indices, fp4_format="nvfp4", causal=False,
    )
    # Dequant: every element is fp4 value 1.0 with scale 1.0 -> q=k=1.0 over 128
    # dims, so each logit = 128. Single tile -> max score 128.
    val = _FP4_VALUES[nibble]
    expected = float(val * val * 128)
    assert scores.shape == (1, 1, 1)
    torch.testing.assert_close(scores[0, 0, 0], torch.tensor(expected, device=dev), atol=1e-3, rtol=1e-3)


def _build_dense_sparse_case(dev, *, dtype=torch.bfloat16):
    total_q, num_kv_heads, h_ratio, head_dim, page_size, topk = 3, 1, 4, 128, 128, 4
    num_qo_heads = num_kv_heads * h_ratio
    n_pages = 4
    kv_len = n_pages * page_size
    cu_q = torch.tensor([0, total_q], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device=dev)
    q = (torch.randn(total_q, num_qo_heads, head_dim, device=dev, dtype=torch.bfloat16) * 0.3).to(dtype)
    k = torch.randn(kv_len, num_kv_heads, head_dim, device=dev, dtype=torch.bfloat16) * 0.3
    v = torch.randn_like(k)
    q2k = torch.full((num_kv_heads, total_q, topk), -1, dtype=torch.int32, device=dev)
    q2k[0, :, 0] = 0
    q2k[0, :, 1] = 3
    return dict(
        q=q, k=k, v=v, q2k=q2k, cu_q=cu_q, cu_k=cu_k, topk=topk,
        page_size=page_size, kv_len=kv_len, total_q=total_q,
        sm_scale=1.0 / math.sqrt(head_dim),
    )


def _fp8_available():
    return hasattr(torch, "float8_e4m3fn")


def test_sparse_atten_fp8_kv_matches_dequantized_bf16():
    # bf16 Q + fp8 E4M3 K/V cache must equal the explicitly dequantized-bf16
    # K/V (SM12x stages fp8 -> bf16, so this is exact).
    _need_cuda()
    if not _fp8_available():
        pytest.skip("float8_e4m3fn unavailable")
    from fmha_sm12x import build_k2q_csr, sparse_atten_func

    torch.manual_seed(5)
    c = _build_dense_sparse_case(torch.device("cuda"))
    row_ptr, q_idx = build_k2q_csr(c["q2k"], c["cu_q"], c["cu_k"], c["page_size"])
    k_fp8 = c["k"].to(torch.float8_e4m3fn)
    v_fp8 = c["v"].to(torch.float8_e4m3fn)
    common = dict(
        cu_seqlens_q=c["cu_q"], cu_seqlens_k=c["cu_k"], max_seqlen_q=c["total_q"],
        max_seqlen_k=c["kv_len"], blk_kv=c["page_size"], causal=True, softmax_scale=c["sm_scale"],
    )
    out_fp8 = sparse_atten_func(c["q"], k_fp8, v_fp8, row_ptr, q_idx, c["topk"], **common)
    out_ref = sparse_atten_func(
        c["q"], k_fp8.to(torch.bfloat16), v_fp8.to(torch.bfloat16), row_ptr, q_idx, c["topk"], **common
    )
    torch.testing.assert_close(out_fp8, out_ref, atol=0, rtol=0)


def test_sparse_atten_fp8_qkv_matches_dequantized_bf16():
    # All-fp8 Q/K/V must equal the dequantized-bf16 inputs.
    _need_cuda()
    if not _fp8_available():
        pytest.skip("float8_e4m3fn unavailable")
    from fmha_sm12x import build_k2q_csr, sparse_atten_func

    torch.manual_seed(6)
    c = _build_dense_sparse_case(torch.device("cuda"))
    row_ptr, q_idx = build_k2q_csr(c["q2k"], c["cu_q"], c["cu_k"], c["page_size"])
    q_fp8 = c["q"].to(torch.float8_e4m3fn)
    k_fp8 = c["k"].to(torch.float8_e4m3fn)
    v_fp8 = c["v"].to(torch.float8_e4m3fn)
    common = dict(
        cu_seqlens_q=c["cu_q"], cu_seqlens_k=c["cu_k"], max_seqlen_q=c["total_q"],
        max_seqlen_k=c["kv_len"], blk_kv=c["page_size"], causal=True, softmax_scale=c["sm_scale"],
    )
    out_fp8 = sparse_atten_func(q_fp8, k_fp8, v_fp8, row_ptr, q_idx, c["topk"], **common)
    out_ref = sparse_atten_func(
        q_fp8.to(torch.bfloat16), k_fp8.to(torch.bfloat16), v_fp8.to(torch.bfloat16),
        row_ptr, q_idx, c["topk"], **common,
    )
    torch.testing.assert_close(out_fp8, out_ref, atol=0, rtol=0)


def test_sparse_atten_rejects_unsupported_mixed_dtypes():
    # fp16 Q with fp8 K (not an SM100-supported combination) is rejected.
    _need_cuda()
    if not _fp8_available():
        pytest.skip("float8_e4m3fn unavailable")
    from fmha_sm12x import build_k2q_csr, sparse_atten_func

    c = _build_dense_sparse_case(torch.device("cuda"), dtype=torch.float16)
    row_ptr, q_idx = build_k2q_csr(c["q2k"], c["cu_q"], c["cu_k"], c["page_size"])
    with pytest.raises(TypeError, match="share a dtype"):
        sparse_atten_func(
            c["q"], c["k"].to(torch.float8_e4m3fn), c["v"].to(torch.float8_e4m3fn),
            row_ptr, q_idx, c["topk"], cu_seqlens_q=c["cu_q"], cu_seqlens_k=c["cu_k"],
            max_seqlen_q=c["total_q"], max_seqlen_k=c["kv_len"], blk_kv=c["page_size"], causal=True,
        )


def test_sparse_atten_nvfp4_kv_matches_dequantized_bf16():
    # The NVFP4 K/V entry must equal sparse_atten_func on the dequantized BF16
    # K/V (mirrors the SM100 nvfp4-matches-dequant check).
    _need_cuda()
    from fmha_sm12x import build_k2q_csr, sparse_atten_func, sparse_atten_nvfp4_kv_func
    from fmha_sm12x._nvfp4 import dequantize_nvfp4_128x4

    dev = torch.device("cuda")
    head_dim, page_size, topk = 128, 128, 4
    kv_len = page_size  # one page
    q = torch.ones((1, 1, head_dim), dtype=torch.bfloat16, device=dev)
    k_fp4 = torch.full((kv_len, 1, 64), 2 | (2 << 4), dtype=torch.uint8, device=dev)
    v_fp4 = torch.full((kv_len, 1, 64), 4 | (4 << 4), dtype=torch.uint8, device=dev)
    scale = torch.ones((kv_len, 8), device=dev).to(torch.float8_e4m3fn)
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device=dev)
    q2k = torch.tensor([[[0, -1, -1, -1]]], dtype=torch.int32, device=dev)
    row_ptr, q_idx = build_k2q_csr(q2k, cu_q, cu_k, page_size)
    common = dict(
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=1, max_seqlen_k=kv_len,
        blk_kv=page_size, causal=False, softmax_scale=1.0 / math.sqrt(head_dim),
    )

    out_nvfp4 = sparse_atten_nvfp4_kv_func(
        q, k_fp4, v_fp4, scale, scale, None, None, row_ptr, q_idx, topk, **common
    )
    logical = (*k_fp4.shape[:-1], head_dim)
    k_bf16 = dequantize_nvfp4_128x4(k_fp4, scale, None, original_shape=logical)
    v_bf16 = dequantize_nvfp4_128x4(v_fp4, scale, None, original_shape=logical)
    out_ref = sparse_atten_func(q, k_bf16, v_bf16, row_ptr, q_idx, topk, **common)
    torch.testing.assert_close(out_nvfp4, out_ref, atol=0, rtol=0)


def test_sparse_decode_full_matches_torch_sdpa():
    # sparse_decode_atten_func with no block selection == dense causal SDPA over
    # seqused_k tokens (paged KV decode reference).
    _need_cuda()
    from fmha_sm12x import sparse_decode_atten_func

    torch.manual_seed(4)
    dev = torch.device("cuda")
    num_kv_heads, h_ratio, head_dim, page_size = 1, 2, 128, 128
    num_qo_heads = num_kv_heads * h_ratio
    seqlen_q, used = 1, 200
    n_pages = (used + page_size - 1) // page_size
    q = torch.randn(seqlen_q, num_qo_heads, head_dim, device=dev, dtype=torch.bfloat16) * 0.3
    k = torch.randn(n_pages, num_kv_heads, page_size, head_dim, device=dev, dtype=torch.bfloat16) * 0.3
    v = torch.randn_like(k)
    page_table = torch.arange(n_pages, device=dev, dtype=torch.int32).reshape(1, n_pages)
    seqused = torch.tensor([used], dtype=torch.int32, device=dev)
    sm_scale = 1.0 / math.sqrt(head_dim)

    out = sparse_decode_atten_func(
        q, k, v, page_table=page_table, seqused_k=seqused, seqlen_q=seqlen_q,
        max_seqlen_k=n_pages * page_size, blk_kv=page_size, causal=True, softmax_scale=sm_scale,
    )

    # Flatten paged KV to [used, Hkv, D] and run dense causal SDPA for the single
    # decode token (it sees all `used` tokens).
    k_flat = k.permute(0, 2, 1, 3).reshape(n_pages * page_size, num_kv_heads, head_dim)[:used]
    v_flat = v.permute(0, 2, 1, 3).reshape(n_pages * page_size, num_kv_heads, head_dim)[:used]
    ref = torch.zeros((seqlen_q, num_qo_heads, head_dim), dtype=torch.float32, device=dev)
    for h in range(num_qo_heads):
        kvh = h // h_ratio
        logits = (k_flat[:, kvh].float() @ q[0, h].float()) * sm_scale
        ref[0, h] = torch.softmax(logits, dim=0) @ v_flat[:, kvh].float()
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
