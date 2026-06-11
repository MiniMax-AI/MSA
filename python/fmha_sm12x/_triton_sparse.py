# SPDX-License-Identifier: MIT

"""Triton block-sparse attention for SM12x dense-KV prefill/decode.

This is the semi-optimized SM120/SM121 path: a fused flash-attention kernel
that attends each query only to its top-k selected KV blocks, matching the
Torch reference in ``_reference.py`` / ``_lse.py`` numerically. Triton is an
optional accelerator (it ships with torch on Linux); callers fall back to the
Torch reference when :func:`triton_dense_supported` returns False.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only without triton
    _HAS_TRITON = False


def triton_available() -> bool:
    return _HAS_TRITON


def triton_dense_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_size: int,
) -> bool:
    """Whether the dense Triton fast path can serve this call.

    Dense KV only (rank-3 ``[total_k, Hkv, D]``); paged KV and exotic dtypes
    route to the Torch reference.
    """

    if not _HAS_TRITON:
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        return False
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    head_dim = int(q.shape[-1])
    if head_dim > 256 or int(v.shape[-1]) != head_dim:
        return False
    if int(page_size) <= 0 or int(page_size) > 256:
        return False
    return True


if _HAS_TRITON:

    @triton.jit
    def _sparse_attn_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        idx_ptr,
        o_ptr,
        lse_ptr,
        tlse_ptr,
        kv_start_ptr,
        kv_len_ptr,
        vis_ptr,
        sm_scale,
        inv_temp,
        h_ratio,
        topk,
        page_size,
        stride_qn,
        stride_qh,
        stride_qd,
        stride_kn,
        stride_kh,
        stride_kd,
        stride_vn,
        stride_vh,
        stride_vd,
        stride_in,
        stride_ih,
        stride_it,
        stride_on,
        stride_oh,
        stride_od,
        stride_ln,
        stride_lh,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_H: tl.constexpr,
        CAUSAL: tl.constexpr,
        RETURN_LSE: tl.constexpr,
        RETURN_TLSE: tl.constexpr,
    ):
        # One program owns a single query token and one KV head; it attends
        # the whole GQA group (h_ratio q-heads that share this KV head)
        # against that query's selected blocks. Per-token selection keeps the
        # result identical to the reference even when tokens in a block differ.
        q_index = tl.program_id(0)
        kv_head = tl.program_id(1)
        kv_start = tl.load(kv_start_ptr + q_index)
        kv_len = tl.load(kv_len_ptr + q_index)
        vis = tl.load(vis_ptr + q_index)

        off_h = tl.arange(0, BLOCK_H)
        off_d = tl.arange(0, BLOCK_D)
        off_k = tl.arange(0, BLOCK_K)
        h_mask = off_h < h_ratio
        d_mask = off_d < D
        qh = kv_head * h_ratio + off_h

        q = tl.load(
            q_ptr + q_index * stride_qn + qh[:, None] * stride_qh + off_d[None, :] * stride_qd,
            mask=h_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        m_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)
        m2 = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
        l2 = tl.zeros((BLOCK_H,), dtype=tl.float32)

        for t in range(topk):
            bid = tl.load(idx_ptr + q_index * stride_in + kv_head * stride_ih + t * stride_it)
            base = bid * page_size
            if (bid >= 0) and (base < kv_len):
                pos = base + off_k
                pos_mask = (off_k < page_size) & (pos < kv_len)
                if CAUSAL:
                    pos_mask = pos_mask & (pos <= vis)
                kv_row = kv_start + pos
                k = tl.load(
                    k_ptr + kv_row[None, :] * stride_kn + kv_head * stride_kh + off_d[:, None] * stride_kd,
                    mask=d_mask[:, None] & pos_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                qk = tl.dot(q, k, input_precision="ieee") * sm_scale
                qk = tl.where(pos_mask[None, :], qk, float("-inf"))

                m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
                # m_ij == -inf only when no position has ever been visible;
                # force alpha=1 there so exp(-inf - -inf) never yields NaN.
                alpha = tl.where(m_ij == float("-inf"), 1.0, tl.exp(m_i - m_ij))
                p = tl.where(pos_mask[None, :], tl.exp(qk - m_ij[:, None]), 0.0)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None]
                v = tl.load(
                    v_ptr + kv_row[:, None] * stride_vn + kv_head * stride_vh + off_d[None, :] * stride_vd,
                    mask=pos_mask[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc += tl.dot(p, v, input_precision="ieee")
                m_i = m_ij

                if RETURN_TLSE:
                    tqk = qk * inv_temp
                    m2_ij = tl.maximum(m2, tl.max(tqk, axis=1))
                    alpha2 = tl.where(m2_ij == float("-inf"), 1.0, tl.exp(m2 - m2_ij))
                    p2 = tl.where(pos_mask[None, :], tl.exp(tqk - m2_ij[:, None]), 0.0)
                    l2 = l2 * alpha2 + tl.sum(p2, axis=1)
                    m2 = m2_ij

        has_mass = l_i > 0
        out = tl.where(has_mass[:, None], acc / tl.where(has_mass[:, None], l_i[:, None], 1.0), 0.0)
        tl.store(
            o_ptr + q_index * stride_on + qh[:, None] * stride_oh + off_d[None, :] * stride_od,
            out.to(o_ptr.dtype.element_ty),
            mask=h_mask[:, None] & d_mask[None, :],
        )
        if RETURN_LSE:
            lse = tl.where(has_mass, m_i + tl.log(tl.where(has_mass, l_i, 1.0)), float("-inf"))
            tl.store(lse_ptr + q_index * stride_ln + qh * stride_lh, lse, mask=h_mask)
        if RETURN_TLSE:
            has_mass2 = l2 > 0
            tlse = tl.where(has_mass2, m2 + tl.log(tl.where(has_mass2, l2, 1.0)), float("-inf"))
            tl.store(tlse_ptr + q_index * stride_ln + qh * stride_lh, tlse, mask=h_mask)


def _per_query_geometry(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    total_q: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-query kv_start, kv_len and causal visibility limit.

    ``vis[q] = (kv_len - qo_len) + local_q`` reproduces the reference default
    ``qo_offset = kv_lens - qo_lens`` so causal masking matches token-for-token.
    """

    device = cu_seqlens_q.device
    cu_q = cu_seqlens_q.to(torch.int64)
    cu_k = cu_seqlens_k.to(torch.int64)
    qo_len = cu_q[1:] - cu_q[:-1]
    kv_len_b = cu_k[1:] - cu_k[:-1]
    offset_b = kv_len_b - qo_len
    batch_id = torch.repeat_interleave(torch.arange(qo_len.numel(), device=device), qo_len)
    if int(batch_id.numel()) != int(total_q):
        raise ValueError("cu_seqlens_q does not sum to total_q")
    local_q = torch.arange(total_q, device=device) - cu_q[batch_id]
    kv_start_q = cu_k[batch_id].to(torch.int32)
    kv_len_q = kv_len_b[batch_id].to(torch.int32)
    vis_q = (offset_b[batch_id] + local_q).to(torch.int32)
    return kv_start_q.contiguous(), kv_len_q.contiguous(), vis_q.contiguous()


def triton_sparse_atten_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indexes: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_kv_heads: int,
    page_size: int,
    causal: bool = False,
    sm_scale: float | None = None,
    return_lse: bool = False,
    lse_temperature_scale: float = 1.0,
    return_temperature_lse: bool = False,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Fused dense block-sparse attention.

    ``block_indexes`` is int32 ``[total_q, num_kv_heads, topk]`` with -1
    padding (q2k selections, shared across each GQA group). Returns
    ``(out, lse_or_None, temperature_lse_or_None)``; LSE tensors are
    ``[total_q, Hq]`` float32 and -inf where a query selects no visible block.
    ``out_dtype`` defaults to ``q.dtype``; pass it to round the float32
    accumulator straight to the final dtype (the SM12x surface uses bf16).
    """

    if not _HAS_TRITON:
        raise RuntimeError("triton_sparse_atten_dense requires Triton")
    if block_indexes.dtype != torch.int32 or block_indexes.ndim != 3:
        raise ValueError("block_indexes must be int32 [total_q, num_kv_heads, topk]")
    total_q, num_qo_heads, head_dim = (int(x) for x in q.shape)
    num_kv_heads = int(num_kv_heads)
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    if int(block_indexes.shape[0]) != total_q or int(block_indexes.shape[1]) != num_kv_heads:
        raise ValueError("block_indexes shape must be [total_q, num_kv_heads, topk]")
    h_ratio = num_qo_heads // num_kv_heads
    topk = int(block_indexes.shape[2])
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    lse_temperature_scale = float(lse_temperature_scale)
    return_temperature_lse = bool(return_temperature_lse) and bool(return_lse)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    block_indexes = block_indexes.contiguous()
    kv_start_q, kv_len_q, vis_q = _per_query_geometry(cu_seqlens_q, cu_seqlens_k, total_q)

    out = torch.empty(
        (total_q, num_qo_heads, v.shape[-1]),
        dtype=out_dtype if out_dtype is not None else q.dtype,
        device=q.device,
    )
    lse = (
        torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    tlse = (
        torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)
        if return_temperature_lse
        else None
    )
    lse_view = lse if lse is not None else out  # unused when RETURN_LSE is False
    tlse_view = tlse if tlse is not None else out
    stride_ln = lse_view.stride(0) if lse is not None else 0
    stride_lh = lse_view.stride(1) if lse is not None else 0

    # tl.dot needs all three tile dims (M GQA-group rows, N KV columns, and
    # the K contraction = head dim) >= 16, so pad up; extra rows/columns/lanes
    # are masked off in the kernel.
    block_h = max(16, triton.next_power_of_2(h_ratio))
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_k = max(16, triton.next_power_of_2(int(page_size)))
    grid = (total_q, num_kv_heads)
    _sparse_attn_kernel[grid](
        q,
        k,
        v,
        block_indexes,
        out,
        lse_view,
        tlse_view,
        kv_start_q,
        kv_len_q,
        vis_q,
        float(sm_scale),
        1.0 / lse_temperature_scale,
        h_ratio,
        topk,
        int(page_size),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        block_indexes.stride(0),
        block_indexes.stride(1),
        block_indexes.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        stride_ln,
        stride_lh,
        D=head_dim,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        BLOCK_H=block_h,
        CAUSAL=bool(causal),
        RETURN_LSE=bool(return_lse),
        RETURN_TLSE=bool(return_temperature_lse),
        num_warps=4,
    )
    return out, lse, tlse


__all__ = ["triton_available", "triton_dense_supported", "triton_sparse_atten_dense"]
