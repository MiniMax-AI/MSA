# SPDX-License-Identifier: MIT

"""SM12x paged decode reference implementation."""

from __future__ import annotations

import torch

from ._lse import run_lse
from .api import fmha_sm12x, fmha_sm12x_plan


def _compact_page_table(page_table: torch.Tensor, seqused_k: torch.Tensor, page_size: int) -> torch.Tensor:
    pages: list[torch.Tensor] = []
    lengths = seqused_k.to("cpu", dtype=torch.int64, non_blocking=False).tolist()
    for batch, kv_len in enumerate(lengths):
        page_count = (int(kv_len) + int(page_size) - 1) // int(page_size)
        if page_count > 0:
            pages.append(page_table[batch, :page_count])
    if not pages:
        return torch.empty((0,), dtype=torch.int32, device=page_table.device)
    return torch.cat(pages).contiguous()


def sparse_decode_atten_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor | None = None,
    *,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    seqlen_q: int,
    max_seqlen_k: int,
    blk_kv: int = 128,
    causal: bool = True,
    softmax_scale: float | None = None,
    return_softmax_lse: bool = False,
    **_kwargs,
):
    """Run SM12x paged decode via the dequantized Torch reference path."""

    if q.ndim != 3 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("decode expects q [B*S,Hq,D] and paged k/v [P,Hkv,page,D]")
    # FP8 E4M3 K/V cache (and FP8 Q) is staged to BF16, matching the SM100
    # decode path; the reference then runs in BF16.
    q = q.to(torch.bfloat16) if q.dtype == torch.float8_e4m3fn else q
    k = k.to(torch.bfloat16) if k.dtype == torch.float8_e4m3fn else k
    v = v.to(torch.bfloat16) if v.dtype == torch.float8_e4m3fn else v
    if page_table.dtype != torch.int32 or seqused_k.dtype != torch.int32:
        raise TypeError("page_table and seqused_k must be int32")
    if page_table.device != q.device or seqused_k.device != q.device:
        raise ValueError("decode metadata must be on q.device")
    batch = int(page_table.shape[0])
    if int(q.shape[0]) != batch * int(seqlen_q):
        raise ValueError("q.shape[0] must equal batch * seqlen_q")
    if int(k.shape[2]) != int(blk_kv) or k.shape != v.shape:
        raise ValueError("paged k/v shapes must match and use blk_kv page size")
    if max_seqlen_k <= 0:
        raise ValueError("max_seqlen_k must be positive")
    qo_lens = torch.full((batch,), int(seqlen_q), dtype=torch.int32, device=q.device)
    kv_lens = seqused_k.to(dtype=torch.int32)
    kv_indices = _compact_page_table(page_table.contiguous(), seqused_k.contiguous(), int(blk_kv))
    plan = fmha_sm12x_plan(
        qo_lens, kv_lens, int(q.shape[1]), int(k.shape[1]), page_size=int(blk_kv),
        causal=bool(causal),
    )
    block_indexes = None
    if q2k_indices is not None:
        if q2k_indices.dtype != torch.int32 or q2k_indices.ndim != 3:
            raise ValueError("q2k_indices must be int32 with shape [Hkv, total_q, topK]")
        block_indexes = q2k_indices.permute(1, 0, 2).contiguous()
    out, _ = fmha_sm12x(
        q, k, v, plan, kv_indices=kv_indices, kv_block_indexes=block_indexes,
        sm_scale=softmax_scale,
    )
    if return_softmax_lse:
        lse = run_lse(
            q, k, v, plan, kv_indices=kv_indices, kv_block_indexes=block_indexes,
            sm_scale=softmax_scale,
        )
        return out, lse
    return out


class SparseDecodePagedAttentionWrapper:
    """Plan/run wrapper matching the SM100 decode surface for SM12x."""

    def __init__(self, *, blk_kv: int = 128, causal: bool = True) -> None:
        self.blk_kv = int(blk_kv)
        self.causal = bool(causal)
        self.page_table: torch.Tensor | None = None
        self.seqused_k: torch.Tensor | None = None
        self.q2k_indices: torch.Tensor | None = None
        self.seqlen_q: int | None = None
        self.max_seqlen_k: int | None = None

    def plan(
        self,
        *,
        page_table: torch.Tensor,
        seqused_k: torch.Tensor,
        seqlen_q: int,
        max_seqlen_k: int,
        q2k_indices: torch.Tensor | None = None,
        **_kwargs,
    ) -> "SparseDecodePagedAttentionWrapper":
        self.page_table = page_table.contiguous()
        self.seqused_k = seqused_k.contiguous()
        self.q2k_indices = None if q2k_indices is None else q2k_indices.contiguous()
        self.seqlen_q = int(seqlen_q)
        self.max_seqlen_k = int(max_seqlen_k)
        return self

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        softmax_scale: float | None = None,
        return_softmax_lse: bool = False,
        **_kwargs,
    ):
        if self.page_table is None or self.seqused_k is None or self.seqlen_q is None or self.max_seqlen_k is None:
            raise RuntimeError("decode wrapper must be planned before run")
        return sparse_decode_atten_func(
            q, k, v, self.q2k_indices, page_table=self.page_table, seqused_k=self.seqused_k,
            seqlen_q=self.seqlen_q, max_seqlen_k=self.max_seqlen_k, blk_kv=self.blk_kv,
            causal=self.causal, softmax_scale=softmax_scale, return_softmax_lse=return_softmax_lse,
        )
