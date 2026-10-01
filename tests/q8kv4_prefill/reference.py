# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Independent fp32 reference for Q8KV4 sparse prefill.

Dequantization follows the cache's data contract: each value is ``E2M1(code) x E4M3(block_scale)
x global_scale``, with ``code x block_scale`` requantized to E4M3 (the FP8 tensor core's input)
after staging the block scale by ``2**-block_scale_shift``. Each selected page is one split: its
softmax uses the page's own maximum, the probabilities are quantized as ``E4M3(p * 448) / 448``
for the PV product while the normalizer sums them unquantized, and the split output is rounded
to bf16 before the splits are merged by their log-sum-exp.
"""

from __future__ import annotations

import torch

from .cases import HEAD_DIM, PAGE_SIZE, Q_HEADS_PER_KV, SM_SCALE, PrefillInputs, page_counts, query_positions

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
E4M3_MAX = 448.0
FP8_PROBABILITY_SCALE = 448.0


def dequantize(
    codes: torch.Tensor,
    scale: torch.Tensor,
    *,
    global_scale: float = 1.0,
    block_scale_shift: int = 0,
    fp8_datapath: bool = True,
) -> torch.Tensor:
    """``[..., 128, 64]`` packed codes and ``[..., 128, 8]`` linear scales to fp32 values; without
    ``fp8_datapath`` the exact ``code x scale x global_scale``."""
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=codes.device)
    values = torch.stack((codes & 0x0F, codes >> 4), dim=-1).reshape(*codes.shape[:-1], HEAD_DIM)
    if not fp8_datapath:
        return lut[values.long()] * scale.float().repeat_interleave(16, dim=-1) * global_scale
    staged_scale = (scale.float() / (1 << block_scale_shift)).clamp(-E4M3_MAX, E4M3_MAX)
    staged_scale = staged_scale.to(torch.float8_e4m3fn).float()
    products = lut[values.long()] * staged_scale.repeat_interleave(16, dim=-1)
    requantized = products.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    return requantized * float(1 << block_scale_shift) * global_scale


def sparse_prefill_reference(
    inputs: PrefillInputs,
    *,
    sm_scale: float = SM_SCALE,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    k_global_scale: float = 1.0,
    v_global_scale: float = 1.0,
    block_scale_shift: int = 0,
    topk_indices: torch.Tensor | None = None,
    seqused_k: list[int] | None = None,
    output_scale: float = 1.0,
    fp8_datapath: bool = True,
    query_chunk: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(out [total_q, Hq, 128] bf16, lse [total_q, Hq] fp32, natural log)``.

    ``seqused_k`` overrides each request's KV length for masking and for the causal alignment
    (query ``i`` of a ``q_len`` chunk sits at ``seqused_k - q_len + i``), as ``fmha_sm100``'s
    ``qo_offset`` does. A selected page outside the request's pages or past the query is empty.
    ``fp8_datapath=False`` drops the kernel's roundings (E4M3 requant, probability quantization,
    bf16 splits): the attention the cache's values define.
    """
    case = inputs.case
    device = inputs.q.device
    shift = block_scale_shift
    k_values = dequantize(inputs.k_codes, inputs.k_scale if k_scale is None else k_scale,
                          global_scale=k_global_scale, block_scale_shift=shift,
                          fp8_datapath=fp8_datapath)
    v_values = dequantize(inputs.v_codes, inputs.v_scale if v_scale is None else v_scale,
                          global_scale=v_global_scale, block_scale_shift=shift,
                          fp8_datapath=fp8_datapath)
    lists = (inputs.topk_indices if topk_indices is None else topk_indices).long()
    heads = inputs.k_codes.shape[1]
    q = inputs.q.float()
    out = torch.zeros((case.total_q, heads * Q_HEADS_PER_KV, HEAD_DIM), dtype=torch.float32,
                      device=device)
    lse = torch.full((case.total_q, heads * Q_HEADS_PER_KV), -torch.inf, dtype=torch.float32,
                     device=device)
    token_in_page = torch.arange(PAGE_SIZE, device=device)
    counts = page_counts(case.k_lens)
    kv_lens = list(case.k_lens) if seqused_k is None else list(seqused_k)

    row_begin = 0
    for batch, q_len in enumerate(case.q_lens):
        page_count = counts[batch]
        table = inputs.page_table[batch, :page_count].long()
        for chunk_begin in range(0, q_len, query_chunk):
            chunk_end = min(chunk_begin + query_chunk, q_len)
            rows = slice(row_begin + chunk_begin, row_begin + chunk_end)
            positions = torch.arange(kv_lens[batch] - q_len + chunk_begin,
                                     kv_lens[batch] - q_len + chunk_end, device=device)
            for head in range(heads):
                pages = lists[head, rows]  # [n, topk]
                present = (pages >= 0) & (pages < page_count)
                physical = table[pages.clamp(0, page_count - 1)]
                keys = k_values[physical, head]  # [n, topk, 128, 128]
                values = v_values[physical, head]
                key_positions = pages[..., None] * PAGE_SIZE + token_in_page
                visible = (present[..., None] & (key_positions < kv_lens[batch])
                           & (key_positions <= positions[:, None, None]))  # [n, topk, 128]
                q_heads = slice(head * Q_HEADS_PER_KV, (head + 1) * Q_HEADS_PER_KV)
                scores = torch.einsum("nhd,ntkd->nhtk", q[rows, q_heads], keys) * sm_scale
                scores.masked_fill_(~visible[:, None], -torch.inf)
                page_max = scores.amax(dim=-1)  # [n, h, topk]
                finite = torch.isfinite(page_max)
                probability = torch.exp(scores - torch.where(finite, page_max, 0.0)[..., None])
                probability.masked_fill_(~visible[:, None], 0.0)
                page_sum = probability.sum(dim=-1)
                if fp8_datapath:
                    quantized = (probability * FP8_PROBABILITY_SCALE).to(torch.float8_e4m3fn)
                    probability = quantized.float() / FP8_PROBABILITY_SCALE
                partial = torch.einsum("nhtk,ntkd->nhtd", probability, values)
                partial = partial / page_sum.clamp_min(1e-30)[..., None] * output_scale
                if fp8_datapath:
                    partial = partial.to(torch.bfloat16).float()
                page_lse = torch.where(finite, page_max + torch.log(page_sum), -torch.inf)
                row_lse = torch.logsumexp(page_lse, dim=-1)  # [n, h]
                weights = torch.exp(page_lse - row_lse[..., None]).nan_to_num(0.0)
                out[rows, q_heads] = torch.einsum("nht,nhtd->nhd", weights, partial)
                lse[rows, q_heads] = row_lse
        row_begin += q_len
    return out.to(torch.bfloat16), lse


def assert_close_to_reference(
    out: torch.Tensor,
    reference: torch.Tensor,
    *,
    label: str,
    lse: torch.Tensor | None = None,
    reference_lse: torch.Tensor | None = None,
) -> None:
    """Finite output within ``0.05 + 0.05 |reference|`` (the decode tests' bound) everywhere and
    within ``2e-3 + 2e-2 |reference|`` (nv_dev's) on all but ``max(4, 1e-4 * vectors)`` of the
    ``(query, head)`` output vectors; LSE within ``1e-4`` absolute where the reference is finite.

    The rare outlier vectors come from a probability on an E4M3 rounding boundary (3 mantissa
    bits) that the kernel's ``ex2.approx`` and the reference's fp32 ``exp`` round to neighbouring
    codes; one such key shifts its head's whole output vector, most visibly for queries that see
    few keys.
    """
    assert torch.isfinite(out).all(), f"{label}: non-finite output"
    diff = (out.float() - reference.float()).abs()
    magnitude = reference.float().abs()
    loose = int((diff > 0.05 + 0.05 * magnitude).sum())
    tight_vectors = int((diff > 2e-3 + 2e-2 * magnitude).any(dim=-1).sum())
    vectors = diff[..., 0].numel()
    assert loose == 0 and tight_vectors <= max(4, vectors * 1e-4), (
        f"{label}: {tight_vectors} of {vectors} (query, head) vectors outside the tight "
        f"tolerance, {loose} elements outside the loose one, max |diff| {diff.max().item():.4g}"
    )
    if lse is not None:
        finite = torch.isfinite(reference_lse)
        assert torch.equal(torch.isfinite(lse), finite), f"{label}: LSE finiteness differs"
        lse_diff = (lse[finite] - reference_lse[finite]).abs()
        assert lse_diff.numel() == 0 or lse_diff.max().item() <= 1e-4, (
            f"{label}: max LSE |diff| {lse_diff.max().item():.3g}"
        )


__all__ = ["assert_close_to_reference", "dequantize", "query_positions", "sparse_prefill_reference"]
