"""Independent, bounded-memory FP32 sparse attention reference."""

from __future__ import annotations

from collections.abc import Callable

import torch

_PAGE_SIZE = 128
_P_SCALE = 448.0
_REFERENCE_ROWS = 128


@torch.no_grad()
def sparse_decode_reference(
    inputs,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    softmax_scale: float,
    dequantize: Callable | None = None,
) -> torch.Tensor:
    """Batch independent query/head rows without sampling the output."""
    total_q, q_heads, head_dim = inputs.q.shape
    kv_heads = k_cache.shape[1]
    group_size = q_heads // kv_heads
    rows = total_q * kv_heads
    q = inputs.q.reshape(rows, group_size, head_dim)
    topk = inputs.topk_indices.reshape(rows, -1)
    output = torch.empty_like(inputs.q, dtype=torch.bfloat16).reshape(
        rows, group_size, head_dim
    )
    token = torch.arange(_PAGE_SIZE, device=q.device)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for begin in range(0, rows, _REFERENCE_ROWS):
            end = min(begin + _REFERENCE_ROWS, rows)
            flat_ids = torch.arange(begin, end, device=q.device)
            query_ids = flat_ids // kv_heads
            batch_ids = query_ids // inputs.q_len_per_req
            head_ids = flat_ids % kv_heads
            positions = (
                inputs.seq_lens[batch_ids].to(torch.int64)
                - inputs.q_len_per_req
                + query_ids % inputs.q_len_per_req
            )
            logical = topk[begin:end].to(torch.int64)
            physical = inputs.page_table[batch_ids[:, None], logical.clamp_min(0)].to(
                torch.int64
            )
            k = k_cache[physical, head_ids[:, None]]
            v = v_cache[physical, head_ids[:, None]]
            if dequantize is not None:
                k = dequantize(k, inputs.k_scale[physical, head_ids[:, None]])
                v = dequantize(v, inputs.v_scale[physical, head_ids[:, None]])
            else:
                k, v = k.float(), v.float()
            scores = (
                torch.einsum("rgd,rptd->rpgt", q[begin:end].float(), k) * softmax_scale
            )
            valid = (logical >= 0)[:, :, None] & (
                logical[:, :, None] * _PAGE_SIZE + token <= positions[:, None, None]
            )
            scores.masked_fill_(~valid[:, :, None, :], -torch.inf)
            maxima = torch.full(
                (end - begin, group_size),
                -torch.inf,
                dtype=torch.float32,
                device=q.device,
            )
            denominator = torch.zeros_like(maxima)
            accumulator = torch.zeros(
                (end - begin, group_size, head_dim),
                dtype=torch.float32,
                device=q.device,
            )
            for page in range(logical.shape[1]):
                page_scores = scores[:, page]
                new_maxima = torch.maximum(maxima, page_scores.amax(-1))
                safe_maxima = torch.where(torch.isfinite(new_maxima), new_maxima, 0.0)
                correction = torch.exp(maxima - safe_maxima)
                probability = (
                    torch.exp(page_scores - safe_maxima[:, :, None]) * _P_SCALE
                )
                probability_fp8 = probability.to(torch.float8_e4m3fn).float()
                accumulator = accumulator * correction[:, :, None] + torch.bmm(
                    probability_fp8, v[:, page]
                )
                denominator = denominator * correction + probability.sum(-1)
                maxima = new_maxima
            output[begin:end] = (
                accumulator / denominator.clamp_min(1.0)[:, :, None]
            ).to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return output.reshape(total_q, q_heads, head_dim)
