"""Independent full-output reference for Q8KV4 paged sparse decode attention."""

from __future__ import annotations

import torch

from tests.inference.msa_v1.attention.decode.q8kv4.real_cases import (
    DecodeAttentionInputs,
)
from tests.inference.msa_v1.attention.decode.reference import sparse_decode_reference

E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
HEAD_DIM = 128
PAGE_SIZE = 128
TOPK = 16


def dequantize_cache(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Match QMUL4 E2M1-times-E4M3 with E4M3 output rounding."""

    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1)
    codes = codes.reshape(*packed.shape[:-1], HEAD_DIM).to(torch.int64)
    expanded_scale = scale.float().repeat_interleave(16, dim=-1)
    return (
        (lut[codes] * expanded_scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .float()
    )


def decode_attention_reference(
    inputs: DecodeAttentionInputs,
    *,
    softmax_scale: float = HEAD_DIM**-0.5,
) -> torch.Tensor:
    """Compute all rows with an independent FP32 paged reference."""
    return sparse_decode_reference(
        inputs,
        inputs.packed_k,
        inputs.packed_v,
        softmax_scale=softmax_scale,
        dequantize=dequantize_cache,
    )


def assert_decode_topk_contract(inputs: DecodeAttentionInputs) -> None:
    """Check valid prefixes, arbitrary unique history, and forced local tails."""

    batch = inputs.seq_lens.numel()
    query_ids = torch.arange(
        inputs.q_len_per_req,
        dtype=torch.int64,
        device=inputs.seq_lens.device,
    ).repeat(batch)
    positions = inputs.seq_lens.to(torch.int64).repeat_interleave(inputs.q_len_per_req)
    positions = positions - inputs.q_len_per_req + query_ids
    local_page = torch.div(positions, PAGE_SIZE, rounding_mode="floor")
    valid_count = (local_page + 1).clamp(max=TOPK)
    slots = torch.arange(TOPK, device=inputs.seq_lens.device).reshape(1, 1, -1)
    expected_valid = slots < valid_count.reshape(-1, 1, 1)
    assert torch.equal(
        inputs.topk_indices >= 0,
        expected_valid.expand(-1, inputs.topk_indices.shape[1], -1),
    )
    tail = inputs.topk_indices.gather(
        2,
        (valid_count - 1)
        .reshape(-1, 1, 1)
        .expand(-1, inputs.topk_indices.shape[1], -1),
    ).squeeze(-1)
    expected_tail = local_page.to(torch.int32).reshape(-1, 1)
    assert torch.equal(tail, expected_tail.expand(-1, inputs.topk_indices.shape[1]))

    history = inputs.topk_indices[:, :, : TOPK - 1]
    history_mask = (
        slots[:, :, : TOPK - 1] < (valid_count - 1).reshape(-1, 1, 1)
    ).expand(-1, inputs.topk_indices.shape[1], -1)
    limit = local_page.reshape(-1, 1, 1).expand(
        -1, inputs.topk_indices.shape[1], TOPK - 1
    )
    assert bool(torch.all(history[history_mask] >= 0))
    assert bool(torch.all(history[history_mask] < limit[history_mask]))
    sentinel = torch.iinfo(torch.int32).max
    ordered = torch.where(history_mask, history, sentinel).sort(dim=-1).values
    adjacent_valid = ordered[:, :, 1:] != sentinel
    assert bool(
        torch.all(
            ordered[:, :, 1:][adjacent_valid] > ordered[:, :, :-1][adjacent_valid]
        )
    )


__all__ = [
    "assert_decode_topk_contract",
    "decode_attention_reference",
    "dequantize_cache",
]
