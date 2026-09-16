"""Independent FP32 reference for Q8K8 paged sparse decode attention."""

from __future__ import annotations

import torch

from tests.inference.msa_v1.attention.decode.q8kv8.real_cases import (
    DecodeAttentionInputs,
)
from tests.inference.msa_v1.attention.decode.reference import sparse_decode_reference
from tests.inference.msa_v1.decode.metadata import PAGE_SIZE, TOPK

HEAD_DIM = 128


def decode_attention_reference(
    inputs: DecodeAttentionInputs,
    *,
    softmax_scale: float = HEAD_DIM**-0.5,
) -> torch.Tensor:
    """Compute all rows with an independent FP32 paged reference."""
    return sparse_decode_reference(
        inputs, inputs.k_cache, inputs.v_cache, softmax_scale=softmax_scale
    )


def assert_decode_topk_contract(inputs: DecodeAttentionInputs) -> None:
    """Check valid prefixes, local-page tails, and sparse history bounds."""

    batch = inputs.seq_lens.numel()
    query_ids = torch.arange(
        inputs.q_len_per_req, dtype=torch.int64, device=inputs.seq_lens.device
    ).repeat(batch)
    positions = inputs.seq_lens.to(torch.int64).repeat_interleave(inputs.q_len_per_req)
    positions = positions - inputs.q_len_per_req + query_ids
    local_page = torch.div(positions, PAGE_SIZE, rounding_mode="floor")
    valid_count = (local_page + 1).clamp(max=TOPK)
    slots = torch.arange(TOPK, device=inputs.seq_lens.device).reshape(1, 1, -1)
    valid = slots < valid_count.reshape(-1, 1, 1)
    assert torch.equal(
        inputs.topk_indices >= 0, valid.expand(-1, inputs.topk_indices.shape[1], -1)
    )
    tail = inputs.topk_indices.gather(
        2,
        (valid_count - 1)
        .reshape(-1, 1, 1)
        .expand(-1, inputs.topk_indices.shape[1], -1),
    ).squeeze(-1)
    assert torch.equal(
        tail,
        local_page.to(torch.int32)
        .reshape(-1, 1)
        .expand(-1, inputs.topk_indices.shape[1]),
    )


__all__ = ["assert_decode_topk_contract", "decode_attention_reference"]
