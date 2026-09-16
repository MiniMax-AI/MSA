"""Shared paged sparse metadata generation for MSA v1 decode tests."""

from __future__ import annotations

import torch

PAGE_SIZE = 128
KV_HEADS = 4
TOPK = 16


def make_decode_topk(
    seq_lens: torch.Tensor,
    *,
    q_len_per_req: int,
    seed: int,
    num_kv_heads: int = KV_HEADS,
) -> torch.Tensor:
    """Build unique unordered histories with the local page forced last."""

    device = seq_lens.device
    batch = seq_lens.numel()
    batch_ids = torch.arange(batch, dtype=torch.int64, device=device).repeat_interleave(
        q_len_per_req
    )
    query_ids = torch.arange(
        q_len_per_req,
        dtype=torch.int64,
        device=device,
    ).repeat(batch)
    query_positions = seq_lens.to(torch.int64).repeat_interleave(q_len_per_req)
    query_positions = query_positions - q_len_per_req + query_ids
    local_page = torch.div(query_positions, PAGE_SIZE, rounding_mode="floor")
    history_count = local_page.clamp(max=TOPK - 1)
    safe_history_count = history_count.clamp(min=1)

    head = torch.arange(num_kv_heads, dtype=torch.int64, device=device).reshape(
        1, num_kv_heads, 1
    )
    slot = torch.arange(TOPK - 1, dtype=torch.int64, device=device).reshape(
        1, 1, TOPK - 1
    )
    count_3d = history_count.reshape(-1, 1, 1)
    safe_count_3d = safe_history_count.reshape(-1, 1, 1)
    salt = seed % 104729
    position = torch.remainder(
        slot + head + batch_ids.reshape(-1, 1, 1) + salt,
        safe_count_3d,
    )
    reverse = torch.bitwise_and(
        head + query_ids.reshape(-1, 1, 1) + batch_ids.reshape(-1, 1, 1) + salt,
        1,
    ).bool()
    position = torch.where(reverse, safe_count_3d - 1 - position, position)
    history_page = torch.div(
        position * local_page.reshape(-1, 1, 1),
        safe_count_3d,
        rounding_mode="floor",
    )
    topk = torch.full(
        (batch * q_len_per_req, num_kv_heads, TOPK),
        -1,
        dtype=torch.int32,
        device=device,
    )
    topk[:, :, : TOPK - 1] = torch.where(
        slot < count_3d,
        history_page,
        -1,
    ).to(torch.int32)
    topk.scatter_(
        2,
        history_count.reshape(-1, 1, 1).expand(-1, num_kv_heads, -1),
        local_page.reshape(-1, 1, 1).expand(-1, num_kv_heads, -1).to(torch.int32),
    )
    return topk.contiguous()


__all__ = ["KV_HEADS", "PAGE_SIZE", "TOPK", "make_decode_topk"]
