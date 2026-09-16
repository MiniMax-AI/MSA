"""Operator-independent tensor metadata builders for inference workloads."""

from __future__ import annotations

from collections.abc import Sequence


def cumulative_lengths(lengths: Sequence[int]) -> tuple[int, ...]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return tuple(offsets)


def make_disjoint_page_table(
    final_kv_lens: Sequence[int],
    *,
    max_cols: int,
    generator,
    device,
    page_size: int = 128,
):
    """Build disjoint active mappings and one valid shared suffix page."""

    import torch

    active_pages = tuple(
        (length + page_size - 1) // page_size for length in final_kv_lens
    )
    physical_pages = sum(active_pages) + 1
    physical_ids = torch.randperm(
        physical_pages,
        generator=generator,
        device=device,
        dtype=torch.int64,
    )
    scratch_page = physical_ids[-1]
    page_table = scratch_page.expand(len(active_pages), max_cols).clone()
    offset = 0
    for batch_idx, page_count in enumerate(active_pages):
        page_table[batch_idx, :page_count] = physical_ids[offset : offset + page_count]
        offset += page_count
    return page_table.to(torch.int32).contiguous(), physical_pages


def make_attention_topk(
    case,
    *,
    device,
    kv_heads: int = 4,
    topk: int = 16,
    page_size: int = 128,
):
    """Build unordered history followed by the forced causal page."""

    import torch

    batch_ids = []
    query_ids = []
    query_positions = []
    for batch, (query_len, prefix_len) in enumerate(
        zip(case.query_lens, case.prefix_lens, strict=True)
    ):
        query_idx = torch.arange(query_len, dtype=torch.int64, device=device)
        batch_ids.append(torch.full_like(query_idx, batch))
        query_ids.append(query_idx)
        query_positions.append(query_idx + prefix_len)
    batch_ids = torch.cat(batch_ids)
    query_ids = torch.cat(query_ids)
    query_positions = torch.cat(query_positions)
    local_page = torch.div(query_positions, page_size, rounding_mode="floor")
    history_count = local_page.clamp(max=topk - 1)
    safe_history_count = history_count.clamp(min=1)
    head = torch.arange(kv_heads, dtype=torch.int64, device=device).reshape(
        kv_heads, 1, 1
    )
    slot = torch.arange(topk - 1, dtype=torch.int64, device=device).reshape(
        1, 1, topk - 1
    )
    count = history_count.reshape(1, -1, 1)
    safe_count = safe_history_count.reshape(1, -1, 1)
    salt = case.seed % 104729
    position = torch.remainder(
        slot + head + batch_ids.reshape(1, -1, 1) + salt,
        safe_count,
    )
    reverse = torch.bitwise_and(
        head
        + query_ids.reshape(1, -1, 1)
        + batch_ids.reshape(1, -1, 1)
        + salt,
        1,
    ).bool()
    position = torch.where(reverse, safe_count - 1 - position, position)
    history_page = torch.div(
        position * local_page.reshape(1, -1, 1),
        safe_count,
        rounding_mode="floor",
    )
    result = torch.full(
        (kv_heads, case.total_q, topk),
        -1,
        dtype=torch.int32,
        device=device,
    )
    result[:, :, : topk - 1] = torch.where(
        slot < count,
        history_page,
        -1,
    ).to(torch.int32)
    result.scatter_(
        2,
        history_count.reshape(1, -1, 1).expand(kv_heads, -1, -1),
        local_page.reshape(1, -1, 1).expand(kv_heads, -1, -1).to(torch.int32),
    )
    return result.contiguous()


__all__ = [
    "cumulative_lengths",
    "make_attention_topk",
    "make_disjoint_page_table",
]
