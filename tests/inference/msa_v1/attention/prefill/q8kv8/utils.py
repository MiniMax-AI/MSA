"""Shared Q8KV8 PageKV test data builders."""

from __future__ import annotations

import torch


PAGE_SIZE = 128
TOPK = 16


def make_cu_seqlens(lengths: tuple[int, ...], device: torch.device) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32, device=device)


def make_topk(
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Build deterministic unordered history followed by the causal page."""

    total_q = sum(q_lens)
    topk_host = torch.full((4, total_q, TOPK), -1, dtype=torch.int32)
    q_offset = 0
    for q_len, k_len in zip(q_lens, k_lens):
        for q_idx in range(q_len):
            current_page = (k_len - q_len + q_idx) // PAGE_SIZE
            history = list(range(current_page))
            for kv_head in range(4):
                if history:
                    shift = (q_idx * 5 + kv_head * 3) % len(history)
                    shuffled = history[shift:] + history[:shift]
                    if (q_idx + kv_head) & 1:
                        shuffled.reverse()
                else:
                    shuffled = []
                selected = shuffled[: TOPK - 1]
                pages = selected + [current_page]
                topk_host[kv_head, q_offset + q_idx, : len(pages)] = torch.tensor(
                    pages,
                    dtype=torch.int32,
                )
        q_offset += q_len
    return topk_host.to(device=device)


def make_shuffled_page_table(
    batch: int,
    max_pages: int,
    *,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(batch * max_pages, generator=generator)
    return permutation.reshape(batch, max_pages).to(device=device, dtype=torch.int32)


__all__ = ["make_cu_seqlens", "make_shuffled_page_table", "make_topk"]
