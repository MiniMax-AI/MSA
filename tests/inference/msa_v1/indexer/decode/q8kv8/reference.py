"""FP8-world reference for Q8KV8 page-max scores."""

from __future__ import annotations

import torch


def indexer_gemm_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Compute every page-max score without materializing the logical K cache."""

    del seq_lens
    batch_size, max_pages = page_table.shape
    output = torch.empty(
        (q.shape[2], batch_size * q.shape[1], max_pages),
        dtype=torch.float32,
        device=q.device,
    )
    batch_chunk = 4
    page_chunk = 64
    for batch_start in range(0, batch_size, batch_chunk):
        batch_end = min(batch_start + batch_chunk, batch_size)
        q_chunk = q[batch_start:batch_end].float()
        for page_start in range(0, max_pages, page_chunk):
            page_end = min(page_start + page_chunk, max_pages)
            physical_ids = page_table[batch_start:batch_end, page_start:page_end].to(
                torch.int64
            )
            logical_k = k_cache.index_select(0, physical_ids.reshape(-1)).float()
            logical_k = logical_k.reshape(
                batch_end - batch_start,
                page_end - page_start,
                128,
                128,
            )
            token_scores = torch.einsum("bqhd,bptd->hbqpt", q_chunk, logical_k)
            output[
                :,
                batch_start * q.shape[1] : batch_end * q.shape[1],
                page_start:page_end,
            ] = token_scores.amax(dim=-1).reshape(
                q.shape[2],
                (batch_end - batch_start) * q.shape[1],
                page_end - page_start,
            )
    return output


def make_inputs(
    batch: int,
    max_pages: int,
    seq_lens: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
    num_index_heads: int = 1,
    query_length: int = 8,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, ...]:
    """Create deterministic inputs with non-identity page tables."""

    generator = torch.Generator(device=device).manual_seed(seed)
    physical_pages = max_pages + 3
    q = (
        torch.randn(
            batch,
            query_length,
            num_index_heads,
            128,
            generator=generator,
            device=device,
        )
        * 0.5
    ).to(dtype)
    k_cache = (
        torch.randn(
            physical_pages,
            128,
            128,
            generator=generator,
            device=device,
        )
        * 0.5
    ).to(dtype)
    page_table = torch.stack(
        [
            torch.randperm(physical_pages, generator=generator, device=device)[
                :max_pages
            ]
            for _ in range(batch)
        ]
    ).to(torch.int32)
    return q, k_cache, page_table, seq_lens.to(device=device)
