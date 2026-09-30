"""FP8-world reference for Q8KV4 page-max scores."""

from __future__ import annotations

import torch

_E2M1_VALUES = (
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


def dequantize_k(packed_k: torch.Tensor, k_scale: torch.Tensor) -> torch.Tensor:
    """Match QMUL4 E2M1-times-E4M3 with E4M3 output rounding."""

    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed_k.device)
    codes = torch.stack((packed_k & 0x0F, packed_k >> 4), dim=-1)
    codes = codes.reshape(packed_k.shape[0], 128, 128).to(torch.int64)
    scale = k_scale.float().repeat_interleave(16, dim=-1)
    limit = torch.finfo(torch.float8_e4m3fn).max
    return (lut[codes] * scale).clamp(-limit, limit).to(torch.float8_e4m3fn).float()


def indexer_gemm_reference(
    q: torch.Tensor,
    packed_k: torch.Tensor,
    k_scale: torch.Tensor,
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
            flat_ids = physical_ids.reshape(-1)
            logical_k = dequantize_k(
                packed_k.index_select(0, flat_ids),
                k_scale.index_select(0, flat_ids),
            ).reshape(batch_end - batch_start, page_end - page_start, 128, 128)
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
) -> tuple[torch.Tensor, ...]:
    """Create deterministic quantized inputs with non-identity page tables."""

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
    ).to(torch.float8_e4m3fn)
    packed_k = torch.randint(
        0,
        256,
        (physical_pages, 128, 64),
        generator=generator,
        dtype=torch.uint8,
        device=device,
    )
    k_scale = (
        torch.rand(physical_pages, 128, 8, generator=generator, device=device) * 0.5
        + 0.125
    ).to(torch.float8_e4m3fn)
    page_table = torch.stack(
        [
            torch.randperm(physical_pages, generator=generator, device=device)[
                :max_pages
            ]
            for _ in range(batch)
        ]
    ).to(torch.int32)
    return q, packed_k, k_scale, page_table, seq_lens.to(device=device)
