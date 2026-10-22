"""One launch that copies a verify step's inputs into the binding's persistent tensors.

The step's KV length, page table, positions and slots used to take seven eager
ops (add, gather, floor-divide and four copies); each costs a launch gap on the
host-bound bs=1 step.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _prepare_kernel(
    seq_lens,
    req_pool_indices,
    req_to_token,
    req_to_token_stride,
    req_to_token_width,
    positions_in,
    slots_in,
    kv_len,
    kv_indices,
    positions_out,
    slots_out,
    pages,
    PAGE: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    page = block * BLOCK + tl.arange(0, BLOCK)
    in_table = page < pages
    row = tl.load(req_pool_indices).to(tl.int64)
    column = tl.minimum(page.to(tl.int64) * PAGE, req_to_token_width - 1)
    slot = tl.load(req_to_token + row * req_to_token_stride + column, in_table, 0).to(tl.int64)
    # floor division, as torch.div(rounding_mode="floor") for a negative slot
    index = slot // PAGE
    index = tl.where((slot < 0) & (index * PAGE != slot), index - 1, index)
    tl.store(kv_indices + page, index.to(kv_indices.dtype.element_ty), in_table)
    if block == 0:
        # verify seq_lens exclude the rows being verified
        tl.store(kv_len, (tl.load(seq_lens) + ROWS).to(kv_len.dtype.element_ty))
        r = tl.arange(0, ROWS)
        tl.store(positions_out + r, tl.load(positions_in + r).to(tl.int64))
        tl.store(slots_out + r, tl.load(slots_in + r).to(tl.int64))


def prepare_step_inputs(
    *,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    kv_len: torch.Tensor,
    kv_indices: torch.Tensor,
    positions_out: torch.Tensor,
    slot_mapping: torch.Tensor,
    page: int,
    rows: int,
) -> None:
    """``kv_len = seq_lens[0] + rows``; ``kv_indices[0, p]`` is the page of the
    request's token ``min(p * page, width - 1)``; positions and slots copied."""
    pages = kv_indices.shape[-1]
    assert kv_indices.is_contiguous() and req_to_token.stride(1) == 1
    assert positions.numel() == out_cache_loc.numel() == rows
    assert positions.is_contiguous() and out_cache_loc.is_contiguous()
    _prepare_kernel[(triton.cdiv(pages, _BLOCK),)](
        seq_lens,
        req_pool_indices,
        req_to_token,
        req_to_token.stride(0),
        req_to_token.shape[1],
        positions,
        out_cache_loc,
        kv_len,
        kv_indices,
        positions_out,
        slot_mapping,
        pages,
        PAGE=page,
        ROWS=rows,
        BLOCK=_BLOCK,
        num_warps=4,
    )
