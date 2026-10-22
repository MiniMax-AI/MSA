"""The one-launch step input copy against the eager ops it replaces."""

from __future__ import annotations

import pytest
import torch
from sglang_msa_mono.minimax_m3 import PAGE, ROWS
from sglang_msa_mono.minimax_m3.prepare import prepare_step_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _eager(seq_lens, req_pool_indices, req_to_token, positions, out_cache_loc, pages):
    columns = torch.arange(pages, dtype=torch.int64, device=req_to_token.device) * PAGE
    row = req_pool_indices[:1].long()
    slots = req_to_token[row][:, columns.clamp(max=req_to_token.shape[1] - 1)]
    return (
        (seq_lens[:1] + ROWS).to(torch.int32),
        torch.div(slots, PAGE, rounding_mode="floor").to(torch.int32),
        positions.long(),
        out_cache_loc.long(),
    )


# pages wider than the table clamp to its last column; more than one kernel block
@pytest.mark.parametrize("width,pages", [(4096, 16), (1 << 17, 1024), (3000, 2100)])
def test_matches_the_eager_copy(width, pages):
    g = torch.Generator(device="cuda").manual_seed(width)
    reqs = 5
    req_to_token = torch.randint(0, 1 << 30, (reqs, width), device="cuda", generator=g).int()
    req_to_token[3, :7] = -1  # unallocated slots floor to page -1
    seq_lens = torch.tensor([width - 9], device="cuda", dtype=torch.int64)
    req_pool_indices = torch.tensor([3], device="cuda", dtype=torch.int64)
    positions = torch.arange(100, 100 + ROWS, device="cuda", dtype=torch.int64)
    out_cache_loc = torch.randint(0, 1 << 30, (ROWS,), device="cuda", generator=g)

    kv_len = torch.zeros(1, dtype=torch.int32, device="cuda")
    kv_indices = torch.zeros((1, pages), dtype=torch.int32, device="cuda")
    positions_out = torch.zeros(ROWS, dtype=torch.int64, device="cuda")
    slot_mapping = torch.full((ROWS,), -1, dtype=torch.int64, device="cuda")
    prepare_step_inputs(
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        positions=positions,
        out_cache_loc=out_cache_loc,
        kv_len=kv_len,
        kv_indices=kv_indices,
        positions_out=positions_out,
        slot_mapping=slot_mapping,
        page=PAGE,
        rows=ROWS,
    )
    want = _eager(seq_lens, req_pool_indices, req_to_token, positions, out_cache_loc, pages)
    for got, ref in zip((kv_len, kv_indices, positions_out, slot_mapping), want):
        assert torch.equal(got, ref)
