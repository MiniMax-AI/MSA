"""The sparse prefill split combine on its own.

Random partials and partial LSEs are merged against an FP32 torch reference, with NaN in every
partial slot past a row's valid split count so a masking defect shows up as a non-finite output.
Covers the split counts, head counts, partial and output dtypes, padded split slabs, head counts
whose LSE slabs are not 16-byte aligned, and the per-domain tile claiming of a locality-striped
workspace (with a synthetic SM table, so it runs on any supported GPU).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import pytest
import torch

from inference.msa_v1.attention.prefill._common.combine import LocalityHints, combine

logger = logging.getLogger(__name__)

HEAD_DIM = 128
Q_HEADS_PER_KV = 16
# One 2 MB page of 256-byte rows: the stripe granularity of a locality-domain workspace.
PAGE_BYTES = 2 << 20
PAGE_ROWS = PAGE_BYTES // (HEAD_DIM * 2)


@dataclass(frozen=True)
class CombineCase:
    name: str
    total_q: int
    num_kv_heads: int
    topk: int
    partial_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = torch.bfloat16
    # Query heads per KV head; a total head count that is not a multiple of 4 leaves the flat
    # row count unaligned for a 2-D LSE box.
    q_heads_per_kv: int = Q_HEADS_PER_KV
    split_counts: bool = True
    lse_out: bool = True
    # Pad every split slab to a multiple of this many flat rows (0: contiguous slabs).
    slab_rows: int = 0
    # Claim tiles per memory locality domain over a synthetic SM table (0: tile = CTA index).
    domains: int = 0


CASES = (
    CombineCase("h64_topk16", 333, 4, 16),
    # 16 heads: the flat rows are not a multiple of the tile, so the last tile is partial
    CombineCase("h16_topk16_partial_tile", 171, 1, 16),
    CombineCase("h32_topk4", 100, 2, 4),
    CombineCase("h64_topk8_f16_partials", 257, 4, 8, partial_dtype=torch.float16),
    CombineCase("h64_topk32", 129, 4, 32),
    CombineCase("h64_topk16_every_split", 65, 4, 16, split_counts=False, lse_out=False),
    CombineCase("one_query_f16_out", 1, 1, 16, out_dtype=torch.float16),
    CombineCase("h64_topk16_padded_slabs", 333, 4, 16, slab_rows=1024),
    CombineCase("h64_topk16_two_domains", 333, 4, 16, slab_rows=2 * PAGE_ROWS, domains=2),
    CombineCase(
        "h16_topk8_two_domains_partial_tile", 171, 1, 8, slab_rows=2 * PAGE_ROWS, domains=2
    ),
    # Flat row counts 333, 342 and 606: no split slab starts 16-byte aligned
    CombineCase("h1_topk16_odd_rows", 333, 1, 16, q_heads_per_kv=1),
    CombineCase("h2_topk8_f16_partials", 171, 1, 8, partial_dtype=torch.float16, q_heads_per_kv=2),
    CombineCase("h6_topk32_odd_rows", 101, 2, 32, q_heads_per_kv=3),
)


def _require_supported_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3), (10, 7)}:
        pytest.skip("the prefill combine requires SM100, SM103 or SM107")
    return device


def _real_to_fake_col(device: torch.device) -> torch.Tensor:
    """Fake (stored) column of every real column: inside each 32-column group the forward
    epilogue stores a 4x4 transpose of the 32-bit pairs."""
    col = torch.arange(HEAD_DIM, device=device)
    group_base, col32 = col // 32 * 32, col % 32
    return group_base + (col32 % 8) // 2 * 8 + col32 // 8 * 2 + col32 % 2


def _reference(partial, lse, valid):
    """FP32 merge of the valid splits: softmax weights over the partial LSEs, final LSE."""
    lse = torch.where(valid, lse, torch.full_like(lse, -math.inf))
    row_max = lse.amax(dim=0)
    safe_max = torch.where(torch.isinf(row_max), torch.zeros_like(row_max), row_max)
    weight = torch.exp(lse - safe_max)
    total = weight.sum(dim=0)
    inv_total = torch.where(total > 0, 1.0 / total, torch.zeros_like(total))
    partial = torch.where(
        valid[..., None], partial.float(), torch.zeros_like(partial, dtype=torch.float32)
    )
    out = ((weight * inv_total)[..., None] * partial).sum(dim=0)
    final_lse = torch.where(
        total > 0, row_max + torch.log(total), torch.full_like(row_max, -math.inf)
    )
    return out, final_lse


def _nan_fill(tensor: torch.Tensor, mask: torch.Tensor) -> None:
    """Write a quiet NaN into every element selected by ``mask`` (broadcast over trailing dims)."""
    nan = torch.tensor(math.nan, dtype=tensor.dtype, device=tensor.device)
    tensor.masked_fill_(mask.reshape(mask.shape + (1,) * (tensor.ndim - mask.ndim)), nan)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_combine_merges_the_valid_splits(case: CombineCase) -> None:
    device = _require_supported_device()
    gen = torch.Generator(device=device).manual_seed(hash(case.name) & 0xFFFF)
    heads = case.num_kv_heads * case.q_heads_per_kv
    shape = (case.topk, case.total_q, heads)

    partial_real = torch.randn(*shape, HEAD_DIM, generator=gen, device=device).to(
        case.partial_dtype
    )
    # Partial LSEs spread over a few units with a row offset, so the row max sits in a random split.
    lse_partial = torch.randn(*shape, generator=gen, device=device) * 2.0
    lse_partial += torch.randn(case.total_q, heads, generator=gen, device=device) * 4.0
    if case.split_counts:
        split_counts = torch.randint(
            1,
            case.topk + 1,
            (case.total_q, case.num_kv_heads),
            generator=gen,
            device=device,
            dtype=torch.int32,
        )
        split_counts[0::7] = case.topk
        split_counts[3::11] = 1
        counts_q = split_counts.repeat_interleave(case.q_heads_per_kv, dim=1)
    else:
        split_counts = None
        counts_q = torch.full((case.total_q, heads), case.topk, device=device, dtype=torch.int32)
    valid = torch.arange(case.topk, device=device)[:, None, None] < counts_q[None]

    ref_out, ref_lse = _reference(partial_real, lse_partial, valid)

    # The kernel's inputs: fake column order, and NaN wherever a split is past the row's count
    # (and in the slab padding, which no tile may read).
    flat_rows = case.total_q * heads
    slab_rows = (
        flat_rows if case.slab_rows == 0 else -(-flat_rows // case.slab_rows) * case.slab_rows
    )
    o_partial = torch.full(
        (case.topk * slab_rows * HEAD_DIM,), math.nan, dtype=case.partial_dtype, device=device
    ).as_strided(
        (case.topk, case.total_q, heads, HEAD_DIM),
        (slab_rows * HEAD_DIM, heads * HEAD_DIM, HEAD_DIM, 1),
    )
    o_partial[..., _real_to_fake_col(device)] = partial_real
    _nan_fill(o_partial, ~valid)
    locality = None
    if case.domains:
        locality = LocalityHints(
            sm_domains=torch.arange(4096, dtype=torch.int32, device=device) % case.domains,
            tile_counters=torch.zeros(case.domains, dtype=torch.int32, device=device),
            page_bytes=PAGE_BYTES,
        )
    lse_in = lse_partial.clone()
    _nan_fill(lse_in, ~valid)
    out = torch.empty(case.total_q, heads, HEAD_DIM, dtype=case.out_dtype, device=device)
    lse_out = (
        torch.empty(case.total_q, heads, dtype=torch.float32, device=device)
        if case.lse_out
        else None
    )

    def run():
        if locality is not None:
            # Zero at launch is the contract; the forward kernel does it in production.
            locality.tile_counters.zero_()
        combine(
            o_partial,
            lse_in,
            out,
            lse_out,
            split_counts=split_counts,
            use_pdl=True,
            locality=locality,
        )
        return out.clone(), None if lse_out is None else lse_out.clone()

    torch.cuda.synchronize()
    started = time.perf_counter()
    first = run()
    torch.cuda.synchronize()
    logger.info("combine %s: first call (compile + run) %.3fs", case.name, time.perf_counter() - started)
    started = time.perf_counter()
    second = run()
    torch.cuda.synchronize()
    logger.info("combine %s: ran in %.3fms", case.name, (time.perf_counter() - started) * 1e3)

    if locality is not None:  # every CTA claimed exactly once
        assert int(locality.tile_counters.sum()) >= -(-case.total_q * heads // 64)
    assert torch.isfinite(first[0]).all(), "a stale partial reached the output"
    # The output is the FP32 merge rounded once to the output dtype.
    torch.testing.assert_close(first[0].float(), ref_out, rtol=1e-2, atol=2e-2)
    if case.lse_out:
        assert torch.isfinite(first[1]).all()
        torch.testing.assert_close(first[1], ref_lse, rtol=1e-5, atol=1e-4)
    # Same inputs, same split order: the merge is deterministic.
    for a, b in zip(first, second):
        assert a is None or torch.equal(a, b), f"{case.name}: the combine is not deterministic"
