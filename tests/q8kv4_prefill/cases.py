# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Self-contained inputs for the Q8KV4 sparse prefill tests.

Random E2M1 codes and E4M3 block scales, a scattered physical page table, and chunked-prefill
requests: each request appends ``q_len`` tokens to ``k_len - q_len`` cached ones. Every query
selects up to ``topk`` pages at or before its own: history pages in random order, then its own
page, then ``-1`` padding (``ascending=True`` sorts the valid prefix, the ``fmha_sm100``
contract).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

import torch

from fmha_sm100.prefill_q8kv4 import interleave_v_scales

PAGE_SIZE = 128
HEAD_DIM = 128
Q_HEADS_PER_KV = 16
SCALE_GROUPS = HEAD_DIM // 16
SM_SCALE = HEAD_DIM**-0.5
# Block-scale bytes in [2^-4, 1): code x scale fits E4M3 without staging, and scaling the bytes
# by 2^8 (the TE-style tests) stays finite.
SCALE_BYTE_RANGE = (0x18, 0x39)


@dataclass(frozen=True)
class PrefillCase:
    name: str
    q_lens: tuple[int, ...]
    k_lens: tuple[int, ...]
    topk: int = 16
    num_kv_heads: int = 4
    seed: int = 0
    ascending: bool = False

    @property
    def batch_size(self) -> int:
        return len(self.q_lens)

    @property
    def total_q(self) -> int:
        return sum(self.q_lens)

    @property
    def num_q_heads(self) -> int:
        return self.num_kv_heads * Q_HEADS_PER_KV


SMOKE_CASES = (
    # nv_dev's chunked case: two requests, partial last pages, one query per page boundary.
    PrefillCase("b2_q8_9_k131_260", (8, 9), (131, 260), seed=1),
    PrefillCase("b3_ragged_prefix", (300, 64, 513), (300, 2_000, 4_609), seed=2),
    PrefillCase("b1_q2048_k8192_kv1", (2_048,), (8_192,), num_kv_heads=1, seed=3),
)
FULL_CASES = (
    PrefillCase("b1_q1_k1", (1,), (1,), seed=11),
    PrefillCase("b1_q128_k128", (128,), (128,), seed=12),
    PrefillCase("b4_fresh_prefill", (1_000, 257, 128, 3_000), (1_000, 257, 128, 3_000), seed=13),
    PrefillCase("b2_long_prefix", (512, 1_024), (65_536, 20_000), seed=14),
    PrefillCase("b2_topk4", (700, 300), (2_700, 900), topk=4, seed=15),
    PrefillCase("b2_topk8", (700, 300), (2_700, 900), topk=8, seed=16),
    PrefillCase("b2_topk32", (700, 300), (9_700, 4_900), topk=32, seed=17),
    PrefillCase("b8_ragged_kv1", (64, 1, 200, 129, 7, 512, 33, 1_000),
                (64, 5_000, 200, 1_129, 7, 512, 3_033, 1_000), num_kv_heads=1, seed=18),
    PrefillCase("b1_q4096_k32768", (4_096,), (32_768,), seed=19),
)


@dataclass
class PrefillInputs:
    case: PrefillCase
    q: torch.Tensor  # [total_q, Hq, 128] E4M3
    k_codes: torch.Tensor  # [P, Hkv, 128, 64] uint8, two E2M1 per byte
    v_codes: torch.Tensor
    k_scale: torch.Tensor  # [P, Hkv, 128, 8] E4M3, linear token rows
    v_scale: torch.Tensor  # same layout; callers interleave it for the kernel
    page_table: torch.Tensor  # [B, max_pages] int32, zero past a request's pages
    kv_indices: torch.Tensor  # the table as a flat physical page list
    kv_indptr: torch.Tensor  # [B + 1] int32, each request's first entry in kv_indices
    cu_seqlens_q: torch.Tensor  # [B + 1] int32
    cu_seqlens_k: torch.Tensor  # [B + 1] int32
    topk_indices: torch.Tensor  # [Hkv, total_q, topk] int32, the wrapper's (q2k) layout

    @property
    def v_scale_kernel(self) -> torch.Tensor:
        return interleave_v_scales(self.v_scale)

    @property
    def kv_block_indexes(self) -> torch.Tensor:
        """The TopK lists in ``fmha_sm100``'s ``[total_q, Hkv, topk]`` layout."""
        return self.topk_indices.permute(1, 0, 2).contiguous()

    @property
    def total_rows(self) -> int:
        return sum(page_counts(self.case.k_lens))


def page_counts(k_lens) -> list[int]:
    return [(int(length) + PAGE_SIZE - 1) // PAGE_SIZE for length in k_lens]


def query_positions(case: PrefillCase) -> list[int]:
    """Each query's position in its request's KV sequence (bottom-right causal alignment)."""
    return [
        k_len - q_len + index
        for q_len, k_len in zip(case.q_lens, case.k_lens, strict=True)
        for index in range(q_len)
    ]


def make_topk_lists(case: PrefillCase, rng: random.Random) -> torch.Tensor:
    """``[Hkv, total_q, topk]``: history pages, then the query's own page, then ``-1``."""
    lists = torch.full((case.num_kv_heads, case.total_q, case.topk), -1, dtype=torch.int32)
    for row, position in enumerate(query_positions(case)):
        local_page = position // PAGE_SIZE
        for head in range(case.num_kv_heads):
            history = rng.sample(range(local_page), min(case.topk - 1, local_page))
            entries = (sorted(history) if case.ascending else history) + [local_page]
            lists[head, row, : len(entries)] = torch.tensor(entries, dtype=torch.int32)
    return lists


def make_inputs(case: PrefillCase, device: torch.device) -> PrefillInputs:
    assert all(q <= k for q, k in zip(case.q_lens, case.k_lens, strict=True)), "q_len > k_len"
    generator = torch.Generator(device=device).manual_seed(case.seed)
    rng = random.Random(case.seed)
    counts = page_counts(case.k_lens)
    total_pages = sum(counts) + 8  # spare pages nobody references
    permutation = torch.randperm(total_pages, generator=generator, device=device).to(torch.int32)
    page_table = torch.zeros((case.batch_size, max(counts)), dtype=torch.int32, device=device)
    offset = 0
    for batch, count in enumerate(counts):
        page_table[batch, :count] = permutation[offset : offset + count]
        offset += count
    kv_indices = permutation[:offset].contiguous()
    heads = case.num_kv_heads
    q = torch.randn((case.total_q, case.num_q_heads, HEAD_DIM), generator=generator, device=device)
    q = (q * 0.5).to(torch.float8_e4m3fn)
    data_shape = (total_pages, heads, PAGE_SIZE, HEAD_DIM // 2)
    scale_shape = (total_pages, heads, PAGE_SIZE, SCALE_GROUPS)
    k_codes, v_codes = (
        torch.randint(0, 256, data_shape, dtype=torch.uint8, generator=generator, device=device)
        for _ in range(2)
    )
    k_scale, v_scale = (
        torch.randint(*SCALE_BYTE_RANGE, scale_shape, dtype=torch.uint8, generator=generator,
                      device=device).view(torch.float8_e4m3fn)
        for _ in range(2)
    )

    def prefix_sums(lengths):
        return torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device=device)

    return PrefillInputs(
        case=case,
        q=q,
        k_codes=k_codes,
        v_codes=v_codes,
        k_scale=k_scale,
        v_scale=v_scale,
        page_table=page_table,
        kv_indices=kv_indices,
        kv_indptr=prefix_sums(counts),
        cu_seqlens_q=prefix_sums(case.q_lens),
        cu_seqlens_k=prefix_sums(case.k_lens),
        topk_indices=make_topk_lists(case, rng).to(device),
    )


def scale_bytes_times_pow2(scale: torch.Tensor, exponent: int) -> torch.Tensor:
    """Multiply E4M3 block scales by 2**exponent exactly (an exponent-field shift)."""
    bits = scale.view(torch.uint8)
    assert int(bits.max()) + (exponent << 3) <= 0x7E, "E4M3 scale overflow"
    return (bits + (exponent << 3)).view(torch.float8_e4m3fn)


def global_scale(value: float, device: torch.device) -> torch.Tensor:
    return torch.full((1,), value, dtype=torch.float32, device=device)


def pack_vllm_pages(
    inputs: PrefillInputs, *, pad_bytes: int = 0, offset_bytes: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """The side-packed NVFP4 layout: per page all heads' data blocks then all heads' scale
    blocks, K scales linear and V scales in token-quad order, as two ``[P, H, 128, 72]`` uint8
    tensors. The direct Q8KV4 API reads it through ``unpack_views``; ``fmha_sm100`` takes
    ``pack_head_slot_pages`` instead.

    ``pad_bytes`` widens the page stride; ``offset_bytes`` places the pages inside a larger
    buffer, so both layout freedoms of the kernel are exercised.
    """
    total_pages, heads = inputs.k_codes.shape[:2]
    stride = heads * (PAGE_SIZE * 64 + PAGE_SIZE * 8) + pad_bytes
    packed = []
    for codes, scale in ((inputs.k_codes, inputs.k_scale), (inputs.v_codes, inputs.v_scale_kernel)):
        buffer = torch.zeros(offset_bytes + total_pages * stride, dtype=torch.uint8,
                             device=codes.device)
        buffer.as_strided((total_pages, heads, PAGE_SIZE, 64), (stride, PAGE_SIZE * 64, 64, 1),
                          offset_bytes).copy_(codes)
        buffer.as_strided((total_pages, heads, PAGE_SIZE, 8), (stride, PAGE_SIZE * 8, 8, 1),
                          offset_bytes + heads * PAGE_SIZE * 64).copy_(scale.view(torch.uint8))
        packed.append(buffer.as_strided((total_pages, heads, PAGE_SIZE, 72),
                                        (stride, 72 * PAGE_SIZE, 72, 1), offset_bytes))
    return packed[0], packed[1]


def pack_head_slot_pages(inputs: PrefillInputs) -> tuple[torch.Tensor, torch.Tensor]:
    """The NVFP4 cache ``fmha_sm100`` reads: one ``[P, 2 * H, 128, 72]`` uint8 buffer where slot
    ``2 * h`` is head ``h``'s K (its data block, then its scale block) and slot ``2 * h + 1`` its
    V. Returns the K and V slot views ``cache[:, 0::2]`` and ``cache[:, 1::2]``.
    """
    total_pages, heads = inputs.k_codes.shape[:2]
    cache = torch.zeros(total_pages, 2 * heads, PAGE_SIZE, 72, dtype=torch.uint8,
                        device=inputs.k_codes.device)
    k, v = cache[:, 0::2], cache[:, 1::2]
    for slots, codes, scale in ((k, inputs.k_codes, inputs.k_scale),
                                (v, inputs.v_codes, inputs.v_scale_kernel)):
        slot_bytes = slots.flatten(2)
        slot_bytes[..., :PAGE_SIZE * 64].copy_(codes.flatten(2))
        slot_bytes[..., PAGE_SIZE * 64:].copy_(scale.view(torch.uint8).flatten(2))
    return k, v


def unpack_views(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Data and scale views over a ``pack_vllm_pages`` ``[P, H, 128, 72]`` tensor, for the
    direct Q8KV4 API."""
    total_pages, heads = packed.shape[:2]
    data = packed.as_strided((total_pages, heads, PAGE_SIZE, 64), (packed.stride(0), 8192, 64, 1))
    scale = packed.as_strided((total_pages, heads, PAGE_SIZE, 8), (packed.stride(0), 1024, 8, 1),
                              packed.storage_offset() + heads * 8192)
    return data, scale.view(torch.float8_e4m3fn)
