# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Self-contained inputs for the Q8KV4 sparse decode tests.

Random E2M1 codes and E4M3 block scales, a scattered physical page table, and TopK lists that
follow the kernel's contract: a prefix of valid logical page ids (history pages in arbitrary
order, the query's own page last) followed by ``-1`` padding.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

import torch

from fmha_sm100.decode_q8kv4 import interleave_v_scales

PAGE_SIZE = 128
HEAD_DIM = 128
SCALE_GROUPS = HEAD_DIM // 16
KV_HEADS = 4
SM_SCALE = HEAD_DIM**-0.5
# Block-scale bytes in [2^-4, 1): code x scale fits E4M3 without staging, and scaling the bytes
# by 2^8 (the TE-style tests) stays finite.
SCALE_BYTE_RANGE = (0x18, 0x39)


@dataclass(frozen=True)
class DecodeCase:
    name: str
    seq_lens: tuple[int, ...]
    q_len: int = 1
    topk: int = 16
    seed: int = 0

    @property
    def batch_size(self) -> int:
        return len(self.seq_lens)


SMOKE_CASES = (
    DecodeCase("b2_s700_1100_q4", (700, 1_100), q_len=4, seed=1),
    DecodeCase("b4_ragged_q1", (129, 2_048, 40_000, 5_000), q_len=1, seed=2),
    DecodeCase("b8_s100000_q8", (100_000,) * 8, q_len=8, seed=3),
)
FULL_CASES = (
    DecodeCase("b1_one_page", (128,), seed=11),
    DecodeCase("b1_one_token", (1,), seed=12),
    DecodeCase("b16_ragged_q4", tuple(range(1_000, 17_000, 1_000)), q_len=4, seed=13),
    DecodeCase("b32_s10000_q8", (10_000,) * 32, q_len=8, seed=14),
    DecodeCase("b64_s1000_q8", (1_000,) * 64, q_len=8, seed=15),
    DecodeCase("b128_s1000_q1", (1_000,) * 128, q_len=1, seed=16),
    DecodeCase("b8_s20000_topk8", (20_000,) * 8, q_len=4, topk=8, seed=17),
    DecodeCase("b4_s50000_topk32", (50_000,) * 4, q_len=4, topk=32, seed=18),
    DecodeCase("b2_s100000_topk64", (100_000,) * 2, q_len=2, topk=64, seed=19),
    DecodeCase("b4_s3000_topk1", (3_000,) * 4, q_len=4, topk=1, seed=20),
)


@dataclass
class DecodeInputs:
    case: DecodeCase
    gqa: int
    q: torch.Tensor  # [B * q_len, Hq, 128] E4M3
    k_codes: torch.Tensor  # [P, Hkv, 128, 64] uint8, two E2M1 per byte
    v_codes: torch.Tensor
    k_scale: torch.Tensor  # [P, Hkv, 128, 8] E4M3, linear token rows
    v_scale: torch.Tensor  # same layout; callers interleave it for the kernel
    page_table: torch.Tensor  # [B, max_pages] int32, zero past a request's pages
    seq_lens: torch.Tensor  # [B] int32
    topk_indices: torch.Tensor  # [B * q_len, Hkv, topk] int32

    @property
    def num_q_heads(self) -> int:
        return KV_HEADS * self.gqa

    @property
    def v_scale_kernel(self) -> torch.Tensor:
        return interleave_v_scales(self.v_scale)


def page_counts(seq_lens) -> list[int]:
    return [(int(length) + PAGE_SIZE - 1) // PAGE_SIZE for length in seq_lens]


def make_topk_lists(case: DecodeCase, num_kv_heads: int, rng: random.Random) -> torch.Tensor:
    """History pages in random order, the query's own page last, ``-1`` padding."""
    rows = []
    for length in case.seq_lens:
        for token in range(case.q_len):
            position = length - case.q_len + token
            local_page = position // PAGE_SIZE
            for _ in range(num_kv_heads):
                history = rng.sample(range(local_page), min(case.topk - 1, local_page))
                entries = history + [local_page]
                rows.append(entries + [-1] * (case.topk - len(entries)))
    return torch.tensor(rows, dtype=torch.int32).reshape(
        case.batch_size * case.q_len, num_kv_heads, case.topk
    )


def make_inputs(case: DecodeCase, device: torch.device, *, gqa: int = 16) -> DecodeInputs:
    assert all(length >= case.q_len for length in case.seq_lens), "q_len exceeds a KV length"
    generator = torch.Generator(device=device).manual_seed(case.seed)
    rng = random.Random(case.seed)
    counts = page_counts(case.seq_lens)
    total_pages = sum(counts) + 8  # spare pages nobody references
    permutation = torch.randperm(total_pages, generator=generator, device=device).to(torch.int32)
    page_table = torch.zeros((case.batch_size, max(counts)), dtype=torch.int32, device=device)
    offset = 0
    for batch, count in enumerate(counts):
        page_table[batch, :count] = permutation[offset : offset + count]
        offset += count
    rows = case.batch_size * case.q_len
    q = (torch.randn((rows, KV_HEADS * gqa, HEAD_DIM), generator=generator, device=device) * 0.5)
    q = q.to(torch.float8_e4m3fn)
    data_shape = (total_pages, KV_HEADS, PAGE_SIZE, HEAD_DIM // 2)
    scale_shape = (total_pages, KV_HEADS, PAGE_SIZE, SCALE_GROUPS)
    k_codes, v_codes = (
        torch.randint(0, 256, data_shape, dtype=torch.uint8, generator=generator, device=device)
        for _ in range(2)
    )
    k_scale, v_scale = (
        torch.randint(*SCALE_BYTE_RANGE, scale_shape, dtype=torch.uint8, generator=generator,
                      device=device).view(torch.float8_e4m3fn)
        for _ in range(2)
    )
    return DecodeInputs(
        case=case,
        gqa=gqa,
        q=q,
        k_codes=k_codes,
        v_codes=v_codes,
        k_scale=k_scale,
        v_scale=v_scale,
        page_table=page_table,
        seq_lens=torch.tensor(case.seq_lens, dtype=torch.int32, device=device),
        topk_indices=make_topk_lists(case, KV_HEADS, rng).to(device),
    )


def flat_page_table(page_table: torch.Tensor, seq_lens) -> tuple[torch.Tensor, torch.Tensor]:
    """The 2-D table as a flat physical page list plus per-request bases."""
    counts = page_counts(seq_lens)
    kv_indices = torch.cat([page_table[batch, :count] for batch, count in enumerate(counts)])
    kv_indptr = torch.tensor([0, *itertools.accumulate(counts)], dtype=torch.int32,
                             device=page_table.device)
    return kv_indices.contiguous(), kv_indptr


def scale_bytes_times_pow2(scale: torch.Tensor, exponent: int) -> torch.Tensor:
    """Multiply E4M3 block scales by 2**exponent exactly (an exponent-field shift)."""
    bits = scale.view(torch.uint8)
    assert int(bits.max()) + (exponent << 3) <= 0x7E, "E4M3 scale overflow"
    return (bits + (exponent << 3)).view(torch.float8_e4m3fn)


def global_scale(value: float, device: torch.device) -> torch.Tensor:
    return torch.full((1,), value, dtype=torch.float32, device=device)


def pack_vllm_pages(
    inputs: DecodeInputs, *, pad_bytes: int = 0, offset_bytes: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """The vLLM NVFP4 layout: per page all heads' data blocks then all heads' scale blocks,
    K scales linear and V scales in token-quad order, as two ``[P, H, 128, 72]`` uint8 tensors.

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
        packed.append(buffer.as_strided((total_pages, heads, PAGE_SIZE, 72), (stride, 72 * PAGE_SIZE, 72, 1), offset_bytes))
    return packed[0], packed[1]


def pack_head_slot_pages(inputs: DecodeInputs) -> tuple[torch.Tensor, torch.Tensor]:
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
