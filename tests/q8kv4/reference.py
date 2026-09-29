# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Independent fp32 reference for Q8KV4 sparse decode.

Dequantization follows the cache's data contract rather than the kernel: each value is
``E2M1(code) x E4M3(block_scale) x global_scale``. The kernel feeds ``code x block_scale`` to an
FP8 tensor core, so that product is requantized to E4M3 here as well, with the block scale
staged by ``2**-block_scale_shift`` first when the cache uses that convention.
"""

from __future__ import annotations

import torch

from .cases import HEAD_DIM, PAGE_SIZE, SM_SCALE, DecodeInputs

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
E4M3_MAX = 448.0


def dequantize(
    codes: torch.Tensor,
    scale: torch.Tensor,
    *,
    global_scale: float = 1.0,
    block_scale_shift: int = 0,
) -> torch.Tensor:
    """``[..., 128, 64]`` packed codes and ``[..., 128, 8]`` linear scales to fp32 values."""
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=codes.device)
    values = torch.stack((codes & 0x0F, codes >> 4), dim=-1).reshape(*codes.shape[:-1], HEAD_DIM)
    staged_scale = (scale.float() / (1 << block_scale_shift)).clamp(-E4M3_MAX, E4M3_MAX)
    staged_scale = staged_scale.to(torch.float8_e4m3fn).float()
    products = lut[values.long()] * staged_scale.repeat_interleave(16, dim=-1)
    requantized = products.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    return requantized * float(1 << block_scale_shift) * global_scale


class PageDequantizer:
    """Dequantizes one (physical page, head) block on demand and caches it.

    The reference only touches the selected pages, so this keeps the fp32 values bounded by the
    pages a case actually reads instead of the whole cache.
    """

    def __init__(self, codes: torch.Tensor, scale: torch.Tensor, *, global_scale: float = 1.0,
                 block_scale_shift: int = 0):
        self._codes = codes
        self._scale = scale
        self._global_scale = global_scale
        self._block_scale_shift = block_scale_shift
        self._cache: dict[tuple[int, int], torch.Tensor] = {}

    def __call__(self, physical_page: int, head: int) -> torch.Tensor:
        key = (physical_page, head)
        block = self._cache.get(key)
        if block is None:
            block = dequantize(self._codes[physical_page, head], self._scale[physical_page, head],
                               global_scale=self._global_scale, block_scale_shift=self._block_scale_shift)
            self._cache[key] = block
        return block


def sparse_decode_reference(
    inputs: DecodeInputs,
    k_values: PageDequantizer,
    v_values: PageDequantizer,
    *,
    sm_scale: float = SM_SCALE,
    topk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attention of every query over its selected pages, fp32 math, bf16 output.

    The selection is the list's leading entries that lie in ``[0, local_page]``; the local page
    is visible up to the query's own token, every other selected page in full. An empty
    selection yields zeros.
    """
    case = inputs.case
    topk = inputs.topk_indices if topk_indices is None else topk_indices
    num_kv_heads = inputs.k_codes.shape[1]
    gqa = inputs.num_q_heads // num_kv_heads
    q = inputs.q.float()
    out = torch.zeros((q.shape[0], inputs.num_q_heads, HEAD_DIM), dtype=torch.float32,
                      device=q.device)
    seq_lens = inputs.seq_lens.tolist()
    page_table = inputs.page_table.tolist()
    lists = topk.tolist()
    for batch, length in enumerate(seq_lens):
        for token in range(case.q_len):
            row = batch * case.q_len + token
            position = length - case.q_len + token
            local_page = position // PAGE_SIZE
            for head in range(num_kv_heads):
                keys, values = [], []
                for page in lists[row][head]:
                    if page < 0 or page > local_page:
                        break
                    physical = page_table[batch][page]
                    visible = PAGE_SIZE if page < local_page else position % PAGE_SIZE + 1
                    keys.append(k_values(physical, head)[:visible])
                    values.append(v_values(physical, head)[:visible])
                if not keys:
                    continue
                k = torch.cat(keys)
                v = torch.cat(values)
                q_group = q[row, head * gqa : (head + 1) * gqa]
                probabilities = torch.softmax((q_group @ k.T) * sm_scale, dim=-1)
                out[row, head * gqa : (head + 1) * gqa] = probabilities @ v
    return out.to(torch.bfloat16)


def assert_close_to_reference(out: torch.Tensor, reference: torch.Tensor, *, label: str) -> None:
    """Whole-tensor comparison: finite output, |diff| <= 0.05 + 0.05 |reference| everywhere."""
    assert torch.isfinite(out).all(), f"{label}: non-finite output"
    diff = (out.float() - reference.float()).abs()
    violations = int((diff > 0.05 + 0.05 * reference.float().abs()).sum())
    assert violations == 0, (
        f"{label}: {violations} of {diff.numel()} elements outside tolerance, "
        f"max |diff| {diff.max().item():.4f}"
    )
