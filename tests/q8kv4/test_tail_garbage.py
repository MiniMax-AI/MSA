# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Cache bytes past a request's length never reach the output.

The query's own page is only partly written. vLLM reuses cache blocks across cache groups of
different formats, so the unwritten token slots can hold any bytes, including E4M3 NaN block
scales (0x7F / 0xFF). Those tokens are masked (P = 0), but the PV MMA still multiplies their V
rows, so a NaN there used to turn the whole output row into NaN. Every poisoned run must be
bitwise equal to the clean one.
"""

import pytest
import torch

from .cases import SMOKE_CASES, DecodeCase, make_inputs, scale_bytes_times_pow2
from .runners import run_wrapper

PAGE_SIZE = 128
CASES = [pytest.param(case, id=case.name) for case in SMOKE_CASES[:2]] + [
    pytest.param(DecodeCase("b6_mid_page_q4", (1_000, 1_001, 1_026, 1_150, 3_000, 4_095), q_len=4,
                            seed=51), id="b6_mid_page_q4"),
]


def _poison_past_length(inputs, *, sides, generator) -> None:
    """NaN block scales and random codes in every slot of the local page past the length."""
    for batch, length in enumerate(inputs.case.seq_lens):
        page, slot = (length - 1) // PAGE_SIZE, (length - 1) % PAGE_SIZE
        if slot + 1 == PAGE_SIZE:
            continue
        physical = int(inputs.page_table[batch, page])
        for side in sides:
            codes = inputs.k_codes if side == "k" else inputs.v_codes
            scale = (inputs.k_scale if side == "k" else inputs.v_scale).view(torch.uint8)
            tail = codes[physical, :, slot + 1:]
            tail.copy_(torch.randint(0, 256, tail.shape, dtype=torch.uint8, generator=generator,
                                     device=codes.device))
            nan = torch.tensor((0x7F, 0xFF), dtype=torch.uint8, device=codes.device)
            tail_scale = scale[physical, :, slot + 1:]
            pick = torch.randint(0, 2, tail_scale.shape, generator=generator, device=codes.device)
            tail_scale.copy_(nan[pick])


@pytest.mark.parametrize("block_scale_shift", (0, 3), ids=("shift0", "shift3"))
@pytest.mark.parametrize("num_kv_splits", (None, 4), ids=("auto", "split4"))
@pytest.mark.parametrize("gqa", (16, 8), ids=("gqa16", "gqa8"))
@pytest.mark.parametrize("sides", (("v",), ("k", "v")), ids=("v", "kv"))
@pytest.mark.parametrize("case", CASES)
def test_garbage_past_length_is_ignored(device, case, sides, gqa, num_kv_splits, block_scale_shift):
    inputs = make_inputs(case, device, gqa=gqa)
    if block_scale_shift:
        inputs.k_scale = scale_bytes_times_pow2(inputs.k_scale, block_scale_shift)
        inputs.v_scale = scale_bytes_times_pow2(inputs.v_scale, block_scale_shift)
    label = f"{case.name} gqa{gqa} {'+'.join(sides)} splits={num_kv_splits} shift{block_scale_shift}"
    clean = run_wrapper(inputs, num_kv_splits=num_kv_splits, block_scale_shift=block_scale_shift,
                        label=f"{label} clean")
    assert torch.isfinite(clean).all(), f"{label}: clean run is not finite"
    generator = torch.Generator(device=device).manual_seed(case.seed + 7)
    _poison_past_length(inputs, sides=sides, generator=generator)
    poisoned = run_wrapper(inputs, num_kv_splits=num_kv_splits, block_scale_shift=block_scale_shift,
                           label=f"{label} poisoned")
    assert torch.isfinite(poisoned).all(), f"{label}: bytes past the length reached the output"
    assert torch.equal(poisoned, clean), f"{label}: bytes past the length changed the output"
