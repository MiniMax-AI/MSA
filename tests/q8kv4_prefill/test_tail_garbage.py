# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Cache bytes past a request's KV length never reach the prefill output.

The last page of a request is only partly written, and vLLM reuses cache blocks across cache
groups of different formats, so its unwritten token slots can hold any bytes, including E4M3 NaN
block scales (0x7F / 0xFF). Every poisoned run must be bitwise equal to the clean one.
"""

import pytest
import torch

from .cases import PAGE_SIZE, SMOKE_CASES, PrefillCase, make_inputs
from .runners import run_wrapper

CASES = [pytest.param(case, id=case.name) for case in SMOKE_CASES] + [
    pytest.param(PrefillCase("b3_mid_page_kv1", (5, 130, 64), (1_005, 2_130, 4_064), num_kv_heads=1,
                             seed=31), id="b3_mid_page_kv1"),
]


def _poison_past_length(inputs, *, sides, generator) -> None:
    for batch, length in enumerate(inputs.case.k_lens):
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


@pytest.mark.parametrize("sides", (("v",), ("k", "v")), ids=("v", "kv"))
@pytest.mark.parametrize("case", CASES)
def test_garbage_past_length_is_ignored(device, case, sides):
    inputs = make_inputs(case, device)
    label = f"{case.name} {'+'.join(sides)}"
    clean, clean_lse = run_wrapper(inputs, label=f"{label} clean")
    assert torch.isfinite(clean).all(), f"{label}: clean run is not finite"
    generator = torch.Generator(device=device).manual_seed(case.seed + 7)
    _poison_past_length(inputs, sides=sides, generator=generator)
    poisoned, poisoned_lse = run_wrapper(inputs, label=f"{label} poisoned")
    assert torch.isfinite(poisoned).all(), f"{label}: bytes past the length reached the output"
    assert torch.equal(poisoned, clean), f"{label}: bytes past the length changed the output"
    assert torch.equal(poisoned_lse, clean_lse), f"{label}: bytes past the length changed the LSE"
