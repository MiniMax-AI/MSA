# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Cache layouts and page indexing: every variant must reproduce the contiguous run bitwise."""

import pytest
import torch

from fmha_sm100.api import nvfp4_head_slot_views

from .cases import SMOKE_CASES, make_inputs, pack_head_slot_pages, pack_vllm_pages, unpack_views
from .runners import run_wrapper

CASES = [pytest.param(case, id=case.name) for case in SMOKE_CASES[:2]]


@pytest.mark.parametrize("case", CASES)
def test_flat_list_and_packed_pages_match_the_contiguous_run(device, case):
    inputs = make_inputs(case, device)
    baseline = run_wrapper(inputs, label=f"{case.name} contiguous")
    flat = run_wrapper(inputs, plan_kwargs={"flat": True}, label=f"{case.name} flat list")
    assert all(map(torch.equal, flat, baseline)), "flat page list with per-request bases"
    for label, layout in (("packed", {}), ("padded page stride", {"pad_bytes": 256}),
                          ("storage offset", {"offset_bytes": 4096})):
        k_packed, v_packed = pack_vllm_pages(inputs, **layout)
        k_data, k_sf = unpack_views(k_packed)
        v_data, v_sf = unpack_views(v_packed)
        out = run_wrapper(inputs, kv=(k_data, v_data), kv_sf=(k_sf, v_sf),
                          plan_kwargs={"flat": True}, label=f"{case.name} {label}")
        assert all(map(torch.equal, out, baseline)), f"side-packed pages, {label}"
    k_data, k_sf, v_data, v_sf = nvfp4_head_slot_views(*pack_head_slot_pages(inputs))
    out = run_wrapper(inputs, kv=(k_data, v_data),
                      kv_sf=(k_sf.view(torch.float8_e4m3fn), v_sf.view(torch.float8_e4m3fn)),
                      plan_kwargs={"flat": True}, label=f"{case.name} head slots")
    assert all(map(torch.equal, out, baseline)), "per-head K/V slot pages (head stride of two slots)"


@pytest.mark.parametrize("case", CASES)
def test_strided_page_table_matches_the_flat_list(device, case):
    """vLLM passes a block table that is a row/column slice of a wider persistent buffer."""
    inputs = make_inputs(case, device)
    flat = run_wrapper(inputs, plan_kwargs={"flat": True}, label=f"{case.name} flat list")
    batch, width = inputs.page_table.shape
    buffer = torch.full((batch + 3, width + 37), -1, dtype=torch.int32, device=device)
    buffer[2 : 2 + batch, :width] = inputs.page_table
    inputs.page_table = buffer[2 : 2 + batch, :width]
    assert not inputs.page_table.is_contiguous()
    strided = run_wrapper(inputs, label=f"{case.name} strided table")
    assert all(map(torch.equal, strided, flat)), "strided [batch, max_pages] table"
