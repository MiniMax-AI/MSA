# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Cache layouts and page indexing: every variant must reproduce the contiguous run bitwise."""

import pytest
import torch

from fmha_sm100.api import nvfp4_head_slot_views
from fmha_sm100.decode_q8kv4 import BatchDecodeWithPagedKVCacheWrapper

from .cases import KV_HEADS, SMOKE_CASES, flat_page_table, make_inputs, pack_head_slot_pages, pack_vllm_pages, unpack_views
from .conftest import run_timed
from .runners import run_wrapper

CASES = [pytest.param(case, id=case.name) for case in SMOKE_CASES[:2]]


def _run_views(inputs, k_data, v_data, k_sf, v_sf, kv_indices, kv_indptr, label):
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    wrapper.plan(inputs.topk_indices, kv_indices, inputs.seq_lens, q_len_per_req=inputs.case.q_len,
                 num_q_heads=inputs.num_q_heads, num_kv_heads=KV_HEADS, kv_indptr=kv_indptr)
    return run_timed(label, lambda: wrapper.run(inputs.q, (k_data, v_data), kv_cache_sf=(k_sf, v_sf)).clone())


@pytest.mark.parametrize("case", CASES)
def test_flat_list_and_packed_pages_match_the_contiguous_run(device, case):
    inputs = make_inputs(case, device)
    baseline = run_wrapper(inputs, label=f"{case.name} contiguous")
    kv_indices, kv_indptr = flat_page_table(inputs.page_table, case.seq_lens)
    out = _run_views(inputs, inputs.k_codes, inputs.v_codes, inputs.k_scale, inputs.v_scale_kernel,
                     kv_indices, kv_indptr, f"{case.name} flat list")
    assert torch.equal(out, baseline), "flat page list with per-request bases"
    for label, layout in (("packed", {}), ("padded page stride", {"pad_bytes": 256}),
                          ("storage offset", {"offset_bytes": 4096})):
        k_packed, v_packed = pack_vllm_pages(inputs, **layout)
        k_data, k_sf = unpack_views(k_packed)
        v_data, v_sf = unpack_views(v_packed)
        out = _run_views(inputs, k_data, v_data, k_sf, v_sf, kv_indices, kv_indptr, f"{case.name} {label}")
        assert torch.equal(out, baseline), f"vLLM packed pages, {label}"
    k_data, k_sf, v_data, v_sf = nvfp4_head_slot_views(*pack_head_slot_pages(inputs))
    out = _run_views(inputs, k_data, v_data, k_sf.view(torch.float8_e4m3fn), v_sf.view(torch.float8_e4m3fn),
                     kv_indices, kv_indptr, f"{case.name} head slots")
    assert torch.equal(out, baseline), "per-head K/V slot pages (head stride of two slots)"
