# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Selection-list contract and interface validation of the Q8KV4 wrapper."""

import random

import pytest
import torch

from fmha_sm100.decode_q8kv4 import BatchDecodeWithPagedKVCacheWrapper, plan_decode

from .cases import KV_HEADS, PAGE_SIZE, SMOKE_CASES, make_inputs
from .reference import PageDequantizer, assert_close_to_reference, sparse_decode_reference
from .runners import run_wrapper

CASE = SMOKE_CASES[1]  # ragged lengths, one query token each


def _positions(inputs):
    case = inputs.case
    return [length - case.q_len + token for length in case.seq_lens for token in range(case.q_len)]


def _variant(inputs, kind: str) -> torch.Tensor:
    """Lists outside the generator's habits but inside the kernel's contract."""
    rng = random.Random(7)
    lists = inputs.topk_indices.clone().cpu()
    rows, heads, width = lists.shape
    local_pages = [position // PAGE_SIZE for position in _positions(inputs)]
    for row in range(rows):
        local = local_pages[row]
        for head in range(heads):
            valid = [page for page in lists[row, head].tolist() if page >= 0]
            history = [page for page in valid if page != local]
            if kind == "unsorted":  # local page anywhere in the prefix
                entries = valid[:]
                rng.shuffle(entries)
            elif kind == "no_local":  # a history page replaces the query's own page
                pool = [page for page in range(local) if page not in valid]
                entries = history + ([rng.choice(pool)] if pool else [])
            elif kind == "short":  # a prefix of five entries
                entries = valid[:5]
            elif kind == "future":  # pages past the query's own page end the prefix
                entries = sorted(valid) + [local + 1, local + 2]
            else:
                raise ValueError(kind)
            entries = entries[:width]
            lists[row, head] = torch.tensor(entries + [-1] * (width - len(entries)), dtype=torch.int32)
    return lists.to(inputs.topk_indices.device)


@pytest.mark.parametrize("kind", ("unsorted", "no_local", "short", "future"))
def test_selection_lists_follow_the_list_not_the_position(device, kind):
    inputs = make_inputs(CASE, device)
    inputs.topk_indices = _variant(inputs, kind)
    reference = sparse_decode_reference(
        inputs, PageDequantizer(inputs.k_codes, inputs.k_scale), PageDequantizer(inputs.v_codes, inputs.v_scale)
    )
    out = run_wrapper(inputs, label=f"contract {kind}")
    assert_close_to_reference(out, reference, label=f"contract {kind}")


def test_plan_rejects_invalid_metadata(device):
    inputs = make_inputs(CASE, device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called"):
        wrapper.run(inputs.q, (inputs.k_codes, inputs.v_codes), kv_cache_sf=(inputs.k_scale, inputs.v_scale))
    with pytest.raises(ValueError, match="8 or 16 Q heads"):
        wrapper.plan(inputs.topk_indices, inputs.page_table, inputs.seq_lens, q_len_per_req=1, num_q_heads=16)
    with pytest.raises(ValueError, match="kv_indptr applies only"):
        wrapper.plan(inputs.topk_indices, inputs.page_table, inputs.seq_lens, q_len_per_req=1,
                     kv_indptr=torch.zeros(5, dtype=torch.int32, device=device))
    with pytest.raises(ValueError, match="requires kv_indptr"):
        wrapper.plan(inputs.topk_indices, inputs.page_table.reshape(-1), inputs.seq_lens, q_len_per_req=1)
    with pytest.raises(ValueError, match="block_scale_shift"):
        wrapper.plan(inputs.topk_indices, inputs.page_table, inputs.seq_lens, q_len_per_req=1,
                     block_scale_shift=8)
    with pytest.raises(ValueError, match="topk"):
        plan_decode(batch_size=4, q_len_per_req=1, topk=65, device=device)
    with pytest.raises(TypeError, match="int32"):
        wrapper.plan(inputs.topk_indices.long(), inputs.page_table, inputs.seq_lens, q_len_per_req=1)


def test_run_rejects_mismatched_tensors(device):
    inputs = make_inputs(CASE, device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    wrapper.plan(inputs.topk_indices, inputs.page_table, inputs.seq_lens, q_len_per_req=1,
                 num_q_heads=inputs.num_q_heads, num_kv_heads=KV_HEADS)
    scales = (inputs.k_scale, inputs.v_scale_kernel)
    with pytest.raises(ValueError, match="contiguously"):
        wide_rows = torch.zeros(inputs.k_codes.numel() * 2, dtype=torch.uint8, device=device)
        wide_rows = wide_rows.as_strided(tuple(inputs.k_codes.shape), (KV_HEADS * PAGE_SIZE * 128, PAGE_SIZE * 128, 128, 1))
        wrapper.run(inputs.q, (wide_rows, inputs.v_codes), kv_cache_sf=scales)
    with pytest.raises(ValueError, match="multiples of 16"):
        odd = torch.zeros(inputs.k_scale.numel() + 8 * KV_HEADS * inputs.k_scale.shape[0], dtype=torch.uint8, device=device)
        odd = odd.as_strided(tuple(inputs.k_scale.shape), (KV_HEADS * 1032, 1032, 8, 1)).view(torch.float8_e4m3fn)
        wrapper.run(inputs.q, (inputs.k_codes, inputs.v_codes), kv_cache_sf=(odd, inputs.v_scale_kernel))
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        wrapper.run(inputs.q.to(torch.bfloat16), (inputs.k_codes, inputs.v_codes), kv_cache_sf=scales)
    with pytest.raises(ValueError, match="one-element torch.float32"):
        wrapper.run(inputs.q, (inputs.k_codes, inputs.v_codes), kv_cache_sf=scales,
                    kv_global_scale=(torch.ones(2, device=device), torch.ones(1, device=device)))
    with pytest.raises(ValueError, match="topk_indices must have shape"):
        wrapper.run(inputs.q, (inputs.k_codes, inputs.v_codes), kv_cache_sf=scales,
                    topk_indices=inputs.topk_indices[:, :, :8].contiguous())
