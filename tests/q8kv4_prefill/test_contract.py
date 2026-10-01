# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Selection-list contract and interface validation of the Q8KV4 prefill wrapper."""

import pytest
import torch

from fmha_sm100.prefill_q8kv4 import BatchPrefillWithPagedKVCacheWrapper

from .cases import PAGE_SIZE, SMOKE_CASES, global_scale, make_inputs, query_positions
from .reference import assert_close_to_reference, sparse_prefill_reference
from .runners import plan_wrapper, run_wrapper

CASE = SMOKE_CASES[0]


def _lists_with(inputs, kind: str) -> torch.Tensor:
    """Lists outside the generator's habits but inside the kernel's contract."""
    lists = inputs.topk_indices.clone()
    heads, rows, width = lists.shape
    local_pages = [position // PAGE_SIZE for position in query_positions(inputs.case)]
    for row, local in enumerate(local_pages):
        for head in range(heads):
            valid = [page for page in lists[head, row].tolist() if page >= 0]
            if kind == "future":  # the request's next page is selected but not yet visible
                entries = valid + [local + 1]
            elif kind == "empty":  # nothing visible: zero output, -inf LSE
                entries = [local + 1] if row % 2 else []
            elif kind == "local_first":
                entries = [local] + [page for page in valid if page != local]
            else:
                raise ValueError(kind)
            entries = entries[:width]
            lists[head, row] = torch.tensor(entries + [-1] * (width - len(entries)), dtype=torch.int32)
    return lists


@pytest.mark.parametrize("kind", ("future", "empty", "local_first"))
def test_selection_lists_follow_the_list(device, kind):
    inputs = make_inputs(CASE, device)
    inputs.topk_indices = _lists_with(inputs, kind)
    out, lse = run_wrapper(inputs, label=f"contract {kind}")
    reference, reference_lse = sparse_prefill_reference(inputs)
    assert_close_to_reference(out, reference, label=f"contract {kind}", lse=lse, reference_lse=reference_lse)
    if kind == "empty":
        assert not out.any() and torch.isinf(lse).all()


def test_plan_rejects_invalid_metadata(device):
    inputs = make_inputs(CASE, device)
    with pytest.raises(RuntimeError, match="before run"):
        BatchPrefillWithPagedKVCacheWrapper().run(inputs.q, (inputs.k_codes, inputs.v_codes),
                                                  kv_cache_sf=(inputs.k_scale, inputs.v_scale_kernel))
    with pytest.raises(NotImplementedError, match="causal"):
        plan_wrapper(inputs, causal=False)
    with pytest.raises(ValueError, match="topk"):
        plan_wrapper(inputs, topk_indices=inputs.topk_indices[..., :12].contiguous())
    with pytest.raises(ValueError, match="kv_indptr"):
        plan_wrapper(inputs, flat=True, kv_indptr=None)
    with pytest.raises(TypeError, match="int32"):
        wrapper = BatchPrefillWithPagedKVCacheWrapper()
        wrapper.plan(inputs.topk_indices, inputs.cu_seqlens_q, inputs.cu_seqlens_k,
                     inputs.page_table.long(), total_k=sum(CASE.k_lens), total_rows=inputs.total_rows,
                     max_seqlen_q=max(CASE.q_lens), max_seqlen_k=max(CASE.k_lens))
    with pytest.raises(ValueError, match="block_scale_shift"):
        plan_wrapper(inputs, block_scale_shift=8)
    with pytest.raises(ValueError, match="seqused_k"):
        plan_wrapper(inputs, seqused_k=torch.zeros(3, dtype=torch.int32, device=device))


def test_run_rejects_mismatched_tensors(device):
    inputs = make_inputs(CASE, device)
    wrapper = plan_wrapper(inputs)
    kv, sf = (inputs.k_codes, inputs.v_codes), (inputs.k_scale, inputs.v_scale_kernel)
    with pytest.raises(TypeError, match="float8_e4m3fn"):
        wrapper.run(inputs.q.to(torch.bfloat16), kv, kv_cache_sf=sf)
    with pytest.raises(ValueError, match="shape"):
        wrapper.run(inputs.q[:, :32].contiguous(), kv, kv_cache_sf=sf)
    with pytest.raises(ValueError, match="contiguous"):  # token rows must stay contiguous
        wrapper.run(inputs.q, (inputs.k_codes.transpose(2, 3).contiguous().transpose(2, 3), inputs.v_codes),
                    kv_cache_sf=sf)
    padded = torch.zeros((*inputs.k_codes.shape[:3], 72), dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="16-byte"):  # a head stride of 72 * 128 + 8 bytes
        misaligned = torch.empty(padded.numel() + 8 * padded.shape[0] * padded.shape[1], dtype=torch.uint8,
                                 device=device)
        view = misaligned.as_strided(inputs.k_codes.shape, (padded.shape[1] * (9216 + 8), 9216 + 8, 64, 1))
        wrapper.run(inputs.q, (view, inputs.v_codes), kv_cache_sf=sf)
    with pytest.raises(TypeError, match="float32"):
        wrapper.run(inputs.q, kv, kv_cache_sf=sf,
                    kv_global_scale=(global_scale(1.0, device).double(), None))
