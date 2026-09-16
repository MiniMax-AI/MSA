"""Correctness and CUDA Graph tests for Q8KV4 paged sparse decode."""

from __future__ import annotations

import math

import pytest
import torch

from inference.msa_v1.attention.decode.q8kv4 import (
    BatchDecodeWithPagedKVCacheWrapper,
)
from tests.inference.msa_v1.attention.decode.q8kv4.plan_inspection import (
    assert_split_count,
)
from tests.inference.msa_v1.attention.decode.q8kv4.runtime import (
    compile_modules,
)
from tests.inference.msa_v1.attention.decode.runtime import run_cuda


def _make_constant_inputs(
    batch: int, q_len_per_req: int, gqa_ratio: int
) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    total_q = batch * q_len_per_req
    topk_indices = torch.full((total_q, 4, 16), -1, dtype=torch.int32, device=device)
    num_pages = math.ceil(q_len_per_req / 128)
    query_in_request = torch.arange(total_q, device=device) % q_len_per_req
    local_pages = torch.div(query_in_request, 128, rounding_mode="floor")
    for logical_page in range(num_pages):
        topk_indices[local_pages >= logical_page, :, logical_page] = logical_page
    page_table = (
        torch.arange(num_pages, dtype=torch.int32, device=device)
        .expand(batch, -1)
        .contiguous()
    )
    seq_lens = torch.full((batch,), q_len_per_req, dtype=torch.int32, device=device)
    q = torch.zeros(
        (total_q, gqa_ratio * 4, 128), dtype=torch.float8_e4m3fn, device=device
    )
    packed_k = torch.zeros((num_pages, 4, 128, 64), dtype=torch.uint8, device=device)
    packed_v = torch.full_like(packed_k, 0x22)
    scale = torch.ones((num_pages, 4, 128, 8), dtype=torch.float8_e4m3fn, device=device)
    return topk_indices, page_table, seq_lens, q, packed_k, packed_v, scale


@pytest.mark.parametrize("q_len_per_req", (1, 3, 8, 17, 127, 128, 129, 257))
@pytest.mark.parametrize("gqa_ratio", (8, 16), ids=("gqa8", "gqa16"))
@pytest.mark.parametrize("num_kv_splits", (1, 4, 8))
def test_arbitrary_query_lengths_and_preallocated_output(
    q_len_per_req: int,
    gqa_ratio: int,
    num_kv_splits: int,
) -> None:
    batch = math.ceil(1024 / q_len_per_req)
    topk, page_table, seq_lens, q, packed_k, packed_v, scale = _make_constant_inputs(
        batch, q_len_per_req, gqa_ratio
    )
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    wrapper.plan(
        topk,
        page_table,
        seq_lens,
        q_len_per_req=q_len_per_req,
        num_q_heads=gqa_ratio * 4,
        num_kv_splits=num_kv_splits,
    )
    out = torch.empty_like(q, dtype=torch.bfloat16)
    compile_modules(gqa_ratio, q.device)
    actual = run_cuda(
        "constant boundary",
        lambda: wrapper.run(
            q, (packed_k, packed_v), kv_cache_sf=(scale, scale), out=out
        ),
    )
    assert_split_count(wrapper, num_kv_splits)
    assert actual.data_ptr() == out.data_ptr()
    assert torch.all(out == 1)


@pytest.mark.parametrize("gqa_ratio", (8, 16), ids=("gqa8", "gqa16"))
def test_cuda_graph_capture_replays_nonempty_work(gqa_ratio: int) -> None:
    topk, page_table, seq_lens, q, packed_k, packed_v, scale = _make_constant_inputs(
        1, 8, gqa_ratio
    )
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    wrapper.plan(
        topk,
        page_table,
        seq_lens,
        q_len_per_req=8,
        num_q_heads=gqa_ratio * 4,
        num_kv_splits=1,
    )
    out = torch.empty_like(q, dtype=torch.bfloat16)
    compile_modules(gqa_ratio, q.device)
    run_cuda(
        "graph warmup",
        lambda: wrapper.run(
            q, (packed_k, packed_v), kv_cache_sf=(scale, scale), out=out
        ),
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(
            q,
            (packed_k, packed_v),
            kv_cache_sf=(scale, scale),
            out=out,
        )
    out.fill_(-777)
    run_cuda("graph replay", graph.replay)
    assert torch.all(out == 1)
