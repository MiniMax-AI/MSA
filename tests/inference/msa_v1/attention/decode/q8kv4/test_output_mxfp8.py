"""MXFP8 output must match BF16 attention followed by FlashInfer quantization."""

from __future__ import annotations

import math
import os
import random
import time

import pytest
import torch

from inference.msa_v1.attention.decode.q8kv4 import BatchDecodeWithPagedKVCacheWrapper
from inference.msa_v1.attention.decode.q8kv4 import jit
from tests.inference.msa_v1.attention.decode.runtime import run_cuda


def _inputs(q_len: int, kv_len: int, gqa: int):
    pages = math.ceil(kv_len / 128)
    topk = torch.full((q_len, 4, 16), -1, dtype=torch.int32, device="cuda")
    selected = list(range(min(15, pages - 1))) + [pages - 1]
    topk[:, :, : len(selected)] = torch.tensor(selected, dtype=torch.int32, device="cuda")
    table = torch.arange(pages, dtype=torch.int32, device="cuda").unsqueeze(0)
    lengths = torch.tensor([kv_len], dtype=torch.int32, device="cuda")
    q = (torch.randn(q_len, 4 * gqa, 128, device="cuda") * 0.125).to(
        torch.float8_e4m3fn
    )
    k = torch.randint(0, 256, (pages, 4, 128, 64), dtype=torch.uint8, device="cuda")
    v = torch.randint(0, 256, (pages, 4, 128, 64), dtype=torch.uint8, device="cuda")
    sf = torch.ones((pages, 4, 128, 8), dtype=torch.float8_e4m3fn, device="cuda")
    return topk, table, lengths, q, k, v, sf


def _valid_scale_indices(rows: int, heads: int) -> torch.Tensor:
    offsets = [
        (row // 128) * heads * 512 + head * 512 +
        (row % 32) * 16 + ((row % 128) // 32) * 4 + col
        for row in range(rows)
        for head in range(heads)
        for col in range(4)
    ]
    return torch.tensor(offsets, dtype=torch.int64, device="cuda")


@pytest.mark.parametrize("gqa", (8, 16), ids=("gqa8", "gqa16"))
@pytest.mark.parametrize("q_len", (1, 8))
@pytest.mark.parametrize("kv_len", (2048, 2051, 8192))
def test_mxfp8_matches_flashinfer(gqa: int, q_len: int, kv_len: int) -> None:
    flashinfer = pytest.importorskip("flashinfer")
    torch.manual_seed(2026 + kv_len + q_len + gqa)
    topk, table, lengths, q, k, v, sf = _inputs(q_len, kv_len, gqa)
    bf16 = BatchDecodeWithPagedKVCacheWrapper()
    bf16.plan(topk, table, lengths, q_len_per_req=q_len, num_q_heads=4 * gqa)
    fused = BatchDecodeWithPagedKVCacheWrapper()
    fused.plan(
        topk, table, lengths, q_len_per_req=q_len, num_q_heads=4 * gqa,
        output_mode="mxfp8",
    )
    for split in (False, True):
        jit.get_fmha_fwd_variant(gqa_ratio=gqa, split_kv=split, output_mode="mxfp8")
    out_bf16 = run_cuda(
        "BF16 output", lambda: bf16.run(q, (k, v), kv_cache_sf=(sf, sf))
    )
    expected, expected_sf = flashinfer.mxfp8_quantize(
        out_bf16.view(q_len, -1), is_sf_swizzled_layout=True,
        sf_swizzle_layout=flashinfer.SfLayout.layout_128x4,
        backend="cute-dsl",
    )
    actual, actual_sf = run_cuda(
        "MXFP8 output", lambda: fused.run(q, (k, v), kv_cache_sf=(sf, sf))
    )
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    valid = _valid_scale_indices(q_len, 4 * gqa)
    assert torch.equal(actual_sf[valid], expected_sf[valid])


def test_legacy_multi_split_rejected() -> None:
    topk, table, lengths, *_ = _inputs(8, 2048, 16)
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    with pytest.raises(ValueError, match="legacy multi-split"):
        wrapper.plan(
            topk, table, lengths, q_len_per_req=8, num_q_heads=64,
            num_kv_splits=4, output_mode="mxfp8",
        )


@pytest.mark.parametrize("gqa", (8, 16))
@pytest.mark.parametrize("q_len,kv_len", ((1, 2048), (8, 1)))
def test_mxfp8_explicit_single_split_and_empty_segment(
    gqa: int, q_len: int, kv_len: int
) -> None:
    flashinfer = pytest.importorskip("flashinfer")
    topk, table, lengths, q, k, v, sf = _inputs(q_len, kv_len, gqa)
    if kv_len < q_len:
        topk[:-kv_len].fill_(-1)
    bf16 = BatchDecodeWithPagedKVCacheWrapper()
    bf16.plan(topk, table, lengths, q_len_per_req=q_len,
              num_q_heads=4 * gqa, num_kv_splits=1)
    fused = BatchDecodeWithPagedKVCacheWrapper()
    fused.plan(topk, table, lengths, q_len_per_req=q_len,
               num_q_heads=4 * gqa, num_kv_splits=1, output_mode="mxfp8")
    out_bf16 = run_cuda("single-split BF16", lambda: bf16.run(
        q, (k, v), kv_cache_sf=(sf, sf)))
    expected, expected_sf = flashinfer.mxfp8_quantize(
        out_bf16.view(q_len, -1), is_sf_swizzled_layout=True,
        sf_swizzle_layout=flashinfer.SfLayout.layout_128x4,
        backend="cute-dsl",
    )
    actual, actual_sf = run_cuda("single-split MXFP8", lambda: fused.run(
        q, (k, v), kv_cache_sf=(sf, sf)))
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    valid = _valid_scale_indices(q_len, 4 * gqa)
    assert torch.equal(actual_sf[valid], expected_sf[valid])


def test_mxfp8_graph_reuses_merge_counters() -> None:
    topk, table, lengths, q, k, v, sf = _inputs(8, 32768, 16)
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    wrapper.plan(topk, table, lengths, q_len_per_req=8, num_q_heads=64,
                 output_mode="mxfp8")
    jit.get_fmha_fwd_variant(gqa_ratio=16, split_kv=True, output_mode="mxfp8")
    data, scales = run_cuda(
        "MXFP8 graph warmup", lambda: wrapper.run(q, (k, v), kv_cache_sf=(sf, sf))
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, (k, v), kv_cache_sf=(sf, sf), out=data, out_scale=scales)
    reference = data.view(torch.uint8).clone()
    reference_scales = scales[_valid_scale_indices(8, 64)].clone()
    for _ in range(32):
        run_cuda("MXFP8 graph replay", graph.replay)
        assert torch.equal(data.view(torch.uint8), reference)
        assert torch.equal(scales[_valid_scale_indices(8, 64)], reference_scales)


def test_mxfp8_two_way_streamk_merge() -> None:
    flashinfer = pytest.importorskip("flashinfer")
    topk, table, lengths, q, k, v, sf = _inputs(8, 8192, 16)
    bf16 = BatchDecodeWithPagedKVCacheWrapper()
    bf16.plan(topk, table, lengths, q_len_per_req=8, num_q_heads=64,
              usable_sm_count=64)
    fused = BatchDecodeWithPagedKVCacheWrapper()
    fused.plan(topk, table, lengths, q_len_per_req=8, num_q_heads=64,
               usable_sm_count=64, output_mode="mxfp8")
    out_bf16 = run_cuda("two-way BF16 merge", lambda: bf16.run(
        q, (k, v), kv_cache_sf=(sf, sf)))
    expected, expected_sf = flashinfer.mxfp8_quantize(
        out_bf16.view(8, -1), is_sf_swizzled_layout=True,
        sf_swizzle_layout=flashinfer.SfLayout.layout_128x4,
        backend="cute-dsl",
    )
    actual, actual_sf = run_cuda("two-way MXFP8 merge", lambda: fused.run(
        q, (k, v), kv_cache_sf=(sf, sf)))
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    valid = _valid_scale_indices(8, 64)
    assert torch.equal(actual_sf[valid], expected_sf[valid])


@pytest.mark.skipif(
    os.environ.get("MSA_MXFP8_STRESS") != "1", reason="requires explicit stress run"
)
def test_mxfp8_random_2048_and_graph_deadlock() -> None:
    flashinfer = pytest.importorskip("flashinfer")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    shard = int(os.environ.get("MSA_MXFP8_STRESS_SHARD", "0"))
    shards = int(os.environ.get("MSA_MXFP8_STRESS_SHARDS", "1"))
    torch.cuda.set_device(local_rank)
    rng = random.Random(2026)
    max_pages = 1048576 // 128
    table = torch.arange(max_pages, dtype=torch.int32, device="cuda").unsqueeze(0)
    k = torch.randint(0, 256, (max_pages, 4, 128, 64), dtype=torch.uint8, device="cuda")
    v = torch.randint(0, 256, (max_pages, 4, 128, 64), dtype=torch.uint8, device="cuda")
    sf = torch.ones((max_pages, 4, 128, 8), dtype=torch.float8_e4m3fn, device="cuda")
    boundaries = (8, 127, 128, 129, 2047, 2048, 2049, 32767, 32768,
                  32769, 131071, 131072, 131073, 1048575, 1048576)
    for case in range(shard, 2048, shards):
        kv_len = boundaries[case] if case < len(boundaries) else rng.randint(8, 1048576)
        q_len = 1 + case % 8
        gqa = 8 if (case // 8) % 2 == 0 else 16
        topk = torch.full((q_len, 4, 16), -1, dtype=torch.int32, device="cuda")
        for qidx in range(q_len):
            visible = kv_len - q_len + qidx + 1
            if visible <= 0:
                continue
            pages = math.ceil(visible / 128)
            count = min(15, pages - 1)
            selected = rng.sample(range(pages - 1), count) + [pages - 1]
            topk[qidx, :, :len(selected)] = torch.tensor(
                selected, dtype=torch.int32, device="cuda"
            )
        lengths = torch.tensor([kv_len], dtype=torch.int32, device="cuda")
        q = (torch.randn(q_len, 4 * gqa, 128, device="cuda") * 0.125).to(
            torch.float8_e4m3fn
        )
        bf16 = BatchDecodeWithPagedKVCacheWrapper()
        bf16.plan(topk, table, lengths, q_len_per_req=q_len, num_q_heads=4 * gqa)
        fused = BatchDecodeWithPagedKVCacheWrapper()
        fused.plan(topk, table, lengths, q_len_per_req=q_len,
                   num_q_heads=4 * gqa, output_mode="mxfp8")
        base = bf16.run(q, (k, v), kv_cache_sf=(sf, sf))
        expected, expected_sf = flashinfer.mxfp8_quantize(
            base.view(q_len, -1), is_sf_swizzled_layout=True,
            sf_swizzle_layout=flashinfer.SfLayout.layout_128x4,
            backend="cute-dsl",
        )
        actual, actual_sf = fused.run(q, (k, v), kv_cache_sf=(sf, sf))
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), case
        valid = _valid_scale_indices(q_len, 4 * gqa)
        assert torch.equal(actual_sf[valid], expected_sf[valid]), case

    # Reuse the last plan, output pointers, and Stream-K counters for at least ten seconds.
    jit.get_fmha_fwd_variant(gqa_ratio=gqa, split_kv=True, output_mode="mxfp8")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused.run(q, (k, v), kv_cache_sf=(sf, sf), out=actual, out_scale=actual_sf)
    start = time.monotonic()
    while time.monotonic() - start < 10:
        run_cuda("MXFP8 stress graph replay", graph.replay)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
