"""CUDA Graph capture, replay, and metadata replan tests."""

from __future__ import annotations

import pytest
import torch

from inference.msa_v1.indexer.decode.q8kv4.interface import (
    BatchDecodeIndexerWithPagedKVCacheWrapper,
    _BatchDecodeProxyScoreWrapper,
)
from tests.inference.msa_v1.indexer.decode.q8kv4.reference import (
    indexer_gemm_reference,
    make_inputs,
)


def _make_graph_wrapper(
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> _BatchDecodeProxyScoreWrapper:
    return _BatchDecodeProxyScoreWrapper(
        use_cuda_graph=True,
        page_table_buffer=torch.empty_like(page_table),
        seq_lens_buffer=torch.empty_like(seq_lens),
    )


@pytest.mark.parametrize("batch", (32, 129, 1025, 4097))
def test_cuda_graph_replay_with_device_lengths(batch: int) -> None:
    device = torch.device("cuda")
    max_pages = 4
    lengths = torch.arange(batch, dtype=torch.int32) % (max_pages * 128 - 7) + 8
    q, packed_k, k_scale, page_table, seq_lens = make_inputs(
        batch, max_pages, lengths, seed=7, device=device
    )
    output = torch.full(
        (1, batch * 8, max_pages), 123.0, dtype=torch.float32, device=device
    )
    wrapper = _make_graph_wrapper(page_table, seq_lens)
    wrapper.plan(page_table, seq_lens)
    wrapper.run(q, packed_k, k_scale, out=output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, packed_k, k_scale, out=output)
    graph.replay()
    torch.cuda.synchronize()
    first = output.clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, first, atol=0, rtol=0)


@pytest.mark.parametrize("batch", (32, 129))
def test_replan_between_graph_replays(batch: int) -> None:
    device = torch.device("cuda")
    max_pages = 4
    first_lengths = torch.full((batch,), 256, dtype=torch.int32)
    second_lengths = torch.arange(batch, dtype=torch.int32) % 377 + 128
    q, packed_k, k_scale, first_page_table, first_seq_lens = make_inputs(
        batch, max_pages, first_lengths, seed=31, device=device
    )
    _, _, _, second_page_table, second_seq_lens = make_inputs(
        batch, max_pages, second_lengths, seed=32, device=device
    )
    output = torch.full(
        (1, batch * 8, max_pages), 123.0, dtype=torch.float32, device=device
    )
    wrapper = _make_graph_wrapper(first_page_table, first_seq_lens)
    wrapper.plan(first_page_table, first_seq_lens)
    wrapper.run(q, packed_k, k_scale, out=output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, packed_k, k_scale, out=output)

    wrapper.plan(second_page_table, second_seq_lens)
    output.fill_(123.0)
    graph.replay()
    torch.cuda.synchronize()
    expected = indexer_gemm_reference(
        q, packed_k, k_scale, second_page_table, second_seq_lens
    )
    pages = torch.arange(max_pages, device=device).reshape(1, 1, -1)
    query_positions = (
        second_seq_lens.reshape(-1, 1)
        - 8
        + torch.arange(8, dtype=torch.int32, device=device).reshape(1, 8)
    )
    local_pages = torch.div(query_positions, 128, rounding_mode="floor").reshape(
        1, -1, 1
    )
    torch.testing.assert_close(
        output.masked_select(pages < local_pages),
        expected.masked_select(pages < local_pages),
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.all(output.masked_select(pages >= local_pages) == 123.0)


def test_capture_requires_preallocated_output() -> None:
    device = torch.device("cuda")
    q, packed_k, k_scale, page_table, seq_lens = make_inputs(
        1,
        2,
        torch.tensor([256], dtype=torch.int32),
        seed=41,
        device=device,
    )
    wrapper = _make_graph_wrapper(page_table, seq_lens)
    wrapper.plan(page_table, seq_lens)
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="requires a preallocated out"),
        torch.cuda.graph(graph),
    ):
        wrapper.run(q, packed_k, k_scale)


def test_plan_is_rejected_during_capture() -> None:
    device = torch.device("cuda")
    q, packed_k, k_scale, page_table, seq_lens = make_inputs(
        1,
        2,
        torch.tensor([256], dtype=torch.int32),
        seed=42,
        device=device,
    )
    output = torch.empty((1, 8, 2), dtype=torch.float32, device=device)
    wrapper = _make_graph_wrapper(page_table, seq_lens)
    wrapper.plan(page_table, seq_lens)
    wrapper.run(q, packed_k, k_scale, out=output)
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match=r"plan\(\) must be called outside"),
        torch.cuda.graph(graph),
    ):
        wrapper.plan(page_table, seq_lens)


@pytest.mark.parametrize("page_capacity", (4, 40))
def test_public_wrapper_cuda_graph_replays_topk(page_capacity: int) -> None:
    device = torch.device("cuda")
    q, packed_k, k_scale, page_table, seq_lens = make_inputs(
        32,
        page_capacity,
        (torch.arange(32, dtype=torch.int32) * 137) % (page_capacity * 128 - 128) + 128,
        seed=71,
        device=device,
    )
    wrapper = BatchDecodeIndexerWithPagedKVCacheWrapper(
        use_cuda_graph=True,
        page_table_buffer=torch.empty_like(page_table),
        seq_lens_buffer=torch.empty_like(seq_lens),
    )
    wrapper.plan(page_table, seq_lens)
    output = torch.empty((1, 32 * 8, 16), dtype=torch.int32, device=device)
    wrapper.run(q, packed_k, k_scale=k_scale, out=output)
    torch.cuda.synchronize()
    expected = output.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, packed_k, k_scale=k_scale, out=output)
    output.fill_(-777)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, expected)

    # Independent wrappers and graphs must not alias mutable score/TopK buffers.
    other = BatchDecodeIndexerWithPagedKVCacheWrapper()
    other.plan(page_table, seq_lens)
    other_q = q.flip(0).contiguous()
    other_output = torch.empty_like(output)
    other.run(other_q, packed_k, k_scale=k_scale, out=other_output)
    torch.cuda.synchronize()
    other_expected = other_output.clone()
    other_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(other_graph):
        other.run(other_q, packed_k, k_scale=k_scale, out=other_output)
    torch.cuda.synchronize()
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    for _ in range(3):
        with torch.cuda.stream(streams[0]):
            graph.replay()
        with torch.cuda.stream(streams[1]):
            other_graph.replay()
    for stream in streams:
        stream.synchronize()
    assert torch.equal(output, expected)
    assert torch.equal(other_output, other_expected)
