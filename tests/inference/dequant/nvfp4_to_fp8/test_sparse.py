"""Correctness tests for selected-page paged NVFP4 conversion."""

from __future__ import annotations

import pytest
import torch

from inference.dequant import SparsePagedNvfp4ToFp8Wrapper
from tests.inference.dequant.nvfp4_to_fp8.reference import dequantize_reference
from tests.inference.msa_v1.decode.metadata import make_decode_topk


pytestmark = pytest.mark.gpu


def _require_sm100_family() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
        pytest.skip("NVFP4 dequant requires SM100 or SM103")
    return device


def _make_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    seq_lens = torch.tensor([385, 257], dtype=torch.int32, device=device)
    q_len_per_req = 2
    topk = make_decode_topk(
        seq_lens,
        q_len_per_req=q_len_per_req,
        seed=1701,
    )[:, :, :4].contiguous()
    page_table = torch.tensor(
        [[4, 1, 3, 0], [4, 5, 3, 2]],
        dtype=torch.int32,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(1701)
    packed_shape = (6, 4, 128, 64)
    scale_shape = (6, 4, 128, 8)
    packed_k = torch.randint(
        0,
        256,
        packed_shape,
        dtype=torch.uint8,
        generator=generator,
        device=device,
    )
    packed_v = torch.randint(
        0,
        256,
        packed_shape,
        dtype=torch.uint8,
        generator=generator,
        device=device,
    )
    k_scale = torch.randint(
        0,
        127,
        scale_shape,
        dtype=torch.uint8,
        generator=generator,
        device=device,
    ).view(torch.float8_e4m3fn)
    v_scale = torch.randint(
        0,
        127,
        scale_shape,
        dtype=torch.uint8,
        generator=generator,
        device=device,
    ).view(torch.float8_e4m3fn)
    return (
        topk,
        page_table,
        seq_lens,
        packed_k,
        packed_v,
        k_scale,
        v_scale,
    )


def test_selected_physical_page_head_pairs_are_bitwise_exact() -> None:
    device = _require_sm100_family()
    topk, page_table, seq_lens, packed_k, packed_v, k_scale, v_scale = _make_inputs(
        device
    )
    wrapper = SparsePagedNvfp4ToFp8Wrapper()
    wrapper.plan(topk, page_table, seq_lens, q_len_per_req=2)
    result = wrapper.run(
        (packed_k, packed_v),
        kv_cache_sf=(k_scale, v_scale),
    )
    torch.cuda.synchronize()

    positions = seq_lens.to(torch.int64).repeat_interleave(2)
    positions += torch.tensor([[-2, -1]], device=device).expand(2, -1).reshape(-1)
    valid_count = (positions // 128 + 1).clamp(max=topk.shape[2])
    for query in range(topk.shape[0]):
        batch = query // 2
        for head in range(topk.shape[1]):
            for slot in range(int(valid_count[query])):
                logical_page = int(topk[query, head, slot])
                physical_page = int(page_table[batch, logical_page])
                compact_page = int(result.block_tables[head, query, slot])
                expected_k = dequantize_reference(
                    packed_k[physical_page, head],
                    k_scale[physical_page, head],
                )
                expected_v = dequantize_reference(
                    packed_v[physical_page, head],
                    v_scale[physical_page, head],
                )
                assert torch.equal(
                    result.k_cache[compact_page, head].view(torch.uint8),
                    expected_k.view(torch.uint8),
                )
                assert torch.equal(
                    result.v_cache[compact_page, head].view(torch.uint8),
                    expected_v.view(torch.uint8),
                )

    expected_lens = ((valid_count - 1) * 128 + positions.remainder(128) + 1).to(
        torch.int32
    )
    assert torch.equal(result.seq_lens, expected_lens.expand(4, -1))
    assert result.max_seq_len == 4 * 128


def test_sparse_run_is_cuda_graph_deterministic() -> None:
    device = _require_sm100_family()
    topk, page_table, seq_lens, packed_k, packed_v, k_scale, v_scale = _make_inputs(
        device
    )
    wrapper = SparsePagedNvfp4ToFp8Wrapper()
    wrapper.plan(topk, page_table, seq_lens, q_len_per_req=2)
    eager = wrapper.run(
        (packed_k, packed_v),
        kv_cache_sf=(k_scale, v_scale),
    )
    torch.cuda.synchronize()
    valid = topk.permute(1, 0, 2) >= 0
    heads = (
        torch.arange(4, device=device).reshape(4, 1, 1).expand_as(eager.block_tables)
    )
    expected_k = eager.k_cache[eager.block_tables[valid], heads[valid]].clone()
    expected_v = eager.v_cache[eager.block_tables[valid], heads[valid]].clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(
            (packed_k, packed_v),
            kv_cache_sf=(k_scale, v_scale),
        )
    graph.replay()
    torch.cuda.synchronize()
    actual_k = captured.k_cache[captured.block_tables[valid], heads[valid]]
    actual_v = captured.v_cache[captured.block_tables[valid], heads[valid]]
    assert torch.equal(actual_k.view(torch.uint8), expected_k.view(torch.uint8))
    assert torch.equal(actual_v.view(torch.uint8), expected_v.view(torch.uint8))


def test_plan_rejects_non_prefix_and_wrong_local_page() -> None:
    device = _require_sm100_family()
    topk, page_table, seq_lens, *_ = _make_inputs(device)
    wrapper = SparsePagedNvfp4ToFp8Wrapper()

    non_prefix = topk.clone()
    non_prefix[0, 0, 0] = -1
    with pytest.raises(ValueError, match="prefix"):
        wrapper.plan(non_prefix, page_table, seq_lens, q_len_per_req=2)

    wrong_local = topk.clone()
    wrong_local[0, 0, 2] = 0
    with pytest.raises(ValueError, match="local page"):
        wrapper.plan(wrong_local, page_table, seq_lens, q_len_per_req=2)
