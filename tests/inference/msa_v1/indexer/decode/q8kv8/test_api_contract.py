"""Public API validation tests."""

from __future__ import annotations

import inspect

import pytest
import torch

from inference.msa_v1.indexer.decode import q8kv8 as indexer
from tests.inference.msa_v1.indexer.decode.q8kv8.reference import make_inputs

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("capability", ((10, 0), (10, 3), (9, 0), (12, 0)))
def test_plan_checks_device_architecture(monkeypatch, capability) -> None:
    device = torch.device("cuda")
    page_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    seq_lens = torch.full((1,), 128, dtype=torch.int32, device=device)
    wrapper = indexer.BatchDecodeIndexerWithPagedKVCacheWrapper()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    if capability in ((10, 0), (10, 3)):
        wrapper.plan(page_table, seq_lens)
    else:
        with pytest.raises(ValueError, match="SM100 or SM103"):
            wrapper.plan(page_table, seq_lens)


def test_public_entrypoints_match_direct_e4m3_contract() -> None:
    assert indexer.__all__ == ["BatchDecodeIndexerWithPagedKVCacheWrapper"]
    assert not hasattr(indexer, "forward")
    assert not hasattr(indexer, "BatchDecodeIndexerGemmWrapper")
    wrapper_type = indexer.BatchDecodeIndexerWithPagedKVCacheWrapper
    assert tuple(inspect.signature(wrapper_type.plan).parameters) == (
        "self",
        "page_table",
        "seq_lens",
        "num_index_heads",
        "query_length",
        "shared_plan",
    )
    assert tuple(inspect.signature(wrapper_type.run).parameters) == (
        "self",
        "q",
        "paged_k_cache",
        "out",
    )


def test_preallocated_output_and_shape() -> None:
    device = torch.device("cuda")
    q, k_cache, page_table, seq_lens = make_inputs(
        2,
        3,
        torch.tensor([129, 256], dtype=torch.int32),
        seed=1,
        device=device,
    )
    output = torch.full((1, 16, 16), -777, dtype=torch.int32, device=device)
    wrapper = indexer.BatchDecodeIndexerWithPagedKVCacheWrapper()
    wrapper.plan(page_table, seq_lens)
    result = wrapper.run(q, k_cache, out=output)
    torch.cuda.synchronize()
    assert result.data_ptr() == output.data_ptr()
    assert result.shape == (1, 16, 16)
    query_positions = (
        seq_lens.cpu().reshape(-1, 1)
        - 8
        + torch.arange(8, dtype=torch.int32).reshape(1, 8)
    )
    num_valid_pages = (
        torch.div(
            query_positions,
            128,
            rounding_mode="floor",
        )
        .add_(1)
        .reshape(-1)
    )
    slots = torch.arange(16, dtype=torch.int32)
    expected = torch.where(
        slots[None, :] < num_valid_pages[:, None],
        slots[None, :],
        -1,
    )
    assert torch.equal(output.cpu(), expected.unsqueeze(0))


def test_out_none_allocates_output() -> None:
    device = torch.device("cuda")
    q, k_cache, page_table, seq_lens = make_inputs(
        2,
        3,
        torch.tensor([129, 256], dtype=torch.int32),
        seed=2,
        device=device,
    )
    wrapper = indexer.BatchDecodeIndexerWithPagedKVCacheWrapper()
    wrapper.plan(page_table, seq_lens)
    result = wrapper.run(q, k_cache)
    assert result.shape == (1, 16, 16)
    assert result.dtype == torch.int32
    assert result.is_cuda
    assert result.device.index == torch.cuda.current_device()


def test_user_workspace_buffer_is_reused() -> None:
    device = torch.device("cuda")
    q, k_cache, page_table, seq_lens = make_inputs(
        2,
        3,
        torch.tensor([129, 256], dtype=torch.int32),
        seed=3,
        device=device,
    )
    workspace = torch.empty(
        indexer.BatchDecodeIndexerWithPagedKVCacheWrapper.workspace_size(2),
        dtype=torch.uint8,
        device=device,
    )
    wrapper = indexer.BatchDecodeIndexerWithPagedKVCacheWrapper(workspace)
    wrapper.plan(page_table, seq_lens)
    result = wrapper.run(q, k_cache)
    torch.cuda.synchronize()
    assert result.shape == (1, 16, 16)
    assert wrapper._proxy_score._workspace_buffer.data_ptr() == workspace.data_ptr()


def test_run_requires_plan() -> None:
    device = torch.device("cuda")
    q, k_cache, _, _ = make_inputs(
        1,
        1,
        torch.tensor([128], dtype=torch.int32),
        seed=4,
        device=device,
    )
    wrapper = indexer.BatchDecodeIndexerWithPagedKVCacheWrapper()
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called"):
        wrapper.run(q, k_cache)


def test_rejects_invalid_dtype() -> None:
    device = torch.device("cuda")
    q, k_cache, page_table, seq_lens = make_inputs(
        1,
        1,
        torch.tensor([128], dtype=torch.int32),
        seed=5,
        device=device,
    )
    wrapper = indexer.BatchDecodeIndexerWithPagedKVCacheWrapper()
    wrapper.plan(page_table, seq_lens)
    with pytest.raises(RuntimeError, match="q must have dtype"):
        wrapper.run(q.float(), k_cache)
    with pytest.raises(RuntimeError, match="k_cache must have dtype"):
        wrapper.run(q, k_cache.float())
    with pytest.raises(ValueError, match="seq_lens must have dtype"):
        wrapper.plan(page_table, seq_lens.to(torch.int64))


def test_cuda_graph_mode_requires_fixed_metadata_buffers() -> None:
    with pytest.raises(ValueError, match="requires page_table_buffer"):
        indexer.BatchDecodeIndexerWithPagedKVCacheWrapper(use_cuda_graph=True)
