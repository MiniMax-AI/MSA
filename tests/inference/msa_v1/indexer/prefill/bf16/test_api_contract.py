"""Public contract tests for the BF16 paged-prefill indexer."""

import inspect

import pytest
import torch

from inference.msa_v1.indexer.prefill import bf16 as bf16_package
from inference.msa_v1.indexer.prefill.bf16 import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)

pytestmark = pytest.mark.gpu


def test_wrapper_is_the_only_public_entrypoint() -> None:
    assert bf16_package.__all__ == ["BatchPrefillIndexerWithPagedKVCacheWrapper"]
    wrapper_type = BatchPrefillIndexerWithPagedKVCacheWrapper
    assert tuple(inspect.signature(wrapper_type.run).parameters) == (
        "self",
        "q",
        "paged_k_cache",
        "out",
    )


@pytest.mark.parametrize("num_index_heads", (1, 4))
def test_output_is_always_head_major_3d(num_index_heads: int) -> None:
    cu_seqlens = torch.tensor((0, 1), dtype=torch.int32, device="cuda")
    page_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    q = torch.zeros(
        (1, num_index_heads, 128),
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_cache = torch.zeros((1, 1, 128, 128), dtype=torch.bfloat16, device="cuda")
    wrapper = BatchPrefillIndexerWithPagedKVCacheWrapper()
    wrapper.plan(
        cu_seqlens,
        cu_seqlens,
        page_table,
        total_q=1,
        max_seqlen_q=1,
        max_seqlen_k=1,
        num_index_heads=num_index_heads,
    )
    out = wrapper.run(q, k_cache)
    torch.cuda.synchronize()
    assert out.shape == (num_index_heads, 1, 16)
    assert torch.equal(out[:, :, 0], torch.zeros_like(out[:, :, 0]))
    assert torch.all(out[:, :, 1:] == -1)


def test_run_requires_plan() -> None:
    wrapper = BatchPrefillIndexerWithPagedKVCacheWrapper()
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called"):
        wrapper.run(torch.empty(0), torch.empty(0))


def test_run_rejects_misaligned_q() -> None:
    cu_seqlens = torch.tensor((0, 1), dtype=torch.int32, device="cuda")
    page_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    k_cache = torch.zeros((1, 1, 128, 128), dtype=torch.bfloat16, device="cuda")
    wrapper = BatchPrefillIndexerWithPagedKVCacheWrapper()
    wrapper.plan(
        cu_seqlens,
        cu_seqlens,
        page_table,
        total_q=1,
        max_seqlen_q=1,
        max_seqlen_k=1,
        num_index_heads=1,
    )
    storage = torch.empty(129, dtype=torch.bfloat16, device="cuda")
    misaligned_q = storage[1:].view(1, 1, 128)
    assert misaligned_q.is_contiguous()
    with pytest.raises(ValueError, match="16-byte aligned"):
        wrapper.run(misaligned_q, k_cache)
