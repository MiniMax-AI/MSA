"""Public API validation tests."""

from __future__ import annotations

import inspect

import pytest
import torch

from inference.msa_v1.indexer.decode import tp4_q8kv4 as indexer
from inference.msa_v1.indexer.decode.tp4_q8kv4 import jit
from tests.inference.msa_v1.indexer.decode.tp4_q8kv4.reference import make_inputs

pytestmark = pytest.mark.gpu


def test_wrapper_is_the_only_public_entrypoint() -> None:
    assert indexer.__all__ == ["BatchDecodeIndexerWithPagedKVCacheWrapper"]
    assert not hasattr(indexer, "forward")
    assert not hasattr(indexer, "BatchDecodeIndexerGemmWrapper")
    wrapper_type = indexer.BatchDecodeIndexerWithPagedKVCacheWrapper
    assert tuple(inspect.signature(wrapper_type.plan).parameters) == (
        "self",
        "page_table",
        "seq_lens",
    )
    assert tuple(inspect.signature(wrapper_type.run).parameters) == (
        "self",
        "q",
        "paged_k_cache",
        "k_scale",
        "out",
    )


def test_preallocated_output_and_shape() -> None:
    device = torch.device("cuda")
    seq_lens_cpu = torch.tensor([129, 256], dtype=torch.int32)
    q, packed_k, k_scale, page_table, seq_lens = make_inputs(
        2, 3, seq_lens_cpu, seed=1, device=device
    )
    output = torch.full((16, 16), -777, dtype=torch.int32, device=device)
    wrapper = indexer.BatchDecodeIndexerWithPagedKVCacheWrapper()
    wrapper.plan(page_table, seq_lens)
    result = wrapper.run(q, packed_k, k_scale=k_scale, out=output)
    torch.cuda.synchronize()
    assert result.data_ptr() == output.data_ptr()
    assert result.shape == (16, 16)
    query_positions = seq_lens_cpu.reshape(-1, 1) - 8 + torch.arange(
        8, dtype=torch.int32
    ).reshape(1, 8)
    num_valid_pages = torch.div(
        query_positions,
        128,
        rounding_mode="floor",
    ).add_(1).reshape(-1)
    slots = torch.arange(16, dtype=torch.int32)
    expected = torch.where(
        slots[None, :] < num_valid_pages[:, None],
        slots[None, :],
        -1,
    )
    assert torch.equal(output.cpu(), expected)


def test_out_none_allocates_only_output() -> None:
    device = torch.device("cuda")
    inputs = make_inputs(
        2,
        3,
        torch.tensor([129, 256], dtype=torch.int32),
        seed=2,
        device=device,
    )
    q, packed_k, k_scale, page_table, seq_lens = inputs
    wrapper = indexer.BatchDecodeIndexerWithPagedKVCacheWrapper()
    wrapper.plan(page_table, seq_lens)
    result = wrapper.run(q, packed_k, k_scale=k_scale)
    assert result.shape == (16, 16)
    assert result.dtype == torch.int32
    assert result.is_cuda
    assert result.device.index == torch.cuda.current_device()


def test_external_workspace_and_size_validation() -> None:
    device = torch.device("cuda")
    inputs = make_inputs(
        129,
        2,
        torch.full((129,), 256, dtype=torch.int32),
        seed=3,
        device=device,
    )
    _, _, _, page_table, seq_lens = inputs
    required_bytes = (
        indexer.BatchDecodeIndexerWithPagedKVCacheWrapper.workspace_size(129)
    )
    workspace = torch.empty(required_bytes, dtype=torch.uint8, device=device)
    wrapper = indexer.BatchDecodeIndexerWithPagedKVCacheWrapper(workspace)
    wrapper.plan(page_table, seq_lens)

    too_small = torch.empty(required_bytes - 1, dtype=torch.uint8, device=device)
    wrapper = indexer.BatchDecodeIndexerWithPagedKVCacheWrapper(too_small)
    with pytest.raises(ValueError, match="workspace_buffer is too small"):
        wrapper.plan(page_table, seq_lens)


def test_run_requires_plan() -> None:
    device = torch.device("cuda")
    q, packed_k, k_scale, _, _ = make_inputs(
        1,
        1,
        torch.tensor([128], dtype=torch.int32),
        seed=4,
        device=device,
    )
    wrapper = indexer.BatchDecodeIndexerWithPagedKVCacheWrapper()
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called"):
        wrapper.run(q, packed_k, k_scale=k_scale)


def test_rejects_invalid_dtype() -> None:
    device = torch.device("cuda")
    q, packed_k, k_scale, page_table, seq_lens = make_inputs(
        1,
        1,
        torch.tensor([128], dtype=torch.int32),
        seed=5,
        device=device,
    )
    wrapper = indexer.BatchDecodeIndexerWithPagedKVCacheWrapper()
    wrapper.plan(page_table, seq_lens)
    with pytest.raises(RuntimeError, match="q must have dtype"):
        wrapper.run(q.float(), packed_k, k_scale=k_scale)

    with pytest.raises(ValueError, match="seq_lens must have dtype"):
        wrapper.plan(page_table, seq_lens.to(torch.int64))


def test_cuda_graph_mode_requires_fixed_metadata_buffers() -> None:
    with pytest.raises(ValueError, match="requires page_table_buffer"):
        indexer.BatchDecodeIndexerWithPagedKVCacheWrapper(use_cuda_graph=True)


def test_jit_spec_contains_only_compile_time_configuration() -> None:
    spec = jit.gen_jit_spec()

    assert tuple(spec.__dataclass_fields__) == (
        "dequant_mode",
        "target_arch",
        "variant_name",
        "query_length",
        "head_dim",
        "page_tokens",
        "scale_group_size",
        "pages_per_work_tile",
        "maximum_pages",
    )
    assert spec.variant_name == "indexer_gemm_tp4_q8kv4"
    assert spec.dequant_mode in {"qmul4", "fp16_fallback"}
    assert spec.query_length == 8
    assert spec.maximum_pages == 8192


@pytest.mark.parametrize(
    ("supports_qmul4", "expected_mode"),
    ((True, "qmul4"), (False, "fp16_fallback")),
)
def test_jit_selects_dequant_mode_from_compiler_capability(
    monkeypatch: pytest.MonkeyPatch,
    supports_qmul4: bool,
    expected_mode: str,
) -> None:
    monkeypatch.setattr(jit, "_target_arch", lambda device=None: "103a")
    monkeypatch.setattr(jit, "_supports_qmul4", lambda arch: supports_qmul4)
    jit._dequant_mode.cache_clear()
    try:
        spec = jit.gen_jit_spec()
        assert spec.dequant_mode == expected_mode
        assert f"_{expected_mode}_103a_" in spec.uri
    finally:
        jit._dequant_mode.cache_clear()


def test_lazy_jit_uses_tensor_device_over_offline_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:5")
    monkeypatch.setenv("MM_SPARSE_TARGET_ARCH", "100a")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda requested: (10, 3) if requested == device else (10, 0),
    )
    jit._target_arch.cache_clear()
    try:
        assert jit._target_arch(device) == "103a"
    finally:
        jit._target_arch.cache_clear()
