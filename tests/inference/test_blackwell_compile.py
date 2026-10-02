"""Compile affected kernels for both Blackwell targets without executing them."""

from __future__ import annotations

import importlib
import logging
import time

import cutlass
import cutlass.cute as cute  # noqa: PLR0402
import pytest
import torch
from packaging.version import Version

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a"))
@pytest.mark.parametrize("num_index_heads", (1, 2, 4))
@pytest.mark.parametrize("query_length", range(1, 17))
@pytest.mark.parametrize("input_dtype", (cutlass.Float8E4M3FN, cutlass.BFloat16))
def test_decode_indexer_compiles_for_blackwell(
    arch: str, num_index_heads: int, query_length: int, input_dtype: type
) -> None:
    from inference.msa_v1.indexer.decode.indexer_gemm import DecodeIndexerGemmSm100

    assert Version(cutlass.__version__) >= Version("4.5.2")

    def tensor(dtype, shape, alignment=16):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=alignment,
        )

    kernel = DecodeIndexerGemmSm100(
        num_index_heads=num_index_heads,
        sm_count=148,
        query_columns=((query_length * num_index_heads + 7) // 8) * 8,
        input_dtype=input_dtype,
    )
    args = (
        tensor(input_dtype, (2, query_length * num_index_heads, 128)),
        tensor(input_dtype, (37, 128, 128)),
        tensor(cutlass.Int32, (2, 17), 4),
        tensor(cutlass.Int32, (2,), 4),
        tensor(cutlass.Float32, (num_index_heads, 2 * query_length, 17)),
        tensor(cutlass.Uint8, ((2 + 4 * 148 + 2) * 4,)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    started_at = time.perf_counter()
    compiled = cute.compile(
        kernel, *args, options=f"--enable-tvm-ffi --opt-level 2 --gpu-arch={arch}"
    )
    logger.info("%s indexer compiled in %.3fs", arch, time.perf_counter() - started_at)
    assert compiled is not None


@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a"))
def test_decode_shared_plan_compiles_for_blackwell(arch):
    from inference.msa_v1.indexer.decode.plan_kernel import DecodeIndexerPlanSm100

    def tensor(size):
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (size,), stride_order=(0,), assumed_align=4
        )

    started_at = time.perf_counter()
    compiled = cute.compile(
        DecodeIndexerPlanSm100(),
        tensor(513),
        tensor(513),
        tensor(513 + 4 * 148 + 2),
        tensor(4 * 513 * 16),
        cutlass.Int32(16),
        cutlass.Int32(4),
        cutlass.Int32(8192),
        cutlass.Int32(148),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --opt-level 2 --gpu-arch={arch}",
    )
    logger.info(
        "%s shared plan compiled in %.3fs", arch, time.perf_counter() - started_at
    )
    assert compiled is not None


@pytest.mark.parametrize("capability", ((10, 0), (10, 3)))
@pytest.mark.parametrize("num_index_heads", (1, 2, 4))
def test_q8kv8_prefill_indexer_compiles_for_blackwell(capability, num_index_heads):
    from inference.msa_v1.indexer.prefill.q8kv8.indexer_gemm import (
        PrefillIndexerGemmSm100,
    )

    assert Version(cutlass.__version__) >= Version("4.5.2")

    def tensor(dtype, shape):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=16,
        )

    kernel = PrefillIndexerGemmSm100(
        compute_capability=capability,
        num_persistent_clusters=74,
        num_index_heads=num_index_heads,
    )
    arch = f"sm_{capability[0]}{capability[1]}a"
    started = time.perf_counter()
    compiled = cute.compile(
        kernel,
        tensor(cutlass.Float8E4M3FN, (257 * num_index_heads, 128)),
        tensor(cutlass.Float8E4M3FN, (37, 1, 128, 128)),
        tensor(cutlass.Int32, (2, 17)),
        tensor(cutlass.Float32, (num_index_heads, 257, 17)),
        tensor(cutlass.Int32, (8, 16, 4)),
        tensor(cutlass.Int32, (8,)),
        cute.runtime.make_fake_stream(),
        options=f"--gpu-arch={arch}",
    )
    logger.info(
        "%s H=%d compiled in %.3fs",
        arch,
        num_index_heads,
        time.perf_counter() - started,
    )
    assert compiled is not None


def _compile_factory_for_arch(monkeypatch, module, capability, cache, invoke):
    assert Version(cutlass.__version__) >= Version("4.5.2")
    original_compile = cute.compile
    actual_capability = torch.cuda.get_device_capability()
    arch = f"sm_{capability[0]}{capability[1]}a"
    compiled_kernels = []

    def compile_target(kernel, *args, options):
        assert kernel.use_tmem_load_reduce == (capability == (10, 3))
        if capability != actual_capability:
            # DSL 4.5.2 cannot construct default-argument FFI wrappers without
            # a native JIT engine. Cross compilation checks device code only.
            options = options.replace("--enable-tvm-ffi", "")
        started_at = time.perf_counter()
        result = original_compile(kernel, *args, options=f"{options} --gpu-arch={arch}")
        logger.info(
            "%s %s compiled in %.3fs",
            arch,
            module.__name__,
            time.perf_counter() - started_at,
        )
        compiled_kernels.append(result)
        return result

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    monkeypatch.setattr(module, "compile_or_load", lambda _key, fn, **_kwargs: fn())
    monkeypatch.setattr(cute, "compile", compile_target)
    cache.clear()
    try:
        invoke()
    finally:
        cache.clear()
    assert len(compiled_kernels) == 1


@pytest.mark.parametrize("capability", ((10, 0), (10, 3)))
@pytest.mark.parametrize("tree", (False, True))
@pytest.mark.parametrize("fp16_score", (False, True))
def test_training_indexer_compiles_for_blackwell(
    monkeypatch, capability, tree, fp16_score
):
    module = importlib.import_module(
        f"msa_v1.{'indexer_tree' if tree else 'indexer'}.m3_indexer"
    )
    kwargs = {"use_fp16_score": fp16_score}
    if not tree:
        kwargs["has_fragment_indices"] = True
    _compile_factory_for_arch(
        monkeypatch,
        module,
        capability,
        module._GEMM_COMPILE_CACHE,
        lambda: module._compile_gemm(torch.device("cuda:0"), **kwargs),
    )


@pytest.mark.parametrize("capability", ((10, 0), (10, 3)))
@pytest.mark.parametrize("heads", (1, 2, 4))
def test_bf16_prefill_indexer_compiles_for_blackwell(capability, heads):
    assert Version(cutlass.__version__) >= Version("4.5.2")
    from inference.msa_v1.indexer.prefill.bf16.indexer_gemm import M3IndexerGemmSm100

    def tensor(dtype, shape, order=None, alignment=16):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))) if order is None else order,
            assumed_align=alignment,
        )

    arch = f"sm_{capability[0]}{capability[1]}a"
    kernel = M3IndexerGemmSm100(compute_capability=capability, num_index_heads=heads)
    args = (
        tensor(cutlass.BFloat16, (512 * heads, 128)),
        tensor(cutlass.BFloat16, (128, 128, 64), (1, 0, 2)),
        tensor(cutlass.Int32, (2, 32), alignment=4),
        tensor(cutlass.Float32, (heads, 512, 32)),
        tensor(cutlass.Int32, (3,), alignment=4),
        tensor(cutlass.Int32, (3,), alignment=4),
        tensor(cutlass.Int32, (32,), alignment=4),
        tensor(cutlass.Int32, (32,), alignment=4),
        None,
        cutlass.Int32(32),
        cutlass.Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    started_at = time.perf_counter()
    result = cute.compile(kernel, *args, options=f"--enable-tvm-ffi --gpu-arch={arch}")
    logger.info(
        "%s BF16 indexer heads=%d compiled in %.3fs",
        arch,
        heads,
        time.perf_counter() - started_at,
    )
    assert result is not None
