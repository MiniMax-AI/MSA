# SPDX-License-Identifier: MIT

"""Tests for runtime CUDA architecture flag selection."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from minimax_msa import arch


def test_default_arch_flags_when_no_cuda_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default builds preserve the original SM100/SM103 csrc targets."""

    monkeypatch.delenv("MSA_CUDA_ARCH", raising=False)
    monkeypatch.delenv("FMHA_SM100_CUDA_ARCH", raising=False)
    monkeypatch.delenv("MSA_NVCC_GENCODES", raising=False)
    monkeypatch.delenv("FMHA_SM100_NVCC_GENCODES", raising=False)
    monkeypatch.setattr(arch, "_detect_device_arch", lambda: None)

    assert arch.nvcc_gencode_flags() == [
        "-gencode=arch=compute_100a,code=sm_100a",
        "-gencode=arch=compute_103a,code=sm_103a",
    ]
    assert arch.cpp_extension_arch_flag() == "-arch=sm_100"
    assert arch.cuda_arch_cache_suffix() == ""


def test_explicit_sm121_arch_selects_single_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """MSA_CUDA_ARCH=sm_121 selects SM121 flags and cache names."""

    monkeypatch.setenv("MSA_CUDA_ARCH", "sm_121")
    monkeypatch.delenv("FMHA_SM100_CUDA_ARCH", raising=False)
    monkeypatch.delenv("MSA_NVCC_GENCODES", raising=False)
    monkeypatch.delenv("FMHA_SM100_NVCC_GENCODES", raising=False)

    assert arch.nvcc_gencode_flags() == ["-gencode=arch=compute_121,code=sm_121"]
    assert arch.cpp_extension_arch_flag() == "-arch=sm_121"
    assert arch.cuda_arch_cache_suffix() == "_sm121"



def test_sm12x_topk_loader_targets_sm121(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The SM12x sparse_topk loader compiles the shared kernel for SM121.

    fmha_sm100 is untouched (its csrc still targets SM100/SM103); the SM12x
    arch routing lives entirely in fmha_sm12x._topk.
    """

    monkeypatch.setenv("MSA_CUDA_ARCH", "sm_121")
    monkeypatch.delenv("FMHA_SM100_CUDA_ARCH", raising=False)
    monkeypatch.delenv("MSA_NVCC_GENCODES", raising=False)
    monkeypatch.delenv("FMHA_SM100_NVCC_GENCODES", raising=False)

    from fmha_sm12x import _topk

    flags = _topk._nvcc_flags(tmp_path, tmp_path, tmp_path)
    assert "-gencode=arch=compute_121,code=sm_121" in flags
    assert "-gencode=arch=compute_100a,code=sm_100a" not in flags


def test_fmha_sm100_jit_is_arch_agnostic_source() -> None:
    """fmha_sm100 carries no SM12x/arch-routing coupling (zero-diff PR goal)."""

    jit_src = (
        Path(__file__).resolve().parents[1] / "python/fmha_sm100/jit.py"
    ).read_text()
    assert "minimax_msa" not in jit_src
    assert "-gencode=arch=compute_100a,code=sm_100a" in jit_src


def test_sm12x_csrc_guard_accepts_sm121(monkeypatch: pytest.MonkeyPatch) -> None:
    """SM12x helper kernels accept SM121 targets."""

    monkeypatch.setenv("MSA_CUDA_ARCH", "sm_121")
    monkeypatch.delenv("FMHA_SM100_CUDA_ARCH", raising=False)
    monkeypatch.delenv("MSA_NVCC_GENCODES", raising=False)
    monkeypatch.delenv("FMHA_SM100_NVCC_GENCODES", raising=False)

    arch.require_sm12x_csrc_arch("fmha_sm12x.k2q_csr")


def test_sm12x_csrc_guard_rejects_default_sm100(monkeypatch: pytest.MonkeyPatch) -> None:
    """SM12x helper kernels reject default SM100-only gencodes."""

    monkeypatch.delenv("MSA_CUDA_ARCH", raising=False)
    monkeypatch.delenv("FMHA_SM100_CUDA_ARCH", raising=False)
    monkeypatch.delenv("MSA_NVCC_GENCODES", raising=False)
    monkeypatch.delenv("FMHA_SM100_NVCC_GENCODES", raising=False)
    monkeypatch.setattr(arch, "_detect_device_arch", lambda: None)

    with pytest.raises(arch.UnsupportedCudaArchError, match="SM120/SM121"):
        arch.require_sm12x_csrc_arch("fmha_sm12x.k2q_csr")


def test_explicit_gencodes_override_single_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    """MSA_NVCC_GENCODES overrides the generated gencode list."""

    gencodes = "-gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_121,code=sm_121"
    monkeypatch.setenv("MSA_CUDA_ARCH", "sm_100a")
    monkeypatch.setenv("MSA_NVCC_GENCODES", gencodes)
    monkeypatch.delenv("FMHA_SM100_CUDA_ARCH", raising=False)
    monkeypatch.delenv("FMHA_SM100_NVCC_GENCODES", raising=False)

    assert arch.nvcc_gencode_flags() == [
        "-gencode=arch=compute_120,code=sm_120",
        "-gencode=arch=compute_121,code=sm_121",
    ]
    assert "compute_120" in arch.cuda_arch_cache_suffix()


def test_invalid_explicit_arch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid arch names fail before invoking nvcc."""

    monkeypatch.setenv("MSA_CUDA_ARCH", "blackwell")
    monkeypatch.delenv("FMHA_SM100_CUDA_ARCH", raising=False)
    monkeypatch.delenv("MSA_NVCC_GENCODES", raising=False)
    monkeypatch.delenv("FMHA_SM100_NVCC_GENCODES", raising=False)

    with pytest.raises(ValueError, match="MSA_CUDA_ARCH"):
        arch.nvcc_gencode_flags()


def test_sm12x_facade_resolves_parallel_kernel_routes() -> None:
    """The SM12x namespace resolves SM12x routes without aliasing SM100 sparse."""

    import fmha_sm12x

    assert "fmha_sm12x" in fmha_sm12x.__all__
    assert "sparse_topk_select" in fmha_sm12x.__all__
    assert "Nvfp4QuantizedTensor" in fmha_sm12x.__all__
    assert callable(fmha_sm12x.dequantize_nvfp4_128x4_to_bf16)
    assert callable(fmha_sm12x.nvfp4_scale_128x4_offset)
    assert callable(fmha_sm12x.fp4_indexer_block_scores)
    assert callable(fmha_sm12x.sparse_atten_nvfp4_kv_func)
    assert callable(fmha_sm12x.sparse_decode_atten_func)
    assert callable(fmha_sm12x.SparseDecodePagedAttentionWrapper)


def test_sm12x_arch_facade_imports_without_heavy_cuda_deps() -> None:
    """The SM12x namespace exposes the shared arch helper as a light import."""

    import fmha_sm12x.arch as sm12x_arch

    assert sm12x_arch.nvcc_gencode_flags is arch.nvcc_gencode_flags
