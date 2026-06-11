# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""SM120/SM121 package facade for MiniMax Sparse Attention.

The upstream MiniMax kernels in ``fmha_sm100`` rely on SM100-only tcgen05/TMEM
MMA and copy operations. SM12x support therefore cannot safely alias the SM100
implementation; it needs separate kernels, following the same split used by the
SGLang fork for SM100 versus SM12x code.

This package currently exposes only architecture/build helpers. Kernel names are
reserved and fail loudly if accessed before a real SM12x implementation exists.
"""

from __future__ import annotations

from minimax_msa.arch import (
    CudaArch,
    cpp_extension_arch_flag,
    cuda_arch_cache_suffix,
    nvcc_arch_define_flags,
    nvcc_gencode_flags,
    selected_cuda_arch,
)

_KERNEL_EXPORTS = frozenset(
    {
        "fmha_sm100",
        "fmha_sm100_plan",
        "fmha_sm12x",
        "fmha_sm12x_plan",
        "sparse_topk_select",
        "sparse_atten_func",
        "sparse_atten_nvfp4_kv_func",
        "sparse_decode_atten_func",
        "SparseDecodePagedAttentionWrapper",
        "fp4_indexer_block_scores",
        "build_k2q_csr",
        "SparseK2qCsrBuilderSm100",
    }
)

__all__ = [
    "CudaArch",
    "cpp_extension_arch_flag",
    "cuda_arch_cache_suffix",
    "nvcc_arch_define_flags",
    "nvcc_gencode_flags",
    "selected_cuda_arch",
]


def __getattr__(name: str):
    if name in _KERNEL_EXPORTS:
        raise AttributeError(
            f"{name!r} is not available from {__name__!r}: the existing "
            "MiniMax kernels are SM100-only tcgen05/TMEM kernels. Add a "
            "separate SM12x implementation instead of aliasing fmha_sm100."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
