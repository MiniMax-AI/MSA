# SPDX-License-Identifier: MIT

"""SM120/SM121 package facade for MiniMax Sparse Attention."""

from __future__ import annotations

from minimax_msa.arch import (
    CudaArch,
    cpp_extension_arch_flag,
    cuda_arch_cache_suffix,
    nvcc_gencode_flags,
    selected_cuda_arch,
)

_API_EXPORTS = frozenset(
    {
        "Sm12xPlan",
        "fmha_sm12x",
        "fmha_sm12x_plan",
        "sparse_topk_select",
    }
)
_SPARSE_EXPORTS = frozenset(
    {
        "Nvfp4QuantizedTensor",
        "SparseDecodePagedAttentionWrapper",
        "SparseK2qCsrBuilderSm12x",
        "build_k2q_csr",
        "dequantize_nvfp4_128x4_to_bf16",
        "fp4_indexer_block_scores",
        "nvfp4_global_scale_from_amax",
        "nvfp4_scale_128x4_offset",
        "quantize_bf16_to_nvfp4_128x4",
        "quantize_kv_bf16_to_nvfp4_128x4",
        "sparse_atten_func",
        "sparse_atten_nvfp4_kv_func",
        "sparse_decode_atten_func",
        "swizzle_nvfp4_scale_to_128x4",
    }
)

__all__ = [
    "CudaArch",
    "Nvfp4QuantizedTensor",
    "Sm12xPlan",
    "SparseDecodePagedAttentionWrapper",
    "SparseK2qCsrBuilderSm12x",
    "build_k2q_csr",
    "cpp_extension_arch_flag",
    "cuda_arch_cache_suffix",
    "dequantize_nvfp4_128x4_to_bf16",
    "fmha_sm12x",
    "fmha_sm12x_plan",
    "fp4_indexer_block_scores",
    "nvcc_gencode_flags",
    "nvfp4_global_scale_from_amax",
    "nvfp4_scale_128x4_offset",
    "quantize_bf16_to_nvfp4_128x4",
    "quantize_kv_bf16_to_nvfp4_128x4",
    "selected_cuda_arch",
    "sparse_atten_func",
    "sparse_atten_nvfp4_kv_func",
    "sparse_decode_atten_func",
    "sparse_topk_select",
    "swizzle_nvfp4_scale_to_128x4",
]


def __getattr__(name: str):
    if name in _API_EXPORTS:
        from . import api as _api

        return getattr(_api, name)
    if name in _SPARSE_EXPORTS:
        from . import sparse as _sparse

        return getattr(_sparse, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
