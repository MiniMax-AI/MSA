# SPDX-License-Identifier: MIT

"""SM12x-facing architecture helpers."""

from minimax_msa.arch import (  # noqa: F401
    CudaArch,
    cpp_extension_arch_flag,
    cuda_arch_cache_suffix,
    nvcc_gencode_flags,
    require_sm12x_csrc_arch,
    selected_cuda_arch,
)

__all__ = [
    "CudaArch",
    "cpp_extension_arch_flag",
    "cuda_arch_cache_suffix",
    "nvcc_gencode_flags",
    "require_sm12x_csrc_arch",
    "selected_cuda_arch",
]
