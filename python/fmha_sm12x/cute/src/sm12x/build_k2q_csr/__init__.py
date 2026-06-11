# SPDX-License-Identifier: MIT

"""JIT-loaded CUDA C++ extension for the SM12x q2k -> k2q CSR builder."""

from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load

from minimax_msa.arch import (
    cpp_extension_arch_flag,
    cuda_arch_cache_suffix,
    require_sm12x_csrc_arch,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_THIS_DIR, "build_k2q_csr.cu")

_EXTRA_CFLAGS = ["-O3"]
_EXTRA_CUDA_CFLAGS_BASE = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "--ptxas-options=-v",
    "--expt-relaxed-constexpr",
]

_ext = None


def _cccl_include_flags() -> list[str]:
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cccl = os.path.join(cuda_home, "include", "cccl")
    return [f"-I{cccl}"] if os.path.isdir(cccl) else []


def _load_ext():
    global _ext
    if _ext is None:
        require_sm12x_csrc_arch("fmha_sm12x.k2q_csr")
        _ext = load(
            name=f"sparse_build_k2q_csr_sm12x_ext{cuda_arch_cache_suffix()}",
            sources=[_SRC],
            extra_cflags=_EXTRA_CFLAGS,
            extra_cuda_cflags=[
                *_EXTRA_CUDA_CFLAGS_BASE,
                cpp_extension_arch_flag(),
                *_cccl_include_flags(),
            ],
            verbose=False,
        )
    return _ext


def run_build_k2q_csr(
    q2k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    row_ptr: torch.Tensor,
    q_idx: torch.Tensor,
    topk: int,
    blk_kv: int,
    total_rows: int,
    max_kv_blocks: int,
) -> None:
    """In-place fill of ``row_ptr`` and ``q_idx`` using the SM12x CUDA helper."""

    _load_ext().run_build_k2q_csr(
        q2k,
        cu_seqlens_q,
        cu_seqlens_k,
        row_ptr,
        q_idx,
        int(topk),
        int(blk_kv),
        int(total_rows),
        int(max_kv_blocks),
    )


def run_build_k2q_csr_with_schedule(
    q2k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    row_ptr: torch.Tensor,
    q_idx: torch.Tensor,
    scheduler_metadata: torch.Tensor,
    work_count: torch.Tensor,
    qsplit_idx: torch.Tensor,
    split_counts: torch.Tensor,
    topk: int,
    blk_kv: int,
    total_rows: int,
    max_kv_blocks: int,
    target_q_per_cta: int,
    work_capacity: int,
    max_seqlen_q: int,
) -> None:
    """In-place fill of CSR plus sparse-attention schedule metadata."""

    _load_ext().run_build_k2q_csr_with_schedule(
        q2k,
        cu_seqlens_q,
        cu_seqlens_k,
        row_ptr,
        q_idx,
        scheduler_metadata,
        work_count,
        qsplit_idx,
        split_counts,
        int(topk),
        int(blk_kv),
        int(total_rows),
        int(max_kv_blocks),
        int(target_q_per_cta),
        int(work_capacity),
        int(max_seqlen_q),
    )


def is_supported(topk: int, blk_kv: int) -> bool:
    return int(topk) in (4, 8, 16, 32) and int(blk_kv) == 128


__all__ = ["run_build_k2q_csr", "run_build_k2q_csr_with_schedule", "is_supported"]
