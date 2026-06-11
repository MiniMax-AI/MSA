# SPDX-License-Identifier: MIT

"""JIT-loaded CUDA/C++ extension for SM12x paged decode split-KV scheduling."""

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
_SRC = os.path.join(_THIS_DIR, "build_decode_schedule.cu")

_EXTRA_CFLAGS = ["-O3"]
_EXTRA_CUDA_CFLAGS_BASE = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "--ptxas-options=-v",
    "--expt-relaxed-constexpr",
]

_ext = None


def _validate_decode_seqused_k(
    seqused_k: torch.Tensor, *, seqlen_q: int, page_size: int, max_seqlen_k: int
) -> None:
    """Reject seqused_k values that hang the kernel or overflow the pad.

    These are device-data-dependent guards enforced at the raw launch
    boundary so that *every* caller — the high-level
    ``prepare_decode_schedule`` wrapper and any direct user of this raw
    entrypoint alike — is protected (the C++ kernel only TORCH_CHECKs
    structural invariants and would otherwise spin on an all-masked row or
    scatter past the worst-case-padded output arrays).

    (0) max(seqused_k) <= max_seqlen_k.  The host sizes the work-tile arrays
        from max_pages_global = ceil(max_seqlen_k / page_size); a longer
        seqused_k produces work_count > pad_work and the wrapper's
        narrow(0, 0, padded_work_count) / kernel scatter run out of bounds.

    (1) seqused_k[b] >= seqlen_q.  The kernel's causal col_limit for the
        first packed q-token is seqlen_k - seqlen_q + 1, which goes <= 0
        when seqlen_k < seqlen_q.  That all-masked row hits a mask-codegen
        path with PTX-undefined shift counts and the kernel hangs.  It is
        also a batched-decode invariant: seqlen_k must include the
        seqlen_q new tokens being emitted.

    (2) seqused_k[b] % page_size in {0, 8, 16, ..., 120}.  The same hang
        fires when the last partial page has < q_tokens_per_group=8 valid
        columns, because the last MMA tile then hits the all-masked row
        case for the trailing q-tokens.
    """

    max_seqlen_k_i = int(max_seqlen_k)
    max_used_k = int(seqused_k.max().item()) if seqused_k.numel() > 0 else 0
    if max_used_k > max_seqlen_k_i:
        raise ValueError(
            f"max_seqlen_k must cover max(seqused_k), got {max_seqlen_k_i} "
            f"for max seqused_k {max_used_k}"
        )
    seqlen_q_i = int(seqlen_q)
    bad_q = seqused_k < seqlen_q_i
    if bool(bad_q.any().item()):
        bad_idx = int(torch.nonzero(bad_q, as_tuple=True)[0][0].item())
        bad_val = int(seqused_k[bad_idx].item())
        raise ValueError(
            f"decode kernel requires seqused_k[b] >= seqlen_q (= {seqlen_q_i}) "
            f"for every batch.  Got seqused_k[{bad_idx}]={bad_val}.  "
            f"This is also a batched-decode invariant: seqlen_k must include "
            f"the seqlen_q new tokens being emitted."
        )
    page_size_i = int(page_size)
    rem = seqused_k % page_size_i
    bad_rem = (rem > 0) & (rem < seqlen_q_i)
    if bool(bad_rem.any().item()):
        bad_idx = int(torch.nonzero(bad_rem, as_tuple=True)[0][0].item())
        bad_val = int(seqused_k[bad_idx].item())
        raise ValueError(
            f"decode kernel requires seqused_k[b] % page_size in "
            f"{{0, {seqlen_q_i}, {seqlen_q_i*2}, ..., {max(page_size_i//seqlen_q_i, 1)*seqlen_q_i}}}.  "
            f"Got seqused_k[{bad_idx}]={bad_val}, last partial page has "
            f"{bad_val % page_size_i} valid columns (< seqlen_q={seqlen_q_i}). "
            f"Round seqused_k up to the next multiple of {seqlen_q_i} OR to "
            f"a multiple of {page_size_i}."
        )


def _cccl_include_flags() -> list[str]:
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cccl = os.path.join(cuda_home, "include", "cccl")
    return [f"-I{cccl}"] if os.path.isdir(cccl) else []


def _load_ext():
    global _ext
    if _ext is None:
        require_sm12x_csrc_arch("fmha_sm12x.decode_schedule")
        _ext = load(
            name=f"sparse_decode_schedule_sm12x_ext{cuda_arch_cache_suffix()}",
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


def build_decode_schedule(
    seqused_k: torch.Tensor,
    *,
    page_size: int,
    seqlen_q: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seqlen_k: int,
    enable_cuda_graph: bool = False,
    max_grid_size: int = 0,
    fixed_split_size: int = -1,
    disable_split_kv: bool = False,
) -> dict[str, object]:
    """Build paged decode schedule arrays on device with the SM12x helper."""

    # Device-data-dependent hang guards enforced here (the lowest common
    # launch boundary) so direct callers can't spin the kernel on an
    # all-masked row.  Skip when the config is non-positive so the C++
    # TORCH_CHECKs surface the clean structural error instead.
    if int(seqlen_q) > 0 and int(page_size) > 0:
        _validate_decode_seqused_k(
            seqused_k, seqlen_q=int(seqlen_q), page_size=int(page_size),
            max_seqlen_k=int(max_seqlen_k),
        )

    raw = _load_ext().build_decode_schedule(
        seqused_k,
        int(page_size),
        int(seqlen_q),
        int(num_qo_heads),
        int(num_kv_heads),
        int(head_dim),
        int(max_seqlen_k),
        bool(enable_cuda_graph),
        int(max_grid_size),
        int(fixed_split_size),
        bool(disable_split_kv),
    )
    pad = int(raw["padded_work_count"])
    for key in (
        "request_indices",
        "qo_tile_indices",
        "kv_tile_indices",
        "block_valid_mask",
    ):
        raw[key] = raw[key].narrow(0, 0, pad)
    return raw


__all__ = ["build_decode_schedule"]
