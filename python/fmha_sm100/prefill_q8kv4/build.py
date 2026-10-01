# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Offline cache builder for Q8KV4 sparse prefill attention."""

from __future__ import annotations

import argparse
import os

from .jit import FMHA_SM100_PREFILL_Q8KV4_ARCH


def precompile(arch: str | None = None, block_scale_shifts=(0, 3)) -> list[str]:
    """Build the cache entries used by the lazy runtime JIT (one per block-scale shift)."""

    previous_arch = os.environ.get(FMHA_SM100_PREFILL_Q8KV4_ARCH)
    if arch is not None:
        os.environ[FMHA_SM100_PREFILL_Q8KV4_ARCH] = arch
    try:
        from . import jit

        jit._target_arch.cache_clear()
        jit._load_extension_for_arch.cache_clear()
        uris = []
        for shift in block_scale_shifts:
            spec = jit.gen_jit_spec(block_scale_shift=shift)
            spec.build_and_load()
            uris.append(spec.uri)
        return uris
    finally:
        if previous_arch is None:
            os.environ.pop(FMHA_SM100_PREFILL_Q8KV4_ARCH, None)
        else:
            os.environ[FMHA_SM100_PREFILL_Q8KV4_ARCH] = previous_arch
        if "jit" in locals():
            jit._target_arch.cache_clear()


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default=None, help="Target SM, for example 103a")
    parser.add_argument("--block-scale-shifts", default="0,3",
                        help="Comma-separated block-scale shifts to build")
    args = parser.parse_args()
    shifts = tuple(int(value) for value in args.block_scale_shifts.split(","))
    print("\n".join(precompile(args.arch, shifts)))


if __name__ == "__main__":
    _main()


__all__ = ["precompile"]
