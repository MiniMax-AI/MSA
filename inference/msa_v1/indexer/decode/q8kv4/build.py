"""Offline cache builder for Q8KV4 decode indexer GEMM."""

from __future__ import annotations

import argparse
import os


def precompile(arch: str | None = None, *, num_index_heads: int = 1) -> str:
    """Build the same cache entry used by the lazy runtime JIT."""

    previous_arch = os.environ.get("MM_SPARSE_TARGET_ARCH")
    if arch is not None:
        os.environ["MM_SPARSE_TARGET_ARCH"] = arch
    try:
        from . import jit

        jit._target_arch.cache_clear()
        jit._load_extension_for_arch.cache_clear()
        spec = jit.gen_jit_spec(num_index_heads=num_index_heads)
        spec.build_and_load()
        return spec.uri
    finally:
        if previous_arch is None:
            os.environ.pop("MM_SPARSE_TARGET_ARCH", None)
        else:
            os.environ["MM_SPARSE_TARGET_ARCH"] = previous_arch
        if "jit" in locals():
            jit._target_arch.cache_clear()


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default=None, help="Target SM, for example 103a")
    parser.add_argument("--num-index-heads", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    print(precompile(args.arch, num_index_heads=args.num_index_heads))


if __name__ == "__main__":
    _main()


__all__ = ["precompile"]
