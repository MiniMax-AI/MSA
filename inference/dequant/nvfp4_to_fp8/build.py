"""Offline cache builder for reusable NVFP4-to-E4M3 conversion."""

from __future__ import annotations

import argparse
import os


def precompile(arch: str | None = None) -> str:
    previous_arch = os.environ.get("MM_SPARSE_TARGET_ARCH")
    if arch is not None:
        os.environ["MM_SPARSE_TARGET_ARCH"] = arch
    try:
        from . import jit

        jit._target_arch.cache_clear()
        jit._clear_extension_cache()
        spec = jit.gen_jit_spec()
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
    args = parser.parse_args()
    print(precompile(args.arch))


if __name__ == "__main__":
    _main()


__all__ = ["precompile"]
