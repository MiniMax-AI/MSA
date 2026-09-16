"""Offline cache builder for Q8KV4 decode attention."""

from __future__ import annotations

import argparse
import os


def precompile(arch: str | None = None) -> tuple[str, ...]:
    """Build the same cache entries used by the lazy runtime JIT."""

    previous_arch = os.environ.get("MM_SPARSE_TARGET_ARCH")
    if arch is not None:
        os.environ["MM_SPARSE_TARGET_ARCH"] = arch
    try:
        from . import jit

        jit._target_arch.cache_clear()
        jit._clear_loaded_extensions()
        specs = (
            jit.gen_jit_spec(topk=16, split_kv=False),
            jit.gen_jit_spec(topk=16, split_kv=True),
        )
        for spec in specs:
            spec.build_and_load()
        jit.get_plan_fn()
        jit.get_reduction_module()
        return tuple(spec.uri for spec in specs)
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
    for uri in precompile(args.arch):
        print(uri)


if __name__ == "__main__":
    _main()


__all__ = ["precompile"]
