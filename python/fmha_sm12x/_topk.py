# SPDX-License-Identifier: MIT

"""SM12x JIT loader for the ``sparse_topk_select`` csrc kernel.

The kernel source is arch-agnostic (no tcgen05/TMEM) and ships inside the
``fmha_sm100`` package.  SM12x reuses that source read-only but compiles it for
the SM120/SM121 target via :mod:`minimax_msa.arch`, with its own cache dir, so
``fmha_sm100`` itself needs no SM12x-specific code and its SM100 build is
unaffected.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import threading
from pathlib import Path

from minimax_msa.arch import (
    cuda_arch_cache_suffix,
    nvcc_gencode_flags,
    require_sm12x_csrc_arch,
)


def _fmha_sm100_dir() -> Path:
    spec = importlib.util.find_spec("fmha_sm100")
    if spec is None or spec.origin is None:
        raise RuntimeError("fmha_sm100 package not found; cannot locate sparse_topk_select.cu")
    return Path(spec.origin).resolve().parent


def _cache_base() -> Path:
    explicit = os.environ.get("MINFER_FMHA_CACHE_DIR")
    base = Path(explicit) if explicit else Path(os.path.expanduser("~/.cache/minfer/fmha_sm12x"))
    suffix = cuda_arch_cache_suffix()
    return base.parent / (base.name + suffix) if suffix else base


def _tvm_ffi_include() -> str:
    import tvm_ffi

    tvm_dir = Path(tvm_ffi.__path__[0])
    for inc in (tvm_dir / "include", tvm_dir.parent / "include"):
        if inc.exists():
            return str(inc)
    raise RuntimeError("Cannot find TVM-FFI include directory; install apache-tvm-ffi")


def _cuda_home() -> str:
    if "CUDA_HOME" in os.environ:
        return os.environ["CUDA_HOME"]
    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).resolve().parent.parent)
    for p in ("/usr/local/cuda", "/opt/cuda"):
        if os.path.isdir(p):
            return p
    raise RuntimeError("Cannot find CUDA toolkit. Set CUDA_HOME.")


def _nvcc_flags(cache_dir: Path, csrc: Path, cutlass: Path) -> str:
    flags = [
        "-O3", "-std=c++20",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        *nvcc_gencode_flags(),
        "-static-global-template-stub=false",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
        "-DFLASHINFER_ENABLE_FP8_E8M0",
        "-DFLASHINFER_ENABLE_FP4_E2M1",
        "-DFLASHINFER_ENABLE_F16",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-Xcudafe", "--diag_suppress=2908",
        f"-I{csrc / 'include'}",
        f"-I{cutlass / 'include'}",
        f"-I{cutlass / 'tools' / 'util' / 'include'}",
        f"-I{_tvm_ffi_include()}",
        f"-I{cache_dir}",
        "-use_fast_math",
        "-DNDEBUG", "-Xptxas", "-O3",
        "-Xcompiler", "-fPIC",
    ]
    return " ".join(flags)


_module = None
_lock = threading.Lock()


def _compile() -> Path:
    require_sm12x_csrc_arch("fmha_sm12x.sparse_topk")
    pkg = _fmha_sm100_dir()
    csrc = pkg / "csrc"
    cutlass = pkg / "cutlass"
    cache_dir = _cache_base() / "sparse_topk"
    so_path = cache_dir / "sparse_topk_select.so"
    if so_path.exists():
        return so_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csrc / "sparse_topk_select.cu", cache_dir / "sparse_topk_select.cu")
    # tvm_ffi_utils.h is included by relative name; csrc/ is not on the include
    # path, so stage it inside cache_dir (which is) like the source build does.
    header = csrc / "tvm_ffi_utils.h"
    dst = cache_dir / "tvm_ffi_utils.h"
    if not dst.exists() or dst.read_text() != header.read_text():
        shutil.copy2(header, dst)

    nvcc = os.path.join(_cuda_home(), "bin", "nvcc")
    obj = cache_dir / "sparse_topk_select.o"
    ninja_content = f"""ninja_required_version = 1.5

nvcc = {nvcc}
nvcc_flags = {_nvcc_flags(cache_dir, csrc, cutlass)}

rule nvcc_compile
  command = $nvcc $nvcc_flags -c $in -o $out
  description = Compiling $in

rule nvcc_link
  command = $nvcc -shared $in -o $out -lcuda
  description = Linking $out

build {obj}: nvcc_compile {cache_dir / "sparse_topk_select.cu"}
build {so_path}: nvcc_link {obj}
"""
    (cache_dir / "build.ninja").write_text(ninja_content)
    result = subprocess.run(["ninja", "-j1"], cwd=str(cache_dir), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"sparse_topk_select compilation failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return so_path


def get_sparse_topk_module():
    """Compile (once) and load the SM12x ``sparse_topk_select`` module."""

    global _module
    if _module is not None:
        return _module
    with _lock:
        if _module is not None:
            return _module
        so_path = _compile()
        import tvm_ffi

        _module = tvm_ffi.load_module(str(so_path))
        return _module


__all__ = ["get_sparse_topk_module"]
