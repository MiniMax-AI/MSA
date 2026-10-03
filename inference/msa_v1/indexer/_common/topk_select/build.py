"""Lazy JIT builder for the internal SM100-family indexer TopK."""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from pathlib import Path

from torch.utils import cpp_extension

from inference.msa_v1._build_utils import cuda_home

_LOGGER = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parent
_CSRC = _ROOT / "csrc"
_CACHE_ABI = "msa_v1_indexer_topk_decode_pdl_v3"


@lru_cache(maxsize=1)
def load_extension():
    """Build and load the internal TopK extension once per process."""

    selected_cuda_home = cuda_home()
    build_root = Path(
        os.environ.get(
            "MINIMAX_MSA_V1_INDEXER_TOPK_BUILD_DIR",
            Path.home() / ".cache" / "minimax_msa" / _CACHE_ABI,
        )
    ).expanduser()
    build_root.mkdir(parents=True, exist_ok=True)
    sources = [
        _CSRC / "api/indexer_topk_api.cpp",
        _CSRC / "api/indexer_topk_pybind.cpp",
        _CSRC / "kernel/indexer_topk.cu",
    ]
    include_paths = [
        str(_CSRC / "api"),
        str(_CSRC / "kernel"),
        str(_CSRC / "kernel/m3"),
    ]
    started_at = time.time()
    previous_cuda_home = cpp_extension.CUDA_HOME
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    cpp_extension.CUDA_HOME = str(selected_cuda_home)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a;10.3a"
    try:
        extension = cpp_extension.load(
            name="minimax_msa_v1_indexer_topk",
            sources=[str(path) for path in sources],
            extra_include_paths=include_paths,
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++20",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-Xptxas=-v",
            ],
            extra_ldflags=[
                "-lcuda",
                f"-Wl,-rpath,{selected_cuda_home / 'lib64'}",
            ],
            build_directory=str(build_root),
            verbose=(
                os.environ.get("MINIMAX_MSA_V1_INDEXER_TOPK_VERBOSE_BUILD") == "1"
            ),
        )
    finally:
        cpp_extension.CUDA_HOME = previous_cuda_home
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list
    _LOGGER.info(
        "Compiled msa_v1.indexer._common.topk_select in %.1fs",
        time.time() - started_at,
    )
    return extension
