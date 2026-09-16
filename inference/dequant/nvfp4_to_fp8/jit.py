"""JIT build support for reusable NVFP4-to-E4M3 conversion."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from torch.utils import cpp_extension

from inference.msa_v1._build_utils import cuda_home as _cuda_home
from inference.msa_v1._build_utils import cutlass_root
from inference.msa_v1._build_utils import require_cuda_version
from inference.msa_v1._build_utils import target_arch as _resolve_target_arch
from inference.msa_v1._build_utils import torch_cuda_arch as _resolve_torch_arch

logger = logging.getLogger(__name__)


_ROOT = Path(__file__).resolve().parent
_CSRC = _ROOT / "csrc"
_API = _CSRC / "api"
_INCLUDE = _CSRC / "include"
_SRC = _CSRC / "src"


@lru_cache(maxsize=1)
def _cuda_version() -> tuple[int, int]:
    return require_cuda_version((12, 9), component="NVFP4 dequant")


def _compiled_backend_name() -> str:
    return "qmul4" if _cuda_version() >= (13, 4) else "fp16_fallback"


@lru_cache(maxsize=None)
def _target_arch(device=None) -> str:
    return _resolve_target_arch(device, component="NVFP4 dequant")


def _torch_arch(arch: str) -> str:
    return _resolve_torch_arch(arch, component="NVFP4 dequant")


@lru_cache(maxsize=1)
def _source_digest() -> str:
    import torch

    digest = hashlib.sha256()
    for path in sorted(_CSRC.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(_ROOT)).encode())
            digest.update(path.read_bytes())
    digest.update(Path(__file__).read_bytes())
    digest.update(sys.version.encode())
    digest.update(torch.__version__.encode())
    digest.update(str(torch._C._GLIBCXX_USE_CXX11_ABI).encode())
    digest.update(str(_cuda_home()).encode())
    digest.update(str(_cuda_version()).encode())
    return digest.hexdigest()[:16]


def _cache_root() -> Path:
    root = Path(
        os.environ.get("TORCH_EXTENSIONS_DIR", Path.home() / ".cache/torch_extensions")
    ).expanduser()
    return root / "minimax_msa/dequant_nvfp4_to_fp8"


@dataclass(frozen=True)
class JitSpec:
    """The single architecture- and CUDA-version-selected build."""

    target_arch: str
    variant_name: str = "dequant_nvfp4_to_fp8"

    @property
    def uri(self) -> str:
        version = _cuda_version()
        return (
            f"{self.variant_name}_{self.target_arch}_cuda{version[0]}{version[1]}_"
            f"{_source_digest()}"
        )

    def build_and_load(self):
        cache_dir = _cache_root() / self.uri
        cache_dir.mkdir(parents=True, exist_ok=True)
        module_name = f"minimax_msa_{self.uri}"
        started_at = time.time()
        previous_cuda_home = cpp_extension.CUDA_HOME
        previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
        cpp_extension.CUDA_HOME = str(_cuda_home())
        os.environ["TORCH_CUDA_ARCH_LIST"] = _torch_arch(self.target_arch)
        try:
            extension = cpp_extension.load(
                name=module_name,
                sources=[
                    str(_API / "dequant_api.cpp"),
                    str(_API / "dequant_binding.cpp"),
                    str(_SRC / "dequant.cu"),
                ],
                extra_include_paths=[
                    str(_API),
                    str(_INCLUDE),
                    str(cutlass_root() / "include"),
                ],
                extra_cflags=["-O3", "-DNDEBUG", "-std=c++20"],
                extra_cuda_cflags=[
                    "-O3",
                    "-DNDEBUG",
                    "-lineinfo",
                    "-std=c++20",
                    "-Xptxas=-O3",
                ],
                extra_ldflags=["-lcuda", f"-Wl,-rpath,{_cuda_home() / 'lib64'}"],
                build_directory=str(cache_dir),
                verbose=False,
                with_cuda=True,
            )
        finally:
            cpp_extension.CUDA_HOME = previous_cuda_home
            if previous_arch_list is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list
        logger.info(
            "Compiled inference.dequant.nvfp4_to_fp8 backend=%s in %.1fs",
            _compiled_backend_name(),
            time.time() - started_at,
        )
        return extension


def gen_jit_spec(device=None) -> JitSpec:
    return JitSpec(target_arch=_target_arch(device))


_loaded_extensions = {}
_extension_lock = threading.Lock()


def extension_is_loaded(device=None) -> bool:
    return _target_arch(device) in _loaded_extensions


def load_extension(device=None):
    arch = _target_arch(device)
    if arch in _loaded_extensions:
        return _loaded_extensions[arch]
    with _extension_lock:
        if arch not in _loaded_extensions:
            _loaded_extensions[arch] = JitSpec(target_arch=arch).build_and_load()
    return _loaded_extensions[arch]


def _clear_extension_cache() -> None:
    with _extension_lock:
        _loaded_extensions.clear()


__all__ = ["JitSpec", "extension_is_loaded", "gen_jit_spec", "load_extension"]
