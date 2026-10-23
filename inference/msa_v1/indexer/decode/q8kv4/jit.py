"""Op-local JIT for the SM100 Q8KV4 decode indexer GEMM."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

import jinja2
from torch.utils import cpp_extension

from inference.msa_v1._build_utils import cuda_home as _cuda_home
from inference.msa_v1._build_utils import cutlass_root as _cutlass_root
from inference.msa_v1._build_utils import require_cuda_version
from inference.msa_v1._build_utils import supports_qmul4 as _probe_supports_qmul4
from inference.msa_v1._build_utils import target_arch as _resolve_target_arch
from inference.msa_v1._build_utils import torch_cuda_arch as _resolve_torch_arch

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
_CSRC = _ROOT / "csrc"
_API = _CSRC / "api"
_INCLUDE = _CSRC / "include"
_TEMPLATES = _CSRC / "templates"
_QMUL4_DEQUANT = "qmul4"
_FP16_DEQUANT = "fp16_fallback"


@lru_cache(maxsize=1)
def _cuda_version() -> tuple[int, int]:
    return require_cuda_version(
        (12, 9),
        component="Q8KV4 decode indexer",
    )


@cache
def _target_arch(device=None) -> str:
    return _resolve_target_arch(device, component="Q8KV4 decode indexer")


def _torch_arch(arch: str) -> str:
    return _resolve_torch_arch(arch, component="Q8KV4 decode indexer")


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
    digest.update(str(_cutlass_root()).encode())
    version_header = _cutlass_root() / "include/cutlass/version.h"
    if version_header.is_file():
        digest.update(version_header.read_bytes())
    return digest.hexdigest()[:16]


def _cache_root() -> Path:
    root = Path(
        os.environ.get(
            "TORCH_EXTENSIONS_DIR",
            Path.home() / ".cache/torch_extensions",
        )
    ).expanduser()
    return root / "minimax_msa/indexer_decode_qh_tiles_v1_q8kv4"


def _write_text_if_changed(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


@cache
def _supports_qmul4(arch: str) -> bool:
    """Return whether the selected NVCC accepts the public QMUL4 PTX form."""

    return _probe_supports_qmul4(arch, probe_root=_cache_root() / "capability_probes")


@cache
def _dequant_mode(arch: str) -> str:
    return _QMUL4_DEQUANT if _supports_qmul4(arch) else _FP16_DEQUANT


@dataclass(frozen=True)
class JitSpec:
    """Static architecture, head mapping, and legal MMA column capacity."""

    dequant_mode: str
    target_arch: str
    num_index_heads: int = 1
    query_columns: int = 16
    variant_name: str = "indexer_gemm_q8kv4"
    max_query_length: int = 16
    head_dim: int = 128
    page_tokens: int = 128
    scale_group_size: int = 16
    maximum_pages: int = 16384

    @property
    def uri(self) -> str:
        return (
            f"{self.variant_name}_{self.dequant_mode}_"
            f"{self.target_arch}_h{self.num_index_heads}_n{self.query_columns}_{_source_digest()}"
        )

    def build_and_load(self):
        cache_dir = _cache_root() / self.uri
        cache_dir.mkdir(parents=True, exist_ok=True)
        template = jinja2.Template(
            (_TEMPLATES / "indexer_gemm_inst.cu.jinja").read_text(encoding="utf-8")
        )
        generated_source = cache_dir / "indexer_gemm_inst.cu"
        _write_text_if_changed(
            generated_source,
            template.render(
                max_query_length=self.max_query_length,
                query_columns=self.query_columns,
                head_dim=self.head_dim,
                page_tokens=self.page_tokens,
                scale_group_size=self.scale_group_size,
                maximum_pages=self.maximum_pages,
            ),
        )

        module_name = f"minimax_msa_{self.uri}"
        sources = [
            _API / "indexer_gemm_api.cpp",
            _API / "indexer_gemm_binding.cpp",
            generated_source,
        ]
        include_paths = [
            _API,
            _INCLUDE,
            _cutlass_root() / "include",
            _cutlass_root() / "tools/util/include",
        ]
        started_at = time.time()
        previous_cuda_home = cpp_extension.CUDA_HOME
        previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
        cpp_extension.CUDA_HOME = str(_cuda_home())
        os.environ["TORCH_CUDA_ARCH_LIST"] = _torch_arch(self.target_arch)
        try:
            extension = cpp_extension.load(
                name=module_name,
                sources=[str(path) for path in sources],
                extra_include_paths=[str(path) for path in include_paths],
                extra_cflags=[
                    "-O3",
                    "-DNDEBUG",
                    "-std=c++20",
                    f"-DMINIMAX_MSA_INDEX_HEADS={self.num_index_heads}",
                    f"-DMINIMAX_MSA_QUERY_COLUMNS={self.query_columns}",
                ],
                extra_cuda_cflags=[
                    "-O3",
                    f"-DMINIMAX_MSA_INDEX_HEADS={self.num_index_heads}",
                    f"-DMINIMAX_MSA_QUERY_COLUMNS={self.query_columns}",
                    "-DNDEBUG",
                    "-lineinfo",
                    "-std=c++20",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-static-global-template-stub=false",
                    "-Xptxas=-O3",
                    f"-DMINIMAX_MSA_Q8KV4_INDEXER_HAS_QMUL4={int(self.dequant_mode == _QMUL4_DEQUANT)}",
                ],
                extra_ldflags=[
                    "-lcuda",
                    f"-Wl,-rpath,{_cuda_home() / 'lib64'}",
                ],
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
            "Compiled inference.msa_v1.indexer.decode.q8kv4 in %.1fs",
            time.time() - started_at,
        )
        return extension


def gen_jit_spec(
    device=None, *, num_index_heads: int = 1, query_length: int = 8
) -> JitSpec:
    """Select the smallest legal TMEM-source MMA column capacity."""

    arch = _target_arch(device)
    return JitSpec(
        dequant_mode=_dequant_mode(arch),
        target_arch=arch,
        num_index_heads=num_index_heads,
        query_columns=((query_length * num_index_heads + 15) // 16) * 16,
    )


@cache
def _load_extension_for_arch(arch: str, num_index_heads: int, query_columns: int):
    return JitSpec(
        dequant_mode=_dequant_mode(arch),
        target_arch=arch,
        num_index_heads=num_index_heads,
        query_columns=query_columns,
    ).build_and_load()


def load_extension(device=None, *, num_index_heads: int = 1, query_length: int = 8):
    """Build and load one extension for the selected tensor architecture."""

    query_columns = ((query_length * num_index_heads + 15) // 16) * 16
    return _load_extension_for_arch(
        _target_arch(device), num_index_heads, query_columns
    )


__all__ = ["JitSpec", "gen_jit_spec", "load_extension"]
