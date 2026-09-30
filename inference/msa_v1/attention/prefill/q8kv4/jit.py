"""Op-local JIT for the SM100 Q8KV4 sparse prefill attention kernel."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jinja2
from torch.utils import cpp_extension

logger = logging.getLogger(__name__)

from inference.msa_v1._build_utils import cuda_home as _cuda_home
from inference.msa_v1._build_utils import cutlass_root as _cutlass_root
from inference.msa_v1._build_utils import require_cuda_version
from inference.msa_v1._build_utils import target_arch as _resolve_target_arch
from inference.msa_v1._build_utils import torch_cuda_arch as _resolve_torch_arch


_ROOT = Path(__file__).resolve().parent
_CSRC = _ROOT / "csrc"
_API = _CSRC / "api"
_INCLUDE = _CSRC / "include"
_TEMPLATES = _CSRC / "templates"


@lru_cache(maxsize=1)
def _cuda_version() -> tuple[int, int]:
    return require_cuda_version(
        (13, 4),
        component="Q8KV4 prefill attention with QMUL4",
    )


@lru_cache(maxsize=None)
def _target_arch(device=None) -> str:
    return _resolve_target_arch(device, component="Q8KV4 prefill attention")


def _torch_arch(arch: str) -> str:
    return _resolve_torch_arch(arch, component="Q8KV4 prefill attention")


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
    return root / "minimax_msa/attention_prefill_q8kv4"


def _write_text_if_changed(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


@dataclass(frozen=True)
class JitSpec:
    """The single compile-time configuration supported by this op."""

    target_arch: str
    variant_name: str = "prefill_attention_q8kv4"
    q_heads_per_kv: int = 16
    head_dim: int = 128
    page_size: int = 128
    topk: int = 16
    q_stages: int = 3
    score_stages: int = 2

    @property
    def uri(self) -> str:
        return f"{self.variant_name}_{self.target_arch}_{_source_digest()}"

    def build_and_load(self):
        import torch

        cache_dir = _cache_root() / self.uri
        cache_dir.mkdir(parents=True, exist_ok=True)
        template = jinja2.Template(
            (_TEMPLATES / "prefill_attention_inst.cu.jinja").read_text(encoding="utf-8")
        )
        generated_source = cache_dir / "prefill_attention_inst.cu"
        _write_text_if_changed(
            generated_source,
            template.render(
                q_heads_per_kv=self.q_heads_per_kv,
                head_dim=self.head_dim,
                page_size=self.page_size,
                topk=self.topk,
                q_stages=self.q_stages,
                score_stages=self.score_stages,
            ),
        )

        module_name = f"minimax_msa_{self.uri}"
        sources = [
            _API / "prefill_attention_api.cpp",
            _API / "prefill_attention_binding.cpp",
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
                extra_cflags=["-O3", "-DNDEBUG", "-std=c++20"],
                extra_cuda_cflags=[
                    "-O3",
                    "-DNDEBUG",
                    "-lineinfo",
                    "-std=c++20",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-static-global-template-stub=false",
                    "-Xptxas=-O3",
                    "-DCUTLASS_ENABLE_GDC_FOR_SM100",
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
            "Compiled inference.msa_v1.attention.prefill.q8kv4 in %.1fs",
            time.time() - started_at,
        )
        return extension


def gen_jit_spec(device=None) -> JitSpec:
    """Return the only production compile-time configuration."""

    return JitSpec(target_arch=_target_arch(device))


@lru_cache(maxsize=None)
def _load_extension_for_arch(arch: str):
    return JitSpec(target_arch=arch).build_and_load()


def load_extension(device=None):
    """Build and load one extension for the selected tensor architecture."""

    return _load_extension_for_arch(_target_arch(device))


__all__ = ["JitSpec", "gen_jit_spec", "load_extension"]
