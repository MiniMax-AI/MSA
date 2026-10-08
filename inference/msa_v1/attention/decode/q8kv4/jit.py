"""Op-local JIT for the SM100 Q8KV4 sparse decode attention kernels."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

import jinja2

from inference.msa_v1._build_utils import cuda_home as _cuda_home
from inference.msa_v1._build_utils import cutlass_root as _cutlass_root
from inference.msa_v1._build_utils import require_cuda_version
from inference.msa_v1._build_utils import target_arch as _resolve_target_arch

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
_CSRC = _ROOT / "csrc"
_TEMPLATES = _CSRC / "templates"
_SOURCES = _CSRC / "src"
_SM100_INCLUDE = _CSRC / "include/sm100"
_SPARSE_VARIANTS = {
    False: "decode_attention_q8kv4_topk16",
    True: "decode_attention_q8kv4_topk16_split",
}
_QMUL4_DEQUANT = "qmul4"
_FP16_DEQUANT = "fp16_fallback"
_QMUL4_PROBE_SOURCE = r"""
#include <cstdint>

__global__ void qmul4_probe(uint32_t* output) {
  uint32_t result;
  uint16_t packed_e2m1 = 0;
  uint32_t scale_e4m3 = 0;
  asm volatile(
      "mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, %1, %2;"
      : "=r"(result)
      : "h"(packed_e2m1), "r"(scale_e4m3));
  output[0] = result;
}
""".lstrip()


def _write_text_if_changed(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


@contextmanager
def _build_lock(cache_dir: Path):
    """Keep a shared library immutable while another rank builds or loads it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "build.lock").open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@lru_cache(maxsize=1)
def _cuda_version() -> tuple[int, int]:
    return require_cuda_version(
        (12, 9),
        component="Q8KV4 decode attention",
    )


@cache
def _target_arch(device=None) -> str:
    arch = _resolve_target_arch(
        device,
        component="Q8KV4 decode attention",
        supported_arches=("100a", "103a", "107a"),
    )
    if arch == "107a" and _cuda_version() < (13, 5):
        raise RuntimeError(
            "Q8KV4 decode attention on SM107 requires CUDA 13.5 or newer"
        )
    return arch


def _tvm_ffi_include() -> Path:
    import tvm_ffi

    tvm_root = Path(tvm_ffi.__path__[0])
    for candidate in (tvm_root / "include", tvm_root.parent / "include"):
        if candidate.is_dir():
            return candidate
    raise RuntimeError("Cannot find TVM-FFI headers")


@lru_cache(maxsize=1)
def _source_digest() -> str:
    import torch
    import tvm_ffi

    digest = hashlib.sha256()
    for path in sorted(_CSRC.rglob("*")):
        # csrc/api is compiled by torch.utils.cpp_extension (host side); it does not feed the
        # kernel modules, so it must not invalidate their cache.
        if path.is_file() and "api" not in path.relative_to(_CSRC).parts:
            digest.update(str(path.relative_to(_ROOT)).encode())
            digest.update(path.read_bytes())
    digest.update(Path(__file__).read_bytes())
    digest.update(sys.version.encode())
    digest.update(torch.__version__.encode())
    digest.update(str(torch._C._GLIBCXX_USE_CXX11_ABI).encode())
    digest.update(getattr(tvm_ffi, "__version__", "unknown").encode())
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
    return root / "minimax_msa/attention_decode_q8kv4"


@cache
def _supports_qmul4(arch: str) -> bool:
    """Return whether the selected NVCC accepts the public QMUL4 PTX form."""

    nvcc = _cuda_home() / "bin/nvcc"
    probe_key = hashlib.sha256()
    probe_key.update(str(nvcc.resolve()).encode())
    probe_key.update(str(_cuda_version()).encode())
    probe_key.update(arch.encode())
    probe_key.update(_QMUL4_PROBE_SOURCE.encode())
    probe_dir = _cache_root() / "capability_probes" / probe_key.hexdigest()[:16]
    with _build_lock(probe_dir):
        return _probe_qmul4(nvcc, arch, probe_dir)


def _probe_qmul4(nvcc: Path, arch: str, probe_dir: Path) -> bool:
    result_path = probe_dir / "qmul4.result"
    if result_path.is_file():
        return result_path.read_text(encoding="utf-8").strip() == "supported"

    probe_dir.mkdir(parents=True, exist_ok=True)
    source_path = probe_dir / "qmul4_probe.cu"
    object_path = probe_dir / f"qmul4_probe.{os.getpid()}.o"
    _write_text_if_changed(source_path, _QMUL4_PROBE_SOURCE)
    result = subprocess.run(
        [
            str(nvcc),
            "-std=c++20",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            "-c",
            str(source_path),
            "-o",
            str(object_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    supported = result.returncode == 0
    if supported:
        object_path.unlink(missing_ok=True)
    else:
        logger.info(
            "Selected NVCC does not accept QMUL4 for SM%s; using FP16 dequant fallback",
            arch,
        )
    _write_text_if_changed(
        result_path,
        "supported\n" if supported else "unsupported\n",
    )
    return supported


@cache
def _dequant_mode(arch: str) -> str:
    return _QMUL4_DEQUANT if _supports_qmul4(arch) else _FP16_DEQUANT


def _cache_dir(component: str, arch: str, dequant_mode: str) -> Path:
    return _cache_root() / f"{component}_{dequant_mode}_{arch}_{_source_digest()}"


def _nvcc_flags(cache_dir: Path, dequant_mode: str, arch: str) -> str:
    include_dirs = [
        _SM100_INCLUDE,
        _SM100_INCLUDE / "common",
        _SM100_INCLUDE / "collective",
        _SM100_INCLUDE / "device",
        _SM100_INCLUDE / "kernel",
        _cutlass_root() / "include",
        _cutlass_root() / "tools/util/include",
        _tvm_ffi_include(),
        cache_dir,
    ]
    flags = [
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
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
        "-Xcudafe",
        "--diag_suppress=2908",
        *[f"-I{path}" for path in include_dirs],
        "-use_fast_math",
        "-DNDEBUG",
        f"-DMINIMAX_MSA_Q8KV4_HAS_QMUL4={int(dequant_mode == _QMUL4_DEQUANT)}",
        f"-DMINIMAX_MSA_Q8KV4_RAW_KV_STAGES={12 if arch == '107a' else 8}",
        "-Xptxas",
        "-O3",
        "-Xcompiler",
        "-fPIC",
    ]
    return " ".join(flags)


def _run_ninja(cache_dir: Path, label: str) -> None:
    started_at = time.time()
    result = subprocess.run(
        ["ninja", "-j1"],
        cwd=cache_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} compilation failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    logger.info("Compiled %s in %.1fs", label, time.time() - started_at)


def _write_ninja(
    cache_dir: Path,
    output: Path,
    sources: list[Path],
    dequant_mode: str,
    arch: str,
) -> None:
    nvcc = _cuda_home() / "bin/nvcc"
    objects = [cache_dir / f"source_{index}.o" for index in range(len(sources))]
    builds = "\n".join(
        f"build {obj}: nvcc_compile {source}"
        for obj, source in zip(objects, sources, strict=True)
    )
    content = f"""ninja_required_version = 1.5

nvcc = {nvcc}
nvcc_flags = {_nvcc_flags(cache_dir, dequant_mode, arch)}

rule nvcc_compile
  command = $nvcc $nvcc_flags -MMD -MF $out.d -c $in -o $out
  description = Compiling $in
  depfile = $out.d
  deps = gcc

rule nvcc_link
  command = $nvcc -shared $in -o $out -lcuda
  description = Linking $out

{builds}
build {output}: nvcc_link {" ".join(str(obj) for obj in objects)}
"""
    _write_text_if_changed(cache_dir / "build.ninja", content)


@dataclass(frozen=True)
class JitSpec:
    """A deterministic recipe for one op-local JIT module."""

    variant_name: str
    split_kv: bool
    dequant_mode: str
    target_arch: str
    gqa_ratio: int = 16
    output_mode: str = "bf16"

    @property
    def uri(self) -> str:
        return (
            f"{self.variant_name}_gqa{self.gqa_ratio}_{self.output_mode}_{self.dequant_mode}_"
            f"{self.target_arch}_{_source_digest()}"
        )

    def build_and_load(self):
        cache_dir = _cache_dir(
            f"{self.variant_name}_gqa{self.gqa_ratio}_{self.output_mode}",
            self.target_arch,
            self.dequant_mode,
        )
        with _build_lock(cache_dir):
            return self._build_and_load_locked(cache_dir)

    def _build_and_load_locked(self, cache_dir: Path):
        params = {
            "variant_name": self.variant_name,
            "func_name": f"decode_attention_{self.variant_name}",
            "tile_q": self.gqa_ratio,
            "tile_kv": 128,
            "thread_shape": "_2, _1, _1",
            "page_size": 128,
            "pack_factor": self.gqa_ratio,
            "single_wg": "false",
            "is_split_kv": "true" if self.split_kv else "false",
            "sparse_mode": "Sparse",
            "sparse_topk": 16,
            "fixed_q_tokens_per_batch": 0,
            "output_mxfp8": self.output_mode == "mxfp8",
        }
        inst_template = jinja2.Template(
            (_TEMPLATES / "decode_attention_inst.cu.jinja").read_text()
        )
        run_template = jinja2.Template(
            (_TEMPLATES / "decode_attention_run.cu.jinja").read_text()
        )
        inst_cu = cache_dir / "decode_attention_inst.cu"
        run_cu = cache_dir / "decode_attention_run.cu"
        _write_text_if_changed(inst_cu, inst_template.render(**params))
        _write_text_if_changed(run_cu, run_template.render(**params))
        so_path = cache_dir / f"{self.variant_name}.so"
        _write_ninja(
            cache_dir,
            so_path,
            [inst_cu, run_cu],
            self.dequant_mode,
            self.target_arch,
        )
        _run_ninja(cache_dir, f"{self.variant_name} JIT module")

        import tvm_ffi

        return tvm_ffi.load_module(str(so_path))


def _validate_gqa_arch(gqa_ratio: int, device=None) -> str:
    arch = _target_arch(device)
    if gqa_ratio == 8 and arch not in ("100a", "103a"):
        raise RuntimeError("Q8KV4 decode attention with GQA=8 requires SM100 or SM103")
    return arch


def gen_jit_spec(
    *,
    topk: int = 16,
    split_kv: bool = False,
    gqa_ratio: int = 16,
    device=None,
    output_mode: str = "bf16",
) -> JitSpec:
    if int(topk) != 16:
        raise ValueError(f"Q8KV4 decode attention requires TopK 16, got {topk}")
    if gqa_ratio not in (8, 16):
        raise ValueError("Q8KV4 decode attention requires GQA ratio 8 or 16")
    if output_mode not in ("bf16", "mxfp8"):
        raise ValueError("output_mode must be 'bf16' or 'mxfp8'")
    arch = _validate_gqa_arch(gqa_ratio, device)
    return JitSpec(
        _SPARSE_VARIANTS[bool(split_kv)],
        bool(split_kv),
        _dequant_mode(arch),
        arch,
        gqa_ratio,
        output_mode,
    )


class _VariantWrapper:
    def __init__(self, fn):
        self._fn = fn

    def run(self, *args):
        return self._fn(*args)


class _VariantManager:
    def __init__(self):
        self._loaded: dict[str, _VariantWrapper] = {}
        self._lock = threading.Lock()

    def get(
        self, *, topk: int, split_kv: bool, gqa_ratio: int = 16,
        output_mode: str = "bf16", device=None
    ) -> _VariantWrapper:
        spec = gen_jit_spec(
            topk=topk, split_kv=split_kv, gqa_ratio=gqa_ratio,
            output_mode=output_mode, device=device
        )
        cached = self._loaded.get(spec.uri)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._loaded.get(spec.uri)
            if cached is not None:
                return cached
            module = spec.build_and_load()
            wrapper = _VariantWrapper(getattr(module, f"run_{spec.variant_name}"))
            self._loaded[spec.uri] = wrapper
            return wrapper


_variant_manager = _VariantManager()


def get_fmha_fwd_variant(
    *,
    topk: int = 16,
    split_kv: bool = False,
    gqa_ratio: int = 16,
    device=None,
    output_mode: str = "bf16",
):
    return _variant_manager.get(
        topk=topk, split_kv=split_kv, gqa_ratio=gqa_ratio,
        output_mode=output_mode, device=device
    )


def _build_fixed_module(component: str, source: Path, label: str, arch: str):
    import tvm_ffi

    dequant_mode = _dequant_mode(arch)
    cache_dir = _cache_dir(component, arch, dequant_mode)
    with _build_lock(cache_dir):
        so_path = cache_dir / f"{component}.so"
        _write_ninja(
            cache_dir,
            so_path,
            [source],
            dequant_mode,
            arch,
        )
        _run_ninja(cache_dir, label)
        return tvm_ffi.load_module(str(so_path))


_plan_modules = {}
_plan_lock = threading.Lock()


def get_plan_fn(device=None):
    arch = _target_arch(device)
    if arch in _plan_modules:
        return _plan_modules[arch]
    with _plan_lock:
        if arch not in _plan_modules:
            _plan_modules[arch] = _build_fixed_module(
                "decode_attention_plan",
                _SOURCES / "decode_attention_plan.cu",
                "Q8KV4 decode attention plan module",
                arch,
            )
    return _plan_modules[arch]


_reduction_modules = {}
_reduction_lock = threading.Lock()


def get_reduction_module(device=None):
    arch = _target_arch(device)
    if arch in _reduction_modules:
        return _reduction_modules[arch]
    with _reduction_lock:
        if arch not in _reduction_modules:
            _reduction_modules[arch] = _build_fixed_module(
                "decode_attention_reduction",
                _SOURCES / "decode_attention_reduction.cu",
                "Q8KV4 decode attention reduction module",
                arch,
            )
    return _reduction_modules[arch]


def _clear_loaded_extensions() -> None:
    _variant_manager._loaded.clear()
    _plan_modules.clear()
    _reduction_modules.clear()


__all__ = [
    "JitSpec",
    "gen_jit_spec",
    "get_fmha_fwd_variant",
    "get_plan_fn",
    "get_reduction_module",
]
