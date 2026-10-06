# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Op-local JIT for the SM100 Q8KV4 sparse decode attention kernels."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

import jinja2

from .. import _jit_cache
from ._build_utils import cuda_home as _cuda_home
from ._build_utils import cutlass_root as _cutlass_root
from ._build_utils import cutlass_version as _cutlass_version
from ._build_utils import require_cuda_version
from ._build_utils import target_arch as _resolve_target_arch

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
_CSRC = _ROOT / "csrc"
_TEMPLATES = _CSRC / "templates"
_SOURCES = _CSRC / "src"
_SM100_INCLUDE = _CSRC / "include/sm100"
_SPARSE_VARIANTS = {
    False: "decode_attention_q8kv4",
    True: "decode_attention_q8kv4_split",
}
# Widest TopK list one kernel binary accepts (kernel template bound and host-API constant
# kMaxSparseTopK); the list width itself is a runtime argument.
MAX_TOPK = 64
# Largest block-scale staging shift (an E4M3 exponent offset, Traits::kBlockScaleShift); each
# shift is its own kernel binary so caches whose products already fit pay nothing.
MAX_BLOCK_SCALE_SHIFT = 7
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
    if arch == "107a":
        if _cuda_version() < (13, 4):
            raise RuntimeError(
                "Q8KV4 decode attention on SM107 requires CUDA 13.4 or newer"
            )
        if _cutlass_version() < (4, 8):
            raise RuntimeError(
                "Q8KV4 decode attention on SM107 requires CUTLASS 4.8 or newer; set "
                f"CUTLASS_ROOT (found {_cutlass_version()} at {_cutlass_root()})"
            )
    return arch


def _tvm_ffi_include() -> Path:
    import tvm_ffi

    tvm_root = Path(tvm_ffi.__path__[0])
    for candidate in (tvm_root / "include", tvm_root.parent / "include"):
        if candidate.is_dir():
            return candidate
    raise RuntimeError("Cannot find TVM-FFI headers")


def _cache_root() -> Path:
    """Share the package JIT cache location; fall back to the torch extension dir."""
    explicit = os.environ.get("MINFER_FMHA_CACHE_DIR") or os.environ.get(
        "TORCH_EXTENSIONS_DIR"
    )
    root = (
        Path(explicit).expanduser()
        if explicit
        else Path.home() / ".cache/minfer/fmha_sm100"
    )
    return root / "decode_q8kv4"


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


# Set to 1 to skip the QMUL4 instruction path and compile the FP16 dequant fallback even when the
# toolchain supports QMUL4: the path a CUDA toolkit older than 13.4 would take.
DISABLE_QMUL4_ENV = "FMHA_SM100_DECODE_Q8KV4_DISABLE_QMUL4"


def _qmul4_disabled() -> bool:
    value = os.environ.get(DISABLE_QMUL4_ENV, "").strip().lower() or "0"
    if value not in ("0", "1", "false", "true"):
        raise ValueError(f"{DISABLE_QMUL4_ENV} must be 0 or 1, got {value!r}")
    return value in ("1", "true")


@cache
def _dequant_mode(arch: str) -> str:
    if _qmul4_disabled() or not _supports_qmul4(arch):
        return _FP16_DEQUANT
    return _QMUL4_DEQUANT


@lru_cache(maxsize=1)
def _namespace() -> _jit_cache.Namespace:
    """The cache namespace of the selected toolchain (see ``fmha_sm100._jit_cache``)."""
    return _jit_cache.Namespace(_cache_root(), _cuda_home())


# Compile and link rules of every library; part of each recipe, so changing them rebuilds.
_NINJA_RULES = """rule nvcc_compile
  command = $nvcc $nvcc_flags -MMD -MF $out.d -c $in -o $out
  description = Compiling $in
  depfile = $out.d
  deps = gcc

rule nvcc_link
  command = $nvcc -shared $in -o $out -lcuda
  description = Linking $out
"""


def _recipe(name: str, nvcc_flags: str, templates=(), **key) -> _jit_cache.Recipe:
    """One library's cache recipe (compiler, flags, build rules, ``key``); the files it compiles
    from are checked per record."""
    return _jit_cache.Recipe(
        _namespace(),
        name,
        {
            "nvcc": str(_cuda_home() / "bin/nvcc"),
            "nvcc_flags": nvcc_flags,
            "rules": _NINJA_RULES,
            **key,
        },
        templates=templates,
    )


def _nvcc_flags(dequant_mode: str, arch: str, block_scale_shift: int = 0) -> str:
    include_dirs = [
        _SM100_INCLUDE,
        _SM100_INCLUDE / "common",
        _SM100_INCLUDE / "collective",
        _SM100_INCLUDE / "device",
        _SM100_INCLUDE / "kernel",
        _cutlass_root() / "include",
        _cutlass_root() / "tools/util/include",
        _tvm_ffi_include(),
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
        f"-DMINIMAX_MSA_Q8KV4_BLOCK_SCALE_SHIFT={block_scale_shift}",
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
    block_scale_shift: int = 0,
) -> None:
    nvcc = _cuda_home() / "bin/nvcc"
    objects = [cache_dir / f"source_{index}.o" for index in range(len(sources))]
    builds = "\n".join(
        f"build {obj}: nvcc_compile {source}"
        for obj, source in zip(objects, sources, strict=True)
    )
    content = f"""ninja_required_version = 1.5

nvcc = {nvcc}
nvcc_flags = {_nvcc_flags(dequant_mode, arch, block_scale_shift)}

{_NINJA_RULES}
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
    block_scale_shift: int = 0

    @property
    def _component(self) -> str:
        return f"{self.variant_name}_gqa{self.gqa_ratio}_shift{self.block_scale_shift}"

    @property
    def uri(self) -> str:
        return f"{self._component}_{self.dequant_mode}_{self.target_arch}"

    @property
    def _params(self) -> dict:
        return {
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
            "sparse_topk": MAX_TOPK,
            "fixed_q_tokens_per_batch": 0,
        }

    def recipe(self) -> _jit_cache.Recipe:
        return _recipe(
            self.uri,
            _nvcc_flags(self.dequant_mode, self.target_arch, self.block_scale_shift),
            templates=(
                _TEMPLATES / "decode_attention_inst.cu.jinja",
                _TEMPLATES / "decode_attention_run.cu.jinja",
            ),
            params=self._params,
        )

    def build(self) -> Path:
        """Return the library, compiling it when no record matches the current sources."""
        return self.recipe().build(self._build)

    def build_and_load(self):
        import tvm_ffi

        return tvm_ffi.load_module(str(self.build()))

    def _build(self, cache_dir: Path) -> Path:
        params = self._params
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
            self.block_scale_shift,
        )
        _run_ninja(cache_dir, f"{self.variant_name} JIT module")
        return so_path


def gen_jit_spec(
    *,
    topk: int = 16,
    split_kv: bool = False,
    gqa_ratio: int = 16,
    device=None,
    block_scale_shift: int = 0,
) -> JitSpec:
    if not 1 <= int(topk) <= MAX_TOPK:
        raise ValueError(f"Q8KV4 decode attention requires 1 <= TopK <= {MAX_TOPK}, got {topk}")
    if gqa_ratio not in (8, 16):
        raise ValueError("Q8KV4 decode attention requires GQA ratio 8 or 16")
    if not 0 <= int(block_scale_shift) <= MAX_BLOCK_SCALE_SHIFT:
        raise ValueError(
            f"block_scale_shift must be in [0, {MAX_BLOCK_SCALE_SHIFT}], got {block_scale_shift}"
        )
    arch = _target_arch(device)
    return JitSpec(
        _SPARSE_VARIANTS[bool(split_kv)],
        bool(split_kv),
        _dequant_mode(arch),
        arch,
        gqa_ratio,
        int(block_scale_shift),
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
        self,
        *,
        topk: int,
        split_kv: bool,
        gqa_ratio: int = 16,
        device=None,
        block_scale_shift: int = 0,
    ) -> _VariantWrapper:
        spec = gen_jit_spec(
            topk=topk,
            split_kv=split_kv,
            gqa_ratio=gqa_ratio,
            device=device,
            block_scale_shift=block_scale_shift,
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
    block_scale_shift: int = 0,
):
    return _variant_manager.get(
        topk=topk,
        split_kv=split_kv,
        gqa_ratio=gqa_ratio,
        device=device,
        block_scale_shift=block_scale_shift,
    )


def _build_fixed_module(component: str, source: Path, label: str, arch: str):
    import tvm_ffi

    dequant_mode = _dequant_mode(arch)

    def build(cache_dir: Path) -> Path:
        so_path = cache_dir / f"{component}.so"
        _write_ninja(cache_dir, so_path, [source], dequant_mode, arch)
        _run_ninja(cache_dir, label)
        return so_path

    recipe = _recipe(f"{component}_{dequant_mode}_{arch}", _nvcc_flags(dequant_mode, arch))
    return tvm_ffi.load_module(str(recipe.build(build)))


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
    "DISABLE_QMUL4_ENV",
    "MAX_BLOCK_SCALE_SHIFT",
    "MAX_TOPK",
    "JitSpec",
    "gen_jit_spec",
    "get_fmha_fwd_variant",
    "get_plan_fn",
    "get_reduction_module",
]
