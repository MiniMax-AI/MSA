"""Shared public-toolchain discovery for MSA v1 JIT extensions."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MM_SPARSE_TARGET_ARCH = "MM_SPARSE_TARGET_ARCH"
QMUL4_PROBE_SOURCE = r"""
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


def normalize_target_arch(
    value: str, *, component: str = "MM-Sparse", supported_arches=("100a", "103a")
) -> str:
    """Normalize a supported architecture into its family-specific target."""

    arch = value.strip().lower().replace("+ptx", "")
    arch = arch.replace("sm_", "").replace("compute_", "").replace(".", "")
    normalized = arch if arch.endswith("a") else f"{arch}a"
    if normalized in supported_arches:
        return normalized
    supported = " and ".join(f"SM{item}" for item in supported_arches)
    raise RuntimeError(f"{component} supports only {supported}; got {value!r}")


def target_arch(
    device=None, *, component: str = "MM-Sparse", supported_arches=("100a", "103a")
) -> str:
    """Select an architecture from a tensor device or the offline-build override.

    Runtime callers must pass the device of an input tensor. The environment
    variable is intentionally considered only when no device is available.
    """

    import torch

    if device is not None:
        device = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        if device.type != "cuda":
            raise RuntimeError(f"{component} requires a CUDA device; got {device}")
        major, minor = torch.cuda.get_device_capability(device)
        return normalize_target_arch(
            f"{major}{minor}a", component=component, supported_arches=supported_arches
        )

    if value := os.environ.get(MM_SPARSE_TARGET_ARCH):
        return normalize_target_arch(
            value, component=component, supported_arches=supported_arches
        )

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return normalize_target_arch(
            f"{major}{minor}a", component=component, supported_arches=supported_arches
        )

    # Preserve the historical offline default. Reproducible AOT builds should
    # always set MM_SPARSE_TARGET_ARCH explicitly.
    return "103a"


def torch_cuda_arch(arch: str, *, component: str = "MM-Sparse") -> str:
    """Return the torch cpp-extension spelling for a normalized target."""

    normalized = normalize_target_arch(arch, component=component)
    return "10.0a" if normalized == "100a" else "10.3a"


def _unique_paths(candidates: list[Path]) -> tuple[Path, ...]:
    unique = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return tuple(unique)


def _nvcc_version(root: Path) -> tuple[int, int] | None:
    nvcc = root / "bin/nvcc"
    if not nvcc.is_file():
        return None
    try:
        result = subprocess.run(
            [str(nvcc), "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"release\s+(\d+)\.(\d+)", result.stdout + result.stderr)
    if result.returncode != 0 or match is None:
        return None
    return int(match.group(1)), int(match.group(2))


@lru_cache(maxsize=1)
def cuda_home() -> Path:
    """Return the first runnable CUDA toolkit selected through public mechanisms."""

    candidates = []
    if value := os.environ.get("CUDA_HOME"):
        candidates.append(Path(value))
    if value := os.environ.get("CUDACXX"):
        candidates.append(Path(value).parent.parent)

    try:
        from torch.utils import cpp_extension

        if cpp_extension.CUDA_HOME:
            candidates.append(Path(cpp_extension.CUDA_HOME))
    except ImportError:
        pass

    if nvcc := shutil.which("nvcc"):
        candidates.append(Path(nvcc).parent.parent)
    candidates.append(Path("/usr/local/cuda"))

    attempted = []
    for candidate in _unique_paths(candidates):
        attempted.append(str(candidate))
        if _nvcc_version(candidate) is not None:
            return candidate
    raise RuntimeError(
        "No runnable CUDA toolkit was found. Set CUDA_HOME or CUDACXX to a "
        f"toolkit containing a host-compatible bin/nvcc. Tried: {attempted}"
    )


@lru_cache(maxsize=1)
def cuda_version() -> tuple[int, int]:
    version = _nvcc_version(cuda_home())
    if version is None:
        raise RuntimeError(f"Unable to determine NVCC version under {cuda_home()}")
    return version


def require_cuda_version(
    minimum: tuple[int, int],
    *,
    component: str,
) -> tuple[int, int]:
    version = cuda_version()
    if version < minimum:
        raise RuntimeError(
            f"{component} requires CUDA Toolkit {minimum[0]}.{minimum[1]} or "
            f"newer; selected {cuda_home()} reports {version[0]}.{version[1]}"
        )
    return version


@lru_cache(maxsize=1)
def cutlass_root() -> Path:
    """Locate public CUTLASS headers from the environment or repository submodule."""

    candidates = []
    for variable in ("CUTLASS_ROOT", "CUTLASS_PATH"):
        if value := os.environ.get(variable):
            candidates.append(Path(value))
    candidates.append(_REPOSITORY_ROOT / "third_party/cutlass")

    attempted = []
    for candidate in _unique_paths(candidates):
        attempted.append(str(candidate))
        if (candidate / "include/cutlass/cutlass.h").is_file():
            return candidate
    raise RuntimeError(
        "CUTLASS headers were not found. Initialize third_party/cutlass or set "
        f"CUTLASS_ROOT. Tried: {attempted}"
    )


@contextmanager
def build_lock(cache_dir: Path):
    """Keep a shared library immutable while another rank builds or loads it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "build.lock").open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_text_if_changed(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _probe_qmul4(nvcc: Path, arch: str, probe_dir: Path) -> bool:
    result_path = probe_dir / "qmul4.result"
    if result_path.is_file():
        return result_path.read_text(encoding="utf-8").strip() == "supported"

    probe_dir.mkdir(parents=True, exist_ok=True)
    source_path = probe_dir / "qmul4_probe.cu"
    object_path = probe_dir / f"qmul4_probe.{os.getpid()}.o"
    _write_text_if_changed(source_path, QMUL4_PROBE_SOURCE)
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


def supports_qmul4(arch: str, *, probe_root: Path) -> bool:
    """Return whether the selected NVCC accepts the public QMUL4 PTX form for ``arch``.

    The answer is recorded under ``probe_root`` per toolkit, architecture and probe source, so
    every op that dequantizes NVFP4 shares one compile of the probe per toolchain.
    """

    nvcc = cuda_home() / "bin/nvcc"
    probe_key = hashlib.sha256()
    probe_key.update(str(nvcc.resolve()).encode())
    probe_key.update(str(cuda_version()).encode())
    probe_key.update(arch.encode())
    probe_key.update(QMUL4_PROBE_SOURCE.encode())
    probe_dir = probe_root / probe_key.hexdigest()[:16]
    with build_lock(probe_dir):
        return _probe_qmul4(nvcc, arch, probe_dir)


__all__ = [
    "MM_SPARSE_TARGET_ARCH",
    "QMUL4_PROBE_SOURCE",
    "build_lock",
    "cuda_home",
    "cuda_version",
    "cutlass_root",
    "normalize_target_arch",
    "require_cuda_version",
    "supports_qmul4",
    "target_arch",
    "torch_cuda_arch",
]
