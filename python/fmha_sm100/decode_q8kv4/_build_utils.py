# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Public-toolchain discovery for the Q8KV4 decode JIT (CUDA, CUTLASS, target arch)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # python/fmha_sm100
FMHA_SM100_DECODE_Q8KV4_ARCH = "FMHA_SM100_DECODE_Q8KV4_ARCH"


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

    if value := os.environ.get(FMHA_SM100_DECODE_Q8KV4_ARCH):
        return normalize_target_arch(
            value, component=component, supported_arches=supported_arches
        )

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return normalize_target_arch(
            f"{major}{minor}a", component=component, supported_arches=supported_arches
        )

    # Preserve the historical offline default. Reproducible AOT builds should
    # always set FMHA_SM100_DECODE_Q8KV4_ARCH explicitly.
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
    candidates.append(_PACKAGE_ROOT / "cutlass")

    attempted = []
    for candidate in _unique_paths(candidates):
        attempted.append(str(candidate))
        if (candidate / "include/cutlass/cutlass.h").is_file():
            return candidate
    raise RuntimeError(
        "CUTLASS headers were not found. Initialize the fmha_sm100/cutlass submodule or set "
        f"CUTLASS_ROOT. Tried: {attempted}"
    )


@lru_cache(maxsize=1)
def cutlass_version() -> tuple[int, int]:
    """Read CUTLASS_MAJOR/CUTLASS_MINOR from the selected headers."""

    header = (cutlass_root() / "include/cutlass/version.h").read_text(encoding="utf-8")
    versions = {}
    for name in ("CUTLASS_MAJOR", "CUTLASS_MINOR"):
        match = re.search(rf"#define\s+{name}\s+(\d+)", header)
        if match is None:
            raise RuntimeError(f"{name} not found in {cutlass_root()}/include/cutlass/version.h")
        versions[name] = int(match.group(1))
    return versions["CUTLASS_MAJOR"], versions["CUTLASS_MINOR"]


__all__ = [
    "FMHA_SM100_DECODE_Q8KV4_ARCH",
    "cutlass_version",
    "cuda_home",
    "cuda_version",
    "cutlass_root",
    "normalize_target_arch",
    "require_cuda_version",
    "target_arch",
    "torch_cuda_arch",
]
