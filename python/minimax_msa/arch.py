# SPDX-License-Identifier: MIT

"""CUDA architecture selection for MiniMax Sparse Attention JIT builds."""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from typing import Final

_CUDA_ARCH_ENV: Final = "MSA_CUDA_ARCH"
_LEGACY_CUDA_ARCH_ENV: Final = "FMHA_SM100_CUDA_ARCH"
_NVCC_GENCODES_ENV: Final = "MSA_NVCC_GENCODES"
_LEGACY_NVCC_GENCODES_ENV: Final = "FMHA_SM100_NVCC_GENCODES"
_ARCH_RE: Final = re.compile(r"^(?:sm_|compute_)?(?P<major>\d{2,3})(?P<suffix>a?)$")
_SM100_GENCODE_CODE_MARKERS: Final = (
    "code=sm_100a",
    "code=sm_103a",
    "code=compute_100a",
    "code=compute_103a",
)
_SUPPORTED_SM100_TARGETS: Final = ("sm_100a", "sm_103a")
_SM12X_GENCODE_CODE_MARKERS: Final = (
    "code=sm_120",
    "code=sm_121",
    "code=compute_120",
    "code=compute_121",
)
_SUPPORTED_SM12X_TARGETS: Final = ("sm_120", "sm_121")


class UnsupportedCudaArchError(RuntimeError):
    """Raised when an architecture-specific kernel is requested for the wrong SM."""


@dataclass(frozen=True, slots=True)
class CudaArch:
    """CUDA virtual and real architecture pair."""

    compute: str
    code: str

    @property
    def cache_suffix(self) -> str:
        return self.code.replace("sm_", "_sm")

    @property
    def arch_flag(self) -> str:
        return f"-arch={self.code}"

    @property
    def gencode_flag(self) -> str:
        return f"-gencode=arch={self.compute},code={self.code}"


_DEFAULT_GENCODES: Final = (
    CudaArch(compute="compute_100a", code="sm_100a"),
    CudaArch(compute="compute_103a", code="sm_103a"),
)


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _parse_cuda_arch(value: str) -> CudaArch:
    normalized = value.strip().lower()
    match = _ARCH_RE.fullmatch(normalized)
    if match is None:
        raise ValueError(
            f"{_CUDA_ARCH_ENV} must look like sm_100a, sm_103a, sm_120, or sm_121; "
            f"got {value!r}"
        )
    digits = match.group("major")
    suffix = match.group("suffix")
    return CudaArch(compute=f"compute_{digits}{suffix}", code=f"sm_{digits}{suffix}")


def _detect_device_arch() -> CudaArch | None:
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    if major == 10 and minor == 0:
        return CudaArch(compute="compute_100a", code="sm_100a")
    if major == 10 and minor == 3:
        return CudaArch(compute="compute_103a", code="sm_103a")
    if major == 12 and minor in (0, 1):
        digits = f"{major}{minor}"
        return CudaArch(compute=f"compute_{digits}", code=f"sm_{digits}")
    return None


def selected_cuda_arch() -> CudaArch | None:
    """Return the explicit or detected single-arch CUDA target."""

    explicit = _first_env(_CUDA_ARCH_ENV, _LEGACY_CUDA_ARCH_ENV)
    if explicit:
        return _parse_cuda_arch(explicit)
    return _detect_device_arch()


def nvcc_gencode_flags() -> list[str]:
    """Return gencode flags for csrc JIT builds."""

    explicit_gencodes = _first_env(_NVCC_GENCODES_ENV, _LEGACY_NVCC_GENCODES_ENV)
    if explicit_gencodes:
        return shlex.split(explicit_gencodes)
    arch = selected_cuda_arch()
    if arch is not None:
        return [arch.gencode_flag]
    return [arch.gencode_flag for arch in _DEFAULT_GENCODES]



def require_sm100_csrc_arch(component: str) -> None:
    """Reject non-SM100-family targets for tcgen05/TMEM kernels."""

    gencodes = nvcc_gencode_flags()
    if all(
        any(marker in flag for marker in _SM100_GENCODE_CODE_MARKERS)
        for flag in gencodes
    ):
        return
    selected = " ".join(gencodes)
    supported = ", ".join(_SUPPORTED_SM100_TARGETS)
    raise UnsupportedCudaArchError(
        f"{component} is an SM100/SM103-only tcgen05/TMEM kernel and does not "
        f"support CUDA target {selected}; supported targets: {supported}. "
        "Add separate SM12x kernels instead of aliasing fmha_sm100 on SM120/SM121."
    )


def require_sm12x_csrc_arch(component: str) -> None:
    """Reject non-SM12x-family targets for SM120/SM121 kernels."""

    gencodes = nvcc_gencode_flags()
    if all(
        any(marker in flag for marker in _SM12X_GENCODE_CODE_MARKERS)
        for flag in gencodes
    ):
        return
    selected = " ".join(gencodes)
    supported = ", ".join(_SUPPORTED_SM12X_TARGETS)
    raise UnsupportedCudaArchError(
        f"{component} is an SM120/SM121 kernel and does not support CUDA "
        f"target {selected}; supported targets: {supported}."
    )


def ensure_sm100_kernel_arch(package: str) -> None:
    """Compatibility alias for SM100-only kernel guards."""

    require_sm100_csrc_arch(package)


def cpp_extension_arch_flag() -> str:
    """Return the torch cpp_extension CUDA arch flag."""

    arch = selected_cuda_arch()
    if arch is not None:
        return arch.arch_flag
    return CudaArch(compute="compute_100", code="sm_100").arch_flag


def cuda_arch_cache_suffix() -> str:
    """Return a stable cache suffix for non-default architecture selections."""

    explicit_gencodes = _first_env(_NVCC_GENCODES_ENV, _LEGACY_NVCC_GENCODES_ENV)
    if explicit_gencodes:
        digest = re.sub(r"[^0-9A-Za-z]+", "_", explicit_gencodes).strip("_")
        return f"_{digest}" if digest else "_custom_arch"
    arch = selected_cuda_arch()
    return arch.cache_suffix if arch is not None else ""
