# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Pytest configuration for architecture-specific kernel tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

# Tests exercising the SM100 tcgen05/TMEM kernels; skipped off SM100/SM103.
_SM100_ONLY_TESTS = (
    "tests/integration/",
    "tests/regression/",
)
# Tests exercising the SM120/SM121 fmha_sm12x package; skipped on SM100/SM103.
_SM12X_ONLY_TESTS = (
    "tests/test_sm12x_reference.py",
    "tests/test_sm12x_triton_sparse.py",
    "tests/test_sm12x_equivalence.py",
)


def _cuda_capability() -> tuple[int, int] | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return int(major), int(minor)


def _is_sm100_family(capability: tuple[int, int] | None) -> bool:
    return capability in ((10, 0), (10, 3))


def _skipped_prefixes(capability: tuple[int, int] | None) -> tuple[str, ...]:
    """Tests for the *other* architecture family are skipped.

    SM100/SM103 runs the SM100 suite and skips the SM12x suite; every other
    device (including SM120/SM121) does the reverse.  Arch-agnostic tests
    (e.g. tests/test_arch.py, the arch-adaptive smoke tests) are in neither
    list and always run.
    """

    if _is_sm100_family(capability):
        return _SM12X_ONLY_TESTS
    return _SM100_ONLY_TESTS


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    _ = config
    path = collection_path.as_posix()
    return any(part in path for part in _skipped_prefixes(_cuda_capability()))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    _ = config
    capability = _cuda_capability()
    skipped = _skipped_prefixes(capability)
    reason = (
        "SM12x-only test; requires SM120/SM121"
        if _is_sm100_family(capability)
        else "SM100-only tcgen05/TMEM kernel; requires SM100/SM103"
    )
    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        path = item.path.as_posix()
        if any(part in path for part in skipped):
            item.add_marker(skip_marker)
