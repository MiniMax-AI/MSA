"""Tests for public CUDA and CUTLASS toolchain discovery."""

from pathlib import Path

import pytest
import torch
from torch.utils import cpp_extension

from inference.msa_v1 import _build_utils


@pytest.fixture(autouse=True)
def _clear_discovery_caches():
    _build_utils.cuda_home.cache_clear()
    _build_utils.cuda_version.cache_clear()
    _build_utils.cutlass_root.cache_clear()
    yield
    _build_utils.cuda_home.cache_clear()
    _build_utils.cuda_version.cache_clear()
    _build_utils.cutlass_root.cache_clear()


def test_cuda_discovery_skips_non_runnable_toolkits(monkeypatch) -> None:
    invalid = Path("/tmp/non-runnable-cuda").resolve()
    fallback = Path("/usr/local/cuda").resolve()
    monkeypatch.setenv("CUDA_HOME", str(invalid))
    monkeypatch.delenv("CUDACXX", raising=False)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", str(invalid))
    monkeypatch.setattr(_build_utils.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        _build_utils,
        "_nvcc_version",
        lambda root: (12, 9) if root == fallback else None,
    )

    assert _build_utils.cuda_home() == fallback
    assert _build_utils.cuda_version() == (12, 9)


def test_cutlass_discovery_uses_repository_submodule(monkeypatch) -> None:
    monkeypatch.setenv("CUTLASS_ROOT", "/tmp/missing-cutlass")
    monkeypatch.delenv("CUTLASS_PATH", raising=False)
    expected = Path(__file__).resolve().parents[3] / "third_party/cutlass"
    assert _build_utils.cutlass_root() == expected.resolve()


def test_required_cuda_version_reports_selected_toolkit(monkeypatch) -> None:
    monkeypatch.setattr(_build_utils, "cuda_version", lambda: (12, 9))
    monkeypatch.setattr(_build_utils, "cuda_home", lambda: Path("/opt/cuda-12.9"))
    with pytest.raises(RuntimeError, match="requires CUDA Toolkit 13.4"):
        _build_utils.require_cuda_version(
            (13, 4),
            component="QMUL4 test kernel",
        )


def test_tensor_device_overrides_offline_target(monkeypatch) -> None:
    requested_devices = []
    monkeypatch.setenv("MM_SPARSE_TARGET_ARCH", "100a")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device=None: requested_devices.append(device) or (10, 3),
    )

    device = torch.device("cuda:7")
    assert _build_utils.target_arch(device) == "103a"
    assert requested_devices == [device]


def test_offline_target_uses_namespaced_environment(monkeypatch) -> None:
    monkeypatch.setenv("MM_SPARSE_TARGET_ARCH", "sm_100a")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "10.3a")

    assert _build_utils.target_arch() == "100a"
    assert _build_utils.torch_cuda_arch("103a") == "10.3a"


def test_unsupported_tensor_device_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device=None: (9, 0),
    )

    with pytest.raises(RuntimeError, match="supports only SM100a and SM103a"):
        _build_utils.target_arch(torch.device("cuda:0"))
