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


def _hold_jit_build_lock_then_die(directory, ready):
    import os

    from inference.msa_v1._build_utils import jit_build_lock

    with jit_build_lock(directory):
        # What torch's FileBaton leaves behind when its builder is killed mid-compile.
        (Path(directory) / "lock").touch()
        ready.set()
        os._exit(137)


def test_jit_build_lock_recovers_from_a_killed_builder(tmp_path, monkeypatch):
    import multiprocessing

    from inference.msa_v1._build_utils import jit_build_lock

    monkeypatch.setattr(_build_utils, "_torch_uses_file_baton", lambda: True)

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    builder = context.Process(
        target=_hold_jit_build_lock_then_die, args=(str(tmp_path), ready)
    )
    builder.start()
    builder.join(timeout=60)
    assert ready.is_set() and builder.exitcode == 137
    assert (tmp_path / "lock").exists()

    # The fcntl lock died with its holder; the stale torch baton must not block us.
    with jit_build_lock(tmp_path):
        assert not (tmp_path / "lock").exists()


def test_jit_build_lock_keeps_no_baton_when_none_exists(tmp_path):
    from inference.msa_v1._build_utils import jit_build_lock

    with jit_build_lock(tmp_path / "fresh"):
        assert (tmp_path / "fresh" / "build.lock").exists()
        assert not (tmp_path / "fresh" / "lock").exists()


def test_jit_build_lock_keeps_torch_lock_file_without_file_baton(tmp_path, monkeypatch):
    from inference.msa_v1._build_utils import jit_build_lock

    # torch >= 2.14 keeps a persistent "lock" file for its own OS-level lock.
    monkeypatch.setattr(_build_utils, "_torch_uses_file_baton", lambda: False)
    (tmp_path / "lock").touch()
    with jit_build_lock(tmp_path):
        assert (tmp_path / "lock").exists()
