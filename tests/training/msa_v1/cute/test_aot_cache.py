"""Production logging and persistent CuTe AOT cache contracts."""

import logging
from pathlib import Path

import pytest

from msa_v1._common import aot_cache


def _write_source(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class _FakeCompiled:
    def export_to_c(self, object_file_path: str, function_name: str) -> None:
        Path(object_file_path).write_text(function_name, encoding="utf-8")


def test_msa_v1_environment_names() -> None:
    assert aot_cache._AOT_CACHE_ENV == "MSA_V1_AOT_CACHE"
    assert aot_cache._AOT_DISABLE_ENV == "MSA_V1_AOT_DISABLE"


def test_source_fingerprint_tracks_package_sources(tmp_path: Path) -> None:
    source_root = tmp_path / "msa_v1"
    kernel_path = source_root / "attention" / "kernel.py"
    _write_source(source_root / "__init__.py", "")
    _write_source(source_root / "attention" / "__init__.py", "")
    _write_source(kernel_path, "VALUE = 1\n")

    initial = aot_cache._hash_source_tree(source_root)
    _write_source(kernel_path, "VALUE = 2\n")

    assert aot_cache._hash_source_tree(source_root) != initial


def test_source_fingerprint_ignores_local_snapshot_dirs(tmp_path: Path) -> None:
    source_root = tmp_path / "msa_v1"
    _write_source(source_root / "__init__.py", "")
    initial = aot_cache._hash_source_tree(source_root)

    _write_source(source_root / "kl.0" / "kernel.py", "LOCAL_ONLY = True\n")

    assert aot_cache._hash_source_tree(source_root) == initial


def test_source_fingerprint_tracks_runtime_abi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "msa_v1"
    _write_source(source_root / "__init__.py", "")
    initial = aot_cache._hash_source_tree(source_root)

    monkeypatch.setattr(aot_cache, "_CUTLASS_DSL_VERSION", "next-version")

    assert aot_cache._hash_source_tree(source_root) != initial

    monkeypatch.setattr(aot_cache, "_CUDA_BACKEND_VERSION", "12.9")
    cuda12 = aot_cache._hash_source_tree(source_root)
    monkeypatch.setattr(aot_cache, "_CUDA_BACKEND_VERSION", "13.1")
    assert aot_cache._hash_source_tree(source_root) != cuda12


def test_cache_path_is_namespaced_by_source_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aot_cache, "_AOT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        aot_cache,
        "_compute_source_fingerprint",
        lambda: "source-fingerprint",
    )

    cache_path = Path(aot_cache._key_to_path(("kernel/name", "config")))

    assert cache_path.parent == tmp_path / "source-fingerprint"
    assert cache_path.name.startswith("kernel_name_")


def test_compile_or_load_is_silent_without_logging_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(aot_cache, "_AOT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(aot_cache, "_AOT_DISABLE", False)
    key = ("test_v1_aot_kernel", "config")

    compiled = aot_cache.compile_or_load(key, _FakeCompiled)

    assert isinstance(compiled, _FakeCompiled)
    assert Path(aot_cache._key_to_path(key) + ".o").is_file()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_compile_or_load_emits_debug_records_only_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(aot_cache, "_AOT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(aot_cache, "_AOT_DISABLE", False)

    with caplog.at_level(logging.DEBUG, logger=aot_cache.__name__):
        aot_cache.compile_or_load(("test_v1_debug_kernel",), _FakeCompiled)

    assert "AOT cache miss" in caplog.text
    assert "cute.compile done" in caplog.text
    assert "Saved AOT kernel" in caplog.text


def test_missing_aot_raises_when_jit_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(aot_cache, "_AOT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(aot_cache, "_AOT_DISABLE", False)
    monkeypatch.delenv("FMHA_SM100_ALLOW_JIT", raising=False)

    with pytest.raises(RuntimeError, match="JIT compilation is disabled"):
        aot_cache.try_load_aot(("missing_v1_aot_kernel",))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
