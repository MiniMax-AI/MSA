"""Shared MSA v1 inference AOT cache contracts."""

from pathlib import Path

import pytest

from inference.msa_v1 import aot_cache


class _FakeCompiled:
    def export_to_c(self, object_file_path: str, function_name: str) -> None:
        Path(object_file_path).write_text(function_name, encoding="utf-8")


def _write_source(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_aot_cache_uses_msa_v1_namespace_and_full_inference_source_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert aot_cache._AOT_CACHE_ENV == "MSA_V1_AOT_CACHE"
    assert aot_cache._AOT_DISABLE_ENV == "MSA_V1_AOT_DISABLE"
    assert aot_cache._SOURCE_ROOT == Path(aot_cache.__file__).resolve().parent

    source_root = tmp_path / "msa_v1"
    kernel_path = source_root / "indexer" / "prefill" / "kernel.py"
    _write_source(source_root / "__init__.py", "")
    _write_source(kernel_path, "VALUE = 1\n")
    initial = aot_cache._hash_source_tree(source_root)
    _write_source(kernel_path, "VALUE = 2\n")
    assert aot_cache._hash_source_tree(source_root) != initial

    monkeypatch.setattr(aot_cache, "_CUTLASS_DSL_VERSION", "4.5.2")
    minimum_version = aot_cache._hash_source_tree(source_root)
    monkeypatch.setattr(aot_cache, "_CUTLASS_DSL_VERSION", "4.7.1")
    assert aot_cache._hash_source_tree(source_root) != minimum_version

    monkeypatch.setattr(aot_cache, "_CUDA_BACKEND_VERSION", "12.9")
    cuda12 = aot_cache._hash_source_tree(source_root)
    monkeypatch.setattr(aot_cache, "_CUDA_BACKEND_VERSION", "13.1")
    assert aot_cache._hash_source_tree(source_root) != cuda12


def test_compile_or_load_persists_under_source_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aot_cache, "_AOT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(aot_cache, "_AOT_DISABLE", False)
    monkeypatch.setattr(
        aot_cache,
        "_compute_source_fingerprint",
        lambda: "source-fingerprint",
    )
    monkeypatch.setenv("FMHA_SM100_ALLOW_JIT", "1")
    key = ("msa_v1_test_inference_kernel", "config")

    compiled = aot_cache.compile_or_load(key, _FakeCompiled)
    cache_path = Path(aot_cache._key_to_path(key) + ".o")

    assert isinstance(compiled, _FakeCompiled)
    assert cache_path.parent == tmp_path / "source-fingerprint"
    assert cache_path.name.startswith("msa_v1_test_inference_kernel_")
    assert cache_path.read_text(encoding="utf-8") == key[0]
