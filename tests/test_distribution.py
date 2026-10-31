"""CPU-only checks for relocatable wheels and pinned JIT resources."""

import importlib.util
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def build_utils():
    spec = importlib.util.spec_from_file_location(
        "msa_build_utils", ROOT / "inference/msa_v1/_build_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cutlass_resources_and_overrides(tmp_path, monkeypatch, build_utils):
    bundled = tmp_path / "site-packages/inference/_vendor/cutlass"
    override = tmp_path / "custom-cutlass"
    for root in (bundled, override):
        header = root / "include/cutlass/cutlass.h"
        header.parent.mkdir(parents=True)
        header.write_text("header")
    monkeypatch.setattr(build_utils, "_BUNDLED_CUTLASS", bundled)
    monkeypatch.setattr(build_utils, "_REPOSITORY_ROOT", tmp_path / "absent")
    monkeypatch.delenv("CUTLASS_ROOT", raising=False)
    monkeypatch.delenv("CUTLASS_PATH", raising=False)
    assert build_utils.cutlass_root() == bundled
    first = build_utils.cutlass_fingerprint()
    (bundled / "include/cutlass/cutlass.h").write_text("changed header")
    build_utils.cutlass_fingerprint.cache_clear()
    assert build_utils.cutlass_fingerprint() != first
    monkeypatch.setenv("CUTLASS_ROOT", str(override))
    build_utils.cutlass_root.cache_clear()
    assert build_utils.cutlass_root() == override


def test_missing_cutlass_reports_installation_remedy(
    tmp_path, monkeypatch, build_utils
):
    monkeypatch.setattr(build_utils, "_BUNDLED_CUTLASS", tmp_path / "absent")
    monkeypatch.setattr(build_utils, "_REPOSITORY_ROOT", tmp_path / "absent")
    monkeypatch.delenv("CUTLASS_ROOT", raising=False)
    monkeypatch.delenv("CUTLASS_PATH", raising=False)
    with pytest.raises(RuntimeError, match="Reinstall the complete MSA wheel"):
        build_utils.cutlass_root()


def test_cutlass_pin_detects_modified_headers(tmp_path, monkeypatch):
    build = runpy.run_path(str(ROOT / "setup.py"))
    verify = build["verify_cutlass"]
    verify()
    root = tmp_path / "cutlass"
    root.mkdir()
    (root / "LICENSE.txt").write_text("altered dependency")
    monkeypatch.setitem(verify.__globals__, "CUTLASS", root)
    with pytest.raises(RuntimeError, match="differ from release/cutlass.json"):
        verify()


def test_wheel_contains_all_production_sources():
    """The artifact path is explicit so source-only runs cannot fake acceptance."""
    import os
    import zipfile

    wheel = os.environ.get("MSA_TEST_WHEEL")
    if not wheel:
        pytest.skip("Set MSA_TEST_WHEEL to validate an actual built wheel")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert not any({"msa_v2", "high_tps"}.intersection(Path(n).parts) for n in names)
        for source_root, prefix in (
            (ROOT / "training/msa_v1", "msa_v1"),
            (ROOT / "inference", "inference"),
        ):
            for source in source_root.rglob("*"):
                relative = source.relative_to(source_root)
                if "__pycache__" in relative.parts or not source.is_file():
                    continue
                if source.suffix in {
                    ".py",
                    ".cpp",
                    ".cu",
                    ".hpp",
                    ".cuh",
                    ".h",
                    ".jinja",
                }:
                    assert f"{prefix}/{relative.as_posix()}" in names
        pin = json.loads(archive.read("inference/_vendor/cutlass/provenance.json"))
        assert pin == json.loads((ROOT / "release/cutlass.json").read_text())
        for suffix in (
            "licenses/LICENSE",
            "licenses/NOTICE",
            "licenses/licenses/CUTLASS-LICENSE.txt",
        ):
            assert any(n.endswith(".dist-info/" + suffix) for n in names)
