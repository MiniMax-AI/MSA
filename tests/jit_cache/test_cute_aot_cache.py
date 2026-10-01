# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""The CuTe DSL AOT cache serves an exported kernel only while the modules it is generated
from are unchanged: the import closure, within cute/, of the module that defines it."""

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("cutlass")

CUTE = Path(__file__).resolve().parents[2] / "python" / "fmha_sm100" / "cute"
if str(CUTE) not in sys.path:
    sys.path.insert(0, str(CUTE))
aot_cache = importlib.import_module("src.common.aot_cache")


def test_the_closure_follows_the_kernel_module_imports():
    closure = {str(path.relative_to(CUTE))
               for path in aot_cache._import_closure([CUTE / "src/sm100/fwd/combine.py"])}
    assert {"src/sm100/fwd/combine.py", "src/common/utils.py", "src/common/seqlen_info.py",
            "src/__init__.py", "src/common/__init__.py"} <= closure
    assert "src/sm100/fwd/atten_fwd.py" not in closure, "another kernel"
    assert "src/common/aot_cache.py" not in closure, "the cache does not generate code"


class _Compiled:
    def export_to_c(self, path, function_name):
        Path(path).write_bytes(f"object of {function_name}".encode())


@pytest.fixture
def package(tmp_path, monkeypatch):
    root = tmp_path / "cute"
    (root / "src" / "common").mkdir(parents=True)
    (root / "src" / "__init__.py").write_text("")
    (root / "src" / "common" / "__init__.py").write_text("")
    (root / "src" / "common" / "helpers.py").write_text("X = 1\n")
    (root / "src" / "kernel_a.py").write_text("from src.common import helpers\n")
    (root / "src" / "kernel_b.py").write_text("from .common.helpers import X\n")
    monkeypatch.setattr(aot_cache, "_CUTE_ROOT", root)
    monkeypatch.setattr(aot_cache, "_AOT_CACHE_DIR", str(tmp_path / "aot"))
    monkeypatch.setattr(aot_cache, "_AOT_DISABLE", False)
    monkeypatch.setattr(aot_cache, "_file_hashes", {})
    return root


def _fresh(monkeypatch):
    monkeypatch.setattr(aot_cache, "_file_hashes", {})


def test_an_object_is_served_while_its_closure_is_unchanged(package, monkeypatch):
    key_a, key_b = ("kernel_a", 128), ("kernel_b", 128)
    aot_cache.save_aot(key_a, _Compiled(), sources=["src/kernel_a.py"])
    aot_cache.save_aot(key_b, _Compiled(), sources=["src/kernel_b.py"])
    assert aot_cache.aot_object_path(key_a) and aot_cache.aot_object_path(key_b)
    assert aot_cache.aot_object_path(("kernel_a", 64)) == "", "another compile key"

    (package / "src" / "kernel_b.py").write_text("from .common.helpers import X\nY = 2\n")
    _fresh(monkeypatch)
    assert aot_cache.aot_object_path(key_a), "kernel_b's module is not in kernel_a's closure"
    assert aot_cache.aot_object_path(key_b) == ""

    (package / "src" / "common" / "helpers.py").write_text("X = 3\n")
    _fresh(monkeypatch)
    assert aot_cache.aot_object_path(key_a) == "", "both kernels import the helper"
    aot_cache.save_aot(key_a, _Compiled(), sources=["src/kernel_a.py"])
    assert aot_cache.aot_object_path(key_a)


def test_unreadable_entries_are_misses(package):
    key = ("kernel_a", 1)
    aot_cache.save_aot(key, _Compiled(), sources=["src/kernel_a.py"])
    Path(aot_cache._key_to_path(key) + ".json").write_text("{not json")
    assert aot_cache.aot_object_path(key) == ""
    assert aot_cache.try_load_aot(key) is None


def test_the_compile_call_site_is_an_input(package, monkeypatch):
    """The module that calls save_aot builds the tensors and options cute.compile sees."""
    frontend = package / "frontend.py"
    frontend.write_text("def compile_a(aot, key, compiled):\n"
                        "    aot.save_aot(key, compiled, sources=['src/kernel_a.py'])\n")
    spec = importlib.util.spec_from_file_location("frontend", frontend)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    key = ("kernel_a", 7)
    module.compile_a(aot_cache, key, _Compiled())
    assert aot_cache.aot_object_path(key)
    frontend.write_text(frontend.read_text() + "# alignment changed\n")
    _fresh(monkeypatch)
    assert aot_cache.aot_object_path(key) == ""
