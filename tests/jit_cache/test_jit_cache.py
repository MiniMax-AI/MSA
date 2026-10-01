# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""fmha_sm100._jit_cache serves a library only while the files it was built from are unchanged.

GPU-free: the libraries are toy C++ builds (host compiler + ninja, the JIT's own rule shape), so
these run anywhere the JIT could build. Header edits keep the file's mtime, the case timestamps
cannot catch (a checkout or an install that preserves mtimes).
"""

import json
import os
import shutil

import pytest

from fmha_sm100 import _jit_cache

pytestmark = pytest.mark.skipif(not (shutil.which("ninja") and shutil.which("c++")),
                                reason="needs ninja and a host C++ compiler")


def _edit(path, text):
    """Change a file's content but keep its mtime."""
    stat = path.stat()
    path.write_text(text)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def _new_process():
    _jit_cache.clear_memo()


class Toy:
    """A recipe whose library links one object per source, compiled with -MMD like the JIT's."""

    def __init__(self, namespace, name, sources, includes, flags="-O0"):
        self.recipe = _jit_cache.Recipe(namespace, name, {"flags": flags})
        self.sources = sources
        self.includes = " ".join(f"-I{path}" for path in includes)
        self.flags = flags
        self.builds = 0

    def _builder(self, build_dir):
        self.builds += 1
        library = build_dir / f"{self.recipe.name}.so"
        objects = [(build_dir / f"{source.stem}.o", source) for source in self.sources]
        compiles = "\n".join(f"build {obj}: cc {source}" for obj, source in objects)
        (build_dir / "build.ninja").write_text(f"""rule cc
  command = c++ {self.flags} -fPIC {self.includes} -MMD -MF $out.d -c $in -o $out
  depfile = $out.d
  deps = gcc
rule link
  command = c++ -shared $in -o $out
{compiles}
build {library}: link {" ".join(str(obj) for obj, _ in objects)}
""")
        _jit_cache.run_ninja(build_dir, self.recipe.name)
        return library

    def build(self):
        return self.recipe.build(self._builder)

    def entry(self):
        return json.loads((self.recipe.dir / "entry.json").read_text())


@pytest.fixture
def tree(tmp_path):
    src = tmp_path / "src"
    include = src / "include"
    include.mkdir(parents=True)
    (include / "a.h").write_text("constexpr int a = 1;\n")
    (include / "b.h").write_text("constexpr int b = 2;\n")
    (src / "x.cc").write_text('#include "a.h"\nint x() { return a; }\n')
    (src / "y.cc").write_text('#include "b.h"\nint y() { return b; }\n')
    _new_process()
    yield src, include, _jit_cache.Namespace(tmp_path / "cache", cuda_home=None)
    _new_process()


def test_a_library_is_reused_by_later_processes(tree):
    src, include, namespace = tree
    toy = Toy(namespace, "x", [src / "x.cc"], [include])
    library = toy.build()
    assert library.is_file() and toy.builds == 1
    assert (namespace.path / "manifest.json").is_file()
    inputs = toy.entry()["records"][0]["inputs"]
    assert {os.path.basename(path) for path in inputs} == {"x.cc", "a.h"}
    _new_process()
    assert toy.recipe.lookup() == library
    assert toy.build() == library and toy.builds == 1


def test_editing_a_header_rebuilds_only_the_libraries_that_include_it(tree):
    src, include, namespace = tree
    x = Toy(namespace, "x", [src / "x.cc"], [include])
    y = Toy(namespace, "y", [src / "y.cc"], [include])
    x_library, y_library = x.build(), y.build()
    _edit(include / "a.h", "constexpr int a = 3;\n")
    _new_process()
    assert x.recipe.lookup() is None
    assert y.recipe.lookup() == y_library
    rebuilt = x.build()
    assert rebuilt != x_library and x.builds == 2
    assert y.build() == y_library and y.builds == 1


def test_earlier_sources_find_their_library_again(tree):
    src, include, namespace = tree
    x = Toy(namespace, "x", [src / "x.cc"], [include])
    original = x.build()
    _edit(include / "a.h", "constexpr int a = 3;\n")
    _new_process()
    edited = x.build()
    _edit(include / "a.h", "constexpr int a = 1;\n")  # back to the first version
    _new_process()
    assert x.recipe.lookup() == original and original.is_file()
    _edit(include / "a.h", "constexpr int a = 3;\n")
    _new_process()
    assert x.recipe.lookup() == edited
    assert x.builds == 2


def test_only_objects_whose_inputs_changed_recompile(tree):
    src, include, namespace = tree
    xy = Toy(namespace, "xy", [src / "x.cc", src / "y.cc"], [include])
    xy.build()
    x_obj, y_obj = xy.recipe.dir / "x.o", xy.recipe.dir / "y.o"
    x_before, y_before = x_obj.stat(), y_obj.stat()
    _edit(include / "b.h", "constexpr int b = 4;\n")  # same mtime: ninja alone keeps y.o
    _new_process()
    xy.build()
    assert xy.builds == 2
    assert x_obj.stat().st_mtime_ns == x_before.st_mtime_ns, "x.o does not include b.h"
    assert y_obj.stat().st_ino != y_before.st_ino or \
        y_obj.stat().st_mtime_ns != y_before.st_mtime_ns, "y.o includes the edited b.h"


def test_unreadable_or_foreign_entries_are_misses(tree):
    src, include, namespace = tree
    x = Toy(namespace, "x", [src / "x.cc"], [include])
    x.build()
    entry = x.recipe.dir / "entry.json"
    entry.write_text("{not json")
    _new_process()
    assert x.recipe.lookup() is None
    x.build()
    foreign = x.entry()
    foreign["recipe"]["key"] = {"flags": "-O3"}
    entry.write_text(json.dumps(foreign))
    _new_process()
    assert x.recipe.lookup() is None
    assert x.build().is_file() and x.builds == 3


def test_each_recipe_has_its_own_directory(tree):
    src, include, namespace = tree
    o0 = Toy(namespace, "x", [src / "x.cc"], [include], flags="-O0")
    o1 = Toy(namespace, "x", [src / "x.cc"], [include], flags="-O1")
    assert o0.recipe.dir != o1.recipe.dir
    first, second = o0.build(), o1.build()
    _new_process()
    assert o0.recipe.lookup() == first and o1.recipe.lookup() == second


def test_toolchain_headers_are_left_to_the_namespace(tree, tmp_path):
    src, include, _ = tree
    toolkit = tmp_path / "toolkit"
    (toolkit / "include").mkdir(parents=True)
    (toolkit / "include" / "t.h").write_text("constexpr int t = 5;\n")
    (src / "z.cc").write_text('#include "t.h"\n#include "a.h"\nint z() { return t + a; }\n')
    namespace = _jit_cache.Namespace(tmp_path / "cache", cuda_home=toolkit)
    assert namespace.manifest["nvcc"] == str(toolkit.resolve() / "bin" / "nvcc")
    z = Toy(namespace, "z", [src / "z.cc"], [include, toolkit / "include"])
    library = z.build()
    assert all("toolkit" not in path for path in z.entry()["records"][0]["inputs"])
    _edit(toolkit / "include" / "t.h", "constexpr int t = 6;\n")
    _new_process()
    assert z.recipe.lookup() == library


def test_toolchains_get_separate_namespaces(tmp_path):
    plain = _jit_cache.Namespace(tmp_path, cuda_home=None)
    other = _jit_cache.Namespace(tmp_path, cuda_home=None, extra={"compiler": "other"})
    assert plain.path != other.path and plain.path.parent == other.path.parent


def test_records_and_libraries_are_bounded(tree):
    src, include, namespace = tree
    x = Toy(namespace, "x", [src / "x.cc"], [include])
    for value in range(_jit_cache.MAX_RECORDS + 2):
        _edit(include / "a.h", f"constexpr int a = {value + 10};\n")
        _new_process()
        x.build()
    assert len(x.entry()["records"]) == _jit_cache.MAX_RECORDS
    assert len(list(x.recipe.dir.glob("x-*.so"))) == _jit_cache.MAX_RECORDS
