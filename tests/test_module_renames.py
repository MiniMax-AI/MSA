"""Naming, import-boundary, and legacy CLI contracts for module migrations."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAPPING = json.loads((ROOT / "release/module-renames.json").read_text())["files"]


def module_name(path):
    return path.removeprefix("training/").removesuffix(".py").replace("/", ".")


def test_rename_manifest_and_compatibility_boundaries():
    """Aliases must be explicit, one-way, and stay within their original package."""
    assert not set(MAPPING).intersection(MAPPING.values())
    for old, new in MAPPING.items():
        assert (ROOT / new).is_file()
        assert Path(old).parent == Path(new).parent
        assert re.fullmatch(r"[a-z][a-z0-9_]*\.py", Path(new).name)
        assert not Path(new).stem.endswith("_cutedsl")
        assert not Path(new).name.startswith(("m3_indexer", "msa_indexer"))
        tree = ast.parse((ROOT / old).read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
        ]
        assert len(calls) == 1
        assert ast.literal_eval(calls[0].args[0]) == module_name(new)
        assert not any(
            isinstance(n, (ast.ClassDef, ast.FunctionDef)) for n in ast.walk(tree)
        )

    symbols = json.loads((ROOT / "release/module-renames.json").read_text())["symbols"]
    for old, new in symbols.items():
        old_path, old_name = old.split(":")
        new_path, new_name = new.split(":")
        assert old_name == new_name
        assert Path(old_path).parent == Path(new_path).parent
        tree = ast.parse((ROOT / new_path).read_text())
        assert any(
            isinstance(node, ast.ClassDef) and node.name == new_name
            for node in tree.body
        )
        old_tree = ast.parse((ROOT / old_path).read_text())
        assert any(
            isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
            for node in old_tree.body
        )


def test_internal_imports_use_canonical_names():
    """Legacy module imports are permitted only in the compatibility layer."""
    old_modules = {module_name(p) for p in MAPPING}
    violations = []
    for directory in ("training", "inference", "tests", "benchmarks"):
        for path in (ROOT / directory).rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            if relative in MAPPING:
                continue
            if directory in {"training", "inference"}:
                assert not path.stem.endswith("_cutedsl"), relative
                assert not path.stem.startswith(("m3_", "msa_indexer_")), relative
            current = module_name(relative)
            package = (
                current.removesuffix(".__init__")
                if path.name == "__init__.py"
                else current.rsplit(".", 1)[0]
            )
            for node in ast.walk(ast.parse(path.read_text())):
                candidates = []
                if isinstance(node, ast.Import):
                    candidates = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    if node.level:
                        base = importlib.util.resolve_name(
                            "." * node.level + base, package
                        )
                    candidates = [base] + [base + "." + a.name for a in node.names]
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    candidates = [node.value]
                for name in candidates:
                    if name in old_modules:
                        violations.append((relative, node.lineno, name))
    assert not violations, violations


@pytest.mark.parametrize(
    "old,new",
    [
        pair
        for pair in MAPPING.items()
        if Path(pair[0]).name
        in {"bench_cpp_baseline.py", "benchmark.py", "bench_kl_v1_topk_pathologies.py"}
    ],
)
def test_legacy_cli_help_and_errors(old, new):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    for args in (["--help"], ["--msa-invalid-option"]):
        outputs = []
        for path in (old, new):
            result = subprocess.run(
                [sys.executable, str(ROOT / path), *args],
                cwd=ROOT.parent,
                text=True,
                capture_output=True,
                timeout=60,
                env=env,
                check=False,
            )
            assert result.returncode == (0 if args == ["--help"] else 2), result.stderr
            # argparse derives the executable name from argv[0].
            output = re.sub(r"[\w-]+\.py", "<command>", result.stdout + result.stderr)
            outputs.append((result.returncode, " ".join(output.split())))
        assert outputs[0] == outputs[1]
