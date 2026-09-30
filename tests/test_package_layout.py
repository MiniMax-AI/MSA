"""Repository layout and public package contract tests."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import msa_v1

try:
    import tomllib
except ImportError:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"
INFERENCE_ROOT = REPO_ROOT / "inference"
MSA_V1_INFERENCE_ROOT = INFERENCE_ROOT / "msa_v1"
MSA_V1_OPERATOR_ROOTS = (
    MSA_V1_INFERENCE_ROOT / "attention/decode/q8kv4",
    MSA_V1_INFERENCE_ROOT / "attention/decode/q8kv8",
    MSA_V1_INFERENCE_ROOT / "attention/prefill/bf16",
    MSA_V1_INFERENCE_ROOT / "attention/prefill/q8kv4",
    MSA_V1_INFERENCE_ROOT / "attention/prefill/q8kv8",
    MSA_V1_INFERENCE_ROOT / "indexer/decode/q8kv4",
    MSA_V1_INFERENCE_ROOT / "indexer/decode/q8kv8",
    MSA_V1_INFERENCE_ROOT / "indexer/prefill/bf16",
    MSA_V1_INFERENCE_ROOT / "indexer/prefill/q8kv8",
)


def _import_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


def test_training_source_root_preserves_public_package_names() -> None:
    assert not (TRAINING_ROOT / "__init__.py").exists()
    assert msa_v1.__name__ == "msa_v1"
    assert Path(msa_v1.__file__).parent == TRAINING_ROOT / "msa_v1"

    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_find = config["tool"]["setuptools"]["packages"]["find"]
    assert package_find["where"] == ["training", "."]
    assert package_find["namespaces"] is False


def test_local_inference_dumps_are_excluded_from_package_data() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    datas = config["tool"]["setuptools"]["package-data"]["datas"]
    assert "inference/*.jsonl" not in datas
    assert "inference/prefill_cases_v1.jsonl" not in datas
    assert "inference/prefill_cases_v1/*.jsonl" in datas


def test_production_packages_exclude_test_and_benchmark_trees() -> None:
    roots = (TRAINING_ROOT / "msa_v1", INFERENCE_ROOT)
    for root in roots:
        forbidden = [
            path
            for path in root.rglob("*")
            if path.is_dir()
            and path.name in {"test", "tests", "benchmark", "benchmarks"}
        ]
        assert not forbidden


def test_training_and_inference_packages_do_not_cross_import() -> None:
    violations: list[str] = []
    for path in TRAINING_ROOT.rglob("*.py"):
        if "inference" in _import_roots(path):
            violations.append(f"{path.relative_to(REPO_ROOT)} imports inference")
    for path in INFERENCE_ROOT.rglob("*.py"):
        forbidden = _import_roots(path).intersection({"msa_v1", "training"})
        if forbidden:
            violations.append(
                f"{path.relative_to(REPO_ROOT)} imports {', '.join(sorted(forbidden))}"
            )
    assert not violations, "\n".join(violations)


def _top_level_print_lines(path: Path) -> list[int]:
    lines: list[int] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                lines.append(node.lineno)
            self.generic_visit(node)

    Visitor().visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    return lines


def test_production_modules_have_no_import_time_print_calls() -> None:
    violations: list[str] = []
    for root in (TRAINING_ROOT, INFERENCE_ROOT):
        for path in root.rglob("*.py"):
            for lineno in _top_level_print_lines(path):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
    assert not violations, "\n".join(violations)


def test_compile_caches_have_no_manual_abi_version_constants() -> None:
    violations: list[str] = []
    for root in (TRAINING_ROOT, INFERENCE_ROOT):
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                targets: list[ast.expr] = []
                if isinstance(node, ast.Assign):
                    targets.extend(node.targets)
                elif isinstance(node, ast.AnnAssign):
                    targets.append(node.target)
                for target in targets:
                    if isinstance(target, ast.Name) and "ABI_VERSION" in target.id:
                        violations.append(
                            f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {target.id}"
                        )
    assert not violations, "\n".join(violations)


def test_kernel_identifiers_have_no_manual_version_suffixes() -> None:
    pattern = re.compile(r"_sm\d+a?_v\d+\b")
    violations: list[str] = []
    for path in MSA_V1_INFERENCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and pattern.search(node.value)
            ):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {node.value}"
                )
    assert not violations, "\n".join(violations)


def test_msa_v1_inference_layout_is_canonical() -> None:
    msa_v1_root = INFERENCE_ROOT / "msa_v1"
    package_dirs = {
        path.name
        for path in msa_v1_root.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert package_dirs == {"attention", "indexer"}

    forbidden_files = {"kernel.py", "wrapper.py", "launcher.py", "qat.py"}
    assert not [
        path for path in msa_v1_root.rglob("*.py") if path.name in forbidden_files
    ]

    assert (msa_v1_root / "attention/prefill/_common/atten_fwd_sm100.py").is_file()
    assert not (msa_v1_root / "attention/prefill/q8kv8/atten_fwd.py").exists()
    assert (msa_v1_root / "attention/prefill/bf16/interface.py").is_file()
    assert (msa_v1_root / "indexer/decode/q8kv8/indexer_gemm.py").is_file()
    assert (msa_v1_root / "indexer/prefill/q8kv8/indexer_gemm.py").is_file()
    assert (msa_v1_root / "indexer/prefill/bf16/indexer_gemm.py").is_file()


def test_msa_v1_legacy_inference_packages_are_absent() -> None:
    msa_v1_root = INFERENCE_ROOT / "msa_v1"
    legacy_packages = {
        "decode_atten_tp1_q8kv4",
        "decode_indexer_tp4_q8kv4",
        "decode_indexer_tp4_q8kv8",
        "indexer_topk",
        "prefill_atten_tp1_q8kv4",
        "prefill_atten_tp1_q8kv8",
        "prefill_indexer_tp4_q8kv8",
    }
    assert not [
        package for package in legacy_packages if (msa_v1_root / package).exists()
    ]
    assert not [path for path in msa_v1_root.rglob("_vendor") if path.is_dir()]
    assert not [path for path in msa_v1_root.rglob("tp*") if path.is_dir()]


def test_decode_attention_q8kv4_csrc_uses_op_local_flashinfer_layout() -> None:
    csrc = INFERENCE_ROOT / "msa_v1/attention/decode/q8kv4/csrc"
    assert {path.name for path in csrc.iterdir() if path.is_dir()} == {
        "api",
        "include",
        "src",
        "templates",
    }
    assert (csrc / "api/decode_attention_api.cpp").is_file()
    assert (csrc / "api/decode_attention_binding.cpp").is_file()
    assert (csrc / "src/decode_attention_plan.cu").is_file()
    assert (csrc / "src/decode_attention_reduction.cu").is_file()
    assert (csrc / "templates/decode_attention_inst.cu.jinja").is_file()
    assert (csrc / "templates/decode_attention_run.cu.jinja").is_file()
    assert (csrc / "include/sm100/common/nvfp4_to_e4m3.cuh").is_file()
    assert not (csrc / "include/minimax").exists()
    assert not (csrc / "kernel").exists()
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    patterns = config["tool"]["setuptools"]["package-data"][
        "inference.msa_v1.attention.decode.q8kv4"
    ]
    packaged = {path for pattern in patterns for path in csrc.parent.glob(pattern)}
    required = {path for path in csrc.rglob("*") if path.is_file()}
    assert required <= packaged, sorted(str(path) for path in required - packaged)


def test_prefill_attention_q8kv4_csrc_uses_op_local_flashinfer_layout() -> None:
    csrc = INFERENCE_ROOT / "msa_v1/attention/prefill/q8kv4/csrc"
    assert {path.name for path in csrc.iterdir() if path.is_dir()} == {
        "api",
        "include",
        "templates",
    }
    assert (csrc / "api/prefill_attention_api.cpp").is_file()
    assert (csrc / "api/prefill_attention_binding.cpp").is_file()
    assert (csrc / "templates/prefill_attention_inst.cu.jinja").is_file()
    assert (csrc / "include/sm100/common/nvfp4_to_e4m3.cuh").is_file()
    for layer in ("common", "collective", "device", "kernel"):
        assert (csrc / "include/sm100" / layer).is_dir()
    assert not (csrc / "include/minimax").exists()
    assert not (csrc / "kernel").exists()


def test_decode_indexer_q8kv4_csrc_uses_op_local_flashinfer_layout() -> None:
    csrc = INFERENCE_ROOT / "msa_v1/indexer/decode/q8kv4/csrc"
    assert {path.name for path in csrc.iterdir() if path.is_dir()} == {
        "api",
        "include",
        "templates",
    }
    assert (csrc / "api/indexer_gemm_api.cpp").is_file()
    assert (csrc / "api/indexer_gemm_binding.cpp").is_file()
    shared_plan = INFERENCE_ROOT / "msa_v1/indexer/decode"
    assert (shared_plan / "plan.py").is_file()
    assert (shared_plan / "plan_kernel.py").is_file()
    assert (csrc / "templates/indexer_gemm_inst.cu.jinja").is_file()
    assert (csrc / "include/sm100/common/nvfp4_to_e4m3.cuh").is_file()
    for layer in ("common", "collective", "device", "kernel"):
        assert (csrc / "include/sm100" / layer).is_dir()
    assert not (csrc / "include/minimax").exists()
    assert not (csrc / "kernel").exists()


def test_nvfp4_dequant_csrc_uses_shared_cutlass_layout() -> None:
    csrc = INFERENCE_ROOT / "dequant/nvfp4_to_fp8/csrc"
    assert {path.name for path in csrc.iterdir() if path.is_dir()} == {
        "api",
        "include",
        "src",
    }
    assert (csrc / "api/dequant_api.cpp").is_file()
    assert (csrc / "api/dequant_binding.cpp").is_file()
    assert (csrc / "src/dequant.cu").is_file()
    for layer in ("common", "collective", "device", "kernel"):
        assert (csrc / "include/sm100" / layer).is_dir()


def test_q8kv8_decode_adapter_does_not_vendor_flashinfer() -> None:
    root = INFERENCE_ROOT / "msa_v1/attention/decode/q8kv8"
    assert not (root / "csrc").exists()
    assert not [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name in {"third_party", "vendor", "_vendor"}
    ]
    assert not [
        path
        for path in root.rglob("*")
        if path.suffix in {".cubin", ".fatbin", ".ptx", ".so"}
    ]
    dependencies = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["dependencies"]
    assert not [
        dependency for dependency in dependencies if "flashinfer" in dependency.lower()
    ]


def test_msa_v1_operator_readmes_use_one_structure() -> None:
    expected_headings = ["## 功能", "## 公开接口", "## 数据契约", "## 运行约束"]
    development_terms = (
        "PASS",
        "TFLOPS",
        "TB/s",
        "profile",
        "迁移",
    )
    for operator_root in MSA_V1_OPERATOR_ROOTS:
        readme = operator_root / "README.md"
        text = readme.read_text(encoding="utf-8")
        headings = [line for line in text.splitlines() if line.startswith("## ")]
        assert headings in (
            expected_headings,
            [*expected_headings, "## 验收命令"],
        ), readme.relative_to(REPO_ROOT)
        assert not any(term in text for term in development_terms), readme.relative_to(
            REPO_ROOT
        )
        overview = text.partition("## 验收命令")[0]
        assert not any(term in overview for term in ("benchmark", "验收")), (
            readme.relative_to(REPO_ROOT)
        )


def test_msa_v1_tracked_document_style_is_portable() -> None:
    forbidden = ("/home/", "worktrees/")
    legacy_development_names = {
        "optimization.md",
        "validation.md",
        "result.md",
        "results.md",
    }
    tracked_output = subprocess.run(
        ["git", "ls-files", "--", "inference/msa_v1/**/*.md"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tracked_markdown = [REPO_ROOT / path for path in tracked_output.splitlines()]
    assert not [
        path for path in tracked_markdown if path.name in legacy_development_names
    ]

    for path in tracked_markdown:
        if path.name in {"design.md", "development.md"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path.relative_to(
            REPO_ROOT
        )

    gitignore_lines = (
        (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    )
    assert "**/design.md" in gitignore_lines
    assert "**/development.md" in gitignore_lines
