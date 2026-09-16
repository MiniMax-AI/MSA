"""MSA v1 public API and production-path contract tests."""

import ast
import inspect
from pathlib import Path

import cutlass
from packaging.version import Version

import msa_v1
from msa_v1.attention import prepare_scheduler
from msa_v1.attention.interface import sparse_atten_func


def test_public_api_is_training_only() -> None:
    assert Version(str(cutlass.__version__)) >= Version("4.5.2")
    assert msa_v1.__all__ == ["attention", "indexer", "indexer_tree", "kl"]
    assert msa_v1.attention.__all__ == [
        "AttentionMetadata",
        "backward",
        "forward",
        "prepare",
    ]
    assert msa_v1.indexer.__all__ == [
        "IndexerForwardWorkspace",
        "IndexerSchedule",
        "allocate_indexer_schedule",
        "allocate_indexer_workspace",
        "forward",
        "prepare_indexer_schedule",
    ]
    assert msa_v1.indexer_tree.__all__ == ["IndexerPlan", "compile_plan", "forward"]
    assert inspect.signature(msa_v1.indexer.forward).parameters[
        "use_fp16_score"
    ].default is False
    assert inspect.signature(msa_v1.indexer_tree.forward).parameters[
        "use_fp16_score"
    ].default is False
    local_block_positions = inspect.signature(msa_v1.indexer_tree.compile_plan).parameters[
        "local_block_positions"
    ]
    assert local_block_positions.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert local_block_positions.default is None
    block_bases = inspect.signature(msa_v1.indexer_tree.forward).parameters[
        "block_bases"
    ]
    assert block_bases.kind is inspect.Parameter.KEYWORD_ONLY
    assert block_bases.default is None
    forbidden = {
        "causal",
        "schedule",
        "page_table",
        "seqused_k",
        "qk_dtype",
        "pv_dtype",
    }
    for public_fn in (
        msa_v1.attention.prepare,
        msa_v1.attention.forward,
        msa_v1.attention.backward,
    ):
        assert forbidden.isdisjoint(inspect.signature(public_fn).parameters)


def test_package_has_no_src_imports_or_inference_modules() -> None:
    package_root = Path(msa_v1.__file__).parent
    source_roots = tuple(
        package_root / name
        for name in ("_common", "attention", "indexer", "indexer_tree", "kl")
    )
    source_files = list(package_root.glob("*.py"))
    for root in source_roots:
        source_files.extend(path for path in root.rglob("*") if path.is_file())
    native_suffixes = {".cu", ".cuh", ".cpp", ".cc", ".cxx"}
    assert not any(path.suffix in native_suffixes for path in source_files)
    for path in (path for path in source_files if path.suffix == ".py"):
        assert "torch.utils.cpp_extension" not in path.read_text()
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("src")
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("src") for alias in node.names)
    assert not (package_root / "fwd_decode").exists()
    assert not (package_root / "attention" / "fwd" / "atten_fwd_nvfp4_kv.py").exists()


def test_production_modules_have_no_direct_print_calls() -> None:
    package_root = Path(msa_v1.__file__).parent
    violations = []
    for path in package_root.rglob("*.py"):
        relative = path.relative_to(package_root)
        if not path.stem.isidentifier() or any(
            not part.isidentifier() for part in relative.parts[:-1]
        ):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                violations.append(f"{path}:{node.lineno}: print")
            if isinstance(node.func, ast.Attribute) and node.func.attr == "printf":
                violations.append(f"{path}:{node.lineno}: printf")
    assert not violations, "\n".join(violations)


def test_attention_core_requires_the_prepared_schedule() -> None:
    schedule = inspect.signature(sparse_atten_func).parameters["schedule"]

    assert schedule.default is inspect.Parameter.empty
    assert prepare_scheduler.__all__ == [
        "SPARSE_SCHEDULE_MODEL",
        "SparseAttentionSchedule",
        "SparseAttentionScheduleModel",
    ]
    assert not hasattr(prepare_scheduler, "prepare_sparse_fwd_schedule")
    assert not hasattr(prepare_scheduler, "prepare_sparse_fwd_schedule_and_split")


_BANNED_D2H_METHODS = {"item", "cpu", "tolist", "numpy"}
_OFFLINE_APIS = {"compile_plan", "compile_m3_arbitrary_mask_plan"}
_BANNED_COMPILE_KEY_NAMES = {
    "batch",
    "batch_count",
    "batch_size",
    "cu_seqlens",
    "cu_seqlens_k",
    "cu_seqlens_kv",
    "cu_seqlens_q",
    "max_seqlen_k",
    "max_seqlen_kv",
    "max_seqlen_q",
    "nnz",
    "seqlen_k",
    "seqlen_kv",
    "seqlen_q",
    "total_k",
    "total_k_padded",
    "total_kv",
    "total_q",
    "total_q_padded",
    "total_rows",
    "work_capacity",
    "work_count",
}
_CODEGEN_STATIC_SHAPE_INDICES = {
    "D": {-1},
    "head_dim": {-1},
    "head_q": {1},
    "num_splits": {0},
    "page_size": {2},
}
_CODEGEN_STATIC_PRESENCE_NAMES = {"has_cu_seqlens"}
_COMPILE_KEY_CALL_NAMES = {
    "compile_or_load",
    "save_aot",
    "try_load_aot",
}


class _D2hVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.functions: list[str] = []
        self.violations: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call) -> None:
        method = node.func.attr if isinstance(node.func, ast.Attribute) else None
        offline = any(name in _OFFLINE_APIS for name in self.functions)
        if method in _BANNED_D2H_METHODS and not offline:
            self.violations.append(f"{self.path}:{node.lineno}: production .{method}()")
        if self._qualified_name(node.func) == "torch.cuda.synchronize" and not offline:
            self.violations.append(
                f"{self.path}:{node.lineno}: production torch.cuda.synchronize()"
            )
        self.generic_visit(node)

    @staticmethod
    def _qualified_name(node: ast.expr) -> str:
        names = []
        while isinstance(node, ast.Attribute):
            names.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            names.append(node.id)
        return ".".join(reversed(names))


def _assigned_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    else:
        return set()
    return {
        child.id
        for target in targets
        for child in ast.walk(target)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    }


def _assignment_value(node: ast.AST) -> ast.expr | None:
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return node.value
    return None


def _expression_runtime_references(
    node: ast.AST,
    *,
    assigned_name: str | None,
) -> set[str]:
    violations = {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
        and child.id in _BANNED_COMPILE_KEY_NAMES
    }
    violations.update(
        child.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and child.attr in _BANNED_COMPILE_KEY_NAMES
    )
    if assigned_name in _CODEGEN_STATIC_PRESENCE_NAMES:
        violations.clear()

    allowed_indices = _CODEGEN_STATIC_SHAPE_INDICES.get(assigned_name, set())
    for child in ast.walk(node):
        if not isinstance(child, ast.Subscript):
            continue
        if not (
            isinstance(child.value, ast.Attribute)
            and child.value.attr == "shape"
        ):
            continue
        index = child.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, int):
            index_value = index.value
        elif (
            isinstance(index, ast.UnaryOp)
            and isinstance(index.op, ast.USub)
            and isinstance(index.operand, ast.Constant)
            and isinstance(index.operand.value, int)
        ):
            index_value = -index.operand.value
        else:
            index_value = None
        if index_value == -1 or index_value in allowed_indices:
            continue
        violations.add("runtime tensor shape")
    return violations


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _is_compile_cache(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, ast.Attribute):
        name = node.attr
    else:
        return False
    return "compile_cache" in name.lower()


def _actual_compile_key_expressions(function: ast.AST) -> list[ast.expr]:
    expressions = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and node.args:
            if _call_name(node) in _COMPILE_KEY_CALL_NAMES:
                expressions.append(node.args[0])
            elif (
                _call_name(node) == "get"
                and isinstance(node.func, ast.Attribute)
                and _is_compile_cache(node.func.value)
            ):
                expressions.append(node.args[0])
        if isinstance(node, ast.Subscript) and _is_compile_cache(node.value):
            expressions.append(node.slice)
    return expressions


def _compile_key_violations(tree: ast.AST, path: Path) -> list[str]:
    violations = []
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        key_expressions = _actual_compile_key_expressions(function)
        if not key_expressions:
            continue

        assignments: dict[str, list[ast.expr]] = {}
        for node in ast.walk(function):
            value = _assignment_value(node)
            if value is None:
                continue
            for name in _assigned_names(node):
                assignments.setdefault(name, []).append(value)

        pending = [(expression, None) for expression in key_expressions]
        expanded_names = set()
        while pending:
            expression, assigned_name = pending.pop()
            runtime_references = sorted(
                _expression_runtime_references(
                    expression,
                    assigned_name=assigned_name,
                )
            )
            if runtime_references:
                details = ", ".join(runtime_references)
                if assigned_name is not None:
                    details = f"{assigned_name} <- {details}"
                violations.append(
                    f"{path}:{expression.lineno}: {details}"
                )
            for child in ast.walk(expression):
                if not isinstance(child, ast.Name):
                    continue
                name = child.id
                if name in expanded_names or name not in assignments:
                    continue
                expanded_names.add(name)
                pending.extend((value, name) for value in assignments[name])
    return sorted(set(violations))


def test_training_modules_do_not_synchronize_device_metadata_to_host() -> None:
    repo = Path(__file__).resolve().parents[4]
    roots = (
        repo / "training" / "msa_v1" / "_common",
        repo / "training" / "msa_v1" / "attention",
        repo / "training" / "msa_v1" / "indexer",
        repo / "training" / "msa_v1" / "indexer_tree",
        repo / "training" / "msa_v1" / "kl",
    )
    violations = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            visitor = _D2hVisitor(path.relative_to(repo))
            visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
            violations.extend(visitor.violations)
    assert not violations, "\n".join(violations)


def test_compile_keys_only_contain_codegen_static_configuration() -> None:
    package_root = Path(msa_v1.__file__).parent
    violations = []
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(_compile_key_violations(tree, path))
    assert not violations, "\n".join(violations)


def test_compile_key_contract_detects_alias_attribute_and_augassign() -> None:
    source = """
def compile_bad(args, q):
    total_rows_alias = args.total_q
    compile_key = ("bad",)
    compile_key += (total_rows_alias, q.shape[0])
    compile_or_load(compile_key, lambda: None)

def compile_good(q):
    key = q.shape[0]
    compile_key = ("static",)
    compile_or_load(compile_key, lambda: None)

def compile_static(q):
    compile_key = ("static", q.shape[-1])
    compile_or_load(compile_key, lambda: None)
"""
    violations = _compile_key_violations(ast.parse(source), Path("fixture.py"))
    assert violations == [
        "fixture.py:3: total_rows_alias <- total_q",
        "fixture.py:5: compile_key <- runtime tensor shape",
    ]
