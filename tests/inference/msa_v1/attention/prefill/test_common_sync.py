"""CPU-only source-sync and production dependency gates."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
PREFILL_ROOT = REPO_ROOT / "inference/msa_v1/attention/prefill"
COMMON_ROOT = PREFILL_ROOT / "_common"
MANIFEST = COMMON_ROOT / "vendor_manifest.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _expected_common_text(entry: dict[str, str]) -> str:
    manifest = _manifest()
    source_path = entry["source"]
    source = REPO_ROOT / manifest["source_root"] / source_path
    text = source.read_text()
    for old, new in manifest["namespace_rewrites"]:
        text = text.replace(old, new)
    for replacement in manifest["recorded_replacements"].get(source_path, []):
        assert text.count(replacement["source"]) == 1
        text = text.replace(replacement["source"], replacement["destination"])
    return text


def test_manifest_covers_every_shared_python_file() -> None:
    manifest = _manifest()
    actual = {
        path.relative_to(COMMON_ROOT).as_posix() for path in COMMON_ROOT.rglob("*.py")
    }
    expected = (
        {entry["destination"] for entry in manifest["files"]}
        | set(manifest["organization_files"])
        | set(manifest["local_implementation_files"])
    )
    assert actual == expected


def test_shared_files_match_recorded_training_transform() -> None:
    manifest = _manifest()
    mismatches = [
        entry["destination"]
        for entry in manifest["files"]
        if (COMMON_ROOT / entry["destination"]).read_text()
        != _expected_common_text(entry)
    ]
    assert not mismatches, f"shared sources drifted: {mismatches}"


def test_production_has_no_training_or_bare_msa_v1_imports() -> None:
    violations = []
    for path in PREFILL_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules.append(node.module)
            for module in modules:
                if module == "training" or module.startswith("training."):
                    violations.append((path, node.lineno, module))
                if module == "msa_v1" or module.startswith("msa_v1."):
                    violations.append((path, node.lineno, module))
    assert not violations


def test_production_has_no_device_to_host_metadata_reads() -> None:
    forbidden = (".item(", ".cpu(", ".tolist(", ".numpy(")
    violations = []
    for path in PREFILL_ROOT.rglob("*.py"):
        text = path.read_text()
        for token in forbidden:
            if token in text:
                violations.append((path, token))
    assert not violations


def test_package_import_does_not_load_training_namespace() -> None:
    script = f"""
import importlib.abc
import pathlib
import sys

repo = pathlib.Path({str(REPO_ROOT)!r})
training = (repo / 'training').resolve()
sys.path = [str(repo)] + [
    path for path in sys.path
    if path and pathlib.Path(path).resolve() != training
]

class ForbiddenImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'training' or fullname.startswith('training.'):
            raise AssertionError(f'forbidden import: {{fullname}}')
        if fullname == 'msa_v1' or fullname.startswith('msa_v1.'):
            raise AssertionError(f'forbidden import: {{fullname}}')
        return None

sys.meta_path.insert(0, ForbiddenImportFinder())
from inference.msa_v1.attention.prefill.q8kv8 import (
    BatchPrefillWithPagedKVCacheWrapper,
)
assert BatchPrefillWithPagedKVCacheWrapper.__name__ == (
    'BatchPrefillWithPagedKVCacheWrapper'
)
from inference.msa_v1.attention.prefill.q8kv4 import (
    BatchPrefillWithPagedKVCacheWrapper as Q8KV4Wrapper,
)
assert Q8KV4Wrapper.__name__ == 'BatchPrefillWithPagedKVCacheWrapper'
"""
    subprocess.run([sys.executable, "-c", script], check=True, cwd=REPO_ROOT)


def test_prefill_architecture_dispatch(monkeypatch) -> None:
    """Select Rubin options from the tensor device without changing Blackwell defaults."""
    import pytest
    import torch

    from inference.msa_v1.attention.prefill._common.atten_fwd_sm100 import (
        _check_architecture,
        _resolve_rubin_options,
    )
    from inference.msa_v1.attention.prefill.q8kv8 import (
        BatchPrefillWithPagedKVCacheWrapper,
    )

    device = torch.device("cuda:1")
    for capability in ((10, 0), (10, 3), (10, 7)):

        def get_capability(actual_device):
            assert actual_device == device
            return capability

        monkeypatch.setattr(torch.cuda, "get_device_capability", get_capability)
        assert _check_architecture(device) == capability
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            supported = capability == (10, 7) and dtype == torch.float8_e4m3fn
            assert _resolve_rubin_options(capability, dtype, None, None) == (
                supported,
                supported,
            )
            assert _resolve_rubin_options(capability, dtype, False, False) == (
                False,
                False,
            )
            if not supported:
                with pytest.raises(NotImplementedError, match="SM107"):
                    _resolve_rubin_options(capability, dtype, True, None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    with pytest.raises(RuntimeError, match="SM90"):
        _check_architecture(device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    assert wrapper.enable_fp16_softmax is None
    assert wrapper.enable_2x_fp8 is None
    with pytest.raises(TypeError, match="bool or None"):
        BatchPrefillWithPagedKVCacheWrapper(enable_2x_fp8=1)
