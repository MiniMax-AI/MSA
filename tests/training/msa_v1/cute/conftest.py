"""Pytest configuration for the MSA v1 CuTe release suites."""

from __future__ import annotations

import inspect
import os
import time
from collections.abc import Iterator

import pytest
from packaging.version import Version

os.environ.setdefault("FMHA_SM100_ALLOW_JIT", "1")

from datas.training.cases import metadata_digest
from tests.training.msa_v1.cute.cases import (
    EXPECTED_MANIFEST_DIGEST,
    FULL_CASE_COUNT,
    SMOKE_CASE_COUNT,
    full_manifest,
    manifest_digest,
    smoke_manifest,
    validate_manifest,
)

EXPECTED_METADATA_DIGEST = (
    "116faa50d1a8de18aaab87cd8ff0179750a942802a62d131ec31dcdc990800d7"
)
_SESSION_STARTED = time.monotonic()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("msa-v1")
    group.addoption(
        "--msa-v1-suite",
        choices=("full", "smoke"),
        default="full",
        help=f"run the {FULL_CASE_COUNT}-case manifest or {SMOKE_CASE_COUNT}-case smoke subset",
    )
    group.addoption("--msa-v1-case-id", type=int)


def pytest_configure(config: pytest.Config) -> None:
    config._msa_v1_started = _SESSION_STARTED  # type: ignore[attr-defined]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "msa_case" not in metafunc.fixturenames:
        return
    case_id = metafunc.config.getoption("--msa-v1-case-id")
    if case_id is not None:
        if not 0 <= case_id < FULL_CASE_COUNT:
            raise pytest.UsageError(
                f"--msa-v1-case-id must be in [0, {FULL_CASE_COUNT})"
            )
        cases = (full_manifest()[case_id],)
    elif metafunc.config.getoption("--msa-v1-suite") == "smoke":
        cases = smoke_manifest()
    else:
        cases = full_manifest()
    metafunc.parametrize("msa_case", cases, ids=lambda item: item.pytest_id)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    elapsed = time.monotonic() - session.config._msa_v1_started  # type: ignore[attr-defined]
    suite = session.config.getoption("--msa-v1-suite")
    limit = 300.0 if suite == "smoke" else 3600.0
    print(f"MSA v1 {suite} suite elapsed: {elapsed:.1f}s (limit {limit:.0f}s)")
    if exitstatus == pytest.ExitCode.OK and elapsed > limit:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        print(f"MSA v1 {suite} suite exceeded its runtime limit")


def _validate_public_api() -> None:
    import msa_v1
    from msa_v1 import attention, indexer, indexer_tree, kl

    assert msa_v1.__all__ == ["attention", "indexer", "indexer_tree", "kl"]
    assert attention.__all__ == ["AttentionMetadata", "backward", "forward", "prepare"]
    assert indexer_tree.__all__ == ["IndexerPlan", "compile_plan", "forward"]
    prepare_parameters = inspect.signature(attention.prepare).parameters
    assert "fragment_indices" in prepare_parameters
    for function in (attention.forward, attention.backward):
        assert "metadata" in inspect.signature(function).parameters
        assert "deterministic" in inspect.signature(function).parameters
    indexer_parameters = inspect.signature(indexer.forward).parameters
    assert "cu_seqlens_q" in indexer_parameters
    assert "cu_seqlens_kv" in indexer_parameters
    assert "fragment_indices" in indexer_parameters
    assert "schedule" in indexer_parameters
    assert "lse_temperature" in indexer_parameters
    assert "deterministic" in indexer_parameters
    assert indexer_parameters["use_fp16_score"].default is False
    tree_indexer_parameters = inspect.signature(indexer_tree.forward).parameters
    assert "lse_temperature" in tree_indexer_parameters
    assert "deterministic" in tree_indexer_parameters
    assert tree_indexer_parameters["use_fp16_score"].default is False
    assert kl.__all__ == ["backward"]
    assert "deterministic" in inspect.signature(kl.backward).parameters


@pytest.fixture(scope="session", autouse=True)
def release_preflight() -> Iterator[None]:
    """Validate hardware, data, manifest, and public API once."""

    import cutlass
    import torch

    if not torch.cuda.is_available():
        pytest.skip("MSA v1 release tests require CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(device)
    if capability not in {(10, 0), (10, 3)}:
        pytest.skip(f"MSA v1 requires SM100/SM103, got SM{capability}")
    if Version(str(cutlass.__version__)) < Version("4.5.2"):
        raise AssertionError(f"CuTe DSL >=4.5.2 is required, got {cutlass.__version__}")
    manifest = full_manifest()
    validate_manifest(manifest)
    assert manifest_digest(manifest) == EXPECTED_MANIFEST_DIGEST
    assert (
        metadata_digest(item.rank_case for item in manifest) == EXPECTED_METADATA_DIGEST
    )
    _validate_public_api()
    yield
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def cuda_device():
    import torch

    return torch.device("cuda", torch.cuda.current_device())
