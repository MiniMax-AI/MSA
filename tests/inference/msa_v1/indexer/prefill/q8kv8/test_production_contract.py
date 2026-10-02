"""Static production-path contracts that must not depend on host metadata."""

from pathlib import Path

import pytest

from inference.msa_v1.indexer.prefill.q8kv8.indexer_gemm import (
    PrefillIndexerGemmSm100,
)

_PRODUCTION_DIR = (
    Path(__file__).resolve().parents[6] / "inference/msa_v1/indexer/prefill/q8kv8"
)
_BANNED_HOST_MATERIALIZATION = (
    ".item(",
    ".cpu(",
    ".tolist(",
    ".numpy(",
    "torch.cuda.synchronize(",
)


def test_production_path_has_no_metadata_d2h_or_host_synchronize() -> None:
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in _PRODUCTION_DIR.glob("*.py")
    }
    assert sources
    for name, source in sources.items():
        for banned in _BANNED_HOST_MATERIALIZATION:
            assert banned not in source, f"{name} contains forbidden {banned}"


def test_plan_and_gemm_use_independent_persistent_aot_entries() -> None:
    source = (_PRODUCTION_DIR / "interface.py").read_text(encoding="utf-8")
    expected_names = (
        "msa_v1_indexer_prefill_q8kv8_multihead_v1_plan_reset_sm100",
        "msa_v1_indexer_prefill_q8kv8_multihead_v1_plan_build_sm100",
        "msa_v1_indexer_prefill_q8kv8_multihead_v1_gemm_sm100",
    )

    assert source.count("compile_or_load(") == len(expected_names)
    assert "_CUTE_DSL_VERSION" not in source
    assert "cutlass.__version__" not in source
    for name in expected_names:
        assert name in source


def test_sm100_family_uses_architecture_specific_reduction_and_dynamic_grid(
    monkeypatch, tmp_path
) -> None:
    import torch

    from inference.msa_v1 import _build_utils
    from inference.msa_v1.indexer.prefill.q8kv8 import interface

    headers = tmp_path / "external_cutlass/include/cutlass"
    headers.mkdir(parents=True)
    (headers / "cutlass.h").touch()
    monkeypatch.setenv("CUTLASS_ROOT", str(headers.parents[1]))
    monkeypatch.setenv("CUTLASS_PATH", "")
    monkeypatch.delenv("CUTLASS_PATH", raising=False)
    monkeypatch.setattr(_build_utils, "_REPOSITORY_ROOT", tmp_path / "wheel_install")
    _build_utils.cutlass_root.cache_clear()
    try:
        for capability in ((10, 0), (10, 3), (10, 7)):
            monkeypatch.setattr(
                torch.cuda, "get_device_capability", lambda _: capability
            )
            assert interface._check_runtime(torch.device("cuda")) == capability
        assert _build_utils.cutlass_root() == headers.parents[1]
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
        with pytest.raises(RuntimeError, match="SM100, SM103 and SM107"):
            interface._check_runtime(torch.device("cuda"))
    finally:
        _build_utils.cutlass_root.cache_clear()

    sm100 = PrefillIndexerGemmSm100(
        compute_capability=(10, 0),
        num_persistent_clusters=71,
    )
    sm103 = PrefillIndexerGemmSm100(
        compute_capability=(10, 3),
        num_persistent_clusters=73,
    )

    assert sm100.num_persistent_clusters == 71
    assert not sm100.use_tmem_load_reduce
    assert sm103.num_persistent_clusters == 73
    assert sm103.use_tmem_load_reduce
    with pytest.raises(ValueError, match="SM100, SM103 or SM107"):
        PrefillIndexerGemmSm100(
            compute_capability=(9, 0),
            num_persistent_clusters=1,
        )
    with pytest.raises(ValueError, match="must be positive"):
        PrefillIndexerGemmSm100(
            compute_capability=(10, 0),
            num_persistent_clusters=0,
        )

    interface_source = (_PRODUCTION_DIR / "interface.py").read_text(encoding="utf-8")
    assert "multi_processor_count" in interface_source
    assert "PERSISTENT_CLUSTERS_BY_CAPABILITY" not in interface_source
    assert not (_PRODUCTION_DIR / "_config.py").exists()
