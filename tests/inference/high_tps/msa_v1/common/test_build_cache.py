"""Guard dependency tracking after the source-tree split; no GPU required."""

import shutil
from pathlib import Path

from inference.high_tps.msa_v1.common import jit


def test_source_dependency_invalidation(tmp_path, monkeypatch):
    project = (
        Path(__file__).resolve().parents[5] / "inference/high_tps/msa_v1"
    )
    shutil.copytree(project / "csrc", tmp_path / "csrc")
    (tmp_path / "common").mkdir()
    shutil.copy2(project / "common/jit.py", tmp_path / "common/jit.py")
    monkeypatch.setattr(jit, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(jit, "_CSRC_DIR", tmp_path / "csrc")
    monkeypatch.setattr(jit, "__file__", str(tmp_path / "common/jit.py"))
    manager = jit.M3DecodeVariantManager()

    def fingerprints():
        result = {}
        for mode in ("mega_atten",):
            name, config = manager._variant_info(kernel_mode=mode)
            result[mode] = jit._build_fingerprint(name, config)
        return result

    before = fingerprints()
    assert before == fingerprints()
    leaf = tmp_path / "csrc/attn/kernel/include/sm100_m3_decode_mainloop.hpp"
    leaf.write_text(leaf.read_text() + "\n// Dependency test.\n")
    after = fingerprints()
    assert before["mega_atten"] != after["mega_atten"]
    shared = tmp_path / "csrc/common/kernel/include/sm100_m3_decode_traits.hpp"
    shared.write_text(shared.read_text() + "\n// Shared dependency test.\n")
    final = fingerprints()
    assert all(after[mode] != final[mode] for mode in after)


def test_templates_reference_their_owner():
    manager = jit.M3DecodeVariantManager()
    manager._load_templates()
    for mode, owner in jit._SOURCE_OWNERS.items():
        _, config = manager._variant_info(kernel_mode=mode)
        text = manager._inst_templates[mode].render(
            **config, source_owner=owner, variant_name="check", func_name="check"
        )
        assert f'#include "{owner}/kernel/include/m3_decode_sm100.cuh"' in text


def test_attention_variants_depend_only_on_static_modes():
    manager = jit.M3DecodeVariantManager()
    manager._load_templates()
    names = set()
    for diagnostic, placement in ((False, False), (False, True), (True, False), (True, True)):
        expected = manager._variant_info(
            kernel_mode="mega_atten", wait_deadline=diagnostic, enable_placement=placement
        )
        names.add(expected[0])
        source = manager._inst_templates["mega_atten"].render(
            **expected[1], source_owner="attn", variant_name=expected[0], func_name="check"
        )
        assert f"#define M3_DECODE_WAIT_DEADLINE {int(diagnostic)}" in source
        assert f"#define M3_DECODE_ENABLE_PLACEMENT {int(placement)}" in source
        assert "run_m3_decode_mega_atten(p)" in source
        assert "run_m3_decode_mega_atten<" not in source
    assert len(names) == 4
    assert manager._variant_info(kernel_mode="mega_atten")[1]["enable_placement"] == "0"
