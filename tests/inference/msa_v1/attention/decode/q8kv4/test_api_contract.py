"""Public API contract tests for Q8KV4 paged sparse decode."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from inference.msa_v1.attention.decode import q8kv4
from inference.msa_v1.attention.decode.q8kv4 import (
    BatchDecodeWithPagedKVCacheWrapper,
    interface,
    jit,
)


def test_public_package_exports_only_wrapper() -> None:
    assert q8kv4.__all__ == ["BatchDecodeWithPagedKVCacheWrapper"]
    for old_name in (
        "prepare",
        "decode_attention",
        "fmha_fwd_plan",
        "fmha_fwd_run",
        "fmha_fwd",
    ):
        assert not hasattr(q8kv4, old_name)


def test_plan_and_run_signatures_use_canonical_names() -> None:
    assert tuple(
        inspect.signature(BatchDecodeWithPagedKVCacheWrapper.plan).parameters
    ) == (
        "self",
        "topk_indices",
        "page_table",
        "seq_lens",
        "q_len_per_req",
        "num_q_heads",
        "num_kv_heads",
        "num_kv_splits",
        "usable_sm_count",
        "sm_scale",
        "output_mode",
    )
    assert tuple(
        inspect.signature(BatchDecodeWithPagedKVCacheWrapper.run).parameters
    ) == ("self", "q", "paged_kv_cache", "kv_cache_sf", "out", "out_scale")


def test_run_requires_plan_and_aligned_data(monkeypatch) -> None:
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called"):
        wrapper.run(
            torch.empty(0),
            (torch.empty(0), torch.empty(0)),
            kv_cache_sf=(torch.empty(0), torch.empty(0)),
        )

    monkeypatch.setattr(
        interface, "_prepare_decode_plan", lambda *args, **kwargs: object()
    )

    def unexpected_launch(*args, **kwargs):
        pytest.fail("Misaligned data reached the kernel backend")

    monkeypatch.setattr(interface, "_run_backend", unexpected_launch)
    device = torch.device("cuda")
    topk = torch.full((8, 1, 16), -1, dtype=torch.int32, device=device)
    topk[:, :, 0] = 0
    pages = torch.zeros((1, 1), dtype=torch.int32, device=device)
    lengths = torch.full((1,), 8, dtype=torch.int32, device=device)
    for ratio in (8, 16):
        wrapper.plan(
            topk, pages, lengths, q_len_per_req=8, num_q_heads=ratio, num_kv_heads=1
        )
        data = {
            "q": torch.empty((8, ratio, 128), dtype=torch.float8_e4m3fn, device=device),
            "k_cache": torch.empty((1, 1, 128, 64), dtype=torch.uint8, device=device),
            "v_cache": torch.empty((1, 1, 128, 64), dtype=torch.uint8, device=device),
            "k_scale": torch.empty(
                (1, 1, 128, 8), dtype=torch.float8_e4m3fn, device=device
            ),
            "v_scale": torch.empty(
                (1, 1, 128, 8), dtype=torch.float8_e4m3fn, device=device
            ),
            "out": torch.empty((8, ratio, 128), dtype=torch.bfloat16, device=device),
        }
        for name, tensor in data.items():
            misaligned = torch.empty(
                tensor.numel() + 1, dtype=tensor.dtype, device=device
            )[1:].view(tensor.shape)
            assert misaligned.is_contiguous() and misaligned.data_ptr() % 16 != 0
            invalid = {**data, name: misaligned}
            with pytest.raises(
                ValueError, match=f"{name} must have a 16-byte aligned address"
            ):
                wrapper.run(
                    invalid["q"],
                    (invalid["k_cache"], invalid["v_cache"]),
                    kv_cache_sf=(invalid["k_scale"], invalid["v_scale"]),
                    out=invalid["out"],
                )


def test_plan_rejects_invalid_metadata_before_compilation(monkeypatch) -> None:
    device = torch.device("cuda")
    topk = torch.zeros((8, 4, 16), dtype=torch.int32, device=device)
    page_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    seq_lens = torch.full((1,), 8, dtype=torch.int32, device=device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper()

    with pytest.raises(ValueError, match="q_len_per_req"):
        wrapper.plan(topk, page_table, seq_lens, q_len_per_req=0)
    with pytest.raises(TypeError, match="seq_lens must be torch.int32"):
        wrapper.plan(
            topk,
            page_table,
            seq_lens.to(torch.int64),
            q_len_per_req=8,
        )
    with pytest.raises(ValueError, match="topk_indices must have shape"):
        wrapper.plan(topk[:7], page_table, seq_lens, q_len_per_req=8)
    with pytest.raises(ValueError, match="8 or 16 Q heads per KV head"):
        wrapper.plan(
            topk[:, :1].contiguous(),
            page_table,
            seq_lens,
            q_len_per_req=8,
            num_q_heads=12,
            num_kv_heads=1,
        )
    monkeypatch.setattr(jit, "_target_arch", lambda device=None: "107a")
    with pytest.raises(RuntimeError, match="GQA=8 requires SM100 or SM103"):
        wrapper.plan(topk, page_table, seq_lens, q_len_per_req=8, num_q_heads=32)


def test_jit_spec_contains_only_compile_time_configuration(monkeypatch) -> None:
    monkeypatch.setattr(jit, "_source_digest", lambda: "test-source")
    monkeypatch.setattr(jit, "_target_arch", lambda device=None: "103a")
    monkeypatch.setattr(jit, "_dequant_mode", lambda arch: "fp16_fallback")
    no_split = jit.gen_jit_spec(topk=16, split_kv=False)
    split = jit.gen_jit_spec(topk=16, split_kv=True)

    assert tuple(no_split.__dataclass_fields__) == (
        "variant_name",
        "split_kv",
        "dequant_mode",
        "target_arch",
        "gqa_ratio",
        "output_mode",
    )
    assert no_split.variant_name == "decode_attention_q8kv4_topk16"
    assert no_split.dequant_mode == "fp16_fallback"
    assert split.variant_name == "decode_attention_q8kv4_topk16_split"
    assert no_split.uri != split.uri
    native_eight = jit.gen_jit_spec(topk=16, split_kv=False, gqa_ratio=8)
    assert native_eight.uri != no_split.uri
    assert native_eight.gqa_ratio == 8
    mxfp8 = jit.gen_jit_spec(output_mode="mxfp8")
    assert mxfp8.uri != no_split.uri
    assert mxfp8.output_mode == "mxfp8"
    with pytest.raises(ValueError, match="GQA ratio"):
        jit.gen_jit_spec(gqa_ratio=12)
    monkeypatch.setattr(jit, "_target_arch", lambda device=None: "107a")
    assert jit.gen_jit_spec(gqa_ratio=16).target_arch == "107a"
    with pytest.raises(RuntimeError, match="GQA=8 requires SM100 or SM103"):
        jit.gen_jit_spec(gqa_ratio=8)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        jit.get_fmha_fwd_variant(q_tokens_per_batch=3)


def test_lazy_jit_uses_tensor_device_over_offline_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:5")
    monkeypatch.setenv("MM_SPARSE_TARGET_ARCH", "100a")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda requested: (10, 3) if requested == device else (10, 0),
    )
    jit._target_arch.cache_clear()
    try:
        assert jit._target_arch(device) == "103a"
    finally:
        jit._target_arch.cache_clear()


@pytest.mark.parametrize(
    ("supports_qmul4", "expected"),
    ((False, "fp16_fallback"), (True, "qmul4")),
)
def test_dequant_mode_follows_compiler_capability(
    monkeypatch: pytest.MonkeyPatch,
    supports_qmul4: bool,
    expected: str,
) -> None:
    monkeypatch.setattr(jit, "_supports_qmul4", lambda arch: supports_qmul4)
    jit._dequant_mode.cache_clear()
    try:
        assert jit._dequant_mode("103a") == expected
    finally:
        jit._dequant_mode.cache_clear()


@pytest.mark.parametrize("splits", (None, 1, 2, 4, 8))
@pytest.mark.parametrize(
    "batch,gqa_ratio,default_mode",
    ((1, 8, "streamk"), (64, 8, "legacy"), (64, 16, "streamk")),
)
@pytest.mark.parametrize("mode_override", (None, "streamk", "legacy"))
def test_explicit_split_count_overrides_automatic_schedule(
    monkeypatch,
    splits,
    batch,
    gqa_ratio,
    default_mode,
    mode_override,
):
    monkeypatch.delenv("MSA_Q8KV4_SPLIT_MODE", raising=False)
    if mode_override is not None:
        monkeypatch.setenv("MSA_Q8KV4_SPLIT_MODE", mode_override)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=208),
    )
    monkeypatch.setattr(interface, "_make_backend_plan", lambda *args, **kwargs: kwargs)
    plan = interface._prepare_decode_plan(
        batch,
        8,
        device=torch.device("cuda:0"),
        num_q_heads=gqa_ratio * 4,
        num_kv_heads=4,
        num_kv_splits=splits,
    )
    expected_mode = mode_override or default_mode
    assert plan["split_mode"] == (expected_mode if splits is None else "legacy")
    if splits is not None:
        assert plan["num_kv_splits"] == splits


@pytest.mark.parametrize("version", ((13, 4), (13, 5)))
def test_rubin_checks_toolkit_before_build(monkeypatch, version):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (10, 7))
    monkeypatch.setattr(jit, "_cuda_version", lambda: version)
    jit._target_arch.cache_clear()
    try:
        if version < (13, 5):
            with pytest.raises(RuntimeError, match="SM107 requires CUDA 13.5"):
                jit._target_arch(torch.device("cuda:0"))
        else:
            assert jit._target_arch(torch.device("cuda:0")) == "107a"
    finally:
        jit._target_arch.cache_clear()
