"""Architecture selection tests for the NVFP4 dequant JIT."""

from __future__ import annotations

import torch

from inference.dequant.nvfp4_to_fp8 import jit


def test_tensor_device_overrides_offline_build_arch(monkeypatch) -> None:
    device = torch.device("cuda:4")
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


def test_offline_precompile_uses_requested_arch(monkeypatch) -> None:
    monkeypatch.setenv("MM_SPARSE_TARGET_ARCH", "10.0a")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    jit._target_arch.cache_clear()
    try:
        assert jit._target_arch() == "100a"
    finally:
        jit._target_arch.cache_clear()
