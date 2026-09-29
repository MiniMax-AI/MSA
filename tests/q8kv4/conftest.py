# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

import logging
import time

import pytest
import torch

logger = logging.getLogger("tests.q8kv4")


def pytest_configure(config):
    config.addinivalue_line("markers", "full: large matrices; `-m 'not full'` keeps the smoke set")


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    return torch.device("cuda")


@pytest.fixture(scope="session", autouse=True)
def compiled_kernels(device):
    """Build every Q8KV4 kernel variant the tests use before any run is timed."""
    from fmha_sm100.decode_q8kv4 import jit

    started = time.perf_counter()
    jit.get_plan_fn(device)
    jit.get_reduction_module(device)
    for gqa_ratio in (16, 8):
        for split_kv in (False, True):
            for block_scale_shift in (0, 3):
                jit.get_fmha_fwd_variant(gqa_ratio=gqa_ratio, split_kv=split_kv, device=device,
                                         block_scale_shift=block_scale_shift)
    logger.info("Q8KV4 kernels ready in %.1fs (%s)", time.perf_counter() - started,
                jit._dequant_mode(jit._target_arch(device)))


def run_timed(label: str, fn):
    """Run ``fn`` twice with the GPU synchronized: the first call may still JIT-compile another
    backend and is only logged; the second call is the kernel run, which must finish within 30 s
    (the project's deadlock threshold). Returns the second result.
    """
    torch.cuda.synchronize()
    started = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    logger.info("%s first call (incl. any compile) %.3f s", label, time.perf_counter() - started)
    started = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    logger.info("%s ran in %.3f ms", label, elapsed * 1e3)
    assert elapsed < 30.0, f"{label} took {elapsed:.1f}s: treat as a deadlock"
    return result
