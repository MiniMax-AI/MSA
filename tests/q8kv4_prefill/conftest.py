# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

import logging
import time

import pytest
import torch

logger = logging.getLogger("tests.q8kv4_prefill")


def pytest_configure(config):
    config.addinivalue_line("markers", "full: large matrices; `-m 'not full'` keeps the smoke set")


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    return torch.device("cuda")


@pytest.fixture(scope="session", autouse=True)
def compiled_kernels(device):
    """Build the Q8KV4 prefill kernel and load the CuTe-DSL sparse stack before any run is timed."""
    from fmha_sm100.prefill_q8kv4 import jit
    from fmha_sm100.prefill_q8kv4.interface import _sparse_stack

    started = time.perf_counter()
    for block_scale_shift in (0, 3):
        jit.load_extension(device, block_scale_shift)
    _sparse_stack()
    logger.info("Q8KV4 prefill kernel ready in %.1fs", time.perf_counter() - started)


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
