"""Compile-time helpers shared by CuTe kernels."""

import logging
import time
from typing import Optional

import cutlass.cute as cute

logger = logging.getLogger(__name__)


def compile_with_timing(*args, **kwargs):
    """Compile one CuTe program and log host-side latency at DEBUG."""
    started_at = time.perf_counter()
    compiled = cute.compile(*args, **kwargs)
    logger.debug("Compiled in %.1fs", time.perf_counter() - started_at)
    return compiled


def make_fake_tensor(dtype, shape, divisibility=1, leading_dim=-1) -> Optional[cute.Tensor]:
    if leading_dim < 0:
        leading_dim = len(shape) + leading_dim
    if dtype is None:
        return None
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=divisibility * dtype.width // 8
    )


__all__ = ["compile_with_timing", "make_fake_tensor"]
