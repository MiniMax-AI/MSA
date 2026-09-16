"""Separate compilation from synchronized, deadlock-guarded decode execution."""

import logging
import time

import cutlass
from packaging.version import Version
import torch

from inference.msa_v1.attention.decode.q8kv4 import jit

logger = logging.getLogger(__name__)


def compile_modules(gqa_ratio: int, device: torch.device) -> None:
    assert Version(cutlass.__version__) >= Version("4.5.2")
    logger.info("CuTe DSL %s loaded from %s", cutlass.__version__, cutlass.__file__)
    started = time.perf_counter()
    for split in (False, True):
        jit.get_fmha_fwd_variant(gqa_ratio=gqa_ratio, split_kv=split, device=device)
    jit.get_plan_fn(device)
    jit.get_reduction_module(device)
    logger.info(
        "Compilation preparation completed in %.3fs", time.perf_counter() - started
    )
