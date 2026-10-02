"""Prepare FlashInfer before timing its public decode call."""

import logging
import time

import cutlass
from packaging.version import Version

from inference.msa_v1.attention.decode._flashinfer import load_backend
from tests.inference.msa_v1.attention.decode.runtime import run_cuda

logger = logging.getLogger(__name__)


def compile_module() -> None:
    assert Version(cutlass.__version__) >= Version("4.5.2")
    logger.info("CuTe DSL %s loaded from %s", cutlass.__version__, cutlass.__file__)
    started = time.perf_counter()
    load_backend()
    from flashinfer.decode import get_trtllm_gen_fmha_module

    get_trtllm_gen_fmha_module()
    logger.info(
        "FlashInfer compilation preparation completed in %.3fs",
        time.perf_counter() - started,
    )


def run_decode(wrapper, inputs, *, out=None):
    return run_cuda(
        "FlashInfer decode",
        lambda: wrapper.run(inputs.q, (inputs.k_cache, inputs.v_cache), out=out),
    )
