"""Separate compilation from bounded, synchronized correctness execution."""

import logging
import os
import threading
import time

import torch

from inference.msa_v1.indexer._common.topk_select.build import load_extension
from inference.msa_v1.indexer.prefill.q8kv8.interface import _compile_kernel

logger = logging.getLogger(__name__)


def run_checked(wrapper, q, k_cache, *, out=None):
    state = wrapper._proxy_score.plan_state
    _compile_kernel(q.view(-1, 128), k_cache, state, state.proxy_scores)
    load_extension()

    def abort():
        os.write(2, b"Q8KV8 kernel execution exceeded 30 seconds; possible deadlock\n")
        os._exit(124)

    watchdog = threading.Timer(30.0, abort)
    watchdog.daemon = True
    watchdog.start()
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = wrapper.run(q, k_cache, out=out)
        torch.cuda.synchronize()
        logger.info("Ran in %.3fms", (time.perf_counter() - started) * 1000)
        return result
    finally:
        watchdog.cancel()
