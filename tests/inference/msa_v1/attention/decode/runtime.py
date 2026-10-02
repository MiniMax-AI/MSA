"""Synchronized execution timing with an active kernel deadlock watchdog."""

import logging
import os
import threading
import time
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


def _terminate_deadlock() -> None:
    logger.error("Kernel execution exceeded 30 seconds; terminating deadlocked process")
    os._exit(124)


@contextmanager
def cuda_execution(label: str):
    """Time a compiled launch region, including completion synchronization."""
    watchdog = threading.Timer(30, _terminate_deadlock)
    watchdog.daemon = True
    watchdog.start()
    started = time.perf_counter()
    try:
        torch.cuda.synchronize()
        yield
        torch.cuda.synchronize()
    finally:
        watchdog.cancel()
    elapsed = time.perf_counter() - started
    logger.info("Ran %s in %.3fms", label, elapsed * 1.0e3)
    if elapsed > 30:
        raise RuntimeError("kernel run exceeded the 30-second deadlock threshold")


def run_cuda(label: str, operation):
    with cuda_execution(label):
        return operation()
