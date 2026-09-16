"""Cold JIT publication must be safe across serving ranks."""

import multiprocessing
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from inference.msa_v1.attention.decode.q8kv4 import jit


def _build_lock_worker(directory, start):
    cache_dir = Path(directory)
    start.wait()
    for _ in range(3):
        with jit._build_lock(cache_dir):
            marker = cache_dir / "building"
            with marker.open("x"):
                time.sleep(0.02)
            marker.unlink()


def test_build_lock_serializes_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    workers = [
        context.Process(target=_build_lock_worker, args=(str(tmp_path), start))
        for _ in range(4)
    ]
    try:
        for worker in workers:
            worker.start()
        start.set()
        for worker in workers:
            worker.join(timeout=45)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)


@pytest.mark.parametrize("component", ("attention", "plan", "reduction"))
def test_compile_and_load_share_the_build_lock(tmp_path, monkeypatch, component):
    import tvm_ffi

    held = False
    events = []
    result = object()

    @contextmanager
    def lock(directory):
        nonlocal held
        directory.mkdir(parents=True, exist_ok=True)
        held = True
        try:
            yield
        finally:
            held = False

    def check(stage, value=None):
        assert held, f"{stage} ran outside the process lock"
        events.append(stage)
        return value

    monkeypatch.setattr(jit, "_build_lock", lock)
    monkeypatch.setattr(jit, "_cache_dir", lambda *args: tmp_path / component)
    monkeypatch.setattr(jit, "_dequant_mode", lambda arch: "qmul4")
    monkeypatch.setattr(jit, "_write_ninja", lambda *args: check("generate"))
    monkeypatch.setattr(jit, "_run_ninja", lambda *args: check("compile"))
    monkeypatch.setattr(tvm_ffi, "load_module", lambda *args: check("load", result))
    if component == "attention":
        actual = jit.JitSpec(
            "decode_attention_q8kv4_topk16", False, "qmul4", "103a"
        ).build_and_load()
    else:
        actual = jit._build_fixed_module(
            component, tmp_path / "source.cu", "test", "103a"
        )
    assert actual is result
    assert events == ["generate", "compile", "load"]
    assert not held
