"""Formal benchmark resource checks without requiring a CUDA device."""

import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from benchmarks.inference.msa_v1.indexer.decode import plan as plan_benchmark
from benchmarks.inference.msa_v1.indexer.decode import runtime


@pytest.mark.parametrize("uuid", ("test-uuid", "GPU-test-uuid"))
@pytest.mark.parametrize("node", ("gb300-nvl-012-compute03", "nvl72d152-T08"))
def test_benchmark_device_guard(monkeypatch, uuid, node):
    monkeypatch.setattr(runtime.socket, "gethostname", lambda: node)
    monkeypatch.setattr(
        runtime.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(uuid=uuid, name="NVIDIA GB300"),
    )
    monkeypatch.setattr(runtime.torch.cuda, "get_device_capability", lambda: (10, 3))
    monkeypatch.setattr(runtime.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(runtime.torch, "empty", lambda *args, **kwargs: None)
    processes = "123\n"
    topology = "GPU0 GPU1\nGPU0 X NV18\nGPU1 NV18 X\n"

    def query(command, **kwargs):
        if command == ["nvidia-smi", "topo", "-m"]:
            return topology
        assert command[1] == "--id=GPU-test-uuid"
        if command[2] == "--query-gpu=index":
            return "0\n"
        return processes

    monkeypatch.setattr(runtime.subprocess, "check_output", query)
    assert runtime.check_benchmark_device()[0] == node
    topology = "GPU0 GPU1 GPU2\nGPU0 X SYS SYS\nGPU1 SYS X NV18\nGPU2 SYS NV18 X\n"
    with pytest.raises(RuntimeError, match="topology"):
        runtime.check_benchmark_device()
    topology = "GPU0 GPU1\nGPU0 X NV18\nGPU1 NV18 X\n"
    processes = "123\n456\n"
    with pytest.raises(RuntimeError, match="exclusive"):
        runtime.check_benchmark_device()
    for hostname in ("galaxy-ts3-043", "2u2g-gen-0176", "2u2g-gen-0300"):
        monkeypatch.setattr(runtime.socket, "gethostname", lambda name=hostname: name)
        with pytest.raises(RuntimeError, match="excludes"):
            runtime.check_benchmark_device()
    monkeypatch.setattr(runtime.socket, "gethostname", lambda: node)
    monkeypatch.setattr(
        runtime.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(uuid=uuid, name="NVIDIA B300A"),
    )
    with pytest.raises(RuntimeError, match="excluding B300A"):
        runtime.check_benchmark_device()


@pytest.mark.parametrize("valid", (False, True))
def test_plan_benchmark_retains_device_results(monkeypatch, tmp_path, valid):
    output = tmp_path / "plan.json"
    monkeypatch.setattr(
        sys, "argv", ["plan", "--precision", "q8kv8", "--out", str(output)]
    )
    monkeypatch.setattr(plan_benchmark, "LOW_LATENCY_CASES", (object(),))
    monkeypatch.setattr(
        plan_benchmark,
        "check_benchmark_device",
        lambda: ("gb300-nvl-012-compute03", SimpleNamespace(name="GB300", uuid="test")),
    )
    result = {
        "update_host_device_valid": valid,
        "update": {"median_us": 1.0, "cv": 0.0},
    }
    monkeypatch.setattr(plan_benchmark, "run_case", lambda *args, **kwargs: result)
    context = nullcontext() if valid else pytest.raises(SystemExit, match="CV=3%")
    with context:
        plan_benchmark.main()
    assert json.loads(output.read_text())["results"] == [result]
