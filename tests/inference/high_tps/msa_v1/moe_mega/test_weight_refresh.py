"""CPU checks of refresh ownership and failure handling; GPU numerical tests are separate."""

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from inference.high_tps.msa_v1.moe.backends.moe_mega.backend import MegaBackend
from inference.high_tps.msa_v1.moe.interface import MoeConfig, MoeRunner
from inference.high_tps.msa_v1.moe.weights import MoeWeights


@pytest.fixture
def runner(monkeypatch):
    # Small CPU operands exercise actual refresh arithmetic. CUDA input-domain
    # validation is covered by the runner's GPU tests and is stubbed here only.
    device = torch.device("cuda", 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(MoeWeights, "validate", lambda self: device)
    weights = MoeWeights(**{field.name: torch.ones(4) for field in fields(MoeWeights)})
    result = MoeRunner.__new__(MoeRunner)
    result.weights = weights
    result.device = device
    result.tp_context = SimpleNamespace(closed=False)
    result._weight_bindings = result._weight_storage_bindings()
    result._weights_valid = True
    result._backends = {"mega": MegaBackend(weights, MoeConfig())}
    return result


def test_refresh_updates_alpha_without_replacing_storage(runner):
    prepared = runner._backends["mega"].fc2_alpha
    pointer = prepared.data_ptr()
    runner.weights.routed_fc2_alpha.fill_(3.0)
    runner.refresh_weights()
    assert prepared.data_ptr() == pointer
    torch.testing.assert_close(prepared, torch.full_like(prepared, 6.0))
    runner._require_valid_weights()


def test_refresh_rejects_replaced_storage(runner):
    runner.weights.routed_fc2_alpha.data = torch.full((4,), 3.0)
    with pytest.raises(ValueError, match="storage cannot be replaced"):
        runner.refresh_weights()
    with pytest.raises(RuntimeError, match="refresh failed"):
        runner._require_valid_weights()


def test_refresh_failure_invalidates_all_plans_until_success(runner):
    def fail():
        raise RuntimeError("injected refresh failure")

    runner._backends["faulty"] = SimpleNamespace(refresh=fail)
    with pytest.raises(RuntimeError, match="injected"):
        runner.refresh_weights()
    with pytest.raises(RuntimeError, match="refresh failed"):
        runner._require_valid_weights()
    del runner._backends["faulty"]
    runner.refresh_weights()
    runner._require_valid_weights()


def test_refresh_rejects_capture_before_mutating_prepared_storage(runner, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    prepared = runner._backends["mega"].fc2_alpha.clone()
    with pytest.raises(RuntimeError, match="capture"):
        runner.refresh_weights()
    torch.testing.assert_close(runner._backends["mega"].fc2_alpha, prepared)


def run_graph_refresh(backend):
    """Run under torchrun with TP4; compare retained Graphs with a fresh runner."""
    import os

    import torch.distributed as dist

    from inference.high_tps.msa_v1.moe import create_moe_runner, create_moe_tp_context
    from tests.inference.high_tps.msa_v1.common.m3_decode_fixture import (
        synchronize_checked,
    )
    from tests.inference.high_tps.msa_v1.moe_mega.test_reference import make_operands

    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    context = create_moe_tp_context(
        backend=backend, group_id=f"refresh-{backend}-{os.environ['MASTER_PORT']}"
    )
    weights, hidden, residual = make_operands(device, rank, 8, 20261021, random_routes=False)
    runner = create_moe_runner(weights=weights, tp_context=context, backend=backend)
    plan = runner.prepare(q_len=8, hidden=hidden, residual=residual)
    dist.barrier()
    plan.run()
    synchronize_checked(device, "refresh warmup")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run()
    graph.replay()
    synchronize_checked(device, "refresh original Graph")
    before = plan.output.clone()
    weights.routed_fc1_weight.bitwise_xor_(1)
    weights.shared_fc1_weight_sf.add_(1)
    weights.routed_fc2_alpha.mul_(1.25)
    runner.refresh_weights()
    synchronize_checked(device, "refresh prepared weights")
    dist.barrier()
    fresh = create_moe_runner(weights=weights, tp_context=context, backend=backend)
    fresh_plan = fresh.prepare(q_len=8, hidden=hidden, residual=residual)
    fresh_plan.run()
    synchronize_checked(device, "refresh fresh runner")
    expected = fresh_plan.output.clone()
    graph.replay()
    synchronize_checked(device, "refresh retained Graph")
    actual = plan.output
    assert torch.isfinite(actual).all()
    assert not torch.equal(before, actual), "the update did not affect the output"
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=0.0625)
    relative = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
        expected.float()
    ).clamp_min(1e-20)
    assert relative <= 0.002
    print(f"PASS {backend} refresh retained Graph rank={rank}", flush=True)
    graph.reset()
    context.close()


if __name__ == "__main__":
    import argparse
    import os
    import traceback

    import torch.distributed as dist

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mega",), default="mega")
    options = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    try:
        run_graph_refresh(options.backend)
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - abort communicating peers on any failure
        traceback.print_exc()
        os._exit(1)
