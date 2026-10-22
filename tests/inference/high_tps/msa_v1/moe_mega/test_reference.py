"""Check the Q8 mega MoE against an independent FP32 reference on TP4.

Run with four single-node processes:
    torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.moe_mega.test_reference
"""

import argparse
from dataclasses import replace
from datetime import timedelta
import json
import logging
import os
import time
import uuid

import torch
import torch.distributed as dist

from inference.high_tps.msa_v1.moe import (
    MoeConfig,
    MoeWeights,
    create_moe_runner,
    create_moe_tp_context,
)

from ..common.m3_decode_fixture import route_ids
from .reference import E, H, I, K, R, T, local_partial, ordered_output

logger = logging.getLogger(__name__)


def make_operands(device, rank, active_experts, seed, *, random_routes, zero_scales=False):
    """Nonzero random weights with a router that selects a known expert set."""
    common = torch.Generator(device=device).manual_seed(seed)
    local = torch.Generator(device=device).manual_seed(seed + 101 * (rank + 1))
    hidden = (torch.randn((T, H), device=device, generator=common) * 0.7).bfloat16()
    residual = (torch.randn((T, H), device=device, generator=common) * 0.35).bfloat16()
    experts = torch.randperm(E, device=device, generator=common)[:active_experts]
    if random_routes:
        # The fixture's random routes: a cyclic layout with shuffled tokens and slots.
        ids = route_ids(active_experts, device, "random").long()
    else:
        ids = experts[torch.arange(R, device=device) % active_experts].reshape(T, K)
    target = torch.full((T, E), -12.0, device=device)
    target.scatter_(1, ids, torch.tensor([12.0, 10.0, 8.0, 6.0], device=device).expand(T, K))
    x = hidden.float()
    router = (torch.linalg.solve(x @ x.T, target).T @ x).contiguous()

    def normal(shape):
        return (torch.randn(shape, device=device, generator=local) * 0.025).to(torch.float8_e4m3fn)

    def codes(shape, low, high):
        return torch.randint(low, high, shape, device=device, dtype=torch.uint8, generator=local)

    tensors = dict(
        router_weight=router,
        router_correction_bias=torch.zeros(E, device=device),
        shared_fc1_weight=normal((2 * I, H)),
        shared_fc1_weight_sf=codes((2 * I, H // 32), 126, 129),
        shared_fc2_weight=normal((H, I)),
        shared_fc2_weight_sf=codes((H, I // 32), 126, 129),
        routed_fc1_weight=codes((E, 2 * I, H // 2), 0, 256),
        routed_fc1_weight_sf=codes((E, 2 * I, H // 16), 48, 65).view(torch.float8_e4m3fn),
        routed_fc1_alpha=torch.linspace(0.001, 0.008, E, device=device),
        routed_fc2_weight=codes((E, H, I // 2), 0, 256),
        routed_fc2_weight_sf=codes((E, H, I // 16), 48, 65).view(torch.float8_e4m3fn),
        routed_fc2_alpha=torch.linspace(0.00025, 0.002, E, device=device),
    )
    if zero_scales:
        for name in (
            "shared_fc1_weight_sf",
            "shared_fc2_weight_sf",
            "routed_fc1_weight_sf",
            "routed_fc2_weight_sf",
        ):
            tensors[name].view(torch.uint8).zero_()
        # Signed E2M1 zero with nonzero scales in one selected expert.
        for prefix in ("routed_fc1", "routed_fc2"):
            tensors[prefix + "_weight"][experts[0]].fill_(0x88)
            tensors[prefix + "_weight_sf"][experts[0]].view(torch.uint8).fill_(56)
    return MoeWeights(**tensors), hidden, residual


def expected_output(plan, group):
    weights = plan.runner.weights
    fc2_alpha = weights.routed_fc2_alpha * plan.runner.config.routed_scaling_factor
    partial, active = local_partial(weights, plan.hidden, fc2_alpha)
    partials = [None] * dist.get_world_size(group)
    dist.all_gather_object(partials, partial.cpu(), group=group)
    return ordered_output(partials, plan.residual), active


def check(plan, group, label, *, expected_active=None):
    expected, active = expected_output(plan, group)
    actual = plan.output.float()
    reference = expected.float()
    failures = []
    if not torch.isfinite(actual).all():
        failures.append("non-finite output")
    # Only the GEMM accumulation order differs from the reference.
    relative = torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(
        reference
    ).clamp_min(1e-20)
    if not (torch.isclose(actual, reference, rtol=0.02, atol=0.0625).all() and relative <= 0.002):
        failures.append(f"output differs from the reference (relative L2 {relative.item():.3g})")
    if expected_active is not None and active != expected_active:
        failures.append(f"constructed {active} active experts, expected {expected_active}")
    reports = [None] * dist.get_world_size(group)
    dist.all_gather_object(reports, failures, group=group)
    if any(reports):
        raise AssertionError(json.dumps({"case": label, "rank_failures": reports}))
    return {"case": label, "passed": True}


def timed_run(plan, label):
    torch.cuda.synchronize()
    start = time.time()
    plan.run()  # A run exceeding 30 seconds is a deadlock.
    torch.cuda.synchronize()
    logger.info("%s ran in %.3f ms", label, (time.time() - start) * 1e3)


def check_determinism(plan, group, label):
    timed_run(plan, label)
    first = plan.output.clone()
    plan.output.fill_(float("nan"))
    timed_run(plan, label)
    if not torch.equal(first, plan.output):
        raise AssertionError(f"{label}: repeated execution is not bitwise identical")


def check_graph(plan, group, calls, replays, label):
    plan.run()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            plan.run()
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()
    results = [check(plan, group, label)]
    replayed = plan.output.clone()
    plan.run()
    torch.cuda.synchronize()
    if not torch.equal(replayed, plan.output):
        raise AssertionError(f"{label}: graph replay differs from eager execution")
    plan.hidden.neg_()
    plan.residual.add_(0.125)
    plan.output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    results.append(check(plan, group, label + "_changed_inputs"))
    del graph
    return results


def expect_error(label, error_types, function):
    try:
        function()
    except error_types:
        return {"case": label, "passed": True}
    raise AssertionError(f"{label}: invalid usage was accepted")


def run_suite(group, rank, args):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    identifiers = ["moe-reference-" + uuid.uuid4().hex if rank == 0 else None]
    dist.broadcast_object_list(identifiers, src=dist.get_global_rank(group, 0), group=group)
    context = create_moe_tp_context(group=group, group_id=identifiers[0], backend="mega")
    results, plans = [], []
    try:
        for random_routes in (False, True):
            for active in args.experts:
                for seed in args.seeds:
                    label = f"experts_{active}_{'random' if random_routes else 'balanced'}_{seed}"
                    weights, hidden, residual = make_operands(
                        device, rank, active, seed + 31 * active, random_routes=random_routes
                    )
                    runner = create_moe_runner(weights=weights, tp_context=context, backend="mega")
                    plan = runner.prepare(q_len=T, hidden=hidden, residual=residual)
                    plan.output.fill_(float("nan"))
                    timed_run(plan, label)
                    results.append(check(plan, group, label, expected_active=active))
                    if seed == args.seeds[0]:
                        check_determinism(plan, group, label)
                        plans.append(plan)
        results.extend(check_graph(plans[1], group, args.graph_calls, args.graph_replays, "graph"))
        first, second = plans[0], plans[-1]
        for index, plan in enumerate((first, second, first, second)):
            plan.output.fill_(float("nan"))
            plan.run()
            results.append(check(plan, group, f"alternating_plan_{index}"))
        results.append(
            expect_error(
                "output_alias_rejected",
                ValueError,
                lambda: first.runner.prepare(q_len=T, hidden=first.output, residual=first.residual),
            )
        )
        results.append(
            expect_error(
                "q_len_7_rejected",
                ValueError,
                lambda: first.runner.prepare(
                    q_len=7, hidden=first.hidden[1:], residual=first.residual[1:]
                ),
            )
        )
        invalid = replace(
            first.runner.weights,
            routed_fc1_alpha=torch.zeros_like(first.runner.weights.routed_fc1_alpha),
        )
        results.append(
            expect_error(
                "nonpositive_alpha_rejected",
                ValueError,
                lambda: create_moe_runner(weights=invalid, tp_context=context, backend="mega"),
            )
        )
        tiny_runner = create_moe_runner(
            weights=first.runner.weights,
            config=MoeConfig(routed_scaling_factor=2**-200),
            tp_context=context,
            backend="mega",
        )
        tiny = tiny_runner.prepare(q_len=T, hidden=first.hidden, residual=first.residual)
        tiny.output.fill_(float("nan"))
        tiny.run()
        results.append(check(tiny, group, "tiny_routing_factor"))
        weights, hidden, residual = make_operands(
            device, rank, 10, args.seeds[0] + 200, random_routes=False, zero_scales=True
        )
        runner = create_moe_runner(weights=weights, tp_context=context, backend="mega")
        edge = runner.prepare(q_len=T, hidden=hidden, residual=residual)
        edge.output.fill_(float("nan"))
        edge.run()
        results.append(check(edge, group, "zero_scales_signed_zero"))
        hidden.zero_()
        edge.output.fill_(float("nan"))
        edge.run()
        torch.cuda.synchronize()
        if not torch.equal(edge.output, residual):
            raise AssertionError("zero input must return the residual exactly")
        results.append({"case": "zero_input_returns_residual", "passed": True})
    finally:
        torch.cuda.synchronize()
        context.close()
    results.append(expect_error("closed_context_rejected", RuntimeError, plans[0].run))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, nargs="+", default=[4, 10, 20, 28, 30, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[4879, 9133])
    parser.add_argument("--graph-calls", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=3)
    args = parser.parse_args()
    if len(args.experts) < 2 or any(not 4 <= value <= R for value in args.experts):
        parser.error("provide at least two expert counts in [4, 32]")
    logging.basicConfig(level=logging.INFO if os.environ.get("RANK") == "0" else logging.WARNING)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("gloo", timeout=timedelta(minutes=30))
    try:
        if dist.get_world_size() != 4:
            raise ValueError("run four single-node processes")
        group = dist.new_group(ranks=[0, 1, 2, 3], backend="gloo")
        results = run_suite(group, dist.get_rank(), args)
        if dist.get_rank() == 0:
            logger.info(json.dumps({"passed": True, "checks": results}))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
