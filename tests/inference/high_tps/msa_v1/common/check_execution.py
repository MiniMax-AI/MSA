"""TP4 analytic checks, fixed-Q plan reuse, and CUDA Graph topology."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from inference.high_tps.msa_v1.common.jit import (
    get_m3_decode_mega_atten_variant,
)

from .execution import ExecutionResources, PreparedOperation
from .m3_decode_fixture import (
    graph_summary,
    synchronize_checked,
)


def main(kind):
    parser = argparse.ArgumentParser(description=f"Validate {kind} on TP4")
    # MegaMoE accepts exactly eight query rows.
    fixed_q8 = kind == "moe_mega" or kind.endswith("_moe_mega")
    parser.add_argument(
        "--q", type=int, nargs="+", default=[8] if fixed_q8 else list(range(1, 9))
    )
    parser.add_argument("--kv-len", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--nonzero",
        action="store_true",
        help="Dump finite nonzero outputs for baseline comparison",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disable-pdl", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("gloo")
    if dist.get_world_size() != 4:
        raise RuntimeError("validation requires single-node TP4")
    resources = ExecutionResources(
        kind, rank, device, f"verify-{os.environ['MASTER_PORT']}"
    )
    cases, graphs, records, outputs = [], [], {}, {}
    try:
        if kind == "attn_mega" or kind.startswith("attn_mega_"):
            get_m3_decode_mega_atten_variant()
        dist.barrier()
        for q in args.q:
            case = PreparedOperation(
                resources,
                q=q,
                kv_len=args.kv_len,
                seed=args.seed,
                zero_math=not args.nonzero,
                active_experts=min(20, 4 * q),
                enable_pdl=not args.disable_pdl,
            )
            cases.append(case)
            dist.barrier()
            expected = case.run().clone()
            synchronize_checked(device, f"{kind}_q{q}_direct")
            if not torch.isfinite(expected).all():
                raise AssertionError("non-finite output")
            if not args.nonzero:
                torch.testing.assert_close(
                    expected,
                    case.expected_zero_math(),
                    rtol=0.008 if kind == "attn_mega" else 0,
                    atol=0.008 if kind == "attn_mega" else 0,
                )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                case.run()
            synchronize_checked(device, f"{kind}_q{q}_warmup")
            dist.barrier()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            graphs.append((graph, stream))
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(10):
                    output = case.run()
            info = graph_summary(graph)
            assert info["kernel_count"] == 10 * case.kernel_count, info
            assert set(info["node_types"]) == {"kernel"}, info
            args.output.mkdir(parents=True, exist_ok=True)
            graph.debug_dump(str(args.output / f"{kind}-q{q}-rank{rank}.dot"))
            with torch.cuda.stream(stream):
                graph.replay()
                graph.replay()
            synchronize_checked(device, f"{kind}_q{q}_graph")
            torch.testing.assert_close(output, expected, rtol=0.008, atol=0.008)
            outputs[f"q{q}"] = expected.cpu()
            records[f"q{q}"] = info
        for case in reversed(cases):
            output = case.run()
            synchronize_checked(device, f"{kind}_q{case.q}_reverse")
            torch.testing.assert_close(
                output.cpu(), outputs[f"q{case.q}"], rtol=0.008, atol=0.008
            )
        torch.save(outputs, args.output / f"{kind}-rank{rank}.pt")
        (args.output / f"{kind}-rank{rank}.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "rank": rank,
                    "kind": kind,
                    "nonzero": args.nonzero,
                    "gpu": torch.cuda.get_device_name(device),
                    "cases": records,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"PASS kind={kind} rank={rank}", flush=True)
    finally:
        synchronize_checked(device, "validation_cleanup")
        for graph, _ in graphs:
            graph.reset()
        graphs.clear()
        resources.close()
        dist.destroy_process_group()
