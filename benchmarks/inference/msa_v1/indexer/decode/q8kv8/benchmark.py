"""CUDA Graph decode benchmark with rotating cold-L2 input slots."""

from __future__ import annotations

# Imports follow the standalone-script path bootstrap.
import argparse
import importlib.metadata
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.inference.msa_v1.hardware import roofline_metrics
from benchmarks.inference.msa_v1.indexer.decode.metrics import (
    compare_low_latency,
    compare_mbu,
    necessary_bytes,
)
from benchmarks.inference.msa_v1.indexer.decode.q8kv8.cases import (
    BATCH_SIZES,
    LOW_LATENCY_BATCH_SIZES,
    LOW_LATENCY_SEQ_LENGTHS,
    SEQ_LENGTHS,
    DecodeCase,
    benchmark_cases,
    make_seq_lens,
)
from benchmarks.inference.msa_v1.indexer.decode.runtime import check_benchmark_device
from inference.msa_v1.indexer.decode.q8kv8.interface import (
    BatchDecodeIndexerWithPagedKVCacheWrapper,
)

PAGE_SIZE = 128
HEAD_DIM = 128
PAGE_BYTES = PAGE_SIZE * HEAD_DIM
MAX_CV = 0.03
MAX_TIMING_ATTEMPTS = 3


def _make_slot(
    case: DecodeCase,
    lengths_cpu: torch.Tensor,
    device: torch.device,
    seed: int,
    num_index_heads: int = 1,
    query_length: int = 8,
) -> tuple[torch.Tensor, ...]:
    max_pages = (int(lengths_cpu.max()) + PAGE_SIZE - 1) // PAGE_SIZE
    physical_pages = case.batch * max_pages
    generator = torch.Generator(device=device).manual_seed(seed)
    q = (
        torch.randn(
            case.batch,
            query_length,
            HEAD_DIM,
            generator=generator,
            device=device,
        )
        * 0.5
    ).to(torch.float8_e4m3fn)
    k_cache = (
        torch.randn(
            physical_pages,
            PAGE_SIZE,
            HEAD_DIM,
            generator=generator,
            device=device,
        )
        * 0.5
    ).to(torch.float8_e4m3fn)
    physical_indices = torch.arange(
        physical_pages,
        dtype=torch.int32,
        device=device,
    ).reshape(case.batch, max_pages)
    page_table = torch.stack(
        [
            physical_indices[batch_index][
                torch.randperm(max_pages, generator=generator, device=device)
            ]
            for batch_index in range(case.batch)
        ]
    ).contiguous()
    seq_lens = lengths_cpu.to(device=device)
    q = q.unsqueeze(2)
    if num_index_heads > 1:
        additional = (
            torch.randn(
                case.batch,
                query_length,
                num_index_heads - 1,
                HEAD_DIM,
                generator=generator,
                device=device,
            )
            * 0.5
        ).to(q.dtype)
        q = torch.cat((q, additional), dim=2)
    output = torch.empty(
        num_index_heads,
        case.batch * query_length,
        16,
        dtype=torch.int32,
        device=device,
    )
    return q, k_cache, page_table, seq_lens, output


def _time_graphs(
    graphs: list[torch.cuda.CUDAGraph],
    *,
    graph_calls: int,
    warmup: int,
    replays: int,
) -> tuple[list[float], float]:
    """Rotate graph banks and retry unstable measurements up to three times."""

    last_cv = float("inf")
    for _ in range(MAX_TIMING_ATTEMPTS):
        check_benchmark_device()
        for _ in range(warmup):
            for graph in graphs:
                graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(replays):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for graph in graphs:
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(
                start.elapsed_time(end) * 1000.0 / (graph_calls * len(graphs))
            )
        check_benchmark_device()
        sample_mean = statistics.fmean(samples)
        last_cv = statistics.pstdev(samples) / sample_mean
        if last_cv <= MAX_CV:
            return samples, last_cv
    raise RuntimeError(
        f"unstable timing after {MAX_TIMING_ATTEMPTS} attempts: "
        f"CV {last_cv:.4f} > {MAX_CV:.4f}"
    )


def _resolve_slots(
    *,
    requested_slots: int,
    graph_calls: int,
    read_bytes_per_call: int,
    l2_bytes: int,
) -> int:
    minimum_slots = (2 * l2_bytes) // read_bytes_per_call + 2
    if requested_slots:
        if requested_slots < minimum_slots:
            raise ValueError(
                f"--slots={requested_slots} is too small; at least {minimum_slots} "
                "disjoint slots are required"
            )
        slots = requested_slots
    elif minimum_slots <= graph_calls:
        slots = next(
            divisor
            for divisor in range(minimum_slots, graph_calls + 1)
            if graph_calls % divisor == 0
        )
    else:
        slots = ((minimum_slots + graph_calls - 1) // graph_calls) * graph_calls
    if slots <= graph_calls and graph_calls % slots:
        raise ValueError("slots must divide graph_calls when slots <= graph_calls")
    if slots > graph_calls and slots % graph_calls:
        raise ValueError(
            "slots must be a multiple of graph_calls when slots > graph_calls"
        )
    return slots


def run_case(
    case: DecodeCase,
    *,
    slots_count: int,
    graph_calls: int,
    warmup: int,
    replays: int,
    num_index_heads: int = 1,
    query_length: int = 8,
    verify: bool = False,
) -> dict[str, object]:
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    lengths_cpu = make_seq_lens(case)
    max_pages = (int(lengths_cpu.max()) + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = torch.div(
        lengths_cpu + PAGE_SIZE - 1, PAGE_SIZE, rounding_mode="floor"
    )
    scored_pages = torch.clamp(total_pages - 1, min=0)
    valid_pages = int(scored_pages.sum())
    query_positions = (
        lengths_cpu.reshape(-1, 1)
        - query_length
        + torch.arange(
            query_length,
            dtype=torch.int32,
        ).reshape(1, query_length)
    )
    query_local_pages = torch.div(
        query_positions,
        PAGE_SIZE,
        rounding_mode="floor",
    )
    valid_scores = int(query_local_pages.sum())
    read_bytes_per_call = (
        valid_pages * PAGE_BYTES
        + case.batch * query_length * HEAD_DIM
        + valid_pages * 4
        + case.batch * 4
    )
    reuse_distance_bytes = (slots_count - 1) * valid_pages * PAGE_BYTES
    l2_bytes = properties.L2_cache_size
    if reuse_distance_bytes <= 2 * l2_bytes:
        raise RuntimeError(
            "slot rotation does not establish a greater-than-2x-L2 reuse distance: "
            f"reuse={reuse_distance_bytes}, L2={l2_bytes}"
        )

    slots = [
        _make_slot(
            case,
            lengths_cpu,
            device,
            seed=case.seed * 100 + slot,
            num_index_heads=num_index_heads,
            query_length=query_length,
        )
        for slot in range(slots_count)
    ]
    wrappers = [
        BatchDecodeIndexerWithPagedKVCacheWrapper(
            use_cuda_graph=True,
            page_table_buffer=torch.empty_like(slot[2]),
            seq_lens_buffer=torch.empty_like(slot[3]),
        )
        for slot in slots
    ]
    for wrapper, slot in zip(wrappers, slots, strict=True):
        wrapper.plan(
            slot[2], slot[3], num_index_heads=num_index_heads, query_length=query_length
        )
    expected_outputs = []
    for wrapper, slot in zip(wrappers, slots, strict=True):
        q, k_cache, _, _, output = slot
        wrapper.run(q, k_cache, out=output)
        expected_outputs.append(output.clone())
    torch.cuda.synchronize()

    graph_count = max(1, slots_count // graph_calls)
    graphs = []
    for graph_index in range(graph_count):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for call in range(graph_calls):
                slot_index = (
                    graph_index * graph_calls + call
                    if slots_count > graph_calls
                    else call % slots_count
                )
                q, k_cache, _, _, output = slots[slot_index]
                wrappers[slot_index].run(q, k_cache, out=output)
        graphs.append(graph)
    for graph in graphs:
        graph.replay()
    torch.cuda.synchronize()
    for expected, slot in zip(expected_outputs, slots, strict=True):
        if not torch.equal(expected, slot[-1]):
            raise RuntimeError("CUDA Graph replay changed the full TopK output")

    per_call_microseconds, cv = _time_graphs(
        graphs,
        graph_calls=graph_calls,
        warmup=warmup,
        replays=replays,
    )
    latency_us = statistics.median(per_call_microseconds)
    sample_mean = statistics.fmean(per_call_microseconds)
    sample_std = statistics.pstdev(per_call_microseconds)
    flops = 2 * num_index_heads * valid_scores * PAGE_SIZE * HEAD_DIM
    output_bytes = num_index_heads * case.batch * query_length * 16 * 4
    logical_bytes = necessary_bytes(
        lengths_cpu, num_index_heads, PAGE_BYTES, query_length=query_length
    )
    effective_tb_s = logical_bytes / latency_us / 1.0e6
    if verify:
        from tests.inference.msa_v1.indexer._common.topk_select.reference import (
            assert_quantized_topk_contract,
        )
        from tests.inference.msa_v1.indexer.decode.q8kv8.reference import (
            indexer_gemm_reference,
        )

        q, k, table, lengths, output = slots[0]
        expected = indexer_gemm_reference(q, k, table, lengths)
        local = (
            lengths[:, None] - query_length + torch.arange(query_length, device=device)
        ) // 128
        mask = torch.arange(max_pages, device=device).view(1, 1, -1) < local.reshape(
            1, -1, 1
        )
        torch.testing.assert_close(
            wrappers[0]._proxy_scores.masked_select(mask),
            expected.masked_select(mask),
            atol=1e-4,
            rtol=1e-4,
        )
        assert_quantized_topk_contract(
            expected.reshape(-1, max_pages).cpu().numpy(),
            (local + 1).reshape(-1).repeat(num_index_heads).cpu().numpy(),
            output.reshape(-1, 16).cpu().numpy(),
        )
    return {
        **asdict(case),
        "q_len_per_req": query_length,
        "name": case.name,
        "length_min": int(lengths_cpu.min()),
        "length_max": int(lengths_cpu.max()),
        "length_mean": float(lengths_cpu.to(torch.float64).mean()),
        "valid_pages": valid_pages,
        "valid_scores": valid_scores * num_index_heads,
        "necessary_bytes": logical_bytes,
        "num_index_heads": num_index_heads,
        "max_pages": max_pages,
        "slots": slots_count,
        "graph_count": graph_count,
        "graph_calls": graph_calls,
        "replays": replays,
        "latency_us_samples": per_call_microseconds,
        "latency_us": latency_us,
        "latency_us_mean": sample_mean,
        "latency_us_std": sample_std,
        "cv": cv,
        "tflops": flops / latency_us / 1.0e6,
        "read_bytes_per_call": read_bytes_per_call,
        "output_bytes_per_call": output_bytes,
        "effective_gb_s": logical_bytes / latency_us / 1.0e3,
        "effective_tb_s": effective_tb_s,
        "l2_bytes": l2_bytes,
        "reuse_distance_bytes": reuse_distance_bytes,
        "reuse_distance_over_l2": reuse_distance_bytes / l2_bytes,
        "device": properties.name,
        "e2e_scope": "BatchDecodeIndexerWithPagedKVCacheWrapper.run",
        "correctness": "full reference passed" if verify else "TopK graph replay only",
        **roofline_metrics(
            device_name=properties.name,
            useful_flops=flops,
            logical_bytes=logical_bytes,
            latency_us=latency_us,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=("smoke", "full", "low-latency"), default="smoke"
    )
    parser.add_argument(
        "--batch",
        type=int,
        choices=tuple(sorted(set(BATCH_SIZES + LOW_LATENCY_BATCH_SIZES))),
        action="append",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        choices=tuple(sorted(set(SEQ_LENGTHS + LOW_LATENCY_SEQ_LENGTHS))),
        action="append",
    )
    parser.add_argument("--slots", type=int, default=0)
    parser.add_argument("--graph-calls", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--num-index-heads", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--query-length", type=int, choices=range(1, 17), default=8)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.slots < 0 or args.graph_calls < 1:
        parser.error("--slots must be non-negative and --graph-calls must be positive")
    if args.warmup < 1 or args.replays < 2:
        parser.error("--warmup must be positive and --replays must be at least two")
    check_benchmark_device()
    cases = benchmark_cases(args.suite)
    if args.batch:
        cases = tuple(case for case in cases if case.batch_size in set(args.batch))
    if args.seq_len:
        cases = tuple(
            case for case in cases if case.nominal_seq_len in set(args.seq_len)
        )
    if not cases:
        parser.error("benchmark selection is empty")
    results = []
    for case in cases:
        lengths_cpu = make_seq_lens(case)
        total_pages = torch.div(
            lengths_cpu + PAGE_SIZE - 1, PAGE_SIZE, rounding_mode="floor"
        )
        valid_pages = int(torch.clamp(total_pages - 1, min=0).sum())
        try:
            slots_count = _resolve_slots(
                requested_slots=args.slots,
                graph_calls=args.graph_calls,
                read_bytes_per_call=valid_pages * PAGE_BYTES,
                l2_bytes=torch.cuda.get_device_properties("cuda").L2_cache_size,
            )
        except ValueError as error:
            parser.error(str(error))
        result = run_case(
            case,
            slots_count=slots_count,
            graph_calls=args.graph_calls,
            warmup=args.warmup,
            replays=args.replays,
            num_index_heads=args.num_index_heads,
            query_length=args.query_length,
            verify=args.verify,
        )
        result["suite"] = args.suite
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    formal = args.suite == "full" and not args.batch and not args.seq_len
    aggregate = None
    if formal:
        total_weight = sum(case.weight for case in cases)
        weighted_latency = (
            sum(
                case.weight * float(result["latency_us"])
                for case, result in zip(cases, results, strict=True)
            )
            / total_weight
        )
        aggregate = {
            "total_weight": total_weight,
            "weighted_mean_e2e_latency_us": weighted_latency,
            "weighted_effective_mbu": sum(
                case.weight * row["necessary_bytes"]
                for case, row in zip(cases, results, strict=True)
            )
            / sum(
                case.weight * row["latency_us"]
                for case, row in zip(cases, results, strict=True)
            )
            / 1e6
            / results[0]["peak_hbm_tb_s"],
        }
    payload = {
        "schema_version": 2,
        "num_index_heads": args.num_index_heads,
        "precision": "q8kv8",
        "device_uuid": str(torch.cuda.get_device_properties("cuda").uuid),
        "device": torch.cuda.get_device_properties("cuda").name,
        "dsl_version": importlib.metadata.version("nvidia-cutlass-dsl"),
        "protocol": {
            "suite": args.suite,
            "formal": formal
            or (
                args.suite == "low-latency"
                and not args.batch
                and not args.seq_len
                and args.warmup == 5
                and args.replays == 20
                and args.graph_calls == 120
            ),
            "e2e_scope": "public wrapper.run (proxy GEMM + TopK)",
            "cuda_graph_calls": args.graph_calls,
            "query_length": args.query_length,
            "warmup_replays": args.warmup,
            "timed_replays": args.replays,
            "maximum_cv": MAX_CV,
            "cold_cache_reuse_distance_over_l2": 2.0,
        },
        "aggregate": aggregate,
        "results": results,
    }
    acceptance_failed = False
    if args.baseline is not None:
        baseline_payload = json.loads(args.baseline.read_text())
        comparator = compare_low_latency if args.suite == "low-latency" else compare_mbu
        payload["acceptance"] = comparator(payload, baseline_payload)
        acceptance_failed = not payload["acceptance"]["passed"]
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({key: value for key, value in payload.items() if key != "results"})
    )
    if acceptance_failed:
        raise SystemExit("E2E performance acceptance failed")


if __name__ == "__main__":
    main()
