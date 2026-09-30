"""Measure shared decode-plan updates and explicit multi-layer Graph consumers."""

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path

import cutlass
import torch

from benchmarks.inference.msa_v1.decode.cases import LOW_LATENCY_CASES, make_seq_lens
from benchmarks.inference.msa_v1.indexer.decode.runtime import check_benchmark_device
from inference.msa_v1.indexer.decode import BatchDecodeIndexerPlan


def run_case(case, *, precision, heads, query_length, layers):
    harness = importlib.import_module(
        f"benchmarks.inference.msa_v1.indexer.decode.{precision}.benchmark"
    )
    lengths = make_seq_lens(case)
    history_pages = int(((lengths - 1) // 128).sum())
    properties = torch.cuda.get_device_properties("cuda")
    # The first layer alone establishes the cold-cache distance for every slot.
    slots_count = harness._resolve_slots(
        requested_slots=0,
        graph_calls=120,
        read_bytes_per_call=history_pages * harness.PAGE_BYTES,
        l2_bytes=properties.L2_cache_size,
    )
    plans, consumers = [], []
    for slot in range(slots_count):
        source = lengths.to(device="cuda")
        plan = BatchDecodeIndexerPlan(
            source,
            max_pages=(int(lengths.max()) + 127) // 128,
            num_index_heads=heads,
            query_length=query_length,
        )
        plan.update()
        layer_consumers = []
        for layer in range(layers):
            values = harness._make_slot(
                case,
                lengths,
                torch.device("cuda"),
                seed=case.seed * 100 + slot * layers + layer,
                num_index_heads=heads,
                query_length=query_length,
            )
            if precision == "q8kv8":
                q, k, table, _, output = values
                kwargs = {}
            else:
                q, k, scale, table, _, output = values
                kwargs = {"k_scale": scale}
            wrapper = harness.BatchDecodeIndexerWithPagedKVCacheWrapper()
            wrapper.plan(
                table,
                source,
                num_index_heads=heads,
                query_length=query_length,
                shared_plan=plan,
            )
            wrapper.run(q, k, out=output, **kwargs)
            layer_consumers.append((wrapper, q, k, kwargs, output))
        plans.append(plan)
        consumers.append(layer_consumers)
    torch.cuda.synchronize()
    for _ in range(5):
        plans[0].update()
    torch.cuda.synchronize()
    for _ in range(harness.MAX_TIMING_ATTEMPTS):
        host_samples = []
        for _ in range(20):
            started = time.perf_counter_ns()
            plans[0].update()
            torch.cuda.synchronize()
            host_samples.append((time.perf_counter_ns() - started) / 1000)
        host_cv = statistics.pstdev(host_samples) / statistics.fmean(host_samples)
        if host_cv <= harness.MAX_CV:
            break
    result = {
        "name": case.name,
        "batch_size": case.batch,
        "nominal_seq_len": case.nominal_seq_len,
        "slots": slots_count,
        "update_host_device_median_us": statistics.median(host_samples),
        "update_host_device_cv": host_cv,
        "update_host_device_valid": host_cv <= harness.MAX_CV,
        "update_cache": "metadata hot cache",
        "run_cache": "disjoint K rotation, reuse distance greater than 2x L2",
    }
    for scope in ("update", "run", "update_run"):
        graphs = []
        for bank in range(max(1, slots_count // 120)):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for call in range(120):
                    slot = (
                        bank * 120 + call if slots_count > 120 else call % slots_count
                    )
                    if scope != "run":
                        plans[slot].update()
                    if scope != "update":
                        for wrapper, q, k, kwargs, output in consumers[slot]:
                            wrapper.run(q, k, out=output, **kwargs)
            graphs.append(graph)
        check_benchmark_device()
        samples, cv = harness._time_graphs(
            graphs, graph_calls=120, warmup=5, replays=20
        )
        result[scope] = {"median_us": statistics.median(samples), "cv": cv}
        del graphs, graph
    reference = importlib.import_module(
        f"tests.inference.msa_v1.indexer.decode.{precision}.reference"
    )
    from tests.inference.msa_v1.indexer._common.topk_select.reference import (
        assert_quantized_topk_contract,
    )

    for wrapper, q, k, kwargs, output in consumers[0]:
        proxy = wrapper._proxy_score
        if precision == "q8kv8":
            expected = reference.indexer_gemm_reference(
                q, k, proxy._page_table, proxy._seq_lens
            )
        else:
            expected = reference.indexer_gemm_reference(
                q, k, kwargs["k_scale"], proxy._page_table, proxy._seq_lens
            )
        local = (
            proxy._seq_lens[:, None]
            - query_length
            + torch.arange(query_length, device="cuda")
        ) // 128
        pages = expected.shape[-1]
        mask = torch.arange(pages, device="cuda").view(1, 1, -1) < local.reshape(
            1, -1, 1
        )
        torch.testing.assert_close(
            wrapper._proxy_scores.masked_select(mask),
            expected.masked_select(mask),
            atol=1e-4,
            rtol=1e-4,
        )
        assert_quantized_topk_contract(
            expected.reshape(-1, pages).cpu().numpy(),
            (local + 1).reshape(-1).repeat(heads).cpu().numpy(),
            output.reshape(-1, 16).cpu().numpy(),
        )
    result["correctness"] = "all layers, all rows in the first rotation slot"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("q8kv8", "q8kv4"), required=True)
    parser.add_argument("--num-index-heads", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--query-length", type=int, choices=range(1, 17), default=8)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.layers < 1:
        parser.error("--layers must be positive")
    hostname, properties = check_benchmark_device()
    payload = {
        "precision": args.precision,
        "num_index_heads": args.num_index_heads,
        "query_length": args.query_length,
        "layers": args.layers,
        "node": hostname,
        "device": properties.name,
        "device_uuid": str(properties.uuid),
        "dsl_version": cutlass.__version__,
        "cuda_backend": str(cutlass.CUDA_VERSION),
        "results": [],
    }
    for case in LOW_LATENCY_CASES:
        result = run_case(
            case,
            precision=args.precision,
            heads=args.num_index_heads,
            query_length=args.query_length,
            layers=args.layers,
        )
        payload["results"].append(result)
        print(json.dumps(result), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
    if not all(row["update_host_device_valid"] for row in payload["results"]):
        raise SystemExit(
            "host+device update timing exceeded CV=3%; device results retained"
        )


if __name__ == "__main__":
    main()
