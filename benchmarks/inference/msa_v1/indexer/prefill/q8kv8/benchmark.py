"""CUDA Graph E2E benchmark for the Q8KV8 prefill indexer."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.inference.msa_v1.hardware import roofline_metrics
from datas.inference.cases import (
    InferencePrefillBenchmarkCase,
    InferencePrefillCase,
    load_prefill_benchmark_cases,
)
from datas.inference.tensors import (
    cumulative_lengths,
    make_disjoint_page_table,
)
from inference.msa_v1.indexer.prefill.q8kv8 import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)
from inference.msa_v1.indexer.prefill.q8kv8.indexer_gemm import (
    PrefillIndexerGemmSm100,
)

PAGE_SIZE = 128
HEAD_DIM = 128
TOPK = 16
MAX_CV = 0.03
MAX_TIMING_ATTEMPTS = 3
MINIMUM_THROUGHPUT_RATIO = 0.99


def _make_inputs(
    case: InferencePrefillCase, num_index_heads: int = 1
) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(case.seed)
    page_table, physical_pages = make_disjoint_page_table(
        case.final_kv_lens,
        max_cols=case.max_cols,
        generator=generator,
        device=device,
    )
    q = (
        torch.randn(
            (case.total_q, 1, HEAD_DIM),
            generator=generator,
            device=device,
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    k_cache = (
        torch.randn(
            (physical_pages, 1, PAGE_SIZE, HEAD_DIM),
            generator=generator,
            device=device,
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    if num_index_heads > 1:
        # Keep head zero and K identical to the original H=1 baseline.
        additional_q = (
            torch.randn(
                (case.total_q, num_index_heads - 1, HEAD_DIM),
                generator=generator,
                device=device,
            )
            * 0.25
        )
        q = torch.cat((q.float(), additional_q), dim=1).to(torch.float8_e4m3fn)
    return (
        q,
        k_cache,
        page_table,
        torch.tensor(
            cumulative_lengths(case.query_lens),
            dtype=torch.int32,
            device=device,
        ),
        torch.tensor(
            cumulative_lengths(case.final_kv_lens),
            dtype=torch.int32,
            device=device,
        ),
    )


def _plan(
    wrapper: BatchPrefillIndexerWithPagedKVCacheWrapper,
    case: InferencePrefillCase,
    inputs: tuple[torch.Tensor, ...],
) -> None:
    q, _, page_table, cu_seqlens_q, cu_seqlens_k = inputs
    wrapper.plan(
        cu_seqlens_q,
        cu_seqlens_k,
        page_table,
        total_q=case.total_q,
        max_seqlen_q=case.max_query_len,
        max_seqlen_k=case.max_final_kv,
        num_index_heads=q.shape[1],
    )


def _logical_work(
    case: InferencePrefillCase, num_index_heads: int = 1
) -> tuple[int, int]:
    candidate_entries = int(case.metrics["topk_candidate_entries"])
    logical_bytes = (
        candidate_entries * PAGE_SIZE * HEAD_DIM
        + case.total_q * HEAD_DIM
        + candidate_entries * 4
        + case.total_q * TOPK * 4
    )
    return case.useful_flops * num_index_heads, logical_bytes * num_index_heads


def _unique_working_set(case: InferencePrefillCase) -> int:
    historical_pages = sum((length - 1) // PAGE_SIZE for length in case.final_kv_lens)
    return historical_pages * PAGE_SIZE * HEAD_DIM + case.total_q * (TOPK + 1) * 4


def _resolve_slots(
    case: InferencePrefillCase,
    *,
    requested_slots: int,
    l2_bytes: int,
) -> int:
    # Count distinct historical K bytes, not repeated per-query reads. Use the
    # H=1 footprint for every H so baseline and candidates rotate equally.
    # No historical K is read in this boundary case; K-cache rotation is vacuous.
    if all(length <= PAGE_SIZE for length in case.final_kv_lens):
        return requested_slots or 1
    unique_bytes = _unique_working_set(case)
    minimum_slots = (2 * l2_bytes) // unique_bytes + 2
    if requested_slots and requested_slots < minimum_slots:
        raise ValueError(
            f"--slots={requested_slots} is too small; case {case.case_id} requires "
            f"at least {minimum_slots} disjoint tensor slots"
        )
    return requested_slots or minimum_slots


def _time_graphs(
    graphs: list[torch.cuda.CUDAGraph],
    *,
    warmup: int,
    replays: int,
    calls_per_graph: int = 1,
) -> dict[str, object]:
    last_cv = float("inf")
    for _ in range(MAX_TIMING_ATTEMPTS):
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
                start.elapsed_time(end) * 1000.0 / (len(graphs) * calls_per_graph)
            )
        sample_mean = statistics.fmean(samples)
        last_cv = statistics.pstdev(samples) / sample_mean
        if last_cv <= MAX_CV:
            return {
                "latency_us": statistics.median(samples),
                "latency_us_mean": sample_mean,
                "latency_us_min": min(samples),
                "latency_us_max": max(samples),
                "latency_us_samples": samples,
                "cv": last_cv,
                "timed_replays": replays,
                "graph_slots": len(graphs),
            }
    raise RuntimeError(
        f"unstable timing after {MAX_TIMING_ATTEMPTS} attempts: "
        f"CV {last_cv:.4f} > {MAX_CV:.4f}"
    )


def _select_cases(
    args: argparse.Namespace,
) -> tuple[InferencePrefillBenchmarkCase, ...]:
    cases = load_prefill_benchmark_cases()
    if args.suite == "smoke":
        cases = tuple(case for case in cases if "batch_anchor" in case.tags)
    if args.case_id:
        requested = set(args.case_id)
        cases = tuple(case for case in cases if case.case.case_id in requested)
    if args.tag:
        requested_tags = set(args.tag)
        cases = tuple(case for case in cases if requested_tags & set(case.tags))
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("benchmark selection is empty")
    return cases


@torch.inference_mode()
def run_case(
    selection: InferencePrefillBenchmarkCase,
    *,
    slots: int,
    warmup: int,
    replays: int,
    num_index_heads: int = 1,
    verify: bool = False,
) -> dict[str, object]:
    case = selection.case
    input_slots = [_make_inputs(case, num_index_heads) for _ in range(slots)]
    wrappers = [BatchPrefillIndexerWithPagedKVCacheWrapper() for _ in range(slots)]
    for wrapper, inputs in zip(wrappers, input_slots, strict=True):
        _plan(wrapper, case, inputs)

    def launch(slot: int) -> torch.Tensor:
        q, k_cache, _, _, _ = input_slots[slot]
        state = wrappers[slot]._proxy_score.plan_state
        return wrappers[slot].run(q, k_cache, out=state.topk_indices)

    baseline_outputs = []
    for slot in range(slots):
        baseline_outputs.append(launch(slot).clone())
    torch.cuda.synchronize()

    calls_per_graph = 120 if all(x <= PAGE_SIZE for x in case.final_kv_lens) else 1
    graphs = []
    graph_outputs = []
    for slot in range(slots):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls_per_graph):
                output = launch(slot)
        graphs.append(graph)
        graph_outputs.append(output)
    for graph in graphs:
        graph.replay()
    torch.cuda.synchronize()
    for expected, actual in zip(baseline_outputs, graph_outputs, strict=True):
        if not torch.equal(expected, actual):
            raise AssertionError(
                f"{case.case_id}: full TopK output changed after graph replay"
            )

    stats = _time_graphs(
        graphs, warmup=warmup, replays=replays, calls_per_graph=calls_per_graph
    )
    if verify:
        from tests.inference.msa_v1.indexer.prefill.q8kv8.cases import RealPrefillInputs
        from tests.inference.msa_v1.indexer.prefill.q8kv8.reference import (
            assert_full_scores,
            assert_full_topk_quality,
            assert_topk_structure,
        )

        q, k, pages, cu_q, cu_k = input_slots[0]
        state = wrappers[0]._proxy_score.plan_state
        state.proxy_scores.fill_(float("nan"))
        output = launch(0)
        expected = assert_full_scores(
            case, RealPrefillInputs(q, k, pages, cu_q, cu_k), state.proxy_scores
        )
        assert_topk_structure(output, state.num_valid_pages)
        assert_full_topk_quality(expected, state.num_valid_pages, output)
    latency_us = float(stats["latency_us"])
    useful_flops, logical_bytes = _logical_work(case, num_index_heads)
    properties = torch.cuda.get_device_properties("cuda")
    row = {
        "case_id": case.case_id,
        "num_index_heads": num_index_heads,
        "tags": selection.tags,
        "representative_weight": selection.representative_weight,
        "stratum": selection.stratum,
        "count": case.count,
        "batch_size": case.batch_size,
        "query_lens": case.query_lens,
        "prefix_lens": case.prefix_lens,
        "final_kv_lens": case.final_kv_lens,
        "total_q": case.total_q,
        "max_query_len": case.max_query_len,
        "max_final_kv": case.max_final_kv,
        "max_cols": case.max_cols,
        "useful_flops": useful_flops,
        "useful_tflops": useful_flops / latency_us / 1.0e6,
        "logical_bytes": logical_bytes,
        "logical_tb_s": logical_bytes / latency_us / 1.0e6,
        "cold_cache_slots": slots,
        "calls_per_graph": calls_per_graph,
        "reuse_distance_bytes": (slots - 1) * _unique_working_set(case),
        "reuse_distance_over_l2": (slots - 1)
        * _unique_working_set(case)
        / properties.L2_cache_size,
        "e2e_scope": "BatchPrefillIndexerWithPagedKVCacheWrapper.run",
        "correctness": "full independent reference passed"
        if verify
        else "graph replay consistency only",
        **roofline_metrics(
            device_name=properties.name,
            useful_flops=useful_flops,
            logical_bytes=logical_bytes,
            latency_us=latency_us,
        ),
        **stats,
    }
    del graphs, graph_outputs, baseline_outputs
    wrappers.clear()
    input_slots.clear()
    torch.cuda.empty_cache()
    return row


def _compare_throughput(rows: list[dict], baseline: dict, num_index_heads: int) -> dict:
    """Compare weighted useful work per E2E second against the original H=1 implementation."""
    baseline_rows = {row["case_id"]: row for row in baseline["results"]}
    if set(baseline_rows) != {row["case_id"] for row in rows}:
        raise ValueError("baseline/candidate case mismatch")
    weighted_time = 0.0
    baseline_time = 0.0
    weighted_flops = 0.0
    regressed = []
    for row in rows:
        previous = baseline_rows[row["case_id"]]
        if max(row["cv"], previous["cv"]) > MAX_CV:
            raise ValueError("baseline/candidate CV exceeds the timing gate")
        if row["cold_cache_slots"] != previous["cold_cache_slots"]:
            raise ValueError("baseline/candidate graph rotation mismatch")
        weight = int(row["representative_weight"] or 0)
        if weight != int(previous["representative_weight"] or 0):
            raise ValueError("baseline/candidate weight mismatch")
        if row["useful_flops"] != num_index_heads * previous["useful_flops"]:
            raise ValueError("baseline must contain single-head useful FLOPs")
        weighted_time += weight * row["latency_us"]
        baseline_time += weight * previous["latency_us"]
        weighted_flops += weight * row["useful_flops"]
        if num_index_heads == 1 and row["latency_us"] > previous["latency_us"] * 1.05:
            regressed.append(row["case_id"])
    if min(weighted_time, baseline_time, weighted_flops) <= 0:
        raise ValueError("weighted work and times must be positive")
    candidate_tflops = weighted_flops / weighted_time / 1.0e6
    baseline_tflops = weighted_flops / num_index_heads / baseline_time / 1.0e6
    return {
        "passed": candidate_tflops >= baseline_tflops * MINIMUM_THROUGHPUT_RATIO
        and not regressed,
        "minimum_throughput_ratio": MINIMUM_THROUGHPUT_RATIO,
        "candidate_weighted_useful_tflops": candidate_tflops,
        "baseline_weighted_useful_tflops": baseline_tflops,
        "throughput_ratio": candidate_tflops / baseline_tflops,
        "h1_regressed_cases": regressed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--tag", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--slots", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--num-index-heads", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    if args.slots < 0 or args.warmup < 1 or args.replays < 2:
        parser.error("slots must be non-negative, warmup positive, and replays >= 2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability not in PrefillIndexerGemmSm100.supported_compute_capabilities:
        raise RuntimeError("Q8KV8 prefill benchmark requires SM100, SM103 or SM107")
    try:
        selections = _select_cases(args)
    except ValueError as error:
        parser.error(str(error))

    l2_bytes = torch.cuda.get_device_properties("cuda").L2_cache_size
    rows = []
    for selection in selections:
        try:
            slots = _resolve_slots(
                selection.case,
                requested_slots=args.slots,
                l2_bytes=l2_bytes,
            )
        except ValueError as error:
            parser.error(str(error))
        row = run_case(
            selection,
            slots=slots,
            warmup=args.warmup,
            replays=args.replays,
            num_index_heads=args.num_index_heads,
            verify=args.verify,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    is_formal = (
        args.suite == "full"
        and not args.case_id
        and not args.tag
        and args.limit is None
        and len(rows) == 128
    )
    representative = (
        [row for row in rows if "representative" in row["tags"]] if is_formal else []
    )
    represented_calls = sum(int(row["representative_weight"]) for row in representative)
    weighted_latency = (
        sum(
            int(row["representative_weight"]) * float(row["latency_us"])
            for row in representative
        )
        / represented_calls
        if represented_calls
        else None
    )
    result = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(capability),
        "cutlass_dsl_version": importlib.metadata.version("nvidia-cutlass-dsl"),
        "num_index_heads": args.num_index_heads,
        "protocol": {
            "suite": args.suite,
            "formal_full_selection": is_formal,
            "e2e_scope": "public wrapper.run: proxy GEMM + TopK",
            "compile_plan_allocate_capture_in_timing": False,
            "cuda_graph": True,
            "warmup_replays": args.warmup,
            "timed_replays": args.replays,
            "maximum_cv": MAX_CV,
            "maximum_timing_attempts": MAX_TIMING_ATTEMPTS,
            "cold_cache": "disjoint rotation above 2x L2; no-history uses one slot with 120 calls",
        },
        "aggregate": {
            "formal": is_formal,
            "representative_cases": len(representative),
            "represented_calls": represented_calls,
            "weighted_mean_e2e_latency_us": weighted_latency,
            "weighted_useful_tflops": (
                sum(
                    int(row["representative_weight"]) * row["useful_flops"]
                    for row in representative
                )
                / (represented_calls * weighted_latency * 1.0e6)
                if represented_calls
                else None
            ),
        },
        "results": rows,
    }
    acceptance_failed = False
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text())
        if not is_formal or not baseline["protocol"]["formal_full_selection"]:
            raise ValueError(
                "performance acceptance requires both full 128-case selections"
            )
        for key in ("device", "compute_capability", "cutlass_dsl_version"):
            if result[key] != baseline[key]:
                raise ValueError(f"baseline/candidate {key} mismatch")
        if result["protocol"] != baseline["protocol"]:
            raise ValueError("baseline/candidate timing protocol mismatch")
        result["acceptance"] = _compare_throughput(rows, baseline, args.num_index_heads)
        acceptance_failed = not result["acceptance"]["passed"]
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    if acceptance_failed:
        raise SystemExit("E2E performance acceptance failed")


if __name__ == "__main__":
    main()
