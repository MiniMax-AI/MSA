#!/usr/bin/env python3
"""CUDA Graph E2E benchmark for the BF16 prefill indexer."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.inference.msa_v1.acceptance import compare_e2e_results
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
from inference.msa_v1.indexer.prefill.bf16 import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)
from inference.msa_v1.indexer.prefill.bf16.indexer_gemm import (
    M3_PAGED_DIRECT_SCORE_NUM_HEADS,
)

PAGE_SIZE = 128
HEAD_DIM = 128
TOPK = 16
DEFAULT_NUM_INDEX_HEADS = 4
MAX_CV = 0.03
MAX_TIMING_ATTEMPTS = 3


def _ceil_div(dividend: int, divisor: int) -> int:
    return (dividend + divisor - 1) // divisor


def _make_inputs(
    case: InferencePrefillCase,
    num_index_heads: int,
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
            (case.total_q, num_index_heads, HEAD_DIM),
            generator=generator,
            device=device,
        )
        * 0.25
    ).to(torch.bfloat16)
    k_cache = (
        torch.randn(
            (physical_pages, 1, PAGE_SIZE, HEAD_DIM),
            generator=generator,
            device=device,
        )
        * 0.25
    ).to(torch.bfloat16)
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
    num_index_heads: int,
) -> None:
    _, _, page_table, cu_seqlens_q, cu_seqlens_k = inputs
    wrapper.plan(
        cu_seqlens_q,
        cu_seqlens_k,
        page_table,
        total_q=case.total_q,
        max_seqlen_q=case.max_query_len,
        max_seqlen_k=case.max_final_kv,
        num_index_heads=num_index_heads,
    )


def _logical_work(
    case: InferencePrefillCase,
    num_index_heads: int,
) -> tuple[int, int]:
    candidate_entries = int(case.metrics["topk_candidate_entries"])
    useful_flops = 2 * candidate_entries * num_index_heads * PAGE_SIZE * HEAD_DIM
    return useful_flops, _working_set_bytes(case, num_index_heads)


def _working_set_bytes(
    case: InferencePrefillCase,
    num_index_heads: int,
) -> int:
    physical_pages = (
        sum(_ceil_div(length, PAGE_SIZE) for length in case.final_kv_lens) + 1
    )
    q_bytes = case.total_q * num_index_heads * HEAD_DIM * 2
    k_bytes = physical_pages * PAGE_SIZE * HEAD_DIM * 2
    score_bytes = case.total_q * num_index_heads * case.max_cols * 4
    output_bytes = case.total_q * num_index_heads * (4 + TOPK * 4)
    metadata_bytes = case.batch_size * case.max_cols * 4 + 2 * (case.batch_size + 1) * 4
    return q_bytes + k_bytes + score_bytes + output_bytes + metadata_bytes


def _resolve_slots(
    case: InferencePrefillCase,
    *,
    requested_slots: int,
    l2_bytes: int,
    num_index_heads: int,
) -> int:
    working_set_bytes = _working_set_bytes(case, num_index_heads)
    minimum_slots = (2 * l2_bytes) // working_set_bytes + 2
    if requested_slots and requested_slots < minimum_slots:
        raise ValueError(
            f"--slots={requested_slots} is too small; case {case.case_id} requires "
            f"at least {minimum_slots} disjoint tensor slots"
        )
    return requested_slots or minimum_slots


def _time_graph_rotation(
    graph: torch.cuda.CUDAGraph,
    *,
    calls_per_replay: int,
    warmup: int,
    replays: int,
) -> dict[str, object]:
    last_cv = float("inf")
    for _ in range(MAX_TIMING_ATTEMPTS):
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(replays):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0 / calls_per_replay)
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
                "graph_slots": calls_per_replay,
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
    num_index_heads: int,
) -> dict[str, object]:
    case = selection.case
    input_slots = [_make_inputs(case, num_index_heads) for _ in range(slots)]
    wrappers = [BatchPrefillIndexerWithPagedKVCacheWrapper() for _ in range(slots)]
    for wrapper, inputs in zip(wrappers, input_slots, strict=True):
        _plan(wrapper, case, inputs, num_index_heads)

    def launch(slot: int) -> torch.Tensor:
        q, k_cache, _, _, _ = input_slots[slot]
        return wrappers[slot].run(q, k_cache)

    baseline_outputs = []
    for slot in range(slots):
        baseline_outputs.append(launch(slot).clone())
    torch.cuda.synchronize()

    graph_outputs = []
    rotation_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(rotation_graph):
        for slot in range(slots):
            graph_outputs.append(launch(slot))
    rotation_graph.replay()
    torch.cuda.synchronize()
    for expected, actual in zip(baseline_outputs, graph_outputs, strict=True):
        if not torch.equal(expected, actual):
            raise AssertionError(
                f"{case.case_id}: full TopK output changed after graph replay"
            )

    stats = _time_graph_rotation(
        rotation_graph,
        calls_per_replay=slots,
        warmup=warmup,
        replays=replays,
    )
    latency_us = float(stats["latency_us"])
    useful_flops, logical_bytes = _logical_work(case, num_index_heads)
    working_set_bytes = _working_set_bytes(case, num_index_heads)
    properties = torch.cuda.get_device_properties("cuda")
    row = {
        "case_id": case.case_id,
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
        "num_index_heads": num_index_heads,
        "useful_flops": useful_flops,
        "useful_tflops": useful_flops / latency_us / 1.0e6,
        "logical_bytes": logical_bytes,
        "logical_tb_s": logical_bytes / latency_us / 1.0e6,
        "per_slot_working_set_bytes": working_set_bytes,
        "cold_cache_slots": slots,
        "reuse_distance_bytes": (slots - 1) * working_set_bytes,
        "reuse_distance_over_l2": (slots - 1)
        * working_set_bytes
        / properties.L2_cache_size,
        "e2e_scope": "BatchPrefillIndexerWithPagedKVCacheWrapper.run",
        "correctness": "passed (full TopK bitwise graph replay)",
        **roofline_metrics(
            device_name=properties.name,
            useful_flops=useful_flops,
            logical_bytes=logical_bytes,
            latency_us=latency_us,
            compute_dtype="bf16",
        ),
        **stats,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--tag", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--slots", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument(
        "--num-index-heads",
        type=int,
        choices=M3_PAGED_DIRECT_SCORE_NUM_HEADS,
        default=DEFAULT_NUM_INDEX_HEADS,
        help="Number of local index heads: 1 or 4",
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.slots < 0 or args.warmup < 1 or args.replays < 2:
        parser.error("slots must be non-negative, warmup positive, and replays >= 2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError("BF16 prefill benchmark requires SM100 or SM103")
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
                num_index_heads=args.num_index_heads,
            )
        except ValueError as error:
            parser.error(str(error))
        row = run_case(
            selection,
            slots=slots,
            warmup=args.warmup,
            replays=args.replays,
            num_index_heads=args.num_index_heads,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        torch.cuda.empty_cache()

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
        "protocol": {
            "suite": args.suite,
            "nvidia_cutlass_dsl": importlib_metadata.version("nvidia-cutlass-dsl"),
            "formal_full_selection": is_formal,
            "e2e_scope": "public wrapper.run: proxy GEMM + TopK",
            "num_index_heads": args.num_index_heads,
            "compile_plan_allocate_capture_in_timing": False,
            "cuda_graph": True,
            "warmup_replays": args.warmup,
            "timed_replays": args.replays,
            "maximum_cv": MAX_CV,
            "maximum_timing_attempts": MAX_TIMING_ATTEMPTS,
            "cold_cache": "disjoint multi-tensor graph rotation above 2x L2",
        },
        "aggregate": {
            "formal": is_formal,
            "representative_cases": len(representative),
            "represented_calls": represented_calls,
            "weighted_mean_e2e_latency_us": weighted_latency,
        },
        "results": rows,
    }
    acceptance_failed = False
    if args.baseline is not None:
        result["acceptance"] = compare_e2e_results(
            rows,
            json.loads(args.baseline.read_text()),
            case_key="case_id",
            latency_key="latency_us",
            weight_key="representative_weight",
        )
        acceptance_failed = not result["acceptance"]["passed"]
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    if acceptance_failed:
        raise SystemExit("E2E performance acceptance failed")


if __name__ == "__main__":
    main()
