#!/usr/bin/env python3
"""CUDA Graph E2E benchmark for Q8KV8 paged sparse prefill."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from benchmarks.inference.msa_v1.acceptance import compare_e2e_results  # noqa: E402
from benchmarks.inference.msa_v1.attention.prefill.q8kv8.cases import (  # noqa: E402
    HEAD_DIM,
    PAGE_SIZE,
    Q_HEADS,
    TOPK,
    PrefillCase,
    real_prefill_cases,
    warmup_case,
)
from benchmarks.inference.msa_v1.hardware import roofline_metrics  # noqa: E402
from datas.inference.tensors import (  # noqa: E402
    cumulative_lengths,
    make_attention_topk,
    make_disjoint_page_table,
)
from inference.msa_v1.attention.prefill.q8kv8 import (  # noqa: E402
    BatchPrefillWithPagedKVCacheWrapper,
)

KV_HEADS = 4
MAX_CV = 0.03
MAX_TIMING_ATTEMPTS = 3
KV_HEAD_PAGE_BYTES = 2 * PAGE_SIZE * HEAD_DIM


def _make_finite_e4m3(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    bits = torch.randint(
        0,
        128,
        shape,
        dtype=torch.uint8,
        generator=generator,
        device=device,
    )
    sign = torch.bitwise_left_shift(torch.bitwise_and(bits, 0x40), 1)
    bits.bitwise_and_(0x3F)
    bits.bitwise_or_(sign)
    return bits.view(torch.float8_e4m3fn)


def _make_case_tensors(
    case: PrefillCase,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(case.seed)
    page_table, physical_pages = make_disjoint_page_table(
        case.final_kv_lens,
        max_cols=case.max_cols,
        generator=generator,
        device=device,
    )
    q = _make_finite_e4m3(
        (case.total_q, Q_HEADS, HEAD_DIM),
        generator=generator,
        device=device,
    )
    k_cache = _make_finite_e4m3(
        (physical_pages, KV_HEADS, PAGE_SIZE, HEAD_DIM),
        generator=generator,
        device=device,
    )
    v_cache = _make_finite_e4m3(
        k_cache.shape,
        generator=generator,
        device=device,
    )
    cu_seqlens_q = torch.tensor(
        cumulative_lengths(case.query_lens), dtype=torch.int32, device=device
    )
    cu_seqlens_k = torch.tensor(
        cumulative_lengths(case.final_kv_lens), dtype=torch.int32, device=device
    )
    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "page_table": page_table,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "topk": make_attention_topk(case, device=device),
    }


def _plan(
    wrapper: BatchPrefillWithPagedKVCacheWrapper,
    case: PrefillCase,
    tensors: dict[str, torch.Tensor],
) -> None:
    wrapper.plan(
        tensors["topk"],
        tensors["cu_seqlens_q"],
        tensors["cu_seqlens_k"],
        tensors["page_table"],
        total_k=sum(case.final_kv_lens),
        total_rows=case.num_active_pages,
        max_seqlen_q=case.max_query_len,
        max_seqlen_k=case.max_final_kv,
    )


def _logical_work(case: PrefillCase) -> tuple[int, int]:
    selected_pages = 0
    for query_len, prefix_len in zip(
        case.query_lens, case.prefix_lens, strict=True
    ):
        for query_idx in range(query_len):
            local_page = (prefix_len + query_idx) // PAGE_SIZE
            selected_pages += min(TOPK, local_page + 1)
    logical_bytes = (
        selected_pages * KV_HEADS * KV_HEAD_PAGE_BYTES
        + case.total_q * Q_HEADS * HEAD_DIM
        + case.total_q * Q_HEADS * HEAD_DIM * 2
        + case.total_q * Q_HEADS * 4
    )
    return case.useful_flops, logical_bytes


def _resolve_slots(
    case: PrefillCase,
    *,
    requested_slots: int,
    l2_bytes: int,
) -> int:
    _, logical_bytes = _logical_work(case)
    minimum_slots = (2 * l2_bytes) // logical_bytes + 2
    if requested_slots and requested_slots < minimum_slots:
        raise ValueError(
            f"--slots={requested_slots} is too small; case {case.name} requires "
            f"at least {minimum_slots} disjoint tensor slots"
        )
    return requested_slots or minimum_slots


def _time_graphs(
    graphs: list[torch.cuda.CUDAGraph],
    *,
    warmup: int,
    replays: int,
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
            samples.append(start.elapsed_time(end) * 1000.0 / len(graphs))
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


def _warm_compilation(device: torch.device) -> None:
    case = warmup_case()
    tensors = _make_case_tensors(case, device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    _plan(wrapper, case, tensors)
    wrapper.run(
        tensors["q"],
        (tensors["k_cache"], tensors["v_cache"]),
        return_lse=True,
    )
    torch.cuda.synchronize()
    del wrapper, tensors
    torch.cuda.empty_cache()


@torch.inference_mode()
def run_case(
    case: PrefillCase,
    *,
    slots: int,
    warmup: int,
    replays: int,
) -> dict[str, object]:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    tensor_slots = [_make_case_tensors(case, device) for _ in range(slots)]
    wrappers = [BatchPrefillWithPagedKVCacheWrapper() for _ in range(slots)]
    for wrapper, tensors in zip(wrappers, tensor_slots, strict=True):
        _plan(wrapper, case, tensors)

    def launch(slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = tensor_slots[slot]
        return wrappers[slot].run(
            tensors["q"],
            (tensors["k_cache"], tensors["v_cache"]),
            return_lse=True,
        )

    baseline_outputs = []
    for slot in range(slots):
        out, lse = launch(slot)
        baseline_outputs.append((out.clone(), lse.clone()))
    torch.cuda.synchronize()
    for out, lse in baseline_outputs:
        if not bool(torch.isfinite(out).all()) or not bool(torch.isfinite(lse).all()):
            raise AssertionError(f"{case.name}: eager output contains non-finite values")

    graphs = []
    graph_outputs = []
    for slot in range(slots):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = launch(slot)
        graphs.append(graph)
        graph_outputs.append(outputs)
    for graph in graphs:
        graph.replay()
    torch.cuda.synchronize()
    for (expected_out, expected_lse), (out, lse) in zip(
        baseline_outputs, graph_outputs, strict=True
    ):
        if not torch.equal(out, expected_out):
            raise AssertionError(f"{case.name}: full output changed after graph replay")
        if not torch.equal(lse, expected_lse):
            raise AssertionError(f"{case.name}: full LSE changed after graph replay")

    stats = _time_graphs(graphs, warmup=warmup, replays=replays)
    latency_us = float(stats["latency_us"])
    useful_flops, logical_bytes = _logical_work(case)
    properties = torch.cuda.get_device_properties(device)
    row = {
        "case": case.name,
        "tags": case.tags,
        "representative_weight": case.representative_weight,
        "stratum": case.stratum,
        "count": case.shape.count,
        "batch_size": case.batch,
        "query_lens": case.query_lens,
        "prefix_lens": case.prefix_lens,
        "final_kv_lens": case.final_kv_lens,
        "total_q": case.total_q,
        "active_pages": case.num_active_pages,
        "topk": TOPK,
        "average_selected_tokens_per_query": case.average_selected_tokens,
        "sparse_density_vs_dense_causal": case.sparse_density,
        "useful_flops": useful_flops,
        "useful_tflops": useful_flops / latency_us / 1.0e6,
        "query_tokens_per_second": case.total_q / latency_us * 1.0e6,
        "logical_bytes": logical_bytes,
        "logical_tb_s": logical_bytes / latency_us / 1.0e6,
        "cold_cache_slots": slots,
        "reuse_distance_bytes": (slots - 1) * logical_bytes,
        "reuse_distance_over_l2": (slots - 1)
        * logical_bytes
        / properties.L2_cache_size,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "correctness": "passed (finite full output + bitwise full graph replay)",
        **roofline_metrics(
            device_name=properties.name,
            useful_flops=useful_flops,
            logical_bytes=logical_bytes,
            latency_us=latency_us,
        ),
        **stats,
    }
    del graphs, graph_outputs, baseline_outputs, wrappers, tensor_slots
    torch.cuda.empty_cache()
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
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.slots < 0 or args.warmup < 1 or args.replays < 2:
        parser.error("slots must be non-negative, warmup positive, and replays >= 2")
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("Q8KV8 performance benchmark requires GB300/SM103")

    device = torch.device("cuda")
    _warm_compilation(device)
    cases = real_prefill_cases(args.suite)
    if args.case_id:
        requested = set(args.case_id)
        cases = tuple(case for case in cases if case.name in requested)
    if args.tag:
        requested_tags = set(args.tag)
        cases = tuple(case for case in cases if requested_tags & set(case.tags))
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        parser.error("benchmark selection is empty")

    rows = []
    for case in cases:
        try:
            slots = _resolve_slots(
                case,
                requested_slots=args.slots,
                l2_bytes=torch.cuda.get_device_properties(device).L2_cache_size,
            )
        except ValueError as error:
            parser.error(str(error))
        row = run_case(case, slots=slots, warmup=args.warmup, replays=args.replays)
        rows.append(row)
        print(json.dumps(row))

    is_formal = (
        args.suite == "full"
        and not args.case_id
        and not args.tag
        and args.limit is None
        and len(rows) == 128
    )
    representative = (
        [row for row in rows if "representative" in row["tags"]]
        if is_formal
        else []
    )
    represented_calls = sum(
        int(row["representative_weight"]) for row in representative
    )
    weighted_latency = (
        sum(
            int(row["representative_weight"]) * float(row["latency_us"])
            for row in representative
        )
        / represented_calls
        if represented_calls
        else None
    )
    properties = torch.cuda.get_device_properties(device)
    result = {
        "device": properties.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "protocol": {
            "suite": args.suite,
            "formal_full_selection": is_formal,
            "e2e_scope": "public wrapper.run: attention K1 + split combine",
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
            case_key="case",
            latency_key="latency_us",
            weight_key="representative_weight",
        )
        acceptance_failed = not result["acceptance"]["passed"]
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.out}")
    if acceptance_failed:
        raise SystemExit("E2E performance acceptance failed")


if __name__ == "__main__":
    main()
