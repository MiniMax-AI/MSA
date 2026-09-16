#!/usr/bin/env python3
"""CUDA Graph E2E benchmark for BF16 paged sparse prefill."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from benchmarks.inference.msa_v1.acceptance import compare_e2e_results
from benchmarks.inference.msa_v1.attention.prefill.bf16.cases import (
    HEAD_DIM,
    PAGE_SIZE,
    TOPK,
    PrefillCase,
    real_prefill_cases,
    warmup_case,
)
from benchmarks.inference.msa_v1.hardware import roofline_metrics
from datas.inference.tensors import (
    cumulative_lengths,
    make_attention_topk,
    make_disjoint_page_table,
)
from inference.msa_v1.attention.prefill.bf16 import (
    BatchPrefillWithPagedKVCacheWrapper,
)

GQA_GROUP_SIZE = 16
DEFAULT_NUM_KV_HEADS = 4
MAX_CV = 0.03
MAX_TIMING_ATTEMPTS = 3


def _make_bf16(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return (torch.randn(shape, generator=generator, device=device) * 0.25).to(
        torch.bfloat16
    )


def _make_case_tensors(
    case: PrefillCase,
    device: torch.device,
    num_kv_heads: int,
) -> dict[str, torch.Tensor]:
    num_q_heads = num_kv_heads * GQA_GROUP_SIZE
    generator = torch.Generator(device=device).manual_seed(case.seed)
    page_table, physical_pages = make_disjoint_page_table(
        case.final_kv_lens,
        max_cols=case.max_cols,
        generator=generator,
        device=device,
    )
    q = _make_bf16(
        (case.total_q, num_q_heads, HEAD_DIM),
        generator=generator,
        device=device,
    )
    k_cache = _make_bf16(
        (physical_pages, num_kv_heads, PAGE_SIZE, HEAD_DIM),
        generator=generator,
        device=device,
    )
    v_cache = _make_bf16(
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
        "topk": make_attention_topk(
            case,
            device=device,
            kv_heads=num_kv_heads,
        ),
    }


def _plan(
    wrapper: BatchPrefillWithPagedKVCacheWrapper,
    case: PrefillCase,
    tensors: dict[str, torch.Tensor],
    num_kv_heads: int,
) -> None:
    wrapper.plan(
        tensors["topk"],
        tensors["cu_seqlens_q"],
        tensors["cu_seqlens_k"],
        tensors["page_table"],
        num_q_heads=num_kv_heads * GQA_GROUP_SIZE,
        num_kv_heads=num_kv_heads,
        total_k=sum(case.final_kv_lens),
        total_rows=case.num_active_pages,
        max_seqlen_q=case.max_query_len,
        max_seqlen_k=case.max_final_kv,
    )


def _logical_work(case: PrefillCase, num_kv_heads: int) -> tuple[int, int]:
    useful_flops = case.useful_flops * num_kv_heads // DEFAULT_NUM_KV_HEADS
    return useful_flops, _working_set_bytes(case, num_kv_heads)


def _working_set_bytes(case: PrefillCase, num_kv_heads: int) -> int:
    num_q_heads = num_kv_heads * GQA_GROUP_SIZE
    physical_pages = case.num_active_pages + 1
    q_bytes = case.total_q * num_q_heads * HEAD_DIM * 2
    kv_bytes = physical_pages * num_kv_heads * PAGE_SIZE * HEAD_DIM * 4
    topk_bytes = num_kv_heads * case.total_q * TOPK * 4
    page_table_bytes = case.batch * case.max_cols * 4
    sequence_bytes = 2 * (case.batch + 1) * 4
    partial_bytes = TOPK * case.total_q * num_q_heads * (HEAD_DIM * 2 + 4)
    output_bytes = case.total_q * num_q_heads * (HEAD_DIM * 2 + 4)
    csr_bytes = num_kv_heads * case.total_q * TOPK * 8
    return (
        q_bytes
        + kv_bytes
        + topk_bytes
        + page_table_bytes
        + sequence_bytes
        + partial_bytes
        + output_bytes
        + csr_bytes
    )


def _resolve_slots(
    case: PrefillCase,
    *,
    requested_slots: int,
    l2_bytes: int,
    num_kv_heads: int,
) -> int:
    working_set_bytes = _working_set_bytes(case, num_kv_heads)
    minimum_slots = (2 * l2_bytes) // working_set_bytes + 2
    if requested_slots and requested_slots < minimum_slots:
        raise ValueError(
            f"--slots={requested_slots} is too small; case {case.name} requires "
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


def _warm_compilation(device: torch.device, num_kv_heads: int) -> None:
    case = warmup_case()
    tensors = _make_case_tensors(case, device, num_kv_heads)
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    _plan(wrapper, case, tensors, num_kv_heads)
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
    num_kv_heads: int,
) -> dict[str, object]:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    tensor_slots = [
        _make_case_tensors(case, device, num_kv_heads) for _ in range(slots)
    ]
    wrappers = [BatchPrefillWithPagedKVCacheWrapper() for _ in range(slots)]
    for wrapper, tensors in zip(wrappers, tensor_slots, strict=True):
        _plan(wrapper, case, tensors, num_kv_heads)

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
            raise AssertionError(
                f"{case.name}: eager output contains non-finite values"
            )

    graph_outputs = []
    rotation_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(rotation_graph):
        for slot in range(slots):
            graph_outputs.append(launch(slot))
    rotation_graph.replay()
    torch.cuda.synchronize()
    for (expected_out, expected_lse), (out, lse) in zip(
        baseline_outputs, graph_outputs, strict=True
    ):
        if not torch.equal(out, expected_out):
            raise AssertionError(f"{case.name}: full output changed after graph replay")
        if not torch.equal(lse, expected_lse):
            raise AssertionError(f"{case.name}: full LSE changed after graph replay")

    stats = _time_graph_rotation(
        rotation_graph,
        calls_per_replay=slots,
        warmup=warmup,
        replays=replays,
    )
    latency_us = float(stats["latency_us"])
    useful_flops, logical_bytes = _logical_work(case, num_kv_heads)
    working_set_bytes = _working_set_bytes(case, num_kv_heads)
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
        "num_q_heads": num_kv_heads * GQA_GROUP_SIZE,
        "num_kv_heads": num_kv_heads,
        "average_selected_tokens_per_query": case.average_selected_tokens,
        "sparse_density_vs_dense_causal": case.sparse_density,
        "useful_flops": useful_flops,
        "useful_tflops": useful_flops / latency_us / 1.0e6,
        "query_tokens_per_second": case.total_q / latency_us * 1.0e6,
        "logical_bytes": logical_bytes,
        "logical_tb_s": logical_bytes / latency_us / 1.0e6,
        "per_slot_working_set_bytes": working_set_bytes,
        "cold_cache_slots": slots,
        "reuse_distance_bytes": (slots - 1) * working_set_bytes,
        "reuse_distance_over_l2": (slots - 1)
        * working_set_bytes
        / properties.L2_cache_size,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "correctness": "passed (finite full output + bitwise full graph replay)",
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
        "--num-kv-heads",
        type=int,
        choices=(1, 4),
        default=DEFAULT_NUM_KV_HEADS,
        help="Local KV heads: 4 for TP1 and 1 for TP4",
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.slots < 0 or args.warmup < 1 or args.replays < 2:
        parser.error("slots must be non-negative, warmup positive, and replays >= 2")
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("BF16 performance benchmark requires GB300/SM103")

    device = torch.device("cuda")
    _warm_compilation(device, args.num_kv_heads)
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
                num_kv_heads=args.num_kv_heads,
            )
        except ValueError as error:
            parser.error(str(error))
        row = run_case(
            case,
            slots=slots,
            warmup=args.warmup,
            replays=args.replays,
            num_kv_heads=args.num_kv_heads,
        )
        rows.append(row)
        print(json.dumps(row))
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
    properties = torch.cuda.get_device_properties(device)
    result = {
        "device": properties.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "protocol": {
            "suite": args.suite,
            "nvidia_cutlass_dsl": importlib_metadata.version("nvidia-cutlass-dsl"),
            "formal_full_selection": is_formal,
            "e2e_scope": "public wrapper.run: attention K1 + split combine",
            "tp_degree": DEFAULT_NUM_KV_HEADS // args.num_kv_heads,
            "num_q_heads": args.num_kv_heads * GQA_GROUP_SIZE,
            "num_kv_heads": args.num_kv_heads,
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
