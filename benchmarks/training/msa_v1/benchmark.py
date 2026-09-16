#!/usr/bin/env python3
"""Single-GPU MSA v1 benchmark over Magi-dispatched CP rank metadata."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import cutlass
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = REPO_ROOT / "training"
for source_root in (REPO_ROOT, TRAINING_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from msa_v1 import attention, indexer, kl

from benchmarks.training.msa_v1.cases import (
    BENCHMARK_CASE_COUNT,
    EXPECTED_BENCHMARK_DIGEST,
    MsaBenchmarkCase,
    benchmark_manifest,
    benchmark_manifest_digest,
)
from datas.training.cases import (
    DEFAULT_SCENARIO,
    MSA_V1_SPEC,
    OFFICIAL_SCENARIOS,
    SYNTHETIC_SCENARIOS,
    CpRankCase,
    iter_rank_cases,
    make_torch_metadata,
    make_torch_topk,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel",
        choices=("attention-fwd", "attention-bwd", "indexer", "kl"),
        required=True,
    )
    workload = parser.add_mutually_exclusive_group()
    workload.add_argument(
        "--scenario",
        choices=tuple(OFFICIAL_SCENARIOS),
        default=DEFAULT_SCENARIO,
    )
    workload.add_argument(
        "--synthetic-scenario",
        choices=tuple(SYNTHETIC_SCENARIOS),
        help="run one supplemental synthetic workload instead of the official workload",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--case-suite", choices=("smoke", "full", "all"), default="smoke"
    )
    parser.add_argument("--benchmark-case-id", type=int)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument(
        "--use-fp16-score",
        action="store_true",
        help="store and rank MSA v1 indexer scores in FP16",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _time_ms(call, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def _make_inputs(case: CpRankCase, device: torch.device) -> dict[str, object]:
    metadata = make_torch_metadata(case, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(1_000_003 * case.case_id + case.rank)

    def random(shape: tuple[int, ...]) -> torch.Tensor:
        tensor = torch.empty(shape, dtype=torch.bfloat16, device=device)
        tensor.normal_(generator=generator)
        return tensor.mul_(0.2)

    return {
        **metadata,
        "q": random((case.total_q, 64, 128)),
        "k": random((case.total_kv, 4, 128)),
        "v": random((case.total_kv, 4, 128)),
        "qi": random((case.total_q, 4, 128)),
        "ki": random((case.total_kv, 1, 128)),
        "topk": make_torch_topk(case, device=device),
    }


def _benchmark_rank(
    kernel_name: str,
    case: CpRankCase,
    device: torch.device,
    warmup: int,
    iterations: int,
    benchmark_case_id: int | None,
    representative_weight: int,
    stratum: str,
    use_fp16_score: bool,
) -> dict[str, float | int | str | None]:
    values = _make_inputs(case, device)
    cu_q = values["cu_seqlens_q"]
    cu_kv = values["cu_seqlens_kv"]
    fragments = values["fragment_indices"]
    topk = values["topk"]
    preprocess_call = None
    base_preprocess_call = None
    cold_e2e_call = None

    if kernel_name in ("attention-fwd", "attention-bwd"):

        def preprocess_call():
            return attention.prepare(
                topk,
                cu_q,
                cu_kv,
                total_k=case.total_kv,
                total_rows=case.total_kv_rows,
                max_seqlen_q=case.max_seqlen_q,
                max_seqlen_k=case.max_seqlen_kv,
                fragment_indices=fragments,
            )

        prepared = preprocess_call()
        if kernel_name == "attention-fwd":
            call = lambda: attention.forward(
                values["q"], values["k"], values["v"], prepared
            )
            flops = 2 * (128 + 128) * 64 * case.sparse_attention_elements
            flops_model = "sparse useful: QK + PV"
        else:
            out, lse = attention.forward(
                values["q"],
                values["k"],
                values["v"],
                prepared,
                return_softmax_lse=True,
            )
            dout = torch.randn_like(out)
            call = lambda: attention.backward(
                values["q"],
                values["k"],
                values["v"],
                dout,
                out,
                lse,
                prepared,
            )
            flops = 2 * (3 * 128 + 2 * 128) * 64 * case.sparse_attention_elements
            flops_model = "sparse useful: QK recompute + dP + dV + dQ + dK"
        work_elements = case.sparse_attention_elements
    elif kernel_name == "indexer":
        schedule_storage = indexer.allocate_indexer_schedule(
            total_q=case.total_q,
            batch=len(case.fragments),
            device=device,
        )

        def preprocess_call():
            return indexer.prepare_indexer_schedule(
                cu_q,
                cu_kv,
                total_q=case.total_q,
                fragment_indices=fragments,
                schedule=schedule_storage,
            )

        schedule = preprocess_call()
        workspace = indexer.allocate_indexer_workspace(
            total_q=case.total_q,
            batch=len(case.fragments),
            max_seqlen_kv=case.max_seqlen_kv,
            device=device,
            use_fp16_score=use_fp16_score,
        )
        call = lambda: indexer.forward(
            values["qi"],
            values["ki"],
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=case.max_seqlen_q,
            max_seqlen_kv=case.max_seqlen_kv,
            fragment_indices=fragments,
            schedule=schedule,
            workspace=workspace,
            use_fp16_score=use_fp16_score,
        )

        def cold_e2e_call():
            preprocess_call()
            return call()

        work_elements = case.causal_elements
        flops = 2 * 128 * 4 * work_elements
        flops_model = "dense causal K1 GEMM; runtime includes K1 + K2 + K3"
    else:
        schedule = indexer.prepare_indexer_schedule(
            cu_q,
            cu_kv,
            total_q=case.total_q,
            fragment_indices=fragments,
        )
        workspace = indexer.allocate_indexer_workspace(
            total_q=case.total_q,
            batch=len(case.fragments),
            max_seqlen_kv=case.max_seqlen_kv,
            device=device,
        )
        topk, indexer_lse = indexer.forward(
            values["qi"],
            values["ki"],
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=case.max_seqlen_q,
            max_seqlen_kv=case.max_seqlen_kv,
            fragment_indices=fragments,
            schedule=schedule,
            workspace=workspace,
        )

        def prepare(prepare_kl_schedule: bool):
            return attention.prepare(
                topk,
                cu_q,
                cu_kv,
                total_k=case.total_kv,
                total_rows=case.total_kv_rows,
                max_seqlen_q=case.max_seqlen_q,
                max_seqlen_k=case.max_seqlen_kv,
                fragment_indices=fragments,
                prepare_kl_schedule=prepare_kl_schedule,
            )

        base_preprocess_call = lambda: prepare(False)
        preprocess_call = lambda: prepare(True)
        prepared = preprocess_call()
        _, teacher_lse = attention.forward(
            values["q"],
            values["k"],
            values["v"],
            prepared,
            return_softmax_lse=True,
        )
        call = lambda: kl.backward(
            values["q"],
            values["k"],
            teacher_lse,
            values["qi"],
            values["ki"],
            indexer_lse,
            prepared,
        )
        work_elements = case.sparse_attention_elements
        flops = 2 * 128 * (64 + 3 * 4) * work_elements
        flops_model = "sparse useful: teacher QK + student QK + dQI + dKI"

    runtime_ms = _time_ms(call, warmup, iterations)
    row: dict[str, float | int | str | None] = {
        "scenario": case.scenario,
        "benchmark_case_id": benchmark_case_id,
        "case_id": case.case_id,
        "rank": case.rank,
        "representative_weight": representative_weight,
        "stratum": stratum,
        "runtime_ms": runtime_ms,
        "useful_flops": flops,
        "useful_tflops": flops / runtime_ms / 1e9,
        "flops_model": flops_model,
        "work_elements": work_elements,
        "causal_elements": case.causal_elements,
        "sparse_attention_elements": case.sparse_attention_elements,
        "physical_kv_tokens": case.total_kv,
        "use_fp16_score": use_fp16_score,
    }
    if preprocess_call is not None:
        preprocess_runtime_ms = _time_ms(preprocess_call, warmup, iterations)
        row["preprocess_runtime_ms"] = preprocess_runtime_ms
        if base_preprocess_call is not None:
            base_preprocess_runtime_ms = _time_ms(
                base_preprocess_call,
                warmup,
                iterations,
            )
            incremental_preprocess_runtime_ms = (
                preprocess_runtime_ms - base_preprocess_runtime_ms
            )
            module_runtime_ms = runtime_ms + incremental_preprocess_runtime_ms
            row["base_preprocess_runtime_ms"] = base_preprocess_runtime_ms
            row["incremental_preprocess_runtime_ms"] = incremental_preprocess_runtime_ms
            row["module_runtime_ms"] = module_runtime_ms
            row["module_useful_tflops"] = flops / module_runtime_ms / 1e9
    if cold_e2e_call is not None:
        row["cold_e2e_runtime_ms"] = _time_ms(cold_e2e_call, warmup, iterations)
    return row


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _select_rank_cases(
    args: argparse.Namespace,
) -> tuple[str, list[MsaBenchmarkCase], str | None]:
    if args.case_limit is not None and args.case_limit < 1:
        raise ValueError("--case-limit must be positive")
    if args.benchmark_case_id is not None:
        if args.case_suite == "all" or args.case_limit is not None:
            raise ValueError("--benchmark-case-id requires the smoke or full suite")
        if not 0 <= args.benchmark_case_id < BENCHMARK_CASE_COUNT:
            raise ValueError("--benchmark-case-id is out of range")
    scenario = args.synthetic_scenario or args.scenario
    synthetic = args.synthetic_scenario is not None
    if args.case_limit is not None:
        cases = [
            MsaBenchmarkCase(-1, case, 1, "unweighted_debug")
            for case in iter_rank_cases(
                scenario,
                sparse_spec=MSA_V1_SPEC,
                allow_synthetic=synthetic,
            )
            if case.case_id < args.case_limit
        ]
        return "global-prefix", cases, None
    if args.case_suite == "all":
        return (
            "all",
            [
                MsaBenchmarkCase(-1, case, 1, "unweighted_all")
                for case in iter_rank_cases(
                    scenario,
                    sparse_spec=MSA_V1_SPEC,
                    allow_synthetic=synthetic,
                )
            ],
            None,
        )
    manifest = benchmark_manifest(scenario, synthetic=synthetic)
    digest = benchmark_manifest_digest(manifest)
    if not synthetic and digest != EXPECTED_BENCHMARK_DIGEST:
        raise RuntimeError(f"{scenario} benchmark manifest digest changed")
    if args.benchmark_case_id is not None:
        manifest = (manifest[args.benchmark_case_id],)
    elif args.case_suite == "smoke":
        manifest = manifest[:16]
    return args.case_suite, list(manifest), digest


def main() -> None:
    args = _parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    if args.use_fp16_score and args.kernel != "indexer":
        raise ValueError("--use-fp16-score is supported only with --kernel indexer")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    print(
        f"CUTLASS DSL {cutlass.__version__}; device={torch.cuda.get_device_name(device)}"
    )
    started = time.time()
    suite, selected, digest = _select_rank_cases(args)
    rows = [
        _benchmark_rank(
            args.kernel,
            item.rank_case,
            device,
            args.warmup,
            args.iterations,
            None if item.benchmark_case_id < 0 else item.benchmark_case_id,
            item.representative_weight,
            item.stratum,
            args.use_fp16_score,
        )
        for item in selected
    ]
    runtimes = [float(row["runtime_ms"]) for row in rows]
    useful_flops = [int(row["useful_flops"]) for row in rows]
    weights = [int(row["representative_weight"]) for row in rows]
    weighted_runtime_ms = sum(
        runtime * weight for runtime, weight in zip(runtimes, weights, strict=True)
    )
    weighted_flops = sum(
        flops * weight for flops, weight in zip(useful_flops, weights, strict=True)
    )
    scenario = args.synthetic_scenario or args.scenario
    payload = {
        "kernel": args.kernel,
        "scenario": scenario,
        "workload_kind": "synthetic" if args.synthetic_scenario else "real",
        "case_suite": suite,
        "formal_full_selection": suite == "full" and len(rows) == 32,
        "manifest_digest": digest,
        "device": torch.cuda.get_device_name(device),
        "capability": torch.cuda.get_device_capability(device),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "use_fp16_score": args.use_fp16_score,
        "compile_and_benchmark_seconds": time.time() - started,
        "summary_runtime_ms": {
            "mean": statistics.fmean(runtimes),
            "weighted_mean": weighted_runtime_ms / sum(weights),
            "p50": _percentile(runtimes, 0.50),
            "p90": _percentile(runtimes, 0.90),
            "p99": _percentile(runtimes, 0.99),
            "max": max(runtimes),
        },
        "summary_useful_tflops": {
            "rank_mean": statistics.fmean(float(row["useful_tflops"]) for row in rows),
            "weighted_aggregate": weighted_flops / weighted_runtime_ms / 1e9,
        },
        "represented_calls": sum(weights),
        "rows": rows,
    }
    if args.kernel == "kl":
        module_runtimes = [float(row["module_runtime_ms"]) for row in rows]
        weighted_module_runtime_ms = sum(
            runtime * weight
            for runtime, weight in zip(module_runtimes, weights, strict=True)
        )
        incremental_preprocess_runtimes = [
            float(row["incremental_preprocess_runtime_ms"]) for row in rows
        ]
        payload["summary_incremental_preprocess_runtime_ms"] = {
            "mean": statistics.fmean(incremental_preprocess_runtimes),
            "p50": _percentile(incremental_preprocess_runtimes, 0.50),
            "p90": _percentile(incremental_preprocess_runtimes, 0.90),
            "p99": _percentile(incremental_preprocess_runtimes, 0.99),
            "max": max(incremental_preprocess_runtimes),
        }
        payload["summary_module_runtime_ms"] = {
            "mean": statistics.fmean(module_runtimes),
            "weighted_mean": weighted_module_runtime_ms / sum(weights),
            "p50": _percentile(module_runtimes, 0.50),
            "p90": _percentile(module_runtimes, 0.90),
            "p99": _percentile(module_runtimes, 0.99),
            "max": max(module_runtimes),
        }
        payload["summary_module_useful_tflops"] = {
            "rank_mean": statistics.fmean(
                float(row["module_useful_tflops"]) for row in rows
            ),
            "weighted_aggregate": weighted_flops
            / weighted_module_runtime_ms
            / 1e9,
        }
    encoded = json.dumps(payload, indent=2)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
