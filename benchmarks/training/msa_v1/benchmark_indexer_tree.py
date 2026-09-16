#!/usr/bin/env python3
"""Synthetic batch-1 Tree/Func indexer benchmark kept separate from CP training."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = REPO_ROOT / "training"
for source_root in (REPO_ROOT, TRAINING_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from msa_v1 import indexer_tree

Q_LEN = 7168
K_LENS = (32768, 65536, 131072)


def _causal_func(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    func = torch.full(
        (1, 1, 1, q_len + 256), k_len, dtype=torch.int32, device=device
    )
    func[0, 0, 0, :q_len] = torch.arange(
        q_len, dtype=torch.int32, device=device
    ) + (k_len - q_len + 1)
    return func


def _measure(call, warmup: int, iterations: int, rounds: int) -> list[float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) / iterations)
    return samples


def _flops(q_len: int, k_len: int) -> int:
    visible_pairs = q_len * (k_len - q_len) + q_len * (q_len + 1) // 2
    return 2 * 128 * 4 * visible_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--use-fp16-score", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    rows = []
    for k_len in K_LENS:
        plan = indexer_tree.compile_plan(
            _causal_func(Q_LEN, k_len, device), Q_LEN, k_len
        )
        q = torch.randn(Q_LEN, 4, 128, dtype=torch.bfloat16, device=device)
        k = torch.randn(k_len, 1, 128, dtype=torch.bfloat16, device=device)
        score = torch.empty(
            (plan.num_plan_tiles, 2, 128),
            dtype=torch.float16 if args.use_fp16_score else torch.float32,
            device=device,
        )
        block_sum = torch.empty(
            (plan.num_plan_tiles, 2, 128),
            dtype=torch.float32,
            device=device,
        )
        ids = torch.empty((4, Q_LEN, 16), dtype=torch.int32, device=device)
        lse = torch.empty((4, Q_LEN), dtype=torch.float32, device=device)
        call = lambda: indexer_tree.forward(
            q,
            k,
            plan,
            score_workspace=score,
            block_sum_workspace=block_sum,
            topk_indices=ids,
            selected_lse=lse,
            use_fp16_score=args.use_fp16_score,
        )
        samples = _measure(call, args.warmup, args.iterations, args.rounds)
        median_ms = statistics.median(samples)
        rows.append(
            {
                "case": f"q{Q_LEN}_k{k_len}",
                "runtime_ms": median_ms,
                "rounds_ms": samples,
                "useful_tflops": _flops(Q_LEN, k_len) / median_ms / 1e9,
            }
        )
    payload = {
        "kernel": "indexer_tree",
        "device": torch.cuda.get_device_name(device),
        "capability": torch.cuda.get_device_capability(device),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rounds": args.rounds,
        "use_fp16_score": args.use_fp16_score,
        "rows": rows,
    }
    encoded = json.dumps(payload, indent=2)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
