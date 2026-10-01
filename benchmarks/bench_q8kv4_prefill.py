# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Q8KV4 paged sparse prefill benchmark on an NVFP4 KV cache.

Per case: build the inputs of tests/q8kv4_prefill (chunked-prefill requests, random codes and
block scales, scattered pages, ascending TopK lists with the query's own page last) packed in the
vLLM page layout, then run one layer through ``fmha_sm100`` on each backend:

- ``q8kv4``: E4M3 Q on the Q8KV4 prefill kernel (``prefill_backend="q8kv4"``),
- ``cute_e4m3``: E4M3 Q on the CuTe-DSL NVFP4 kernel (``prefill_backend="cute_dsl"``),
- ``cute_bf16``: BF16 Q on the CuTe-DSL NVFP4 kernel, the path BF16-Q prefill takes today.

Every backend's output is compared with ``cute_bf16`` (the closest to exact) by mean, 99.99th
percentile and max |diff|. Timings (plans built outside, lists passed per call as in serving):
``call`` is the median wall time of a synchronized ``fmha_sm100`` call (CSR build, forward,
combine and any host work), ``gpu`` the profiler's kernel time per call split into the forward
kernel and the rest (CSR build, schedule, combine). ``TFLOP/s`` counts the useful QK and PV
work of the visible keys over the forward kernel's time.

Suites: ``smoke`` (3 cases) and ``full`` (fresh and long-prefix chunks, batch 1 to 8, both head
layouts). Requires an exclusive GPU.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
# Import the test helpers as the `q8kv4_prefill` package, the name pytest gives them.
sys.path.insert(0, str(REPO_ROOT / "tests"))

from fmha_sm100.api import fmha_sm100, fmha_sm100_plan  # noqa: E402
from q8kv4_prefill.cases import (  # noqa: E402
    PAGE_SIZE, SM_SCALE, PrefillCase, global_scale, make_inputs, pack_head_slot_pages, query_positions,
)

BACKENDS = ("q8kv4", "cute_e4m3", "cute_bf16")
# name: (q_lens, k_lens, kv_heads)
SMOKE_SUITE = {
    "b1_q4k_fresh_kv4": ((4_096,), (4_096,), 4),
    "b1_q8k_p32k_kv4": ((8_192,), (40_960,), 4),
    "b4_q1k_p16k_kv1": ((1_024,) * 4, (17_408,) * 4, 1),
}
FULL_SUITE = {
    "b1_q2k_fresh_kv4": ((2_048,), (2_048,), 4),
    "b1_q8k_fresh_kv4": ((8_192,), (8_192,), 4),
    "b1_q16k_fresh_kv4": ((16_384,), (16_384,), 4),
    "b1_q8k_p32k_kv4": ((8_192,), (40_960,), 4),
    "b1_q8k_p120k_kv4": ((8_192,), (131_072,), 4),
    "b1_q2k_p200k_kv4": ((2_048,), (202_048,), 4),
    "b2_q4k_p16k_kv4": ((4_096,) * 2, (20_480,) * 2, 4),
    "b4_q1k_p64k_kv4": ((1_024,) * 4, (66_560,) * 4, 4),
    "b8_q512_p8k_kv4": ((512,) * 8, (8_704,) * 8, 4),
    "b8_ragged_kv4": ((64, 300, 1_000, 2_000, 128, 700, 4_000, 33),
                      (64, 20_300, 1_000, 52_000, 9_128, 700, 100_000, 33), 4),
    "b1_q8k_fresh_kv1": ((8_192,), (8_192,), 1),
    "b1_q8k_p120k_kv1": ((8_192,), (131_072,), 1),
    "b4_q1k_p16k_kv1": ((1_024,) * 4, (17_408,) * 4, 1),
}
CALLS = 20
PROFILED_CALLS = 5


def visible_keys(case: PrefillCase, topk_indices: torch.Tensor) -> int:
    """Keys each (query, KV head) actually attends to, summed: the useful work."""
    positions = torch.tensor(query_positions(case), device=topk_indices.device)
    kv_len = torch.repeat_interleave(torch.tensor(case.k_lens, device=positions.device),
                                     torch.tensor(case.q_lens, device=positions.device))
    pages = topk_indices.long()  # [H, total_q, topk]
    start = pages * PAGE_SIZE
    visible = torch.minimum(kv_len[None, :, None], positions[None, :, None] + 1) - start
    return int(visible.clamp(0, PAGE_SIZE).masked_fill(pages < 0, 0).sum())


class Case:
    def __init__(self, name, q_lens, k_lens, kv_heads, seed, device):
        self.case = PrefillCase(name, tuple(q_lens), tuple(k_lens), topk=16, num_kv_heads=kv_heads,
                                seed=seed, ascending=True)
        self.inputs = make_inputs(self.case, device)
        self.k_packed, self.v_packed = pack_head_slot_pages(self.inputs)
        self.unit = global_scale(1.0, device)
        self.q_bf16 = self.inputs.q.to(torch.bfloat16)
        self.lists = self.inputs.kv_block_indexes
        self.flops = 4 * 128 * 16 * visible_keys(self.case, self.inputs.topk_indices)

    def plan(self, backend):
        case = self.case
        kv = torch.tensor(case.k_lens, dtype=torch.int32)
        qo = torch.tensor(case.q_lens, dtype=torch.int32)
        return fmha_sm100_plan(qo, kv, case.num_q_heads, num_kv_heads=case.num_kv_heads,
                               causal=True, qo_offset=kv - qo, page_size=PAGE_SIZE,
                               kv_block_num=case.topk, sparse_kernel_mode="prefill",
                               split_prefill_decode=False,
                               prefill_backend="q8kv4" if backend == "q8kv4" else "cute_dsl")

    def caller(self, backend):
        plan = self.plan(backend)
        q = self.q_bf16 if backend == "cute_bf16" else self.inputs.q
        out = torch.empty((q.shape[0], q.shape[1], 128), dtype=torch.bfloat16, device=q.device)

        def call():
            fmha_sm100(q, self.k_packed, self.v_packed, plan, kv_indices=self.inputs.kv_indices,
                       kv_block_indexes=self.lists, out=out, sm_scale=SM_SCALE,
                       k_scale=self.unit, v_scale=self.unit)
            return out

        return call


def time_calls(call) -> float:
    samples = []
    for _ in range(CALLS):
        torch.cuda.synchronize()
        started = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples) * 1e3


def profile_kernels(call) -> tuple[float, float, list[str]]:
    """(forward ms, other ms, forward kernel names) per call from the profiler."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(PROFILED_CALLS):
            call()
        torch.cuda.synchronize()
    forward = other = 0.0
    names = set()
    for event in prof.key_averages():
        # Kernel entries only: CPU-op entries repeat their kernels' time.
        if event.device_type != DeviceType.CUDA or event.device_time_total <= 0:
            continue
        device_us = event.device_time_total
        name = event.key
        if "prefill_attention_kernel" in name or "sparse_forward" in name.lower() or "SparseAttentionForward" in name:
            forward += device_us
            names.add(name[:80])
        else:
            other += device_us
    return forward / PROFILED_CALLS / 1e3, other / PROFILED_CALLS / 1e3, sorted(names)


def compare(out, reference) -> dict:
    diff = (out.float() - reference.float()).abs().flatten()
    tail = torch.kthvalue(diff, max(1, int(diff.numel() * 0.9999))).values.item()
    return {"mean": diff.mean().item(), "p9999": tail, "max": diff.max().item()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None, help="write the results here")
    args = parser.parse_args()
    device = torch.device("cuda")
    suite = SMOKE_SUITE if args.suite == "smoke" else FULL_SUITE
    print(f"device {torch.cuda.get_device_name(device)}, suite {args.suite}, {len(suite)} cases")
    header = (f"{'case':22s} {'total_q':>7s} {'backend':10s} {'call ms':>8s} {'fwd ms':>7s} "
              f"{'other ms':>8s} {'TFLOP/s':>7s} {'mean|d|':>8s} {'p99.99':>7s} {'max|d|':>7s}")
    print(header)
    results = []
    for index, (name, (q_lens, k_lens, kv_heads)) in enumerate(suite.items()):
        case = Case(name, q_lens, k_lens, kv_heads, args.seed + index, device)
        callers = {backend: case.caller(backend) for backend in BACKENDS}
        outputs = {backend: callers[backend]().clone() for backend in BACKENDS}  # warm-up + JIT
        for backend in BACKENDS:
            call = callers[backend]
            call_ms = time_calls(call)
            forward_ms, other_ms, kernels = profile_kernels(call)
            error = compare(outputs[backend], outputs["cute_bf16"])
            tflops = case.flops / (forward_ms * 1e-3) / 1e12 if forward_ms > 0 else float("nan")
            row = {"case": name, "total_q": case.case.total_q, "backend": backend,
                   "call_ms": call_ms, "forward_ms": forward_ms, "other_ms": other_ms,
                   "tflops": tflops, "error_vs_cute_bf16": error, "forward_kernels": kernels}
            results.append(row)
            print(f"{name:22s} {case.case.total_q:7d} {backend:10s} {call_ms:8.3f} {forward_ms:7.3f} "
                  f"{other_ms:8.3f} {tflops:7.1f} {error['mean']:8.2e} {error['p9999']:7.4f} "
                  f"{error['max']:7.4f}", flush=True)
        del case, callers, outputs
        torch.cuda.empty_cache()
    for backend in BACKENDS[1:]:
        speedups = [
            other["forward_ms"] / row["forward_ms"]
            for row, other in zip(results[0::3], results[BACKENDS.index(backend)::3])
        ]
        calls = [
            other["call_ms"] / row["call_ms"]
            for row, other in zip(results[0::3], results[BACKENDS.index(backend)::3])
        ]
        print(f"q8kv4 vs {backend}: forward speedup geomean "
              f"{statistics.geometric_mean(speedups):.2f}x (min {min(speedups):.2f}x, max "
              f"{max(speedups):.2f}x); call speedup geomean {statistics.geometric_mean(calls):.2f}x")
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
