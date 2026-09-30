# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Q8KV4 paged sparse decode benchmark on an NVFP4 KV cache.

Per case: build the inputs of tests/q8kv4 (random codes, block scales, scattered pages, TopK lists
with the local page last), compare the kernel with the fp32 reference (elements outside the test
tolerance are reported per row, non-finite output aborts), then time CUDA-graph replays. Each graph call rotates over disjoint physical page regions ("slots") so that the KV
bytes between two uses of the same page exceed twice the L2, i.e. the cache is streamed from HBM
as in serving. Plans are built outside capture; the timed interval contains only kernel launches.

Suites: `mtp` (batch 8/16/32, 100k context, 4 query tokens per request), `full` (28 cases, batch
8/32/64/128 x context 1k..200k, 8 query tokens, weighted), `smoke` (3 cases).
Backends: `q8kv4` (fmha_sm100_plan/fmha_sm100 forced to the Q8KV4 kernel), `kv_mode3`
(the same API forced to PR 13's kernel), `wrapper` (decode_q8kv4 directly on contiguous tensors).

Requires an exclusive GPU: the run refuses to start while another compute process is present.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
# Import the test helpers as the `q8kv4` package, the name pytest gives them, so a
# third-party top-level `tests` package on sys.path cannot shadow them.
sys.path.insert(0, str(REPO_ROOT / "tests"))

from fmha_sm100.api import fmha_sm100, fmha_sm100_plan  # noqa: E402
from fmha_sm100.decode_q8kv4 import BatchDecodeWithPagedKVCacheWrapper  # noqa: E402
from q8kv4.cases import (  # noqa: E402
    KV_HEADS, PAGE_SIZE, SM_SCALE, DecodeCase, flat_page_table, global_scale, make_inputs,
    pack_head_slot_pages, page_counts,
)
from q8kv4.reference import PageDequantizer, sparse_decode_reference  # noqa: E402

# (batch, context) -> weight of the full suite; 8 query tokens per request.
FULL_SUITE = {
    (8, 1_000): 30, (32, 1_000): 90, (64, 1_000): 180, (128, 1_000): 300,
    (8, 4_000): 40, (32, 4_000): 120, (64, 4_000): 240, (128, 4_000): 400,
    (8, 5_000): 50, (32, 5_000): 150, (64, 5_000): 300, (128, 5_000): 500,
    (8, 10_000): 140, (32, 10_000): 350, (64, 10_000): 490, (128, 10_000): 420,
    (8, 50_000): 500, (32, 50_000): 800, (64, 50_000): 500, (128, 50_000): 200,
    (8, 100_000): 1260, (32, 100_000): 980, (64, 100_000): 420, (128, 100_000): 140,
    (8, 200_000): 910, (32, 200_000): 350, (64, 200_000): 112, (128, 200_000): 28,
}
MTP_SUITE = {(8, 100_000): 1, (16, 100_000): 1, (32, 100_000): 1}
SMOKE_SUITE = {(8, 1_000): 1, (32, 10_000): 1, (16, 100_000): 1}
HBM_PEAK_TBS = {"B300": 8.0, "B200": 8.0}  # override with --hbm-peak-tbs for other devices
GRAPH_CALLS = 120
BYTES_PER_PAGE_SIDE = PAGE_SIZE * 72  # packed data + block scales, one head


def suite_cases(name: str, q_len: int, seed: int) -> list[tuple[DecodeCase, int]]:
    table = {"full": FULL_SUITE, "mtp": MTP_SUITE, "smoke": SMOKE_SUITE}[name]
    if name == "full":
        q_len = 8
    elif name == "mtp":
        q_len = 4
    return [
        (DecodeCase(f"b{batch}_s{context}_q{q_len}", (context,) * batch, q_len=q_len, seed=seed + i), weight)
        for i, ((batch, context), weight) in enumerate(table.items())
    ]


def ascending(topk: torch.Tensor) -> torch.Tensor:
    """kv_mode 3 needs ascending lists; the Q8KV4 kernel takes any order."""
    big = torch.iinfo(torch.int32).max
    lists = topk.clone()
    lists[lists < 0] = big
    lists, _ = lists.sort(dim=-1)
    lists[lists == big] = -1
    return lists.contiguous()


def sm_clock_mhz() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=10).stdout
        return int(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def ensure_exclusive_gpu(allow_shared: bool) -> None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return
    others = [pid for pid in out.split() if pid.isdigit() and int(pid) != torch.cuda.current_device() and int(pid) != 0]
    if len(others) > 1 and not allow_shared:
        raise SystemExit(f"another compute process is using the GPU ({others}); pass --allow-shared to override")


class Slot:
    """One physical page region with the plan and call closure of a backend."""

    def __init__(self, inputs, backend: str, shift: int, device):
        self.out = torch.empty((inputs.q.shape[0], inputs.num_q_heads, 128), dtype=torch.bfloat16, device=device)
        self.inputs = inputs
        if backend == "wrapper":
            wrapper = BatchDecodeWithPagedKVCacheWrapper()
            wrapper.plan(inputs.topk_indices, inputs.page_table, inputs.seq_lens, q_len_per_req=inputs.case.q_len,
                         num_q_heads=inputs.num_q_heads, num_kv_heads=KV_HEADS, block_scale_shift=shift)
            scales = (inputs.k_scale, inputs.v_scale_kernel)
            self.call = lambda: wrapper.run(inputs.q, (inputs.k_codes, inputs.v_codes), kv_cache_sf=scales, out=self.out)
            return
        k_packed, v_packed = pack_head_slot_pages(inputs)
        kv_indices, _ = flat_page_table(inputs.page_table, inputs.case.seq_lens)
        unit = global_scale(1.0, device)
        kv = inputs.seq_lens.cpu()
        qo = torch.full_like(kv, inputs.case.q_len)
        plan = fmha_sm100_plan(qo, kv, inputs.num_q_heads, num_kv_heads=KV_HEADS, causal=True, qo_offset=kv - qo,
                               page_size=PAGE_SIZE, kv_block_num=inputs.case.topk,
                               decode_backend=backend, block_scale_shift=shift)
        self.call = lambda: fmha_sm100(inputs.q, k_packed, v_packed, plan, kv_indices=kv_indices,
                                       kv_block_indexes=inputs.topk_indices, out=self.out, sm_scale=SM_SCALE,
                                       k_scale=unit, v_scale=unit)[0]


def bytes_read_per_call(case: DecodeCase) -> int:
    items = case.batch_size * case.q_len * KV_HEADS
    pages = min(case.topk, max(page_counts(case.seq_lens)))
    return items * pages * BYTES_PER_PAGE_SIDE * 2


def run_case(case: DecodeCase, backend: str, *, gqa: int, shift: int, device, l2_bytes: int, warmup: int,
             replays: int) -> dict:
    slots_needed = max(1, -(-2 * l2_bytes // bytes_read_per_call(case)))
    slots = next(s for s in range(1, GRAPH_CALLS + 1) if s >= slots_needed and GRAPH_CALLS % s == 0)
    inputs_per_slot = []
    for slot in range(slots):
        inputs = make_inputs(DecodeCase(case.name, case.seq_lens, case.q_len, case.topk, case.seed + 1000 * slot),
                             device, gqa=gqa)
        inputs.topk_indices = ascending(inputs.topk_indices)
        inputs_per_slot.append(Slot(inputs, backend, shift, device))
    # Correctness on one slot against the independent reference before anything is timed.
    check = inputs_per_slot[0]
    out = check.call().clone()
    torch.cuda.synchronize()
    reference = sparse_decode_reference(check.inputs, PageDequantizer(check.inputs.k_codes, check.inputs.k_scale),
                                        PageDequantizer(check.inputs.v_codes, check.inputs.v_scale))
    diff = (out.float() - reference.float()).abs()
    violations = int((diff > 0.05 + 0.05 * reference.float().abs()).sum())
    ref_max_diff = float(diff.max())
    if not bool(torch.isfinite(out).all()):
        raise RuntimeError(f"{case.name} {backend}: non-finite output")
    for slot in inputs_per_slot:
        slot.call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in range(GRAPH_CALLS):
            inputs_per_slot[call % slots].call()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / GRAPH_CALLS)
    latency_us = statistics.median(samples)
    del graph, inputs_per_slot, check, out, reference, diff
    torch.cuda.empty_cache()
    return {"case": case.name, "backend": backend, "latency_us": latency_us,
            "cv": statistics.pstdev(samples) / latency_us, "slots": slots,
            "bytes_per_call": bytes_read_per_call(case), "ref_violations": violations,
            "ref_max_diff": ref_max_diff,
            "allocated_gb_after": torch.cuda.memory_allocated() / 2**30}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=("smoke", "mtp", "full"), default="mtp")
    parser.add_argument("--backends", default="q8kv4,kv_mode3", help="comma list of q8kv4, kv_mode3, wrapper")
    parser.add_argument("--gqa", type=int, choices=(8, 16), default=16)
    parser.add_argument("--shift", type=int, default=3, help="block_scale_shift of the Q8KV4 kernel (0 or 3)")
    parser.add_argument("--q-len", type=int, default=8, help="query tokens per request for the smoke suite")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--hbm-peak-tbs", type=float, default=None)
    parser.add_argument("--allow-shared", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="write per-case rows as JSON")
    args = parser.parse_args()

    device = torch.device("cuda")
    ensure_exclusive_gpu(args.allow_shared)
    props = torch.cuda.get_device_properties(device)
    peak_tbs = args.hbm_peak_tbs or next((v for k, v in HBM_PEAK_TBS.items() if k in props.name), None)
    backends = args.backends.split(",")
    cases = suite_cases(args.suite, args.q_len, args.seed)
    clock_before = sm_clock_mhz()
    rows = []
    started = time.time()
    for case, weight in cases:
        for backend in backends:
            row = run_case(case, backend, gqa=args.gqa, shift=args.shift, device=device, l2_bytes=props.L2_cache_size,
                           warmup=args.warmup, replays=args.replays)
            row.update(weight=weight, gqa=args.gqa, shift=args.shift, **{k: v for k, v in asdict(case).items() if k != "seq_lens"})
            if peak_tbs:
                row["mbu"] = row["bytes_per_call"] / (row["latency_us"] * 1e-6) / (peak_tbs * 1e12)
            rows.append(row)
            print(f"{case.name:20s} {backend:9s} {row['latency_us']:9.2f} us  cv {row['cv']:.3f}  slots {row['slots']:3d}"
                  + (f"  MBU {row['mbu']:.2f}" if peak_tbs else "")
                  + (f"  ref violations {row['ref_violations']} (max |diff| {row['ref_max_diff']:.3f})"
                     if row["ref_violations"] else "")
                  + f"  mem {row['allocated_gb_after']:.1f} GB", flush=True)
    print(f"\n{props.name}, SM clock {clock_before} -> {sm_clock_mhz()} MHz, gqa {args.gqa}, shift {args.shift}, "
          f"{time.time() - started:.0f}s")
    print(f"{'backend':9s} {'weighted latency us':>20s}" + (f" {'weighted MBU':>13s}" if peak_tbs else ""))
    total_weight = sum(weight for _, weight in cases)
    for backend in backends:
        mine = [row for row in rows if row["backend"] == backend]
        latency = sum(row["latency_us"] * row["weight"] for row in mine) / total_weight
        line = f"{backend:9s} {latency:20.2f}"
        if peak_tbs:
            line += f" {sum(row['mbu'] * row['weight'] for row in mine) / total_weight:13.2f}"
        print(line)
    if args.output:
        args.output.write_text(json.dumps({"device": props.name, "torch": torch.__version__, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
