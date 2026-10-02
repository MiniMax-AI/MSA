"""CUDA Graph benchmark for Q8KV8/Q8KV4 paged sparse decode attention.

The timed region contains only ``wrapper.run()``.
Compilation, allocation, planning, CUDA Graph capture, and correctness checks
stay outside the measured interval. Every graph slot uses a disjoint physical
page region so the KV reuse distance exceeds twice the target GPU L2 capacity.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import inspect
import json
import logging
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from benchmarks.inference.msa_v1.acceptance import compare_e2e_results
from benchmarks.inference.msa_v1.attention.decode.q8kv4.cases import (
    BATCH_SIZES,
    FULL_CASES,
    Q_LENGTHS,
    DecodeAttentionCase,
    benchmark_cases,
    make_seq_lens,
)
from benchmarks.inference.msa_v1.hardware import roofline_metrics

PAGE_SIZE = 128
HEAD_DIM = 128
NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
TOPK = 16
KV_FORMAT = "q8kv4"
ENABLE_PDL = True
MAX_CV = 0.03
MAX_TIMING_ATTEMPTS = 3
KV_HEAD_PAGE_BYTES = 2 * (PAGE_SIZE * (HEAD_DIM // 2) + PAGE_SIZE * (HEAD_DIM // 16))


def _wrapper_type():
    module = importlib.import_module(f"inference.msa_v1.attention.decode.{KV_FORMAT}")
    return module.BatchDecodeWithPagedKVCacheWrapper


def _make_wrapper():
    kwargs = {} if KV_FORMAT == "q8kv4" else {"enable_pdl": ENABLE_PDL}
    return _wrapper_type()(**kwargs)


def _ensure_exclusive() -> None:
    import pynvml

    pynvml.nvmlInit()
    uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
    others = [
        p.pid
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if p.pid != os.getpid()
    ]
    if others:
        raise RuntimeError(
            f"formal benchmark requires an exclusive GPU; other processes: {others}"
        )


def _plan(wrapper, topk, page_table, seq_lens, q_len, num_kv_splits):
    kwargs = {
        "q_len_per_req": q_len,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
    }
    if KV_FORMAT == "q8kv4":
        kwargs["num_kv_splits"] = num_kv_splits
    wrapper.plan(topk, page_table, seq_lens, **kwargs)


def _make_query(shape, *, generator, device):
    if KV_FORMAT == "bf16":
        return torch.randn(shape, generator=generator, device=device).to(torch.bfloat16)
    return _make_finite_e4m3(shape, generator=generator, device=device)


def _cache_tensors(packed_shape, scale_shape, generator, device):
    if KV_FORMAT in ("bf16", "q8kv8"):
        shape = (*packed_shape[:-1], HEAD_DIM)
        return (
            _make_query(shape, generator=generator, device=device),
            _make_query(shape, generator=generator, device=device),
            None,
            None,
        )
    k = torch.randint(
        0, 256, packed_shape, dtype=torch.uint8, generator=generator, device=device
    )
    v = torch.randint(
        0, 256, packed_shape, dtype=torch.uint8, generator=generator, device=device
    )
    scales = [
        torch.randint(
            0x18,
            0x39,
            scale_shape,
            dtype=torch.uint8,
            generator=generator,
            device=device,
        ).view(torch.float8_e4m3fn)
        for _ in range(2)
    ]
    return k, v, *scales


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _make_finite_e4m3(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Generate moderate finite E4M3 values without a wider temporary."""

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


def _topk_seed(
    case: DecodeAttentionCase,
    slot: int,
    batch_idx: int,
    q_idx: int,
    kv_head_idx: int,
) -> int:
    return (
        case.seed * 1_000_003
        + case.q_len_per_req * 100_003
        + slot * 10_007
        + batch_idx * 503
        + q_idx * 31
        + kv_head_idx
    )


def _make_topk_slot(
    case: DecodeAttentionCase,
    seq_lens_cpu: torch.Tensor,
    slot: int,
    device: torch.device,
) -> torch.Tensor:
    """Build unordered history pages with the current causal page last."""

    topk = torch.full(
        (case.batch * case.q_len_per_req, NUM_KV_HEADS, TOPK),
        -1,
        dtype=torch.int32,
    )
    for batch_idx, seq_len_tensor in enumerate(seq_lens_cpu):
        seq_len = int(seq_len_tensor)
        for q_idx in range(case.q_len_per_req):
            query_position = seq_len - case.q_len_per_req + q_idx
            local_page = query_position // PAGE_SIZE
            valid_count = min(TOPK, local_page + 1)
            history_count = valid_count - 1
            row = batch_idx * case.q_len_per_req + q_idx
            for kv_head_idx in range(NUM_KV_HEADS):
                rng = random.Random(
                    _topk_seed(case, slot, batch_idx, q_idx, kv_head_idx)
                )
                history = rng.sample(range(local_page), history_count)
                topk[row, kv_head_idx, :history_count] = torch.tensor(
                    history, dtype=torch.int32
                )
                topk[row, kv_head_idx, history_count] = local_page
    return topk.to(device=device, non_blocking=False)


def _assert_topk_contract(
    case: DecodeAttentionCase,
    seq_lens: torch.Tensor,
    topk: torch.Tensor,
) -> None:
    query_ids = torch.arange(
        case.q_len_per_req, dtype=torch.int64, device=seq_lens.device
    ).repeat(case.batch)
    positions = seq_lens.to(torch.int64).repeat_interleave(case.q_len_per_req)
    positions = positions - case.q_len_per_req + query_ids
    local_page = torch.div(positions, PAGE_SIZE, rounding_mode="floor")
    valid_count = (local_page + 1).clamp(max=TOPK)
    slots = torch.arange(TOPK, device=seq_lens.device).reshape(1, 1, TOPK)
    expected_valid = slots < valid_count.reshape(-1, 1, 1)
    if not torch.equal(topk >= 0, expected_valid.expand_as(topk)):
        raise RuntimeError("TopK valid-prefix contract failed")
    tail = topk.gather(
        2,
        (valid_count - 1).reshape(-1, 1, 1).expand(-1, NUM_KV_HEADS, -1),
    ).squeeze(-1)
    expected_tail = local_page.to(torch.int32).reshape(-1, 1)
    if not torch.equal(tail, expected_tail.expand_as(tail)):
        raise RuntimeError("TopK local-page tail contract failed")


@dataclass
class CaseStorage:
    case: DecodeAttentionCase
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor | None
    v_scale: torch.Tensor | None
    page_tables: list[torch.Tensor]
    seq_lens: torch.Tensor
    topk_slots: list[torch.Tensor]
    out: torch.Tensor
    logical_total_pages: int
    page_stride: int

    @classmethod
    def create(
        cls,
        case: DecodeAttentionCase,
        *,
        slots: int,
        device: torch.device,
    ) -> CaseStorage:
        seq_lens_cpu = make_seq_lens(case)
        page_counts = torch.div(
            seq_lens_cpu + PAGE_SIZE - 1,
            PAGE_SIZE,
            rounding_mode="floor",
        )
        logical_total_pages = int(page_counts.sum())
        page_stride = _ceil_div(int(page_counts.max()), 4) * 4
        generator = torch.Generator(device=device).manual_seed(
            case.seed * 1009 + case.q_len_per_req * 17
        )

        page_tables = []
        for slot in range(slots):
            permutation = torch.randperm(
                logical_total_pages,
                dtype=torch.int64,
                generator=generator,
                device=device,
            ).add_(slot * logical_total_pages)
            page_table = torch.zeros(
                (case.batch, page_stride), dtype=torch.int32, device=device
            )
            offset = 0
            for batch_idx, page_count_tensor in enumerate(page_counts):
                page_count = int(page_count_tensor)
                page_table[batch_idx, :page_count] = permutation[
                    offset : offset + page_count
                ].to(torch.int32)
                offset += page_count
            page_tables.append(page_table)

        total_q = case.batch * case.q_len_per_req
        q = _make_query(
            (total_q, NUM_KV_HEADS * 16, HEAD_DIM),
            generator=generator,
            device=device,
        )
        # Preserve the RNG stream and KV payload when comparing GQA 8 against GQA 16.
        if NUM_Q_HEADS // NUM_KV_HEADS == 8:
            q = q.reshape(total_q, NUM_KV_HEADS, 16, HEAD_DIM)[:, :, :8].contiguous()
            q = q.reshape(total_q, NUM_Q_HEADS, HEAD_DIM)
        packed_shape = (
            logical_total_pages * slots,
            NUM_KV_HEADS,
            PAGE_SIZE,
            HEAD_DIM // 2,
        )
        scale_shape = (
            logical_total_pages * slots,
            NUM_KV_HEADS,
            PAGE_SIZE,
            HEAD_DIM // 16,
        )
        k_cache, v_cache, k_scale, v_scale = _cache_tensors(
            packed_shape, scale_shape, generator, device
        )
        seq_lens = seq_lens_cpu.to(device=device)
        topk_slots = [
            _make_topk_slot(case, seq_lens_cpu, slot, device) for slot in range(slots)
        ]
        for topk in topk_slots:
            _assert_topk_contract(case, seq_lens, topk)
        return cls(
            case=case,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=k_scale,
            v_scale=v_scale,
            page_tables=page_tables,
            seq_lens=seq_lens,
            topk_slots=topk_slots,
            out=torch.empty(
                (total_q, NUM_Q_HEADS, HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            ),
            logical_total_pages=logical_total_pages,
            page_stride=page_stride,
        )


def _reuse_distance_bytes(storage: CaseStorage) -> int:
    if len(storage.topk_slots) <= 1:
        return 0
    case = storage.case
    batch_ids = torch.arange(
        case.batch, dtype=torch.int64, device=storage.page_tables[0].device
    ).repeat_interleave(case.q_len_per_req)
    kv_heads = torch.arange(
        NUM_KV_HEADS, dtype=torch.int64, device=storage.page_tables[0].device
    ).reshape(1, NUM_KV_HEADS, 1)
    key_fragments = []
    for slot, topk in enumerate(storage.topk_slots[1:], start=1):
        valid = topk >= 0
        logical_pages = topk.clamp(min=0).to(torch.int64)
        physical_pages = storage.page_tables[slot][
            batch_ids.reshape(-1, 1, 1), logical_pages
        ].to(torch.int64)
        keys = physical_pages * NUM_KV_HEADS + kv_heads
        key_fragments.append(keys[valid])
    unique_pages = torch.unique(torch.cat(key_fragments)).numel()
    return int(unique_pages) * KV_HEAD_PAGE_BYTES


def _run(
    wrapper: object,
    storage: CaseStorage,
) -> torch.Tensor:
    kwargs = (
        {}
        if KV_FORMAT != "q8kv4"
        else {"kv_cache_sf": (storage.k_scale, storage.v_scale)}
    )
    return wrapper.run(
        storage.q, (storage.k_cache, storage.v_cache), out=storage.out, **kwargs
    )


def _make_wrappers(
    storage: CaseStorage,
    *,
    num_kv_splits: int | None,
) -> tuple[list[object], float, float]:
    wrappers = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for topk, page_table in zip(storage.topk_slots, storage.page_tables, strict=True):
        wrapper = _make_wrapper()
        _plan(
            wrapper,
            topk,
            page_table,
            storage.seq_lens,
            storage.case.q_len_per_req,
            num_kv_splits,
        )
        wrappers.append(wrapper)
    end.record()
    end.synchronize()
    return (
        wrappers,
        start.elapsed_time(end) * 1000.0,
        (time.perf_counter() - wall_start) * 1.0e3,
    )


def _warm_compilation(device: torch.device) -> float:
    """Compile and execute both unsplit and split variants on a tiny workload."""

    started = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(7301)
    q = _make_query((1, NUM_Q_HEADS, HEAD_DIM), generator=generator, device=device)
    packed_shape = (TOPK, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM // 2)
    scale_shape = (TOPK, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM // 16)
    k_cache, v_cache, k_scale, v_scale = _cache_tensors(
        packed_shape, scale_shape, generator, device
    )
    page_table = torch.arange(TOPK, dtype=torch.int32, device=device).reshape(1, -1)
    seq_lens = torch.tensor([TOPK * PAGE_SIZE], dtype=torch.int32, device=device)
    topk = torch.arange(TOPK, dtype=torch.int32, device=device)
    topk = topk.reshape(1, 1, TOPK).expand(1, NUM_KV_HEADS, TOPK).contiguous()
    out = torch.empty((1, NUM_Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device)
    split_options = (1, 2) if KV_FORMAT == "q8kv4" else (None,)
    for num_kv_splits in split_options:
        wrapper = _make_wrapper()
        _plan(wrapper, topk, page_table, seq_lens, 1, num_kv_splits)
        kwargs = {} if KV_FORMAT != "q8kv4" else {"kv_cache_sf": (k_scale, v_scale)}
        wrapper.run(q, (k_cache, v_cache), out=out, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(
        f"Compiled and warmed {KV_FORMAT} decode variants in {elapsed:.1f}s", flush=True
    )
    return elapsed


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    graph_calls: int,
    warmup: int,
    replays: int,
) -> tuple[list[float], float]:
    last_cv = float("inf")
    for _ in range(MAX_TIMING_ATTEMPTS):
        _ensure_exclusive()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(replays):
            _ensure_exclusive()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0 / graph_calls)
        sample_mean = statistics.fmean(samples)
        last_cv = statistics.pstdev(samples) / sample_mean
        if last_cv <= MAX_CV:
            return samples, last_cv
    raise RuntimeError(
        f"unstable timing after {MAX_TIMING_ATTEMPTS} attempts: "
        f"CV {last_cv:.4f} > {MAX_CV:.4f}"
    )


def run_case(
    case: DecodeAttentionCase,
    *,
    slots: int,
    graph_calls: int,
    warmup: int,
    replays: int,
    num_kv_splits: int | None,
    enforce_reuse_distance: bool,
) -> dict[str, object]:
    device = torch.device("cuda")
    _ensure_exclusive()
    allocation_start = time.perf_counter()
    storage = CaseStorage.create(case, slots=slots, device=device)
    allocation_seconds = time.perf_counter() - allocation_start
    properties = torch.cuda.get_device_properties(device)
    reuse_distance = _reuse_distance_bytes(storage)
    if enforce_reuse_distance and reuse_distance <= 2 * properties.L2_cache_size:
        raise RuntimeError(
            "disjoint page rotation does not exceed twice L2: "
            f"reuse={reuse_distance}, L2={properties.L2_cache_size}"
        )

    wrappers, plan_gpu_us_total, plan_wall_ms_total = _make_wrappers(
        storage, num_kv_splits=num_kv_splits
    )
    for wrapper in wrappers:
        _run(wrapper, storage)
    torch.cuda.synchronize()
    if not bool(torch.isfinite(storage.out).all()):
        raise RuntimeError("eager output contains non-finite values")

    final_slot = (graph_calls - 1) % slots
    _run(wrappers[final_slot], storage)
    torch.cuda.synchronize()
    expected_output = storage.out.clone()

    from types import SimpleNamespace

    reference_module = importlib.import_module(
        "tests.inference.msa_v1.attention.decode."
        + ("q8kv8" if KV_FORMAT == "bf16" else KV_FORMAT)
        + ".reference"
    )
    inputs = SimpleNamespace(
        q=storage.q,
        k_cache=storage.k_cache,
        v_cache=storage.v_cache,
        packed_k=storage.k_cache,
        packed_v=storage.v_cache,
        k_scale=storage.k_scale,
        v_scale=storage.v_scale,
        page_table=storage.page_tables[final_slot],
        topk_indices=storage.topk_slots[final_slot],
        seq_lens=storage.seq_lens,
        q_len_per_req=case.q_len_per_req,
    )
    reference = reference_module.decode_attention_reference(inputs)
    torch.testing.assert_close(
        expected_output.float(),
        reference.float(),
        atol=0.03 if KV_FORMAT == "bf16" else 0.05,
        rtol=0.03 if KV_FORMAT == "bf16" else 0.05,
    )
    if KV_FORMAT == "q8kv8":
        cosine = torch.nn.functional.cosine_similarity(
            expected_output.float().flatten(), reference.float().flatten(), dim=0
        )
        if float(cosine) < 0.999:
            raise AssertionError(f"cosine {float(cosine)} < 0.999")
    del reference, inputs

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call_idx in range(graph_calls):
            _run(wrappers[call_idx % slots], storage)
    graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(storage.out, expected_output):
        raise RuntimeError("CUDA Graph replay changed the full output tensor")

    latency_samples, cv = _time_graph(
        graph,
        graph_calls=graph_calls,
        warmup=warmup,
        replays=replays,
    )
    sample_mean = statistics.fmean(latency_samples)
    sample_std = statistics.pstdev(latency_samples)
    latency_us = statistics.median(latency_samples)
    total_q = case.batch * case.q_len_per_req
    seq_lens_cpu = make_seq_lens(case)
    query_positions = (
        seq_lens_cpu.reshape(-1, 1)
        - case.q_len_per_req
        + torch.arange(case.q_len_per_req, dtype=torch.int32).reshape(1, -1)
    )
    local_pages = torch.div(query_positions, PAGE_SIZE, rounding_mode="floor")
    valid_pages = (local_pages + 1).clamp(max=TOPK)
    selected_tokens = (
        (valid_pages - 1) * PAGE_SIZE + query_positions.remainder(PAGE_SIZE) + 1
    )
    useful_flops = 4 * NUM_Q_HEADS * HEAD_DIM * int(selected_tokens.sum())
    logical_kv_bytes = int(valid_pages.sum()) * NUM_KV_HEADS * KV_HEAD_PAGE_BYTES
    logical_q_bytes = total_q * NUM_Q_HEADS * HEAD_DIM * storage.q.element_size()
    logical_o_bytes = total_q * NUM_Q_HEADS * HEAD_DIM * 2
    logical_bytes = logical_kv_bytes + logical_q_bytes + logical_o_bytes
    result = {
        **asdict(case),
        "case": case.name,
        "device": properties.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "page_size": PAGE_SIZE,
        "topk": TOPK,
        "page_mode": "disjoint_scattered_slot_rotation",
        "kv_length_min": int(storage.seq_lens.min()),
        "kv_length_max": int(storage.seq_lens.max()),
        "kv_length_mean": float(storage.seq_lens.to(torch.float64).mean()),
        "logical_pages_per_slot": storage.logical_total_pages,
        "physical_pages": storage.logical_total_pages * slots,
        "page_stride": storage.page_stride,
        "slots": slots,
        "graph_calls": graph_calls,
        "warmup_replays": warmup,
        "timed_replays": replays,
        "latency_us_samples": latency_samples,
        "latency_us": latency_us,
        "latency_us_mean": sample_mean,
        "latency_us_std": sample_std,
        "cv": cv,
        "useful_tflops": useful_flops / latency_us / 1.0e6,
        "logical_bytes": logical_bytes,
        "logical_tb_s": logical_bytes / latency_us / 1.0e6,
        "l2_bytes": properties.L2_cache_size,
        "reuse_distance_bytes": reuse_distance,
        "reuse_distance_over_l2": reuse_distance / properties.L2_cache_size,
        "allocation_seconds": allocation_seconds,
        "plan_gpu_us_total": plan_gpu_us_total,
        "plan_gpu_us_per_slot": plan_gpu_us_total / slots,
        "plan_wall_ms_total": plan_wall_ms_total,
        "plan_wall_ms_per_slot": plan_wall_ms_total / slots,
        "num_kv_splits_requested": ("auto" if num_kv_splits is None else num_kv_splits),
        "correctness": "passed (full independent reference + finite output + full graph replay)",
        **roofline_metrics(
            device_name=properties.name,
            useful_flops=useful_flops,
            logical_bytes=logical_bytes,
            latency_us=latency_us,
            compute_dtype="bf16" if KV_FORMAT == "bf16" else "fp8",
        ),
    }
    del graph, wrappers, storage
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _parse_extra_case(spec: str) -> DecodeAttentionCase:
    """Ad-hoc case 'bB_sS_qQ' (diagnostic; weight 1, not part of any suite)."""
    import re

    match = re.fullmatch(r"b(\d+)_s(\d+)_q(\d+)", spec)
    if match is None:
        raise argparse.ArgumentTypeError(f"--extra-case expects bB_sS_qQ, got {spec!r}")
    return DecodeAttentionCase(
        batch_size=int(match.group(1)),
        nominal_seq_len=int(match.group(2)),
        weight=1,
        q_len_per_req=int(match.group(3)),
    )


def _selected_cases(args: argparse.Namespace) -> tuple[DecodeAttentionCase, ...]:
    if getattr(args, "extra_case", None):
        return tuple(_parse_extra_case(spec) for spec in args.extra_case)
    cases = benchmark_cases(args.suite)
    if args.batch:
        requested = set(args.batch)
        cases = tuple(case for case in cases if case.batch in requested)
    if args.q_len:
        requested = set(args.q_len)
        cases = tuple(case for case in cases if case.q_len_per_req in requested)
    if args.case_id:
        requested = set(args.case_id)
        cases = tuple(case for case in cases if case.name in requested)
    if args.limit is not None:
        cases = cases[: args.limit]
    return cases


def _resolve_slots(
    case: DecodeAttentionCase,
    *,
    requested_slots: int,
    graph_calls: int,
    l2_bytes: int,
) -> int:
    """Choose a graph divisor whose disjoint KV regions exceed twice L2."""

    seq_lens = make_seq_lens(case)
    page_counts = torch.div(
        seq_lens + PAGE_SIZE - 1,
        PAGE_SIZE,
        rounding_mode="floor",
    )
    page_stride = _ceil_div(int(page_counts.max()), 4) * 4
    batch_ids = torch.arange(case.batch, dtype=torch.int64).repeat_interleave(
        case.q_len_per_req
    )
    kv_heads = torch.arange(NUM_KV_HEADS, dtype=torch.int64).reshape(1, NUM_KV_HEADS, 1)
    reuse_distance = 0
    minimum_slots = None
    for slot in range(1, graph_calls):
        topk = _make_topk_slot(case, seq_lens, slot, torch.device("cpu"))
        valid = topk >= 0
        logical_pages = topk.clamp_min(0).to(torch.int64)
        keys = (
            batch_ids.reshape(-1, 1, 1) * page_stride + logical_pages
        ) * NUM_KV_HEADS + kv_heads
        reuse_distance += int(torch.unique(keys[valid]).numel()) * KV_HEAD_PAGE_BYTES
        if reuse_distance > 2 * l2_bytes:
            minimum_slots = slot + 1
            break
    if minimum_slots is None:
        raise ValueError(
            f"--graph-calls={graph_calls} cannot establish a >2x-L2 reuse distance"
        )
    if requested_slots:
        if requested_slots < minimum_slots:
            raise ValueError(
                f"--slots={requested_slots} is too small; case {case.name} requires "
                f"at least {minimum_slots} disjoint slots"
            )
        if graph_calls % requested_slots:
            raise ValueError("--graph-calls must be a multiple of --slots")
        return requested_slots
    for slots in range(minimum_slots, graph_calls + 1):
        if graph_calls % slots == 0:
            return slots
    raise ValueError(
        f"--graph-calls={graph_calls} has no divisor >= required slots {minimum_slots}"
    )


def main(kv_format: str = "q8kv4") -> None:
    global KV_FORMAT, NUM_Q_HEADS, NUM_KV_HEADS, KV_HEAD_PAGE_BYTES, ENABLE_PDL
    KV_FORMAT = kv_format
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-q-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--disable-pdl", action="store_true")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--batch", type=int, choices=BATCH_SIZES, action="append")
    parser.add_argument("--q-len", type=int, choices=Q_LENGTHS, action="append")
    parser.add_argument("--case-id", action="append")
    parser.add_argument(
        "--extra-case",
        action="append",
        help="ad-hoc case bB_sS_qQ (diagnostic); when given, only these cases run",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--slots",
        type=int,
        default=0,
        help="disjoint page slots; zero selects the smallest graph-call divisor",
    )
    parser.add_argument("--graph-calls", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument(
        "--num-kv-splits", choices=("auto", "1", "2", "4", "8"), default="auto"
    )
    parser.add_argument("--skip-reuse-check", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--maximum-case-regression", type=float, default=0.05)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if KV_FORMAT == "q8kv4" and args.disable_pdl:
        parser.error("--disable-pdl applies only to BF16/Q8KV8")
    ENABLE_PDL = not args.disable_pdl
    if not 0.0 <= args.maximum_case_regression <= 0.05:
        parser.error("--maximum-case-regression must be in [0, 0.05]")
    NUM_Q_HEADS, NUM_KV_HEADS = args.num_q_heads, args.num_kv_heads
    if (
        NUM_KV_HEADS <= 0
        or NUM_Q_HEADS <= 0
        or NUM_Q_HEADS % NUM_KV_HEADS
        or NUM_Q_HEADS // NUM_KV_HEADS not in (8, 16)
    ):
        parser.error("head ratio must be 8 or 16")
    if KV_FORMAT != "q8kv4" and args.num_kv_splits != "auto":
        parser.error("--num-kv-splits applies only to Q8KV4")
    KV_HEAD_PAGE_BYTES = (
        2
        * PAGE_SIZE
        * (
            {
                "bf16": 2 * HEAD_DIM,
                "q8kv8": HEAD_DIM,
                "q8kv4": HEAD_DIM // 2 + HEAD_DIM // 16,
            }[KV_FORMAT]
        )
    )
    import cutlass
    from packaging.version import Version

    if Version(cutlass.__version__) < Version("4.5.2"):
        raise RuntimeError("benchmark requires CuTe DSL >=4.5.2")
    if args.slots == 1 or args.slots < 0:
        parser.error("--slots must be zero (auto) or at least two")
    if args.slots > args.graph_calls:
        parser.error("--slots cannot exceed --graph-calls")
    if args.warmup < 1 or args.replays < 2:
        parser.error("--warmup must be positive and --replays at least two")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3), (10, 7)}:
        raise RuntimeError("SM100, SM103, or SM107 GPU required")
    cases = _selected_cases(args)
    if not cases:
        parser.error("benchmark selection is empty")

    compile_seconds = _warm_compilation(torch.device("cuda"))
    num_kv_splits = None if args.num_kv_splits == "auto" else int(args.num_kv_splits)
    properties = torch.cuda.get_device_properties(torch.device("cuda"))
    rows = []
    for case in cases:
        try:
            slots = _resolve_slots(
                case,
                requested_slots=args.slots,
                graph_calls=args.graph_calls,
                l2_bytes=properties.L2_cache_size,
            )
        except ValueError as error:
            parser.error(str(error))
        row = run_case(
            case,
            slots=slots,
            graph_calls=args.graph_calls,
            warmup=args.warmup,
            replays=args.replays,
            num_kv_splits=num_kv_splits,
            enforce_reuse_distance=not args.skip_reuse_check,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    result = {
        "schema_version": 1,
        "environment": {
            "cute_dsl": cutlass.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "kv_format": KV_FORMAT,
            "enable_pdl": ENABLE_PDL if KV_FORMAT != "q8kv4" else None,
            "implementation": inspect.getfile(_wrapper_type()),
            "flashinfer": importlib.metadata.version("flashinfer-python")
            if KV_FORMAT != "q8kv4"
            else None,
        },
        "protocol": {
            "suite": args.suite,
            "formal_full_selection": (
                args.suite == "full"
                and not args.batch
                and not args.q_len
                and not args.case_id
                and args.limit is None
                and len(rows) == len(FULL_CASES)
            ),
            "e2e_scope": "wrapper.run only",
            "compile_plan_allocate_capture_in_timing": False,
            "cuda_graph": True,
            "topk_policy": (
                "deterministic unordered history with the current causal page last"
            ),
            "page_policy": (
                "per-request scattered pages with disjoint physical regions per graph slot"
            ),
            "cold_cache_minimum_reuse_distance_over_l2": 2.0,
            "maximum_cv": MAX_CV,
            "maximum_timing_attempts": MAX_TIMING_ATTEMPTS,
        },
        "compile_warmup_seconds": compile_seconds,
        "results": rows,
    }
    if result["protocol"]["formal_full_selection"]:
        total_weight = sum(int(row["weight"]) for row in rows)
        weighted_latency = (
            sum(int(row["weight"]) * float(row["latency_us"]) for row in rows)
            / total_weight
        )
        result["aggregate"] = {
            "total_weight": total_weight,
            "weighted_mean_e2e_latency_us": weighted_latency,
            "case_contributions_us": {
                str(row["case"]): int(row["weight"])
                * float(row["latency_us"])
                / total_weight
                for row in rows
            },
        }
    acceptance_failed = False
    if args.baseline is not None:
        baseline_payload = json.loads(args.baseline.read_text())
        result["acceptance"] = compare_e2e_results(
            rows,
            baseline_payload,
            case_key="case",
            latency_key="latency_us",
            weight_key="weight",
            maximum_case_regression=args.maximum_case_regression,
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
