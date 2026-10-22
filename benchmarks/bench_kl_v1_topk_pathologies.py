"""Search legal TopK patterns that are pathological for the MSA v1 CuTe KL.

This compares the backend introduced by Megatron commit ``cc851edf``
(``msa_v1.kl.backward``) with the previous Gungnir
``SparseAttnKLLossM3`` implementation on identical tensors and TopK indices.

The benchmark intentionally separates four timings:

* ``gungnir_bwd``: old KL backward with an already-built reverse index;
* ``gungnir_full``: old reverse-index build plus old KL backward;
* ``cute_bwd``: new KL backward with the forward-built metadata;
* ``cute_full``: new metadata preparation plus new KL backward.

All generated TopK rows contain unique, causal-valid block indices.  Use a
bottom-right CP-like Q/K shape so every query can see a large common K prefix.

Reproduce the strongest same-shape counterexample on B300:

    PYTHONPATH=$PWD/training CUDA_VISIBLE_DEVICES=7 \
      python benchmarks/bench_kl_v1_topk_pathologies.py \
      --q-len 4096 --k-len 65536 --fragments 128 \
      --patterns recent,spread,hot8_tail8 --warmup 20 --cycles 100

Here ``recent`` is the normal locality-heavy baseline.  ``spread`` and
``hot8_tail8`` keep the same tensor shapes, TopK=16, and total reverse-index
nnz, but distribute those references over many more logical K rows.

Constrain block 0 to a deterministic 99% selection rate and each query's
local block to a 100% selection rate:

    PYTHONPATH=$PWD/training CUDA_VISIBLE_DEVICES=7 \
      python benchmarks/bench_kl_v1_topk_pathologies.py \
      --q-len 4096 --k-len 65536 --fragments 128 \
      --patterns b0p99_local_recent,b0p99_local_spread,b0p99_local_hot6 \
      --warmup 20 --cycles 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Callable

import torch
import tvm_ffi


GUNGNIROPS_ROOT = os.environ.get("GUNGNIROPS_ROOT")
if GUNGNIROPS_ROOT:
    # The image's installed wheel is the default because it matches the
    # Megatron image under investigation.  This override is only for an
    # explicitly requested Gungnir checkout.
    gungnirops_python = os.path.join(GUNGNIROPS_ROOT, "python")
    if gungnirops_python not in sys.path:
        sys.path.insert(0, gungnirops_python)

from msa_v1 import attention, kl


DEVICE = "cuda:0"
BLOCK_K = 128
TOPK = 16
KV_HEADS = 4
Q_HEADS = 64
INDEX_HEADS = 4
HEAD_DIM = 128

DEFAULT_PATTERNS = (
    "recent",
    "spread",
    "pool16",
    "pool32",
    "pool64",
    "pool128",
    "pool256",
    "hot15_tail1",
    "hot8_tail8",
    "random",
    "b0p99_local_recent",
    "b0p99_local_spread",
    "b0p99_local_pool16",
    "b0p99_local_pool32",
    "b0p99_local_pool64",
    "b0p99_local_hot2",
    "b0p99_local_hot4",
    "b0p99_local_hot6",
    "b0p99_local_hot8",
    "b0p99_local_hot12",
    "b0p99_local_hot14",
)


@dataclass
class Inputs:
    q: torch.Tensor
    k: torch.Tensor
    qi: torch.Tensor
    ki: torch.Tensor
    teacher_lse: torch.Tensor
    indexer_lse_qh: torch.Tensor
    topk: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    fragment_indices: torch.Tensor | None
    q_doc_positions: torch.Tensor
    total_rows: int
    max_kvblocks: int
    max_seqlen_q: int
    max_seqlen_k: int
    sm_scale: float
    loss_coeff: float


def _coprime_step(pool: int) -> int:
    for candidate in (127, 61, 31, 17, 13, 7, 5, 3, 1):
        if math.gcd(candidate, pool) == 1:
            return candidate
    raise AssertionError("unreachable")


def _build_block0_local_topk(
    pattern: str,
    visible: torch.Tensor,
    q_idx: torch.Tensor,
    heads: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Build TopK with block 0 at ~99% and every query's local block at 100%."""
    q_len = int(visible.numel())
    common_blocks = int(visible.min().item())
    local = visible - 1
    if int(local.min().item()) <= 0:
        raise ValueError("the constrained patterns require every local block > 0")
    if common_blocks < TOPK + 1:
        raise ValueError(
            "the constrained patterns need at least 17 causal-visible blocks"
        )

    # A deterministic pseudo-random 99%-rate mask keeps runs reproducible.
    # It is shared across index heads because the constraint is per query.
    select_block0 = ((q_idx * 73 + seed) % 100) != 0

    if pattern == "b0p99_local_recent":
        candidate_slots = torch.arange(
            TOPK + 2, device=DEVICE, dtype=torch.int64
        )
        candidates = local[None, :, None] - 1 - candidate_slots[None, None, :]
        candidates = candidates.expand(KV_HEADS, -1, -1)
    else:
        hot_count = 0
        if pattern == "b0p99_local_spread":
            pool_end = common_blocks - 1
        elif pattern.startswith("b0p99_local_pool"):
            pool_end = min(
                int(pattern.removeprefix("b0p99_local_pool")),
                common_blocks - 1,
            )
        elif pattern.startswith("b0p99_local_hot"):
            hot_count = int(pattern.removeprefix("b0p99_local_hot"))
            if not 0 <= hot_count <= TOPK - 2:
                raise ValueError(f"invalid constrained hot count: {hot_count}")
            pool_end = common_blocks - 1
        else:
            raise ValueError(f"unknown constrained pattern: {pattern}")

        if pool_end < TOPK:
            raise ValueError(f"{pattern} has too few nonzero common blocks")
        hot = torch.arange(
            1, hot_count + 1, device=DEVICE, dtype=torch.int64
        )
        tail_begin = hot_count + 1
        tail_size = pool_end - hot_count
        if tail_size <= 0:
            raise ValueError(f"{pattern} has no tail candidate blocks")
        step = _coprime_step(tail_size)
        tail_slots = torch.arange(tail_size, device=DEVICE, dtype=torch.int64)
        base = (q_idx[:, None] * TOPK + heads[None, :] * 37) % tail_size
        tail = tail_begin + (
            base[:, :, None] + tail_slots[None, None, :] * step
        ) % tail_size
        tail = tail.permute(1, 0, 2)
        hot_candidates = hot[None, None, :].expand(KV_HEADS, q_len, -1)
        candidates = torch.cat((hot_candidates, tail), dim=-1)

    valid = (
        (candidates > 0)
        & (candidates < visible[None, :, None])
        & (candidates != local[None, :, None])
    )
    filler_count = TOPK - 1
    if int(valid.sum(dim=-1).min().item()) < filler_count:
        raise ValueError(f"{pattern} cannot supply {filler_count} unique fillers")
    order = torch.arange(candidates.shape[-1], device=DEVICE, dtype=torch.int64)
    priority = order[None, None, :] + (~valid) * (2 * candidates.shape[-1])
    selected = torch.topk(
        priority, k=filler_count, dim=-1, largest=False, sorted=True
    ).indices
    fillers = torch.gather(candidates, dim=-1, index=selected)

    with_block0 = torch.cat(
        (
            torch.zeros((KV_HEADS, q_len, 1), device=DEVICE, dtype=torch.int64),
            fillers[:, :, : TOPK - 2],
        ),
        dim=-1,
    )
    chosen_tail = torch.where(
        select_block0[None, :, None],
        with_block0,
        fillers,
    )
    return torch.cat(
        (local[None, :, None].expand(KV_HEADS, -1, -1), chosen_tail),
        dim=-1,
    )


def build_topk(pattern: str, visible: torch.Tensor, seed: int) -> torch.Tensor:
    """Build unique and bottom-right-causal-valid ``[4, Q, 16]`` TopK."""
    q_len = int(visible.numel())
    if int(visible.min().item()) < TOPK:
        raise ValueError(
            "every query must expose at least 16 K blocks"
        )
    q_idx = torch.arange(q_len, device=DEVICE, dtype=torch.int64)
    heads = torch.arange(KV_HEADS, device=DEVICE, dtype=torch.int64)
    slots = torch.arange(TOPK, device=DEVICE, dtype=torch.int64)
    common_blocks = int(visible.min().item())

    if pattern.startswith("b0p99_local_"):
        result = _build_block0_local_topk(
            pattern, visible, q_idx, heads, seed
        )
    elif pattern == "recent":
        values = visible[:, None] - TOPK + slots[None, :]
        result = values[None, :, :].expand(KV_HEADS, -1, -1).clone()
    elif pattern == "spread":
        pool = common_blocks
        step = _coprime_step(pool)
        base = (q_idx[:, None] * TOPK + heads[None, :] * 37) % pool
        values = (base[:, :, None] + slots[None, None, :] * step) % pool
        result = values.permute(1, 0, 2).contiguous()
    elif pattern.startswith("pool"):
        requested = int(pattern.removeprefix("pool"))
        pool = min(requested, common_blocks)
        if pool < TOPK:
            raise ValueError(f"{pattern} requires at least {TOPK} common blocks")
        step = _coprime_step(pool)
        base = (q_idx[:, None] * TOPK + heads[None, :] * 19) % pool
        values = (base[:, :, None] + slots[None, None, :] * step) % pool
        result = values.permute(1, 0, 2).contiguous()
    elif pattern == "hot15_tail1":
        if common_blocks <= TOPK:
            raise ValueError("hot15_tail1 requires more than 16 common blocks")
        hot = slots[: TOPK - 1]
        tail = TOPK - 1 + (
            q_idx[:, None] + heads[None, :] * 53
        ) % (common_blocks - (TOPK - 1))
        result = torch.empty(
            (KV_HEADS, q_len, TOPK), dtype=torch.int64, device=DEVICE
        )
        result[:, :, : TOPK - 1] = hot
        result[:, :, TOPK - 1] = tail.transpose(0, 1)
    elif pattern == "hot8_tail8":
        hot_count = TOPK // 2
        if common_blocks < TOPK:
            raise ValueError("hot8_tail8 requires at least 16 common blocks")
        hot = slots[:hot_count]
        pool = common_blocks - hot_count
        step = _coprime_step(pool)
        base = (q_idx[:, None] * hot_count + heads[None, :] * 29) % pool
        tail = hot_count + (
            base[:, :, None]
            + slots[None, None, :hot_count] * step
        ) % pool
        result = torch.empty(
            (KV_HEADS, q_len, TOPK), dtype=torch.int64, device=DEVICE
        )
        result[:, :, :hot_count] = hot
        result[:, :, hot_count:] = tail.permute(1, 0, 2)
    elif pattern == "random":
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        keys = torch.rand(
            (KV_HEADS, q_len, common_blocks),
            generator=generator,
            device=DEVICE,
        )
        result = torch.topk(
            keys, k=TOPK, dim=-1, largest=False, sorted=False
        ).indices
    else:
        raise ValueError(f"unknown pattern: {pattern}")

    result = result.to(torch.int32).contiguous()
    if tuple(result.shape) != (KV_HEADS, q_len, TOPK):
        raise AssertionError(f"bad TopK shape: {tuple(result.shape)}")
    sorted_values = result.sort(dim=-1).values
    if torch.any(sorted_values[:, :, 1:] == sorted_values[:, :, :-1]).item():
        raise AssertionError(f"{pattern} generated duplicate TopK indices")
    if torch.any(result < 0).item():
        raise AssertionError(f"{pattern} generated negative TopK indices")
    if torch.any(result >= visible[None, :, None]).item():
        raise AssertionError(f"{pattern} generated a non-causal TopK index")
    return result


@torch.inference_mode()
def make_inputs(
    pattern: str,
    q_len: int,
    k_len: int,
    fragments: int,
    seed: int,
) -> Inputs:
    torch.manual_seed(seed)
    if fragments <= 0 or q_len % fragments:
        raise ValueError("fragments must be positive and divide q_len")
    if fragments == 1:
        max_seqlen_q = q_len
        max_seqlen_k = k_len
        q_doc_positions = torch.arange(
            k_len - q_len,
            k_len,
            dtype=torch.int64,
            device=DEVICE,
        )
        visible = q_doc_positions // BLOCK_K + 1
        cu_q = torch.tensor((0, q_len), dtype=torch.int32, device=DEVICE)
        cu_k = torch.tensor((0, k_len), dtype=torch.int32, device=DEVICE)
        fragment_indices = None
        total_rows = (k_len + BLOCK_K - 1) // BLOCK_K
    else:
        # Model multiple non-contiguous Q fragments from one document sharing
        # the same physical required-K prefix.  This is the Magi layout that
        # exercises ``fragment_indices``: logical K rows grow with every
        # fragment while all physical starts redirect to ``cu_k[0] == 0``.
        q_per_fragment = q_len // fragments
        min_k = max(
            q_per_fragment + (TOPK - 1) * BLOCK_K,
            k_len // 4,
        )
        min_k = ((min_k + BLOCK_K - 1) // BLOCK_K) * BLOCK_K
        if min_k > k_len:
            raise ValueError("k_len is too small for the requested fragment layout")
        endpoints = [
            min_k + (k_len - min_k) * idx // (fragments - 1)
            for idx in range(fragments)
        ]
        endpoints = [
            min(k_len, ((value + BLOCK_K - 1) // BLOCK_K) * BLOCK_K)
            for value in endpoints
        ]
        endpoints[-1] = k_len
        cu_q = torch.arange(
            0,
            q_len + 1,
            q_per_fragment,
            dtype=torch.int32,
            device=DEVICE,
        )
        cu_k = torch.tensor(
            (0, *endpoints), dtype=torch.int32, device=DEVICE
        )
        fragment_indices = torch.zeros(
            fragments, dtype=torch.int32, device=DEVICE
        )
        positions = [
            torch.arange(
                endpoint - q_per_fragment,
                endpoint,
                dtype=torch.int64,
                device=DEVICE,
            )
            for endpoint in endpoints
        ]
        q_doc_positions = torch.cat(positions)
        visible = q_doc_positions // BLOCK_K + 1
        total_rows = sum(
            (endpoint + BLOCK_K - 1) // BLOCK_K for endpoint in endpoints
        )
        max_seqlen_q = q_per_fragment
        max_seqlen_k = k_len

    topk = build_topk(pattern, visible, seed)
    sm_scale = HEAD_DIM**-0.5
    loss_coeff = 0.7

    # Small magnitudes and conservative LSEs keep exponentials finite while
    # avoiding a dense reference attention allocation.
    def randn(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            torch.randn(shape, dtype=torch.float32, device=DEVICE) * 0.1
        ).to(torch.bfloat16)

    q = randn((q_len, Q_HEADS, HEAD_DIM))
    k = randn((k_len, KV_HEADS, HEAD_DIM))
    qi = randn((q_len, INDEX_HEADS, HEAD_DIM))
    ki = randn((k_len, 1, HEAD_DIM))
    lse_value = math.log(max(k_len, 1)) + 1.0
    teacher_lse = torch.full(
        (q_len, Q_HEADS), lse_value, dtype=torch.float32, device=DEVICE
    )
    indexer_lse_qh = torch.full(
        (q_len, INDEX_HEADS), lse_value, dtype=torch.float32, device=DEVICE
    )
    # Match the allocation rule used by the old Gungnir wrapper.
    max_kvblocks = k_len // BLOCK_K + fragments
    return Inputs(
        q=q,
        k=k,
        qi=qi,
        ki=ki,
        teacher_lse=teacher_lse,
        indexer_lse_qh=indexer_lse_qh,
        topk=topk,
        cu_q=cu_q,
        cu_k=cu_k,
        fragment_indices=fragment_indices,
        q_doc_positions=q_doc_positions,
        total_rows=total_rows,
        max_kvblocks=max_kvblocks,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        sm_scale=sm_scale,
        loss_coeff=loss_coeff,
    )


def make_gungnir_calls(
    data: Inputs,
) -> tuple[Callable[[], None], Callable[[], None], tuple[torch.Tensor, ...]]:
    try:
        from gungnir.ops.sparse_attn_kl_loss_m3 import SparseAttnKLLossM3
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Gungnir is required for the comparison; install its wheel or set "
            "GUNGNIROPS_ROOT to a checkout containing python/gungnir"
        ) from exc

    op = SparseAttnKLLossM3()
    row_ptr = torch.empty(
        KV_HEADS * (data.max_kvblocks + 1),
        dtype=torch.int32,
        device=DEVICE,
    )
    q_indices = torch.empty(
        KV_HEADS * data.q.shape[0] * TOPK,
        dtype=torch.int32,
        device=DEVICE,
    )
    dqi = torch.empty_like(data.qi, dtype=torch.float32)
    dki = torch.empty_like(data.ki, dtype=torch.float32)
    avg_loss = torch.empty(Q_HEADS, dtype=torch.float32, device=DEVICE)
    avg_qk_entropy = torch.empty_like(avg_loss)
    avg_idx_entropy = torch.empty(INDEX_HEADS, dtype=torch.float32, device=DEVICE)

    def build_reverse() -> None:
        with tvm_ffi.use_torch_stream():
            if data.fragment_indices is None:
                op.build_reverse_index(
                    data.topk,
                    data.cu_q,
                    data.cu_k,
                    BLOCK_K,
                    row_ptr,
                    q_indices,
                )
            else:
                op.build_reverse_index_fragments(
                    data.topk,
                    data.cu_q,
                    data.cu_k,
                    BLOCK_K,
                    row_ptr,
                    q_indices,
                    data.fragment_indices,
                )

    def backward_only() -> None:
        dqi.zero_()
        dki.zero_()
        avg_loss.zero_()
        avg_qk_entropy.zero_()
        avg_idx_entropy.zero_()
        with tvm_ffi.use_torch_stream():
            common_args = (
                data.q,
                data.k,
                data.qi,
                data.ki,
                row_ptr,
                q_indices,
                data.cu_q,
                data.cu_k,
                BLOCK_K,
                data.sm_scale,
                data.sm_scale,
                data.loss_coeff,
                data.teacher_lse,
                data.indexer_lse_qh,
                dqi,
                dki,
                avg_loss,
                avg_qk_entropy,
                avg_idx_entropy,
            )
            if data.fragment_indices is None:
                op.backward_with_metrics(*common_args)
            else:
                op.backward_with_metrics_fragments(
                    *common_args,
                    data.fragment_indices,
                    data.q_doc_positions,
                )

    def full() -> None:
        build_reverse()
        backward_only()

    build_reverse()
    return backward_only, full, (dqi, dki)


def prepare_cute(data: Inputs):
    return attention.prepare(
        data.topk,
        data.cu_q,
        data.cu_k,
        total_k=int(data.k.shape[0]),
        total_rows=data.total_rows,
        max_seqlen_q=data.max_seqlen_q,
        max_seqlen_k=data.max_seqlen_k,
        fragment_indices=data.fragment_indices,
        prepare_kl_schedule=True,
    )


def make_cute_calls(
    data: Inputs,
) -> tuple[Callable[[], None], Callable[[], None], object, list[torch.Tensor]]:
    metadata = prepare_cute(data)
    last_outputs: list[torch.Tensor] = []

    def run_with_metadata(meta) -> None:
        outputs = kl.backward(
            data.q,
            data.k,
            data.teacher_lse,
            data.qi,
            data.ki,
            data.indexer_lse_qh.transpose(0, 1).contiguous(),
            meta,
            softmax_scale=data.sm_scale,
            indexer_softmax_scale=data.sm_scale,
            loss_coeff=data.loss_coeff,
        )
        last_outputs[:] = outputs

    def backward_only() -> None:
        run_with_metadata(metadata)

    def full() -> None:
        run_with_metadata(prepare_cute(data))

    return backward_only, full, metadata, last_outputs


def summarize(values_ms: list[float]) -> dict[str, float]:
    ordered = sorted(values_ms)
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


@torch.inference_mode()
def benchmark_abba(
    calls: dict[str, Callable[[], None]], warmup: int, cycles: int
) -> dict[str, dict[str, float]]:
    labels = tuple(calls)
    for _ in range(warmup):
        for label in labels:
            calls[label]()
    torch.cuda.synchronize()

    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        label: [] for label in labels
    }
    for cycle in range(cycles):
        order = labels if cycle % 2 == 0 else tuple(reversed(labels))
        for label in (*order, *reversed(order)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            calls[label]()
            end.record()
            events[label].append((start, end))
    torch.cuda.synchronize()
    return {
        label: summarize([start.elapsed_time(end) for start, end in pairs])
        for label, pairs in events.items()
    }


def cosine(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs_f = lhs.float().flatten()
    rhs_f = rhs.float().flatten()
    return float(
        torch.dot(lhs_f, rhs_f)
        .div(lhs_f.norm() * rhs_f.norm())
        .item()
    )


@torch.inference_mode()
def schedule_stats(metadata) -> dict[str, float | int]:
    torch.cuda.synchronize()
    row_lengths = (
        metadata.k2q_row_ptr[:, 1:] - metadata.k2q_row_ptr[:, :-1]
    ).flatten()
    active = row_lengths[row_lengths > 0].float()
    owner_counts = metadata.kl_schedule.dki_owner_counts.float()
    result: dict[str, float | int] = {
        "nnz": int(row_lengths.sum().item()),
        "active_rows": int(active.numel()),
        "work_count": int(metadata.kl_schedule.work_count.item()),
        "dki_active_count": int(metadata.kl_schedule.dki_active_count.item()),
        "max_fanin": int(row_lengths.max().item()),
        "max_owner_count": int(owner_counts.max().item()),
    }
    if active.numel():
        result.update(
            mean_active_fanin=float(active.mean().item()),
            p50_fanin=float(torch.quantile(active, 0.50).item()),
            p95_fanin=float(torch.quantile(active, 0.95).item()),
            p99_fanin=float(torch.quantile(active, 0.99).item()),
        )
    return result


@torch.inference_mode()
def topk_constraint_stats(data: Inputs) -> dict[str, float]:
    block0_selected = (data.topk == 0).any(dim=-1)
    local_blocks = data.q_doc_positions // BLOCK_K
    local_selected = (
        data.topk == local_blocks[None, :, None]
    ).any(dim=-1)
    return {
        "block0_selected_rate": float(block0_selected.float().mean().item()),
        "local_selected_rate": float(local_selected.float().mean().item()),
    }


@torch.inference_mode()
def run_pattern(
    pattern: str,
    q_len: int,
    k_len: int,
    fragments: int,
    seed: int,
    warmup: int,
    cycles: int,
    check: bool,
) -> dict[str, object]:
    data = make_inputs(pattern, q_len, k_len, fragments, seed)
    gungnir_bwd, gungnir_full, gungnir_outputs = make_gungnir_calls(data)
    cute_bwd, cute_full, metadata, cute_outputs = make_cute_calls(data)

    gungnir_bwd()
    cute_bwd()
    torch.cuda.synchronize()
    correctness = {
        "dqi_cos": cosine(gungnir_outputs[0], cute_outputs[0]),
        "dki_cos": cosine(gungnir_outputs[1], cute_outputs[1]),
    }
    if check and min(correctness.values()) < 0.98:
        raise RuntimeError(f"output cosine check failed: {correctness}")

    timing = benchmark_abba(
        {
            "gungnir_bwd": gungnir_bwd,
            "gungnir_full": gungnir_full,
            "cute_bwd": cute_bwd,
            "cute_full": cute_full,
        },
        warmup=warmup,
        cycles=cycles,
    )
    gungnir_ms = timing["gungnir_bwd"]["median_ms"]
    gungnir_full_ms = timing["gungnir_full"]["median_ms"]
    cute_ms = timing["cute_bwd"]["median_ms"]
    cute_full_ms = timing["cute_full"]["median_ms"]
    constraint_stats = topk_constraint_stats(data)
    if pattern.startswith("b0p99_local_") and check:
        rate_tolerance = 1.0 / q_len
        if constraint_stats["local_selected_rate"] != 1.0:
            raise RuntimeError("a constrained TopK row omitted its local block")
        if abs(constraint_stats["block0_selected_rate"] - 0.99) > rate_tolerance:
            raise RuntimeError(
                "block 0 selection rate is outside the finite-Q 99% tolerance: "
                f"{constraint_stats['block0_selected_rate']}"
            )
    result: dict[str, object] = {
        "pattern": pattern,
        "q_len": q_len,
        "k_len": k_len,
        "fragments": fragments,
        **constraint_stats,
        **schedule_stats(metadata),
        **correctness,
        "timing": timing,
        "cute_vs_gungnir_bwd": cute_ms / gungnir_ms,
        "cute_bwd_delta_pct": (cute_ms / gungnir_ms - 1.0) * 100.0,
        "cute_vs_gungnir_full": cute_ms / gungnir_full_ms,
        "cute_full_vs_gungnir_full": cute_full_ms / gungnir_full_ms,
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-len", type=int, default=4096)
    parser.add_argument("--k-len", type=int, default=65536)
    parser.add_argument("--fragments", type=int, default=1)
    parser.add_argument(
        "--patterns",
        default=",".join(DEFAULT_PATTERNS),
        help="comma-separated TopK pattern names",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--no-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patterns = tuple(filter(None, args.patterns.split(",")))
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": torch.cuda.get_device_capability(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "q_len": args.q_len,
                "k_len": args.k_len,
                "fragments": args.fragments,
                "patterns": patterns,
                "warmup": args.warmup,
                "cycles": args.cycles,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    results = []
    for pattern in patterns:
        print(f"\n=== {pattern} ===", flush=True)
        results.append(
            run_pattern(
                pattern,
                args.q_len,
                args.k_len,
                args.fragments,
                args.seed,
                args.warmup,
                args.cycles,
                not args.no_check,
            )
        )
        torch.cuda.empty_cache()
    results.sort(key=lambda item: item["cute_vs_gungnir_bwd"], reverse=True)
    print("\n=== ranked CuTe/Gungnir backward ===")
    for result in results:
        print(
            f"{result['pattern']:>16s}  "
            f"ratio={result['cute_vs_gungnir_bwd']:.3f}x  "
            f"cute={result['timing']['cute_bwd']['median_ms']:.3f} ms  "
            f"gungnir={result['timing']['gungnir_bwd']['median_ms']:.3f} ms  "
            f"work={result['work_count']}  max_fanin={result['max_fanin']}"
        )


if __name__ == "__main__":
    main()
