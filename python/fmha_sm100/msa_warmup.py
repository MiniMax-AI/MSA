# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Build the native kernels one MSA serving configuration needs, in parallel, before serving.

Every native kernel of fmha_sm100 (csrc FMHA variants, their plan/reduction/top-k modules, the
indexer modules, the Q8KV4 decode and prefill kernels, the k2q CSR builder) is JIT-compiled on
first use, one at a time, about a minute each. A serving engine therefore either compiles during
its first requests or warms up by running dummy batches that compile whatever they happen to reach,
serially. ``warmup()`` instead derives, from the same planner and dispatch helpers
``fmha_sm100`` runs, exactly the kernels a configuration can request (split by the KV-cache and
index-cache dtypes, so an NVFP4 deployment builds no FP8 or BF16 variants) and builds them all
at once. CuTe-DSL kernels compile in seconds on first use and are not included.

    from fmha_sm100 import warmup
    warmup(kv_cache_dtype="nvfp4", index_cache_dtype="nvfp4", num_q_heads=16,
           num_kv_heads=1, topk=16, decode_query_lens=(1, 2))

or, for an image build, ``python -m fmha_sm100.msa_warmup --kv-cache-dtype nvfp4 ...``.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

_KV_DTYPES = {
    "auto": "bf16", "bf16": "bf16", "bfloat16": "bf16",
    "fp8": "fp8", "fp8_e4m3": "fp8", "float8_e4m3fn": "fp8",
    "nvfp4": "nvfp4",
}
DECODE_BACKENDS = ("auto", "q8kv4", "kv_mode3")
# Longest prefill chunk that still changes the indexer's variant: beyond 128 packed rows every
# chunk takes the 256-row tile without packing.
_PREFILL_LEN_BUCKET_LIMIT = 129


def _normalize_dtype(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    dtype = _KV_DTYPES.get(str(value).lower())
    if dtype is None:
        raise ValueError(f"{name} must be one of {sorted(_KV_DTYPES)} or None, got {value!r}")
    return dtype


@dataclass(frozen=True)
class WarmupItem:
    """One native build: a label for reports and the call that builds (and loads) it."""

    label: str
    build: Callable[[], object] = field(compare=False, repr=False)
    fmha_variant: str | None = None


@dataclass
class WarmupReport:
    items: list[str]
    fmha_variants: list[str]
    fmha_variants_built: list[str]
    seconds: float

    def __str__(self) -> str:
        return (f"{len(self.items)} native kernels ({len(self.fmha_variants)} FMHA variants, "
                f"{len(self.fmha_variants_built)} compiled) ready in {self.seconds:.0f}s")


def _torch_dtype_code(dtype: str) -> int:
    import torch

    from .jit import _dlpack_dtype_code

    return _dlpack_dtype_code(torch.bfloat16 if dtype == "bf16" else torch.float8_e4m3fn)


def _fmha_item(dtype: str, qo_tile_size: int, single_wg: bool, sparse_mode: int,
               page_size: int, split_kv: bool, pack_factor: int, kv_dtype: str | None,
               purpose: str) -> WarmupItem:
    from . import jit

    name, params = jit._variant_key_from_runtime(
        _torch_dtype_code(dtype), qo_tile_size, single_wg, sparse_mode, page_size, split_kv,
        pack_factor, kv_dtype=kv_dtype)
    return WarmupItem(f"fmha {name} ({purpose})",
                      lambda: jit._variant_manager.compile_locked(name, params), name)


def _sparse_decode_items(dtype: str, kv_format: str | None, num_q_heads: int,
                         num_kv_heads: int, page_size: int, decode_query_lens,
                         purpose: str) -> list[WarmupItem]:
    """Sparse TopK decode: every query token is its own row of ``pack_factor`` packed heads
    (``_expand_for_per_token_sparse``), so the tile follows the pack factor; the planner picks
    split-KV per batch, so both are built."""
    from .api import _compute_pack_factor

    items = []
    for q_len in sorted(set(decode_query_lens)):
        pack_factor = _compute_pack_factor(q_len, num_q_heads, num_kv_heads)
        rows = pack_factor  # per-token expansion
        for split_kv in (False, True):
            items.append(_fmha_item(dtype, 128 if rows <= 128 else 256, rows <= 64, 0,
                                    page_size, split_kv, pack_factor, kv_format, purpose))
    return items


def _index_score_items(dtype: str, num_index_heads: int, page_size: int) -> list[WarmupItem]:
    """The indexer's prefill scoring: MQA over the index-K cache with max-score output only
    and one KV split; the variant follows the longest chunk of the batch."""
    from .api import _compute_pack_factor

    items = {}
    for max_q_len in range(1, _PREFILL_LEN_BUCKET_LIMIT + 1):
        pack_factor = _compute_pack_factor(max_q_len, num_index_heads, 1)
        rows = max_q_len * pack_factor
        item = _fmha_item(dtype, 128 if rows <= 128 else 256, rows <= 64, 2, page_size, False,
                          pack_factor, None, "index score")
        items[item.fmha_variant] = item
    return list(items.values())


def plan_warmup(
    kv_cache_dtype: str,
    index_cache_dtype: str | None,
    num_q_heads: int,
    num_kv_heads: int,
    topk: int,
    decode_query_lens: Sequence[int] = (1,),
    *,
    num_index_heads: int | None = None,
    page_size: int = 128,
    decode_backend: str = "auto",
    block_scale_shift: int = 3,
    sparse_decode: bool = True,
    sparse_prefill: bool = True,
    prefill_backend: str = "auto",
    device=None,
) -> list[WarmupItem]:
    """The native kernels ``warmup`` builds for this configuration (nothing is compiled)."""
    from . import jit

    kv_dtype = _normalize_dtype(kv_cache_dtype, name="kv_cache_dtype")
    index_dtype = _normalize_dtype(index_cache_dtype, name="index_cache_dtype")
    if decode_backend not in DECODE_BACKENDS:
        raise ValueError(f"decode_backend must be one of {DECODE_BACKENDS}, got {decode_backend!r}")
    decode_query_lens = [int(length) for length in decode_query_lens]
    if not decode_query_lens or min(decode_query_lens) < 1:
        raise ValueError("decode_query_lens must be positive query lengths")
    if num_kv_heads < 1 or num_q_heads % num_kv_heads:
        raise ValueError(f"{num_q_heads} Q heads do not group over {num_kv_heads} KV heads")

    items = [WarmupItem("fmha plan", jit.get_plan_fn)]

    # Attention decode.
    q8kv4_decode = False
    if sparse_decode and kv_dtype == "nvfp4" and decode_backend != "kv_mode3":
        from . import q8kv4_decode_adapter

        blockers = {
            q_len: q8kv4_decode_adapter._plan_blocker(
                qo_lens=[q_len], kv_lens=[q_len], num_qo_heads=num_q_heads,
                num_kv_heads=num_kv_heads, page_size=page_size, kv_block_num=topk,
                causal=True, output_maxscore=False)
            for q_len in decode_query_lens
        }
        q8kv4_decode = all(blocker is None for blocker in blockers.values())
        if decode_backend == "q8kv4" and not q8kv4_decode:
            raise ValueError(f"decode_backend='q8kv4' cannot serve this configuration: {blockers}")
    if q8kv4_decode:
        from .decode_q8kv4 import jit as q8kv4_jit

        from .decode_q8kv4 import interface as q8kv4_interface

        gqa = num_q_heads // num_kv_heads
        items += [WarmupItem("q8kv4 decode host API", q8kv4_interface._get_cpp),
                  WarmupItem("q8kv4 decode plan", lambda: q8kv4_jit.get_plan_fn(device)),
                  WarmupItem("q8kv4 decode reduction",
                             lambda: q8kv4_jit.get_reduction_module(device))]
        for split_kv in (False, True):
            items.append(WarmupItem(
                f"q8kv4 decode gqa{gqa} split={split_kv} shift={block_scale_shift}",
                lambda split_kv=split_kv: q8kv4_jit.get_fmha_fwd_variant(
                    topk=topk, split_kv=split_kv, gqa_ratio=gqa,
                    device=device, block_scale_shift=block_scale_shift)))
    elif sparse_decode:
        dtype = "bf16" if kv_dtype == "bf16" else "fp8"
        kv_format = "nvfp4" if kv_dtype == "nvfp4" else None
        items += _sparse_decode_items(dtype, kv_format, num_q_heads, num_kv_heads, page_size,
                                      decode_query_lens, f"{kv_dtype} sparse decode")
        items.append(WarmupItem(f"fmha split-KV reduction ({kv_dtype})",
                                lambda: jit.get_reduction_module(nvfp4=kv_dtype == "nvfp4")))

    # Attention prefill: the CuTe-DSL kernels compile on first use; the CSR builder is native,
    # and so is the Q8KV4 prefill kernel NVFP4 caches take for E4M3 Q (prefill_backend).
    from . import q8kv4_prefill_adapter

    if prefill_backend not in q8kv4_prefill_adapter.PREFILL_BACKENDS:
        raise ValueError(f"prefill_backend must be one of "
                         f"{q8kv4_prefill_adapter.PREFILL_BACKENDS}, got {prefill_backend!r}")
    if sparse_prefill:
        def build_k2q_csr():
            from . import sparse  # noqa: F401  (puts the cute/ modules on sys.path)
            import src.sm100.build_k2q_csr  # noqa: F401  (compiles on import)

        items.append(WarmupItem("k2q CSR builder", build_k2q_csr))
        if kv_dtype == "nvfp4" and prefill_backend != "cute_dsl":
            blocker = q8kv4_prefill_adapter._plan_blocker(
                num_qo_heads=num_q_heads, num_kv_heads=num_kv_heads, page_size=page_size,
                kv_block_num=topk, causal=True, output_maxscore=False,
                device=q8kv4_prefill_adapter._cuda_device(device))
            if blocker is None:
                from .prefill_q8kv4 import jit as prefill_jit

                items.append(WarmupItem(
                    f"q8kv4 prefill shift={block_scale_shift}",
                    lambda: prefill_jit.load_extension(device, block_scale_shift)))
            elif prefill_backend == "q8kv4":
                raise ValueError(f"prefill_backend='q8kv4' cannot serve this configuration: "
                                 f"{blocker}")

    # Indexer.
    if index_dtype == "nvfp4":
        items += [WarmupItem(f"indexer module {name}", lambda name=name: jit.get_indexer_module(name))
                  for name in ("q8kv4_indexer_decode", "indexer_topk_select")]
    elif index_dtype is not None:
        if num_index_heads is None:
            raise ValueError("num_index_heads is required for a BF16 or FP8 index cache")
        items += _index_score_items(index_dtype, num_index_heads, page_size)
        items.append(WarmupItem("sparse top-k select", jit.get_sparse_topk_module))
    # Decode lengths that pack alike share a variant.
    unique = {}
    for item in items:
        unique.setdefault(item.fmha_variant or item.label, item)
    return list(unique.values())


def warmup(
    kv_cache_dtype: str,
    index_cache_dtype: str | None,
    num_q_heads: int,
    num_kv_heads: int,
    topk: int,
    decode_query_lens: Sequence[int] = (1,),
    *,
    num_index_heads: int | None = None,
    page_size: int = 128,
    decode_backend: str = "auto",
    block_scale_shift: int = 3,
    sparse_decode: bool = True,
    sparse_prefill: bool = True,
    prefill_backend: str = "auto",
    device=None,
    max_workers: int | None = None,
) -> WarmupReport:
    """Build, in parallel, every native kernel this MSA configuration can request.

    ``kv_cache_dtype`` (``"auto"``/``"bf16"``, ``"fp8"``, ``"nvfp4"``) selects the attention
    kernels: with ``sparse_decode``, sparse TopK decode for that cache (the Q8KV4 decode kernel
    for NVFP4 when ``decode_backend`` routes there, as ``fmha_sm100_plan`` does; pass False when
    the engine decodes with its own kernel) and, with ``sparse_prefill``, the k2q CSR builder of
    sparse prefill plus, for NVFP4 when ``prefill_backend`` routes there, the Q8KV4 prefill
    kernel (``"cute_dsl"`` skips it, e.g. for engines that prefill with BF16 Q).
    ``index_cache_dtype`` selects the indexer: the Q8KV4/Q8KV8 indexer modules for ``"nvfp4"``,
    the max-score prefill scoring variants over ``num_index_heads`` heads plus sparse top-k for
    ``"bf16"``/``"fp8"``, nothing for ``None``.
    Head counts are per rank; ``decode_query_lens`` are the uniform decode query lengths
    (``1 + num_speculative_tokens``); ``block_scale_shift`` is the NVFP4 staging shift the
    plans use (3 by default). Kernels already in the cache are only loaded. Safe to call from
    every rank at once: builds share the JIT's cache locks.
    """
    started = time.time()
    items = plan_warmup(
        kv_cache_dtype, index_cache_dtype, num_q_heads, num_kv_heads, topk, decode_query_lens,
        num_index_heads=num_index_heads, page_size=page_size, decode_backend=decode_backend,
        block_scale_shift=block_scale_shift, sparse_decode=sparse_decode,
        sparse_prefill=sparse_prefill, prefill_backend=prefill_backend, device=device)
    from . import jit

    variants = [item.fmha_variant for item in items if item.fmha_variant]
    missing = {name for name in variants if not jit._variant_manager.is_cached(name)}
    jit._variant_manager._load_templates()
    to_run = [item for item in items if item.fmha_variant is None or item.fmha_variant in missing]
    workers = max_workers or min(max(len(to_run), 1), os.cpu_count() or 8)
    logger.info("MSA warmup: %d native kernels (%d FMHA variants to compile), %d workers",
                len(items), len(missing), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Each build waits on its own nvcc/ninja subprocess, so threads run them in parallel.
        for future in [pool.submit(item.build) for item in to_run]:
            future.result()
    report = WarmupReport(
        items=[item.label for item in items],
        fmha_variants=variants,
        fmha_variants_built=sorted(missing),
        seconds=time.time() - started,
    )
    logger.info("MSA warmup: %s", report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build the native MSA kernels of one serving "
                                                 "configuration in parallel.")
    parser.add_argument("--kv-cache-dtype", required=True)
    parser.add_argument("--index-cache-dtype", default=None)
    parser.add_argument("--num-q-heads", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--num-index-heads", type=int, default=None)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--decode-query-lens", default="1",
                        help="comma-separated uniform decode query lengths")
    parser.add_argument("--decode-backend", default="auto", choices=DECODE_BACKENDS)
    parser.add_argument("--block-scale-shift", type=int, default=3)
    parser.add_argument("--no-sparse-decode", action="store_true")
    parser.add_argument("--no-sparse-prefill", action="store_true")
    parser.add_argument("--prefill-backend", default="auto", choices=("auto", "q8kv4", "cute_dsl"))
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="list the kernels, build nothing")
    args = parser.parse_args()
    kwargs = dict(
        kv_cache_dtype=args.kv_cache_dtype, index_cache_dtype=args.index_cache_dtype,
        num_q_heads=args.num_q_heads, num_kv_heads=args.num_kv_heads, topk=args.topk,
        decode_query_lens=[int(v) for v in args.decode_query_lens.split(",")],
        num_index_heads=args.num_index_heads, decode_backend=args.decode_backend,
        block_scale_shift=args.block_scale_shift, sparse_decode=not args.no_sparse_decode,
        sparse_prefill=not args.no_sparse_prefill, prefill_backend=args.prefill_backend)
    if args.dry_run:
        for item in plan_warmup(**kwargs):
            print(item.label)
        return
    logging.basicConfig(level=logging.INFO)
    print(warmup(**kwargs, max_workers=args.jobs))


if __name__ == "__main__":
    _main()


__all__ = ["DECODE_BACKENDS", "WarmupItem", "WarmupReport", "plan_warmup", "warmup"]
