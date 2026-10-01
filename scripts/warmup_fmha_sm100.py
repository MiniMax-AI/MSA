#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Pre-compile FMHA SM100 kernel variants into the JIT cache, in parallel.

Usage:
    python3 scripts/warmup_fmha_sm100.py          # Compile all with max parallelism
    python3 scripts/warmup_fmha_sm100.py -j 64    # Limit to 64 parallel compilations
    python3 scripts/warmup_fmha_sm100.py --clear   # Clear cache first, then compile all
"""

import argparse
import glob
import itertools
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from fmha_sm100.jit import _FMHA_SM100_DISPATCH, _FMHA_SM100_IMPOSSIBLE

def enumerate_all_variants():

    dims = _FMHA_SM100_DISPATCH
    all_values = [values for _, values in dims]

    variants = []
    for combo in itertools.product(*all_values):
        params = {}
        for _, template_params in combo:
            params.update(template_params)
        if _FMHA_SM100_IMPOSSIBLE(params):
            continue
        indices = [values.index(val) for (_, values), val in zip(dims, combo)]
        variant_name = "_".join(str(i) for i in indices)
        variant = dict(params)
        variant["func_name"] = "fmha_sm100_" + variant_name
        variant["variant_name"] = variant_name
        variants.append(variant)

    return variants


def select_variants_for_preset(variants, preset: str):
    """Filter FMHA CUTLASS variants for image warmup presets.

    `all` preserves the historical full matrix.  `m3-infer` keeps the paged
    M3 inference paths used by dense FP8KV, sparse scoring, and sparse decode;
    long sparse prefill is handled by MM-SA AOT kernels, not CUTLASS FMHA.
    `aime` is a narrower experiment preset for the current M3 sparse AIME shape.
    """
    if preset == "all":
        return variants

    def common_m3(v):
        if v["page_size"] != 128:
            return False
        if v["sparse_mode"] not in {"Off", "OnlyScore", "Sparse"}:
            return False
        if v["sparse_mode"] == "Sparse" and v["tile_q"] != "_128":
            return False
        return True

    if preset == "m3-infer":
        return [v for v in variants if common_m3(v)]

    if preset == "aime":
        return [
            v for v in variants
            if common_m3(v) and v["pack_factor"] in {1, 4, 16}
        ]

    raise ValueError(f"unknown preset: {preset}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compile all FMHA SM100 kernel variants")
    parser.add_argument("-j", "--jobs", type=int, default=0,
                        help="Parallel compilations (0 = one variant per core)")
    parser.add_argument("--clear", action="store_true",
                        help="Clear JIT cache before compiling")
    parser.add_argument("--all", action="store_true",
                        help="Backward-compatible alias for --include-sparse-aot")
    parser.add_argument("--include-sparse-aot", action="store_true",
                        help="Also build cute AOT kernels")
    parser.add_argument("--preset", choices=["all", "m3-infer", "aime"], default="all",
                        help="FMHA CUTLASS variant preset (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List variants without compiling")
    args = parser.parse_args()
    include_sparse_aot = args.include_sparse_aot or args.all

    _cleanup_nvcc_temps()

    if args.all:
        args.clear = True
    from fmha_sm100 import jit

    all_variants = enumerate_all_variants()
    variants = select_variants_for_preset(all_variants, args.preset)
    print(
        f"Preset: {args.preset} | FMHA variants: {len(variants)}/{len(all_variants)} "
        "+ plan + sparse_topk + reduction"
    )

    if args.dry_run:
        for v in variants:
            print(f"  {v['variant_name']}: dtype={v['dtype_in']} tile={v['tile_q']}x{v['tile_kv']} "
                  f"wg={v['single_wg']} sparse={v['sparse_mode']} page={v['page_size']} "
                  f"split={v['is_split_kv']} pack={v['pack_factor']}")
        return

    if args.clear:
        namespace = jit._namespace().path
        if namespace.exists():
            shutil.rmtree(namespace)
            print(f"Cleared cache: {namespace}")

    # The JIT cache checks each library against the files it was compiled from, so only the
    # variants whose sources changed (or that were never built) compile here. A variant builds
    # its two objects at once, so -j N runs N / 2 variants.
    names = [v["variant_name"] for v in variants]
    modules = ("plan", "sparse_topk", "reduction")
    workers = max(1, args.jobs // 2) if args.jobs > 0 else None
    print(f"Compiling into {jit._namespace().path}", flush=True)
    start = time.time()
    try:
        built = jit.prebuild_fmha_variants(names, max_workers=workers)
        built += jit.prebuild_modules(modules, max_workers=workers)
    except RuntimeError as error:
        print(f"\nBuild failed after {time.time() - start:.1f}s: {error}")
        sys.exit(1)
    cached = len(names) + len(modules) - len(built)
    print(f"\nDone in {time.time() - start:.1f}s: {len(built)} compiled, {cached} cached")

    if include_sparse_aot:
        warmup_sparse_attn(clear=args.clear)

    _cleanup_nvcc_temps()


def _cleanup_nvcc_temps():
    temps = glob.glob("/tmp/tmpxft_*")
    if temps:
        total = sum(os.path.getsize(f) for f in temps if os.path.isfile(f))
        for f in temps:
            try:
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)
            except OSError:
                pass
        print(f"Cleaned {len(temps)} nvcc temp files ({total / 1e9:.1f} GB)")


def warmup_sparse_attn(clear=False):
    """AOT-compile cute CuTe DSL kernels (fwd only).

    Enumerates dtype × qhead_per_kv combinations and runs a minimal forward
    pass for each, which triggers cute.compile() + export_to_c() via aot_cache.
    """
    import torch
    import math
    import random

    cache_dir = os.path.expanduser(
        os.environ.get("MM_SPARSE_ATTN_AOT_CACHE", "~/.cache/minfer/mm_sparse_attn")
    )
    if clear and os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared sparse-attn AOT cache: {cache_dir}")

    mm_sa_dir = str(
        Path(__file__).resolve().parents[1]
        / "python/fmha_sm100/cute"
    )
    if mm_sa_dir not in sys.path:
        sys.path.insert(0, mm_sa_dir)

    from fmha_sm100.sparse_fmha_adapter import (
        sparse_fmha as fmha_sm100,
        sparse_fmha_plan as fmha_sm100_plan,
    )

    dtypes = [torch.bfloat16, torch.float8_e4m3fn]
    gqa_ratios = [1, 2, 4, 8, 16]
    ps, hd = 128, 128
    dev = torch.device("cuda")

    topks = [8, 16]
    total_variants = len(dtypes) * len(gqa_ratios) * len(topks) * 2
    print(f"\n=== Sparse-Attention AOT ({total_variants} forward+combine variants) ===")
    start = time.time()
    compiled = 0

    for dt in dtypes:
        for gqa in gqa_ratios:
            hk, hq = 1, gqa
            for kbn in topks:
                for qo_len, kv_len in [(4, 1024), (128, 8192)]:
                    qo_lens = [qo_len]
                    kv_lens = [kv_len]
                    pages = kv_lens[0] // ps

                    random.seed(42)
                    k = torch.randn(pages, hk, ps, hd, device=dev, dtype=torch.bfloat16)
                    v = torch.randn(pages, hk, ps, hd, device=dev, dtype=torch.bfloat16)
                    q = torch.randn(sum(qo_lens), hq, hd, device=dev, dtype=torch.bfloat16)
                    if dt == torch.float8_e4m3fn:
                        k = k.to(dt)
                        v = v.to(dt)
                        q = q.to(dt)

                    ki = torch.arange(pages, device=dev, dtype=torch.int32)
                    kbi = torch.full(
                        (sum(qo_lens), hk, kbn), -1, device=dev, dtype=torch.int32
                    )
                    for t in range(sum(qo_lens)):
                        blocks = sorted(random.sample(range(pages), min(kbn, pages)))
                        kbi[t, :, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)

                    qo_seg = torch.tensor(qo_lens, device=dev, dtype=torch.int32)
                    kv_seg = torch.tensor(kv_lens, device=dev, dtype=torch.int32)
                    qo_off = torch.tensor(
                        [kv_lens[0] - qo_lens[0]], device=dev, dtype=torch.int32
                    )
                    plan = fmha_sm100_plan(
                        qo_seg,
                        kv_seg,
                        hq,
                        num_kv_heads=hk,
                        qo_offset=qo_off,
                        page_size=ps,
                        kv_block_num=kbn,
                    )
                    fmha_sm100(
                        q,
                        k,
                        v,
                        plan_info=plan,
                        sm_scale=1.0 / math.sqrt(hd),
                        kv_indices=ki,
                        kv_block_indexes=kbi,
                    )
                    compiled += 1
                    print(
                        f"  [{compiled}/{total_variants}] dtype={dt}, GQA={gqa}x, "
                        f"topk={kbn}, qo={qo_len}, kv={kv_len}",
                        flush=True,
                    )

    elapsed = time.time() - start
    cached_files = (
        len(os.listdir(cache_dir)) if os.path.isdir(cache_dir) else 0
    )
    print(f"Sparse-attn AOT done in {elapsed:.1f}s ({cached_files} .o files cached)")


if __name__ == "__main__":
    main()
