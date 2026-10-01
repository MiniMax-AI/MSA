# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""fmha_sm100.warmup builds exactly the native kernels a serving configuration requests.

The plan is split by the KV-cache and index-cache dtypes. After ``warmup()``, sparse decode over
many batch sizes and query lengths and the indexer's max-score prefill scoring over many chunk
lengths must run without compiling anything: every nvcc/ninja subprocess fails the test.
"""

import subprocess

import pytest
import torch

from fmha_sm100 import jit, plan_warmup, warmup
from fmha_sm100.api import _fmha_sm100, _fmha_sm100_plan, fmha_sm100, fmha_sm100_plan

PAGE = 128
TOPK = 16
CONFIGS = {
    "nvfp4_kv4_spec": dict(kv_cache_dtype="nvfp4", index_cache_dtype="nvfp4", num_q_heads=64,
                           num_kv_heads=4, topk=TOPK, decode_query_lens=(1, 2)),
    "fp8_kv1_spec": dict(kv_cache_dtype="fp8", index_cache_dtype="fp8", num_q_heads=16,
                         num_kv_heads=1, topk=TOPK, decode_query_lens=(1, 2), num_index_heads=1),
    "bf16_kv4": dict(kv_cache_dtype="auto", index_cache_dtype="bf16", num_q_heads=64,
                     num_kv_heads=4, topk=TOPK, decode_query_lens=(1,), num_index_heads=4),
}
BATCHES = (1, 8, 64, 256)  # small batches split the KV, large ones do not
PREFILL_CHUNKS = ((1,), (5,), (17,), (33,), (100,), (129,), (700,), (700, 5, 64))


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    return torch.device("cuda", torch.cuda.current_device())  # the planner indexes by device id


@pytest.mark.parametrize("name", CONFIGS)
def test_plan_is_split_by_dtype(name):
    config = CONFIGS[name]
    items = plan_warmup(**config)
    variants = [item.fmha_variant for item in items if item.fmha_variant]
    labels = [item.label for item in items]
    assert len(set(labels)) == len(labels), "duplicate kernels"
    if config["kv_cache_dtype"] == "nvfp4":
        assert not variants, "NVFP4 KV and index caches need no FP8/BF16 FMHA variant"
        assert any(label.startswith("q8kv4 decode gqa") for label in labels)
        assert any(label.startswith("indexer module") for label in labels)
    else:
        prefix = "1_" if config["kv_cache_dtype"] == "fp8" else "0_"
        assert variants and all(v.startswith(prefix) and "nvfp4" not in v for v in variants)
    for variant in variants:  # every name is a buildable variant
        jit._variant_params_from_name(variant)


def _no_native_builds(monkeypatch):
    real_run = subprocess.run

    def guarded(args, *pargs, **kwargs):
        command = args if isinstance(args, str) else " ".join(map(str, args))
        if "ninja" in command or "nvcc" in command:
            raise AssertionError(f"native build after warmup: {command[:200]}")
        return real_run(args, *pargs, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded)


def _cache(kind, pages, heads, device):
    if kind == "nvfp4":  # per slot: packed E2M1 data, then its E4M3 block scales
        cache = torch.randint(0, 256, (pages, heads, PAGE, 72), dtype=torch.uint8, device=device)
        cache[..., 64:] = 0x30
        return cache
    dtype = torch.float8_e4m3fn if kind == "fp8" else torch.bfloat16
    return (torch.randn((pages, heads, PAGE, 128), device=device) * 0.1).to(dtype)


def _run_decode(config, device, batch, q_len):
    kind = config["kv_cache_dtype"].replace("auto", "bf16")
    heads, q_heads = config["num_kv_heads"], config["num_q_heads"]
    kv_len, pages = 32 * PAGE, 64  # requests share a pool of 64 physical pages
    if kind == "nvfp4":  # head-slot pages: slot 2h is head h's K, slot 2h + 1 its V
        cache = _cache(kind, pages, 2 * heads, device)
        k, v = cache[:, 0::2], cache[:, 1::2]
    else:
        k, v = _cache(kind, pages, heads, device), _cache(kind, pages, heads, device)
    rows = batch * q_len
    q_dtype = torch.bfloat16 if kind == "bf16" else torch.float8_e4m3fn
    q = (torch.randn((rows, q_heads, 128), device=device) * 0.1).to(q_dtype)
    kv_indices = torch.arange(batch * 32, device=device, dtype=torch.int32) % pages
    lists = torch.full((rows, heads, TOPK), -1, dtype=torch.int32, device=device)
    for token in range(q_len):  # history pages ascending, the token's own page last
        local = (kv_len - q_len + token) // PAGE
        lists[token::q_len, :, : TOPK - 1] = torch.arange(TOPK - 1, device=device, dtype=torch.int32)
        lists[token::q_len, :, TOPK - 1] = local
    qo = torch.full((batch,), q_len, dtype=torch.int32)
    kvl = torch.full((batch,), kv_len, dtype=torch.int32)
    plan = fmha_sm100_plan(qo, kvl, q_heads, num_kv_heads=heads, qo_offset=kvl - qo,
                           page_size=PAGE, kv_block_num=TOPK, causal=True,
                           use_fp8_kvcache=kind != "bf16", device=device)
    scales = {}
    if kind == "nvfp4":
        one = torch.ones(1, dtype=torch.float32, device=device)
        scales = dict(k_scale=one, v_scale=one)
    fmha_sm100(q, k, v, plan, kv_indices=kv_indices, kv_block_indexes=lists, sm_scale=0.088,
               **scales)


def _run_index_score(config, device, chunks):
    kind = config["index_cache_dtype"]
    heads = config["num_index_heads"]
    dtype = torch.float8_e4m3fn if kind == "fp8" else torch.bfloat16
    qo = torch.tensor(chunks, dtype=torch.int32)
    kvl = qo + 1000
    counts = [(int(length) + PAGE - 1) // PAGE for length in kvl]
    pages = 32
    k_pages = _cache("fp8" if kind == "fp8" else "bf16", pages, 1, device)
    q = (torch.randn((int(qo.sum()), heads, 128), device=device) * 0.1).to(dtype)
    kv_indices = torch.arange(sum(counts), device=device, dtype=torch.int32) % pages
    plan = _fmha_sm100_plan(qo, kvl, heads, num_kv_heads=1, qo_offset=kvl - qo, page_size=PAGE,
                            output_maxscore=True, causal=True, num_kv_splits=1)
    _fmha_sm100(q, k_pages, k_pages, plan, kv_indices=kv_indices, output_o=False,
                output_maxscore=True, sm_scale=0.088)


@pytest.mark.parametrize("name", CONFIGS)
def test_serving_paths_need_no_build_after_warmup(device, monkeypatch, name):
    config = CONFIGS[name]
    report = warmup(**config)
    before = set(jit.requested_fmha_variants())
    _no_native_builds(monkeypatch)
    for q_len in config["decode_query_lens"]:
        for batch in BATCHES:
            _run_decode(config, device, batch, q_len)
    if config["index_cache_dtype"] in ("fp8", "bf16"):
        for chunks in PREFILL_CHUNKS:
            _run_index_score(config, device, chunks)
    torch.cuda.synchronize()
    requested = set(jit.requested_fmha_variants()) - before
    assert requested <= set(report.fmha_variants), sorted(requested - set(report.fmha_variants))
