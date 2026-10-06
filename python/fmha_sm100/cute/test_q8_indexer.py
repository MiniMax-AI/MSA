# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Correctness tests for the Q8KV4/Q8KV8 paged indexers on the vLLM layout."""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import q8_indexer_interface
from q8_indexer_interface import (
    BatchDecodeIndexerQ8KV4Wrapper,
    BatchDecodeIndexerQ8KV8Wrapper,
    BatchPrefillIndexerQ8KV8Wrapper,
    _topk_select,
)

logger = logging.getLogger(__name__)

PAGE_SIZE = 128
HEAD_DIM = 128
TOP_K = 16
MTP = 8
POISON = 123.0
DEADLOCK_SECONDS = 30.0
QUANTIZATION_LEVELS = 65534.0
E4M3_MAX = 448.0
E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
DECODE_FORMATS = ("q8kv8", "q8kv4")
DECODE_WRAPPERS = {"q8kv8": BatchDecodeIndexerQ8KV8Wrapper, "q8kv4": BatchDecodeIndexerQ8KV4Wrapper}
DECODE_HEADS = {"q8kv8": (1, 2, 4), "q8kv4": (1, 2, 4)}
DECODE_CASES = [(fmt, num_heads) for fmt in DECODE_FORMATS for num_heads in DECODE_HEADS[fmt]]
PREFILL_HEADS = (1, 2, 4)


@pytest.fixture(autouse=True)
def _require_sm100_family():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Q8KV4/Q8KV8 indexers require SM100, SM103 or SM107")


def _timed(label, run):
    """Run one launch sequence; exceeding 30 seconds is treated as a deadlock."""

    torch.cuda.synchronize()
    started_at = time.perf_counter()
    result = run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started_at
    logger.info("%s ran in %.3fms", label, elapsed * 1e3)
    assert elapsed < DEADLOCK_SECONDS, f"{label} exceeded the deadlock threshold"
    return result


# ---------------------------------------------------------------------------
# Input construction and independent references
# ---------------------------------------------------------------------------


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _e4m3(shape, generator, scale: float) -> torch.Tensor:
    values = torch.randn(shape, generator=generator, device="cuda") * scale
    return values.clamp_(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)


def _block_table(batch: int, max_pages: int, physical_pages: int, generator) -> torch.Tensor:
    """Scattered, unordered logical-to-physical page maps."""

    rows = [
        torch.randperm(physical_pages, generator=generator, device="cuda")[:max_pages]
        for _ in range(batch)
    ]
    return torch.stack(rows).to(torch.int32)


class _PagedK:
    """Logical K pages plus their vLLM cache encoding for one format."""

    def __init__(self, fmt: str, physical_pages: int, generator, *, page_pad: int, k_scale: float):
        self.page_pad = page_pad
        if fmt == "q8kv8":
            self.set_fp8(_e4m3((physical_pages, PAGE_SIZE, HEAD_DIM), generator, k_scale))
        else:
            packed = torch.randint(
                0,
                256,
                (physical_pages, PAGE_SIZE, HEAD_DIM // 2),
                generator=generator,
                dtype=torch.uint8,
                device="cuda",
            )
            scale = (
                (
                    torch.rand(
                        physical_pages,
                        PAGE_SIZE,
                        HEAD_DIM // 16,
                        generator=generator,
                        device="cuda",
                    )
                    * k_scale
                    + k_scale / 4
                )
                .clamp_(max=E4M3_MAX)
                .to(torch.float8_e4m3fn)
            )
            self.set_nvfp4(packed, scale)

    def set_fp8(self, logical: torch.Tensor) -> None:
        physical_pages = logical.shape[0]
        page_bytes = PAGE_SIZE * HEAD_DIM
        raw = torch.zeros(
            physical_pages, page_bytes + self.page_pad, dtype=torch.uint8, device="cuda"
        )
        raw[:, :page_bytes] = logical.view(torch.uint8).view(physical_pages, page_bytes)
        self.cache = (
            raw[:, :page_bytes].view(torch.float8_e4m3fn).unflatten(1, (PAGE_SIZE, HEAD_DIM))
        )
        self.dequantized = logical.float()

    def set_nvfp4(self, packed: torch.Tensor, scale: torch.Tensor) -> None:
        physical_pages = packed.shape[0]
        data_bytes = PAGE_SIZE * HEAD_DIM // 2
        page_bytes = data_bytes + PAGE_SIZE * HEAD_DIM // 16
        raw = torch.zeros(
            physical_pages, page_bytes + self.page_pad, dtype=torch.uint8, device="cuda"
        )
        raw[:, :data_bytes] = packed.view(physical_pages, data_bytes)
        raw[:, data_bytes:page_bytes] = scale.view(torch.uint8).view(physical_pages, -1)
        self.cache = raw[:, :page_bytes].unflatten(1, (PAGE_SIZE, page_bytes // PAGE_SIZE))
        lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device="cuda")
        codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(
            physical_pages, PAGE_SIZE, HEAD_DIM
        )
        # NVFP4 times the E4M3 block scale, saturated and rounded to E4M3.
        product = lut[codes.long()] * scale.float().repeat_interleave(16, dim=-1)
        self.dequantized = product.clamp_(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()


def _page_scores(q_rows: torch.Tensor, k_pages: torch.Tensor) -> torch.Tensor:
    """FP32 max over each page's tokens: ``[rows, D] x [pages, T, D] -> [rows, pages]``."""

    return torch.einsum("rd,ptd->rpt", q_rows, k_pages).amax(dim=-1)


def _decode_reference(q: torch.Tensor, k: _PagedK, block_table: torch.Tensor) -> torch.Tensor:
    """Scores ``[batch, 8 * num_heads, max_pages]`` with rows ``token * num_heads + head``."""

    batch, max_pages = block_table.shape
    q_rows = q.view(batch, MTP * q.shape[1], HEAD_DIM).float()
    output = torch.empty((batch, q_rows.shape[1], max_pages), dtype=torch.float32, device="cuda")
    page_chunk = 64
    for b in range(batch):
        for begin in range(0, max_pages, page_chunk):
            pages = block_table[b, begin : begin + page_chunk].long()
            output[b, :, begin : begin + page_chunk] = _page_scores(q_rows[b], k.dequantized[pages])
    return output


def _decode_lengths(batch: int, max_pages: int, seed: int) -> torch.Tensor:
    """KV lengths around page and MTP-chunk boundaries, from 8 up to the table capacity."""

    candidates = [8, 9, 15, 120, 121, 127, 128, 129, 135, 136]
    for page in range(1, max_pages + 1):
        boundary = page * PAGE_SIZE
        candidates.extend(
            v for v in (boundary - 7, boundary - 1, boundary, boundary + 1, boundary + 7)
        )
    candidates = [v for v in candidates if MTP <= v <= max_pages * PAGE_SIZE]
    values = [candidates[(i * 17 + seed * 13) % len(candidates)] for i in range(batch)]
    values[-1] = max_pages * PAGE_SIZE
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _decode_local_pages(seq_lens: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Local page of every ``[batch, token * num_heads + head]`` row."""

    positions = seq_lens[:, None] - MTP + torch.arange(MTP, dtype=torch.int32, device="cuda")
    local = torch.div(positions, PAGE_SIZE, rounding_mode="floor")
    return local.repeat_interleave(num_heads, dim=1)


def _decode_valid_pages(seq_lens: torch.Tensor, num_heads: int) -> torch.Tensor:
    return (_decode_local_pages(seq_lens, num_heads) + 1).reshape(-1)


def _assert_decode_scores(actual, expected, seq_lens) -> None:
    pages = torch.arange(actual.shape[-1], device="cuda").view(1, 1, -1)
    scored = pages < _decode_local_pages(seq_lens, actual.shape[1] // MTP)[:, :, None]
    assert torch.isfinite(actual[scored]).all()
    torch.testing.assert_close(actual[scored], expected[scored], atol=1e-4, rtol=1e-4)
    assert torch.all(actual[~scored] == POISON)


def _exact_forced_tail(scores: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    output = np.full((lengths.size, TOP_K), -1, dtype=np.int32)
    for row, length in enumerate(lengths.tolist()):
        if length <= TOP_K:
            output[row, :length] = np.arange(length, dtype=np.int32)
            continue
        order = np.argsort(-scores[row, : length - 1], kind="stable")
        output[row, : TOP_K - 1] = order[: TOP_K - 1]
        output[row, TOP_K - 1] = length - 1
    return output


def _assert_topk_contract(scores: torch.Tensor, lengths: torch.Tensor, topk: torch.Tensor) -> None:
    """Exact structure plus the 16-bit key quantization gate (1.5 steps) for every row."""

    scores = scores.reshape(lengths.numel(), -1).cpu().numpy()
    lengths = lengths.cpu().numpy()
    actual = topk.reshape(lengths.size, TOP_K).cpu().numpy().astype(np.int64)
    expected = _exact_forced_tail(scores, lengths)
    for row, length in enumerate(lengths.tolist()):
        got = actual[row]
        if length <= TOP_K:
            np.testing.assert_array_equal(got, expected[row])
            continue
        ranked = got[: TOP_K - 1]
        assert int(got[TOP_K - 1]) == length - 1
        assert np.all((ranked >= 0) & (ranked < length - 1))
        assert np.unique(ranked).size == TOP_K - 1
        history = scores[row, : length - 1].astype(np.float64)
        selected = scores[row, ranked].astype(np.float64)
        exact_ids = expected[row, : TOP_K - 1].astype(np.int64)
        exact = scores[row, exact_ids].astype(np.float64)
        low, high = float(history.min()), float(history.max())
        tolerance = 1.5 * (high - low) / QUANTIZATION_LEVELS if high > low else 0.0
        threshold = float(exact[TOP_K - 2])
        assert np.all(selected[1:] <= selected[:-1] + tolerance)
        assert np.all(selected >= threshold - tolerance)
        assert set(exact_ids[exact > threshold + tolerance].tolist()) <= set(ranked.tolist())


def _make_decode(
    fmt,
    batch,
    max_pages,
    seed,
    *,
    num_heads=1,
    page_pad=0,
    q_scale=0.5,
    k_scale=0.5,
    seq_lens=None,
):
    generator = _generator(seed)
    physical_pages = max_pages + 3
    q = _e4m3((batch * MTP, num_heads, HEAD_DIM), generator, q_scale)
    k = _PagedK(fmt, physical_pages, generator, page_pad=page_pad, k_scale=k_scale)
    block_table = _block_table(batch, max_pages, physical_pages, generator)
    if seq_lens is None:
        seq_lens = _decode_lengths(batch, max_pages, seed)
    return q, k, block_table, seq_lens


def _run_decode_scores(wrapper, q, k_cache):
    wrapper._scores.fill_(POISON)
    return wrapper._run_scores(q, k_cache)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt,num_heads", DECODE_CASES)
@pytest.mark.parametrize(
    "batch,max_pages,page_pad,q_scale,k_scale",
    [
        (1, 1, 0, 0.5, 0.5),
        (5, 6, 256, 0.25, 1.0),
        (31, 17, 0, 1.0, 0.25),
        (129, 4, 512, 0.5, 2.0),
        (257, 3, 0, 2.0, 0.125),
        (1025, 2, 0, 0.5, 0.5),
    ],
)
def test_decode_matches_reference(fmt, num_heads, batch, max_pages, page_pad, q_scale, k_scale):
    """Full scores and TopK over irregular lengths, scattered pages, and padded pages."""

    seed = 1000 * batch + max_pages
    q, k, block_table, seq_lens = _make_decode(
        fmt,
        batch,
        max_pages,
        seed,
        num_heads=num_heads,
        page_pad=page_pad,
        q_scale=q_scale,
        k_scale=k_scale,
    )
    wrapper = DECODE_WRAPPERS[fmt](num_heads=num_heads)
    wrapper.plan(block_table, seq_lens)
    wrapper.run(q, k.cache)  # compile outside the timed run
    scores = _timed(f"{fmt}-decode-scores", lambda: _run_decode_scores(wrapper, q, k.cache))
    _assert_decode_scores(scores, _decode_reference(q, k, block_table), seq_lens)

    topk = _timed(f"{fmt}-decode-topk", lambda: wrapper.run(q, k.cache).clone())
    assert topk.shape == (batch * MTP, num_heads, TOP_K)
    lengths = _decode_valid_pages(seq_lens, num_heads)
    torch.testing.assert_close(wrapper._num_valid_pages, lengths, atol=0, rtol=0)
    _assert_topk_contract(scores, lengths, topk)

    for _ in range(2):
        repeated_scores = _run_decode_scores(wrapper, q, k.cache).clone()
        assert torch.equal(repeated_scores, scores)
        assert torch.equal(wrapper.run(q, k.cache), topk)


@pytest.mark.parametrize("fmt,num_heads", DECODE_CASES)
def test_decode_long_context_upper_bound(fmt, num_heads):
    """1M-token rows at the 8192-page table limit next to short rows."""

    batch, max_pages = 8, 8192
    seq_lens = torch.tensor(
        [1 << 20, (1 << 19) + 1, 100_001, 129, 8, 9, 16_384, 1_000_000],
        dtype=torch.int32,
        device="cuda",
    )
    q, k, block_table, seq_lens = _make_decode(
        fmt, batch, max_pages, 91, num_heads=num_heads, seq_lens=seq_lens
    )
    wrapper = DECODE_WRAPPERS[fmt](num_heads=num_heads)
    wrapper.plan(block_table, seq_lens)
    wrapper.run(q, k.cache)
    scores = _timed(f"{fmt}-decode-1m", lambda: _run_decode_scores(wrapper, q, k.cache))
    _assert_decode_scores(scores, _decode_reference(q, k, block_table), seq_lens)
    lengths = _decode_valid_pages(seq_lens, num_heads)
    _assert_topk_contract(scores, lengths, wrapper.run(q, k.cache))


@pytest.mark.parametrize("fmt,num_heads", DECODE_CASES)
def test_decode_boundary_values(fmt, num_heads):
    """All-zero K ties break toward lower pages; saturated E4M3 stays finite."""

    batch, max_pages = 4, 40
    seq_lens = torch.full((batch,), max_pages * PAGE_SIZE, dtype=torch.int32, device="cuda")
    q, k, block_table, seq_lens = _make_decode(
        fmt, batch, max_pages, 7, num_heads=num_heads, seq_lens=seq_lens
    )
    wrapper = DECODE_WRAPPERS[fmt](num_heads=num_heads)
    wrapper.plan(block_table, seq_lens)
    lengths = _decode_valid_pages(seq_lens, num_heads)

    pages = k.cache.shape[0]
    if fmt == "q8kv8":
        k.set_fp8(torch.zeros((pages, PAGE_SIZE, HEAD_DIM), device="cuda").to(torch.float8_e4m3fn))
    else:
        k.set_nvfp4(
            torch.zeros((pages, PAGE_SIZE, HEAD_DIM // 2), dtype=torch.uint8, device="cuda"),
            torch.ones((pages, PAGE_SIZE, HEAD_DIM // 16), device="cuda").to(torch.float8_e4m3fn),
        )
    scores = _run_decode_scores(wrapper, q, k.cache)
    _assert_decode_scores(scores, _decode_reference(q, k, block_table), seq_lens)
    expected = torch.cat(
        (
            torch.arange(TOP_K - 1, device="cuda").expand(lengths.numel(), -1),
            (lengths - 1)[:, None],
        ),
        dim=1,
    ).to(torch.int32)
    assert torch.equal(wrapper.run(q, k.cache).view(-1, TOP_K), expected)

    q.copy_(torch.full(q.shape, E4M3_MAX, device="cuda").to(torch.float8_e4m3fn))
    if fmt == "q8kv8":
        k.set_fp8(
            torch.full((pages, PAGE_SIZE, HEAD_DIM), -E4M3_MAX, device="cuda").to(
                torch.float8_e4m3fn
            )
        )
    else:
        # Code 0x7 is 6.0; with a 448 block scale the product saturates to 448.
        k.set_nvfp4(
            torch.full((pages, PAGE_SIZE, HEAD_DIM // 2), 0x77, dtype=torch.uint8, device="cuda"),
            torch.full((pages, PAGE_SIZE, HEAD_DIM // 16), E4M3_MAX, device="cuda").to(
                torch.float8_e4m3fn
            ),
        )
    scores = _run_decode_scores(wrapper, q, k.cache)
    _assert_decode_scores(scores, _decode_reference(q, k, block_table), seq_lens)
    _assert_topk_contract(scores, lengths, wrapper.run(q, k.cache))


@pytest.mark.parametrize("fmt,num_heads", DECODE_CASES)
def test_decode_plan_reuse_and_replan(fmt, num_heads):
    """One plan serves several layers; replanning picks up new lengths and pages."""

    batch, max_pages = 129, 5
    wrapper_class = DECODE_WRAPPERS[fmt]
    workspace = torch.empty(
        wrapper_class.workspace_size(batch, num_heads=num_heads), dtype=torch.uint8, device="cuda"
    )
    wrapper = wrapper_class(workspace, num_heads=num_heads)
    for plan_seed in (11, 12):
        _, _, block_table, seq_lens = _make_decode(fmt, batch, max_pages, plan_seed)
        wrapper.plan(block_table, seq_lens)
        for layer_seed in (21, 22):
            q, k, _, _ = _make_decode(
                fmt, batch, max_pages, layer_seed * plan_seed, num_heads=num_heads
            )
            scores = _run_decode_scores(wrapper, q, k.cache)
            _assert_decode_scores(scores, _decode_reference(q, k, block_table), seq_lens)


@pytest.mark.parametrize("fmt,num_heads", DECODE_CASES)
def test_decode_cuda_graph_replay_and_replan(fmt, num_heads):
    batch, max_pages = 64, 6
    q, k, block_table, seq_lens = _make_decode(fmt, batch, max_pages, 31, num_heads=num_heads)
    wrapper = DECODE_WRAPPERS[fmt](
        num_heads=num_heads,
        use_cuda_graph=True,
        block_table_buffer=torch.empty_like(block_table),
        seq_lens_buffer=torch.empty_like(seq_lens),
    )
    wrapper.plan(block_table, seq_lens)
    out = torch.empty((batch * MTP, num_heads, TOP_K), dtype=torch.int32, device="cuda")
    wrapper.run(q, k.cache, out=out)
    expected = out.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, k.cache, out=out)
        with pytest.raises(RuntimeError, match=r"plan\(\) must be called outside"):
            wrapper.plan(block_table, seq_lens)
    out.fill_(-777)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)

    _, _, next_table, next_lens = _make_decode(fmt, batch, max_pages, 32)
    wrapper.plan(next_table, next_lens)
    wrapper._scores.fill_(POISON)
    graph.replay()
    torch.cuda.synchronize()
    _assert_decode_scores(wrapper._scores, _decode_reference(q, k, next_table), next_lens)
    lengths = _decode_valid_pages(next_lens, num_heads)
    _assert_topk_contract(wrapper._scores, lengths, out)


@pytest.mark.parametrize("fmt,num_heads", DECODE_CASES)
def test_decode_concurrent_streams_do_not_alias(fmt, num_heads):
    """Independent wrappers on two streams match their serial results."""

    inputs = [_make_decode(fmt, 96, 7, seed, num_heads=num_heads) for seed in (41, 42)]
    wrappers = [DECODE_WRAPPERS[fmt](num_heads=num_heads) for _ in inputs]
    serial = []
    for wrapper, (q, k, block_table, seq_lens) in zip(wrappers, inputs, strict=True):
        wrapper.plan(block_table, seq_lens)
        serial.append(wrapper.run(q, k.cache).clone())
    streams = [torch.cuda.Stream() for _ in inputs]
    outputs = [torch.empty_like(result) for result in serial]
    torch.cuda.synchronize()
    for _ in range(3):
        for stream, wrapper, (q, k, _, _), out in zip(
            streams, wrappers, inputs, outputs, strict=True
        ):
            with torch.cuda.stream(stream):
                wrapper.run(q, k.cache, out=out)
    torch.cuda.synchronize()
    for out, expected in zip(outputs, serial, strict=True):
        assert torch.equal(out, expected)


def test_decode_compile_is_shape_independent_and_workspace_is_released():
    """Batch, page-table width, and lengths reuse one compiled kernel per format and heads."""

    baseline_keys = None
    for batch, max_pages in ((3, 2), (77, 9), (300, 4)):
        for fmt, num_heads in DECODE_CASES:
            q, k, block_table, seq_lens = _make_decode(
                fmt, batch, max_pages, batch, num_heads=num_heads
            )
            wrapper = DECODE_WRAPPERS[fmt](num_heads=num_heads)
            wrapper.plan(block_table, seq_lens)
            wrapper.run(q, k.cache)
            del wrapper
        keys = set(q8_indexer_interface._COMPILE_CACHE)
        baseline_keys = keys if baseline_keys is None else baseline_keys
        assert keys == baseline_keys

    torch.cuda.synchronize()
    gc.collect()
    q, k, block_table, seq_lens = _make_decode("q8kv8", 4096, 64, 5)
    before = torch.cuda.memory_allocated()
    wrapper = BatchDecodeIndexerQ8KV8Wrapper()
    wrapper.plan(block_table, seq_lens)
    wrapper.run(q, k.cache)
    assert torch.cuda.memory_allocated() > before
    del wrapper
    gc.collect()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


def _int32_misaligned_copy(tensor: torch.Tensor) -> torch.Tensor:
    """Copy metadata to a view 4 bytes past 16-byte alignment, like vLLM's seq_lens[lo:hi]."""

    storage = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
    view = storage[1:].view(tensor.shape)
    view.copy_(tensor)
    assert view.data_ptr() % 16 == 4
    return view


@pytest.mark.parametrize(
    "phase,num_heads", [*((f"{fmt}-decode", h) for fmt, h in DECODE_CASES), ("prefill", 4)]
)
def test_metadata_slices_need_only_int32_alignment(phase, num_heads):
    if phase == "prefill":
        inputs = _make_prefill([300, 1, 700], [0, 1000, 130], seed=17, num_heads=num_heads)
        metadata = ("cu_seqlens_q", "seq_lens", "block_table")
        results = []
        for shifted in (
            inputs,
            {**inputs, **{n: _int32_misaligned_copy(inputs[n]) for n in metadata}},
        ):
            wrapper = BatchPrefillIndexerQ8KV8Wrapper(num_heads=num_heads)
            _plan_prefill(wrapper, shifted)
            results.append(wrapper.run(shifted["q"], shifted["k"].cache).clone())
    else:
        fmt = phase.split("-")[0]
        q, k, block_table, seq_lens = _make_decode(fmt, 37, 5, 13, num_heads=num_heads)
        results = []
        for table, lens in (
            (block_table, seq_lens),
            tuple(map(_int32_misaligned_copy, (block_table, seq_lens))),
        ):
            wrapper = DECODE_WRAPPERS[fmt](num_heads=num_heads)
            wrapper.plan(table, lens)
            results.append(wrapper.run(q, k.cache).clone())
    assert torch.equal(results[0], results[1])


_VENDORED_LAYOUT_CHECK = """
import importlib.util
import sys

import torch

if importlib.util.find_spec("fmha_sm100") is not None:
    sys.exit(77)
from vendor.fmha_sm100.sparse import BatchDecodeIndexerQ8KV4Wrapper

batch, pages = 2, 3
block_table = torch.arange(batch * pages, dtype=torch.int32, device="cuda").view(batch, pages)
seq_lens = torch.full((batch,), pages * 128, dtype=torch.int32, device="cuda")
q = torch.zeros((batch * 8, 1, 128), device="cuda").to(torch.float8_e4m3fn)
k_cache = torch.zeros((batch * pages, 128, 72), dtype=torch.uint8, device="cuda")
wrapper = BatchDecodeIndexerQ8KV4Wrapper()
wrapper.plan(block_table, seq_lens)
topk = wrapper.run(q, k_cache).view(-1, 16).cpu()
assert topk[:, :3].eq(torch.arange(3)).all() and topk[:, 3:].eq(-1).all(), topk
"""


def test_vendored_package_layout_loads_csrc_modules(tmp_path):
    """Vendored under another package (vLLM), cute/ cannot import ``fmha_sm100`` by name."""

    package_dir = Path(q8_indexer_interface.__file__).resolve().parents[1]
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "__init__.py").write_text("")
    (tmp_path / "vendor" / "fmha_sm100").symlink_to(package_dir, target_is_directory=True)
    result = subprocess.run(
        [sys.executable, "-c", _VENDORED_LAYOUT_CHECK],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode == 77:
        pytest.skip("an installed fmha_sm100 hides the vendored layout")
    assert result.returncode == 0, result.stderr[-4000:]


@pytest.mark.parametrize("fmt", DECODE_FORMATS)
def test_decode_rejects_invalid_inputs(fmt):
    q, k, block_table, seq_lens = _make_decode(fmt, 4, 3, 3)
    wrapper = DECODE_WRAPPERS[fmt]()
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called before run"):
        wrapper.run(q, k.cache)
    with pytest.raises(TypeError, match="block_table"):
        wrapper.plan(block_table.long(), seq_lens)
    with pytest.raises(ValueError, match="max_blocks"):
        wrapper.plan(torch.zeros((4, 8193), dtype=torch.int32, device="cuda"), seq_lens)
    with pytest.raises(ValueError, match="use_cuda_graph"):
        DECODE_WRAPPERS[fmt](block_table_buffer=block_table)
    for num_heads in (0, 3, 8, *({1, 2, 4} - set(DECODE_HEADS[fmt]))):
        with pytest.raises(ValueError, match="num_heads must be one of"):
            DECODE_WRAPPERS[fmt](num_heads=num_heads)
        with pytest.raises(ValueError, match="num_heads must be one of"):
            DECODE_WRAPPERS[fmt].workspace_size(4, num_heads=num_heads)
    wrapper.plan(block_table, seq_lens)
    with pytest.raises(ValueError, match="q must have shape"):
        wrapper.run(q.view(4, MTP, HEAD_DIM), k.cache)
    with pytest.raises(ValueError, match="q must have shape"):
        wrapper.run(q.expand(-1, 4, -1).contiguous(), k.cache)
    with pytest.raises(TypeError, match="q must have dtype"):
        wrapper.run(q.to(torch.bfloat16), k.cache)
    with pytest.raises(ValueError, match="k_cache must have shape"):
        wrapper.run(q, k.cache[:, :64])
    with pytest.raises(ValueError, match="k_cache pages must be contiguous"):
        wrapper.run(q, k.cache.transpose(0, 1).contiguous().transpose(0, 1))
    with pytest.raises(ValueError, match="out must have shape"):
        wrapper.run(q, k.cache, out=torch.empty((4 * MTP, TOP_K), dtype=torch.int32, device="cuda"))
    other = DECODE_FORMATS[1 - DECODE_FORMATS.index(fmt)]
    _, other_k, _, _ = _make_decode(other, 4, 3, 3)
    with pytest.raises((TypeError, ValueError)):
        wrapper.run(q, other_k.cache)


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------

PREFILL_CASES = {
    "single_no_prefix": ([1000], [0]),
    "page_boundaries": ([127, 128, 129, 1, 1, 256], [0, 1, 127, 128, 255, 384]),
    "mixed_prefixes": ([300, 1, 700, 2049], [0, 1000, 130, 5000]),
    "batch_beyond_32": ([17 + 3 * i for i in range(40)], [128 * (i % 7) + i for i in range(40)]),
    "large_q_tile_chunks": ([16_384], [512]),
    "long_prefix": ([513, 2], [200_000, 131_071]),
}


def _make_prefill(
    query_lens, prefix_lens, seed, *, num_heads=1, page_pad=0, q_scale=0.25, k_scale=0.25
):
    generator = _generator(seed)
    batch = len(query_lens)
    seq_lens_list = [q + p for q, p in zip(query_lens, prefix_lens, strict=True)]
    max_pages = max(-(-length // PAGE_SIZE) for length in seq_lens_list)
    physical_pages = max_pages * min(batch, 4) + 5
    k = _PagedK("q8kv8", physical_pages, generator, page_pad=page_pad, k_scale=k_scale)
    total_q = sum(query_lens)
    return {
        "q": _e4m3((total_q, num_heads, HEAD_DIM), generator, q_scale),
        "k": k,
        "block_table": _block_table(batch, max_pages, physical_pages, generator),
        "cu_seqlens_q": torch.tensor(np.cumsum([0, *query_lens]), dtype=torch.int32, device="cuda"),
        "seq_lens": torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda"),
        "total_q": total_q,
        "max_seqlen_q": max(query_lens),
        "max_seqlen_k": max(seq_lens_list),
        "query_lens": query_lens,
        "prefix_lens": prefix_lens,
    }


def _plan_prefill(wrapper, inputs):
    wrapper.plan(
        inputs["cu_seqlens_q"],
        inputs["seq_lens"],
        inputs["block_table"],
        total_q=inputs["total_q"],
        max_seqlen_q=inputs["max_seqlen_q"],
        max_seqlen_k=inputs["max_seqlen_k"],
    )


def _prefill_lengths(inputs) -> torch.Tensor:
    """Candidate pages of every ``token * num_heads + head`` row."""

    values = [
        (prefix + i) // PAGE_SIZE + 1
        for q_len, prefix in zip(inputs["query_lens"], inputs["prefix_lens"], strict=True)
        for i in range(q_len)
    ]
    lengths = torch.tensor(values, dtype=torch.int32, device="cuda")
    return lengths.repeat_interleave(inputs["q"].shape[1])


def _prefill_reference(inputs, max_pages: int) -> torch.Tensor:
    """Historical-page scores with NaN at the local page and beyond."""

    num_heads = inputs["q"].shape[1]
    q_flat = inputs["q"].reshape(-1, HEAD_DIM)
    output = torch.full((q_flat.shape[0], max_pages), float("nan"), device="cuda")
    lengths = _prefill_lengths(inputs)
    offsets = [offset * num_heads for offset in inputs["cu_seqlens_q"].tolist()]
    k = inputs["k"].dequantized
    row_chunk, page_chunk = 1024, 64
    for b in range(len(inputs["query_lens"])):
        for row in range(offsets[b], offsets[b + 1], row_chunk):
            rows = slice(row, min(row + row_chunk, offsets[b + 1]))
            history = int(lengths[rows].max()) - 1
            q_rows = q_flat[rows].float()
            for page in range(0, history, page_chunk):
                cols = slice(page, min(page + page_chunk, history))
                pages = inputs["block_table"][b, cols].long()
                scores = _page_scores(q_rows, k[pages])
                valid = torch.arange(cols.start, cols.stop, device="cuda")[None, :] < (
                    lengths[rows, None] - 1
                )
                output[rows, cols] = torch.where(valid, scores, output[rows, cols])
    return output


def _run_prefill_scores(wrapper, q, k_cache):
    wrapper._state.scores.fill_(float("nan"))
    return wrapper._run_scores(q, k_cache)


@pytest.mark.parametrize("num_heads", PREFILL_HEADS)
@pytest.mark.parametrize("case", sorted(PREFILL_CASES))
def test_prefill_matches_reference(case, num_heads):
    query_lens, prefix_lens = PREFILL_CASES[case]
    inputs = _make_prefill(
        query_lens,
        prefix_lens,
        seed=len(case) * 7919,
        num_heads=num_heads,
        page_pad=256 if "prefix" in case else 0,
    )
    wrapper = BatchPrefillIndexerQ8KV8Wrapper(num_heads=num_heads)
    _plan_prefill(wrapper, inputs)
    q, k_cache = inputs["q"], inputs["k"].cache
    wrapper.run(q, k_cache)
    scores = _timed(
        f"prefill-{case}-scores", lambda: _run_prefill_scores(wrapper, q, k_cache).clone()
    )
    assert int(wrapper._state.plan_error.item()) == 0
    lengths = _prefill_lengths(inputs)
    torch.testing.assert_close(wrapper._state.num_valid_pages, lengths, atol=0, rtol=0)
    expected = _prefill_reference(inputs, scores.shape[1])
    assert torch.equal(torch.isnan(scores), torch.isnan(expected))
    assert torch.isfinite(scores[~torch.isnan(scores)]).all()
    torch.testing.assert_close(scores, expected, atol=2e-4, rtol=2e-4, equal_nan=True)

    topk = _timed(f"prefill-{case}-topk", lambda: wrapper.run(q, k_cache).clone())
    assert topk.shape == (inputs["total_q"], num_heads, TOP_K)
    _assert_topk_contract(scores, lengths, topk)
    for _ in range(2):
        repeated = _run_prefill_scores(wrapper, q, k_cache)
        assert torch.equal(repeated.nan_to_num(POISON), scores.nan_to_num(POISON))
        assert torch.equal(wrapper.run(q, k_cache), topk)


@pytest.mark.parametrize("num_heads", (1, 4))
def test_prefill_plan_reuse_replan_and_cuda_graph(num_heads):
    query_lens, prefix_lens = [300, 45, 900], [100, 2000, 0]
    inputs = _make_prefill(query_lens, prefix_lens, seed=5, num_heads=num_heads)
    wrapper = BatchPrefillIndexerQ8KV8Wrapper(num_heads=num_heads)
    _plan_prefill(wrapper, inputs)
    layer = _make_prefill(query_lens, prefix_lens, seed=6, num_heads=num_heads)
    for q, k in ((inputs["q"], inputs["k"]), (layer["q"], layer["k"])):
        scores = _run_prefill_scores(wrapper, q, k.cache)
        expected = _prefill_reference({**inputs, "q": q, "k": k}, scores.shape[1])
        torch.testing.assert_close(scores, expected, atol=2e-4, rtol=2e-4, equal_nan=True)

    # Shorter prefixes written in place keep the planned buffers valid.
    inputs["seq_lens"].copy_(torch.tensor([350, 1045, 900], dtype=torch.int32))
    inputs["prefix_lens"] = [50, 1000, 0]
    wrapper.replan()
    scores = _run_prefill_scores(wrapper, inputs["q"], inputs["k"].cache)
    torch.testing.assert_close(
        scores, _prefill_reference(inputs, scores.shape[1]), atol=2e-4, rtol=2e-4, equal_nan=True
    )

    out = torch.empty((inputs["total_q"], num_heads, TOP_K), dtype=torch.int32, device="cuda")
    wrapper.run(inputs["q"], inputs["k"].cache, out=out)
    expected = out.clone()
    expected_scores = wrapper._state.scores.nan_to_num(POISON)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(inputs["q"], inputs["k"].cache, out=out)
        with pytest.raises(RuntimeError, match=r"plan\(\) must be called outside"):
            _plan_prefill(BatchPrefillIndexerQ8KV8Wrapper(num_heads=num_heads), inputs)
    # Poison both stages so the replay must rerun the score kernel and the TopK.
    wrapper._state.scores.fill_(float("nan"))
    out.fill_(-777)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    assert torch.equal(wrapper._state.scores.nan_to_num(POISON), expected_scores)


def test_prefill_compile_is_shape_independent():
    """Batch, total_q, lengths, and varlen distribution reuse one compiled kernel set."""

    keys = None
    for query_lens, prefix_lens in (
        ([5], [0]),
        ([700, 3, 90], [128, 0, 4000]),
        ([1] * 33, list(range(33))),
    ):
        for num_heads in PREFILL_HEADS:
            inputs = _make_prefill(
                query_lens, prefix_lens, seed=sum(query_lens), num_heads=num_heads
            )
            wrapper = BatchPrefillIndexerQ8KV8Wrapper(num_heads=num_heads)
            _plan_prefill(wrapper, inputs)
            wrapper.run(inputs["q"], inputs["k"].cache)
        current = set(q8_indexer_interface._COMPILE_CACHE)
        keys = current if keys is None else keys
        assert current == keys


def test_prefill_rejects_invalid_inputs():
    inputs = _make_prefill([10, 20], [0, 100], seed=9)
    for num_heads in (0, 3, 8):
        with pytest.raises(ValueError, match="num_heads must be one of"):
            BatchPrefillIndexerQ8KV8Wrapper(num_heads=num_heads)
    wrapper = BatchPrefillIndexerQ8KV8Wrapper()
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called before"):
        wrapper.run(inputs["q"], inputs["k"].cache)
    with pytest.raises(ValueError, match="max_seqlen_k >= max_seqlen_q"):
        wrapper.plan(
            inputs["cu_seqlens_q"],
            inputs["seq_lens"],
            inputs["block_table"],
            total_q=30,
            max_seqlen_q=20,
            max_seqlen_k=10,
        )
    with pytest.raises(ValueError, match="smaller than"):
        wrapper.plan(
            inputs["cu_seqlens_q"],
            inputs["seq_lens"],
            inputs["block_table"],
            total_q=30,
            max_seqlen_q=20,
            max_seqlen_k=10_000,
        )
    with pytest.raises(ValueError, match="seq_lens must have shape"):
        wrapper.plan(
            inputs["cu_seqlens_q"],
            inputs["seq_lens"][:1],
            inputs["block_table"],
            total_q=30,
            max_seqlen_q=20,
            max_seqlen_k=120,
        )
    _plan_prefill(wrapper, inputs)
    with pytest.raises(ValueError, match="q must have shape"):
        wrapper.run(inputs["q"][:-1], inputs["k"].cache)
    with pytest.raises(ValueError, match="q must have shape"):
        wrapper.run(inputs["q"].expand(-1, 2, -1).contiguous(), inputs["k"].cache)
    with pytest.raises(TypeError, match="k_cache must have dtype"):
        wrapper.run(inputs["q"], inputs["k"].cache.view(torch.uint8))


# ---------------------------------------------------------------------------
# TopK selection
# ---------------------------------------------------------------------------


def _focused_topk_inputs(max_cols: int, num_rows: int, seed: int):
    """Ragged lengths at every width rung with ties, huge ranges, and near-ties."""

    rng = np.random.default_rng(seed)
    boundaries = [1, 2, 15, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 258, max_cols - 1, max_cols]
    lengths = rng.integers(1, max_cols + 1, size=num_rows).astype(np.int32)
    selected = [value for value in boundaries if 1 <= value <= max_cols]
    lengths[: len(selected)] = selected[:num_rows]
    scores = np.full((num_rows, max_cols), np.float32(1.0e30), dtype=np.float32)
    for row, length in enumerate(lengths.tolist()):
        history = length - 1
        if history:
            mode = row % 6
            if mode == 0:
                values = rng.uniform(-100.0, 100.0, history)
            elif mode == 1:
                values = np.full(history, 1.5)
            elif mode == 2:
                values = rng.choice((-100.0, -50.0, 0.0, 50.0), history)
            elif mode == 3:
                values = rng.uniform(-1.0e20, 1.0e20, history)
            elif mode == 4:
                values = (
                    np.linspace(-1.0, 1.0, history)
                    + rng.choice((-0.49, 0.49), history) / QUANTIZATION_LEVELS
                )
            else:
                values = rng.standard_normal(history) * 7.0
            scores[row, :history] = values
        scores[row, history] = np.float32(1000.0)
    return torch.from_numpy(scores).cuda(), torch.from_numpy(lengths).cuda()


@pytest.mark.parametrize("max_cols", (16, 17, 33, 65, 129, 257, 258, 513, 1025, 2049, 4097, 8192))
def test_topk_select_contract(max_cols):
    scores, lengths = _focused_topk_inputs(max_cols, num_rows=91, seed=max_cols)
    padded = torch.full((scores.shape[0], max_cols + 19), -123.0, device="cuda")
    padded[:, :max_cols] = scores
    out = torch.empty((scores.shape[0], 1, TOP_K), dtype=torch.int32, device="cuda")
    _topk_select(padded[:, :max_cols], lengths, out)
    _assert_topk_contract(scores, lengths, out)
    baseline = out.clone()
    for _ in range(3):
        assert torch.equal(_topk_select(scores, lengths, out), baseline)


def test_topk_select_exact_ties():
    num_rows, max_cols = 19, 258
    lengths = torch.full((num_rows,), max_cols, dtype=torch.int32, device="cuda")
    scores = torch.empty((num_rows, max_cols), device="cuda")
    scores[0::2] = torch.tensor((3.0, 3.0, 2.0, 2.0), device="cuda").repeat(max_cols // 4 + 1)[
        :max_cols
    ]
    scores[1::2] = 1.5
    out = torch.empty((num_rows, 1, TOP_K), dtype=torch.int32, device="cuda")
    _topk_select(scores, lengths, out)
    expected = _exact_forced_tail(scores.cpu().numpy(), lengths.cpu().numpy())
    np.testing.assert_array_equal(out.view(num_rows, TOP_K).cpu().numpy(), expected)
