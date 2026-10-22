"""Focused correctness, determinism, striding, and CUDA Graph tests."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import torch

from inference.msa_v1.indexer._common.topk_select import _topk_select
from inference.msa_v1.indexer._common.topk_select.build import load_extension
from tests.inference.msa_v1.attention.decode.runtime import run_cuda
from tests.inference.msa_v1.indexer._common.topk_select.reference import (
    assert_quantized_topk_contract,
    exact_forced_tail_reference,
    focused_lengths,
    make_scores,
)

pytestmark = pytest.mark.gpu


def _run_and_check(scores: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    device_scores = torch.from_numpy(scores).to("cuda")
    device_lengths = torch.from_numpy(lengths).to("cuda")
    load_extension()
    outputs = []
    for compact_grid in (False, True):
        actual = run_cuda(
            "TopK correctness",
            partial(_topk_select, device_scores, device_lengths, compact_grid=compact_grid),
        )
        actual_cpu = actual.cpu().numpy()
        assert_quantized_topk_contract(scores, lengths, actual_cpu)
        outputs.append(actual_cpu)
    np.testing.assert_array_equal(*outputs)
    return actual_cpu


def test_at_least_1000_focused_rows() -> None:
    """Cover ragged rows across buffered and streamed candidate boundaries."""

    tested = 0
    widths = (16, 17, 33, 65, 129, 257, 258, 513, 1025, 2049, 4097, 8192)
    widths += (8193, 8194, 10241, 12289, 14337, 16384)
    for seed, max_cols in enumerate(widths, start=1):
        lengths = focused_lengths(max_cols, num_rows=91, seed=seed)
        scores = make_scores(max_cols, lengths, seed=1000 + seed)
        _run_and_check(scores, lengths)
        tested += lengths.size
    assert tested >= 1000


def test_streamed_candidates_and_row_quantization() -> None:
    lengths = np.full(4, 16384, dtype=np.int32)
    scores = make_scores(16384, lengths, seed=1701)
    scores[0, :-1] = np.float32(-1.0)
    scores[0, 8192:16383:256] = np.arange(32, dtype=np.float32) + 1
    scores[1, :-1] = np.float32(-2.0)
    scores[1, :8192:256] = np.arange(32, dtype=np.float32) + 1
    scores[1, 8192:16383:256] = np.arange(32, dtype=np.float32) + 1.5
    scores[2:, :-1] = np.float32(0.0)
    scores[2:, :16] = np.arange(16, dtype=np.float32) * np.float32(0.01)
    scores[2, 16000] = np.float32(-1.0e6)
    scores[3, 16000] = np.float32(1.0e6)
    actual = _run_and_check(scores, lengths)
    np.testing.assert_array_equal(
        actual[:2], exact_forced_tail_reference(scores, lengths)[:2]
    )
    np.testing.assert_array_equal(actual[2], [*range(15), 16383])
    np.testing.assert_array_equal(actual[3], [16000, *range(14), 16383])


@pytest.mark.parametrize("max_cols", (257, 8192, 8193))
@pytest.mark.parametrize("compact_grid", (False, True))
def test_capacity_does_not_change_valid_candidates(max_cols, compact_grid) -> None:
    lengths = focused_lengths(max_cols, num_rows=37, seed=1701)
    scores = make_scores(max_cols, lengths, seed=1702)
    storage = torch.full((37, 16384), float("nan"), device="cuda")
    storage[:, :max_cols].copy_(torch.from_numpy(scores))
    device_lengths = torch.from_numpy(lengths).to("cuda")
    load_extension()
    outputs = []
    for view in (storage[:, :max_cols], storage):
        outputs.append(
            run_cuda(
                "TopK capacity",
                partial(_topk_select, view, device_lengths, compact_grid=compact_grid),
            )
        )
    assert torch.equal(*outputs)
    assert_quantized_topk_contract(scores, lengths, outputs[1].cpu().numpy())


@pytest.mark.parametrize("max_cols", (258, 16384))
def test_ties_and_all_same_use_column_id_tie_break(max_cols: int) -> None:
    num_rows = 19
    lengths = np.full(num_rows, max_cols, dtype=np.int32)
    scores = make_scores(max_cols, lengths, seed=71)
    for row in range(num_rows):
        if row % 2:
            scores[row, : max_cols - 1] = np.float32(1.5)
        else:
            scores[row, : max_cols - 1] = np.resize(
                np.asarray((3.0, 3.0, 2.0, 2.0), dtype=np.float32),
                max_cols - 1,
            )
    actual = _run_and_check(scores, lengths)
    expected = exact_forced_tail_reference(scores, lengths)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("max_cols", (257, 8193, 16384))
@pytest.mark.parametrize("initialize_padding", (False, True))
@pytest.mark.parametrize("compact_grid", (False, True))
def test_strided_rows_and_poison_suffix(
    max_cols: int, initialize_padding: bool, compact_grid: bool
) -> None:
    num_rows = 23
    padding = 19
    lengths = focused_lengths(max_cols, num_rows, seed=81)
    scores = make_scores(max_cols, lengths, seed=82)
    base = torch.empty(
        (num_rows, max_cols + padding),
        dtype=torch.float32,
        device="cuda",
    )
    view = base[:, :max_cols]
    if initialize_padding:
        base.fill_(-123.0)
        view.copy_(torch.from_numpy(scores))
    else:
        for row, length in enumerate(lengths):
            history = int(length) - 1
            view[row, :history].copy_(torch.from_numpy(scores[row, :history]))
    assert view.stride(0) > max_cols and view.stride(1) == 1
    device_lengths = torch.from_numpy(lengths).to("cuda")
    load_extension()
    actual = run_cuda(
        "TopK strided valid-only scores",
        partial(_topk_select, view, device_lengths, compact_grid=compact_grid),
    )
    assert_quantized_topk_contract(scores, lengths, actual.cpu().numpy())


@pytest.mark.parametrize("max_cols", (513, 8193, 16384))
def test_replay_and_row_permutation_are_deterministic(max_cols: int) -> None:
    num_rows = 37
    lengths = focused_lengths(max_cols, num_rows, seed=91)
    scores = make_scores(max_cols, lengths, seed=92)
    device_scores = torch.from_numpy(scores).to("cuda")
    device_lengths = torch.from_numpy(lengths).to("cuda")
    output = torch.empty(num_rows, 16, dtype=torch.int32, device="cuda")
    _topk_select(device_scores, device_lengths, out=output)
    torch.cuda.synchronize()
    baseline = output.clone()
    for _ in range(5):
        _topk_select(device_scores, device_lengths, out=output)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, baseline, atol=0, rtol=0)

    permutation = torch.randperm(num_rows, device="cuda")
    permuted = _topk_select(
        device_scores[permutation].contiguous(),
        device_lengths[permutation].contiguous(),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(permuted, baseline[permutation], atol=0, rtol=0)


@pytest.mark.parametrize("compact_grid", (False, True))
@pytest.mark.parametrize("max_cols", (1025, 16384))
def test_cuda_graph_capture_and_replay(compact_grid: bool, max_cols: int) -> None:
    num_rows = 29
    lengths = focused_lengths(max_cols, num_rows, seed=101)
    scores = make_scores(max_cols, lengths, seed=102)
    device_scores = torch.from_numpy(scores).to("cuda")
    device_lengths = torch.from_numpy(lengths).to("cuda")
    output = torch.empty(num_rows, 16, dtype=torch.int32, device="cuda")
    _topk_select(device_scores, device_lengths, out=output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = _topk_select(
            device_scores, device_lengths, out=output, compact_grid=compact_grid
        )
    assert result.data_ptr() == output.data_ptr()
    graph.replay()
    torch.cuda.synchronize()
    first = output.clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, first, atol=0, rtol=0)
    assert_quantized_topk_contract(scores, lengths, output.cpu().numpy())

    for seed in (103, 104):
        lengths = focused_lengths(max_cols, num_rows, seed=seed)
        scores = make_scores(max_cols, lengths, seed=seed + 100)
        device_scores.copy_(torch.from_numpy(scores))
        device_lengths.copy_(torch.from_numpy(lengths))
        run_cuda("TopK updated Graph replay", graph.replay)
        assert_quantized_topk_contract(scores, lengths, output.cpu().numpy())
        expected = _topk_select(device_scores, device_lengths, compact_grid=compact_grid)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_cuda_graph_capture_requires_out() -> None:
    scores = torch.randn(3, 33, dtype=torch.float32, device="cuda")
    lengths = torch.full((3,), 33, dtype=torch.int32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.warns(UserWarning, match="The CUDA Graph is empty"),
        pytest.raises(RuntimeError, match="requires a preallocated out"),
        torch.cuda.graph(graph),
    ):
        _topk_select(scores, lengths)
