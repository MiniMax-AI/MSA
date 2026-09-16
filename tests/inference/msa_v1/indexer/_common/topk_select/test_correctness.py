"""Focused correctness, determinism, striding, and CUDA Graph tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from inference.msa_v1.indexer._common.topk_select import _topk_select
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
    actual = _topk_select(device_scores, device_lengths)
    torch.cuda.synchronize()
    actual_cpu = actual.cpu().numpy()
    assert_quantized_topk_contract(scores, lengths, actual_cpu)
    return actual_cpu


def test_at_least_1000_focused_rows() -> None:
    """Cover 1092 ragged rows across every family and width rung."""

    tested = 0
    widths = (16, 17, 33, 65, 129, 257, 258, 513, 1025, 2049, 4097, 8192)
    for seed, max_cols in enumerate(widths, start=1):
        lengths = focused_lengths(max_cols, num_rows=91, seed=seed)
        scores = make_scores(max_cols, lengths, seed=1000 + seed)
        _run_and_check(scores, lengths)
        tested += lengths.size
    assert tested >= 1000


def test_ties_and_all_same_use_column_id_tie_break() -> None:
    num_rows = 19
    max_cols = 258
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


def test_strided_rows_and_poison_suffix() -> None:
    num_rows = 23
    max_cols = 257
    padding = 19
    lengths = focused_lengths(max_cols, num_rows, seed=81)
    scores = make_scores(max_cols, lengths, seed=82)
    base = torch.full(
        (num_rows, max_cols + padding),
        -123.0,
        dtype=torch.float32,
        device="cuda",
    )
    view = base[:, :max_cols]
    view.copy_(torch.from_numpy(scores))
    assert view.stride(0) > max_cols and view.stride(1) == 1
    device_lengths = torch.from_numpy(lengths).to("cuda")
    actual = _topk_select(view, device_lengths)
    torch.cuda.synchronize()
    assert_quantized_topk_contract(scores, lengths, actual.cpu().numpy())


def test_replay_and_row_permutation_are_deterministic() -> None:
    num_rows = 37
    max_cols = 513
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


def test_cuda_graph_capture_and_replay() -> None:
    num_rows = 29
    max_cols = 1025
    lengths = focused_lengths(max_cols, num_rows, seed=101)
    scores = make_scores(max_cols, lengths, seed=102)
    device_scores = torch.from_numpy(scores).to("cuda")
    device_lengths = torch.from_numpy(lengths).to("cuda")
    output = torch.empty(num_rows, 16, dtype=torch.int32, device="cuda")
    _topk_select(device_scores, device_lengths, out=output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = _topk_select(device_scores, device_lengths, out=output)
    assert result.data_ptr() == output.data_ptr()
    graph.replay()
    torch.cuda.synchronize()
    first = output.clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, first, atol=0, rtol=0)
    assert_quantized_topk_contract(scores, lengths, output.cpu().numpy())


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
