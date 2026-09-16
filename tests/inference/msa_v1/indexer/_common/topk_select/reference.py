"""Focused score generators and the PR #23 quantization-accuracy gate."""

from __future__ import annotations

import numpy as np

TOP_K = 16
_QUANTIZATION_LEVELS = 65534.0


def focused_lengths(max_cols: int, num_rows: int, seed: int) -> np.ndarray:
    """Build deterministic ragged lengths with all relevant boundaries."""

    boundaries = (
        1,
        2,
        15,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        258,
        max_cols - 2,
        max_cols - 1,
        max_cols,
    )
    selected = [value for value in boundaries if 1 <= value <= max_cols]
    rng = np.random.default_rng(seed)
    random_values = rng.integers(1, max_cols + 1, size=num_rows, dtype=np.int32)
    result = random_values
    result[: min(len(selected), num_rows)] = selected[:num_rows]
    return np.ascontiguousarray(result, dtype=np.int32)


def make_scores(
    max_cols: int,
    lengths: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Generate finite histories and a high-valued poison suffix per row."""

    rng = np.random.default_rng(seed)
    scores = np.full((lengths.size, max_cols), np.float32(1.0e30), dtype=np.float32)
    for row, length_value in enumerate(lengths):
        length = int(length_value)
        history = length - 1
        mode = row % 6
        if history > 0:
            if mode == 0:
                values = rng.uniform(-100.0, 100.0, history)
            elif mode == 1:
                values = np.full(history, 1.5)
            elif mode == 2:
                values = rng.choice((-100.0, -50.0, 0.0, 50.0), history)
            elif mode == 3:
                values = rng.uniform(-1.0e20, 1.0e20, history)
            elif mode == 4:
                base = np.linspace(-1.0, 1.0, history, dtype=np.float64)
                jitter = rng.choice((-0.49, 0.49), history) / _QUANTIZATION_LEVELS
                values = base + jitter
            else:
                values = rng.standard_normal(history) * 7.0
            scores[row, :history] = np.asarray(values, dtype=np.float32)
        scores[row, history] = np.float32(1000.0)
    return np.ascontiguousarray(scores)


def exact_forced_tail_reference(
    scores: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    """Return exact score-descending/id-ascending forced-tail results."""

    output = np.full((lengths.size, TOP_K), -1, dtype=np.int32)
    for row, length_value in enumerate(lengths):
        length = int(length_value)
        if length <= TOP_K:
            output[row, :length] = np.arange(length, dtype=np.int32)
            continue
        history_scores = scores[row, : length - 1]
        order = np.argsort(-history_scores, kind="stable")
        output[row, : TOP_K - 1] = order[: TOP_K - 1]
        output[row, TOP_K - 1] = length - 1
    return output


def assert_quantized_topk_contract(
    scores: np.ndarray,
    lengths: np.ndarray,
    actual: np.ndarray,
) -> None:
    """Apply the original PR's 1.5-step gate and exact structural checks."""

    expected = exact_forced_tail_reference(scores, lengths)
    assert actual.shape == expected.shape
    for row, length_value in enumerate(lengths):
        length = int(length_value)
        got = actual[row].astype(np.int64)
        if length <= TOP_K:
            np.testing.assert_array_equal(got, expected[row])
            continue

        ranked = got[: TOP_K - 1]
        assert int(got[TOP_K - 1]) == length - 1
        assert np.all((ranked >= 0) & (ranked < length - 1))
        assert np.unique(ranked).size == TOP_K - 1

        history_scores = scores[row, : length - 1].astype(np.float64)
        selected_scores = scores[row, ranked].astype(np.float64)
        exact_ids = expected[row, : TOP_K - 1].astype(np.int64)
        exact_scores = scores[row, exact_ids].astype(np.float64)
        low = float(history_scores.min())
        high = float(history_scores.max())
        tolerance = 1.5 * (high - low) / _QUANTIZATION_LEVELS if high > low else 0.0
        threshold = float(exact_scores[TOP_K - 2])
        assert np.all(selected_scores[1:] <= selected_scores[:-1] + tolerance)
        assert np.all(selected_scores >= threshold - tolerance)
        must_select = exact_ids[exact_scores > threshold + tolerance]
        assert set(must_select.tolist()).issubset(set(ranked.tolist()))
