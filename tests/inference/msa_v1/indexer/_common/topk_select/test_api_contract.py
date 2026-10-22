"""Internal API and tensor-contract tests for forced-tail TopK."""

from __future__ import annotations

import importlib
import inspect

import pytest
import torch

from inference.msa_v1.indexer._common import topk_select
from inference.msa_v1.indexer._common.topk_select.build import load_extension

pytestmark = pytest.mark.gpu


def _valid_inputs(
    num_rows: int = 5,
    max_cols: int = 33,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.randn(num_rows, max_cols, dtype=torch.float32, device="cuda")
    lengths = torch.full((num_rows,), max_cols, dtype=torch.int32, device="cuda")
    return scores, lengths


def test_topk_select_is_the_only_package_entrypoint() -> None:
    assert topk_select.__all__ == ["_topk_select"]
    assert tuple(inspect.signature(topk_select._topk_select).parameters) == (
        "scores",
        "lengths",
        "out",
        "compact_grid",
    )
    for forbidden in (
        "forward",
        "row_n",
        "kv_lengths",
        "seq_lens",
        "prepare",
        "plan",
        "run",
    ):
        assert not hasattr(topk_select, forbidden)


def test_out_none_and_preallocated_out() -> None:
    scores, lengths = _valid_inputs()
    allocated = topk_select._topk_select(scores, lengths)
    assert allocated.shape == (scores.shape[0], 16)
    assert allocated.dtype == torch.int32
    assert allocated.device == scores.device

    output = torch.full_like(allocated, -7)
    result = topk_select._topk_select(scores, lengths, out=output)
    torch.cuda.synchronize()
    assert result.data_ptr() == output.data_ptr()


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda scores, lengths: (scores.cpu(), lengths), "scores must be a CUDA"),
        (lambda scores, lengths: (scores.half(), lengths), "scores must have dtype"),
        (
            lambda scores, lengths: (scores.transpose(0, 1), lengths),
            r"stride\(1\) == 1",
        ),
        (lambda scores, lengths: (scores, lengths.long()), "lengths must have dtype"),
        (
            lambda scores, lengths: (scores, lengths[:-1]),
            r"lengths must have shape \[num_rows\]",
        ),
    ),
)
def test_rejects_invalid_inputs(mutate, message: str) -> None:
    scores, lengths = _valid_inputs()
    bad_scores, bad_lengths = mutate(scores, lengths)
    with pytest.raises(ValueError, match=message):
        topk_select._topk_select(bad_scores, bad_lengths)


def test_rejects_invalid_out() -> None:
    scores, lengths = _valid_inputs()
    with pytest.raises(ValueError, match=r"shape \[num_rows, 16\]"):
        topk_select._topk_select(
            scores,
            lengths,
            out=torch.empty(5, 15, dtype=torch.int32, device="cuda"),
        )
    with pytest.raises(ValueError, match="out must have dtype"):
        topk_select._topk_select(
            scores,
            lengths,
            out=torch.empty(5, 16, dtype=torch.int64, device="cuda"),
        )


def test_rejects_dimensions_outside_contract() -> None:
    lengths = torch.ones(1, dtype=torch.int32, device="cuda")
    output = torch.empty(1, 16, dtype=torch.int32, device="cuda")
    extension = load_extension()
    for columns in (0, 16385):
        scores = torch.empty(1, columns, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match="max_cols must be"):
            topk_select._topk_select(scores, lengths)
        with pytest.raises(RuntimeError, match="max_cols must be"):
            extension._run(scores, lengths, output, False)


@pytest.mark.parametrize(
    ("phase", "precision"),
    (
        ("decode", "bf16"),
        ("decode", "q8kv4"),
        ("decode", "q8kv8"),
        ("prefill", "bf16"),
        ("prefill", "q8kv8"),
    ),
)
def test_indexer_rejects_capacity_above_topk_limit(phase, precision) -> None:
    api = importlib.import_module(f"inference.msa_v1.indexer.{phase}.{precision}")
    name = (
        "BatchDecodeIndexerWithPagedKVCacheWrapper"
        if phase == "decode"
        else "BatchPrefillIndexerWithPagedKVCacheWrapper"
    )
    wrapper = getattr(api, name)()
    table = torch.zeros(1, 16385, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="max_pages|maximum page count|max_cols"):
        if phase == "decode":
            wrapper.plan(table, torch.tensor([8], dtype=torch.int32, device="cuda"))
        else:
            cumulative = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
            wrapper.plan(
                cumulative,
                cumulative,
                table,
                total_q=8,
                max_seqlen_q=8,
                max_seqlen_k=16385 * 128,
            )
