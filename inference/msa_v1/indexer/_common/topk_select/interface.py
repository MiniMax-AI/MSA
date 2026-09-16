"""Internal PyTorch interface for SM100-family forced-tail TopK."""

from __future__ import annotations

import torch

from inference.msa_v1.indexer._common.topk_select.build import load_extension

_TOP_K = 16
_MAXIMUM_COLUMNS = 8192


def _is_capturing(device: torch.device) -> bool:
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def _check_inputs(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor | None,
) -> None:
    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a torch.Tensor")
    if not scores.is_cuda:
        raise ValueError("scores must be a CUDA tensor")
    if scores.dtype != torch.float32:
        raise ValueError("scores must have dtype torch.float32")
    if scores.ndim != 2 or scores.shape[0] <= 0:
        raise ValueError("scores must have shape [num_rows, max_cols]")
    if not 0 < scores.shape[1] <= _MAXIMUM_COLUMNS:
        raise ValueError(f"max_cols must be in [1, {_MAXIMUM_COLUMNS}]")
    if scores.stride(1) != 1:
        raise ValueError("scores must have stride(1) == 1")
    if scores.stride(0) < scores.shape[1]:
        raise ValueError("scores rows must not overlap")

    if not isinstance(lengths, torch.Tensor):
        raise TypeError("lengths must be a torch.Tensor")
    if not lengths.is_cuda or not lengths.is_contiguous():
        raise ValueError("lengths must be a contiguous CUDA tensor")
    if lengths.dtype != torch.int32:
        raise ValueError("lengths must have dtype torch.int32")
    if lengths.ndim != 1 or lengths.shape[0] != scores.shape[0]:
        raise ValueError("lengths must have shape [num_rows]")
    if lengths.device != scores.device:
        raise ValueError("lengths must be on the same device as scores")

    if out is None:
        return
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if not out.is_cuda or not out.is_contiguous():
        raise ValueError("out must be a contiguous CUDA tensor")
    if out.dtype != torch.int32:
        raise ValueError("out must have dtype torch.int32")
    if out.shape != (scores.shape[0], _TOP_K):
        raise ValueError("out must have shape [num_rows, 16]")
    if out.device != scores.device:
        raise ValueError("out must be on the same device as scores")


def _topk_select(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select 15 historical columns and one forced tail from each row.

    ``lengths[row]`` includes the forced tail. Values in ``lengths`` are
    trusted device metadata and are not copied to the host for validation.
    During CUDA Graph capture, callers must provide ``out``.
    """

    _check_inputs(scores, lengths, out)
    if out is None:
        if _is_capturing(scores.device):
            raise RuntimeError("CUDA Graph capture requires a preallocated out tensor")
        out = torch.empty(
            (scores.shape[0], _TOP_K),
            dtype=torch.int32,
            device=scores.device,
        )
    return load_extension()._run(scores, lengths, out)
