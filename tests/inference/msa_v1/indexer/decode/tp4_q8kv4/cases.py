"""Correctness case construction."""

from __future__ import annotations

import torch


def boundary_lengths(batch: int, max_pages: int, seed: int) -> torch.Tensor:
    """Generate lengths around page and eight-query causal boundaries."""

    candidates = [8, 9, 15, 120, 121, 127, 128]
    for page in range(1, max_pages + 1):
        boundary = page * 128
        candidates.extend(
            value
            for value in (
                boundary - 7,
                boundary - 1,
                boundary,
                boundary + 1,
                boundary + 7,
            )
            if 8 <= value <= max_pages * 128
        )
    values = [
        candidates[(index * 17 + seed * 13) % len(candidates)] for index in range(batch)
    ]
    return torch.tensor(values, dtype=torch.int32)
