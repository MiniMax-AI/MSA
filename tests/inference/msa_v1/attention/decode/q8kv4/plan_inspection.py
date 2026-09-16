"""Read-only access used to verify the actual split work selected by the planner."""

from functools import cache
from pathlib import Path

import torch
import tvm_ffi
from torch.utils.cpp_extension import load

from inference.msa_v1.attention.decode import q8kv4


@cache
def _inspection_module():
    return load(
        name="minimax_msa_decode_plan_inspection",
        sources=[str(Path(__file__).with_name("_plan_inspection.cpp"))],
        extra_include_paths=[
            str(Path(q8kv4.__file__).parent / "csrc/api"),
            str(Path(tvm_ffi.__file__).parent / "include"),
        ],
        extra_cflags=["-std=c++20", "-O2"],
    )


def assert_split_count(wrapper, expected: int) -> None:
    if expected == 1:
        return
    counts = _inspection_module().split_counts(wrapper._plan_state.backend_plan)
    assert counts.numel() > 0
    assert bool(torch.all(counts == expected)), counts.unique().cpu().tolist()
