# SPDX-License-Identifier: MIT

"""SM120/SM121 reference API for MiniMax Sparse Attention."""

from __future__ import annotations

from typing import Optional, Union

import torch

from ._topk import get_sparse_topk_module

from ._reference import Sm12xPlan, make_plan, run_plan

__all__ = ["Sm12xPlan", "fmha_sm12x_plan", "fmha_sm12x", "sparse_topk_select"]


def fmha_sm12x_plan(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int = -1,
    qo_offset: Optional[Union[int, torch.Tensor]] = None,
    num_kv_splits: int = -1,
    page_size: int = -1,
    output_maxscore: bool = False,
    kv_block_num: int = -1,
    usable_SM_count: int = -1,
    causal: bool = True,
    **_kwargs,
) -> Sm12xPlan:
    """Build a semantic SM12x reference plan.

    ``num_kv_splits`` and ``usable_SM_count`` are accepted for API compatibility
    but are intentionally ignored by this Torch reference backend.
    """

    _ = (num_kv_splits, usable_SM_count)
    return make_plan(
        qo_segment_lens,
        kv_segment_lens,
        num_qo_heads=int(num_qo_heads),
        num_kv_heads=int(num_kv_heads),
        qo_offset=qo_offset,
        page_size=int(page_size),
        output_maxscore=bool(output_maxscore),
        kv_block_num=int(kv_block_num),
        causal=bool(causal),
    )


def fmha_sm12x(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan_info: Sm12xPlan,
    kv_indices: Optional[torch.Tensor] = None,
    kv_block_indexes: Optional[torch.Tensor] = None,
    q_offset_override: Optional[Union[int, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    max_score: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Run the SM12x Torch reference attention path."""

    if q_offset_override is not None:
        plan_info = make_plan(
            plan_info.qo_segment_lens,
            plan_info.kv_segment_lens,
            num_qo_heads=plan_info.num_qo_heads,
            num_kv_heads=plan_info.num_kv_heads,
            qo_offset=q_offset_override,
            page_size=plan_info.page_size,
            output_maxscore=plan_info.output_maxscore,
            kv_block_num=plan_info.kv_block_num,
            causal=plan_info.causal,
        )
    return run_plan(
        q,
        k,
        v,
        plan_info,
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
        out=out,
        max_score=max_score,
        sm_scale=kwargs.get("sm_scale"),
        output_o=bool(kwargs.get("output_o", True)),
        output_maxscore=bool(kwargs.get("output_maxscore", False)),
    )


def sparse_topk_select(
    max_score: torch.Tensor,
    topk: int,
    num_valid_pages: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
) -> torch.Tensor:
    """SM12x-safe wrapper around the standalone CUDA top-k selector."""

    if max_score.dtype != torch.float32:
        raise TypeError(f"max_score must be float32, got {max_score.dtype}")
    if max_score.ndim != 3 or not max_score.is_contiguous():
        raise ValueError("max_score must be contiguous with shape [heads, max_k_tiles, total_q]")
    if int(topk) != 16:
        raise ValueError(f"topk must be 16, got {topk}")
    heads, max_k_tiles, total_q = (int(v) for v in max_score.shape)
    if output is None:
        output = torch.empty((total_q, heads, int(topk)), dtype=torch.int32, device=max_score.device)
    valid_pages = max_k_tiles if num_valid_pages is None else int(num_valid_pages)
    workspace = torch.empty((heads * max_k_tiles * total_q,), dtype=torch.int32, device=max_score.device)
    get_sparse_topk_module().sparse_topk_select(
        max_score, output, workspace, int(topk), int(valid_pages),
        int(force_begin_blocks), int(force_end_blocks), torch.cuda.current_stream().cuda_stream,
    )
    return output
