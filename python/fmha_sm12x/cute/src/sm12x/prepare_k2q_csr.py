# SPDX-License-Identifier: MIT

"""Sparse k2q CSR builder for SM120/SM121."""

from __future__ import annotations

from typing import Optional

import torch

from ._schedule import SparseAttentionSchedule, SPARSE_SCHEDULE_MODEL

_SUPPORTED_TOPK = (4, 8, 16, 32)
_SUPPORTED_BLK_KV = 128


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _validate_cu_seqlens(
    cu_seqlens: torch.Tensor,
    *,
    name: str,
    expected_total: int | None = None,
) -> int:
    if int(cu_seqlens.shape[0]) < 1:
        raise ValueError(f"{name} must have shape [B + 1]")
    if int(cu_seqlens[0].item()) != 0:
        raise ValueError(f"{name}[0] must be 0")
    diffs = cu_seqlens[1:] - cu_seqlens[:-1]
    if bool((diffs < 0).any().item()):
        raise ValueError(f"{name} must be monotonically non-decreasing")
    total = int(cu_seqlens[-1].item())
    if expected_total is not None and total != int(expected_total):
        raise ValueError(
            f"{name}[-1] must match expected total {int(expected_total)}, got {total}"
        )
    return total


def _row_bounds_from_cu_seqlens(cu_seqlens_k: torch.Tensor, blk_kv: int) -> tuple[int, int, int]:
    lengths = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    if int(lengths.numel()) == 0:
        return 0, 0, 0
    rows = torch.div(lengths + int(blk_kv) - 1, int(blk_kv), rounding_mode="floor")
    return int(rows.sum().item()), int(rows.max().item()), int(lengths.max().item())


class SparseK2qCsrBuilderSm12x:
    """Build the k2q CSR reverse index with the SM12x CUDA helper pipeline."""

    def __init__(self) -> None:
        self._run = None
        self._run_with_schedule = None

    def _ensure_loaded(self) -> None:
        if self._run is None:
            from .build_k2q_csr import (
                run_build_k2q_csr,
                run_build_k2q_csr_with_schedule,
            )

            self._run = run_build_k2q_csr
            self._run_with_schedule = run_build_k2q_csr_with_schedule

    def __call__(
        self,
        q2k_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        *,
        total_k: int,
        blk_kv: int = 128,
        max_seqlen_k: Optional[int] = None,
        max_seqlen_q: Optional[int] = None,
        total_rows: Optional[int] = None,
        qhead_per_kv: int = 1,
        return_schedule: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, SparseAttentionSchedule]:
        if blk_kv != _SUPPORTED_BLK_KV:
            raise ValueError(
                f"SparseK2qCsrBuilderSm12x only supports blk_kv == {_SUPPORTED_BLK_KV}, "
                f"got {blk_kv}"
            )
        if q2k_indices.dtype != torch.int32:
            raise TypeError(f"q2k_indices must be torch.int32, got {q2k_indices.dtype}")
        if q2k_indices.ndim != 3:
            raise ValueError(
                "q2k_indices must be rank-3 [head_kv, total_q, topK], "
                f"got shape {tuple(q2k_indices.shape)}"
            )
        if not q2k_indices.is_contiguous():
            raise ValueError("q2k_indices must be contiguous")
        if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
            raise TypeError("cu_seqlens_q and cu_seqlens_k must be torch.int32")
        if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must be rank-1")
        if cu_seqlens_q.shape != cu_seqlens_k.shape:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must share shape [B + 1]")
        if not (q2k_indices.is_cuda and cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda):
            raise ValueError("all inputs must be CUDA tensors")
        if q2k_indices.device != cu_seqlens_q.device or q2k_indices.device != cu_seqlens_k.device:
            raise ValueError("all inputs must share a device")
        if not cu_seqlens_q.is_contiguous() or not cu_seqlens_k.is_contiguous():
            raise ValueError("cu_seqlens_q and cu_seqlens_k must be contiguous")

        total_k = int(total_k)
        if total_k < 0:
            raise ValueError(f"total_k must be non-negative, got {total_k}")

        head_kv, total_q, topk = (int(v) for v in q2k_indices.shape)
        if topk not in _SUPPORTED_TOPK:
            raise ValueError(f"SparseK2qCsrBuilderSm12x only supports topK in {_SUPPORTED_TOPK}, got {topk}")

        _validate_cu_seqlens(cu_seqlens_q, name="cu_seqlens_q", expected_total=total_q)
        total_k_from_cu = _validate_cu_seqlens(
            cu_seqlens_k,
            name="cu_seqlens_k",
            expected_total=total_k,
        )
        total_rows_from_cu, max_kv_blocks, max_k_tokens_from_cu = _row_bounds_from_cu_seqlens(
            cu_seqlens_k,
            blk_kv,
        )
        if return_schedule and max_seqlen_k is None:
            raise ValueError("build_k2q_csr requires max_seqlen_k when return_schedule=True")
        if max_seqlen_k is not None and int(max_seqlen_k) < max_k_tokens_from_cu:
            raise ValueError("max_seqlen_k must cover every cu_seqlens_k segment")
        if max_seqlen_k is not None:
            max_kv_blocks = max(max_kv_blocks, _ceil_div(int(max_seqlen_k), blk_kv))
        if total_rows is not None and int(total_rows) != total_rows_from_cu:
            raise ValueError(
                "total_rows must match rows implied by cu_seqlens_k, "
                f"got {int(total_rows)} vs {total_rows_from_cu}"
            )
        total_rows = total_rows_from_cu
        total_k = total_k_from_cu
        nnz_upper_bound = total_q * topk
        qhead_per_kv = int(qhead_per_kv)
        if qhead_per_kv <= 0:
            raise ValueError(f"qhead_per_kv must be positive, got {qhead_per_kv}")
        if return_schedule:
            if max_seqlen_q is None:
                raise ValueError("build_k2q_csr requires max_seqlen_q when return_schedule=True")
            max_seqlen_q = int(max_seqlen_q)

        device = q2k_indices.device
        k2q_row_ptr = torch.empty((head_kv, total_rows + 1), dtype=torch.int32, device=device)
        k2q_q_indices = torch.empty((head_kv, nnz_upper_bound), dtype=torch.int32, device=device)
        schedule = None
        if return_schedule:
            target_q_per_cta = SPARSE_SCHEDULE_MODEL.balanced_target_q_per_cta(
                total_q=total_q,
                topk=topk,
                blk_kv=blk_kv,
                head_kv=head_kv,
                qhead_per_kv=qhead_per_kv,
                device=device,
            )
            scheduler_metadata_capacity = SPARSE_SCHEDULE_MODEL.flat_schedule_capacity(
                total_rows=total_rows,
                total_q=total_q,
                topk=topk,
                head_kv=head_kv,
                target_q_per_cta=target_q_per_cta,
            )
            schedule = SparseAttentionSchedule(
                enabled=True,
                scheduler_metadata=torch.empty((scheduler_metadata_capacity, 6), dtype=torch.int32, device=device),
                work_count=torch.empty((1,), dtype=torch.int32, device=device),
                qsplit_indices=torch.empty_like(k2q_q_indices),
                split_counts=torch.empty((total_q, head_kv), dtype=torch.int32, device=device),
                target_q_per_cta=target_q_per_cta,
            )

        if total_rows == 0 or total_q == 0 or head_kv == 0 or topk == 0:
            k2q_row_ptr.zero_()
            k2q_q_indices.fill_(-1)
            if schedule is not None:
                schedule.work_count.zero_()
                schedule.split_counts.zero_()
                return k2q_row_ptr, k2q_q_indices, schedule
            return k2q_row_ptr, k2q_q_indices

        self._ensure_loaded()
        with torch.cuda.nvtx.range("SparseK2qCsrSm12x_Pipeline"):
            if schedule is None:
                self._run(
                    q2k_indices,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    k2q_row_ptr,
                    k2q_q_indices,
                    topk,
                    blk_kv,
                    total_rows,
                    max_kv_blocks,
                )
            else:
                self._run_with_schedule(
                    q2k_indices,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    k2q_row_ptr,
                    k2q_q_indices,
                    schedule.scheduler_metadata,
                    schedule.work_count,
                    schedule.qsplit_indices,
                    schedule.split_counts,
                    topk,
                    blk_kv,
                    total_rows,
                    max_kv_blocks,
                    schedule.target_q_per_cta,
                    schedule.work_capacity,
                    max_seqlen_q,
                )
        if schedule is not None:
            return k2q_row_ptr, k2q_q_indices, schedule
        return k2q_row_ptr, k2q_q_indices
