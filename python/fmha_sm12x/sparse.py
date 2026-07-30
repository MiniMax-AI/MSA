# SPDX-License-Identifier: MIT

"""SM12x sparse-attention public surface."""

from __future__ import annotations

import math

import torch

from ._decode import SparseDecodePagedAttentionWrapper, sparse_decode_atten_func
from ._fp4 import fp4_indexer_block_scores
from ._nvfp4 import (
    Nvfp4QuantizedTensor,
    dequantize_nvfp4_128x4_to_bf16,
    nvfp4_global_scale_from_amax,
    nvfp4_scale_128x4_offset,
    quantize_bf16_to_nvfp4_128x4,
    quantize_kv_bf16_to_nvfp4_128x4,
    sparse_atten_nvfp4_kv_func,
    swizzle_nvfp4_scale_to_128x4,
)
from ._lse import run_lse
from .api import fmha_sm12x, fmha_sm12x_plan, sparse_topk_select

__all__ = [
    "SparseK2qCsrBuilderSm12x",
    "build_k2q_csr",
    "sparse_atten_func",
    "sparse_atten_nvfp4_kv_func",
    "sparse_decode_atten_func",
    "SparseDecodePagedAttentionWrapper",
    "fp4_indexer_block_scores",
    "Nvfp4QuantizedTensor",
    "quantize_bf16_to_nvfp4_128x4",
    "quantize_kv_bf16_to_nvfp4_128x4",
    "dequantize_nvfp4_128x4_to_bf16",
    "swizzle_nvfp4_scale_to_128x4",
    "nvfp4_global_scale_from_amax",
    "nvfp4_scale_128x4_offset",
    "sparse_topk_select",
]


def _rows_per_batch(cu_seqlens_k: torch.Tensor, block_size: int) -> list[int]:
    vals = cu_seqlens_k.to("cpu", dtype=torch.int64, non_blocking=False).tolist()
    return [(int(vals[i + 1]) - int(vals[i]) + block_size - 1) // block_size for i in range(len(vals) - 1)]


def _build_packed_row_map(rows: list[int]) -> tuple[list[list[int]], int]:
    max_rows = max(rows, default=0)
    row_map = [[-1 for _ in range(max_rows)] for _ in rows]
    row_linear = 0
    for block in range(max_rows):
        for batch, row_count in enumerate(rows):
            if block < row_count:
                row_map[batch][block] = row_linear
                row_linear += 1
    return row_map, row_linear


_OPTIMIZED_K2Q_TOPK = (4, 8, 16, 32)
_OPTIMIZED_K2Q_BLOCK_SIZE = 128


def _can_use_optimized_k2q(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    blk_kv: int,
    kwargs: dict,
) -> bool:
    if "total_k" not in kwargs or int(blk_kv) != _OPTIMIZED_K2Q_BLOCK_SIZE:
        return False
    if q2k_indices.dtype != torch.int32 or q2k_indices.ndim != 3:
        return False
    if int(q2k_indices.shape[2]) not in _OPTIMIZED_K2Q_TOPK:
        return False
    return bool(q2k_indices.is_cuda and cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda)


def _validate_cu_seqlens_pair(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> None:
    if cu_seqlens_q.dtype != torch.int32:
        raise TypeError(f"cu_seqlens_q must be torch.int32, got {cu_seqlens_q.dtype}")
    if cu_seqlens_k.dtype != torch.int32:
        raise TypeError(f"cu_seqlens_k must be torch.int32, got {cu_seqlens_k.dtype}")
    if cu_seqlens_q.ndim != 1:
        raise ValueError("cu_seqlens_q must be rank-1")
    if cu_seqlens_k.ndim != 1:
        raise ValueError("cu_seqlens_k must be rank-1")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must share shape [B + 1]")


def _compact_page_table(page_table: torch.Tensor | None, cu_seqlens_k: torch.Tensor, block_size: int) -> torch.Tensor | None:
    if page_table is None:
        return None
    pages: list[torch.Tensor] = []
    rows = _rows_per_batch(cu_seqlens_k, int(block_size))
    for batch, row_count in enumerate(rows):
        if row_count > 0:
            pages.append(page_table[batch, :row_count])
    if not pages:
        return torch.empty((0,), dtype=torch.int32, device=page_table.device)
    return torch.cat(pages).contiguous()


def build_k2q_csr(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    kv_block_size: int,
    *,
    total_k: int | None = None,
    return_schedule: bool = False,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference q2k -> k2q CSR builder for SM12x validation.

    This pure-Torch reference returns only ``(k2q_row_ptr, k2q_q_indices)``.
    The fused sparse-attention schedule (``return_schedule=True`` in the SM100
    surface) requires the optimized CUDA builder, so route through
    ``SparseK2qCsrBuilderSm12x`` for that path rather than failing with a
    silent 2-tuple that a 3-tuple caller would mis-unpack.
    """

    _ = total_k
    if return_schedule:
        raise ValueError(
            "return_schedule=True is not supported by the Torch reference "
            "build_k2q_csr; use SparseK2qCsrBuilderSm12x for the fused schedule."
        )
    if q2k_indices.dtype != torch.int32 or q2k_indices.ndim != 3:
        raise ValueError("q2k_indices must be int32 with shape [Hkv, total_q, topK]")
    _validate_cu_seqlens_pair(cu_seqlens_q, cu_seqlens_k)
    head_kv, total_q, topk = (int(v) for v in q2k_indices.shape)
    rows = _rows_per_batch(cu_seqlens_k, int(kv_block_size))
    row_map, total_rows = _build_packed_row_map(rows)
    row_ptr = torch.zeros((head_kv, total_rows + 1), dtype=torch.int32, device=q2k_indices.device)
    q_indices = torch.full((head_kv, total_q * topk), -1, dtype=torch.int32, device=q2k_indices.device)
    q_starts = cu_seqlens_q.to("cpu", dtype=torch.int64, non_blocking=False).tolist()
    buckets: list[list[list[int]]] = [[[] for _ in range(total_rows)] for _ in range(head_kv)]
    q2k_cpu = q2k_indices.to("cpu", non_blocking=False)
    for batch, row_count in enumerate(rows):
        for local_q in range(int(q_starts[batch + 1]) - int(q_starts[batch])):
            q_global = int(q_starts[batch]) + local_q
            for head in range(head_kv):
                for item in q2k_cpu[head, q_global].tolist():
                    block = int(item)
                    if block >= 0:
                        if block >= row_count:
                            raise ValueError(f"q2k_indices block index out of range for batch {batch}")
                        buckets[head][row_map[batch][block]].append(local_q)
    for head in range(head_kv):
        cursor = 0
        for row, entries in enumerate(buckets[head]):
            row_ptr[head, row] = cursor
            for value in sorted(entries):
                q_indices[head, cursor] = int(value)
                cursor += 1
        row_ptr[head, total_rows] = cursor
    return row_ptr, q_indices


class SparseK2qCsrBuilderSm12x:
    """CSR builder with an optimized SM12x CUDA path and reference fallback."""

    def __init__(self, *, use_optimized: bool = True) -> None:
        self._use_optimized = bool(use_optimized)
        self._optimized = None

    def _optimized_builder(self):
        if self._optimized is None:
            from .cute.src.sm12x.prepare_k2q_csr import (
                SparseK2qCsrBuilderSm12x as _OptimizedSparseK2qCsrBuilderSm12x,
            )

            self._optimized = _OptimizedSparseK2qCsrBuilderSm12x()
        return self._optimized

    def __call__(
        self,
        q2k_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        **kwargs,
    ):
        _validate_cu_seqlens_pair(cu_seqlens_q, cu_seqlens_k)
        blk_kv = int(kwargs.get("blk_kv", _OPTIMIZED_K2Q_BLOCK_SIZE))
        if self._use_optimized and _can_use_optimized_k2q(
            q2k_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            blk_kv,
            kwargs,
        ):
            return self._optimized_builder()(
                q2k_indices,
                cu_seqlens_q,
                cu_seqlens_k,
                **kwargs,
            )
        if kwargs.get("return_schedule", False):
            raise ValueError(
                "return_schedule=True requires the optimized CUDA SM12x CSR builder"
            )
        return build_k2q_csr(q2k_indices, cu_seqlens_q, cu_seqlens_k, blk_kv, **kwargs)


def _q2k_from_csr(k2q_row_ptr: torch.Tensor, k2q_q_indices: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, topk: int, blk_kv: int) -> torch.Tensor:
    head_kv = int(k2q_row_ptr.shape[0])
    total_q = int(cu_seqlens_q[-1].item())
    q2k = torch.full((head_kv, total_q, int(topk)), -1, dtype=torch.int32, device=k2q_row_ptr.device)
    rows = _rows_per_batch(cu_seqlens_k, int(blk_kv))
    row_map, total_rows = _build_packed_row_map(rows)
    if int(k2q_row_ptr.shape[1]) != total_rows + 1:
        raise ValueError("k2q_row_ptr row count does not match cu_seqlens_k")
    q_starts = cu_seqlens_q.to("cpu", dtype=torch.int64, non_blocking=False).tolist()
    for batch, row_count in enumerate(rows):
        qo_len = int(q_starts[batch + 1]) - int(q_starts[batch])
        for block in range(row_count):
            row = row_map[batch][block]
            for head in range(head_kv):
                begin = int(k2q_row_ptr[head, row].item())
                end = int(k2q_row_ptr[head, row + 1].item())
                if begin < 0 or end < begin or end > int(k2q_q_indices.shape[1]):
                    raise ValueError("k2q row pointers are out of range")
                for local_q_value in k2q_q_indices[head, begin:end].to("cpu", non_blocking=False).tolist():
                    local_q = int(local_q_value)
                    if local_q < 0 or local_q >= qo_len:
                        raise ValueError(f"k2q query index out of range for batch {batch} (q index)")
                    q_global = int(q_starts[batch]) + local_q
                    slot = int((q2k[head, q_global] >= 0).sum().item())
                    if slot < topk:
                        q2k[head, q_global, slot] = block
    return q2k


_FP8_E4M3 = torch.float8_e4m3fn


def _stage_attention_dtypes(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accept SM100's forward dtype combinations, staging FP8 to BF16.

    Like the SM100 path, q/k/v may all share a dtype (BF16/FP16/FP8 E4M3) or be
    a BF16 query with an FP8 E4M3 K/V cache.  FP8 operands are dequantized to
    BF16 (the SM100 FP8 path stages QK/PV in BF16), so the downstream Triton /
    Torch reference runs in BF16 and matches the dequantized-BF16 reference.
    """

    same = q.dtype == k.dtype == v.dtype
    fp8_kv_bf16_q = (
        q.dtype == torch.bfloat16 and k.dtype == _FP8_E4M3 and v.dtype == _FP8_E4M3
    )
    if not same and not fp8_kv_bf16_q:
        raise TypeError(
            "q, k, v must share a dtype, except a bf16 query with fp8_e4m3 K/V; "
            f"got q={q.dtype}, k={k.dtype}, v={v.dtype}"
        )

    def _deq(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.bfloat16) if t.dtype == _FP8_E4M3 else t

    return _deq(q), _deq(k), _deq(v)


def sparse_atten_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k2q_row_ptr: torch.Tensor, k2q_q_indices: torch.Tensor, topK: int, *, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, max_seqlen_q: int, max_seqlen_k: int, blk_kv: int = 128, causal: bool = False, softmax_scale: float | None = None, lse_temperature_scale: float = 1.0, return_temperature_lse: bool = False, partial_dtype: torch.dtype = torch.bfloat16, return_softmax_lse: bool = False, page_table: torch.Tensor | None = None, seqused_k: torch.Tensor | None = None, schedule: object | None = None, usable_SM_count: int = -1, qk_dtype: torch.dtype | None = None, pv_dtype: torch.dtype | None = None, **_kwargs):
    """Block-sparse varlen attention for SM12x.

    Uses the fused Triton kernel (:mod:`fmha_sm12x._triton_sparse`) for dense
    KV when Triton is importable, and the Torch reference otherwise (paged KV,
    or no Triton).  Mirrors the SM100 ``sparse_atten_func`` surface:
    ``schedule``, ``usable_SM_count``, ``partial_dtype``, ``qk_dtype`` and
    ``pv_dtype`` are accepted for API compatibility but do not affect this
    forward.  ``return_temperature_lse`` returns a third LSE computed with
    logits scaled by ``softmax_scale / lse_temperature_scale``.
    """

    _ = (max_seqlen_q, max_seqlen_k, seqused_k, schedule, usable_SM_count, partial_dtype, qk_dtype, pv_dtype)
    lse_temperature_scale = float(lse_temperature_scale)
    if not math.isfinite(lse_temperature_scale) or lse_temperature_scale <= 0.0:
        raise ValueError(
            f"lse_temperature_scale must be finite and > 0, got {lse_temperature_scale}"
        )
    if bool(return_temperature_lse) and not bool(return_softmax_lse):
        raise ValueError("return_temperature_lse=True requires return_softmax_lse=True")
    q, k, v = _stage_attention_dtypes(q, k, v)
    q2k = _q2k_from_csr(k2q_row_ptr, k2q_q_indices, cu_seqlens_q, cu_seqlens_k, int(topK), int(blk_kv))
    kv_heads = int(k.shape[1]) if k.ndim == 4 else int(k.shape[-2])
    block_indexes = q2k.permute(1, 0, 2).contiguous()
    resolved_scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(int(q.shape[-1]))

    if page_table is None and int(block_indexes.shape[1]) == kv_heads:
        from ._triton_sparse import triton_dense_supported, triton_sparse_atten_dense

        if triton_dense_supported(q, k, v, int(blk_kv)):
            out, lse, temperature_lse = triton_sparse_atten_dense(
                q, k, v, block_indexes,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                num_kv_heads=kv_heads, page_size=int(blk_kv), causal=bool(causal),
                sm_scale=resolved_scale, return_lse=bool(return_softmax_lse),
                lse_temperature_scale=lse_temperature_scale,
                return_temperature_lse=bool(return_temperature_lse),
                out_dtype=torch.bfloat16,
            )
            if not return_softmax_lse:
                return out
            if bool(return_temperature_lse):
                return out, lse, temperature_lse
            return out, lse

    qo_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    kv_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    plan = fmha_sm12x_plan(qo_lens, kv_lens, int(q.shape[1]), kv_heads, page_size=int(blk_kv), causal=bool(causal))
    kv_indices = _compact_page_table(page_table, cu_seqlens_k, int(blk_kv))
    out, _ = fmha_sm12x(q, k, v, plan, kv_indices=kv_indices, kv_block_indexes=block_indexes, sm_scale=softmax_scale)
    if not return_softmax_lse:
        return out
    lse = run_lse(
        q, k, v, plan, kv_indices=kv_indices, kv_block_indexes=block_indexes,
        sm_scale=resolved_scale,
    )
    if bool(return_temperature_lse):
        temperature_lse = run_lse(
            q, k, v, plan, kv_indices=kv_indices, kv_block_indexes=block_indexes,
            sm_scale=resolved_scale / lse_temperature_scale,
        )
        return out, lse, temperature_lse
    return out, lse
