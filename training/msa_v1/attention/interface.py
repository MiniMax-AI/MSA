"""Sparse attention interface.

Current delivery scope:
    - head dimension is supported only for D=128

Public API:
    sparse_atten_func(...)
Internal autograd core:
    MinMaxSparseAttenCsrVarlen.apply(...)

Preprocessing (external, done once):
    q2k_indices [head_kv, total_q, topK]  ->  attention.prepare()
        -> k2q_row_ptr   [head_kv, total_rows + 1]  int32
        -> k2q_q_indices [head_kv, total_q * topK]  int32
"""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

from msa_v1._common.aot_cache import compile_or_load, save_aot, try_load_aot
from msa_v1._common.compile_utils import compile_with_timing
from msa_v1._common.cute_dsl_utils import to_cute_tensor as to_cute_tensor_kvouter
from msa_v1._common.tma_utils import (
    create_q_gather4_tma_desc,
)
from msa_v1.attention.bwd.atten_bwd import SparseAttentionBackwardSm100
from msa_v1.attention.bwd.postprocess import (
    SparseAttentionBackwardDkvPostprocessSm100,
    SparseAttentionBackwardPostprocessAtomicDqSm100,
    SparseAttentionBackwardPostprocessSm100,
)
from msa_v1.attention.bwd.preprocess import (
    SparseAttentionBackwardDkvSplitZeroSm100,
    SparseAttentionBackwardPreprocessSm100,
)
from msa_v1.attention.fwd.atten_fwd import SparseAttentionForwardSm100
from msa_v1.attention.fwd.combine import combine
from msa_v1.attention.prepare_scheduler import SparseAttentionSchedule

_compile_cache: dict = {}
_bwd_pre_compile_cache: dict = {}
_bwd_dkv_zero_compile_cache: dict = {}
_bwd_atten_compile_cache: dict = {}
_bwd_post_compile_cache: dict = {}
_bwd_dkv_post_compile_cache: dict = {}
_TEMPERATURE_LSE_FAST_PATH_ABS_TOL = 1e-12
_SUPPORTED_SPARSE_TOPK = (4, 8, 16, 32)
_SUPPORTED_FWD_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
_SUPPORTED_FWD_MMA_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
_SUPPORTED_SPARSE_ATTN_P_MODES = ("", "fp8")


def _normalize_sparse_attn_p_mode(sparse_attn_p_mode: str) -> str:
    if not isinstance(sparse_attn_p_mode, str):
        raise TypeError(
            "sparse_attn_p_mode must be a string, "
            f"got {type(sparse_attn_p_mode).__name__}"
        )
    mode = sparse_attn_p_mode.strip().lower()
    if mode not in _SUPPORTED_SPARSE_ATTN_P_MODES:
        raise ValueError(
            "sparse_attn_p_mode must be '' or 'fp8', "
            f"got {sparse_attn_p_mode!r}"
        )
    return mode


def _normalize_partial_dtype(partial_dtype: torch.dtype) -> torch.dtype:
    supported = {torch.bfloat16, torch.float16}
    if partial_dtype not in supported:
        raise TypeError(
            "partial_dtype must be torch.bfloat16 or torch.float16, "
            f"got {partial_dtype}"
        )
    return partial_dtype


def _normalize_forward_mma_dtype(dtype: Optional[torch.dtype], fallback: torch.dtype, name: str) -> torch.dtype:
    dtype = fallback if dtype is None else dtype
    if dtype not in _SUPPORTED_FWD_MMA_DTYPES:
        raise TypeError(
            f"{name} must be one of torch.bfloat16 / torch.float8_e4m3fn, got {dtype}"
        )
    return dtype


def _resolve_forward_mma_dtypes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qk_dtype: Optional[torch.dtype],
    pv_dtype: Optional[torch.dtype],
) -> tuple[torch.dtype, torch.dtype]:
    qk_dtype = _normalize_forward_mma_dtype(qk_dtype, q.dtype, "qk_dtype")
    if pv_dtype is None:
        # Preserve the historical fp8 KV-cache path: BF16 Q with FP8 K/V
        # stages both K and V as BF16 compute operands.
        if (
            q.dtype == torch.bfloat16
            and k.dtype == torch.float8_e4m3fn
            and v.dtype == torch.float8_e4m3fn
        ):
            pv_dtype = torch.bfloat16
        else:
            pv_dtype = v.dtype
    pv_dtype = _normalize_forward_mma_dtype(pv_dtype, pv_dtype, "pv_dtype")

    if q.dtype != qk_dtype:
        raise ValueError(
            "qk_dtype must match q storage dtype; Q fp8->bf16 staging is not supported"
        )
    if k.dtype != qk_dtype:
        if not (k.dtype == torch.float8_e4m3fn and qk_dtype == torch.bfloat16):
            raise ValueError(
                "unsupported K storage/qk_dtype combination; only fp8 K -> bf16 QK staging is supported"
            )
    if v.dtype != pv_dtype:
        if not (v.dtype == torch.float8_e4m3fn and pv_dtype == torch.bfloat16):
            raise ValueError(
                "unsupported V storage/pv_dtype combination; only fp8 V -> bf16 PV staging is supported"
            )
    return qk_dtype, pv_dtype


def _to_cute_tensor_bwd(t: torch.Tensor, assumed_align: int = 16):
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=True)
    return tensor.mark_layout_dynamic(leading_dim=t.ndim - 1)


def _torch_dtype_to_cute_dtype(dtype: torch.dtype):
    if dtype == torch.float32:
        return cutlass.Float32
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float8_e4m3fn:
        return cutlass.Float8E4M3FN
    if dtype == torch.int32:
        return cutlass.Int32
    if dtype == torch.int8:
        return cutlass.Int8
    raise TypeError(f"Unsupported tensor dtype for CuTe descriptor: {dtype}")


def _make_fake_compact_bwd_tensor(
    t: torch.Tensor,
    shape,
    *,
    assumed_align: int = 16,
):
    """Build a compact BWD signature with only semantic extents dynamic."""
    if len(shape) != t.ndim:
        raise ValueError(
            f"fake BWD shape rank mismatch: expected {t.ndim}, got {len(shape)}"
        )
    return cute.runtime.make_fake_compact_tensor(
        _torch_dtype_to_cute_dtype(t.dtype),
        shape,
        stride_order=tuple(range(t.ndim - 1, -1, -1)),
        assumed_align=assumed_align,
    )


def _to_cute_tensor_bwd_int64_shape(t: torch.Tensor, assumed_align: int = 16):
    dtype = _torch_dtype_to_cute_dtype(t.dtype)
    elem_bytes = max(1, (dtype.width + 7) // 8)
    stride_div = max(1, int(assumed_align) // elem_bytes)
    leading_dim = t.ndim - 1
    shape = tuple(cute.sym_int64() for _ in range(t.ndim))
    stride = tuple(
        1 if i == leading_dim else cute.sym_int64(divisibility=stride_div)
        for i in range(t.ndim)
    )
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=stride,
        assumed_align=assumed_align,
    )


def _to_cute_tensor_meta(t: torch.Tensor, assumed_align: int = 4):
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=True)
    return tensor.mark_layout_dynamic(leading_dim=0)


def _torch_dtype_to_cutlass_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float8_e4m3fn:
        return cutlass.Float8E4M3FN
    raise TypeError(
        f"Only torch.bfloat16, torch.float16, torch.float8_e4m3fn supported, got {dtype}"
    )


def _total_padded_workspace(total: int, cu_seqlens: torch.Tensor, tile: int) -> int:
    return ((int(total) + int(cu_seqlens.shape[0]) * int(tile) - 1) // int(tile)) * int(tile)


def _prepare_paged_kv_for_tma(k, v, blk_kv: int):
    page_size = int(k.shape[2])
    if page_size != blk_kv:
        raise ValueError(f"Sparse Page Attention requires page_size == blk_kv, got {page_size} vs {blk_kv}")
    return k, v


def _validate_cu_seqlens(
    cu_seqlens: torch.Tensor,
    *,
    name: str,
    device: torch.device,
) -> None:
    if cu_seqlens.device != device:
        raise ValueError(f"{name} must be on the same device as q")
    if cu_seqlens.dtype != torch.int32:
        raise TypeError(f"{name} must be torch.int32")
    if cu_seqlens.ndim != 1:
        raise ValueError(f"{name} must have shape [B + 1]")
    if cu_seqlens.shape[0] < 1:
        raise ValueError(f"{name} must have at least one element")
    if not cu_seqlens.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_fragment_indices(
    fragment_indices: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    if fragment_indices is None:
        return
    if fragment_indices.dtype != torch.int32:
        raise TypeError("fragment_indices must be torch.int32")
    if tuple(fragment_indices.shape) != (cu_seqlens_q.shape[0] - 1,):
        raise ValueError("fragment_indices must have shape [B]")
    if fragment_indices.device != reference.device:
        raise ValueError("fragment_indices must be on the same device as q")
    if not fragment_indices.is_contiguous():
        raise ValueError("fragment_indices must be contiguous")


def _validate_csr_varlen_bwd_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    topK: int,
    blk_kv: int,
) -> tuple[int, int, int]:
    if q.ndim != 3 or dout.ndim != 3 or out.ndim != 3:
        raise ValueError("CSR sparse backward requires q/dout/out to have shape [total_q, Hq, D]")
    if k.ndim != 3 or v.ndim != 3:
        raise ValueError("CSR sparse backward requires k/v to have shape [total_k, Hkv, D]")
    if q.shape != dout.shape or q.shape != out.shape:
        raise ValueError("q, dout, and out must have the same shape [total_q, Hq, D]")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v must have the same dtype")
    if q.dtype != torch.bfloat16:
        raise NotImplementedError(
            "sparse backward supports only BF16 Q/K/V; native FP8 Q/K/V are unsupported"
        )
    if q.dtype != dout.dtype or q.dtype != out.dtype:
        raise ValueError("q, dout, and out must have the same dtype")
    if q.device != k.device or q.device != v.device or q.device != dout.device or q.device != out.device:
        raise ValueError("q, k, v, dout, and out must be on the same device")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, and v must have the same head dimension")
    dim = q.shape[-1]
    if dim != 128:
        raise NotImplementedError(
            f"CSR sparse backward currently supports only D=128, got D={dim}"
        )
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape [total_k, Hkv, D]")
    if softmax_lse.ndim != 2 or softmax_lse.shape[:2] != q.shape[:2]:
        raise ValueError("softmax_lse must have shape [total_q, Hq]")
    if q.device != k2q_row_ptr.device or q.device != k2q_q_indices.device:
        raise ValueError("CSR metadata must be on the same device as q")
    if k2q_row_ptr.dtype != torch.int32 or k2q_q_indices.dtype != torch.int32:
        raise TypeError("k2q_row_ptr and k2q_q_indices must be torch.int32")
    if k2q_row_ptr.ndim != 2 or k2q_q_indices.ndim != 2:
        raise ValueError("k2q_row_ptr and k2q_q_indices must be rank-2")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise TypeError("cu_seqlens_q and cu_seqlens_k must be torch.int32")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be rank-1")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same shape [B + 1]")
    if q.device != cu_seqlens_q.device or q.device != cu_seqlens_k.device:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be on the same device as q")
    if blk_kv <= 0:
        raise ValueError(f"blk_kv must be > 0, got {blk_kv}")
    batch = int(cu_seqlens_q.shape[0] - 1)
    total_q = q.shape[0]
    head_q = q.shape[1]
    head_kv = k.shape[1]
    if head_q % head_kv != 0:
        raise ValueError("q.shape[1] must be divisible by k.shape[1]")
    if k2q_row_ptr.shape[0] != head_kv or k2q_q_indices.shape[0] != head_kv:
        raise ValueError("CSR metadata head dimension must match KV head count")
    if k2q_row_ptr.shape[1] < 1:
        raise ValueError("k2q_row_ptr must contain at least one row pointer column")
    if k2q_q_indices.shape[1] < total_q * topK:
        raise ValueError(
            f"k2q_q_indices.shape[1] ({k2q_q_indices.shape[1]}) must be >= total_q * topK ({total_q * topK})"
        )
    return batch, head_kv, int(q.shape[0])


def _validate_csr_varlen_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    topK: int,
    blk_kv: int,
    page_table: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_k: Optional[torch.Tensor],
) -> tuple[int, int]:
    if q.ndim != 3:
        raise ValueError("CSR sparse forward requires q to have shape [total_q, Hq, D]")
    if q.dtype not in _SUPPORTED_FWD_DTYPES:
        raise TypeError(
            "CSR sparse forward supports only torch.bfloat16 and "
            f"torch.float8_e4m3fn Q/K/V, got {q.dtype}"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, v must be on the same device")
    mixed_fp8_kv_bf16_q = (
        q.dtype == torch.bfloat16
        and k.dtype == torch.float8_e4m3fn
        and v.dtype == torch.float8_e4m3fn
    )
    if not mixed_fp8_kv_bf16_q and (q.dtype != k.dtype or q.dtype != v.dtype):
        raise ValueError(
            "q, k, v must have the same dtype, except q=bf16 with fp8_e4m3 K/V cache"
        )
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, v must have the same head dimension")
    dim = q.shape[-1]
    if dim != 128:
        raise NotImplementedError(
            f"CSR sparse forward currently supports only D=128, got D={dim}"
        )
    if page_table is None:
        if k.shape[-2] != v.shape[-2] or k.shape[-1] != v.shape[-1]:
            raise ValueError("k and v must have the same [Hkv, D] tail dimensions")
        head_kv = k.shape[-2]
    else:
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "Sparse Page Attention requires k and v to have shape "
                "[num_pages, Hkv, page_size, D]"
            )
        if k.shape[1] != v.shape[1] or k.shape[-1] != v.shape[-1]:
            raise ValueError(
                "Sparse Page Attention k and v must have the same Hkv and D"
            )
        head_kv = k.shape[1]
    if (
        q.device != k2q_row_ptr.device
        or q.device != k2q_q_indices.device
    ):
        raise ValueError("CSR metadata must be on the same device as q")
    if (
        k2q_row_ptr.dtype != torch.int32
        or k2q_q_indices.dtype != torch.int32
    ):
        raise TypeError("CSR metadata tensors must be torch.int32")
    if k2q_row_ptr.ndim != 2 or k2q_q_indices.ndim != 2:
        raise ValueError("k2q_row_ptr and k2q_q_indices must be rank-2")

    _validate_cu_seqlens(cu_seqlens_q, name="cu_seqlens_q", device=q.device)
    _validate_cu_seqlens(cu_seqlens_k, name="cu_seqlens_k", device=q.device)
    if cu_seqlens_k.shape != cu_seqlens_q.shape:
        raise ValueError("cu_seqlens_k must have shape [B + 1] matching cu_seqlens_q")
    batch = int(cu_seqlens_q.shape[0] - 1)
    total_q = q.shape[0]

    head_q = q.shape[1]
    if head_q % head_kv != 0:
        raise ValueError("q.shape[1] must be divisible by Hkv")
    qhead_per_kv = head_q // head_kv
    if qhead_per_kv not in (1, 2, 4, 8, 16):
        raise NotImplementedError(
            "CSR forward is currently supported only for qhead_per_kv in {1, 2, 4, 8, 16}"
        )
    if k2q_row_ptr.shape[0] != head_kv or k2q_q_indices.shape[0] != head_kv:
        raise ValueError("CSR metadata head dimension must match KV head count")
    if k2q_q_indices.shape[1] < total_q * topK:
        raise ValueError(
            f"k2q_q_indices.shape[1] ({k2q_q_indices.shape[1]}) must be >= total_q * topK ({total_q * topK})"
        )
    if k2q_row_ptr.shape[1] < 1:
        raise ValueError("k2q_row_ptr must contain at least one row pointer column")

    if page_table is None:
        if seqused_k is not None:
            raise ValueError("seqused_k is only supported together with page_table")
        total_k = k.shape[0]
        if k.ndim != 3 or v.ndim != 3:
            raise ValueError("Sparse Attention requires k and v to have shape [total_k, Hkv, D]")
        if k.shape != (total_k, head_kv, q.shape[-1]) or v.shape != (total_k, head_kv, q.shape[-1]):
            raise ValueError("Sparse Attention k and v must match [total_k, Hkv, D]")
    else:
        if page_table.device != q.device:
            raise ValueError("page_table must be on the same device as q")
        if page_table.dtype != torch.int32:
            raise TypeError("page_table must be torch.int32")
        if page_table.ndim != 2 or page_table.shape[0] != batch:
            raise ValueError("page_table must have shape [B, max_num_pages_per_seq]")
        if page_table.stride(-1) != 1:
            raise ValueError("page_table must be contiguous in the last dimension")
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "Sparse Page Attention requires k and v to have shape "
                "[num_pages, Hkv, page_size, D]"
            )
        if k.shape != v.shape:
            raise ValueError(f"k and v must have the same shape, got {k.shape} and {v.shape}")
        if k.shape[1] != head_kv or k.shape[3] != q.shape[-1]:
            raise ValueError(
                "Sparse Page Attention k and v must match "
                "[num_pages, Hkv, page_size, D]"
            )
        page_size = int(k.shape[2])
        if page_size != blk_kv:
            raise ValueError(
                f"Unsupported Sparse Page Attention page_size={page_size} for blk_kv={blk_kv}; "
                "require page_size == blk_kv"
            )
        if seqused_k is not None:
            if seqused_k.device != q.device:
                raise ValueError("seqused_k must be on the same device as q")
            if seqused_k.dtype != torch.int32:
                raise TypeError("seqused_k must be torch.int32")
            if seqused_k.shape != (batch,):
                raise ValueError("seqused_k must have shape [B]")
            if not seqused_k.is_contiguous():
                raise ValueError("seqused_k must be contiguous")
    if topK not in _SUPPORTED_SPARSE_TOPK:
        raise ValueError(
            f"CSR sparse forward supports topK in {_SUPPORTED_SPARSE_TOPK}, got {topK}"
        )
    return batch, head_kv



def _validate_schedule_common(
    schedule: SparseAttentionSchedule,
    *,
    device: torch.device,
) -> None:
    if schedule.scheduler_metadata is None:
        raise ValueError("schedule.scheduler_metadata is required")
    if schedule.work_count is None:
        raise ValueError("schedule.work_count is required")
    metadata = schedule.scheduler_metadata
    work_count = schedule.work_count
    if metadata.device != device or work_count.device != device:
        raise ValueError("schedule tensors must be on the same device as q")
    if metadata.dtype != torch.int32 or work_count.dtype != torch.int32:
        raise TypeError("schedule tensors must be torch.int32")
    if metadata.ndim != 2 or metadata.shape[1] != 6:
        raise ValueError("schedule.scheduler_metadata must have shape [capacity, 6]")
    if work_count.shape != (1,):
        raise ValueError("schedule.work_count must have shape [1]")
    if not metadata.is_contiguous() or not work_count.is_contiguous():
        raise ValueError("schedule tensors must be contiguous")


def _validate_fwd_schedule(
    schedule: SparseAttentionSchedule,
    *,
    q: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    head_kv: int,
) -> None:
    _validate_schedule_common(schedule, device=q.device)
    if schedule.qsplit_indices is None:
        raise ValueError("schedule.qsplit_indices is required for forward")
    if schedule.split_counts is None:
        raise ValueError("schedule.split_counts is required for forward")
    qsplit = schedule.qsplit_indices
    split_counts = schedule.split_counts
    if qsplit.device != q.device or split_counts.device != q.device:
        raise ValueError("forward schedule tensors must be on the same device as q")
    if qsplit.dtype != torch.int32 or split_counts.dtype != torch.int32:
        raise TypeError("schedule.qsplit_indices and schedule.split_counts must be torch.int32")
    if qsplit.shape != k2q_q_indices.shape:
        raise ValueError("schedule.qsplit_indices shape must match k2q_q_indices")
    total_q = q.shape[0]
    if split_counts.shape != (total_q, head_kv):
        raise ValueError(
            "schedule.split_counts must have shape "
            f"({total_q}, {head_kv}), got {tuple(split_counts.shape)}"
        )
    if not qsplit.is_contiguous() or not split_counts.is_contiguous():
        raise ValueError("schedule.qsplit_indices and schedule.split_counts must be contiguous")


def sparse_atten_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    topK: int,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    schedule: SparseAttentionSchedule,
    fragment_indices: Optional[torch.Tensor] = None,
    blk_kv: int = 128,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    lse_temperature_scale: float = 1.0,
    return_temperature_lse: bool = False,
    partial_dtype: torch.dtype = torch.bfloat16,
    return_softmax_lse: bool = False,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    qk_dtype: Optional[torch.dtype] = None,
    pv_dtype: Optional[torch.dtype] = None,
    sparse_attn_p_mode: str = "",
    deterministic: bool = False,
):
    """Public sparse attention wrapper.

    Supported mode:
      Sparse Attention forward and backward with CSR varlen metadata for
      qhead_per_kv in {1, 2, 4, 8, 16}.
      Sparse Page Attention remains forward-only for inference.
      When return_softmax_lse=True, LSE_out comes directly from the combine
      kernel. When return_temperature_lse=True, a temperature-scaled LSE_out
      is computed with qk logits scaled by softmax_scale / lse_temperature_scale
      and returned as an additional output.
      Fwd/bwd prepare scheduling uses a flat row-chunk worklist to balance sink rows.
      qk_dtype and pv_dtype select the compile-time QK/PV MMA operand dtypes.
      By default they follow Q/V storage dtype, except the legacy BF16-Q with
      FP8 K/V cache path keeps its historical BF16 compute operands.
      sparse_attn_p_mode="fp8" applies block-local E4M3 fake quantization to
      P for non-paged BF16 forward. E4M3 probability paths quantize P * 448
      and compensate the scale in normalization and backward reconstruction.
      BF16 training uses an attention-level
      identity STE: forward and dV use the quantized P, while dQ/dK use the
      smooth softmax Jacobian evaluated at the logical P. Backward reduces
      D = dO dot (P @ V) directly from block-local logical PV accumulators,
      without materializing a logical output.
      deterministic=True preserves the forward algorithm and selects ordered
      dQ/dK/dV FP32 accumulation for autograd backward.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    lse_temperature_scale = float(lse_temperature_scale)
    if not math.isfinite(lse_temperature_scale) or lse_temperature_scale <= 0.0:
        raise ValueError(
            f"lse_temperature_scale must be finite and > 0, got {lse_temperature_scale}"
        )
    return_temperature_lse = bool(return_temperature_lse)
    if return_temperature_lse and not return_softmax_lse:
        raise ValueError("return_temperature_lse=True requires return_softmax_lse=True")
    partial_dtype = _normalize_partial_dtype(partial_dtype)
    qk_dtype, pv_dtype = _resolve_forward_mma_dtypes(q, k, v, qk_dtype, pv_dtype)
    sparse_attn_p_mode = _normalize_sparse_attn_p_mode(sparse_attn_p_mode)
    if type(deterministic) is not bool:
        raise TypeError("deterministic must be a Python bool")

    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError(
            "sparse_atten_func requires CSR varlen metadata: pass cu_seqlens_q and cu_seqlens_k"
        )
    batch, head_kv = _validate_csr_varlen_inputs(
        q,
        k,
        v,
        k2q_row_ptr,
        k2q_q_indices,
        topK,
        blk_kv,
        page_table,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
    )
    _validate_fragment_indices(fragment_indices, cu_seqlens_q, q)
    if fragment_indices is not None and page_table is not None:
        raise ValueError("fragment_indices is supported only for non-paged packed KV")
    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_k = int(max_seqlen_k)

    return MinMaxSparseAttenCsrVarlen.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        k2q_row_ptr.contiguous(),
        k2q_q_indices.contiguous(),
        int(topK),
        int(blk_kv),
        bool(causal),
        float(softmax_scale),
        lse_temperature_scale,
        return_temperature_lse,
        partial_dtype,
        bool(return_softmax_lse),
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        None if fragment_indices is None else fragment_indices.contiguous(),
        None if page_table is None else page_table.contiguous(),
        None if seqused_k is None else seqused_k.contiguous(),
        schedule,
        int(batch),
        int(head_kv),
        int(max_seqlen_q),
        int(max_seqlen_k),
        qk_dtype,
        pv_dtype,
        sparse_attn_p_mode,
        deterministic,
    )


class MinMaxSparseAttenCsrVarlen(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k2q_row_ptr: torch.Tensor,
        k2q_q_indices: torch.Tensor,
        topK: int,
        blk_kv: int,
        causal: bool,
        softmax_scale: float,
        lse_temperature_scale: float,
        return_temperature_lse: bool,
        partial_dtype: torch.dtype,
        return_softmax_lse: bool,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        fragment_indices: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
        seqused_k: Optional[torch.Tensor],
        schedule: SparseAttentionSchedule,
        batch: int,
        head_kv: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        qk_dtype: torch.dtype,
        pv_dtype: torch.dtype,
        sparse_attn_p_mode: str,
        deterministic: bool,
    ):
        total_q, head_q, dim = q.shape
        if head_q % head_kv != 0:
            raise ValueError("q.shape[1] must be divisible by head_kv")
        temperature_lse_fast_path = (
            return_temperature_lse
            and math.isclose(
                float(lse_temperature_scale),
                1.0,
                rel_tol=0.0,
                abs_tol=_TEMPERATURE_LSE_FAST_PATH_ABS_TOL,
            )
        )
        kernel_return_temperature_lse = (
            return_temperature_lse and not temperature_lse_fast_path
        )
        requires_logical_dpsum = (
            sparse_attn_p_mode == "fp8"
            and page_table is None
            and q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and v.dtype == torch.bfloat16
            and qk_dtype == torch.bfloat16
            and pv_dtype == torch.bfloat16
            and any(ctx.needs_input_grad[:3])
        )
        use_raw_partial_stats = (
            q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and v.dtype == torch.bfloat16
            and partial_dtype == torch.bfloat16
            and qk_dtype == torch.bfloat16
            and pv_dtype == torch.bfloat16
            and dim == 128
            and blk_kv == 128
            and topK == 16
            and head_q == 64
            and head_kv == 4
            and causal
            and not kernel_return_temperature_lse
        )

        O_partial = torch.empty(
            topK, total_q, head_q, dim, dtype=partial_dtype, device=q.device
        )
        LSE_partial = torch.empty(
            (topK, total_q, head_q, 2)
            if use_raw_partial_stats
            else (topK, total_q, head_q),
            dtype=torch.float32,
            device=q.device,
        )
        LSE_temperature_partial = (
            torch.empty(topK, total_q, head_q, dtype=torch.float32, device=q.device)
            if kernel_return_temperature_lse
            else None
        )
        O_out = torch.empty(total_q, head_q, dim, dtype=torch.bfloat16, device=q.device)
        LSE_out = torch.empty(total_q, head_q, dtype=torch.float32, device=q.device)
        LSE_temperature_out = (
            torch.empty_like(LSE_out) if kernel_return_temperature_lse else None
        )
        _validate_fwd_schedule(
            schedule,
            q=q,
            k2q_q_indices=k2q_q_indices,
            head_kv=head_kv,
        )
        k2q_qsplit_indices = schedule.qsplit_indices
        split_counts = schedule.split_counts
        schedule = _call_sparse_forward_sm100_csr_varlen(
            q,
            k,
            v,
            k2q_row_ptr,
            k2q_q_indices,
            k2q_qsplit_indices,
            split_counts,
            cu_seqlens_q,
            cu_seqlens_k,
            fragment_indices,
            page_table,
            seqused_k,
            O_partial,
            LSE_partial,
            LSE_temperature_partial,
            softmax_scale,
            lse_temperature_scale,
            kernel_return_temperature_lse,
            blk_kv,
            head_kv,
            max_seqlen_q,
            causal=causal,
            schedule=schedule,
            qk_dtype=qk_dtype,
            pv_dtype=pv_dtype,
            sparse_attn_p_mode=sparse_attn_p_mode,
        )
        # Sparse Attention and Sparse Page Attention both use the varlen-Q
        # combine path; the kernel-written LSE_out is the final contract.
        combine(
            O_partial,
            LSE_partial,
            O_out,
            LSE_out,
            lse_temperature_partial=LSE_temperature_partial,
            lse_temperature_out=LSE_temperature_out,
            cu_seqlens=cu_seqlens_q,
            split_counts=split_counts,
            use_pdl=True,
            raw_partial_stats=use_raw_partial_stats,
        )
        if temperature_lse_fast_path:
            LSE_temperature_out = LSE_out

        ctx.is_paged = page_table is not None
        if ctx.is_paged:
            ctx.save_for_backward()
        else:
            tensors_to_save = [
                q,
                k,
                v,
                O_out,
                LSE_out,
                k2q_row_ptr,
                k2q_q_indices,
                cu_seqlens_q,
                cu_seqlens_k,
            ]
            if fragment_indices is not None:
                ctx.fragment_indices_offset = len(tensors_to_save)
                tensors_to_save.append(fragment_indices)
            if any(ctx.needs_input_grad[:3]) and any(
                tensor is None
                for tensor in (
                    schedule.dkv_owner_counts,
                    schedule.dkv_split_indices,
                    schedule.dkv_split_count,
                )
            ):
                raise ValueError("attention backward requires prepared dKV owner metadata")
            ctx.schedule_offset = len(tensors_to_save)
            tensors_to_save.extend(
                [
                    schedule.scheduler_metadata,
                    schedule.work_count,
                    schedule.dkv_owner_counts,
                    schedule.dkv_split_indices,
                    schedule.dkv_split_count,
                ]
            )
            if requires_logical_dpsum or deterministic:
                ctx.fwd_qsplit_offset = len(tensors_to_save)
                tensors_to_save.append(schedule.qsplit_indices)
            ctx.save_for_backward(*tensors_to_save)
            ctx.work_capacity = schedule.work_capacity
        ctx.has_fragment_indices = fragment_indices is not None
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.topK = topK
        ctx.blk_kv = blk_kv
        ctx.qk_dtype = qk_dtype
        ctx.pv_dtype = pv_dtype
        ctx.sparse_attn_p_mode = sparse_attn_p_mode
        ctx.deterministic = deterministic
        ctx.requires_logical_dpsum = requires_logical_dpsum
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k

        if return_softmax_lse:
            ctx.mark_non_differentiable(LSE_out)
            if return_temperature_lse:
                ctx.mark_non_differentiable(LSE_temperature_out)
                return O_out, LSE_out, LSE_temperature_out
            return O_out, LSE_out
        return O_out

    @staticmethod
    def backward(ctx, *grad_outputs):
        dout = grad_outputs[0]
        if dout is None:
            return (None,) * 27
        if ctx.is_paged:
            raise NotImplementedError("CSR paged sparse attention supports inference only; backward is not implemented")
        if (
            ctx.qk_dtype != ctx.saved_tensors[0].dtype
            or ctx.pv_dtype != ctx.saved_tensors[2].dtype
        ):
            raise NotImplementedError(
                "CSR sparse backward is not implemented for explicit mixed qk_dtype/pv_dtype forward modes"
            )

        saved = ctx.saved_tensors
        q, k, v, out, softmax_lse, k2q_row_ptr, k2q_q_indices, cu_seqlens_q, cu_seqlens_k = saved[:9]
        fragment_indices = (
            saved[ctx.fragment_indices_offset]
            if ctx.has_fragment_indices
            else None
        )
        schedule_offset = ctx.schedule_offset
        scheduler_metadata = saved[schedule_offset]
        work_count = saved[schedule_offset + 1]
        dkv_owner_counts = saved[schedule_offset + 2]
        dkv_split_indices = saved[schedule_offset + 3]
        dkv_split_count = saved[schedule_offset + 4]
        head_kv = int(k.shape[1])
        qhead_per_kv = int(q.shape[1] // head_kv)
        if qhead_per_kv != 16:
            raise NotImplementedError(
                "Sparse attention backward supports only qhead_per_kv=16"
            )
        k2q_qsplit_indices = (
            saved[ctx.fwd_qsplit_offset]
            if ctx.requires_logical_dpsum or ctx.deterministic
            else None
        )
        dq, dk, dv = _call_sparse_bwd_csr_varlen(
            q,
            k,
            v,
            dout.contiguous(),
            out,
            softmax_lse,
            ctx.softmax_scale,
            k2q_row_ptr,
            k2q_q_indices,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            fragment_indices=fragment_indices,
            topK=ctx.topK,
            blk_kv=ctx.blk_kv,
            causal=ctx.causal,
            use_prepare_scheduler=True,
            scheduler_metadata=scheduler_metadata,
            work_count=work_count,
            work_capacity=ctx.work_capacity,
            dkv_owner_counts=dkv_owner_counts,
            dkv_split_indices=dkv_split_indices,
            dkv_split_count=dkv_split_count,
            k2q_qsplit_indices=k2q_qsplit_indices,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            sparse_attn_p_mode=ctx.sparse_attn_p_mode,
            deterministic=ctx.deterministic,
        )
        return (dq, dk, dv, *([None] * 24))


def _prepare_deterministic_bwd_schedule(
    scheduler_metadata: torch.Tensor,
    work_count: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    fragment_indices: Optional[torch.Tensor],
    *,
    work_capacity: int,
    head_kv: int,
    total_q: int,
    padded_kv_blocks: int,
    max_seqlen_q: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Order v1 K-outer work and build deterministic writer tickets."""

    work_ids = torch.arange(
        work_capacity,
        dtype=torch.int64,
        device=scheduler_metadata.device,
    )
    valid = work_ids < work_count[0].to(torch.int64)
    schedule_storage = scheduler_metadata[:work_capacity]
    schedule_i64 = schedule_storage.to(torch.int64)
    batch_count = int(cu_seqlens_k.shape[0] - 1)
    batch_idx = schedule_i64[:, 4].clamp_(0, max(batch_count - 1, 0))
    physical_batch = batch_idx
    if fragment_indices is not None:
        physical_batch = fragment_indices.to(torch.int64).index_select(
            0, batch_idx
        )
    physical_offset = cu_seqlens_k.to(torch.int64).index_select(
        0, physical_batch
    )
    physical_block = (
        (physical_offset + physical_batch * 128) // 128
        + schedule_i64[:, 5]
    )
    group = schedule_i64[:, 0] * padded_kv_blocks + physical_block
    schedule_key = (
        (group * max(batch_count, 1) + batch_idx)
        * max(max_seqlen_q + 1, 1)
        + schedule_i64[:, 2]
    )
    invalid_key = torch.iinfo(torch.int64).max - work_capacity + work_ids
    schedule_key = torch.where(valid, schedule_key, invalid_key)
    order = torch.argsort(schedule_key, stable=True)
    ordered_schedule = schedule_storage.index_select(0, order).contiguous()

    ordered_valid = valid.index_select(0, order)
    ordered_group = group.index_select(0, order)
    positions = torch.arange(
        work_capacity,
        dtype=torch.int64,
        device=scheduler_metadata.device,
    )
    previous_group = torch.cat(
        (ordered_group[:1] - 1, ordered_group[:-1]),
        dim=0,
    )
    group_start = torch.where(
        ordered_valid & (ordered_group != previous_group),
        positions,
        torch.zeros_like(positions),
    )
    group_start = torch.cummax(group_start, dim=0).values
    dkv_writer_rank = torch.where(
        ordered_valid,
        positions - group_start,
        torch.full_like(positions, -1),
    ).to(torch.int32)

    dq_semaphore = torch.zeros(
        (head_kv, total_q, 2),
        dtype=torch.int32,
        device=scheduler_metadata.device,
    )
    dkv_semaphore = torch.zeros(
        (head_kv, padded_kv_blocks, 2, 2),
        dtype=torch.int32,
        device=scheduler_metadata.device,
    )
    return ordered_schedule, dkv_writer_rank, dq_semaphore, dkv_semaphore


def _call_sparse_bwd_csr_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: float,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    fragment_indices: Optional[torch.Tensor] = None,
    topK: int,
    blk_kv: int = 128,
    causal: bool = False,
    use_prepare_scheduler: bool = True,
    scheduler_metadata: Optional[torch.Tensor] = None,
    work_count: Optional[torch.Tensor] = None,
    work_capacity: int = 0,
    dkv_owner_counts: Optional[torch.Tensor] = None,
    dkv_split_indices: Optional[torch.Tensor] = None,
    dkv_split_count: Optional[torch.Tensor] = None,
    k2q_qsplit_indices: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int,
    sparse_attn_p_mode: str = "",
    q_fp8: Optional[torch.Tensor] = None,
    k_fp8: Optional[torch.Tensor] = None,
    deterministic: bool = False,
):
    sparse_attn_p_mode = _normalize_sparse_attn_p_mode(sparse_attn_p_mode)
    if type(deterministic) is not bool:
        raise TypeError("deterministic must be a Python bool")
    _validate_fragment_indices(fragment_indices, cu_seqlens_q, q)
    _, head_kv, _ = _validate_csr_varlen_bwd_inputs(
        q,
        k,
        v,
        dout,
        out,
        softmax_lse,
        k2q_row_ptr,
        k2q_q_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        topK,
        blk_kv,
    )
    total_q, head_q, head_dim = q.shape
    total_k = k.shape[0]
    qhead_per_kv = head_q // head_kv
    if qhead_per_kv != 16:
        raise NotImplementedError("sparse backward supports only qhead_per_kv=16")
    dtype = q.dtype
    use_logical_dpsum = sparse_attn_p_mode == "fp8"
    if use_logical_dpsum:
        if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v)):
            raise TypeError("probability FP8 backward requires BF16 QDQ Q/K/V")
        if k2q_qsplit_indices is None:
            raise ValueError(
                "k2q_qsplit_indices are required when sparse_attn_p_mode='fp8'"
            )
        if q_fp8 is None or k_fp8 is None:
            raise ValueError(
                "q_fp8 and k_fp8 are required when sparse_attn_p_mode='fp8'"
            )
        if (
            q_fp8.dtype != torch.float8_e4m3fn
            or k_fp8.dtype != torch.float8_e4m3fn
        ):
            raise TypeError("q_fp8 and k_fp8 must use E4M3")
        if q_fp8.shape != q.shape or k_fp8.shape != k.shape:
            raise ValueError("q_fp8 and k_fp8 must match the BF16 Q/K shapes")
        for name, payload in (("q_fp8", q_fp8), ("k_fp8", k_fp8)):
            if payload.device != q.device:
                raise ValueError(f"{name} must be on the same device as q")
            if not payload.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        logical_d_q = q_fp8
        logical_d_k = k_fp8
    out_dtype = dtype
    max_seqlen_k = int(max_seqlen_k)
    max_seqlen_q = int(max_seqlen_q)
    use_atomic_dqaccum = False
    m_block_size = 128
    n_block_size = 128
    total_q_padded = _total_padded_workspace(total_q, cu_seqlens_q, m_block_size)
    total_k_padded = _total_padded_workspace(total_k, cu_seqlens_k, n_block_size)
    use_prepare_scheduler = bool(use_prepare_scheduler)
    if (
        not use_prepare_scheduler
        or scheduler_metadata is None
        or work_count is None
        or dkv_owner_counts is None
        or dkv_split_indices is None
        or dkv_split_count is None
        or int(work_capacity) <= 0
        or k2q_row_ptr.shape[1] <= 1
    ):
        raise RuntimeError("sparse backward requires a non-empty prepared schedule")
    if scheduler_metadata.dtype != torch.int32:
        raise TypeError("scheduler_metadata must be torch.int32")
    if scheduler_metadata.ndim != 2 or scheduler_metadata.shape[1] != 6:
        raise ValueError("scheduler_metadata must have shape [work_capacity, 6]")
    if work_count.dtype != torch.int32 or work_count.shape != (1,):
        raise TypeError("work_count must be torch.int32 with shape [1]")
    if work_count.device != q.device:
        raise ValueError("work_count must be on the same device as q")
    if int(work_capacity) > int(scheduler_metadata.shape[0]):
        raise ValueError("work_capacity exceeds scheduler_metadata capacity")
    padded_kv_blocks = total_k_padded // n_block_size
    if dkv_owner_counts.dtype != torch.int32 or tuple(dkv_owner_counts.shape) != (
        head_kv,
        padded_kv_blocks,
    ):
        raise ValueError(
            "dkv_owner_counts must be int32 [head_kv, padded_kv_blocks]"
        )
    if dkv_split_indices.dtype != torch.int32 or dkv_split_indices.ndim != 1:
        raise ValueError("dkv_split_indices must be a rank-1 int32 tensor")
    if dkv_split_count.dtype != torch.int32 or dkv_split_count.shape != (1,):
        raise ValueError("dkv_split_count must be int32 with shape [1]")
    if any(
        tensor.device != q.device
        for tensor in (dkv_owner_counts, dkv_split_indices, dkv_split_count)
    ):
        raise ValueError("dKV owner metadata must be on the same device as q")
    dkv_writer_rank = None
    dq_semaphore = None
    dkv_semaphore = None
    if deterministic:
        if k2q_qsplit_indices is None:
            raise ValueError(
                "deterministic backward requires prepared qsplit metadata"
            )
        (
            scheduler_metadata,
            dkv_writer_rank,
            dq_semaphore,
            dkv_semaphore,
        ) = _prepare_deterministic_bwd_schedule(
            scheduler_metadata,
            work_count,
            cu_seqlens_k,
            fragment_indices,
            work_capacity=work_capacity,
            head_kv=head_kv,
            total_q=total_q,
            padded_kv_blocks=padded_kv_blocks,
            max_seqlen_q=max_seqlen_q,
        )
    mLSE_log2 = torch.empty(total_q_padded, head_q, dtype=torch.float32, device=q.device)
    mdPsum = torch.empty(total_q_padded, head_q, dtype=torch.float32, device=q.device)
    if use_atomic_dqaccum:
        mdQaccum = torch.empty(
            total_q, head_q, head_dim, dtype=torch.float32, device=q.device
        )
    else:
        mdQaccum = torch.empty(
            head_kv,
            total_q_padded * qhead_per_kv * head_dim,
            dtype=torch.float32,
            device=q.device,
        )
    mdKaccum = torch.empty(
        head_kv,
        total_k_padded * head_dim,
        dtype=torch.float32,
        device=q.device,
    )
    mdVaccum = torch.empty_like(mdKaccum)
    mdK = torch.empty(total_k, head_kv, head_dim, dtype=out_dtype, device=q.device)
    mdV = torch.empty(total_k, head_kv, head_dim, dtype=out_dtype, device=q.device)
    mdQ = torch.empty(total_q, head_q, head_dim, dtype=out_dtype, device=q.device)
    p_quant_scale_scratch = None
    if use_logical_dpsum:
        p_quant_scale_numel = 2 * topK * total_q * head_q
        p_quant_scale_bytes = p_quant_scale_numel * torch.float32.itemsize
        if mdQ.numel() * mdQ.element_size() < p_quant_scale_bytes:
            raise RuntimeError(
                "dQ output storage is too small for in-place QAT P-scale scratch"
            )
        p_quant_scale_scratch = mdQ.view(torch.float32).reshape(-1)[
            :p_quant_scale_numel
        ].view(2, topK, total_q, head_q)

    mQ_flat = q.view(total_q * head_q, head_dim)
    mdO_flat = dout.view(total_q * head_q, head_dim)
    compiled_pre = _get_sparse_bwd_preprocess(
        out,
        dout,
        softmax_lse,
        mLSE_log2,
        mdPsum,
        mdQaccum,
        m_block_size,
        cu_seqlens_q=cu_seqlens_q,
        use_atomic_dqaccum=use_atomic_dqaccum,
        compute_dpsum=not use_logical_dpsum,
    )
    compiled_dkv_zero = _get_sparse_bwd_dkv_split_zero(
        mdKaccum,
        mdVaccum,
        dkv_split_indices,
        dkv_split_count,
        head_dim,
        n_block_size,
    )
    compiled_bwd = _get_sparse_bwd_atten(
        q,
        k,
        v,
        dout,
        mLSE_log2,
        mdPsum,
        mdQaccum,
        mdKaccum,
        mdVaccum,
        mdK,
        mdV,
        dkv_owner_counts,
        dkv_writer_rank,
        dq_semaphore,
        dkv_semaphore,
        mQ_flat,
        mdO_flat,
        k2q_q_indices,
        k2q_row_ptr,
        k2q_qsplit_indices,
        p_quant_scale_scratch,
        qhead_per_kv,
        m_block_size,
        n_block_size,
        causal=causal,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        fragment_indices=fragment_indices,
        softmax_scale=softmax_scale,
        use_prepare_scheduler=use_prepare_scheduler,
        scheduler_metadata=scheduler_metadata,
        work_count=work_count,
        work_capacity=work_capacity,
        sparse_attn_p_mode=sparse_attn_p_mode,
        deterministic=deterministic,
    )
    compiled_post = _get_sparse_bwd_postprocess(
        mdQaccum,
        mdQ,
        out_dtype,
        head_dim,
        m_block_size,
        cu_seqlens_q=cu_seqlens_q,
    )
    compiled_dk_post = _get_sparse_bwd_dkv_postprocess(
        mdKaccum,
        mdK,
        dkv_owner_counts,
        out_dtype,
        head_dim,
        n_block_size,
        max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        fragment_indices=fragment_indices,
    )
    compiled_dv_post = _get_sparse_bwd_dkv_postprocess(
        mdVaccum,
        mdV,
        dkv_owner_counts,
        out_dtype,
        head_dim,
        n_block_size,
        max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        fragment_indices=fragment_indices,
    )

    with torch.cuda.nvtx.range("Bwd_SparseAttn_CsrVarlen_Native"):
        with torch.cuda.nvtx.range("Bwd_SparseAttn_Preprocess"):
            compiled_pre(
                out,
                dout,
                mdPsum,
                softmax_lse,
                mLSE_log2,
                mdQaccum,
                cu_seqlens_q,
            )
        if use_logical_dpsum:
            # The preprocess also initializes LSElog2 and dQaccum. Its Pq-based
            # dPsum is discarded here and replaced by the logical-P identity
            # STE scalar reduction below.
            dpsum_numel = topK * total_q_padded * head_q
            use_dqaccum_scratch = mdQaccum.numel() >= dpsum_numel
            if use_dqaccum_scratch:
                # Reuse the dQaccum prefix as disjoint per-split D storage.
                # The dPsum kernel writes only existing q-splits, so clear the
                # whole logical view before reducing absent splits as zeros.
                # Preprocess initializes the packed dQ regions, whose layout
                # does not cover every slot in this temporary logical view.
                # Clear the prefix again after reduction before dQ accumulation.
                dpsum_output = mdQaccum.view(-1)[:dpsum_numel].view(
                    topK,
                    total_q_padded,
                    head_q,
                )
                dpsum_output.zero_()
            else:
                mdPsum.zero_()
                dpsum_output = mdPsum
            dpsum_schedule = SparseAttentionSchedule(
                enabled=True,
                scheduler_metadata=scheduler_metadata,
                work_count=work_count,
                qsplit_indices=k2q_qsplit_indices,
            )
            with torch.cuda.nvtx.range("Bwd_SparseAttn_LogicalDPsum"):
                _call_sparse_forward_sm100_csr_varlen(
                    logical_d_q,
                    logical_d_k,
                    v,
                    k2q_row_ptr,
                    k2q_q_indices,
                    k2q_qsplit_indices,
                    None,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    fragment_indices,
                    None,
                    None,
                    dout,
                    softmax_lse,
                    dpsum_output,
                    softmax_scale,
                    1.0,
                    False,
                    blk_kv,
                    head_kv,
                    max_seqlen_q,
                    causal=causal,
                    schedule=dpsum_schedule,
                    qk_dtype=logical_d_q.dtype,
                    pv_dtype=v.dtype,
                    sparse_attn_p_mode="",
                    dpsum_only=True,
                    p_quant_scale_scratch=p_quant_scale_scratch,
                )
            if use_dqaccum_scratch:
                with torch.cuda.nvtx.range("Bwd_SparseAttn_ReduceLogicalDPsum"):
                    torch.sum(dpsum_output, dim=0, out=mdPsum)
                    dpsum_output.zero_()
        with torch.cuda.nvtx.range("Bwd_SparseAttn_DKV_SplitZero"):
            compiled_dkv_zero(
                mdKaccum,
                mdVaccum,
                dkv_split_indices,
                dkv_split_count,
            )
        with torch.cuda.nvtx.range("Bwd_SparseAttn_Attention"):
            bwd_common_args = (
                q,
                k,
                v,
                dout,
                mLSE_log2,
                mdPsum,
                mdQaccum,
                mdKaccum,
                mdVaccum,
                mdK,
                mdV,
                dkv_owner_counts,
                dkv_writer_rank,
                dq_semaphore,
                dkv_semaphore,
                softmax_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                fragment_indices,
                mQ_flat,
                mdO_flat,
                k2q_q_indices,
                k2q_row_ptr,
            )
            compiled_bwd(
                *bwd_common_args,
                k2q_qsplit_indices,
                p_quant_scale_scratch,
                scheduler_metadata,
                work_count,
                work_capacity,
            )
        with torch.cuda.nvtx.range("Bwd_SparseAttn_Postprocess"):
            if use_atomic_dqaccum:
                compiled_post(mdQaccum, mdQ, softmax_scale)
            else:
                compiled_post(
                    mdQaccum,
                    mdQ,
                    softmax_scale,
                    cu_seqlens_q,
                )
        with torch.cuda.nvtx.range("Bwd_SparseAttn_DKV_Postprocess"):
            compiled_dk_post(
                mdKaccum,
                mdK,
                dkv_owner_counts,
                softmax_scale,
                cu_seqlens_k,
                fragment_indices,
                max_seqlen_k,
            )
            compiled_dv_post(
                mdVaccum,
                mdV,
                dkv_owner_counts,
                1.0,
                cu_seqlens_k,
                fragment_indices,
                max_seqlen_k,
            )

    return mdQ, mdK, mdV


def _get_sparse_bwd_preprocess(
    out: torch.Tensor,
    dout: torch.Tensor,
    softmax_lse: torch.Tensor,
    mLSE_log2: torch.Tensor,
    mdPsum: torch.Tensor,
    mdQaccum: torch.Tensor,
    m_block_size: int,
    *,
    cu_seqlens_q: torch.Tensor,
    use_atomic_dqaccum: bool,
    compute_dpsum: bool,
):
    head_dim = out.shape[-1]
    key = (
        "sparse_backward_preprocess_sm100_csr_varlen",
        out.dtype,
        head_dim,
        m_block_size,
        bool(use_atomic_dqaccum),
        bool(compute_dpsum),
    )
    if key not in _bwd_pre_compile_cache:
        kernel = SparseAttentionBackwardPreprocessSm100(
            dtype=_torch_dtype_to_cutlass_dtype(out.dtype),
            head_dim=head_dim,
            tile_m=m_block_size,
            num_threads=256,
            use_atomic_dqaccum=use_atomic_dqaccum,
            compute_dpsum=compute_dpsum,
        )
        _bwd_pre_compile_cache[key] = compile_or_load(
            key,
            lambda: compile_with_timing(
                kernel,
                _to_cute_tensor_bwd(out),
                _to_cute_tensor_bwd(dout),
                _to_cute_tensor_bwd(mdPsum),
                _to_cute_tensor_bwd(softmax_lse),
                _to_cute_tensor_bwd(mLSE_log2),
                _to_cute_tensor_bwd_int64_shape(mdQaccum),
                _to_cute_tensor_meta(cu_seqlens_q),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            ),
            log_prefix="sparse_bwd_preprocess",
        )
    return _bwd_pre_compile_cache[key]


def _get_sparse_bwd_dkv_split_zero(
    mdKaccum: torch.Tensor,
    mdVaccum: torch.Tensor,
    dkv_split_indices: torch.Tensor,
    dkv_split_count: torch.Tensor,
    head_dim: int,
    n_block_size: int,
):
    key = (
        "sparse_backward_dkv_split_zero_sm100",
        head_dim,
        n_block_size,
    )
    if key not in _bwd_dkv_zero_compile_cache:
        kernel = SparseAttentionBackwardDkvSplitZeroSm100(
            head_dim=head_dim,
            tile_n=n_block_size,
            num_threads=256,
        )
        _bwd_dkv_zero_compile_cache[key] = compile_or_load(
            key,
            lambda: compile_with_timing(
                kernel,
                _to_cute_tensor_bwd(mdKaccum),
                _to_cute_tensor_bwd(mdVaccum),
                _to_cute_tensor_meta(dkv_split_indices),
                _to_cute_tensor_meta(dkv_split_count),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            ),
            log_prefix="sparse_bwd_dkv_zero",
        )
    return _bwd_dkv_zero_compile_cache[key]


def _get_sparse_bwd_atten(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    mLSE_log2: torch.Tensor,
    mdPsum: torch.Tensor,
    mdQaccum: torch.Tensor,
    mdKaccum: torch.Tensor,
    mdVaccum: torch.Tensor,
    mdK: torch.Tensor,
    mdV: torch.Tensor,
    dkv_owner_counts: torch.Tensor,
    dkv_writer_rank: Optional[torch.Tensor],
    dq_semaphore: Optional[torch.Tensor],
    dkv_semaphore: Optional[torch.Tensor],
    mQ_flat: torch.Tensor,
    mdO_flat: torch.Tensor,
    k2q_indices: torch.Tensor,
    k2q_counts: torch.Tensor,
    k2q_qsplit_indices: Optional[torch.Tensor],
    p_quant_scale_scratch: Optional[torch.Tensor],
    qhead_per_kv: int,
    m_block_size: int,
    n_block_size: int,
    *,
    causal: bool = False,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    fragment_indices: Optional[torch.Tensor],
    softmax_scale: float,
    use_prepare_scheduler: bool = True,
    scheduler_metadata: Optional[torch.Tensor] = None,
    work_count: Optional[torch.Tensor] = None,
    work_capacity: int = 0,
    sparse_attn_p_mode: str = "",
    deterministic: bool = False,
):
    head_dim = q.shape[-1]
    dtype = q.dtype
    if dtype != torch.bfloat16:
        raise NotImplementedError(
            "sparse backward supports only BF16 Q/K/V; native FP8 Q/K/V are unsupported"
        )
    sparse_attn_p_mode = _normalize_sparse_attn_p_mode(sparse_attn_p_mode)
    p_mode_fp8 = sparse_attn_p_mode == "fp8" and dtype == torch.bfloat16
    if k.shape[-1] != head_dim or v.shape[-1] != head_dim:
        raise ValueError("q, k, and v must have the same head dimension")
    if (
        not use_prepare_scheduler
        or scheduler_metadata is None
        or work_count is None
        or int(work_capacity) <= 0
    ):
        raise RuntimeError("sparse backward attention requires a prepared schedule")
    # Runtime tensor extents are dynamic in the CuTe signature; keep only
    # codegen-specializing attributes in the Python compile key.
    key = (
        "sparse_backward_sm100_csr_varlen",
        dtype,
        qhead_per_kv,
        head_dim,
        m_block_size,
        n_block_size,
        bool(causal),
        bool(use_prepare_scheduler),
        bool(fragment_indices is not None),
        bool(deterministic),
    )
    if p_mode_fp8:
        key += (
            "sparse_attn_p_mode",
            sparse_attn_p_mode,
            ("has_p_quant_scale", p_quant_scale_scratch is not None),
        )
    if key not in _bwd_atten_compile_cache:
        kernel_kwargs = dict(
            head_dim=head_dim,
            qhead_per_kvhead=qhead_per_kv,
            tile_m=m_block_size,
            tile_n=n_block_size,
            pack_gqa=True,
            is_causal=causal,
            use_prepare_scheduler=use_prepare_scheduler,
            deterministic=deterministic,
        )
        if p_mode_fp8:
            kernel_kwargs["sparse_attn_p_mode"] = True
        kernel = SparseAttentionBackwardSm100(**kernel_kwargs)

        # Keep the fixed attention ABI dimensions static and share symbols for
        # equal runtime extents. This mirrors the FA4/v2 compile interface and
        # avoids carrying redundant tensor shape/stride values around the
        # persistent scheduler backedge.
        total_q_sym = cute.sym_int64()
        total_k_sym = cute.sym_int64()
        total_q_padded_sym = cute.sym_int64()
        q_accum_flat_sym = cute.sym_int64()
        kv_accum_flat_sym = cute.sym_int64()
        padded_kv_blocks_sym = cute.sym_int64()
        packed_q_sym = cute.sym_int64()
        k2q_nnz_sym = cute.sym_int64()
        k2q_rows_sym = cute.sym_int64()
        batch_plus_one_sym = cute.sym_int64()
        batch_sym = cute.sym_int64()
        work_capacity_sym = cute.sym_int64()
        head_q = int(q.shape[1])
        head_kv = int(k.shape[1])

        fake_q = _make_fake_compact_bwd_tensor(
            q, (total_q_sym, head_q, head_dim)
        )
        fake_k = _make_fake_compact_bwd_tensor(
            k, (total_k_sym, head_kv, head_dim)
        )
        fake_v = _make_fake_compact_bwd_tensor(
            v, (total_k_sym, head_kv, head_dim)
        )
        fake_dout = _make_fake_compact_bwd_tensor(
            dout, (total_q_sym, head_q, head_dim)
        )
        fake_lse = _make_fake_compact_bwd_tensor(
            mLSE_log2, (total_q_padded_sym, head_q)
        )
        fake_dpsum = _make_fake_compact_bwd_tensor(
            mdPsum, (total_q_padded_sym, head_q)
        )
        if mdQaccum.ndim == 3:
            fake_dq_accum = _make_fake_compact_bwd_tensor(
                mdQaccum, (total_q_sym, head_q, head_dim)
            )
        else:
            fake_dq_accum = _make_fake_compact_bwd_tensor(
                mdQaccum, (head_kv, q_accum_flat_sym)
            )
        fake_dk_accum = _make_fake_compact_bwd_tensor(
            mdKaccum, (head_kv, kv_accum_flat_sym)
        )
        fake_dv_accum = _make_fake_compact_bwd_tensor(
            mdVaccum, (head_kv, kv_accum_flat_sym)
        )
        fake_dk = _make_fake_compact_bwd_tensor(
            mdK, (total_k_sym, head_kv, head_dim)
        )
        fake_dv = _make_fake_compact_bwd_tensor(
            mdV, (total_k_sym, head_kv, head_dim)
        )
        fake_owner_counts = _make_fake_compact_bwd_tensor(
            dkv_owner_counts,
            (head_kv, padded_kv_blocks_sym),
            assumed_align=4,
        )
        fake_dkv_writer_rank = (
            None
            if dkv_writer_rank is None
            else _make_fake_compact_bwd_tensor(
                dkv_writer_rank,
                (work_capacity_sym,),
                assumed_align=4,
            )
        )
        fake_dq_semaphore = (
            None
            if dq_semaphore is None
            else _make_fake_compact_bwd_tensor(
                dq_semaphore,
                (head_kv, total_q_sym, 2),
                assumed_align=4,
            )
        )
        fake_dkv_semaphore = (
            None
            if dkv_semaphore is None
            else _make_fake_compact_bwd_tensor(
                dkv_semaphore,
                (head_kv, padded_kv_blocks_sym, 2, 2),
                assumed_align=4,
            )
        )
        fake_cu_q = _make_fake_compact_bwd_tensor(
            cu_seqlens_q, (batch_plus_one_sym,), assumed_align=4
        )
        fake_cu_k = _make_fake_compact_bwd_tensor(
            cu_seqlens_k, (batch_plus_one_sym,), assumed_align=4
        )
        fake_fragments = (
            None
            if fragment_indices is None
            else _make_fake_compact_bwd_tensor(
                fragment_indices, (batch_sym,), assumed_align=4
            )
        )
        fake_q_flat = _make_fake_compact_bwd_tensor(
            mQ_flat, (packed_q_sym, head_dim)
        )
        fake_do_flat = _make_fake_compact_bwd_tensor(
            mdO_flat, (packed_q_sym, head_dim)
        )
        fake_k2q_indices = _make_fake_compact_bwd_tensor(
            k2q_indices, (head_kv, k2q_nnz_sym), assumed_align=4
        )
        fake_k2q_counts = _make_fake_compact_bwd_tensor(
            k2q_counts, (head_kv, k2q_rows_sym), assumed_align=4
        )
        compile_common_args = (
            fake_q,
            fake_k,
            fake_v,
            fake_dout,
            fake_lse,
            fake_dpsum,
            fake_dq_accum,
            fake_dk_accum,
            fake_dv_accum,
            fake_dk,
            fake_dv,
            fake_owner_counts,
            fake_dkv_writer_rank,
            fake_dq_semaphore,
            fake_dkv_semaphore,
            Float32(softmax_scale),
            fake_cu_q,
            fake_cu_k,
            fake_fragments,
            fake_q_flat,
            fake_do_flat,
            fake_k2q_indices,
            fake_k2q_counts,
        )
        fake_scheduler_metadata = (
            None
            if scheduler_metadata is None
            else _make_fake_compact_bwd_tensor(
                scheduler_metadata,
                (work_capacity_sym, 6),
                assumed_align=4,
            )
        )
        fake_work_count = (
            None
            if work_count is None
            else _make_fake_compact_bwd_tensor(
                work_count, (1,), assumed_align=4
            )
        )
        compile_direct_schedule_args = (
            fake_scheduler_metadata,
            fake_work_count,
            Int32(work_capacity),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        )
        compile_args = compile_common_args + (
            None
            if k2q_qsplit_indices is None
            else _make_fake_compact_bwd_tensor(
                k2q_qsplit_indices,
                tuple(cute.sym_int64() for _ in range(k2q_qsplit_indices.ndim)),
                assumed_align=4,
            ),
            None
            if p_quant_scale_scratch is None
            else _make_fake_compact_bwd_tensor(
                p_quant_scale_scratch,
                (
                    2,
                    int(p_quant_scale_scratch.shape[1]),
                    total_q_sym,
                    head_q,
                ),
                assumed_align=4,
            ),
        ) + compile_direct_schedule_args
        _bwd_atten_compile_cache[key] = compile_or_load(
            key,
            lambda: compile_with_timing(
                kernel,
                *compile_args,
                options="--enable-tvm-ffi",
            ),
            log_prefix="sparse_bwd",
        )
    return _bwd_atten_compile_cache[key]


def _get_sparse_bwd_dkv_postprocess(
    dkv_accum: torch.Tensor,
    dkv: torch.Tensor,
    dkv_owner_counts: torch.Tensor,
    dtype: torch.dtype,
    head_dim: int,
    n_block_size: int,
    max_seqlen_k: int,
    *,
    cu_seqlens_k: torch.Tensor,
    fragment_indices: Optional[torch.Tensor],
):
    key = (
        "sparse_backward_dkv_postprocess_sm100_csr_varlen",
        dtype,
        head_dim,
        n_block_size,
        bool(fragment_indices is not None),
    )
    if key not in _bwd_dkv_post_compile_cache:
        kernel = SparseAttentionBackwardDkvPostprocessSm100(
            dtype=_torch_dtype_to_cutlass_dtype(dtype),
            head_dim=head_dim,
            tile_n=n_block_size,
            num_threads=128,
        )
        _bwd_dkv_post_compile_cache[key] = compile_or_load(
            key,
            lambda: compile_with_timing(
                kernel,
                _to_cute_tensor_bwd(dkv_accum),
                _to_cute_tensor_bwd(dkv),
                _to_cute_tensor_bwd(dkv_owner_counts),
                Float32(1.0),
                _to_cute_tensor_meta(cu_seqlens_k),
                (
                    None
                    if fragment_indices is None
                    else _to_cute_tensor_meta(fragment_indices)
                ),
                Int32(max_seqlen_k),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            ),
            log_prefix="sparse_bwd_dkv_postprocess",
        )
    return _bwd_dkv_post_compile_cache[key]


def _get_sparse_bwd_postprocess(
    mdQaccum: torch.Tensor,
    mdQ: torch.Tensor,
    dtype: torch.dtype,
    head_dim: int,
    m_block_size: int,
    *,
    cu_seqlens_q: torch.Tensor,
):
    use_atomic_dqaccum = mdQaccum.ndim == 3
    key = (
        "sparse_backward_postprocess_sm100_csr_varlen",
        dtype,
        head_dim,
        m_block_size,
        bool(use_atomic_dqaccum),
    )
    if key not in _bwd_post_compile_cache:
        if use_atomic_dqaccum:
            kernel = SparseAttentionBackwardPostprocessAtomicDqSm100(
                dtype=_torch_dtype_to_cutlass_dtype(dtype),
                head_dim=head_dim,
                tile_m=m_block_size,
                num_threads=128,
            )
        else:
            kernel = SparseAttentionBackwardPostprocessSm100(
                dtype=_torch_dtype_to_cutlass_dtype(dtype),
                head_dim=head_dim,
                tile_m=m_block_size,
                num_threads=128,
            )
        if use_atomic_dqaccum:
            compile_args = (
                _to_cute_tensor_bwd_int64_shape(mdQaccum),
                _to_cute_tensor_bwd(mdQ),
                Float32(1.0),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            )
        else:
            compile_args = (
                _to_cute_tensor_bwd_int64_shape(mdQaccum),
                _to_cute_tensor_bwd(mdQ),
                Float32(1.0),
                _to_cute_tensor_meta(cu_seqlens_q),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            )
        _bwd_post_compile_cache[key] = compile_or_load(
            key,
            lambda: compile_with_timing(
                kernel,
                *compile_args,
                options="--enable-tvm-ffi",
            ),
            log_prefix="sparse_bwd_postprocess",
        )
    return _bwd_post_compile_cache[key]


def _call_sparse_forward_sm100_csr_varlen(
    q,
    k,
    v,
    k2q_row_ptr,
    k2q_q_indices,
    k2q_qsplit_indices,
    split_counts,
    cu_seqlens_q,
    cu_seqlens_k,
    fragment_indices,
    page_table,
    seqused_k,
    O_partial,
    LSE_partial,
    LSE_temperature_partial,
    softmax_scale,
    lse_temperature_scale,
    return_temperature_lse,
    blk_kv,
    head_kv,
    max_seqlen_q,
    *,
    schedule: SparseAttentionSchedule,
    causal=False,
    qk_dtype: torch.dtype,
    pv_dtype: torch.dtype,
    sparse_attn_p_mode: str = "",
    dpsum_only: bool = False,
    p_quant_scale_scratch: Optional[torch.Tensor] = None,
):
    """Compile and launch the SM100 sparse forward K1 kernel on CSR metadata."""
    head_dim = q.shape[-1]
    dtype = q.dtype
    qk_dtype = _normalize_forward_mma_dtype(qk_dtype, q.dtype, "qk_dtype")
    pv_dtype = _normalize_forward_mma_dtype(pv_dtype, v.dtype, "pv_dtype")
    partial_dtype = O_partial.dtype
    return_temperature_lse = bool(return_temperature_lse)
    dpsum_only = bool(dpsum_only)
    if (
        not dpsum_only
        and return_temperature_lse != (LSE_temperature_partial is not None)
    ):
        raise ValueError(
            "return_temperature_lse must match LSE_temperature_partial presence"
        )
    lse_temperature_scale = float(lse_temperature_scale)
    if not math.isfinite(lse_temperature_scale) or lse_temperature_scale <= 0.0:
        raise ValueError(
            f"lse_temperature_scale must be finite and > 0, got {lse_temperature_scale}"
        )
    lse_temperature_inv_scale = 1.0 / lse_temperature_scale
    n_block_size = int(blk_kv)
    head_q = q.shape[1]
    qhead_per_kv = head_q // head_kv
    paged_kv = page_table is not None
    sparse_attn_p_mode = _normalize_sparse_attn_p_mode(sparse_attn_p_mode)
    if dpsum_only:
        qk_dtype_supported = (
            (
                dtype == torch.bfloat16
                and k.dtype == torch.bfloat16
                and qk_dtype == torch.bfloat16
            )
            or (
                dtype == torch.float8_e4m3fn
                and k.dtype == torch.float8_e4m3fn
                and qk_dtype == torch.float8_e4m3fn
            )
        )
        if (
            paged_kv
            or sparse_attn_p_mode
            or not qk_dtype_supported
            or v.dtype != torch.bfloat16
            or pv_dtype != torch.bfloat16
            or partial_dtype != torch.bfloat16
            or LSE_partial.dtype != torch.float32
            or LSE_temperature_partial is None
            or LSE_temperature_partial.dtype != torch.float32
            or p_quant_scale_scratch is None
            or p_quant_scale_scratch.dtype != torch.float32
            or p_quant_scale_scratch.ndim != 4
        ):
            raise ValueError(
                "dpsum_only requires non-paged matching BF16 or E4M3 Q/K, "
                "BF16 V/dO, logical P, FP32 LSE/dPsum, and rank-4 FP32 "
                "P-scale scratch"
            )
    p_mode_fp8 = (
        sparse_attn_p_mode == "fp8"
        and dtype == torch.bfloat16
        and not paged_kv
    )
    raw_partial_stats = not dpsum_only and LSE_partial.ndim == 4
    if not dpsum_only:
        expected_partial_stats_shape = (
            (O_partial.shape[0], q.shape[0], head_q, 2)
            if raw_partial_stats
            else (O_partial.shape[0], q.shape[0], head_q)
        )
        if tuple(LSE_partial.shape) != tuple(expected_partial_stats_shape):
            raise ValueError(
                "LSE_partial/raw stats shape mismatch: expected "
                f"{expected_partial_stats_shape}, got {tuple(LSE_partial.shape)}"
            )
        if LSE_partial.dtype != torch.float32:
            raise TypeError("LSE_partial/raw stats must be torch.float32")
        if raw_partial_stats:
            raw_target_supported = (
                dtype == torch.bfloat16
                and k.dtype == torch.bfloat16
                and v.dtype == torch.bfloat16
                and qk_dtype == torch.bfloat16
                and pv_dtype == torch.bfloat16
                and partial_dtype == torch.bfloat16
                and head_dim == 128
                and n_block_size == 128
                and O_partial.shape[0] == 16
                and head_q == 64
                and head_kv == 4
                and qhead_per_kv == 16
                and bool(causal)
                and not return_temperature_lse
            )
            if not raw_target_supported:
                raise ValueError(
                    "Raw partial stats are restricted to the target BF16 "
                    "D128, block128, topK16, Hq64/Hkv4 causal varlen "
                    "specialization"
                )
    page_size = int(k.shape[2]) if paged_kv else None
    if paged_kv:
        k_kernel, v_kernel = _prepare_paged_kv_for_tma(k, v, n_block_size)
    else:
        k_kernel = k
        v_kernel = v
    O_partial_flat = O_partial.reshape(-1, head_dim).contiguous()
    Q_flat = q.reshape(-1, head_dim).contiguous()
    Q_gather4_desc = (
        create_q_gather4_tma_desc(
            Q_flat,
            box_x=128 if q.dtype == torch.float8_e4m3fn else 64,
        )
        if qhead_per_kv in (1, 2, 4)
        else None
    )
    _validate_schedule_common(schedule, device=q.device)
    use_prepare_scheduler = schedule.enabled
    scheduler_metadata = schedule.scheduler_metadata
    work_count = schedule.work_count
    work_capacity = schedule.work_capacity
    if (
        not use_prepare_scheduler
        or scheduler_metadata is None
        or work_count is None
        or work_capacity <= 0
    ):
        raise RuntimeError("sparse forward requires a non-empty prepared schedule")
    if dpsum_only:
        kernel_name = (
            "sparse_dpsum_split_cpasync_ca_sm100_csr_varlen"
            if LSE_temperature_partial.ndim == 3
            else "sparse_dpsum_atomic_cpasync_ca_sm100_csr_varlen"
        )
    else:
        kernel_name = "sparse_forward_sm100_csr_varlen"
    key = (
        kernel_name,
        head_dim,
        n_block_size,
        qhead_per_kv,
        dtype,
        k.dtype,
        v.dtype,
        qk_dtype,
        pv_dtype,
        partial_dtype,
        bool(causal),
        bool(paged_kv),
        bool(use_prepare_scheduler),
        page_size,
        bool(seqused_k is not None),
        bool(fragment_indices is not None),
        bool(return_temperature_lse),
        ("has_p_quant_scale", p_quant_scale_scratch is not None),
        bool(raw_partial_stats),
    )
    if p_mode_fp8:
        key += ("sparse_attn_p_mode", sparse_attn_p_mode)
    if key not in _compile_cache:
        loaded = try_load_aot(key)
        if loaded is not None:
            _compile_cache[key] = loaded
        else:
            kernel_kwargs = dict(
                head_dim=head_dim,
                qheadperkv=qhead_per_kv,
                n_block_size=n_block_size,
                paged_kv=paged_kv,
                page_size=page_size,
                has_seqused_k=seqused_k is not None,
                causal=bool(causal),
                use_prepare_scheduler=use_prepare_scheduler,
                qk_dtype=_torch_dtype_to_cutlass_dtype(qk_dtype),
                pv_dtype=_torch_dtype_to_cutlass_dtype(pv_dtype),
                raw_partial_stats=raw_partial_stats,
            )
            if p_mode_fp8:
                kernel_kwargs["sparse_attn_p_mode"] = True
            if dpsum_only:
                kernel_kwargs["dpsum_only"] = True
            kernel = SparseAttentionForwardSm100(**kernel_kwargs)
            compile_args = (
                to_cute_tensor_kvouter(k_kernel),
                to_cute_tensor_kvouter(v_kernel),
                to_cute_tensor_kvouter(k2q_q_indices),
                to_cute_tensor_kvouter(k2q_qsplit_indices),
                to_cute_tensor_kvouter(k2q_row_ptr),
                None if scheduler_metadata is None else to_cute_tensor_kvouter(scheduler_metadata),
                None if work_count is None else to_cute_tensor_kvouter(work_count),
                to_cute_tensor_kvouter(O_partial_flat),
                to_cute_tensor_kvouter(LSE_partial),
                None
                if LSE_temperature_partial is None
                else to_cute_tensor_kvouter(LSE_temperature_partial),
                None
                if p_quant_scale_scratch is None
                else to_cute_tensor_kvouter(p_quant_scale_scratch),
                to_cute_tensor_kvouter(Q_flat),
                None if Q_gather4_desc is None else to_cute_tensor_kvouter(Q_gather4_desc),
                None if page_table is None else to_cute_tensor_kvouter(page_table),
                None if seqused_k is None else to_cute_tensor_kvouter(seqused_k),
                to_cute_tensor_kvouter(cu_seqlens_q),
                to_cute_tensor_kvouter(cu_seqlens_k),
                None
                if fragment_indices is None
                else to_cute_tensor_kvouter(fragment_indices),
                Float32(softmax_scale),
                Float32(lse_temperature_inv_scale),
                Int32(head_kv),
                Int32(max_seqlen_q),
                Int32(work_capacity),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            )
            _compile_cache[key] = compile_with_timing(
                kernel,
                *compile_args,
                options="--enable-tvm-ffi",
            )
            save_aot(key, _compile_cache[key])

    with torch.cuda.nvtx.range("Fwd_SparseAttn_Sm100_CsrVarlen"):
        runtime_args = (
            k_kernel,
            v_kernel,
            k2q_q_indices,
            k2q_qsplit_indices,
            k2q_row_ptr,
            scheduler_metadata,
            work_count,
            O_partial_flat,
            LSE_partial,
            LSE_temperature_partial,
            p_quant_scale_scratch,
            Q_flat,
            Q_gather4_desc,
            page_table,
            seqused_k,
            cu_seqlens_q,
            cu_seqlens_k,
            fragment_indices,
            softmax_scale,
            lse_temperature_inv_scale,
            head_kv,
            max_seqlen_q,
            work_capacity,
        )
        _compile_cache[key](*runtime_args)
    return schedule
