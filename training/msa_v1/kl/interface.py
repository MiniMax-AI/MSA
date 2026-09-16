"""PyTorch interface for the MSA v1 SM100 sparse KL backward kernel."""

from __future__ import annotations

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from msa_v1._common.aot_cache import compile_or_load
from msa_v1._common.compile_utils import compile_with_timing
from msa_v1.attention.metadata import AttentionMetadata
from msa_v1.kl.postprocess import (
    SparseKlDkiPostprocessSm100,
    SparseKlDkiSplitZeroSm100,
    SparseKlDqiPostprocessSm100,
)
from msa_v1.kl.sparse_kl import SparseKlLossBackwardSm100

TEACHER_HEADS = 64
INDEX_HEADS = 4
HEAD_DIM = 128
BLOCK_SIZE = 128
TOPK_CAPACITY = 16

_COMPILE_CACHE: dict[tuple, object] = {}
_POSTPROCESS_COMPILE_CACHE: dict[tuple, object] = {}


def _total_padded_workspace(total: int, cu_seqlens: torch.Tensor) -> int:
    batch_plus_one = int(cu_seqlens.shape[0])
    return ((int(total) + batch_plus_one * BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE


def _make_fake_tensor(dtype, shape, *, assumed_align: int = 16):
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=assumed_align,
    )


def _compile_main_kernel(
    device: torch.device,
    *,
    deterministic: bool,
):
    capability = torch.cuda.get_device_capability(device)
    key = (
        "msa_v1_sparse_kl_bwd_padded_dki_workspace_sm100",
        capability,
        cutlass.BFloat16,
        Float32,
        TEACHER_HEADS,
        INDEX_HEADS,
        HEAD_DIM,
        BLOCK_SIZE,
        TOPK_CAPACITY,
        deterministic,
    )
    compiled = _COMPILE_CACHE.get(key)
    if compiled is not None:
        return compiled

    total_q = cute.sym_int64()
    total_k = cute.sym_int64()
    total_k_padded = cute.sym_int64()
    padded_blocks = cute.sym_int64()
    padded_blocks_plus_one = cute.sym_int64()
    nnz = cute.sym_int64()
    work_items = cute.sym_int64()
    bf16 = cutlass.BFloat16
    op = SparseKlLossBackwardSm100(
        bf16,
        Float32,
        deterministic=deterministic,
    )
    fake_args = (
        _make_fake_tensor(bf16, (total_q, TEACHER_HEADS, HEAD_DIM)),
        _make_fake_tensor(bf16, (total_k, INDEX_HEADS, HEAD_DIM)),
        _make_fake_tensor(Float32, (total_q, TEACHER_HEADS)),
        _make_fake_tensor(bf16, (total_q, INDEX_HEADS, HEAD_DIM)),
        _make_fake_tensor(bf16, (total_k, 1, HEAD_DIM)),
        _make_fake_tensor(Float32, (INDEX_HEADS, total_q)),
        _make_fake_tensor(Int32, (INDEX_HEADS, nnz), assumed_align=4),
        _make_fake_tensor(Int32, (INDEX_HEADS, nnz), assumed_align=4),
        _make_fake_tensor(
            Int32,
            (INDEX_HEADS, padded_blocks_plus_one),
            assumed_align=4,
        ),
        _make_fake_tensor(Int32, (work_items, 4), assumed_align=4),
        _make_fake_tensor(Int32, (1,), assumed_align=4),
        _make_fake_tensor(Int32, (padded_blocks,), assumed_align=4),
        (
            _make_fake_tensor(Int32, (work_items,), assumed_align=4)
            if deterministic
            else None
        ),
        (
            _make_fake_tensor(
                Int32,
                (INDEX_HEADS, total_q, 4),
                assumed_align=4,
            )
            if deterministic
            else None
        ),
        (
            _make_fake_tensor(Int32, (padded_blocks,), assumed_align=4)
            if deterministic
            else None
        ),
        _make_fake_tensor(Float32, (total_q, INDEX_HEADS, HEAD_DIM)),
        _make_fake_tensor(Float32, (total_k_padded, 1, HEAD_DIM)),
        Float32(1.0),
        Float32(1.0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    compiled = compile_or_load(
        key,
        lambda: compile_with_timing(
            op,
            *fake_args,
            options="--enable-tvm-ffi",
        ),
        log_prefix="sparse_kl_bwd",
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


def _compile_dqi_postprocess(device: torch.device):
    capability = torch.cuda.get_device_capability(device)
    key = ("msa_v1_sparse_kl_dqi_postprocess_sm100", capability)
    compiled = _POSTPROCESS_COMPILE_CACHE.get(key)
    if compiled is not None:
        return compiled

    total_q = cute.sym_int64()
    op = SparseKlDqiPostprocessSm100()
    fake_args = (
        _make_fake_tensor(Float32, (total_q, INDEX_HEADS, HEAD_DIM)),
        _make_fake_tensor(cutlass.BFloat16, (total_q, INDEX_HEADS, HEAD_DIM)),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    compiled = compile_or_load(
        key,
        lambda: compile_with_timing(op, *fake_args, options="--enable-tvm-ffi"),
        log_prefix="sparse_kl_dqi_postprocess",
    )
    _POSTPROCESS_COMPILE_CACHE[key] = compiled
    return compiled


def _compile_dki_split_zero(device: torch.device):
    capability = torch.cuda.get_device_capability(device)
    key = ("msa_v1_sparse_kl_dki_split_zero_sm100", capability)
    compiled = _POSTPROCESS_COMPILE_CACHE.get(key)
    if compiled is not None:
        return compiled

    total_k_padded = cute.sym_int64()
    split_capacity = cute.sym_int64()
    op = SparseKlDkiSplitZeroSm100()
    fake_args = (
        _make_fake_tensor(Float32, (total_k_padded, 1, HEAD_DIM)),
        _make_fake_tensor(Int32, (split_capacity,), assumed_align=4),
        _make_fake_tensor(Int32, (1,), assumed_align=4),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    compiled = compile_or_load(
        key,
        lambda: compile_with_timing(op, *fake_args, options="--enable-tvm-ffi"),
        log_prefix="sparse_kl_dki_zero",
    )
    _POSTPROCESS_COMPILE_CACHE[key] = compiled
    return compiled


def _compile_dki_postprocess(device: torch.device, *, has_fragments: bool):
    capability = torch.cuda.get_device_capability(device)
    key = ("msa_v1_sparse_kl_dki_owner_postprocess_sm100", capability, has_fragments)
    compiled = _POSTPROCESS_COMPILE_CACHE.get(key)
    if compiled is not None:
        return compiled

    total_k = cute.sym_int64()
    total_k_padded = cute.sym_int64()
    padded_blocks = cute.sym_int64()
    batch = cute.sym_int64()
    batch_plus_one = cute.sym_int64()
    op = SparseKlDkiPostprocessSm100()
    fake_args = (
        _make_fake_tensor(Float32, (total_k_padded, 1, HEAD_DIM)),
        _make_fake_tensor(cutlass.BFloat16, (total_k, 1, HEAD_DIM)),
        _make_fake_tensor(Int32, (padded_blocks,), assumed_align=4),
        Float32(1.0),
        _make_fake_tensor(Int32, (batch_plus_one,), assumed_align=4),
        (
            _make_fake_tensor(Int32, (batch,), assumed_align=4)
            if has_fragments
            else None
        ),
        Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    compiled = compile_or_load(
        key,
        lambda: compile_with_timing(op, *fake_args, options="--enable-tvm-ffi"),
        log_prefix="sparse_kl_dki_postprocess",
    )
    _POSTPROCESS_COMPILE_CACHE[key] = compiled
    return compiled


def _validate_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}")
    if tensor.device != device or not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor on {device}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    teacher_lse: torch.Tensor,
    qi: torch.Tensor,
    ki: torch.Tensor,
    indexer_lse: torch.Tensor,
    metadata: AttentionMetadata,
    dqi_accum: torch.Tensor,
    dki_accum: torch.Tensor,
    dki: torch.Tensor,
) -> None:
    if not isinstance(metadata, AttentionMetadata):
        raise TypeError("metadata must be msa_v1.attention.AttentionMetadata")
    if not torch.cuda.is_available() or not q.is_cuda:
        raise RuntimeError("MSA v1 KL backward requires CUDA tensors")
    capability = torch.cuda.get_device_capability(q.device)
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError(
            "MSA v1 KL backward supports only SM100/SM103, "
            f"got capability {capability}"
        )
    total_q = int(q.shape[0])
    total_k = int(k.shape[0])
    total_k_padded = _total_padded_workspace(total_k, metadata.cu_seqlens_k)
    device = q.device
    specs = (
        (q, "q", (total_q, TEACHER_HEADS, HEAD_DIM), torch.bfloat16),
        (k, "k", (total_k, INDEX_HEADS, HEAD_DIM), torch.bfloat16),
        (
            teacher_lse,
            "teacher_lse",
            (total_q, TEACHER_HEADS),
            torch.float32,
        ),
        (qi, "qi", (total_q, INDEX_HEADS, HEAD_DIM), torch.bfloat16),
        (ki, "ki", (total_k, 1, HEAD_DIM), torch.bfloat16),
        (
            indexer_lse,
            "indexer_lse",
            (INDEX_HEADS, total_q),
            torch.float32,
        ),
        (
            dqi_accum,
            "dqi_accum",
            (total_q, INDEX_HEADS, HEAD_DIM),
            torch.float32,
        ),
        (
            dki_accum,
            "dki_accum",
            (total_k_padded, 1, HEAD_DIM),
            torch.float32,
        ),
        (dki, "dki", (total_k, 1, HEAD_DIM), torch.bfloat16),
    )
    for tensor, name, shape, dtype in specs:
        _validate_tensor(
            tensor,
            name=name,
            shape=shape,
            dtype=dtype,
            device=device,
        )

    meta_specs = (
        (
            metadata.topk_indices,
            "metadata.topk_indices",
            (INDEX_HEADS, total_q, TOPK_CAPACITY),
        ),
        (
            metadata.cu_seqlens_q,
            "metadata.cu_seqlens_q",
            (metadata.cu_seqlens_q.numel(),),
        ),
        (
            metadata.cu_seqlens_k,
            "metadata.cu_seqlens_k",
            (metadata.cu_seqlens_q.numel(),),
        ),
        (
            metadata.k2q_row_ptr,
            "metadata.k2q_row_ptr",
            (INDEX_HEADS, metadata.total_rows + 1),
        ),
        (
            metadata.k2q_q_indices,
            "metadata.k2q_q_indices",
            (INDEX_HEADS, total_q * TOPK_CAPACITY),
        ),
    )
    for tensor, name, shape in meta_specs:
        _validate_tensor(
            tensor,
            name=name,
            shape=shape,
            dtype=torch.int32,
            device=device,
        )
    if metadata.total_k != total_k:
        raise ValueError(
            f"metadata.total_k ({metadata.total_k}) must equal k.shape[0] ({total_k})"
        )
    if metadata.cu_seqlens_q.numel() < 2:
        raise ValueError("metadata must describe at least one packed sequence")
    kl_schedule = metadata.kl_schedule
    padded_blocks = total_k_padded // BLOCK_SIZE
    if any(
        tensor is None
        for tensor in (
            kl_schedule.physical_row_ptr,
            kl_schedule.physical_q_indices,
            kl_schedule.physical_valid_rows,
            kl_schedule.dki_owner_counts,
            kl_schedule.dki_split_indices,
            kl_schedule.dki_split_count,
        )
    ):
        raise ValueError("KL schedule must include physical ownership metadata")
    for tensor, name, shape in (
        (
            kl_schedule.physical_row_ptr,
            "physical_row_ptr",
            (INDEX_HEADS, padded_blocks + 1),
        ),
        (
            kl_schedule.physical_q_indices,
            "physical_q_indices",
            (INDEX_HEADS, total_q * TOPK_CAPACITY),
        ),
        (
            kl_schedule.physical_valid_rows,
            "physical_valid_rows",
            (INDEX_HEADS, total_q * TOPK_CAPACITY),
        ),
        (kl_schedule.dki_owner_counts, "dki_owner_counts", (padded_blocks,)),
        (kl_schedule.dki_split_indices, "dki_split_indices", (padded_blocks,)),
        (kl_schedule.dki_split_count, "dki_split_count", (1,)),
    ):
        _validate_tensor(
            tensor,
            name=f"metadata.kl_schedule.{name}",
            shape=shape,
            dtype=torch.int32,
            device=device,
        )


def _validate_scalar(value: float, *, name: str, positive: bool) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _resolve_scales(
    total_q: int,
    softmax_scale: Optional[float],
    indexer_softmax_scale: Optional[float],
    loss_coeff: float,
) -> tuple[float, float, float]:
    softmax_scale = _validate_scalar(
        HEAD_DIM**-0.5 if softmax_scale is None else softmax_scale,
        name="softmax_scale",
        positive=True,
    )
    indexer_softmax_scale = _validate_scalar(
        HEAD_DIM**-0.5
        if indexer_softmax_scale is None
        else indexer_softmax_scale,
        name="indexer_softmax_scale",
        positive=True,
    )
    loss_coeff = _validate_scalar(loss_coeff, name="loss_coeff", positive=False)
    grad_scale = indexer_softmax_scale * loss_coeff / INDEX_HEADS / max(total_q, 1)
    return softmax_scale, indexer_softmax_scale, grad_scale


def _prepare_deterministic_backward(
    worklist: torch.Tensor,
    work_count: torch.Tensor,
    *,
    total_q: int,
    padded_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Order KL work and build device-side writer tickets."""

    work_capacity = int(worklist.shape[0])
    work_ids = torch.arange(
        work_capacity,
        dtype=torch.int64,
        device=worklist.device,
    )
    valid = work_ids < work_count[0].to(torch.int64)
    worklist_i64 = worklist.to(torch.int64)
    physical_block = worklist_i64[:, 0].clamp(
        0,
        max(padded_blocks - 1, 0),
    )
    macro_begin = worklist_i64[:, 2].clamp_min(0)
    schedule_key = (physical_block << 32) + macro_begin
    invalid_key = torch.iinfo(torch.int64).max - work_capacity + work_ids
    schedule_key = torch.where(valid, schedule_key, invalid_key)
    order = torch.argsort(schedule_key, stable=True)
    ordered_worklist = worklist.index_select(0, order).contiguous()

    ordered_valid = valid.index_select(0, order)
    ordered_block = physical_block.index_select(0, order)
    positions = torch.arange(
        work_capacity,
        dtype=torch.int64,
        device=worklist.device,
    )
    previous_block = torch.cat(
        (ordered_block[:1] - 1, ordered_block[:-1]),
        dim=0,
    )
    group_start = torch.where(
        ordered_valid & (ordered_block != previous_block),
        positions,
        torch.zeros_like(positions),
    )
    group_start = torch.cummax(group_start, dim=0).values
    dki_writer_rank = torch.where(
        ordered_valid,
        positions - group_start,
        torch.full_like(positions, -1),
    ).to(torch.int32)
    dqi_semaphore = torch.zeros(
        (INDEX_HEADS, total_q, 4),
        dtype=torch.int32,
        device=worklist.device,
    )
    dki_semaphore = torch.zeros(
        (padded_blocks,),
        dtype=torch.int32,
        device=worklist.device,
    )
    return ordered_worklist, dki_writer_rank, dqi_semaphore, dki_semaphore


def _launch_main_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    teacher_lse: torch.Tensor,
    qi: torch.Tensor,
    ki: torch.Tensor,
    indexer_lse: torch.Tensor,
    metadata: AttentionMetadata,
    dqi_accum: torch.Tensor,
    dki_accum: torch.Tensor,
    *,
    worklist: torch.Tensor,
    work_count: torch.Tensor,
    dki_writer_rank: Optional[torch.Tensor],
    dqi_semaphore: Optional[torch.Tensor],
    dki_semaphore: Optional[torch.Tensor],
    softmax_scale: float,
    indexer_softmax_scale: float,
    grad_scale: float,
    deterministic: bool,
) -> None:
    _validate_tensor(
        worklist,
        name="worklist",
        shape=(int(worklist.shape[0]), 4),
        dtype=torch.int32,
        device=q.device,
    )
    _validate_tensor(
        work_count,
        name="work_count",
        shape=(1,),
        dtype=torch.int32,
        device=q.device,
    )
    if deterministic:
        if any(
            tensor is None
            for tensor in (dki_writer_rank, dqi_semaphore, dki_semaphore)
        ):
            raise ValueError(
                "deterministic KL backward requires writer-rank and semaphore tensors"
            )
        deterministic_specs = (
            (dki_writer_rank, "dki_writer_rank", (int(worklist.shape[0]),)),
            (dqi_semaphore, "dqi_semaphore", (INDEX_HEADS, int(q.shape[0]), 4)),
            (
                dki_semaphore,
                "dki_semaphore",
                (int(metadata.kl_schedule.dki_owner_counts.shape[0]),),
            ),
        )
        for tensor, name, shape in deterministic_specs:
            _validate_tensor(
                tensor,
                name=name,
                shape=shape,
                dtype=torch.int32,
                device=q.device,
            )
    no_work = (
        int(q.shape[0]) == 0
        or int(k.shape[0]) == 0
        or metadata.total_rows == 0
        or int(worklist.shape[0]) == 0
    )
    if no_work:
        return

    compiled = _compile_main_kernel(
        q.device,
        deterministic=deterministic,
    )
    compiled(
        q,
        k,
        teacher_lse,
        qi,
        ki,
        indexer_lse,
        metadata.kl_schedule.physical_q_indices,
        metadata.kl_schedule.physical_valid_rows,
        metadata.kl_schedule.physical_row_ptr,
        worklist,
        work_count,
        metadata.kl_schedule.dki_owner_counts,
        dki_writer_rank,
        dqi_semaphore,
        dki_semaphore,
        dqi_accum,
        dki_accum,
        softmax_scale,
        indexer_softmax_scale,
        grad_scale,
    )


def sparse_kl_bwd_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    teacher_lse: torch.Tensor,
    qi: torch.Tensor,
    ki: torch.Tensor,
    indexer_lse: torch.Tensor,
    metadata: AttentionMetadata,
    *,
    softmax_scale: Optional[float] = None,
    indexer_softmax_scale: Optional[float] = None,
    loss_coeff: float = 1.0,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run true-varlen sparse KL backward and return natural-layout BF16 gradients."""

    if type(deterministic) is not bool:
        raise TypeError("deterministic must be a Python bool")
    kl_schedule = metadata.kl_schedule
    if (
        not kl_schedule.enabled
        or kl_schedule.scheduler_metadata is None
        or kl_schedule.work_count is None
        or kl_schedule.physical_row_ptr is None
        or kl_schedule.physical_q_indices is None
        or kl_schedule.physical_valid_rows is None
        or kl_schedule.dki_owner_counts is None
        or kl_schedule.dki_split_indices is None
        or kl_schedule.dki_split_count is None
    ):
        raise ValueError("metadata must include the GPU-prepared KL schedule")

    dqi_accum = torch.zeros(
        (q.shape[0], INDEX_HEADS, HEAD_DIM),
        dtype=torch.float32,
        device=q.device,
    )
    total_k_padded = _total_padded_workspace(
        int(k.shape[0]), metadata.cu_seqlens_k
    )
    dki_accum = torch.empty(
        (total_k_padded, 1, HEAD_DIM),
        dtype=torch.float32,
        device=q.device,
    )
    dqi = torch.empty_like(qi)
    dki = torch.empty_like(ki)
    _validate_inputs(
        q,
        k,
        teacher_lse,
        qi,
        ki,
        indexer_lse,
        metadata,
        dqi_accum,
        dki_accum,
        dki,
    )
    softmax_scale, indexer_softmax_scale, grad_scale = _resolve_scales(
        int(q.shape[0]),
        softmax_scale,
        indexer_softmax_scale,
        loss_coeff,
    )
    if int(k.shape[0]) > 0:
        _compile_dki_split_zero(q.device)(
            dki_accum,
            kl_schedule.dki_split_indices,
            kl_schedule.dki_split_count,
        )
    worklist = kl_schedule.scheduler_metadata
    dki_writer_rank = None
    dqi_semaphore = None
    dki_semaphore = None
    if deterministic:
        (
            worklist,
            dki_writer_rank,
            dqi_semaphore,
            dki_semaphore,
        ) = _prepare_deterministic_backward(
            worklist,
            kl_schedule.work_count,
            total_q=int(q.shape[0]),
            padded_blocks=total_k_padded // BLOCK_SIZE,
        )
    _launch_main_backward(
        q,
        k,
        teacher_lse,
        qi,
        ki,
        indexer_lse,
        metadata,
        dqi_accum,
        dki_accum,
        worklist=worklist,
        work_count=kl_schedule.work_count,
        dki_writer_rank=dki_writer_rank,
        dqi_semaphore=dqi_semaphore,
        dki_semaphore=dki_semaphore,
        softmax_scale=softmax_scale,
        indexer_softmax_scale=indexer_softmax_scale,
        grad_scale=grad_scale,
        deterministic=deterministic,
    )
    if int(q.shape[0]) > 0:
        _compile_dqi_postprocess(q.device)(dqi_accum, dqi, grad_scale)
    if int(k.shape[0]) > 0:
        _compile_dki_postprocess(
            q.device,
            has_fragments=metadata.fragment_indices is not None,
        )(
            dki_accum,
            dki,
            kl_schedule.dki_owner_counts,
            grad_scale,
            metadata.cu_seqlens_k,
            metadata.fragment_indices,
            metadata.max_seqlen_k,
        )
    return dqi, dki


__all__ = ["sparse_kl_bwd_cute"]
