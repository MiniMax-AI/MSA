"""Packed-varlen PyTorch interface for the MiniMax-M3 indexer kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch

from msa_v1._common.aot_cache import compile_or_load
from msa_v1._common.compile_utils import compile_with_timing

from .m3_indexer_gemm import M3IndexerGemmSm100
from .m3_indexer_lse import M3IndexerLseSm100
from .m3_indexer_schedule import M3IndexerScheduleSm100
from .m3_indexer_topk import M3IndexerTopkSm100

_NUM_INDEX_HEADS = 4
_HEAD_DIM = 128
_TOPK = 16
_BLOCK_SIZE = 128
_LOGICAL_Q_TILE = 64
_GATHER_LSE_MAX_Q_LEN = 4_096
_GATHER_LSE_MIN_K_BLOCKS = 256
_SM103_DENSITY_MIN_Q_LEN = 4_096
_SM103_DENSITY_MAX_Q_LEN = 8_192
_SM103_GATHER_LSE_MIN_K_BLOCKS = 56
_MAX_SEQLEN = 4_194_240
_MAX_TOTAL_TOKENS = (1 << 31) - 1
_SCHEDULE_COMPILE_CACHE: dict[tuple, object] = {}
_GEMM_COMPILE_CACHE: dict[tuple, object] = {}
_TOPK_COMPILE_CACHE: dict[tuple, object] = {}
_LSE_COMPILE_CACHE: dict[tuple, object] = {}


@dataclass(eq=False)
class IndexerSchedule:
    """Reusable device-side 64-row task table for one micro-batch."""

    task_batch_idx: torch.Tensor
    task_q_local_begin: torch.Tensor
    num_task_slots: int = 0
    total_q: int = 0
    batch: int = 0
    _cu_seqlens_q: Optional[torch.Tensor] = field(default=None, repr=False)
    _cu_seqlens_kv: Optional[torch.Tensor] = field(default=None, repr=False)
    _fragment_indices: Optional[torch.Tensor] = field(default=None, repr=False)


@dataclass(eq=False)
class IndexerForwardWorkspace:
    """Preallocated K1 statistics, final outputs, and an optional schedule."""

    score: torch.Tensor
    block_sum: torch.Tensor
    topk_indices: torch.Tensor
    selected_lse: torch.Tensor
    schedule: Optional[IndexerSchedule] = None


def _make_fake_tensor(
    dtype,
    shape,
    *,
    stride_order,
    assumed_align: int = 16,
):
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=stride_order,
        assumed_align=assumed_align,
    )


def _device_capability(device: torch.device) -> tuple[int, int]:
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError("MSA v1 indexer requires an SM100 or SM103 GPU")
    return capability


def _num_task_slots(total_q: int, batch: int) -> int:
    return (total_q + batch * (_LOGICAL_Q_TILE - 1)) // _LOGICAL_Q_TILE


def _num_k_blocks(max_seqlen_kv: int) -> int:
    return (max_seqlen_kv + _BLOCK_SIZE - 1) // _BLOCK_SIZE


def _compile_schedule(device: torch.device, *, has_fragment_indices: bool):
    capability = _device_capability(device)
    key = (
        "m3_varlen_indexer_schedule_sm100",
        capability,
        has_fragment_indices,
    )
    if key not in _SCHEDULE_COMPILE_CACHE:
        offsets = cute.sym_int64()
        fragments = cute.sym_int64()
        task_slots = cute.sym_int64()
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compile_args = (
            M3IndexerScheduleSm100(),
            _make_fake_tensor(
                cutlass.Int32,
                (offsets,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (offsets,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (task_slots,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (task_slots,),
                stride_order=(0,),
                assumed_align=4,
            ),
            cutlass.Int32(1),
            cutlass.Int32(1),
            (
                _make_fake_tensor(
                    cutlass.Int32,
                    (fragments,),
                    stride_order=(0,),
                    assumed_align=4,
                )
                if has_fragment_indices
                else None
            ),
            fake_stream,
        )
        _SCHEDULE_COMPILE_CACHE[key] = compile_or_load(
            key,
            lambda: compile_with_timing(*compile_args, options="--enable-tvm-ffi"),
            log_prefix="m3_indexer_schedule",
        )
    return _SCHEDULE_COMPILE_CACHE[key]


def _compile_gemm(
    device: torch.device,
    *,
    has_fragment_indices: bool,
    use_fp16_score: bool,
):
    capability = _device_capability(device)
    key = (
        "m3_varlen_indexer_gemm_sm100",
        capability,
        has_fragment_indices,
        use_fp16_score,
    )
    if key not in _GEMM_COMPILE_CACHE:
        q_rows = cute.sym_int64()
        k_rows = cute.sym_int64()
        score_tiles = cute.sym_int64()
        offsets = cute.sym_int64()
        fragments = cute.sym_int64()
        task_slots = cute.sym_int64()
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compile_args = (
            M3IndexerGemmSm100(
                compute_capability=capability, use_fp16_score=use_fp16_score
            ),
            _make_fake_tensor(
                cutlass.BFloat16,
                (q_rows, _HEAD_DIM),
                stride_order=(1, 0),
            ),
            _make_fake_tensor(
                cutlass.BFloat16,
                (k_rows, _HEAD_DIM),
                stride_order=(1, 0),
            ),
            _make_fake_tensor(
                cutlass.Float16 if use_fp16_score else cutlass.Float32,
                (score_tiles, 2, 128),
                stride_order=(2, 1, 0),
            ),
            _make_fake_tensor(
                cutlass.Float32,
                (score_tiles, 2, 128),
                stride_order=(2, 1, 0),
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (offsets,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (offsets,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (task_slots,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (task_slots,),
                stride_order=(0,),
                assumed_align=4,
            ),
            (
                _make_fake_tensor(
                    cutlass.Int32,
                    (fragments,),
                    stride_order=(0,),
                    assumed_align=4,
                )
                if has_fragment_indices
                else None
            ),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Float32(1.0),
            fake_stream,
        )
        _GEMM_COMPILE_CACHE[key] = compile_or_load(
            key,
            lambda: compile_with_timing(*compile_args, options="--enable-tvm-ffi"),
            log_prefix="m3_indexer_gemm",
        )
    return _GEMM_COMPILE_CACHE[key]


def _compile_topk(
    device: torch.device,
    *,
    has_fragment_indices: bool,
    small_plan: bool,
    gather_lse: bool,
    deterministic: bool,
    use_fp16_score: bool,
):
    capability = _device_capability(device)
    key = (
        "m3_varlen_indexer_topk_sm100",
        capability,
        has_fragment_indices,
        small_plan,
        gather_lse,
        deterministic,
        use_fp16_score,
    )
    if key not in _TOPK_COMPILE_CACHE:
        score_tiles = cute.sym_int64()
        offsets = cute.sym_int64()
        fragments = cute.sym_int64()
        task_slots = cute.sym_int64()
        total_q = cute.sym_int64()
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compile_args = (
            M3IndexerTopkSm100(
                deterministic=deterministic,
                small_plan=small_plan,
                gather_lse=gather_lse,
                use_fp16_score=use_fp16_score,
            ),
            _make_fake_tensor(
                cutlass.Float16 if use_fp16_score else cutlass.Float32,
                (score_tiles, 2, 128),
                stride_order=(2, 1, 0),
            ),
            _make_fake_tensor(
                cutlass.Float32,
                (score_tiles, 2, 128),
                stride_order=(2, 1, 0),
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (offsets,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (offsets,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (task_slots,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (task_slots,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (_NUM_INDEX_HEADS, total_q, _TOPK),
                stride_order=(2, 1, 0),
                assumed_align=16,
            ),
            _make_fake_tensor(
                cutlass.Float32,
                (_NUM_INDEX_HEADS, total_q),
                stride_order=(1, 0),
            ),
            (
                _make_fake_tensor(
                    cutlass.Int32,
                    (fragments,),
                    stride_order=(0,),
                    assumed_align=4,
                )
                if has_fragment_indices
                else None
            ),
            cutlass.Int32(1),
            cutlass.Int32(1),
            fake_stream,
        )
        _TOPK_COMPILE_CACHE[key] = compile_or_load(
            key,
            lambda: compile_with_timing(*compile_args, options="--enable-tvm-ffi"),
            log_prefix=(
                "m3_indexer_topk "
                f"fragment={has_fragment_indices} "
                f"small_plan={small_plan} gather_lse={gather_lse} "
                f"deterministic={deterministic} fp16_score={use_fp16_score}"
            ),
        )
    return _TOPK_COMPILE_CACHE[key]


def _compile_lse(device: torch.device, *, use_fp16_score: bool):
    capability = _device_capability(device)
    key = ("m3_varlen_indexer_lse_sm100", capability, use_fp16_score)
    if key not in _LSE_COMPILE_CACHE:
        score_tiles = cute.sym_int64()
        offsets = cute.sym_int64()
        task_slots = cute.sym_int64()
        total_q = cute.sym_int64()
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compile_args = (
            M3IndexerLseSm100(use_fp16_score=use_fp16_score),
            _make_fake_tensor(
                cutlass.Float16 if use_fp16_score else cutlass.Float32,
                (score_tiles, 2, 128),
                stride_order=(2, 1, 0),
            ),
            _make_fake_tensor(
                cutlass.Float32,
                (score_tiles, 2, 128),
                stride_order=(2, 1, 0),
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (offsets,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (task_slots,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (task_slots,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (_NUM_INDEX_HEADS, total_q, _TOPK),
                stride_order=(2, 1, 0),
                assumed_align=16,
            ),
            _make_fake_tensor(
                cutlass.Float32,
                (_NUM_INDEX_HEADS, total_q),
                stride_order=(1, 0),
            ),
            cutlass.Int32(1),
            cutlass.Int32(1),
            fake_stream,
        )
        _LSE_COMPILE_CACHE[key] = compile_or_load(
            key,
            lambda: compile_with_timing(*compile_args, options="--enable-tvm-ffi"),
            log_prefix="m3_indexer_lse",
        )
    return _LSE_COMPILE_CACHE[key]


def _validate_cu_seqlens(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    *,
    device: torch.device,
) -> int:
    for name, tensor in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_kv", cu_seqlens_kv),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be int32")
        if not tensor.is_cuda or tensor.device != device:
            raise ValueError(f"{name} must be a CUDA tensor on {device}")
        if tensor.ndim != 1 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be a contiguous one-dimensional tensor")
    if cu_seqlens_q.numel() != cu_seqlens_kv.numel():
        raise ValueError("Q and KV cu_seqlens must have the same batch size")
    if cu_seqlens_q.numel() < 2:
        raise ValueError("batch must be positive")
    return cu_seqlens_q.numel() - 1


def _validate_fragment_indices(
    fragment_indices: Optional[torch.Tensor],
    *,
    batch: int,
    device: torch.device,
) -> None:
    if fragment_indices is None:
        return
    if fragment_indices.dtype != torch.int32:
        raise TypeError("fragment_indices must be int32")
    if not fragment_indices.is_cuda or fragment_indices.device != device:
        raise ValueError("fragment_indices must be a CUDA tensor on the Q device")
    if tuple(fragment_indices.shape) != (batch,) or not fragment_indices.is_contiguous():
        raise ValueError("fragment_indices must be contiguous with shape [B]")


def _validate_qk(q: torch.Tensor, k: torch.Tensor) -> tuple[int, int]:
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError("q and k must both be torch.bfloat16")
    if not q.is_cuda or not k.is_cuda or q.device != k.device:
        raise ValueError("q and k must be CUDA tensors on the same device")
    if q.ndim != 3 or tuple(q.shape[1:]) != (_NUM_INDEX_HEADS, _HEAD_DIM):
        raise ValueError("q must have shape [total_q, 4, 128]")
    if k.ndim != 3 or tuple(k.shape[1:]) != (1, _HEAD_DIM):
        raise ValueError("k must have shape [total_k, 1, 128]")
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("q and k must be contiguous")
    total_q = int(q.shape[0])
    total_k = int(k.shape[0])
    if total_q <= 0 or total_q > _MAX_TOTAL_TOKENS:
        raise ValueError(f"total_q must be in [1, {_MAX_TOTAL_TOKENS}]")
    if total_k <= 0 or total_k > _MAX_TOTAL_TOKENS:
        raise ValueError(f"total_k must be in [1, {_MAX_TOTAL_TOKENS}]")
    _device_capability(q.device)
    return total_q, total_k


def _validate_max_seqlens(max_seqlen_q: int, max_seqlen_kv: int) -> None:
    if not isinstance(max_seqlen_q, int) or not isinstance(max_seqlen_kv, int):
        raise TypeError("max_seqlen_q and max_seqlen_kv must be Python ints")
    if max_seqlen_q <= 0 or max_seqlen_q > _MAX_SEQLEN:
        raise ValueError(f"max_seqlen_q must be in [1, {_MAX_SEQLEN}]")
    if max_seqlen_kv < max_seqlen_q or max_seqlen_kv > _MAX_SEQLEN:
        raise ValueError(
            f"max_seqlen_kv must be in [max_seqlen_q, {_MAX_SEQLEN}]"
        )


def _inverse_lse_temperature(lse_temperature: float) -> float:
    if not isinstance(lse_temperature, (int, float)) or isinstance(
        lse_temperature, bool
    ):
        raise TypeError("lse_temperature must be a Python float")
    lse_temperature = float(lse_temperature)
    if not math.isfinite(lse_temperature) or lse_temperature <= 0.0:
        raise ValueError(
            f"lse_temperature must be finite and > 0, got {lse_temperature}"
        )
    return 1.0 / lse_temperature


def allocate_indexer_schedule(
    *,
    total_q: int,
    batch: int,
    device: torch.device,
) -> IndexerSchedule:
    """Allocate reusable schedule storage outside the timed region."""

    if not isinstance(total_q, int) or not (0 < total_q <= _MAX_TOTAL_TOKENS):
        raise ValueError(f"total_q must be in [1, {_MAX_TOTAL_TOKENS}]")
    if not isinstance(batch, int) or batch <= 0:
        raise ValueError("batch must be a positive Python int")
    num_task_slots = _num_task_slots(total_q, batch)
    return IndexerSchedule(
        task_batch_idx=torch.empty(num_task_slots, dtype=torch.int32, device=device),
        task_q_local_begin=torch.empty(
            num_task_slots,
            dtype=torch.int32,
            device=device,
        ),
    )


def _validate_schedule_storage(
    schedule: IndexerSchedule,
    *,
    total_q: int,
    batch: int,
    device: torch.device,
) -> int:
    if not isinstance(schedule, IndexerSchedule):
        raise TypeError("schedule must be an IndexerSchedule")
    num_task_slots = _num_task_slots(total_q, batch)
    for name, tensor in (
        ("task_batch_idx", schedule.task_batch_idx),
        ("task_q_local_begin", schedule.task_q_local_begin),
    ):
        if not tensor.is_cuda or tensor.device != device:
            raise ValueError(f"schedule {name} must be on {device}")
        if tensor.dtype != torch.int32 or tensor.ndim != 1:
            raise ValueError(f"schedule {name} must be one-dimensional int32")
        if tensor.numel() < num_task_slots or not tensor.is_contiguous():
            raise ValueError(
                f"schedule {name} must be contiguous with capacity >= {num_task_slots}"
            )
    return num_task_slots


def _schedule_matches(
    schedule: IndexerSchedule,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    fragment_indices: Optional[torch.Tensor],
    *,
    total_q: int,
) -> bool:
    batch = cu_seqlens_q.numel() - 1
    return (
        schedule._cu_seqlens_q is cu_seqlens_q
        and schedule._cu_seqlens_kv is cu_seqlens_kv
        and schedule._fragment_indices is fragment_indices
        and schedule.total_q == total_q
        and schedule.batch == batch
        and schedule.num_task_slots == _num_task_slots(total_q, batch)
    )


def prepare_indexer_schedule(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    *,
    total_q: int,
    fragment_indices: Optional[torch.Tensor] = None,
    schedule: Optional[IndexerSchedule] = None,
) -> IndexerSchedule:
    """Build a reusable device-side packed-varlen task table."""

    if not isinstance(total_q, int) or not (0 < total_q <= _MAX_TOTAL_TOKENS):
        raise ValueError(f"total_q must be in [1, {_MAX_TOTAL_TOKENS}]")
    device = cu_seqlens_q.device
    batch = _validate_cu_seqlens(cu_seqlens_q, cu_seqlens_kv, device=device)
    _validate_fragment_indices(fragment_indices, batch=batch, device=device)
    required_slots = _num_task_slots(total_q, batch)
    if (
        schedule is None
        or schedule.task_batch_idx.numel() < required_slots
        or schedule.task_q_local_begin.numel() < required_slots
    ):
        schedule = allocate_indexer_schedule(
            total_q=total_q,
            batch=batch,
            device=device,
        )
    _validate_schedule_storage(
        schedule,
        total_q=total_q,
        batch=batch,
        device=device,
    )
    _compile_schedule(
        device,
        has_fragment_indices=fragment_indices is not None,
    )(
        cu_seqlens_q,
        cu_seqlens_kv,
        schedule.task_batch_idx,
        schedule.task_q_local_begin,
        cutlass.Int32(batch),
        cutlass.Int32(required_slots),
        fragment_indices,
    )
    schedule.num_task_slots = required_slots
    schedule.total_q = total_q
    schedule.batch = batch
    schedule._cu_seqlens_q = cu_seqlens_q
    schedule._cu_seqlens_kv = cu_seqlens_kv
    schedule._fragment_indices = fragment_indices
    return schedule


def allocate_indexer_workspace(
    *,
    total_q: int,
    batch: int,
    max_seqlen_kv: int,
    device: torch.device,
    use_fp16_score: bool = False,
) -> IndexerForwardWorkspace:
    """Allocate uninitialized statistics and outputs outside the timed region."""

    if not isinstance(total_q, int) or not (0 < total_q <= _MAX_TOTAL_TOKENS):
        raise ValueError(f"total_q must be in [1, {_MAX_TOTAL_TOKENS}]")
    if not isinstance(batch, int) or batch <= 0:
        raise ValueError("batch must be a positive Python int")
    if type(use_fp16_score) is not bool:
        raise TypeError("use_fp16_score must be a Python bool")
    if not isinstance(max_seqlen_kv, int) or not (0 < max_seqlen_kv <= _MAX_SEQLEN):
        raise ValueError(f"max_seqlen_kv must be in [1, {_MAX_SEQLEN}]")
    num_task_slots = _num_task_slots(total_q, batch)
    max_k_blocks = _num_k_blocks(max_seqlen_kv)
    return IndexerForwardWorkspace(
        score=torch.empty(
            (num_task_slots * max_k_blocks, 2, 128),
            dtype=torch.float16 if use_fp16_score else torch.float32,
            device=device,
        ),
        block_sum=torch.empty(
            (num_task_slots * max_k_blocks, 2, 128),
            dtype=torch.float32,
            device=device,
        ),
        topk_indices=torch.empty(
            (_NUM_INDEX_HEADS, total_q, _TOPK),
            dtype=torch.int32,
            device=device,
        ),
        selected_lse=torch.empty(
            (_NUM_INDEX_HEADS, total_q),
            dtype=torch.float32,
            device=device,
        ),
    )


def _validate_workspace(
    workspace: IndexerForwardWorkspace,
    *,
    total_q: int,
    batch: int,
    max_seqlen_kv: int,
    device: torch.device,
    use_fp16_score: bool,
) -> None:
    if not isinstance(workspace, IndexerForwardWorkspace):
        raise TypeError("workspace must be an IndexerForwardWorkspace")
    num_task_slots = _num_task_slots(total_q, batch)
    max_k_blocks = _num_k_blocks(max_seqlen_kv)
    score_dtype = torch.float16 if use_fp16_score else torch.float32
    expected = (
        (
            workspace.score,
            (num_task_slots * max_k_blocks, 2, 128),
            score_dtype,
            "score",
        ),
        (
            workspace.block_sum,
            (num_task_slots * max_k_blocks, 2, 128),
            torch.float32,
            "block_sum",
        ),
        (
            workspace.topk_indices,
            (_NUM_INDEX_HEADS, total_q, _TOPK),
            torch.int32,
            "topk_indices",
        ),
        (
            workspace.selected_lse,
            (_NUM_INDEX_HEADS, total_q),
            torch.float32,
            "selected_lse",
        ),
    )
    for tensor, shape, dtype, name in expected:
        if not tensor.is_cuda or tensor.device != device:
            raise ValueError(f"workspace {name} must be on {device}")
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise ValueError(
                f"workspace {name} must have shape {shape} and dtype {dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"workspace {name} must be contiguous")


def _use_gather_lse(
    *,
    total_q: int,
    max_k_blocks: int,
    capability: tuple[int, int],
) -> bool:
    if capability == (10, 3) and _SM103_DENSITY_MIN_Q_LEN <= total_q <= _SM103_DENSITY_MAX_Q_LEN:
        return max_k_blocks > _SM103_GATHER_LSE_MIN_K_BLOCKS
    return max_k_blocks > _TOPK and (
        total_q <= _GATHER_LSE_MAX_Q_LEN
        or max_k_blocks >= _GATHER_LSE_MIN_K_BLOCKS
    )


def m3_indexer_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    fragment_indices: Optional[torch.Tensor] = None,
    schedule: Optional[IndexerSchedule] = None,
    workspace: Optional[IndexerForwardWorkspace] = None,
    lse_temperature: float = 1.0,
    deterministic: bool = False,
    use_fp16_score: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run causal packed-varlen K1+K2 and return temperature-scaled LSE."""

    if type(deterministic) is not bool:
        raise TypeError("deterministic must be a Python bool")
    if type(use_fp16_score) is not bool:
        raise TypeError("use_fp16_score must be a Python bool")
    total_q, _ = _validate_qk(q, k)
    inv_lse_temperature = _inverse_lse_temperature(lse_temperature)
    _validate_max_seqlens(max_seqlen_q, max_seqlen_kv)
    batch = _validate_cu_seqlens(cu_seqlens_q, cu_seqlens_kv, device=q.device)
    _validate_fragment_indices(fragment_indices, batch=batch, device=q.device)
    if workspace is None:
        workspace = allocate_indexer_workspace(
            total_q=total_q,
            batch=batch,
            max_seqlen_kv=max_seqlen_kv,
            device=q.device,
            use_fp16_score=use_fp16_score,
        )
    _validate_workspace(
        workspace,
        total_q=total_q,
        batch=batch,
        max_seqlen_kv=max_seqlen_kv,
        device=q.device,
        use_fp16_score=use_fp16_score,
    )
    if schedule is None and workspace.schedule is not None and _schedule_matches(
        workspace.schedule,
        cu_seqlens_q,
        cu_seqlens_kv,
        fragment_indices,
        total_q=total_q,
    ):
        schedule = workspace.schedule
    if schedule is None:
        schedule = prepare_indexer_schedule(
            cu_seqlens_q,
            cu_seqlens_kv,
            total_q=total_q,
            fragment_indices=fragment_indices,
        )
    else:
        _validate_schedule_storage(
            schedule,
            total_q=total_q,
            batch=batch,
            device=q.device,
        )
        if not _schedule_matches(
            schedule,
            cu_seqlens_q,
            cu_seqlens_kv,
            fragment_indices,
            total_q=total_q,
        ):
            raise ValueError(
                "schedule does not match this micro-batch; rebuild it with "
                "prepare_indexer_schedule"
            )
    workspace.schedule = schedule

    max_k_blocks = _num_k_blocks(max_seqlen_kv)
    capability = _device_capability(q.device)
    gather_lse = _use_gather_lse(
        total_q=total_q,
        max_k_blocks=max_k_blocks,
        capability=capability,
    )
    small_plan = max_k_blocks <= _TOPK
    has_fragment_indices = fragment_indices is not None
    _compile_gemm(
        q.device,
        has_fragment_indices=has_fragment_indices,
        use_fp16_score=use_fp16_score,
    )(
        q.view(total_q * _NUM_INDEX_HEADS, _HEAD_DIM),
        k.view(k.shape[0], _HEAD_DIM),
        workspace.score,
        workspace.block_sum,
        cu_seqlens_q,
        cu_seqlens_kv,
        schedule.task_batch_idx,
        schedule.task_q_local_begin,
        fragment_indices,
        cutlass.Int32(schedule.num_task_slots),
        cutlass.Int32(max_k_blocks),
        cutlass.Float32(inv_lse_temperature),
    )
    _compile_topk(
        q.device,
        has_fragment_indices=has_fragment_indices,
        small_plan=small_plan,
        gather_lse=gather_lse,
        deterministic=deterministic,
        use_fp16_score=use_fp16_score,
    )(
        workspace.score,
        workspace.block_sum,
        cu_seqlens_q,
        cu_seqlens_kv,
        schedule.task_batch_idx,
        schedule.task_q_local_begin,
        workspace.topk_indices,
        workspace.selected_lse,
        fragment_indices,
        cutlass.Int32(schedule.num_task_slots),
        cutlass.Int32(max_k_blocks),
    )
    if gather_lse and deterministic:
        _compile_lse(q.device, use_fp16_score=use_fp16_score)(
            workspace.score,
            workspace.block_sum,
            cu_seqlens_q,
            schedule.task_batch_idx,
            schedule.task_q_local_begin,
            workspace.topk_indices,
            workspace.selected_lse,
            cutlass.Int32(schedule.num_task_slots),
            cutlass.Int32(max_k_blocks),
        )
    return workspace.topk_indices, workspace.selected_lse


__all__ = [
    "IndexerForwardWorkspace",
    "IndexerSchedule",
    "allocate_indexer_schedule",
    "allocate_indexer_workspace",
    "m3_indexer_forward",
    "prepare_indexer_schedule",
]
