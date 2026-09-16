"""Public batch-1 arbitrary-mask interface for the MiniMax-M3 indexer."""

from __future__ import annotations

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch

from msa_v1._common.aot_cache import compile_or_load
from msa_v1._common.compile_utils import compile_with_timing

from .m3_arbitrary_plan import M3ArbitraryMaskPlan
from .m3_indexer_gemm import M3IndexerGemmSm100
from .m3_indexer_lse import M3IndexerLseSm100
from .m3_indexer_topk import M3IndexerTopkSm100

_NUM_INDEX_HEADS = 4
_HEAD_DIM = 128
_TOPK = 16
_LOGICAL_Q_TILE = 64
_GATHER_LSE_MAX_Q_LEN = 4_096
_GATHER_LSE_MIN_PLAN_TILES = 256
_SM103_PLAN_DENSITY_MIN_Q_LEN = 4_096
_SM103_PLAN_DENSITY_MAX_Q_LEN = 8_192
_SM103_GATHER_LSE_MIN_PLAN_DENSITY = 56
_SM103_DETERMINISTIC_GATHER_LSE_MIN_PLAN_DENSITY = 112
# Largest sequence whose dense tile plan fits signed int32 offsets.
_MAX_SEQLEN = 4_194_240
_MAX_PLAN_TILES = 32_768

_GEMM_COMPILE_CACHE: dict[tuple, object] = {}
_TOPK_COMPILE_CACHE: dict[tuple, object] = {}
_LSE_COMPILE_CACHE: dict[tuple, object] = {}


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


def _compile_gemm(device: torch.device, *, use_fp16_score: bool):
    capability = torch.cuda.get_device_capability(device)
    key = ("m3_arbitrary_indexer_gemm_sm100", capability, use_fp16_score)
    if key not in _GEMM_COMPILE_CACHE:
        q_rows = cute.sym_int64()
        k_rows = cute.sym_int64()
        tile_offsets = cute.sym_int64()
        partial_tiles = cute.sym_int64()
        full_tiles = cute.sym_int64()
        score_tiles = cute.sym_int64()
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
                cutlass.Int32,
                (1, tile_offsets),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (1, partial_tiles),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Uint32,
                (partial_tiles, 2, 128, 4),
                stride_order=(3, 2, 1, 0),
                assumed_align=16,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (1, tile_offsets),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (1, full_tiles),
                stride_order=(1, 0),
                assumed_align=4,
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
            cutlass.Int32(128),
            cutlass.Float32(1.0),
            fake_stream,
        )
        _GEMM_COMPILE_CACHE[key] = compile_or_load(
            key,
            lambda: compile_with_timing(*compile_args, options="--enable-tvm-ffi"),
            log_prefix="m3_arbitrary_indexer_gemm",
        )
    return _GEMM_COMPILE_CACHE[key]


def _compile_topk(
    device: torch.device,
    *,
    deterministic: bool,
    small_plan: bool,
    gather_lse: bool,
    use_fp16_score: bool,
    rebase_block_ids: bool,
):
    capability = torch.cuda.get_device_capability(device)
    key = (
        "m3_arbitrary_indexer_topk_sm100",
        capability,
        deterministic,
        small_plan,
        gather_lse,
        use_fp16_score,
        rebase_block_ids,
    )
    if key not in _TOPK_COMPILE_CACHE:
        score_tiles = cute.sym_int64()
        tile_offsets = cute.sym_int64()
        partial_tiles = cute.sym_int64()
        full_tiles = cute.sym_int64()
        q_len = cute.sym_int64()
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compile_args = (
            M3IndexerTopkSm100(
                deterministic=deterministic,
                small_plan=small_plan,
                gather_lse=gather_lse,
                use_fp16_score=use_fp16_score,
                rebase_block_ids=rebase_block_ids,
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
                (1, tile_offsets),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (1, partial_tiles),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (1, tile_offsets),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (1, full_tiles),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (_NUM_INDEX_HEADS, q_len, _TOPK),
                stride_order=(2, 1, 0),
                assumed_align=16,
            ),
            _make_fake_tensor(
                cutlass.Float32,
                (_NUM_INDEX_HEADS, q_len),
                stride_order=(1, 0),
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (1, q_len),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            (
                _make_fake_tensor(
                    cutlass.Int32,
                    (q_len,),
                    stride_order=(0,),
                    assumed_align=4,
                )
                if rebase_block_ids
                else None
            ),
            cutlass.Int32(128),
            fake_stream,
        )
        _TOPK_COMPILE_CACHE[key] = compile_or_load(
            key,
            lambda: compile_with_timing(*compile_args, options="--enable-tvm-ffi"),
            log_prefix=(
                "m3_arbitrary_indexer_topk "
                f"deterministic={deterministic} small_plan={small_plan} "
                f"gather_lse={gather_lse} fp16_score={use_fp16_score} "
                f"rebase={rebase_block_ids}"
            ),
        )
    return _TOPK_COMPILE_CACHE[key]


def _compile_lse(
    device: torch.device,
    *,
    rebase_block_ids: bool,
):
    capability = torch.cuda.get_device_capability(device)
    key = ("m3_arbitrary_indexer_lse_sm100", capability, rebase_block_ids)
    if key not in _LSE_COMPILE_CACHE:
        score_tiles = cute.sym_int64()
        partial_tiles = cute.sym_int64()
        full_tiles = cute.sym_int64()
        q_len = cute.sym_int64()
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compile_args = (
            M3IndexerLseSm100(rebase_block_ids=rebase_block_ids),
            _make_fake_tensor(
                cutlass.Float32,
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
                (1, partial_tiles),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (1, full_tiles),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            _make_fake_tensor(
                cutlass.Int32,
                (_NUM_INDEX_HEADS, q_len, _TOPK),
                stride_order=(2, 1, 0),
                assumed_align=16,
            ),
            _make_fake_tensor(
                cutlass.Float32,
                (_NUM_INDEX_HEADS, q_len),
                stride_order=(1, 0),
            ),
            (
                _make_fake_tensor(
                    cutlass.Int32,
                    (q_len,),
                    stride_order=(0,),
                    assumed_align=4,
                )
                if rebase_block_ids
                else None
            ),
            cutlass.Int32(128),
            fake_stream,
        )
        _LSE_COMPILE_CACHE[key] = compile_or_load(
            key,
            lambda: compile_with_timing(*compile_args, options="--enable-tvm-ffi"),
            log_prefix=f"m3_arbitrary_indexer_lse rebase={rebase_block_ids}",
        )
    return _LSE_COMPILE_CACHE[key]


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


def _validate_plan(plan: M3ArbitraryMaskPlan, device: torch.device) -> None:
    if not isinstance(plan, M3ArbitraryMaskPlan):
        raise TypeError("plan must be an M3ArbitraryMaskPlan")
    if not isinstance(plan.q_len, int) or not isinstance(plan.k_len, int):
        raise TypeError("plan.q_len and plan.k_len must be Python ints")
    if plan.q_len <= 0 or plan.q_len > _MAX_SEQLEN:
        raise ValueError(f"plan.q_len must be in [1, {_MAX_SEQLEN}]")
    if plan.k_len < plan.q_len or plan.k_len > _MAX_SEQLEN:
        raise ValueError(f"plan.k_len must be in [plan.q_len, {_MAX_SEQLEN}]")
    num_q_tiles = (plan.q_len + 63) // 64
    num_partial = int(plan.partial_block_indices.numel())
    num_full = int(plan.full_block_indices.numel())
    expected_shapes = (
        ("partial_offsets", plan.partial_offsets, (1, num_q_tiles + 1)),
        (
            "partial_block_indices",
            plan.partial_block_indices,
            (1, num_partial),
        ),
        (
            "partial_masks",
            plan.partial_masks,
            (num_partial, 2, 128, 4),
        ),
        ("full_offsets", plan.full_offsets, (1, num_q_tiles + 1)),
        (
            "full_block_indices",
            plan.full_block_indices,
            (1, num_full),
        ),
        ("local_block_positions", plan.local_block_positions, (1, plan.q_len)),
    )
    for name, tensor, shape in expected_shapes:
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"plan.{name} has an invalid shape")

    expected_dtypes = (
        ("partial_offsets", plan.partial_offsets, torch.int32),
        ("partial_block_indices", plan.partial_block_indices, torch.int32),
        ("partial_masks", plan.partial_masks, torch.uint32),
        ("full_offsets", plan.full_offsets, torch.int32),
        ("full_block_indices", plan.full_block_indices, torch.int32),
        ("local_block_positions", plan.local_block_positions, torch.int32),
    )
    for name, tensor, dtype in expected_dtypes:
        if tensor.device != device:
            raise ValueError(f"plan.{name} must be on the Q/K device")
        if tensor.dtype != dtype:
            raise TypeError(f"plan.{name} must have dtype {dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"plan.{name} must be contiguous")
    if (
        not isinstance(plan.max_plan_tiles, int)
        or plan.max_plan_tiles < 0
        or plan.max_plan_tiles > _MAX_PLAN_TILES
    ):
        raise ValueError(
            f"plan.max_plan_tiles must be a Python int in [0, {_MAX_PLAN_TILES}]"
        )


def _validate_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    plan: M3ArbitraryMaskPlan,
) -> tuple[int, int]:
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError("q and k must both be torch.bfloat16")
    if not q.is_cuda or not k.is_cuda:
        raise ValueError("q and k must be CUDA tensors")
    if q.device != k.device:
        raise ValueError("q and k must be on the same device")
    if q.ndim != 3 or tuple(q.shape[1:]) != (
        _NUM_INDEX_HEADS,
        _HEAD_DIM,
    ):
        raise ValueError("q must have shape [q_len, 4, 128]")
    if k.ndim != 3 or tuple(k.shape[1:]) != (1, _HEAD_DIM):
        raise ValueError("k must have shape [k_len, 1, 128]")
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("q and k must be contiguous")
    q_len = int(q.shape[0])
    k_len = int(k.shape[0])
    if q_len <= 0 or q_len > _MAX_SEQLEN:
        raise ValueError(f"q_len must be in [1, {_MAX_SEQLEN}]")
    if k_len <= 0 or k_len > _MAX_SEQLEN:
        raise ValueError(f"k_len must be in [1, {_MAX_SEQLEN}]")
    if k_len < q_len:
        raise ValueError("k_len must be greater than or equal to q_len")
    capability = torch.cuda.get_device_capability(q.device)
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError("M3 indexer requires an SM100 or SM103 GPU")
    if plan.q_len != q_len or plan.k_len != k_len:
        raise ValueError("plan q_len/k_len must match the Q/K tensor lengths")
    _validate_plan(plan, q.device)
    return q_len, k_len


def _validate_stat_workspace(
    workspace: torch.Tensor,
    plan: M3ArbitraryMaskPlan,
    device: torch.device,
    *,
    name: str,
    dtype: torch.dtype = torch.float32,
) -> None:
    if workspace.dtype != dtype or not workspace.is_cuda:
        raise TypeError(f"{name} must be a CUDA {dtype} tensor")
    if workspace.device != device:
        raise ValueError(f"{name} must be on the plan device")
    if tuple(workspace.shape) != (plan.num_plan_tiles, 2, 128):
        raise ValueError(f"{name} must have shape [num_plan_tiles, 2, 128]")
    if not workspace.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_block_bases(
    block_bases: Optional[torch.Tensor],
    plan: M3ArbitraryMaskPlan,
    device: torch.device,
) -> None:
    if block_bases is None:
        return
    if not isinstance(block_bases, torch.Tensor):
        raise TypeError("block_bases must be a torch.Tensor or None")
    if block_bases.dtype != torch.int32:
        raise TypeError("block_bases must be torch.int32")
    if not block_bases.is_cuda or block_bases.device != device:
        raise ValueError("block_bases must be on the Q/K CUDA device")
    if tuple(block_bases.shape) != (plan.q_len,):
        raise ValueError("block_bases must have shape [q_len]")
    if not block_bases.is_contiguous():
        raise ValueError("block_bases must be contiguous")


def _prepare_topk(
    plan: M3ArbitraryMaskPlan,
    device: torch.device,
    topk_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    if topk_indices is None:
        return torch.empty(
            (_NUM_INDEX_HEADS, plan.q_len, _TOPK),
            dtype=torch.int32,
            device=device,
        )
    if (
        topk_indices.dtype != torch.int32
        or topk_indices.device != device
        or tuple(topk_indices.shape) != (_NUM_INDEX_HEADS, plan.q_len, _TOPK)
        or not topk_indices.is_contiguous()
    ):
        raise ValueError(
            "topk_indices must be contiguous int32 [4, q_len, 16] on the Q/K device"
        )
    return topk_indices


def _prepare_selected_lse(
    plan: M3ArbitraryMaskPlan,
    device: torch.device,
    selected_lse: Optional[torch.Tensor],
) -> torch.Tensor:
    if selected_lse is None:
        return torch.empty(
            (_NUM_INDEX_HEADS, plan.q_len),
            dtype=torch.float32,
            device=device,
        )
    if (
        selected_lse.dtype != torch.float32
        or selected_lse.device != device
        or tuple(selected_lse.shape) != (_NUM_INDEX_HEADS, plan.q_len)
        or not selected_lse.is_contiguous()
    ):
        raise ValueError(
            "selected_lse must be contiguous float32 [4, q_len] on the Q/K device"
        )
    return selected_lse


def _use_gather_lse(
    plan: M3ArbitraryMaskPlan,
    *,
    deterministic: bool,
    capability: tuple[int, int],
) -> bool:
    """Choose the cooperative gather family when its cost is amortized."""

    if (
        capability == (10, 3)
        and _SM103_PLAN_DENSITY_MIN_Q_LEN <= plan.q_len <= _SM103_PLAN_DENSITY_MAX_Q_LEN
    ):
        min_density = (
            _SM103_DETERMINISTIC_GATHER_LSE_MIN_PLAN_DENSITY
            if deterministic
            else _SM103_GATHER_LSE_MIN_PLAN_DENSITY
        )
        num_q_tiles = (plan.q_len + _LOGICAL_Q_TILE - 1) // _LOGICAL_Q_TILE
        return (
            plan.max_plan_tiles > _TOPK
            and plan.num_plan_tiles >= min_density * num_q_tiles
        )

    return plan.max_plan_tiles > _TOPK and (
        plan.q_len <= _GATHER_LSE_MAX_Q_LEN
        or plan.max_plan_tiles >= _GATHER_LSE_MIN_PLAN_TILES
    )


def m3_indexer_gemm(
    q: torch.Tensor,
    k: torch.Tensor,
    plan: M3ArbitraryMaskPlan,
    score_workspace: torch.Tensor,
    block_sum_workspace: torch.Tensor,
    *,
    lse_temperature: float = 1.0,
    use_fp16_score: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write temperature-domain block maxima and normalized exponential sums."""

    q_len, k_len = _validate_qk(q, k, plan)
    if type(use_fp16_score) is not bool:
        raise TypeError("use_fp16_score must be a Python bool")
    inv_lse_temperature = _inverse_lse_temperature(lse_temperature)
    _validate_stat_workspace(
        score_workspace,
        plan,
        q.device,
        name="score_workspace",
        dtype=torch.float16 if use_fp16_score else torch.float32,
    )
    _validate_stat_workspace(
        block_sum_workspace,
        plan,
        q.device,
        name="block_sum_workspace",
    )
    _compile_gemm(q.device, use_fp16_score=use_fp16_score)(
        q.view(q_len * _NUM_INDEX_HEADS, _HEAD_DIM),
        k.view(k_len, _HEAD_DIM),
        plan.partial_offsets,
        plan.partial_block_indices,
        plan.partial_masks,
        plan.full_offsets,
        plan.full_block_indices,
        score_workspace,
        block_sum_workspace,
        cutlass.Int32(q_len),
        cutlass.Float32(inv_lse_temperature),
    )
    return score_workspace, block_sum_workspace


def m3_indexer_topk(
    score_workspace: torch.Tensor,
    block_sum_workspace: torch.Tensor,
    plan: M3ArbitraryMaskPlan,
    topk_indices: Optional[torch.Tensor] = None,
    selected_lse: Optional[torch.Tensor] = None,
    *,
    block_bases: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    use_fp16_score: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select block ids and return their visible-token FP32 natural-log LSE.

    ``deterministic`` is a compile-time specialization. The deterministic
    variant orders non-local blocks by descending score and then ascending
    block id. The non-deterministic variant leaves non-local output unordered.
    Both variants place the local block in the final valid slot.
    """

    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a Python bool")
    if type(use_fp16_score) is not bool:
        raise TypeError("use_fp16_score must be a Python bool")
    _validate_plan(plan, score_workspace.device)
    _validate_block_bases(block_bases, plan, score_workspace.device)
    rebase_block_ids = block_bases is not None
    small_plan = plan.max_plan_tiles <= _TOPK
    gather_lse = _use_gather_lse(
        plan,
        deterministic=deterministic,
        capability=torch.cuda.get_device_capability(score_workspace.device),
    )
    _validate_stat_workspace(
        score_workspace,
        plan,
        score_workspace.device,
        name="score_workspace",
        dtype=torch.float16 if use_fp16_score else torch.float32,
    )
    _validate_stat_workspace(
        block_sum_workspace,
        plan,
        score_workspace.device,
        name="block_sum_workspace",
    )
    topk_indices = _prepare_topk(
        plan,
        score_workspace.device,
        topk_indices,
    )
    selected_lse = _prepare_selected_lse(
        plan,
        score_workspace.device,
        selected_lse,
    )
    _compile_topk(
        score_workspace.device,
        deterministic=deterministic,
        small_plan=small_plan,
        gather_lse=gather_lse,
        use_fp16_score=use_fp16_score,
        rebase_block_ids=rebase_block_ids,
    )(
        score_workspace,
        block_sum_workspace,
        plan.partial_offsets,
        plan.partial_block_indices,
        plan.full_offsets,
        plan.full_block_indices,
        topk_indices,
        selected_lse,
        plan.local_block_positions,
        block_bases,
        cutlass.Int32(plan.q_len),
    )
    if gather_lse and not use_fp16_score:
        _compile_lse(
            score_workspace.device,
            rebase_block_ids=rebase_block_ids,
        )(
            score_workspace,
            block_sum_workspace,
            plan.partial_block_indices,
            plan.full_block_indices,
            topk_indices,
            selected_lse,
            block_bases,
            cutlass.Int32(plan.q_len),
        )
    return topk_indices, selected_lse


def m3_indexer_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    plan: M3ArbitraryMaskPlan,
    *,
    block_bases: Optional[torch.Tensor] = None,
    score_workspace: Optional[torch.Tensor] = None,
    block_sum_workspace: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    selected_lse: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    lse_temperature: float = 1.0,
    use_fp16_score: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run K1 GEMM followed by the selected TopK/LSE pipeline."""

    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a Python bool")
    if type(use_fp16_score) is not bool:
        raise TypeError("use_fp16_score must be a Python bool")
    _validate_qk(q, k, plan)
    _validate_block_bases(block_bases, plan, q.device)
    if score_workspace is None:
        score_workspace = torch.empty(
            (plan.num_plan_tiles, 2, 128),
            dtype=torch.float16 if use_fp16_score else torch.float32,
            device=q.device,
        )
    else:
        _validate_stat_workspace(
            score_workspace,
            plan,
            q.device,
            name="score_workspace",
            dtype=torch.float16 if use_fp16_score else torch.float32,
        )
    if block_sum_workspace is None:
        block_sum_workspace = torch.empty(
            (plan.num_plan_tiles, 2, 128),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        _validate_stat_workspace(
            block_sum_workspace,
            plan,
            q.device,
            name="block_sum_workspace",
        )
    m3_indexer_gemm(
        q,
        k,
        plan,
        score_workspace,
        block_sum_workspace,
        lse_temperature=lse_temperature,
        use_fp16_score=use_fp16_score,
    )
    return m3_indexer_topk(
        score_workspace,
        block_sum_workspace,
        plan,
        topk_indices,
        selected_lse,
        block_bases=block_bases,
        deterministic=deterministic,
        use_fp16_score=use_fp16_score,
    )


__all__ = [
    "m3_indexer_forward",
    "m3_indexer_gemm",
    "m3_indexer_topk",
]
