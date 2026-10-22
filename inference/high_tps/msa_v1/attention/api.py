"""Standalone Attention kernel API."""

import math

import torch

from ..common.backend import _get_cpp
from ..common.communication import Tp4AllReduceWorkspace


def create_attention_plan(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    *,
    topk: int = 16,
    device=None,
):
    device_int = None
    if device is not None:
        device_int = (
            torch.device(device).index if not isinstance(device, int) else device
        )

    return _get_cpp()._m3_decode_plan(
        qo_segment_lens,
        kv_segment_lens,
        topk=topk,
        device=device_int,
    )


def m3_decode_mega_atten(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan_info,
    *,
    kv_indices: torch.Tensor,
    k_scale_tensor: torch.Tensor,
    v_scale_tensor: torch.Tensor,
    output_projection_weight: torch.Tensor,
    output_projection_weight_sf: torch.Tensor,
    tp_workspace: Tp4AllReduceWorkspace,
    residual_in: torch.Tensor,
    rms_weight: torch.Tensor,
    indexer_q: torch.Tensor,
    indexer_k: torch.Tensor,
    indexer_k_scale: torch.Tensor,
    indexer_page_table: torch.Tensor,
    attention_projection_input_norm_weight: torch.Tensor,
    attention_projection_weight: torch.Tensor,
    attention_projection_weight_sf: torch.Tensor,
    attention_projection_workspace_data: torch.Tensor,
    attention_projection_workspace_scale: torch.Tensor,
    attention_projection_grid_epoch: torch.Tensor,
    attention_projection_special_partial: torch.Tensor,
    attention_projection_special_epochs: torch.Tensor,
    attention_projection_positions: torch.Tensor,
    attention_projection_slot_mapping: torch.Tensor,
    attention_projection_q_norm_weight: torch.Tensor,
    attention_projection_k_norm_weight: torch.Tensor,
    attention_projection_qi_norm_weight: torch.Tensor,
    attention_projection_ki_norm_weight: torch.Tensor,
    attention_projection_cos_sin_cache: torch.Tensor,
    attention_residual: torch.Tensor | None = None,
    post_attention_norm: torch.Tensor | None = None,
    kv_block_indexes: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_sf: torch.Tensor | None = None,
    topk: int = 16,
    sm_scale: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    o_scale: float | None = None,
    rms_eps: float = 1e-6,
    attention_projection_input_norm_eps: float = 0.0,
    attention_projection_post_norm_eps: float = 0.0,
    attention_projection_ki_dequant_scale: float = 1.0,
    wait_timeout_s: float = 0.0,
    enable_placement: bool = False,
):
    """Run Projection, Indexer, Attention, OProj, and post-attention norm.

    ``wait_timeout_s`` sets a diagnostic deadline for waits on another CTA or
    TP rank. Zero disables the deadline; a nonzero value selects a diagnostic
    binary that traps when the deadline expires. A trap invalidates the CUDA context.
    ``enable_placement=False`` requires canonical hardware cluster placement;
    enable it to adapt to non-canonical placement. The mode is fixed at capture.
    """

    if not isinstance(tp_workspace, Tp4AllReduceWorkspace):
        raise TypeError("tp_workspace must be a Tp4AllReduceWorkspace")
    if not tp_workspace.ready:
        raise RuntimeError("tp_workspace is not ready")
    if tp_workspace.device != q.device.index:
        raise ValueError("tp_workspace and q must use the same CUDA device")
    if tp_workspace.size < 562688:
        raise ValueError("MegaAttention requires at least 562688 workspace bytes")
    if topk != 16:
        raise ValueError(f"m3_decode_mega_atten requires topK=16, got {topk}")
    if not math.isfinite(wait_timeout_s) or wait_timeout_s < 0 or wait_timeout_s * 1e9 >= 2**63:
        raise ValueError("wait_timeout_s must be finite, non-negative and fit int64 nanoseconds")
    if type(enable_placement) is not bool:
        raise TypeError("enable_placement must be a bool")
    if torch.cuda.is_current_stream_capturing() and (
        attention_residual is None
        or post_attention_norm is None
        or out is None
        or out_sf is None
    ):
        raise RuntimeError(
            "all MegaAtten outputs must be allocated before CUDA Graph capture"
        )
    hidden_shape = (residual_in.shape[0], residual_in.shape[1])
    if attention_residual is None:
        attention_residual = torch.empty(
            hidden_shape, dtype=torch.bfloat16, device=residual_in.device
        )
    if post_attention_norm is None:
        post_attention_norm = torch.empty(
            hidden_shape, dtype=torch.bfloat16, device=residual_in.device
        )
    return _get_cpp().m3_decode_mega_atten(
        q,
        k,
        v,
        plan_info,
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
        k_scale_tensor=k_scale_tensor.view(torch.uint8),
        v_scale_tensor=v_scale_tensor.view(torch.uint8),
        output_projection_weight=output_projection_weight,
        output_projection_weight_sf=output_projection_weight_sf,
        tp_unicast_ptr=tp_workspace.unicast_ptr,
        tp_multicast_ptr=tp_workspace.multicast_ptr,
        tp_rank=tp_workspace.rank,
        residual_in=residual_in,
        attention_residual=attention_residual,
        post_attention_norm=post_attention_norm,
        rms_weight=rms_weight,
        indexer_q=indexer_q,
        indexer_k=indexer_k,
        indexer_k_scale=indexer_k_scale,
        indexer_page_table=indexer_page_table,
        attention_projection_input_norm_weight=attention_projection_input_norm_weight,
        attention_projection_weight=attention_projection_weight,
        attention_projection_weight_sf=attention_projection_weight_sf,
        attention_projection_workspace_data=attention_projection_workspace_data,
        attention_projection_workspace_scale=attention_projection_workspace_scale,
        attention_projection_grid_epoch=attention_projection_grid_epoch,
        attention_projection_special_partial=attention_projection_special_partial,
        attention_projection_special_epochs=attention_projection_special_epochs,
        attention_projection_positions=attention_projection_positions,
        attention_projection_slot_mapping=attention_projection_slot_mapping,
        attention_projection_q_norm_weight=attention_projection_q_norm_weight,
        attention_projection_k_norm_weight=attention_projection_k_norm_weight,
        attention_projection_qi_norm_weight=attention_projection_qi_norm_weight,
        attention_projection_ki_norm_weight=attention_projection_ki_norm_weight,
        attention_projection_cos_sin_cache=attention_projection_cos_sin_cache,
        attention_projection_input_norm_eps=float(attention_projection_input_norm_eps),
        attention_projection_post_norm_eps=float(attention_projection_post_norm_eps),
        attention_projection_ki_dequant_scale=float(
            attention_projection_ki_dequant_scale
        ),
        topk=topk,
        out=out,
        out_sf=out_sf,
        sm_scale=sm_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        o_scale=o_scale,
        rms_eps=rms_eps,
        wait_timeout_ns=math.ceil(wait_timeout_s * 1e9),
        enable_placement=enable_placement,
    )


_CACHE_PAGE_ROWS = 128
_CACHE_DATA_ROW_BYTES = 64
_CACHE_SCALE_ROW_BYTES = 8
_TMA_MAX_STRIDE_BYTES = 1 << 40


def _validate_cache_component(
    tensor: torch.Tensor,
    name: str,
    pages: int,
    row_bytes: int,
    *,
    allow_3d: bool,
    alignment: int,
) -> None:
    if tensor.element_size() != 1:
        raise ValueError(f"{name} must use one-byte storage (uint8 or E4M3)")
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        row_dim = 2
    elif allow_3d and tensor.ndim == 3:
        row_dim = 1
    else:
        shapes = f"[P, 1, 128, {row_bytes}]" + (
            f" or [P, 128, {row_bytes}]" if allow_3d else ""
        )
        raise ValueError(f"{name} must be {shapes}; got {tuple(tensor.shape)}")
    if (
        tensor.shape[0] != pages
        or tensor.shape[row_dim] != _CACHE_PAGE_ROWS
        or tensor.shape[row_dim + 1] != row_bytes
    ):
        raise ValueError(
            f"{name} must have {pages} pages of [128, {row_bytes}] bytes; "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.stride(row_dim + 1) != 1 or tensor.stride(row_dim) != row_bytes:
        raise ValueError(
            f"{name} rows must be dense inside a page; got strides {tensor.stride()}"
        )
    page_stride = tensor.stride(0)
    dense_page_bytes = _CACHE_PAGE_ROWS * row_bytes
    if (
        page_stride < dense_page_bytes
        or page_stride % alignment != 0
        or page_stride >= _TMA_MAX_STRIDE_BYTES
    ):
        raise ValueError(
            f"{name} page stride must be a multiple of {alignment} bytes in "
            f"[{dense_page_bytes}, 2^40); got {page_stride}"
        )
    if tensor.data_ptr() % alignment != 0:
        raise ValueError(f"{name} must be {alignment}-byte aligned")


def validate_paged_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    indexer_k: torch.Tensor,
    indexer_k_scale: torch.Tensor,
) -> None:
    """Check the page-strided NVFP4 cache layout consumed by the fused decode kernels.

    Every component holds ``P`` physical pages. ``k``/``v``/``indexer_k`` pages are
    ``[128, 64]`` packed E2M1 bytes and ``k_scale``/``v_scale``/``indexer_k_scale``
    pages are ``[128, 8]`` E4M3 bytes; rows are dense inside a page. ``stride(0)``
    is the page stride in bytes: at least the dense page size (8192 B for data,
    1024 B for scales), below 2^40, and a multiple of 16 bytes. ``k`` and ``v``
    are read through the packed-FP4 TMA type, so their page stride and base
    address must be multiples of 32 bytes; every other component needs 16-byte
    alignment. Each component may be a view into a larger per-page slab and may
    use its own page stride. Dense tensors satisfy this contract.
    """

    if not isinstance(k, torch.Tensor) or k.ndim != 4:
        raise ValueError("k must be a 4-D [P, 1, 128, 64] uint8 cache")
    pages = k.shape[0]
    if not k.is_cuda or not 1 <= pages <= (1 << 31) - 1:
        raise ValueError("k must be CUDA storage with 1..INT_MAX physical pages")
    for tensor, name, row_bytes, allow_3d, alignment in (
        (k, "k", _CACHE_DATA_ROW_BYTES, False, 32),
        (v, "v", _CACHE_DATA_ROW_BYTES, False, 32),
        (k_scale, "k_scale", _CACHE_SCALE_ROW_BYTES, False, 16),
        (v_scale, "v_scale", _CACHE_SCALE_ROW_BYTES, False, 16),
        (indexer_k, "indexer_k", _CACHE_DATA_ROW_BYTES, True, 16),
        (indexer_k_scale, "indexer_k_scale", _CACHE_SCALE_ROW_BYTES, True, 16),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        allowed = (torch.uint8, torch.float8_e4m3fn) if "scale" in name else (torch.uint8,)
        if tensor.device != k.device or tensor.dtype not in allowed:
            raise ValueError(f"{name} must use {allowed} storage on {k.device}")
        _validate_cache_component(
            tensor, name, pages, row_bytes, allow_3d=allow_3d, alignment=alignment
        )
