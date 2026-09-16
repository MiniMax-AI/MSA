"""Public dense and selected-page NVFP4-to-E4M3 conversion APIs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import jit


_HEAD_DIM = 128
_PACKED_HEAD_DIM = _HEAD_DIM // 2
_SCALE_GROUPS = _HEAD_DIM // 16
_PAGE_SIZE = 128
_MAX_TOPK = 16


def _check_cuda_contiguous(tensor: torch.Tensor, *, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_same_device(
    reference: torch.Tensor,
    tensor: torch.Tensor,
    *,
    name: str,
) -> None:
    if tensor.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}")


def _validate_dense_inputs(
    packed_nvfp4: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor | None,
) -> tuple[int, ...]:
    _check_cuda_contiguous(packed_nvfp4, name="packed_nvfp4")
    _check_cuda_contiguous(scale, name="scale")
    if packed_nvfp4.dtype != torch.uint8:
        raise TypeError("packed_nvfp4 must have dtype torch.uint8")
    if scale.dtype != torch.float8_e4m3fn:
        raise TypeError("scale must have dtype torch.float8_e4m3fn")
    if packed_nvfp4.ndim < 1 or packed_nvfp4.shape[-1] != _PACKED_HEAD_DIM:
        raise ValueError("packed_nvfp4 must have shape [..., 64]")
    if packed_nvfp4.numel() == 0:
        raise ValueError("packed_nvfp4 must contain at least one row")
    expected_scale_shape = (*packed_nvfp4.shape[:-1], _SCALE_GROUPS)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(f"scale must have shape {expected_scale_shape}")
    _check_same_device(packed_nvfp4, scale, name="scale")
    expected_output_shape = (*packed_nvfp4.shape[:-1], _HEAD_DIM)
    if out is not None:
        _check_cuda_contiguous(out, name="out")
        if out.dtype != torch.float8_e4m3fn:
            raise TypeError("out must have dtype torch.float8_e4m3fn")
        if tuple(out.shape) != expected_output_shape:
            raise ValueError(f"out must have shape {expected_output_shape}")
        _check_same_device(packed_nvfp4, out, name="out")
    return expected_output_shape


def dequantize_nvfp4_to_fp8(
    packed_nvfp4: torch.Tensor,
    scale: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert contiguous NVFP4 rows and linear E4M3 block scales to E4M3."""

    output_shape = _validate_dense_inputs(packed_nvfp4, scale, out)
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and not jit.extension_is_loaded(packed_nvfp4.device):
        raise RuntimeError("warm up NVFP4 dequant before CUDA Graph capture")
    if out is None:
        if capturing:
            raise RuntimeError("CUDA Graph capture requires a preallocated out tensor")
        out = torch.empty(
            output_shape,
            dtype=torch.float8_e4m3fn,
            device=packed_nvfp4.device,
        )
    return jit.load_extension(packed_nvfp4.device)._run_dense(packed_nvfp4, scale, out)


@dataclass(frozen=True)
class SparseDequantizedPagedKvCache:
    """Compact selected-page output consumable as an HND FP8 KV cache."""

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int

    @property
    def paged_kv_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache, self.v_cache


@dataclass(frozen=True)
class _SparsePlanState:
    pair_keys: torch.Tensor
    block_tables: torch.Tensor
    sparse_seq_lens: torch.Tensor
    max_seq_len: int
    num_kv_heads: int
    page_size: int
    max_source_page: int
    output_storage_shape: tuple[int, int, int]
    cache_shape: tuple[int, int, int, int]
    cache_stride: tuple[int, int, int, int]
    default_k_storage: torch.Tensor
    default_v_storage: torch.Tensor


def _validate_sparse_metadata(
    topk_indices: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    q_len_per_req: int,
) -> tuple[int, int, int, int]:
    for name, tensor in (
        ("topk_indices", topk_indices),
        ("page_table", page_table),
        ("seq_lens", seq_lens),
    ):
        _check_cuda_contiguous(tensor, name=name)
        _check_same_device(topk_indices, tensor, name=name)
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must have dtype torch.int32")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("plan() must be called outside CUDA Graph capture")
    if page_table.ndim != 2 or page_table.shape[0] <= 0 or page_table.shape[1] <= 0:
        raise ValueError("page_table must have shape [batch, max_pages]")
    batch_size = page_table.shape[0]
    q_len_per_req = int(q_len_per_req)
    if q_len_per_req <= 0:
        raise ValueError("q_len_per_req must be positive")
    if seq_lens.shape != (batch_size,):
        raise ValueError("seq_lens must have shape [batch]")
    if topk_indices.ndim != 3 or topk_indices.shape[0] != batch_size * q_len_per_req:
        raise ValueError(
            "topk_indices must have shape [batch * q_len_per_req, kv_heads, topk]"
        )
    num_kv_heads = topk_indices.shape[1]
    topk = topk_indices.shape[2]
    if num_kv_heads <= 0:
        raise ValueError("topk_indices must contain at least one KV head")
    if topk <= 0 or topk > _MAX_TOPK:
        raise ValueError(f"topk capacity must be in [1, {_MAX_TOPK}]")
    return batch_size, q_len_per_req, num_kv_heads, topk


def _build_sparse_plan(
    topk_indices: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    q_len_per_req: int,
) -> _SparsePlanState:
    batch_size, q_len_per_req, num_kv_heads, topk = _validate_sparse_metadata(
        topk_indices,
        page_table,
        seq_lens,
        q_len_per_req,
    )
    device = topk_indices.device
    total_q = batch_size * q_len_per_req
    query_in_request = torch.arange(
        q_len_per_req, device=device, dtype=torch.int64
    ).repeat(batch_size)
    query_positions = seq_lens.to(torch.int64).repeat_interleave(q_len_per_req)
    query_positions = query_positions - q_len_per_req + query_in_request
    if bool(torch.any(query_positions < 0).item()):
        raise ValueError("every seq_lens entry must include the full query chunk")
    local_pages = torch.div(query_positions, _PAGE_SIZE, rounding_mode="floor")
    if bool(torch.any(local_pages >= page_table.shape[1]).item()):
        raise ValueError("page_table does not cover every query local page")
    valid_count = (local_pages + 1).clamp(max=topk)
    slots = torch.arange(topk, device=device, dtype=torch.int64).reshape(1, 1, topk)
    valid_mask = slots < valid_count.reshape(total_q, 1, 1)
    valid_mask = valid_mask.expand(total_q, num_kv_heads, topk)
    if bool(torch.any(topk_indices[valid_mask] < 0).item()):
        raise ValueError("valid TopK entries must form a non-negative prefix")
    if bool(torch.any(topk_indices[~valid_mask] != -1).item()):
        raise ValueError("TopK padding must be a -1 suffix")
    last_slot = (
        (valid_count - 1).reshape(total_q, 1, 1).expand(total_q, num_kv_heads, 1)
    )
    local_last = topk_indices.gather(2, last_slot).squeeze(2)
    if bool(torch.any(local_last != local_pages.reshape(total_q, 1)).item()):
        raise ValueError("the final valid TopK entry must be the local page")
    if bool(torch.any(topk_indices[valid_mask] >= page_table.shape[1]).item()):
        raise ValueError("valid TopK logical page exceeds page_table width")

    batch_ids = torch.arange(
        batch_size, device=device, dtype=torch.int64
    ).repeat_interleave(q_len_per_req)
    logical_pages = topk_indices.clamp_min(0).to(torch.int64)
    flat_page_slots = (
        batch_ids.reshape(total_q, 1, 1) * page_table.shape[1] + logical_pages
    )
    physical_pages = torch.take(page_table, flat_page_slots).to(torch.int64)
    if bool(torch.any(physical_pages[valid_mask] < 0).item()):
        raise ValueError(
            "page_table must map every valid TopK entry to a physical page"
        )
    heads = torch.arange(num_kv_heads, device=device, dtype=torch.int64).reshape(
        1, num_kv_heads, 1
    )
    pair_keys = physical_pages * num_kv_heads + heads
    unique_pair_keys, inverse = torch.unique(
        pair_keys[valid_mask],
        sorted=True,
        return_inverse=True,
    )
    if unique_pair_keys.numel() == 0:
        raise ValueError("sparse dequant plan must contain at least one valid page")

    remapped = torch.zeros_like(topk_indices)
    remapped[valid_mask] = (
        inverse + num_kv_heads - 1 - heads.expand_as(pair_keys)[valid_mask]
    ).to(torch.int32)
    block_tables = remapped.permute(1, 0, 2).contiguous()
    sparse_lens = (
        (valid_count - 1) * _PAGE_SIZE
        + torch.remainder(query_positions, _PAGE_SIZE)
        + 1
    ).to(torch.int32)
    sparse_seq_lens = (
        sparse_lens.reshape(1, total_q).expand(num_kv_heads, total_q).contiguous()
    )
    pair_count = unique_pair_keys.numel()
    storage_pages = pair_count + 2 * num_kv_heads - 2
    cache_pages = pair_count + num_kv_heads - 1
    output_storage_shape = (storage_pages, _PAGE_SIZE, _HEAD_DIM)
    cache_shape = (cache_pages, num_kv_heads, _PAGE_SIZE, _HEAD_DIM)
    page_stride = _PAGE_SIZE * _HEAD_DIM
    cache_stride = (page_stride, page_stride, _HEAD_DIM, 1)
    default_k_storage = torch.empty(
        output_storage_shape,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    default_v_storage = torch.empty_like(default_k_storage)
    return _SparsePlanState(
        pair_keys=unique_pair_keys,
        block_tables=block_tables,
        sparse_seq_lens=sparse_seq_lens,
        max_seq_len=topk * _PAGE_SIZE,
        num_kv_heads=num_kv_heads,
        page_size=_PAGE_SIZE,
        max_source_page=int(torch.div(unique_pair_keys[-1], num_kv_heads).item()),
        output_storage_shape=output_storage_shape,
        cache_shape=cache_shape,
        cache_stride=cache_stride,
        default_k_storage=default_k_storage,
        default_v_storage=default_v_storage,
    )


class SparsePagedNvfp4ToFp8Wrapper:
    """Plan and run pair-deduplicated sparse paged K/V conversion."""

    def __init__(self) -> None:
        self._plan_state: _SparsePlanState | None = None

    def plan(
        self,
        topk_indices: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        q_len_per_req: int,
    ) -> None:
        """Resolve selected logical pages into unique physical-page/head pairs."""

        self._plan_state = _build_sparse_plan(
            topk_indices,
            page_table,
            seq_lens,
            q_len_per_req=q_len_per_req,
        )

    def run(
        self,
        paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        *,
        kv_cache_sf: tuple[torch.Tensor, torch.Tensor],
        out: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> SparseDequantizedPagedKvCache:
        """Convert the unique selected K/V pairs into compact E4M3 storage."""

        if self._plan_state is None:
            raise RuntimeError("plan() must be called before run()")
        state = self._plan_state
        if not isinstance(paged_kv_cache, (tuple, list)) or len(paged_kv_cache) != 2:
            raise TypeError("paged_kv_cache must be a (packed_k, packed_v) pair")
        if not isinstance(kv_cache_sf, (tuple, list)) or len(kv_cache_sf) != 2:
            raise TypeError("kv_cache_sf must be a (k_scale, v_scale) pair")
        packed_k, packed_v = paged_kv_cache
        k_scale, v_scale = kv_cache_sf
        for name, tensor in (
            ("packed_k", packed_k),
            ("packed_v", packed_v),
            ("k_scale", k_scale),
            ("v_scale", v_scale),
        ):
            _check_cuda_contiguous(tensor, name=name)
            _check_same_device(state.pair_keys, tensor, name=name)
        expected_packed_tail = (state.num_kv_heads, state.page_size, _PACKED_HEAD_DIM)
        expected_scale_tail = (state.num_kv_heads, state.page_size, _SCALE_GROUPS)
        if packed_k.dtype != torch.uint8 or packed_v.dtype != torch.uint8:
            raise TypeError("packed K/V must have dtype torch.uint8")
        if k_scale.dtype != torch.float8_e4m3fn or v_scale.dtype != torch.float8_e4m3fn:
            raise TypeError("K/V scales must have dtype torch.float8_e4m3fn")
        if (
            tuple(packed_k.shape[1:]) != expected_packed_tail
            or packed_v.shape != packed_k.shape
        ):
            raise ValueError(
                f"packed K/V must have shape [pages, {state.num_kv_heads}, "
                f"{state.page_size}, {_PACKED_HEAD_DIM}]"
            )
        if (
            tuple(k_scale.shape[1:]) != expected_scale_tail
            or v_scale.shape != k_scale.shape
        ):
            raise ValueError(
                f"K/V scales must have shape [pages, {state.num_kv_heads}, "
                f"{state.page_size}, {_SCALE_GROUPS}]"
            )
        if (
            packed_k.shape[0] <= state.max_source_page
            or k_scale.shape[0] != packed_k.shape[0]
        ):
            raise ValueError(
                "paged K/V storage does not cover every planned physical page"
            )

        if out is None:
            output_k, output_v = state.default_k_storage, state.default_v_storage
        else:
            if not isinstance(out, (tuple, list)) or len(out) != 2:
                raise TypeError("out must be an (out_k, out_v) pair")
            output_k, output_v = out
        for name, tensor in (("out_k", output_k), ("out_v", output_v)):
            _check_cuda_contiguous(tensor, name=name)
            _check_same_device(state.pair_keys, tensor, name=name)
            if tensor.dtype != torch.float8_e4m3fn:
                raise TypeError(f"{name} must have dtype torch.float8_e4m3fn")
            if tuple(tensor.shape) != state.output_storage_shape:
                raise ValueError(f"{name} must have shape {state.output_storage_shape}")

        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and not jit.extension_is_loaded(state.pair_keys.device):
            raise RuntimeError("warm up sparse NVFP4 dequant before CUDA Graph capture")
        jit.load_extension(state.pair_keys.device)._run_sparse(
            packed_k,
            packed_v,
            k_scale,
            v_scale,
            state.pair_keys,
            output_k,
            output_v,
        )
        k_cache = output_k.as_strided(state.cache_shape, state.cache_stride)
        v_cache = output_v.as_strided(state.cache_shape, state.cache_stride)
        return SparseDequantizedPagedKvCache(
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=state.block_tables,
            seq_lens=state.sparse_seq_lens,
            max_seq_len=state.max_seq_len,
        )


__all__ = [
    "SparseDequantizedPagedKvCache",
    "SparsePagedNvfp4ToFp8Wrapper",
    "dequantize_nvfp4_to_fp8",
]
