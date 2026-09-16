"""Private FlashInfer adapter for Q8KV8 sparse decode."""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import torch

_HEAD_DIM = 128
_PAGE_SIZE = 128
_DEFAULT_WORKSPACE_BYTES = 256 * 1024 * 1024
logger = logging.getLogger(__name__)


def check_alignment(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")


@dataclass(frozen=True)
class FlashInferBackend:
    run: Callable
    version: str


@lru_cache(maxsize=1)
def load_backend() -> FlashInferBackend:
    started = time.perf_counter()
    try:
        import flashinfer
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    except ImportError as error:
        raise RuntimeError(
            "Q8KV8 decode requires an external FlashInfer installation with the "
            "TRTLLM-GEN block-sparse backend"
        ) from error

    required_parameters = {
        "query",
        "kv_cache",
        "workspace_buffer",
        "block_tables",
        "seq_lens",
        "q_len_per_req",
        "multi_ctas_kv_counter_buffer",
        "enable_block_sparse_attention",
    }
    actual_parameters = set(
        inspect.signature(trtllm_batch_decode_with_kv_cache).parameters
    )
    missing = sorted(required_parameters - actual_parameters)
    if missing:
        raise RuntimeError(
            "installed FlashInfer does not provide the required Q8KV8 block-sparse "
            f"decode API parameters: {missing}"
        )
    logger.info("Loaded FlashInfer in %.3fs", time.perf_counter() - started)
    return FlashInferBackend(
        run=trtllm_batch_decode_with_kv_cache,
        version=str(getattr(flashinfer, "__version__", "unknown")),
    )


def _check_cuda_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")


def _check_cuda_contiguous(tensor: torch.Tensor, *, name: str) -> None:
    _check_cuda_tensor(tensor, name=name)
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


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class FlashInferPlan:
    block_tables: torch.Tensor
    sparse_seq_lens: torch.Tensor
    max_sparse_seq_len: int
    max_source_page: int
    batch_size: int
    q_len_per_req: int
    total_q: int
    num_q_heads: int
    num_kv_heads: int
    sm_scale: float
    workspace: torch.Tensor
    counter: torch.Tensor
    out: torch.Tensor


def _build_sparse_metadata(
    topk_indices: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    q_len_per_req: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    batch_size = page_table.shape[0]
    total_q = batch_size * q_len_per_req
    topk = topk_indices.shape[2]
    device = topk_indices.device
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
    if bool(
        torch.any(
            topk_indices.gather(2, last_slot).squeeze(2)
            != local_pages.reshape(total_q, 1)
        ).item()
    ):
        raise ValueError("the final valid TopK entry must be the local page")
    if bool(torch.any(topk_indices[valid_mask] >= page_table.shape[1]).item()):
        raise ValueError("valid TopK logical page exceeds page_table width")

    history_mask = slots < (valid_count - 1).reshape(total_q, 1, 1)
    history_mask = history_mask.expand_as(valid_mask)
    if bool(
        torch.any(history_mask & (topk_indices >= local_pages.reshape(-1, 1, 1))).item()
    ):
        raise ValueError("historical TopK pages must precede the local page")
    pairs = slots.unsqueeze(-1) < slots.unsqueeze(-2)
    duplicates = topk_indices.unsqueeze(-1) == topk_indices.unsqueeze(-2)
    valid_pairs = valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(-2)
    if bool(torch.any(duplicates & valid_pairs & pairs).item()):
        raise ValueError("valid TopK pages must be unique")

    batch_ids = torch.arange(
        batch_size, device=device, dtype=torch.int64
    ).repeat_interleave(q_len_per_req)
    logical_pages = topk_indices.clamp_min(0).to(torch.int64)
    flat_page_slots = (
        batch_ids.reshape(total_q, 1, 1) * page_table.shape[1] + logical_pages
    )
    physical_pages = torch.take(page_table, flat_page_slots)
    if bool(torch.any(physical_pages[valid_mask] < 0).item()):
        raise ValueError(
            "page_table must map every valid TopK entry to a physical page"
        )
    physical_pages = torch.where(valid_mask, physical_pages, 0)
    block_tables = physical_pages.permute(1, 0, 2).to(torch.int32).contiguous()
    sparse_lens = (
        (valid_count - 1) * _PAGE_SIZE
        + torch.remainder(query_positions, _PAGE_SIZE)
        + 1
    ).to(torch.int32)
    sparse_seq_lens = (
        sparse_lens.reshape(1, total_q).expand(num_kv_heads, total_q).contiguous()
    )
    return block_tables, sparse_seq_lens, int(physical_pages[valid_mask].max().item())


def make_flashinfer_plan(
    topk_indices: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    q_len_per_req: int,
    num_q_heads: int,
    num_kv_heads: int,
    sm_scale: float,
    workspace_buffer: torch.Tensor | None = None,
) -> FlashInferPlan:
    """Validate sparse rows and allocate state owned only by the caller."""
    if torch.cuda.get_device_capability(topk_indices.device) not in {(10, 0), (10, 3)}:
        raise ValueError("FlashInfer sparse decode requires SM100 or SM103")
    load_backend()
    batch_size = page_table.shape[0]
    scale = sm_scale
    block_tables, sparse_seq_lens, max_source_page = _build_sparse_metadata(
        topk_indices,
        page_table,
        seq_lens,
        q_len_per_req=q_len_per_req,
        num_kv_heads=num_kv_heads,
    )
    workspace = workspace_buffer
    if workspace is None:
        workspace = torch.empty(
            _DEFAULT_WORKSPACE_BYTES,
            dtype=torch.uint8,
            device=topk_indices.device,
        )
    else:
        _check_cuda_contiguous(workspace, name="workspace_buffer")
        _check_same_device(topk_indices, workspace, name="workspace_buffer")
        check_alignment(workspace, name="workspace_buffer")
        if workspace.numel() == 0:
            raise ValueError("workspace_buffer must not be empty")
        if workspace.dtype != torch.uint8:
            raise TypeError("workspace_buffer must have dtype torch.uint8")
    total_q = batch_size * q_len_per_req
    sm_count = torch.cuda.get_device_properties(
        topk_indices.device
    ).multi_processor_count
    counter_bytes = _round_up(max(total_q * num_q_heads, sm_count), 8) * 4
    counter = torch.zeros(counter_bytes, dtype=torch.uint8, device=topk_indices.device)
    return FlashInferPlan(
        block_tables=block_tables,
        sparse_seq_lens=sparse_seq_lens,
        max_sparse_seq_len=topk_indices.shape[2] * _PAGE_SIZE,
        max_source_page=max_source_page,
        batch_size=batch_size,
        q_len_per_req=q_len_per_req,
        total_q=total_q,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        sm_scale=scale,
        workspace=workspace,
        counter=counter,
        out=torch.empty(
            (total_q, num_q_heads, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=topk_indices.device,
        ),
    )


def run_flashinfer(
    state: FlashInferPlan,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Keep each query's sparse pages and causal length independent."""
    for name, tensor in (
        ("q", q),
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("out", out),
    ):
        check_alignment(tensor, name=name)
    if k_cache.shape[0] <= state.max_source_page:
        raise ValueError("KV cache does not cover every planned physical page")
    return load_backend().run(
        q,
        (k_cache, v_cache),
        state.workspace,
        state.block_tables,
        state.sparse_seq_lens,
        state.max_sparse_seq_len,
        bmm1_scale=state.sm_scale,
        bmm2_scale=1.0,
        out=out,
        out_dtype=torch.bfloat16,
        kv_layout="HND",
        backend="trtllm-gen",
        q_len_per_req=1,
        enable_pdl=True,
        multi_ctas_kv_counter_buffer=state.counter,
        enable_block_sparse_attention=True,
    )
