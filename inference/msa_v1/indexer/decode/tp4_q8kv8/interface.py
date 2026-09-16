"""FlashInfer-style direct-E4M3 paged decode indexer wrapper."""

from __future__ import annotations

import hashlib
import logging
import time
from contextlib import nullcontext
from functools import cache
from pathlib import Path

import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

logger = logging.getLogger(__name__)

try:
    import tvm_ffi
except ImportError:  # pragma: no cover
    tvm_ffi = None

from inference.msa_v1.indexer._common.topk_select import _topk_select
from inference.msa_v1.indexer.decode.tp4_q8kv8.indexer_gemm import (
    DecodeIndexerGemmSm100,
)


_QUERY_LENGTH = 8
_PAGE_SIZE = 128
_HEAD_DIM = 128
_TOP_K = 16
_MAXIMUM_PAGES = 8192
_COMPILE_CACHE: dict[tuple, object] = {}


def _stream_context():
    if tvm_ffi is None:
        return nullcontext()
    return tvm_ffi.use_torch_stream()


def _as_dynamic_tensor(tensor: torch.Tensor, *, assumed_align: int):
    return from_dlpack(
        tensor.detach(),
        assumed_align=assumed_align,
        enable_tvm_ffi=True,
    ).mark_layout_dynamic(leading_dim=tensor.ndim - 1)


@cache
def _source_hash() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted(root.glob("*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()[:16]


def _check_workspace_buffer(workspace_buffer: torch.Tensor) -> None:
    if not workspace_buffer.is_cuda:
        raise ValueError("workspace_buffer must be a CUDA tensor")
    if workspace_buffer.dtype is not torch.uint8:
        raise ValueError("workspace_buffer must have dtype torch.uint8")
    if workspace_buffer.ndim != 1 or not workspace_buffer.is_contiguous():
        raise ValueError("workspace_buffer must be a contiguous one-dimensional buffer")


def _check_metadata(
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    if not page_table.is_cuda or not page_table.is_contiguous():
        raise ValueError("page_table must be a contiguous CUDA tensor")
    if page_table.dtype is not torch.int32:
        raise ValueError("page_table must have dtype torch.int32")
    if page_table.ndim != 2 or page_table.shape[0] <= 0:
        raise ValueError("page_table must have shape [batch, max_pages]")
    if not 0 < page_table.shape[1] <= _MAXIMUM_PAGES:
        raise ValueError(f"max_pages must be in [1, {_MAXIMUM_PAGES}]")
    if not seq_lens.is_cuda or not seq_lens.is_contiguous():
        raise ValueError("seq_lens must be a contiguous CUDA tensor")
    if seq_lens.dtype is not torch.int32:
        raise ValueError("seq_lens must have dtype torch.int32")
    if seq_lens.ndim != 1 or seq_lens.shape[0] != page_table.shape[0]:
        raise ValueError("seq_lens must have shape [batch]")
    if seq_lens.device != page_table.device:
        raise ValueError("seq_lens must be on the same device as page_table")


def _is_capturing(device: torch.device) -> bool:
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


class _BatchDecodeProxyScoreWrapper:
    """Manage metadata and compiled launch state for private proxy scores.

    ``seq_lens`` includes the current eight-token MTP query chunk. For query
    ``q_idx``, only logical pages before
    ``(seq_lens[b] - 8 + q_idx) // 128`` are written. The local page and the
    output suffix remain untouched.

    A wrapper instance is not safe for concurrent use from multiple streams.
    Use one wrapper per concurrent stream.
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor | None = None,
        *,
        use_cuda_graph: bool = False,
        page_table_buffer: torch.Tensor | None = None,
        seq_lens_buffer: torch.Tensor | None = None,
    ) -> None:
        if workspace_buffer is not None:
            _check_workspace_buffer(workspace_buffer)
        if use_cuda_graph and (page_table_buffer is None or seq_lens_buffer is None):
            raise ValueError(
                "CUDA Graph mode requires page_table_buffer and seq_lens_buffer"
            )
        if not use_cuda_graph and (
            page_table_buffer is not None or seq_lens_buffer is not None
        ):
            raise ValueError(
                "page_table_buffer and seq_lens_buffer are only valid when "
                "use_cuda_graph=True"
            )
        if page_table_buffer is not None and seq_lens_buffer is not None:
            _check_metadata(page_table_buffer, seq_lens_buffer)
            if (
                workspace_buffer is not None
                and workspace_buffer.device != page_table_buffer.device
            ):
                raise ValueError(
                    "workspace_buffer must be on the metadata buffer device"
                )

        self._workspace_buffer = workspace_buffer
        self._owns_workspace = workspace_buffer is None
        self._use_cuda_graph = use_cuda_graph
        self._page_table_buffer = page_table_buffer
        self._seq_lens_buffer = seq_lens_buffer
        self._page_table: torch.Tensor | None = None
        self._seq_lens: torch.Tensor | None = None
        self._batch_size: int | None = None
        self._max_pages: int | None = None
        self._device: torch.device | None = None
        self._sm_count: int | None = None

    @staticmethod
    def workspace_size(batch_size: int) -> int:
        """Return the required opaque workspace size in bytes."""

        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return max(16, (batch_size + 1) * 4)

    def plan(
        self,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        """Bind reusable metadata without reading device values on the host."""

        _check_metadata(page_table, seq_lens)
        if torch.cuda.get_device_capability(page_table.device) not in ((10, 0), (10, 3)):
            raise ValueError("Q8KV8 decode indexer requires SM100 or SM103")
        if _is_capturing(page_table.device):
            raise RuntimeError("plan() must be called outside CUDA Graph capture")

        if self._use_cuda_graph:
            assert self._page_table_buffer is not None
            assert self._seq_lens_buffer is not None
            if page_table.shape != self._page_table_buffer.shape:
                raise ValueError(
                    "CUDA Graph mode fixes page_table shape for the wrapper lifetime"
                )
            if seq_lens.shape != self._seq_lens_buffer.shape:
                raise ValueError(
                    "CUDA Graph mode fixes seq_lens shape for the wrapper lifetime"
                )
            if page_table.device != self._page_table_buffer.device:
                raise ValueError(
                    "page_table must be on the fixed metadata buffer device"
                )
            if page_table.data_ptr() != self._page_table_buffer.data_ptr():
                self._page_table_buffer.copy_(page_table, non_blocking=True)
            if seq_lens.data_ptr() != self._seq_lens_buffer.data_ptr():
                self._seq_lens_buffer.copy_(seq_lens, non_blocking=True)
            planned_page_table = self._page_table_buffer
            planned_seq_lens = self._seq_lens_buffer
        else:
            planned_page_table = page_table
            planned_seq_lens = seq_lens

        required_bytes = self.workspace_size(page_table.shape[0])
        if self._workspace_buffer is None:
            self._workspace_buffer = torch.empty(
                required_bytes,
                dtype=torch.uint8,
                device=page_table.device,
            )
        elif self._workspace_buffer.device != page_table.device:
            if not self._owns_workspace:
                raise ValueError(
                    "workspace_buffer must be on the same device as metadata"
                )
            self._workspace_buffer = torch.empty(
                required_bytes,
                dtype=torch.uint8,
                device=page_table.device,
            )
        elif self._workspace_buffer.numel() < required_bytes:
            if not self._owns_workspace:
                raise ValueError(
                    f"workspace_buffer is too small: need {required_bytes} "
                    f"bytes, got {self._workspace_buffer.numel()}"
                )
            self._workspace_buffer = torch.empty(
                required_bytes,
                dtype=torch.uint8,
                device=page_table.device,
            )

        scheduler = self._workspace_buffer[
            : (page_table.shape[0] + 1) * 4
        ].view(torch.int32)
        history_pages = torch.div(
            planned_seq_lens - 1,
            _PAGE_SIZE,
            rounding_mode="floor",
        ).clamp_(min=0, max=page_table.shape[1])
        scheduler.zero_()
        torch.cumsum(history_pages, dim=0, out=scheduler[1:])

        device_index = page_table.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        self._page_table = planned_page_table
        self._seq_lens = planned_seq_lens
        self._batch_size = page_table.shape[0]
        self._max_pages = page_table.shape[1]
        self._device = page_table.device
        self._sm_count = torch.cuda.get_device_properties(
            device_index
        ).multi_processor_count

    def _compile_and_bind(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        out: torch.Tensor,
    ) -> tuple[object, tuple]:
        assert self._page_table is not None
        assert self._seq_lens is not None
        assert self._batch_size is not None
        assert self._max_pages is not None
        assert self._device is not None
        assert self._sm_count is not None
        assert self._workspace_buffer is not None

        q_cute = _as_dynamic_tensor(q, assumed_align=16)
        k_cute = _as_dynamic_tensor(k_cache, assumed_align=16)
        page_table_cute = _as_dynamic_tensor(
            self._page_table,
            assumed_align=4,
        )
        seq_lens_cute = _as_dynamic_tensor(self._seq_lens, assumed_align=4)
        out_cute = _as_dynamic_tensor(out, assumed_align=16)
        scheduler_cute = _as_dynamic_tensor(
            self._workspace_buffer,
            assumed_align=16,
        )
        cute_tensors = (
            q_cute,
            k_cute,
            page_table_cute,
            seq_lens_cute,
            out_cute,
            scheduler_cute,
        )

        capability = torch.cuda.get_device_capability(self._device)
        compile_key = (
            "msa_v1_indexer_decode_tp4_q8kv8",
            _source_hash(),
            capability,
            self._batch_size,
            self._max_pages,
            k_cache.shape[0],
            self._sm_count,
        )
        if compile_key not in _COMPILE_CACHE:
            kernel = DecodeIndexerGemmSm100(
                batch_size=self._batch_size,
                max_pages=self._max_pages,
                sm_count=self._sm_count,
            )
            fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
            started_at = time.time()
            _COMPILE_CACHE[compile_key] = cute.compile(
                kernel,
                *cute_tensors,
                fake_stream,
                options="--enable-tvm-ffi --opt-level 2",
            )
            logger.info("Compiled in %.1fs", time.time() - started_at)
        return _COMPILE_CACHE[compile_key], cute_tensors

    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one layer using the metadata prepared by :meth:`plan`."""

        if (
            self._page_table is None
            or self._seq_lens is None
            or self._workspace_buffer is None
            or self._batch_size is None
            or self._max_pages is None
            or self._device is None
            or self._sm_count is None
        ):
            raise RuntimeError("plan() must be called before run()")
        if not q.is_cuda or not q.is_contiguous():
            raise ValueError("q must be a contiguous CUDA tensor")
        if q.dtype is not torch.float8_e4m3fn:
            raise RuntimeError("q must have dtype torch.float8_e4m3fn")
        if q.shape != (self._batch_size, _QUERY_LENGTH, _HEAD_DIM):
            raise ValueError("q must have shape [batch, 8, 128]")
        if q.device != self._device:
            raise ValueError("q must be on the planned metadata device")
        if not k_cache.is_cuda or not k_cache.is_contiguous():
            raise ValueError("k_cache must be a contiguous CUDA tensor")
        if k_cache.dtype is not torch.float8_e4m3fn:
            raise RuntimeError("k_cache must have dtype torch.float8_e4m3fn")
        if (
            k_cache.ndim != 3
            or k_cache.shape[0] <= 0
            or k_cache.shape[1:] != (_PAGE_SIZE, _HEAD_DIM)
        ):
            raise ValueError("k_cache must have shape [physical_pages, 128, 128]")
        if k_cache.device != self._device:
            raise ValueError("k_cache must be on the planned metadata device")

        if out is None:
            if _is_capturing(q.device):
                raise RuntimeError(
                    "CUDA Graph capture requires a preallocated out tensor"
                )
            out = torch.empty(
                (self._batch_size, _QUERY_LENGTH, self._max_pages),
                dtype=torch.float32,
                device=q.device,
            )
        else:
            if not out.is_cuda or not out.is_contiguous():
                raise ValueError("out must be a contiguous CUDA tensor")
            if out.dtype is not torch.float32:
                raise ValueError("out must have dtype torch.float32")
            if out.shape != (
                self._batch_size,
                _QUERY_LENGTH,
                self._max_pages,
            ):
                raise ValueError("out must have shape [batch, 8, max_pages]")
            if out.device != self._device:
                raise ValueError("out must be on the planned metadata device")

        compiled, cute_tensors = self._compile_and_bind(q, k_cache, out)
        with _stream_context():
            compiled(*cute_tensors)
        return out


class BatchDecodeIndexerWithPagedKVCacheWrapper:
    """Compose direct-E4M3 proxy scores with forced-tail TopK selection."""

    def __init__(
        self,
        workspace_buffer: torch.Tensor | None = None,
        *,
        use_cuda_graph: bool = False,
        page_table_buffer: torch.Tensor | None = None,
        seq_lens_buffer: torch.Tensor | None = None,
    ) -> None:
        self._proxy_score = _BatchDecodeProxyScoreWrapper(
            workspace_buffer,
            use_cuda_graph=use_cuda_graph,
            page_table_buffer=page_table_buffer,
            seq_lens_buffer=seq_lens_buffer,
        )
        self._proxy_scores: torch.Tensor | None = None
        self._num_valid_pages: torch.Tensor | None = None
        self._topk_indices: torch.Tensor | None = None

    @staticmethod
    def workspace_size(batch_size: int) -> int:
        """Return the opaque scheduler workspace size in bytes."""

        return _BatchDecodeProxyScoreWrapper.workspace_size(batch_size)

    def plan(
        self,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        """Prepare reusable metadata, scores, lengths, and TopK output."""

        self._proxy_score.plan(page_table, seq_lens)
        planned_seq_lens = self._proxy_score._seq_lens
        assert planned_seq_lens is not None
        batch_size = page_table.shape[0]
        max_pages = page_table.shape[1]
        options = {"device": page_table.device}

        score_shape = (batch_size, _QUERY_LENGTH, max_pages)
        if (
            self._proxy_scores is None
            or self._proxy_scores.shape != score_shape
            or self._proxy_scores.device != page_table.device
        ):
            self._proxy_scores = torch.empty(
                score_shape,
                dtype=torch.float32,
                **options,
            )
        row_count = batch_size * _QUERY_LENGTH
        if (
            self._num_valid_pages is None
            or self._num_valid_pages.shape != (row_count,)
            or self._num_valid_pages.device != page_table.device
        ):
            self._num_valid_pages = torch.empty(
                (row_count,),
                dtype=torch.int32,
                **options,
            )
        if (
            self._topk_indices is None
            or self._topk_indices.shape != (row_count, _TOP_K)
            or self._topk_indices.device != page_table.device
        ):
            self._topk_indices = torch.empty(
                (row_count, _TOP_K),
                dtype=torch.int32,
                **options,
            )

        query_indices = torch.arange(
            _QUERY_LENGTH,
            dtype=torch.int32,
            device=page_table.device,
        )
        num_valid_pages = torch.div(
            planned_seq_lens[:, None] - _QUERY_LENGTH + query_indices[None, :],
            _PAGE_SIZE,
            rounding_mode="floor",
        ).add_(1)
        num_valid_pages.clamp_(min=1, max=max_pages)
        self._num_valid_pages.copy_(
            num_valid_pages.reshape(row_count),
            non_blocking=True,
        )

    def run(
        self,
        q: torch.Tensor,
        paged_k_cache: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return forced-tail TopK logical page indices for one layer."""

        if (
            self._proxy_scores is None
            or self._num_valid_pages is None
            or self._topk_indices is None
        ):
            raise RuntimeError("plan() must be called before run()")
        proxy_scores = self._proxy_score.run(
            q,
            paged_k_cache,
            out=self._proxy_scores,
        )
        topk_out = self._topk_indices if out is None else out
        return _topk_select(
            proxy_scores.view(-1, proxy_scores.shape[-1]),
            self._num_valid_pages,
            out=topk_out,
        )


__all__ = ["BatchDecodeIndexerWithPagedKVCacheWrapper"]
