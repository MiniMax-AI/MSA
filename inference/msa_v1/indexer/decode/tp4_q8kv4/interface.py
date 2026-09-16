"""FlashInfer-style Q8KV4 paged decode indexer wrapper."""

from __future__ import annotations

import torch

from inference.msa_v1.indexer._common.topk_select import _topk_select
from inference.msa_v1.indexer.decode.tp4_q8kv4.jit import load_extension

_QUERY_LENGTH = 8
_PAGE_SIZE = 128
_TOP_K = 16
_MAXIMUM_PAGES = 8192


def _check_workspace_buffer(workspace_buffer: torch.Tensor) -> None:
    if not workspace_buffer.is_cuda:
        raise ValueError("workspace_buffer must be a CUDA tensor")
    if workspace_buffer.dtype is not torch.uint8:
        raise ValueError("workspace_buffer must have dtype torch.uint8")
    if workspace_buffer.ndim != 1 or not workspace_buffer.is_contiguous():
        raise ValueError(
            "workspace_buffer must be a contiguous one-dimensional buffer"
        )


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
    """Manage metadata and scheduler state for the private proxy-score stage.

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
        if use_cuda_graph and (
            page_table_buffer is None or seq_lens_buffer is None
        ):
            raise ValueError(
                "CUDA Graph mode requires page_table_buffer and "
                "seq_lens_buffer"
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
        device = torch.device("cuda", torch.cuda.current_device())
        return int(load_extension(device)._workspace_size(batch_size))

    def plan(
        self,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        """Prepare scheduler state for reusable metadata.

        This method must run outside CUDA Graph capture. In CUDA Graph mode,
        metadata is copied asynchronously into the fixed constructor buffers.
        """

        _check_metadata(page_table, seq_lens)
        if _is_capturing(page_table.device):
            raise RuntimeError("plan() must be called outside CUDA Graph capture")

        if self._use_cuda_graph:
            assert self._page_table_buffer is not None
            assert self._seq_lens_buffer is not None
            if page_table.shape != self._page_table_buffer.shape:
                raise ValueError(
                    "CUDA Graph mode fixes page_table shape for the wrapper "
                    "lifetime"
                )
            if seq_lens.shape != self._seq_lens_buffer.shape:
                raise ValueError(
                    "CUDA Graph mode fixes seq_lens shape for the wrapper "
                    "lifetime"
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

        extension = load_extension(page_table.device)
        required_bytes = int(extension._workspace_size(page_table.shape[0]))
        if self._workspace_buffer is None:
            self._workspace_buffer = torch.empty(
                required_bytes, dtype=torch.uint8, device=page_table.device
            )
        elif self._workspace_buffer.device != page_table.device:
            if not self._owns_workspace:
                raise ValueError(
                    "workspace_buffer must be on the same device as metadata"
                )
            self._workspace_buffer = torch.empty(
                required_bytes, dtype=torch.uint8, device=page_table.device
            )
        elif self._workspace_buffer.numel() < required_bytes:
            if not self._owns_workspace:
                raise ValueError(
                    f"workspace_buffer is too small: need {required_bytes} "
                    f"bytes, got {self._workspace_buffer.numel()}"
                )
            self._workspace_buffer = torch.empty(
                required_bytes, dtype=torch.uint8, device=page_table.device
            )

        sm_count = int(
            extension._plan(
                planned_page_table,
                planned_seq_lens,
                self._workspace_buffer,
            )
        )
        self._page_table = planned_page_table
        self._seq_lens = planned_seq_lens
        self._batch_size = page_table.shape[0]
        self._max_pages = page_table.shape[1]
        self._device = page_table.device
        self._sm_count = sm_count

    def run(
        self,
        q: torch.Tensor,
        packed_k_cache: torch.Tensor,
        k_scale: torch.Tensor,
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
        if not q.is_cuda:
            raise ValueError("q must be a CUDA tensor")
        if q.device != self._device:
            raise ValueError("q must be on the planned metadata device")
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
        return load_extension(q.device)._run(
            q,
            packed_k_cache,
            k_scale,
            self._page_table,
            self._seq_lens,
            self._workspace_buffer,
            self._sm_count,
            out,
        )


class BatchDecodeIndexerWithPagedKVCacheWrapper:
    """Compose Q8KV4 proxy scores with the forced-tail TopK selection."""

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
        k_scale: torch.Tensor,
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
            k_scale,
            out=self._proxy_scores,
        )
        topk_out = self._topk_indices if out is None else out
        return _topk_select(
            proxy_scores.view(-1, proxy_scores.shape[-1]),
            self._num_valid_pages,
            out=topk_out,
        )


__all__ = ["BatchDecodeIndexerWithPagedKVCacheWrapper"]
