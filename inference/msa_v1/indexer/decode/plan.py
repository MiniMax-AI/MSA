"""Shared, graph-capturable metadata updates for decode indexers."""

import logging
import time
from contextlib import nullcontext

import cutlass
import cutlass.cute as cute  # noqa: PLR0402
import torch
from cutlass.cute.runtime import from_dlpack

from .plan_kernel import DecodeIndexerPlanSm100

try:
    import tvm_ffi
except ImportError:
    tvm_ffi = None

logger = logging.getLogger(__name__)
_COMPILE_CACHE: dict[tuple, object] = {}


class BatchDecodeIndexerPlan:
    """Own one step's logical schedule, shared read-only by layer wrappers.

    Construction and binding happen outside capture. ``update`` reads the bound
    source lengths and launches exactly one kernel on the current stream. Callers
    must order all consumers before the next update, including across streams.
    """

    def __init__(
        self,
        seq_lens_buffer: torch.Tensor,
        *,
        max_pages: int,
        num_index_heads: int = 1,
        query_length: int = 8,
        workspace_buffer: torch.Tensor | None = None,
    ):
        if not seq_lens_buffer.is_cuda or seq_lens_buffer.dtype != torch.int32:
            raise ValueError("seq_lens_buffer must be a CUDA int32 tensor")
        if (
            seq_lens_buffer.ndim != 1
            or not seq_lens_buffer.is_contiguous()
            or not seq_lens_buffer.numel()
        ):
            raise ValueError(
                "seq_lens_buffer must be nonempty, contiguous and one-dimensional"
            )
        if type(query_length) is not int or not 1 <= query_length <= 16:
            raise ValueError("query_length must be an integer in [1, 16]")
        if type(num_index_heads) is not int or num_index_heads not in (1, 2, 4):
            raise ValueError("num_index_heads must be 1, 2, or 4")
        if type(max_pages) is not int or not 1 <= max_pages <= 8192:
            raise ValueError("max_pages must be an integer in [1, 8192]")
        self.device = seq_lens_buffer.device
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("construct the plan outside CUDA Graph capture")
        arch = torch.cuda.get_device_capability(self.device)
        if arch not in ((10, 0), (10, 3)):
            raise ValueError("decode indexer plan requires SM100 or SM103")
        self.batch_size = seq_lens_buffer.numel()
        if self.batch_size * max(max_pages, num_index_heads * query_length) > 2**31 - 1:
            raise ValueError("plan capacity exceeds int32 indexing")
        self.max_pages = max_pages
        self.num_index_heads = num_index_heads
        self.query_length = query_length
        self.num_workers = torch.cuda.get_device_properties(
            self.device
        ).multi_processor_count
        self._source_lengths = seq_lens_buffer
        scheduler_words = self.batch_size + 4 * self.num_workers + 2
        required = self.workspace_size(
            self.batch_size,
            num_index_heads=num_index_heads,
            query_length=query_length,
            device=self.device,
        )
        if workspace_buffer is None:
            workspace_buffer = torch.empty(
                required, dtype=torch.uint8, device=self.device
            )
        if workspace_buffer.numel() < required:
            raise ValueError(f"workspace_buffer is too small: need {required} bytes")
        if (
            workspace_buffer.device != self.device
            or workspace_buffer.dtype != torch.uint8
            or workspace_buffer.ndim != 1
            or not workspace_buffer.is_contiguous()
            or workspace_buffer.data_ptr() % 16
        ):
            raise ValueError(
                "workspace_buffer must be aligned contiguous CUDA uint8 with sufficient capacity"
            )
        self._workspace = workspace_buffer
        self._scheduler_buffer = workspace_buffer[: scheduler_words * 4]
        words = workspace_buffer[:required].view(torch.int32)
        self._scheduler = words[:scheduler_words]
        self._seq_lens = words[scheduler_words : scheduler_words + self.batch_size]
        self._num_valid_pages = words[scheduler_words + self.batch_size :].view(
            num_index_heads, self.batch_size * query_length
        )
        self._tensors = tuple(
            from_dlpack(
                t.detach(), assumed_align=4, enable_tvm_ffi=True
            ).mark_layout_dynamic(leading_dim=0)
            for t in (
                self._source_lengths,
                self._seq_lens,
                self._scheduler,
                self._num_valid_pages.view(-1),
            )
        )
        self._args = (
            cutlass.Int32(query_length),
            cutlass.Int32(num_index_heads),
            cutlass.Int32(max_pages),
            cutlass.Int32(self.num_workers),
        )
        key = ("decode_shared_plan_v1", arch)
        if key not in _COMPILE_CACHE:
            started = time.perf_counter()
            with torch.cuda.device(self.device):
                _COMPILE_CACHE[key] = cute.compile(
                    DecodeIndexerPlanSm100(),
                    *self._tensors,
                    *self._args,
                    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                    options="--enable-tvm-ffi --opt-level 2",
                )
            logger.info("Compiled decode plan in %.3fs", time.perf_counter() - started)
        self._compiled = _COMPILE_CACHE[key]
        self._initialized = False

    @staticmethod
    def workspace_size(batch_size, *, num_index_heads=1, query_length=8, device=None):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if type(num_index_heads) is not int or num_index_heads not in (1, 2, 4):
            raise ValueError("num_index_heads must be 1, 2, or 4")
        if type(query_length) is not int or not 1 <= query_length <= 16:
            raise ValueError("query_length must be in [1, 16]")
        workers = torch.cuda.get_device_properties(device).multi_processor_count
        return (
            2 * batch_size
            + 4 * workers
            + 2
            + num_index_heads * batch_size * query_length
        ) * 4

    def update(self):
        """Refresh metadata without allocation, compilation or host synchronization."""
        context = tvm_ffi.use_torch_stream() if tvm_ffi is not None else nullcontext()
        with torch.cuda.device(self.device), context:
            self._compiled(*self._tensors, *self._args)
        self._initialized = True

    def _require_updated(self):
        if not self._initialized:
            raise RuntimeError("shared_plan.update() must precede run()")

    def _check_binding(self, seq_lens, *, max_pages, num_index_heads, query_length):
        if (
            seq_lens.device != self.device
            or seq_lens.data_ptr() != self._source_lengths.data_ptr()
            or seq_lens.shape != self._source_lengths.shape
            or max_pages != self.max_pages
            or num_index_heads != self.num_index_heads
            or query_length != self.query_length
        ):
            raise ValueError("shared_plan metadata binding does not match this wrapper")


__all__ = ["BatchDecodeIndexerPlan"]
