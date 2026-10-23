"""FlashInfer-style wrapper for Q8KV4 paged sparse prefill."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

import torch

from inference.msa_v1.attention.prefill._common.combine import LocalityHints, combine
from inference.msa_v1.attention.prefill._common.metadata import (
    AttentionMetadata,
    prepare,
)
from inference.msa_v1.attention.prefill.q8kv4.jit import load_extension

logger = logging.getLogger(__name__)

_HEAD_DIM = 128
_Q_HEADS_PER_KV = 16
_PAGE_SIZE = 128
_TOPK = 16
# Set to 0 to keep the split workspace in interleaved device memory on a GPU with several memory
# locality domains, the layout single-domain GPUs use (for A/B measurements).
LOCALITY_DOMAINS_ENV = "MM_SPARSE_PREFILL_Q8KV4_LOCALITY_DOMAINS"


def _check_cuda_contiguous(tensor: torch.Tensor, *, name: str) -> None:
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


def _locality_domains_disabled() -> bool:
    value = os.environ.get(LOCALITY_DOMAINS_ENV, "").strip().lower() or "1"
    if value not in ("0", "1", "false", "true"):
        raise ValueError(f"{LOCALITY_DOMAINS_ENV} must be 0 or 1, got {value!r}")
    return value in ("0", "false")


@dataclass(frozen=True)
class _DeviceLocality:
    """The memory locality domains of one device: immutable host metadata, no device tensors."""

    num_domains: int
    page_bytes: int
    sm_domains: tuple[int, ...]


# Probed once per device and process; holds no CUDA tensors, device pointers or workload state.
_device_locality: dict[int, _DeviceLocality | None] = {}


def _device_locality_domains(device: torch.device) -> _DeviceLocality | None:
    """The locality domains of ``device``, probed once per process; None when it has one."""

    if _locality_domains_disabled():
        return None
    index = device.index
    if index not in _device_locality:
        extension = load_extension(device)
        locality = None
        num_domains = int(extension.locality_domain_count(index))
        if num_domains > 1:
            try:
                locality = _DeviceLocality(
                    num_domains=num_domains,
                    page_bytes=int(extension.locality_page_bytes(index)),
                    sm_domains=tuple(extension.sm_locality_domains(index)),
                )
            except RuntimeError as error:
                # The probe needs green contexts, which some profilers' driver shims refuse;
                # interleaved memory serves every domain then.
                logger.warning(
                    "Q8KV4 prefill: locality-domain probe failed, the split workspace stays "
                    "interleaved: %s",
                    error,
                )
        if locality is not None:
            logger.info(
                "Q8KV4 prefill: device %d has %d memory locality domains, the split workspace "
                "is striped over them",
                index,
                locality.num_domains,
            )
        _device_locality[index] = locality
    return _device_locality[index]


class PrefillWorkspace:
    """Split workspace of the Q8KV4 prefill: the partial outputs and partial log-sum-exps the
    forward kernel writes and the combine reads.

    One workspace serves any number of layers and batches that run in order on one stream; two
    streams need two workspaces. On a GPU with several memory locality domains (Rubin, Blackwell)
    the partial pages alternate between the domains and the combine reads every tile from the
    domain of its own SM; elsewhere the partials are ordinary device memory. ``partial_bytes``
    reserves the partial buffer up front (:meth:`required_bytes` of the largest batch); a batch
    that needs more grows it, which on the striped path synchronizes the device first.
    """

    def __init__(self, device: torch.device | str | int, *, partial_bytes: int = 0) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"the prefill workspace lives on a CUDA device, got {device}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self._locality = _device_locality_domains(device)
        self._partials: torch.Tensor | None = None
        self._lse: torch.Tensor | None = None
        self._hints: LocalityHints | None = None
        if self._locality is not None:
            # The kernel's copy of the SM table and its claim counters, exclusive to this workspace.
            self._hints = LocalityHints(
                sm_domains=torch.tensor(
                    self._locality.sm_domains, dtype=torch.int32, device=device
                ),
                tile_counters=torch.zeros(
                    self._locality.num_domains, dtype=torch.int32, device=device
                ),
                page_bytes=self._locality.page_bytes,
            )
        if partial_bytes > 0:
            self._reserve_partials(int(partial_bytes))

    @property
    def num_domains(self) -> int:
        """Memory locality domains the partial buffer is striped over (1: interleaved)."""

        return 1 if self._locality is None else self._locality.num_domains

    def required_bytes(self, total_q: int, num_q_heads: int) -> int:
        """Partial-buffer bytes of a batch: 16 slabs of ``total_q x num_q_heads`` BF16 rows,
        each padded to the domain stripe."""

        return _TOPK * self._slab_bytes(total_q, num_q_heads)

    def _slab_bytes(self, total_q: int, num_q_heads: int) -> int:
        slab = total_q * num_q_heads * _HEAD_DIM * 2
        if self._locality is None:
            return slab
        stripe = self._locality.num_domains * self._locality.page_bytes
        return -(-slab // stripe) * stripe

    def _reserve_partials(self, nbytes: int) -> None:
        if self._locality is None:
            self._partials = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
            return
        stripe = self._locality.num_domains * self._locality.page_bytes
        nbytes = -(-nbytes // stripe) * stripe
        if self._partials is not None:
            # The buffer being replaced may still be read by the previous batch's kernels.
            torch.cuda.synchronize(self.device)
        extension = load_extension(self.device)
        try:
            self._partials = extension.alloc_locality_striped(nbytes, self.device.index)
        except RuntimeError as error:
            # A driver that counts domains but cannot place memory in them: interleaved memory
            # serves every domain, without the steering.
            logger.warning(
                "Q8KV4 prefill: localized allocation failed, the split workspace stays "
                "interleaved: %s",
                error,
            )
            self._locality = None
            self._hints = None
            self._partials = torch.empty(nbytes, dtype=torch.uint8, device=self.device)

    def split_buffers(self, total_q: int, num_q_heads: int):
        """``(o_partial, lse_partial, locality)`` for one batch: ``o_partial`` is
        ``[16, total_q, num_q_heads, 128]`` BF16 with padded split slabs, ``lse_partial``
        ``[16, total_q, num_q_heads]`` FP32, ``locality`` the combine's steering hints or None."""

        slab_bytes = self._slab_bytes(total_q, num_q_heads)
        partial_bytes = _TOPK * slab_bytes
        if self._partials is None or self._partials.numel() < partial_bytes:
            self._reserve_partials(partial_bytes)
            # The reservation may have fallen back to interleaved memory, whose slabs are not padded.
            slab_bytes = self._slab_bytes(total_q, num_q_heads)
            partial_bytes = _TOPK * slab_bytes
        o_partial = (
            self._partials[:partial_bytes]
            .view(torch.bfloat16)
            .as_strided(
                (_TOPK, total_q, num_q_heads, _HEAD_DIM),
                (slab_bytes // 2, num_q_heads * _HEAD_DIM, _HEAD_DIM, 1),
            )
        )
        lse_elements = _TOPK * total_q * num_q_heads
        if self._lse is None or self._lse.numel() < lse_elements:
            self._lse = torch.empty(lse_elements, dtype=torch.float32, device=self.device)
        lse_partial = self._lse[:lse_elements].view(_TOPK, total_q, num_q_heads)
        return o_partial, lse_partial, self._hints


@dataclass(frozen=True)
class _PlanState:
    metadata: AttentionMetadata
    page_table: torch.Tensor
    sm_scale: float
    out: torch.Tensor
    lse: torch.Tensor


class BatchPrefillWithPagedKVCacheWrapper:
    """Stateful Q8KV4 sparse-prefill wrapper with reusable scheduling state.

    ``workspace`` holds the split partials between ``plan`` and ``run``; wrappers that run in
    order on one stream may share one, and a wrapper without one creates its own.
    """

    def __init__(self, workspace: PrefillWorkspace | None = None) -> None:
        self._plan_state: _PlanState | None = None
        self._workspace = workspace

    def plan(
        self,
        topk_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        page_table: torch.Tensor,
        *,
        total_k: int,
        total_rows: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool = True,
        sm_scale: float | None = None,
    ) -> None:
        """Prepare reusable q2k-to-k2q metadata and reserve the split workspace."""

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("plan() must be called outside CUDA Graph capture")
        if not causal:
            raise NotImplementedError("Q8KV4 prefill v1 supports only causal attention")
        _check_cuda_contiguous(page_table, name="page_table")
        _check_same_device(topk_indices, page_table, name="page_table")
        if page_table.dtype != torch.int32:
            raise TypeError("page_table must be torch.int32")
        batch = cu_seqlens_q.numel() - 1
        if page_table.ndim != 2 or page_table.shape[0] != batch:
            raise ValueError("page_table must have shape [batch, max_pages]")
        if page_table.shape[1] <= 0:
            raise ValueError("page_table must contain at least one logical page slot")

        metadata = prepare(
            topk_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            total_k=total_k,
            total_rows=total_rows,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            fragment_indices=None,
            prepare_kl_schedule=False,
        )
        total_q = int(topk_indices.shape[1])
        num_q_heads = int(topk_indices.shape[0]) * _Q_HEADS_PER_KV
        options = {"device": topk_indices.device}
        scale = 1.0 / math.sqrt(_HEAD_DIM) if sm_scale is None else float(sm_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("sm_scale must be finite and positive")
        device = topk_indices.device
        if self._workspace is None or self._workspace.device != device:
            self._workspace = PrefillWorkspace(device)
        # Reserve now so run() never grows the buffer, not even under graph capture.
        self._workspace.split_buffers(total_q, num_q_heads)
        self._plan_state = _PlanState(
            metadata=metadata,
            page_table=page_table,
            sm_scale=scale,
            out=torch.empty(
                (total_q, num_q_heads, _HEAD_DIM),
                dtype=torch.bfloat16,
                **options,
            ),
            lse=torch.empty(
                (total_q, num_q_heads),
                dtype=torch.float32,
                **options,
            ),
        )

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        *,
        kv_cache_sf: tuple[torch.Tensor, torch.Tensor],
        out: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run one layer with the metadata and workspace cached by ``plan``."""

        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        if len(paged_kv_cache) != 2 or len(kv_cache_sf) != 2:
            raise ValueError("paged_kv_cache and kv_cache_sf must be (K, V) pairs")
        k_cache, v_cache = paged_kv_cache
        k_scale, v_scale = kv_cache_sf
        _check_cuda_contiguous(q, name="q")
        _check_same_device(state.page_table, q, name="q")
        # Cache devices, page strides and alignment are validated by the C++ entry;
        # repeating them here would add host time to every layer.

        if q.dtype != torch.float8_e4m3fn:
            raise TypeError("q must be torch.float8_e4m3fn")
        total_q, num_q_heads = state.out.shape[0], state.out.shape[1]
        num_kv_heads = num_q_heads // _Q_HEADS_PER_KV
        if q.shape != state.out.shape:
            raise ValueError(f"q must have shape {tuple(state.out.shape)}")
        for name, tensor in (("k_cache", k_cache), ("v_cache", v_cache)):
            if tensor.dtype != torch.uint8:
                raise TypeError(f"{name} must be torch.uint8")
            if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (
                num_kv_heads,
                _PAGE_SIZE,
                _HEAD_DIM // 2,
            ):
                raise ValueError(
                    f"{name} must have shape [num_pages, {num_kv_heads}, 128, 64]"
                )
        if k_cache.shape != v_cache.shape:
            raise ValueError("k_cache and v_cache must have the same shape")
        for name, tensor in (("k_scale", k_scale), ("v_scale", v_scale)):
            if tensor.dtype != torch.float8_e4m3fn:
                raise TypeError(f"{name} must be torch.float8_e4m3fn")
            if tuple(tensor.shape) != (
                k_cache.shape[0],
                num_kv_heads,
                _PAGE_SIZE,
                _HEAD_DIM // 16,
            ):
                raise ValueError(
                    f"{name} must have shape [num_pages, {num_kv_heads}, 128, 8]"
                )

        out_tensor = state.out if out is None else out
        lse_tensor = state.lse if lse is None else lse
        for name, tensor, dtype, shape in (
            ("out", out_tensor, torch.bfloat16, state.out.shape),
            ("lse", lse_tensor, torch.float32, state.lse.shape),
        ):
            _check_cuda_contiguous(tensor, name=name)
            _check_same_device(q, tensor, name=name)
            if tensor.dtype != dtype or tensor.shape != shape:
                raise ValueError(
                    f"{name} must have dtype {dtype} and shape {tuple(shape)}"
                )

        schedule = state.metadata.schedule
        if (
            schedule.scheduler_metadata is None
            or schedule.work_count is None
            or schedule.qsplit_indices is None
            or schedule.split_counts is None
        ):
            raise RuntimeError("prepare() returned an incomplete forward schedule")

        o_partial, lse_partial, locality = self._workspace.split_buffers(total_q, num_q_heads)
        load_extension(q.device).run(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            state.page_table,
            state.metadata.cu_seqlens_q,
            state.metadata.cu_seqlens_k,
            state.metadata.k2q_row_ptr,
            schedule.qsplit_indices,
            schedule.scheduler_metadata,
            schedule.work_count,
            o_partial,
            lse_partial,
            state.sm_scale,
            None if locality is None else locality.tile_counters,
        )
        combine(
            o_partial,
            lse_partial,
            out_tensor,
            lse_tensor if return_lse or lse is not None else None,
            split_counts=schedule.split_counts,
            use_pdl=True,
            locality=locality,
        )
        if return_lse:
            return out_tensor, lse_tensor
        return out_tensor


__all__ = ["LOCALITY_DOMAINS_ENV", "BatchPrefillWithPagedKVCacheWrapper", "PrefillWorkspace"]
