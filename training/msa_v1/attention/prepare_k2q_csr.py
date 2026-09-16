"""Fixed MSA v1 q2k-to-k2q CSR preparation on SM100."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch

from msa_v1._common.aot_cache import compile_or_load
from msa_v1._common.compile_utils import compile_with_timing
from msa_v1._common.cute_dsl_utils import to_cute_tensor
from msa_v1.attention.prepare_k2q_csr_cutedsl import SparseK2qCsrPipelineSm100
from msa_v1.attention.prepare_scheduler import (
    SPARSE_SCHEDULE_MODEL,
    SparseAttentionSchedule,
)

_HEAD_KV = 4
_QHEAD_PER_KV = 16
_TOPK = 16
_BLOCK_K = 128
_WARPS = 4
_THREADS = _WARPS * cute.arch.WARP_SIZE
_INT32_BYTES = cutlass.Int32.width // 8
# A KL work owns at most 16 query macros for one physical KV block.
_KL_Q_PER_MACRO = 16
_KL_MACROS_PER_WORK = 16
_KL_TARGET_Q_PER_CTA = _KL_Q_PER_MACRO * _KL_MACROS_PER_WORK
_COMPILE_CACHE: dict[tuple, object] = {}
_PIPELINE = SparseK2qCsrPipelineSm100()


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _as_cute_tensor(tensor: torch.Tensor | None) -> cute.Tensor | None:
    if tensor is None:
        return None
    return to_cute_tensor(tensor, leading_dim=tensor.ndim - 1)


def _get_compiled_pipeline(
    tensors: tuple[torch.Tensor | None, ...],
    scalar_args: tuple[int, ...],
    *,
    reorder_schedule: bool,
    prepare_kl_schedule: bool,
) -> object:
    device = tensors[0].device
    has_fragment_indices = tensors[3] is not None
    key = (
        "sparse_k2q_csr_physical_kl_sm100",
        torch.cuda.get_device_capability(device),
        reorder_schedule,
        prepare_kl_schedule,
        has_fragment_indices,
    )
    if key not in _COMPILE_CACHE:
        _COMPILE_CACHE[key] = compile_or_load(
            key,
            lambda: compile_with_timing(
                _PIPELINE,
                *(_as_cute_tensor(tensor) for tensor in tensors),
                *(cutlass.Int32(value) for value in scalar_args),
                reorder_schedule,
                prepare_kl_schedule,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            ),
            log_prefix="sparse_k2q_csr_prepare",
        )
    return _COMPILE_CACHE[key]


class SparseK2qCsrBuilderSm100:
    """Build fixed-configuration packed-varlen CSR and schedule metadata."""

    def __call__(
        self,
        q2k_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        *,
        total_k: int,
        total_rows: int,
        blk_kv: int = _BLOCK_K,
        max_seqlen_k: int | None = None,
        max_seqlen_q: int | None = None,
        qhead_per_kv: int = _QHEAD_PER_KV,
        fragment_indices: torch.Tensor | None = None,
        prepare_kl_schedule: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        SparseAttentionSchedule,
        SparseAttentionSchedule,
    ]:
        self._validate_inputs(
            q2k_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            fragment_indices,
            blk_kv=blk_kv,
            qhead_per_kv=qhead_per_kv,
        )
        if max_seqlen_q is None or max_seqlen_k is None:
            raise ValueError("max_seqlen_q and max_seqlen_k are required")

        total_k = int(total_k)
        total_rows = int(total_rows)
        max_seqlen_q = int(max_seqlen_q)
        max_seqlen_k = int(max_seqlen_k)
        if min(total_k, total_rows, max_seqlen_q, max_seqlen_k) < 0:
            raise ValueError("sequence sizes must be non-negative")

        batch = int(cu_seqlens_q.numel() - 1)
        total_q = int(q2k_indices.shape[1])
        max_kv_blocks = _ceil_div(max(max_seqlen_k, _BLOCK_K), _BLOCK_K)
        padded_kv_blocks = (total_k + (batch + 1) * _BLOCK_K - 1) // _BLOCK_K
        nnz_upper_bound = total_q * _TOPK
        device = q2k_indices.device
        target_q_per_cta = SPARSE_SCHEDULE_MODEL.balanced_target_q_per_cta(
            total_q=total_q,
            topk=_TOPK,
            head_kv=_HEAD_KV,
            qhead_per_kv=_QHEAD_PER_KV,
            device=device,
        )
        work_capacity = SPARSE_SCHEDULE_MODEL.flat_schedule_capacity(
            total_rows=total_rows,
            total_q=total_q,
            topk=_TOPK,
            head_kv=_HEAD_KV,
            target_q_per_cta=target_q_per_cta,
        )
        kl_work_capacity = SPARSE_SCHEDULE_MODEL.physical_kl_schedule_capacity(
            total_q=total_q,
            topk=_TOPK,
            head_kv=_HEAD_KV,
            padded_kv_blocks=padded_kv_blocks,
            q_per_macro=_KL_Q_PER_MACRO,
            macros_per_work=_KL_MACROS_PER_WORK,
        )
        partitions_per_batch, q_per_cta, q_per_warp = self._partition_config(
            batch=batch,
            max_seqlen_q=max_seqlen_q,
            max_kv_blocks=max_kv_blocks,
            device=device,
        )
        # Dynamic workers process the largest rows first to minimize the final-wave tail.
        reorder_schedule = True
        row_ptr = torch.empty(
            (_HEAD_KV, total_rows + 1),
            dtype=torch.int32,
            device=device,
        )
        q_idx = torch.empty(
            (_HEAD_KV, nnz_upper_bound),
            dtype=torch.int32,
            device=device,
        )

        scheduler_metadata = torch.empty(
            (work_capacity, 6),
            dtype=torch.int32,
            device=device,
        )
        work_count = torch.empty((1,), dtype=torch.int32, device=device)
        dkv_owner_counts = torch.empty(
            (_HEAD_KV, padded_kv_blocks),
            dtype=torch.int32,
            device=device,
        )
        dkv_split_indices = torch.empty(
            (work_capacity,), dtype=torch.int32, device=device
        )
        dkv_split_count = torch.empty((1,), dtype=torch.int32, device=device)
        kl_scheduler_metadata = torch.empty(
            (kl_work_capacity, 4),
            dtype=torch.int32,
            device=device,
        )
        kl_work_count = torch.empty((1,), dtype=torch.int32, device=device)
        kl_physical_row_ptr = torch.empty(
            (_HEAD_KV, padded_kv_blocks + 1),
            dtype=torch.int32,
            device=device,
        )
        kl_physical_q_indices = torch.empty_like(q_idx)
        kl_physical_valid_rows = torch.empty_like(q_idx)
        kl_dki_owner_counts = torch.empty(
            (padded_kv_blocks,), dtype=torch.int32, device=device
        )
        kl_dki_split_indices = torch.empty_like(kl_dki_owner_counts)
        kl_dki_split_count = torch.empty((1,), dtype=torch.int32, device=device)
        qsplit_idx = torch.empty_like(q_idx)
        split_counts = torch.empty(
            (total_q, _HEAD_KV),
            dtype=torch.int32,
            device=device,
        )
        schedule = SparseAttentionSchedule(
            enabled=True,
            scheduler_metadata=scheduler_metadata,
            work_count=work_count,
            qsplit_indices=qsplit_idx,
            split_counts=split_counts,
            dkv_owner_counts=dkv_owner_counts,
            dkv_split_indices=dkv_split_indices,
            dkv_split_count=dkv_split_count,
            target_q_per_cta=target_q_per_cta,
        )
        kl_schedule = SparseAttentionSchedule(
            enabled=prepare_kl_schedule,
            scheduler_metadata=kl_scheduler_metadata,
            work_count=kl_work_count,
            physical_row_ptr=kl_physical_row_ptr,
            physical_q_indices=kl_physical_q_indices,
            physical_valid_rows=kl_physical_valid_rows,
            dki_owner_counts=kl_dki_owner_counts,
            dki_split_indices=kl_dki_split_indices,
            dki_split_count=kl_dki_split_count,
            target_q_per_cta=_KL_TARGET_Q_PER_CTA,
        )

        if total_rows == 0 or total_q == 0:
            row_ptr.zero_()
            q_idx.fill_(-1)
            work_count.zero_()
            dkv_owner_counts.zero_()
            dkv_split_count.zero_()
            kl_work_count.zero_()
            kl_physical_row_ptr.zero_()
            kl_dki_owner_counts.zero_()
            kl_dki_split_count.zero_()
            split_counts.zero_()
            return row_ptr, q_idx, schedule, kl_schedule

        row_map = torch.empty(
            (batch, max_kv_blocks),
            dtype=torch.int32,
            device=device,
        )
        row_coords = torch.empty(
            (total_rows, 2),
            dtype=torch.int32,
            device=device,
        )
        row_counts = torch.empty(
            (_HEAD_KV, batch, max_kv_blocks),
            dtype=torch.int32,
            device=device,
        )
        tile_counts = torch.empty(
            (
                batch * partitions_per_batch * _WARPS,
                _HEAD_KV,
                max_kv_blocks,
            ),
            dtype=torch.int32,
            device=device,
        )
        scheduler_workspace = (
            torch.empty_like(scheduler_metadata) if reorder_schedule else None
        )
        kl_physical_row_counts = torch.empty(
            (_HEAD_KV, padded_kv_blocks),
            dtype=torch.int32,
            device=device,
        )
        kl_physical_k_offsets = torch.empty(
            (padded_kv_blocks,),
            dtype=torch.int32,
            device=device,
        )
        scheduler_unsorted = (
            scheduler_workspace if reorder_schedule else scheduler_metadata
        )
        tensors = (
            q2k_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            fragment_indices,
            row_map,
            row_coords,
            q_idx,
            qsplit_idx,
            split_counts,
            row_counts,
            tile_counts,
            row_ptr,
            scheduler_unsorted,
            scheduler_metadata,
            work_count,
            dkv_owner_counts,
            dkv_split_indices,
            dkv_split_count,
            kl_scheduler_metadata,
            kl_work_count,
            kl_physical_row_counts,
            kl_physical_row_ptr,
            kl_physical_k_offsets,
            kl_physical_q_indices,
            kl_physical_valid_rows,
            kl_dki_owner_counts,
            kl_dki_split_indices,
            kl_dki_split_count,
        )
        scalar_args = (
            batch,
            total_q,
            total_rows,
            max_kv_blocks,
            partitions_per_batch,
            q_per_cta,
            q_per_warp,
            target_q_per_cta,
            work_capacity,
            padded_kv_blocks,
            _KL_TARGET_Q_PER_CTA,
            kl_work_capacity,
        )
        compiled = _get_compiled_pipeline(
            tensors,
            scalar_args,
            reorder_schedule=reorder_schedule,
            prepare_kl_schedule=prepare_kl_schedule,
        )
        with torch.cuda.nvtx.range("SparseK2qCsr_Pipeline"):
            compiled(*tensors, *scalar_args)
        return row_ptr, q_idx, schedule, kl_schedule

    @staticmethod
    def _validate_inputs(
        q2k_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        fragment_indices: torch.Tensor | None,
        *,
        blk_kv: int,
        qhead_per_kv: int,
    ) -> None:
        if q2k_indices.ndim != 3 or tuple(q2k_indices.shape[::2]) != (
            _HEAD_KV,
            _TOPK,
        ):
            raise ValueError("q2k_indices must have shape [4, total_q, 16]")
        if q2k_indices.dtype != torch.int32 or not q2k_indices.is_contiguous():
            raise ValueError("q2k_indices must be contiguous torch.int32")
        if not q2k_indices.is_cuda:
            raise ValueError("q2k_indices must be a CUDA tensor")
        for name, tensor in (
            ("cu_seqlens_q", cu_seqlens_q),
            ("cu_seqlens_k", cu_seqlens_k),
        ):
            if tensor.dtype != torch.int32 or tensor.ndim != 1:
                raise ValueError(f"{name} must be rank-1 torch.int32")
            if not tensor.is_cuda or tensor.device != q2k_indices.device:
                raise ValueError(f"{name} must share the q2k CUDA device")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if cu_seqlens_q.shape != cu_seqlens_k.shape:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must share shape")
        if cu_seqlens_q.numel() < 2:
            raise ValueError("packed varlen inputs require batch >= 1")
        if fragment_indices is not None:
            if fragment_indices.dtype != torch.int32:
                raise TypeError("fragment_indices must be torch.int32")
            if (
                not fragment_indices.is_cuda
                or fragment_indices.device != q2k_indices.device
            ):
                raise ValueError("fragment_indices must share the q2k CUDA device")
            if (
                tuple(fragment_indices.shape) != (cu_seqlens_q.numel() - 1,)
                or not fragment_indices.is_contiguous()
            ):
                raise ValueError("fragment_indices must be contiguous with shape [B]")
        if int(blk_kv) != _BLOCK_K:
            raise ValueError("only block_size=128 is supported")
        if int(qhead_per_kv) != _QHEAD_PER_KV:
            raise ValueError("only qhead_per_kv=16 is supported")

    @staticmethod
    def _partition_config(
        *,
        batch: int,
        max_seqlen_q: int,
        max_kv_blocks: int,
        device: torch.device,
    ) -> tuple[int, int, int]:
        properties = torch.cuda.get_device_properties(device)
        num_sms = properties.multi_processor_count
        packed_per_warp = _ceil_div(max_kv_blocks, 2)
        per_cta_smem_bytes = _WARPS * packed_per_warp * _INT32_BYTES
        smem_ctas_per_sm = max(
            1,
            properties.shared_memory_per_multiprocessor // per_cta_smem_bytes,
        )
        thread_ctas_per_sm = max(
            1,
            properties.max_threads_per_multi_processor // _THREADS,
        )
        resident_ctas = num_sms * min(smem_ctas_per_sm, thread_ctas_per_sm)
        partition_q = max(max_seqlen_q, 1)
        # Keep at least one query per histogram thread while filling every CTA
        # slot permitted by the SMEM and thread limits of the target device.
        target_partitions_per_batch = _ceil_div(resident_ctas, batch)
        max_partitions_for_q = _ceil_div(partition_q, _THREADS)
        partitions_per_batch = max(
            1,
            min(target_partitions_per_batch, max_partitions_for_q, partition_q),
        )
        q_per_cta = _ceil_div(partition_q, partitions_per_batch)
        partitions_per_batch = _ceil_div(partition_q, q_per_cta)
        q_per_warp = _ceil_div(q_per_cta, _WARPS)
        return partitions_per_batch, q_per_cta, q_per_warp


__all__ = ["SparseK2qCsrBuilderSm100"]
