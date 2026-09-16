"""Fixed-configuration CuTe DSL q2k-to-k2q metadata pipeline."""

from __future__ import annotations

import math
from typing import Optional

import cutlass
from cutlass import Int32, const_expr
import cutlass.cute as cute
from cutlass.cute import nvgpu
import cuda.bindings.driver as cuda

from msa_v1.attention.prepare_scheduler import (
    KL_VALID_ROWS_MASK,
    KL_WRITER_RANK_MASK,
    KL_WRITER_RANK_SHIFT,
)

_HEAD_KV = 4
_TOPK = 16
_BLOCK_K = 128
_KL_Q_PER_MACRO = 16
# The histogram assigns one warp to each fixed KV head.
_WARPS = 4
_THREADS = _WARPS * cute.arch.WARP_SIZE
# The row-prefix scan uses the SM100 architectural thread limit. Auxiliary
# row kernels use an eight-warp layout and keep one prefix row per warp.
_SM100_MAX_THREADS_PER_CTA = 1024
_AUXILIARY_WARPS = 8
_PREFIX_THREADS = _SM100_MAX_THREADS_PER_CTA
_KL_SCHEDULE_THREADS = _AUXILIARY_WARPS * cute.arch.WARP_SIZE
_TILE_PREFIX_THREADS = _KL_SCHEDULE_THREADS
_ROWS_PER_PREFIX_CTA = _TILE_PREFIX_THREADS // cute.arch.WARP_SIZE
# A bucket spans at most the accepted 5% scheduling-cost quantization error.
_MAX_SCHEDULE_BUCKET_WIDTH = 0.05
_SCHEDULE_BUCKETS = math.ceil(1.0 / _MAX_SCHEDULE_BUCKET_WIDTH)
# The reorder CTA uses two Int32 matrices (counts and offsets) plus one bucket
# base vector. Derive its maximum row domain from the SM100 per-CTA SMEM limit.
_SM100_SHARED_MEMORY_BYTES = 232448
_INT32_BYTES = Int32.width // 8
_VECTOR_BITS = 128
_VECTOR_BYTES = _VECTOR_BITS // 8
_INT32_VECTOR_ELEMENTS = _VECTOR_BITS // Int32.width
_WARP_THREADS = cute.arch.WARP_SIZE
_REORDER_THREADS = _TILE_PREFIX_THREADS
_REORDER_WARPS = _REORDER_THREADS // _WARP_THREADS
_REORDER_FIXED_WORDS = _SCHEDULE_BUCKETS
_REORDER_WORDS_PER_CHUNK = 2 * _SCHEDULE_BUCKETS
_REORDER_MAX_CHUNKS = (
    _SM100_SHARED_MEMORY_BYTES // _INT32_BYTES - _REORDER_FIXED_WORDS
) // _REORDER_WORDS_PER_CHUNK
_SCHEDULE_REORDER_CAPACITY = _REORDER_MAX_CHUNKS * _WARP_THREADS


class SparseK2qCsrPipelineSm100:
    """Launch the fixed MSA v1 CSR and schedule preparation pipeline."""

    @cute.jit
    def _rows_in_batch(
        self,
        mCuSeqlensK: cute.Tensor,
        batch_idx: Int32,
        mFragmentIndices: Optional[cute.Tensor],
    ) -> Int32:
        start_idx = (
            batch_idx
            if const_expr(mFragmentIndices is None)
            else mFragmentIndices[batch_idx]
        )
        seqlen = mCuSeqlensK[batch_idx + Int32(1)] - mCuSeqlensK[start_idx]
        return (seqlen + Int32(_BLOCK_K - 1)) // Int32(_BLOCK_K)

    @cute.jit
    def _row_linear_for_batch_level(
        self,
        mCuSeqlensK: cute.Tensor,
        batch_idx: Int32,
        level: Int32,
        batch: Int32,
        mFragmentIndices: Optional[cute.Tensor],
    ) -> Int32:
        rows_before = Int32(0)
        active_before = Int32(0)
        for b in cutlass.range(batch, unroll=1):
            rows = self._rows_in_batch(mCuSeqlensK, b, mFragmentIndices)
            rows_before += cutlass.min(rows, level)
            if b < batch_idx and rows > level:
                active_before += Int32(1)
        return rows_before + active_before

    @cute.jit
    def _atomic_inc_packed_i16(
        self,
        sPacked: cute.Tensor,
        row: Int32,
    ) -> Int32:
        word_idx = row // Int32(2)
        shift = (row & Int32(1)) * Int32(16)
        delta = Int32(1) << shift
        old = cute.arch.atomic_add(
            (sPacked.iterator + word_idx).llvm_ptr,
            delta,
            sem="relaxed",
            scope="cta",
        )
        return (old >> shift) & Int32(0xFFFF)

    @cute.jit
    def _read_packed_i16(
        self,
        sPacked: cute.Tensor,
        row: Int32,
    ) -> Int32:
        shift = (row & Int32(1)) * Int32(16)
        return (sPacked[row // Int32(2)] >> shift) & Int32(0xFFFF)

    @cute.jit
    def __call__(
        self,
        mQ2K: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mFragmentIndices: Optional[cute.Tensor],
        mRowMap: cute.Tensor,
        mRowCoords: cute.Tensor,
        mQIdx: cute.Tensor,
        mQSplitIdx: cute.Tensor,
        mSplitCounts: cute.Tensor,
        mRowCounts: cute.Tensor,
        mTileCounts: cute.Tensor,
        mRowPtr: cute.Tensor,
        mSchedulerMetadataUnsorted: cute.Tensor,
        mSchedulerMetadata: cute.Tensor,
        mWorkCount: cute.Tensor,
        mDkvOwnerCounts: cute.Tensor,
        mDkvSplitIndices: cute.Tensor,
        mDkvSplitCount: cute.Tensor,
        mKlSchedulerMetadata: cute.Tensor,
        mKlWorkCount: cute.Tensor,
        mKlPhysicalRowCounts: cute.Tensor,
        mKlPhysicalRowPtr: cute.Tensor,
        mKlPhysicalKOffsets: cute.Tensor,
        mKlPhysicalQIndices: cute.Tensor,
        mKlPhysicalValidRows: cute.Tensor,
        mKlDkiOwnerCounts: cute.Tensor,
        mKlDkiSplitIndices: cute.Tensor,
        mKlDkiSplitCount: cute.Tensor,
        batch: Int32,
        total_q: Int32,
        total_rows: Int32,
        max_kv_blocks: Int32,
        partitions_per_batch: Int32,
        q_per_cta: Int32,
        q_per_warp: Int32,
        target_q_per_cta: Int32,
        work_capacity: Int32,
        padded_kv_blocks: Int32,
        kl_target_q_per_cta: Int32,
        kl_work_capacity: Int32,
        reorder_schedule: cutlass.Constexpr[bool],
        prepare_kl_schedule: cutlass.Constexpr[bool],
        stream: cuda.CUstream = None,
    ):
        num_q_idx = Int32(_HEAD_KV * _TOPK) * total_q
        num_row_counts = Int32(_HEAD_KV) * batch * max_kv_blocks
        num_owner_counts = Int32(_HEAD_KV) * padded_kv_blocks
        fill_copy_atom = cute.make_copy_atom(
            nvgpu.CopyUniversalOp(),
            Int32,
            num_bits_per_copy=128,
        )
        self.init_kernel(
            fill_copy_atom,
            mQIdx,
            mRowCounts,
            mWorkCount,
            mDkvOwnerCounts,
            mDkvSplitCount,
            mKlWorkCount,
            mKlPhysicalRowCounts,
            mKlPhysicalKOffsets,
            mKlDkiOwnerCounts,
            mKlDkiSplitCount,
            num_q_idx,
            num_row_counts,
            cutlass.max(num_owner_counts, padded_kv_blocks),
            padded_kv_blocks,
            prepare_kl_schedule,
        ).launch(
            grid=(
                cute.ceil_div(
                    cutlass.max(
                        cutlass.max(
                            num_q_idx // Int32(_INT32_VECTOR_ELEMENTS),
                            num_row_counts,
                        ),
                        num_owner_counts,
                    ),
                    _KL_SCHEDULE_THREADS,
                ),
                1,
                1,
            ),
            block=(_KL_SCHEDULE_THREADS, 1, 1),
            stream=stream,
        )
        self.histogram_kernel(
            mQ2K,
            mCuSeqlensQ,
            mCuSeqlensK,
            mFragmentIndices,
            mRowMap,
            mRowCoords,
            mRowCounts,
            mTileCounts,
            batch,
            max_kv_blocks,
            partitions_per_batch,
            q_per_cta,
            q_per_warp,
        ).launch(
            grid=(batch * partitions_per_batch, 1, 1),
            block=(_THREADS, 1, 1),
            smem=(
                Int32(_WARPS)
                * ((max_kv_blocks + Int32(1)) // Int32(2))
                * Int32(_INT32_BYTES)
            ),
            stream=stream,
        )
        self.row_prefix_kernel(
            mRowCounts,
            mRowPtr,
            mRowCoords,
            mSchedulerMetadataUnsorted,
            mWorkCount,
            mDkvOwnerCounts,
            mDkvSplitIndices,
            mDkvSplitCount,
            mCuSeqlensK,
            mFragmentIndices,
            mKlPhysicalRowCounts,
            mKlPhysicalKOffsets,
            total_rows,
            target_q_per_cta,
            work_capacity,
            padded_kv_blocks,
            prepare_kl_schedule,
        ).launch(
            grid=(_HEAD_KV, 1, 1),
            block=(_PREFIX_THREADS, 1, 1),
            smem=_PREFIX_THREADS * _INT32_BYTES,
            stream=stream,
        )
        if const_expr(prepare_kl_schedule):
            self.kl_physical_prefix_kernel(
                mKlPhysicalRowCounts,
                mKlPhysicalRowPtr,
                padded_kv_blocks,
            ).launch(
                grid=(_HEAD_KV, 1, 1),
                block=(_PREFIX_THREADS, 1, 1),
                smem=_PREFIX_THREADS * _INT32_BYTES,
                stream=stream,
            )
            self.kl_physical_schedule_kernel(
                mKlPhysicalRowPtr,
                mKlPhysicalKOffsets,
                mKlSchedulerMetadata,
                mKlWorkCount,
                mKlDkiOwnerCounts,
                mKlDkiSplitIndices,
                mKlDkiSplitCount,
                padded_kv_blocks,
                kl_target_q_per_cta,
                kl_work_capacity,
            ).launch(
                grid=(
                    cute.ceil_div(padded_kv_blocks, _KL_SCHEDULE_THREADS),
                    1,
                    1,
                ),
                block=(_KL_SCHEDULE_THREADS, 1, 1),
                smem=0,
                stream=stream,
            )
        if const_expr(reorder_schedule):
            self.reorder_schedule_kernel(
                mSchedulerMetadataUnsorted,
                mSchedulerMetadata,
                mWorkCount,
                work_capacity,
                target_q_per_cta,
            ).launch(
                grid=(1, 1, 1),
                block=(_REORDER_THREADS, 1, 1),
                smem=(
                    _REORDER_MAX_CHUNKS * _REORDER_WORDS_PER_CHUNK
                    + _REORDER_FIXED_WORDS
                )
                * _INT32_BYTES,
                stream=stream,
            )
        self.tile_prefix_kernel(
            mTileCounts,
            mRowPtr,
            mRowMap,
            batch,
            total_rows,
            max_kv_blocks,
            partitions_per_batch,
        ).launch(
            grid=(
                _HEAD_KV
                * batch
                * cute.ceil_div(max_kv_blocks, _ROWS_PER_PREFIX_CTA),
                1,
                1,
            ),
            block=(_TILE_PREFIX_THREADS, 1, 1),
            smem=(
                _ROWS_PER_PREFIX_CTA
                * partitions_per_batch
                * _WARPS
                * _INT32_BYTES
            ),
            stream=stream,
        )
        self.scatter_kernel(
            mQ2K,
            mCuSeqlensQ,
            mCuSeqlensK,
            mFragmentIndices,
            mRowMap,
            mTileCounts,
            mQIdx,
            mQSplitIdx,
            mSplitCounts,
            mRowCounts,
            mRowPtr,
            mKlPhysicalRowPtr,
            mKlPhysicalQIndices,
            mKlPhysicalValidRows,
            total_rows,
            max_kv_blocks,
            padded_kv_blocks,
            partitions_per_batch,
            q_per_cta,
            q_per_warp,
            prepare_kl_schedule,
        ).launch(
            grid=(batch * partitions_per_batch, 1, 1),
            block=(_THREADS, 1, 1),
            smem=(
                Int32(_WARPS)
                * ((max_kv_blocks + Int32(1)) // Int32(2))
                * Int32(_INT32_BYTES)
            ),
            stream=stream,
        )

    @cute.kernel
    def init_kernel(
        self,
        fill_copy_atom: cute.CopyAtom,
        mQIdx: cute.Tensor,
        mRowCounts: cute.Tensor,
        mWorkCount: cute.Tensor,
        mDkvOwnerCounts: cute.Tensor,
        mDkvSplitCount: cute.Tensor,
        mKlWorkCount: cute.Tensor,
        mKlPhysicalRowCounts: cute.Tensor,
        mKlPhysicalKOffsets: cute.Tensor,
        mKlDkiOwnerCounts: cute.Tensor,
        mKlDkiSplitCount: cute.Tensor,
        num_q_idx: Int32,
        num_row_counts: Int32,
        num_owner_counts: Int32,
        padded_kv_blocks: Int32,
        prepare_kl_schedule: cutlass.Constexpr[bool],
    ):
        tidx = cute.arch.thread_idx()[0]
        block_idx = cute.arch.block_idx()[0]
        grid_dim = cute.arch.grid_dim()[0]
        idx = block_idx * Int32(_KL_SCHEDULE_THREADS) + tidx
        stride = grid_dim * Int32(_KL_SCHEDULE_THREADS)
        gRowCounts = cute.make_tensor(
            mRowCounts.iterator,
            cute.make_layout((num_row_counts,)),
        )
        vector_idx = idx
        num_vectors = num_q_idx // Int32(_INT32_VECTOR_ELEMENTS)
        tFill = cute.make_rmem_tensor((_INT32_VECTOR_ELEMENTS,), Int32)
        for value_idx in cutlass.range_constexpr(_INT32_VECTOR_ELEMENTS):
            tFill[value_idx] = Int32(-1)
        while vector_idx < num_vectors:
            q_idx = vector_idx * Int32(_INT32_VECTOR_ELEMENTS)
            qidx_ptr = cute.make_ptr(
                mQIdx.element_type,
                (mQIdx.iterator + q_idx).toint(),
                cute.AddressSpace.gmem,
                assumed_align=_VECTOR_BYTES,
            )
            gFill = cute.make_tensor(qidx_ptr, (_INT32_VECTOR_ELEMENTS,))
            fill_tiled_copy = cute.make_cotiled_copy(
                fill_copy_atom,
                cute.make_layout((1, _INT32_VECTOR_ELEMENTS)),
                tFill.layout,
            )
            fill_thr_copy = fill_tiled_copy.get_slice(0)
            tFrFill = fill_thr_copy.partition_S(tFill)
            tFgFill = fill_thr_copy.partition_D(gFill)
            cute.copy(fill_copy_atom, tFrFill, tFgFill)
            vector_idx += stride
        while idx < num_row_counts:
            gRowCounts[idx] = Int32(0)
            idx += stride
        owner_idx = block_idx * Int32(_KL_SCHEDULE_THREADS) + tidx
        while owner_idx < num_owner_counts:
            mDkvOwnerCounts[owner_idx] = Int32(0)
            owner_idx += stride
        if const_expr(prepare_kl_schedule):
            kl_idx = block_idx * Int32(_KL_SCHEDULE_THREADS) + tidx
            num_physical_counts = Int32(_HEAD_KV) * padded_kv_blocks
            while kl_idx < num_physical_counts:
                mKlPhysicalRowCounts[kl_idx] = Int32(0)
                kl_idx += stride
            physical_idx = block_idx * Int32(_KL_SCHEDULE_THREADS) + tidx
            while physical_idx < padded_kv_blocks:
                mKlPhysicalKOffsets[physical_idx] = Int32(-1)
                mKlDkiOwnerCounts[physical_idx] = Int32(0)
                physical_idx += stride
        if block_idx == Int32(0) and tidx == Int32(0):
            mWorkCount[Int32(0)] = Int32(0)
            mDkvSplitCount[Int32(0)] = Int32(0)
            mKlWorkCount[Int32(0)] = Int32(0)
            mKlDkiSplitCount[Int32(0)] = Int32(0)

    @cute.kernel
    def histogram_kernel(
        self,
        mQ2K: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mFragmentIndices: Optional[cute.Tensor],
        mRowMap: cute.Tensor,
        mRowCoords: cute.Tensor,
        mRowCounts: cute.Tensor,
        mTileCounts: cute.Tensor,
        batch: Int32,
        max_kv_blocks: Int32,
        partitions_per_batch: Int32,
        q_per_cta: Int32,
        q_per_warp: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        cta_idx = cute.arch.block_idx()[0]
        warp_idx = tidx // Int32(32)
        lane_idx = tidx % Int32(32)
        batch_idx = cta_idx // partitions_per_batch
        partition_idx = cta_idx - batch_idx * partitions_per_batch
        q_start_cta = mCuSeqlensQ[batch_idx] + partition_idx * q_per_cta
        q_end_cta = cutlass.min(
            q_start_cta + q_per_cta,
            mCuSeqlensQ[batch_idx + Int32(1)],
        )
        q_start_warp = cutlass.min(
            q_start_cta + warp_idx * q_per_warp,
            q_end_cta,
        )
        q_end_warp = cutlass.min(q_start_warp + q_per_warp, q_end_cta)
        packed_per_warp = (max_kv_blocks + Int32(1)) // Int32(2)
        smem_ptr = cute.arch.get_dyn_smem(Int32, alignment=16)
        sPacked = cute.make_tensor(
            smem_ptr,
            cute.make_layout(
                (_WARPS, packed_per_warp),
                stride=(packed_per_warp, 1),
            ),
        )
        sMyPacked = sPacked[warp_idx, None]

        if cta_idx == Int32(0):
            map_idx = tidx
            map_entries = batch * max_kv_blocks
            while map_idx < map_entries:
                map_batch = map_idx // max_kv_blocks
                level = map_idx - map_batch * max_kv_blocks
                rows = self._rows_in_batch(
                    mCuSeqlensK,
                    map_batch,
                    mFragmentIndices,
                )
                row = Int32(-1)
                if level < rows:
                    row = self._row_linear_for_batch_level(
                        mCuSeqlensK,
                        map_batch,
                        level,
                        batch,
                        mFragmentIndices,
                    )
                    mRowCoords[row, Int32(0)] = map_batch
                    mRowCoords[row, Int32(1)] = level
                mRowMap[map_batch, level] = row
                map_idx += Int32(_THREADS)

        batch_rows = self._rows_in_batch(
            mCuSeqlensK,
            batch_idx,
            mFragmentIndices,
        )
        for head_idx in cutlass.range_constexpr(_HEAD_KV):
            packed_idx = lane_idx
            while packed_idx < packed_per_warp:
                sMyPacked[packed_idx] = Int32(0)
                packed_idx += Int32(32)
            cute.arch.sync_threads()

            q_idx = q_start_warp + lane_idx
            while q_idx < q_end_warp:
                for topk_idx in cutlass.range_constexpr(_TOPK):
                    kv_block = mQ2K[head_idx, q_idx, topk_idx]
                    if kv_block >= Int32(0) and kv_block < batch_rows:
                        self._atomic_inc_packed_i16(sMyPacked, kv_block)
                q_idx += Int32(32)
            cute.arch.sync_threads()

            row_idx = lane_idx
            tile_idx = cta_idx * Int32(_WARPS) + warp_idx
            while row_idx < max_kv_blocks:
                mTileCounts[tile_idx, head_idx, row_idx] = (
                    self._read_packed_i16(sMyPacked, row_idx)
                )
                row_idx += Int32(32)
            cute.arch.sync_threads()

            row_idx = tidx
            while row_idx < max_kv_blocks:
                row_sum = Int32(0)
                for source_warp in cutlass.range_constexpr(_WARPS):
                    row_sum += self._read_packed_i16(
                        sPacked[source_warp, None],
                        row_idx,
                    )
                if row_sum > Int32(0):
                    ptr = (
                        mRowCounts.iterator
                        + head_idx * batch * max_kv_blocks
                        + batch_idx * max_kv_blocks
                        + row_idx
                    )
                    cute.arch.atomic_add(
                        ptr.llvm_ptr,
                        row_sum,
                        sem="relaxed",
                        scope="gpu",
                    )
                row_idx += Int32(_THREADS)
            if const_expr(head_idx + 1 < _HEAD_KV):
                cute.arch.sync_threads()

    @cute.kernel
    def row_prefix_kernel(
        self,
        mRowCounts: cute.Tensor,
        mRowPtr: cute.Tensor,
        mRowCoords: cute.Tensor,
        mSchedulerMetadata: cute.Tensor,
        mWorkCount: cute.Tensor,
        mDkvOwnerCounts: cute.Tensor,
        mDkvSplitIndices: cute.Tensor,
        mDkvSplitCount: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mFragmentIndices: Optional[cute.Tensor],
        mKlPhysicalRowCounts: cute.Tensor,
        mKlPhysicalKOffsets: cute.Tensor,
        total_rows: Int32,
        target_q_per_cta: Int32,
        work_capacity: Int32,
        padded_kv_blocks: Int32,
        prepare_kl_schedule: cutlass.Constexpr[bool],
    ):
        head_idx = cute.arch.block_idx()[0]
        tidx = cute.arch.thread_idx()[0]
        smem_ptr = cute.arch.get_dyn_smem(Int32, alignment=16)
        sScan = cute.make_tensor(
            smem_ptr,
            cute.make_layout((_PREFIX_THREADS,)),
        )
        if tidx == Int32(0):
            mRowPtr[head_idx, Int32(0)] = Int32(0)
        chunk = (total_rows + Int32(_PREFIX_THREADS - 1)) // Int32(
            _PREFIX_THREADS
        )
        row_begin = tidx * chunk
        row_end = cutlass.min(row_begin + chunk, total_rows)
        local_sum = Int32(0)
        row_idx = row_begin
        while row_idx < row_end:
            batch_idx = mRowCoords[row_idx, Int32(0)]
            kv_block_idx = mRowCoords[row_idx, Int32(1)]
            local_sum += mRowCounts[head_idx, batch_idx, kv_block_idx]
            row_idx += Int32(1)
        sScan[tidx] = local_sum
        cute.arch.sync_threads()

        for scan_step in cutlass.range_constexpr(10):
            offset = Int32(1 << scan_step)
            addend = Int32(0)
            if tidx >= offset:
                addend = sScan[tidx - offset]
            cute.arch.sync_threads()
            sScan[tidx] += addend
            cute.arch.sync_threads()

        running = sScan[tidx] - local_sum
        row_idx = row_begin
        while row_idx < row_end:
            batch_idx = mRowCoords[row_idx, Int32(0)]
            kv_block_idx = mRowCoords[row_idx, Int32(1)]
            row_count = mRowCounts[head_idx, batch_idx, kv_block_idx]
            running += row_count
            mRowPtr[head_idx, row_idx + Int32(1)] = running
            if row_count > Int32(0):
                num_chunks = (
                    row_count + target_q_per_cta - Int32(1)
                ) // target_q_per_cta
                physical_batch_idx = (
                    batch_idx
                    if const_expr(mFragmentIndices is None)
                    else mFragmentIndices[batch_idx]
                )
                physical_offset_k = mCuSeqlensK[physical_batch_idx]
                physical_block_idx = (
                    physical_offset_k
                    + physical_batch_idx * Int32(_BLOCK_K)
                ) // Int32(_BLOCK_K) + kv_block_idx
                if const_expr(prepare_kl_schedule):
                    physical_count_ptr = (
                        mKlPhysicalRowCounts.iterator
                        + head_idx * padded_kv_blocks
                        + physical_block_idx
                    )
                    logical_physical_offset = cute.arch.atomic_add(
                        physical_count_ptr.llvm_ptr,
                        row_count,
                        sem="relaxed",
                        scope="gpu",
                    )
                    # Row counts are dead after this prefix pass. Reuse their
                    # storage for the row-local offset in the physical CSR row.
                    mRowCounts[head_idx, batch_idx, kv_block_idx] = (
                        logical_physical_offset
                    )
                    cute.arch.atomic_cas(
                        (
                            mKlPhysicalKOffsets.iterator
                            + physical_block_idx
                        ).llvm_ptr,
                        cmp=Int32(-1),
                        val=physical_offset_k + kv_block_idx * Int32(_BLOCK_K),
                        sem="relaxed",
                        scope="gpu",
                    )
                owner_ptr = (
                    mDkvOwnerCounts.iterator
                    + head_idx * padded_kv_blocks
                    + physical_block_idx
                )
                old_owner_count = cute.arch.atomic_add(
                    owner_ptr.llvm_ptr,
                    num_chunks,
                    sem="relaxed",
                    scope="gpu",
                )
                if (
                    old_owner_count < Int32(2)
                    and old_owner_count + num_chunks >= Int32(2)
                ):
                    split_idx = cute.arch.atomic_add(
                        mDkvSplitCount.iterator.llvm_ptr,
                        Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                    if split_idx < work_capacity:
                        mDkvSplitIndices[split_idx] = (
                            head_idx * padded_kv_blocks + physical_block_idx
                        )
                work_base = cute.arch.atomic_add(
                    mWorkCount.iterator.llvm_ptr,
                    num_chunks,
                    sem="relaxed",
                    scope="gpu",
                )
                chunk_idx = Int32(0)
                while chunk_idx < num_chunks:
                    work_idx = work_base + chunk_idx
                    if work_idx < work_capacity:
                        q_begin = chunk_idx * target_q_per_cta
                        q_count = cutlass.min(
                            target_q_per_cta,
                            row_count - q_begin,
                        )
                        mSchedulerMetadata[work_idx, Int32(0)] = head_idx
                        mSchedulerMetadata[work_idx, Int32(1)] = row_idx
                        mSchedulerMetadata[work_idx, Int32(2)] = q_begin
                        mSchedulerMetadata[work_idx, Int32(3)] = q_count
                        mSchedulerMetadata[work_idx, Int32(4)] = batch_idx
                        mSchedulerMetadata[work_idx, Int32(5)] = kv_block_idx
                    chunk_idx += Int32(1)
            row_idx += Int32(1)

    @cute.kernel
    def kl_physical_prefix_kernel(
        self,
        mPhysicalRowCounts: cute.Tensor,
        mPhysicalRowPtr: cute.Tensor,
        padded_kv_blocks: Int32,
    ):
        head_idx = cute.arch.block_idx()[0]
        tidx = cute.arch.thread_idx()[0]
        smem_ptr = cute.arch.get_dyn_smem(Int32, alignment=16)
        sScan = cute.make_tensor(smem_ptr, cute.make_layout((_PREFIX_THREADS,)))
        if tidx == Int32(0):
            mPhysicalRowPtr[head_idx, Int32(0)] = Int32(0)
        chunk = (padded_kv_blocks + Int32(_PREFIX_THREADS - 1)) // Int32(
            _PREFIX_THREADS
        )
        block_begin = tidx * chunk
        block_end = cutlass.min(block_begin + chunk, padded_kv_blocks)
        local_sum = Int32(0)
        physical_block = block_begin
        while physical_block < block_end:
            local_sum += mPhysicalRowCounts[head_idx, physical_block]
            physical_block += Int32(1)
        sScan[tidx] = local_sum
        cute.arch.sync_threads()

        for scan_step in cutlass.range_constexpr(10):
            offset = Int32(1 << scan_step)
            addend = Int32(0)
            if tidx >= offset:
                addend = sScan[tidx - offset]
            cute.arch.sync_threads()
            sScan[tidx] += addend
            cute.arch.sync_threads()

        running = sScan[tidx] - local_sum
        physical_block = block_begin
        while physical_block < block_end:
            count = mPhysicalRowCounts[head_idx, physical_block]
            running += count
            mPhysicalRowPtr[head_idx, physical_block + Int32(1)] = running
            physical_block += Int32(1)

    @cute.kernel
    def kl_physical_schedule_kernel(
        self,
        mPhysicalRowPtr: cute.Tensor,
        mPhysicalKOffsets: cute.Tensor,
        mKlSchedulerMetadata: cute.Tensor,
        mKlWorkCount: cute.Tensor,
        mKlDkiOwnerCounts: cute.Tensor,
        mKlDkiSplitIndices: cute.Tensor,
        mKlDkiSplitCount: cute.Tensor,
        padded_kv_blocks: Int32,
        kl_target_q_per_cta: Int32,
        kl_work_capacity: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        physical_block = (
            cute.arch.block_idx()[0] * Int32(_KL_SCHEDULE_THREADS) + tidx
        )
        block_stride = cute.arch.grid_dim()[0] * Int32(_KL_SCHEDULE_THREADS)
        q_per_macro = Int32(_KL_Q_PER_MACRO)
        macros_per_work = kl_target_q_per_cta // q_per_macro
        while physical_block < padded_kv_blocks:
            total_macros = Int32(0)
            for head_idx in cutlass.range_constexpr(_HEAD_KV):
                row_count = (
                    mPhysicalRowPtr[head_idx, physical_block + Int32(1)]
                    - mPhysicalRowPtr[head_idx, physical_block]
                )
                total_macros += (
                    row_count + q_per_macro - Int32(1)
                ) // q_per_macro
            owner_count = (
                total_macros + macros_per_work - Int32(1)
            ) // macros_per_work
            mKlDkiOwnerCounts[physical_block] = owner_count
            if owner_count > Int32(1):
                split_idx = cute.arch.atomic_add(
                    mKlDkiSplitCount.iterator.llvm_ptr,
                    Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
                if split_idx < padded_kv_blocks:
                    mKlDkiSplitIndices[split_idx] = physical_block
            if owner_count > Int32(0):
                work_base = cute.arch.atomic_add(
                    mKlWorkCount.iterator.llvm_ptr,
                    owner_count,
                    sem="relaxed",
                    scope="gpu",
                )
                owner_idx = Int32(0)
                while owner_idx < owner_count:
                    work_idx = work_base + owner_idx
                    if work_idx < kl_work_capacity:
                        macro_begin = owner_idx * macros_per_work
                        macro_count = cutlass.min(
                            macros_per_work,
                            total_macros - macro_begin,
                        )
                        mKlSchedulerMetadata[work_idx, Int32(0)] = physical_block
                        mKlSchedulerMetadata[work_idx, Int32(1)] = (
                            mPhysicalKOffsets[physical_block]
                        )
                        mKlSchedulerMetadata[work_idx, Int32(2)] = macro_begin
                        mKlSchedulerMetadata[work_idx, Int32(3)] = macro_count
                    owner_idx += Int32(1)
            physical_block += block_stride

    @cute.jit
    def _copy_schedule_row(
        self,
        mSrc: cute.Tensor,
        mDst: cute.Tensor,
        src_idx: Int32,
        dst_idx: Int32,
    ) -> None:
        for field_idx in cutlass.range_constexpr(6):
            mDst[dst_idx, field_idx] = mSrc[src_idx, field_idx]

    @cute.jit
    def _schedule_bucket(
        self,
        mSchedulerMetadata: cute.Tensor,
        work_idx: Int32,
        target_q_per_cta: Int32,
    ) -> Int32:
        q_count = mSchedulerMetadata[work_idx, Int32(3)]
        return cutlass.min(
            Int32(_SCHEDULE_BUCKETS - 1),
            cutlass.max(
                Int32(0),
                q_count * Int32(_SCHEDULE_BUCKETS) // target_q_per_cta,
            ),
        )

    @cute.jit
    def _reorder_schedule(
        self,
        mSchedulerMetadataIn: cute.Tensor,
        mSchedulerMetadataOut: cute.Tensor,
        mWorkCount: cute.Tensor,
        work_capacity: Int32,
        target_q_per_cta: Int32,
    ) -> None:
        tidx = cute.arch.thread_idx()[0]
        warp_idx = tidx // Int32(32)
        lane_idx = tidx % Int32(32)
        num_work = cutlass.min(
            cutlass.max(mWorkCount[Int32(0)], Int32(0)),
            work_capacity,
        )
        if num_work > Int32(_SCHEDULE_REORDER_CAPACITY):
            work_idx = tidx
            while work_idx < num_work:
                self._copy_schedule_row(
                    mSchedulerMetadataIn,
                    mSchedulerMetadataOut,
                    work_idx,
                    work_idx,
                )
                work_idx += Int32(_REORDER_THREADS)
            num_work = Int32(0)
        max_chunks = _REORDER_MAX_CHUNKS
        smem_ptr = cute.arch.get_dyn_smem(Int32, alignment=16)
        chunk_counts = cute.make_tensor(
            smem_ptr,
            cute.make_layout((max_chunks, _SCHEDULE_BUCKETS)),
        )
        chunk_offsets = cute.make_tensor(
            smem_ptr + Int32(max_chunks * _SCHEDULE_BUCKETS),
            cute.make_layout((max_chunks, _SCHEDULE_BUCKETS)),
        )
        bucket_bases = cute.make_tensor(
            smem_ptr + Int32(2 * max_chunks * _SCHEDULE_BUCKETS),
            cute.make_layout((_SCHEDULE_BUCKETS,)),
        )
        num_chunks = (
            num_work + Int32(_WARP_THREADS - 1)
        ) // Int32(_WARP_THREADS)
        chunk_idx = warp_idx
        while chunk_idx < num_chunks:
            work_idx = chunk_idx * Int32(_WARP_THREADS) + lane_idx
            bucket = Int32(-1)
            if work_idx < num_work:
                bucket = self._schedule_bucket(
                    mSchedulerMetadataIn,
                    work_idx,
                    target_q_per_cta,
                )
            for bucket_idx in cutlass.range_constexpr(_SCHEDULE_BUCKETS):
                bucket_mask = cute.arch.vote_ballot_sync(
                    bucket == Int32(bucket_idx)
                )
                if lane_idx == Int32(0):
                    chunk_counts[chunk_idx, bucket_idx] = cute.arch.popc(
                        bucket_mask
                    )
            chunk_idx += Int32(_REORDER_WARPS)
        cute.arch.sync_threads()

        if tidx == Int32(0):
            bucket_base = Int32(0)
            for rev_bucket in cutlass.range_constexpr(_SCHEDULE_BUCKETS):
                bucket_idx = Int32(_SCHEDULE_BUCKETS - 1 - rev_bucket)
                bucket_bases[bucket_idx] = bucket_base
                prefix = Int32(0)
                chunk_idx = Int32(0)
                while chunk_idx < num_chunks:
                    chunk_offsets[chunk_idx, bucket_idx] = prefix
                    prefix += chunk_counts[chunk_idx, bucket_idx]
                    chunk_idx += Int32(1)
                bucket_base += prefix
        cute.arch.sync_threads()

        chunk_idx = warp_idx
        while chunk_idx < num_chunks:
            src_idx = chunk_idx * Int32(_WARP_THREADS) + lane_idx
            bucket = Int32(-1)
            if src_idx < num_work:
                bucket = self._schedule_bucket(
                    mSchedulerMetadataIn,
                    src_idx,
                    target_q_per_cta,
                )
            local_rank = Int32(0)
            for bucket_idx in cutlass.range_constexpr(_SCHEDULE_BUCKETS):
                bucket_mask = cute.arch.vote_ballot_sync(
                    bucket == Int32(bucket_idx)
                )
                if bucket == Int32(bucket_idx):
                    lower_lane_mask = (Int32(1) << lane_idx) - Int32(1)
                    if lane_idx == Int32(0):
                        lower_lane_mask = Int32(0)
                    local_rank = cute.arch.popc(bucket_mask & lower_lane_mask)
            if src_idx < num_work:
                dst_idx = (
                    bucket_bases[bucket]
                    + chunk_offsets[chunk_idx, bucket]
                    + local_rank
                )
                self._copy_schedule_row(
                    mSchedulerMetadataIn,
                    mSchedulerMetadataOut,
                    src_idx,
                    dst_idx,
                )
            chunk_idx += Int32(_REORDER_WARPS)

    @cute.kernel
    def reorder_schedule_kernel(
        self,
        mSchedulerMetadataIn: cute.Tensor,
        mSchedulerMetadataOut: cute.Tensor,
        mWorkCount: cute.Tensor,
        work_capacity: Int32,
        target_q_per_cta: Int32,
    ):
        self._reorder_schedule(
            mSchedulerMetadataIn,
            mSchedulerMetadataOut,
            mWorkCount,
            work_capacity,
            target_q_per_cta,
        )

    @cute.kernel
    def tile_prefix_kernel(
        self,
        mTileCounts: cute.Tensor,
        mRowPtr: cute.Tensor,
        mRowMap: cute.Tensor,
        batch: Int32,
        total_rows: Int32,
        max_kv_blocks: Int32,
        partitions_per_batch: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        lane_idx = tidx % Int32(32)
        warp_idx = tidx // Int32(32)
        job_idx = cute.arch.block_idx()[0]
        blocks_per_batch = (
            max_kv_blocks + Int32(_ROWS_PER_PREFIX_CTA - 1)
        ) // Int32(_ROWS_PER_PREFIX_CTA)
        jobs_per_head = batch * blocks_per_batch
        head_idx = job_idx // jobs_per_head
        job_in_head = job_idx - head_idx * jobs_per_head
        batch_idx = job_in_head // blocks_per_batch
        block_in_batch = job_in_head - batch_idx * blocks_per_batch
        base_row = block_in_batch * Int32(_ROWS_PER_PREFIX_CTA)
        actual_rows = cutlass.min(
            Int32(_ROWS_PER_PREFIX_CTA),
            max_kv_blocks - base_row,
        )
        partitions_per_batch_warp = partitions_per_batch * Int32(_WARPS)
        smem_ptr = cute.arch.get_dyn_smem(Int32, alignment=16)
        sTile = cute.make_tensor(
            smem_ptr,
            cute.make_layout(
                (_ROWS_PER_PREFIX_CTA, partitions_per_batch_warp),
                stride=(partitions_per_batch_warp, 1),
            ),
        )
        total_items = actual_rows * partitions_per_batch_warp
        item_idx = tidx
        while item_idx < total_items:
            row_offset = item_idx % actual_rows
            partition_warp = item_idx // actual_rows
            tile_idx = batch_idx * partitions_per_batch_warp + partition_warp
            sTile[row_offset, partition_warp] = mTileCounts[
                tile_idx,
                head_idx,
                base_row + row_offset,
            ]
            item_idx += Int32(_TILE_PREFIX_THREADS)
        cute.arch.sync_threads()

        if warp_idx < actual_rows:
            local_row = base_row + warp_idx
            row = mRowMap[batch_idx, local_row]
            running = Int32(0)
            if row >= Int32(0) and row < total_rows:
                running = mRowPtr[head_idx, row]
            partition_begin = Int32(0)
            while partition_begin < partitions_per_batch_warp:
                partition_warp = partition_begin + lane_idx
                value = Int32(0)
                if partition_warp < partitions_per_batch_warp:
                    value = sTile[warp_idx, partition_warp]
                inclusive = value
                for scan_step in cutlass.range_constexpr(5):
                    offset = 1 << scan_step
                    neighbor = cute.arch.shuffle_sync_up(
                        inclusive,
                        offset=offset,
                        mask_and_clamp=0,
                    )
                    if lane_idx >= Int32(offset):
                        inclusive += neighbor
                exclusive = running + inclusive - value
                if partition_warp < partitions_per_batch_warp:
                    sTile[warp_idx, partition_warp] = exclusive
                running += cute.arch.shuffle_sync(inclusive, offset=31)
                partition_begin += Int32(32)
        cute.arch.sync_threads()

        item_idx = tidx
        while item_idx < total_items:
            row_offset = item_idx % actual_rows
            partition_warp = item_idx // actual_rows
            tile_idx = batch_idx * partitions_per_batch_warp + partition_warp
            mTileCounts[
                tile_idx,
                head_idx,
                base_row + row_offset,
            ] = sTile[row_offset, partition_warp]
            item_idx += Int32(_TILE_PREFIX_THREADS)

    @cute.kernel
    def scatter_kernel(
        self,
        mQ2K: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mFragmentIndices: Optional[cute.Tensor],
        mRowMap: cute.Tensor,
        mAbsoluteBase: cute.Tensor,
        mQIdx: cute.Tensor,
        mQSplitIdx: cute.Tensor,
        mSplitCounts: cute.Tensor,
        mKlLogicalPhysicalOffsets: cute.Tensor,
        mRowPtr: cute.Tensor,
        mKlPhysicalRowPtr: cute.Tensor,
        mKlPhysicalQIndices: cute.Tensor,
        mKlPhysicalValidRows: cute.Tensor,
        total_rows: Int32,
        max_kv_blocks: Int32,
        padded_kv_blocks: Int32,
        partitions_per_batch: Int32,
        q_per_cta: Int32,
        q_per_warp: Int32,
        prepare_kl_schedule: cutlass.Constexpr[bool],
    ):
        tidx = cute.arch.thread_idx()[0]
        cta_idx = cute.arch.block_idx()[0]
        warp_idx = tidx // Int32(32)
        lane_idx = tidx % Int32(32)
        batch_idx = cta_idx // partitions_per_batch
        partition_idx = cta_idx - batch_idx * partitions_per_batch
        q_start_cta = mCuSeqlensQ[batch_idx] + partition_idx * q_per_cta
        q_end_cta = cutlass.min(
            q_start_cta + q_per_cta,
            mCuSeqlensQ[batch_idx + Int32(1)],
        )
        q_start_warp = cutlass.min(
            q_start_cta + warp_idx * q_per_warp,
            q_end_cta,
        )
        q_end_warp = cutlass.min(q_start_warp + q_per_warp, q_end_cta)
        q_in_iter = lane_idx // Int32(_TOPK)
        topk_idx = lane_idx % Int32(_TOPK)
        packed_per_warp = (max_kv_blocks + Int32(1)) // Int32(2)
        smem_ptr = cute.arch.get_dyn_smem(Int32, alignment=16)
        sPacked = cute.make_tensor(
            smem_ptr,
            cute.make_layout(
                (_WARPS, packed_per_warp),
                stride=(packed_per_warp, 1),
            ),
        )
        sMyPacked = sPacked[warp_idx, None]
        physical_batch_idx = (
            batch_idx
            if const_expr(mFragmentIndices is None)
            else mFragmentIndices[batch_idx]
        )
        physical_offset_k = mCuSeqlensK[physical_batch_idx]
        physical_block_base = (
            physical_offset_k + physical_batch_idx * Int32(_BLOCK_K)
        ) // Int32(_BLOCK_K)
        q_len = mCuSeqlensQ[batch_idx + Int32(1)] - mCuSeqlensQ[batch_idx]
        k_len = mCuSeqlensK[batch_idx + Int32(1)] - physical_offset_k

        for head_idx in cutlass.range_constexpr(_HEAD_KV):
            packed_idx = lane_idx
            while packed_idx < packed_per_warp:
                sMyPacked[packed_idx] = Int32(0)
                packed_idx += Int32(32)
            cute.arch.sync_warp()

            q_base = q_start_warp
            while q_base < q_end_warp:
                # Process the two 16-lane query groups in fixed order. TopK
                # entries are unique within one query, so this removes the
                # only same-half packed-atomic race while retaining both
                # queries per warp iteration.
                for query_group in cutlass.range_constexpr(2):
                    q_idx = q_base + Int32(query_group)
                    valid_q = (
                        q_in_iter == Int32(query_group)
                        and q_idx < q_end_warp
                    )
                    q_local = Int32(0)
                    kv_block = Int32(-1)
                    if valid_q:
                        q_local = q_idx - mCuSeqlensQ[batch_idx]
                        kv_block = mQ2K[head_idx, q_idx, topk_idx]
                    global_row = Int32(-1)
                    if kv_block >= Int32(0) and kv_block < max_kv_blocks:
                        global_row = mRowMap[batch_idx, kv_block]
                    valid_edge = (
                        valid_q
                        and global_row >= Int32(0)
                        and global_row < total_rows
                    )
                    valid_mask = cute.arch.vote_ballot_sync(valid_edge)
                    group_mask = Int32(0xFFFF) << Int32(query_group * 16)
                    valid_count = cute.arch.popc(valid_mask & group_mask)
                    writer_rank = Int32(0)
                    for other_slot in cutlass.range_constexpr(_TOPK):
                        other_block = cute.arch.shuffle_sync(
                            kv_block,
                            query_group * _TOPK + other_slot,
                        )
                        if (
                            other_block >= Int32(0)
                            and other_block < max_kv_blocks
                            and other_block < kv_block
                        ):
                            writer_rank += Int32(1)
                    if valid_q and topk_idx == Int32(0):
                        mSplitCounts[q_idx, head_idx] = valid_count
                    if valid_edge:
                        local_slot = self._atomic_inc_packed_i16(
                            sMyPacked,
                            kv_block,
                        )
                        tile_idx = cta_idx * Int32(_WARPS) + warp_idx
                        out_idx = (
                            mAbsoluteBase[tile_idx, head_idx, kv_block]
                            + local_slot
                        )
                        mQIdx[head_idx, out_idx] = q_local
                        mQSplitIdx[head_idx, out_idx] = (
                            q_local
                            | ((writer_rank & Int32(0xFF)) << Int32(24))
                        )
                        if const_expr(prepare_kl_schedule):
                            physical_block = physical_block_base + kv_block
                            if physical_block < padded_kv_blocks:
                                logical_row_begin = mRowPtr[head_idx, global_row]
                                physical_out_idx = (
                                    mKlPhysicalRowPtr[head_idx, physical_block]
                                    + mKlLogicalPhysicalOffsets[
                                        head_idx, batch_idx, kv_block
                                    ]
                                    + out_idx
                                    - logical_row_begin
                                )
                                block_begin = kv_block * Int32(_BLOCK_K)
                                physical_rows = cutlass.min(
                                    Int32(_BLOCK_K),
                                    cutlass.max(Int32(0), k_len - block_begin),
                                )
                                q_logical = k_len - q_len + q_local
                                valid_rows = cutlass.min(
                                    physical_rows,
                                    cutlass.max(
                                        Int32(0),
                                        q_logical - block_begin + Int32(1),
                                    ),
                                )
                                mKlPhysicalQIndices[
                                    head_idx, physical_out_idx
                                ] = q_idx
                                mKlPhysicalValidRows[
                                    head_idx, physical_out_idx
                                ] = (
                                    valid_rows & Int32(KL_VALID_ROWS_MASK)
                                ) | (
                                    (writer_rank & Int32(KL_WRITER_RANK_MASK))
                                    << Int32(KL_WRITER_RANK_SHIFT)
                                )
                    cute.arch.sync_warp()
                q_base += Int32(2)
            if const_expr(head_idx + 1 < _HEAD_KV):
                cute.arch.sync_threads()


__all__ = ["SparseK2qCsrPipelineSm100"]
