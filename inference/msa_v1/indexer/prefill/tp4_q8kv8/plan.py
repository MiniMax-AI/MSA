"""Device-side varlen task planning for the Blackwell prefill indexer."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

from inference.msa_v1.indexer.prefill.tp4_q8kv8.indexer_gemm import (
    PrefillIndexerGemmSm100,
)


class PrefillIndexerPlanReset:
    """Reset device-resident queue counters and error state."""

    num_buckets = PrefillIndexerGemmSm100.num_task_buckets
    threads_per_cta = cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mTaskCounts: cute.Tensor,
        mPlanError: cute.Tensor,
        stream: cuda.CUstream = None,
    ) -> None:
        self.kernel(mTaskCounts, mPlanError).launch(
            grid=(1, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mTaskCounts: cute.Tensor,
        mPlanError: cute.Tensor,
    ) -> None:
        lane_idx = cute.arch.lane_idx()
        if lane_idx < cutlass.Int32(self.num_buckets):
            mTaskCounts[lane_idx] = cutlass.Int32(0)
        if lane_idx == cutlass.Int32(0):
            mPlanError[0] = cutlass.Int32(0)


class PrefillIndexerPlanBuild:
    """Expand device varlen metadata into page-range task buckets."""

    q_tile = PrefillIndexerGemmSm100.q_tile
    page_size = PrefillIndexerGemmSm100.k_tile
    num_buckets = PrefillIndexerGemmSm100.num_task_buckets
    split_page_chunk = 64
    large_page_chunk = 128
    split_q_tile_threshold = 60
    descriptor_words = 4
    threads_per_cta = cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mLengths: cute.Tensor,
        mTaskDescriptors: cute.Tensor,
        mTaskCounts: cute.Tensor,
        mPlanError: cute.Tensor,
        num_candidate_q_tiles: cutlass.Int32,
        task_capacity: cutlass.Int32,
        stream: cuda.CUstream = None,
    ) -> None:
        self.kernel(
            mCuSeqlensQ,
            mCuSeqlensK,
            mLengths,
            mTaskDescriptors,
            mTaskCounts,
            mPlanError,
            task_capacity,
        ).launch(
            grid=(num_candidate_q_tiles, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _cost_bucket(
        self,
        q_rows: cutlass.Int32,
        page_count: cutlass.Int32,
    ) -> cutlass.Int32:
        work = q_rows * page_count
        bucket = cutlass.Int32(0)
        if work > cutlass.Int32(1 << 10):
            bucket = cutlass.Int32(1)
        if work > cutlass.Int32(1 << 11):
            bucket = cutlass.Int32(2)
        if work > cutlass.Int32(1 << 12):
            bucket = cutlass.Int32(3)
        if work > cutlass.Int32(1 << 13):
            bucket = cutlass.Int32(4)
        if work > cutlass.Int32(1 << 14):
            bucket = cutlass.Int32(5)
        if work > cutlass.Int32(1 << 15):
            bucket = cutlass.Int32(6)
        if work > cutlass.Int32(1 << 16):
            bucket = cutlass.Int32(7)
        return bucket

    @cute.jit
    def _pack_descriptor(
        self,
        batch_idx: cutlass.Int32,
        q_rows: cutlass.Int32,
        page_count: cutlass.Int32,
        bucket: cutlass.Int32,
    ) -> cutlass.Int32:
        return (
            batch_idx
            | ((q_rows - cutlass.Int32(1)) << cutlass.Int32(5))
            | ((page_count - cutlass.Int32(1)) << cutlass.Int32(13))
            | (bucket << cutlass.Int32(25))
        )

    @cute.kernel
    def kernel(
        self,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mLengths: cute.Tensor,
        mTaskDescriptors: cute.Tensor,
        mTaskCounts: cute.Tensor,
        mPlanError: cute.Tensor,
        task_capacity: cutlass.Int32,
    ) -> None:
        lane_idx = cute.arch.lane_idx()
        candidate_idx, _, _ = cute.arch.block_idx()
        batch = cute.size(mCuSeqlensQ) - 1
        batch_idx = candidate_idx % batch
        q_tile_idx = candidate_idx // batch
        q_local_begin = q_tile_idx * cutlass.Int32(self.q_tile)

        q_start = cutlass.Int32(0)
        seq_q = cutlass.Int32(0)
        seq_k = cutlass.Int32(0)
        q_rows = cutlass.Int32(0)
        num_pages = cutlass.Int32(0)
        num_q_tiles = cutlass.Int32(0)
        if lane_idx == cutlass.Int32(0):
            q_start = mCuSeqlensQ[batch_idx]
            q_end = mCuSeqlensQ[batch_idx + cutlass.Int32(1)]
            seq_q = q_end - q_start
            k_start = mCuSeqlensK[batch_idx]
            k_end = mCuSeqlensK[batch_idx + cutlass.Int32(1)]
            seq_k = k_end - k_start
            if q_local_begin < seq_q:
                q_rows = seq_q - q_local_begin
                if q_rows > cutlass.Int32(self.q_tile):
                    q_rows = cutlass.Int32(self.q_tile)
                last_position = (
                    seq_k
                    - seq_q
                    + q_local_begin
                    + q_rows
                    - cutlass.Int32(1)
                )
                num_pages = last_position // cutlass.Int32(self.page_size)
                if num_pages < cutlass.Int32(0):
                    num_pages = cutlass.Int32(0)
                num_q_tiles = cute.ceil_div(seq_q, self.q_tile)

        q_start = cute.arch.shuffle_sync(q_start, 0)
        seq_q = cute.arch.shuffle_sync(seq_q, 0)
        seq_k = cute.arch.shuffle_sync(seq_k, 0)
        q_rows = cute.arch.shuffle_sync(q_rows, 0)
        num_pages = cute.arch.shuffle_sync(num_pages, 0)
        num_q_tiles = cute.arch.shuffle_sync(num_q_tiles, 0)

        for row_iter in cutlass.range_constexpr(self.q_tile // cute.arch.WARP_SIZE):
            q_local = (
                q_local_begin
                + lane_idx
                + cutlass.Int32(row_iter * cute.arch.WARP_SIZE)
            )
            if q_local < seq_q:
                query_position = seq_k - seq_q + q_local
                local_block = query_position // cutlass.Int32(self.page_size)
                mLengths[q_start + q_local] = local_block + cutlass.Int32(1)

        if lane_idx == cutlass.Int32(0) and q_rows > cutlass.Int32(0):
            page_chunk = cutlass.Int32(self.split_page_chunk)
            if num_q_tiles >= cutlass.Int32(self.split_q_tile_threshold):
                page_chunk = cutlass.Int32(self.large_page_chunk)
            if page_chunk > num_pages:
                page_chunk = num_pages
            page_begin = cutlass.Int32(0)
            while page_begin < num_pages:
                page_count = num_pages - page_begin
                if page_count > page_chunk:
                    page_count = page_chunk
                bucket = self._cost_bucket(q_rows, page_count)
                slot = cute.arch.atomic_add(
                    (mTaskCounts.iterator + bucket).llvm_ptr,
                    cutlass.Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
                if slot < task_capacity:
                    mTaskDescriptors[bucket, slot, 0] = q_start + q_local_begin
                    mTaskDescriptors[bucket, slot, 1] = (
                        seq_k - seq_q + q_local_begin
                    )
                    mTaskDescriptors[bucket, slot, 2] = page_begin
                    mTaskDescriptors[bucket, slot, 3] = self._pack_descriptor(
                        batch_idx,
                        q_rows,
                        page_count,
                        bucket,
                    )
                else:
                    cute.arch.atomic_exch(
                        mPlanError.iterator.llvm_ptr,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                page_begin += page_count


__all__ = [
    "PrefillIndexerPlanBuild",
    "PrefillIndexerPlanReset",
]
