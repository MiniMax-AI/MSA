"""Build reusable packed-varlen task metadata for the MSA v1 indexer."""

from typing import Optional

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

from msa_v1._common.utils import warp_prefix_sum


class M3IndexerScheduleSm100:
    """Expand packed Q fragments into 64-row indexer tasks."""

    def __init__(self, *, q_per_cluster: int = 64) -> None:
        if q_per_cluster != 64:
            raise ValueError("MSA v1 indexer schedule requires q_per_cluster=64")
        self.q_per_cluster = q_per_cluster
        self.threads_per_cta = cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mTaskBatchIdx: cute.Tensor,
        mTaskQLocalBegin: cute.Tensor,
        batch: cutlass.Int32,
        num_task_slots: cutlass.Int32,
        mFragmentIndices: Optional[cute.Tensor] = None,
        stream: cuda.CUstream = None,
    ):
        self.kernel(
            mCuSeqlensQ,
            mCuSeqlensK,
            mTaskBatchIdx,
            mTaskQLocalBegin,
            batch,
            num_task_slots,
            mFragmentIndices,
        ).launch(
            grid=(1, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _lpt_lane_metadata(
        self,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        batch: cutlass.Int32,
        mFragmentIndices: Optional[cute.Tensor],
    ) -> tuple[cutlass.Int32, cutlass.Int32]:
        """Sort up to 31 fragments by descending effective KV length."""

        lane_idx = cute.arch.lane_idx()
        lane_batch_idx = lane_idx
        num_tasks = cutlass.Int32(0)
        priority = cutlass.Int32(-1)
        if lane_idx < cutlass.Int32(cute.arch.WARP_SIZE - 1) and lane_idx < batch:
            seq_q = mCuSeqlensQ[lane_idx + cutlass.Int32(1)] - mCuSeqlensQ[lane_idx]
            num_tasks = cute.ceil_div(seq_q, self.q_per_cluster)
            k_start_idx = (
                lane_idx
                if cutlass.const_expr(mFragmentIndices is None)
                else mFragmentIndices[lane_idx]
            )
            priority = mCuSeqlensK[lane_idx + cutlass.Int32(1)] - mCuSeqlensK[
                k_start_idx
            ]

        for sort_span in (2, 4, 8, 16, 32):
            sort_stride = sort_span // 2
            while sort_stride > 0:
                partner_lane = lane_idx ^ cutlass.Int32(sort_stride)
                partner_priority = cute.arch.shuffle_sync(priority, partner_lane)
                partner_batch_idx = cute.arch.shuffle_sync(lane_batch_idx, partner_lane)
                partner_num_tasks = cute.arch.shuffle_sync(num_tasks, partner_lane)
                self_precedes_partner = priority > partner_priority
                if priority == partner_priority:
                    self_precedes_partner = lane_batch_idx < partner_batch_idx
                lane_is_lower = (
                    lane_idx // cutlass.Int32(sort_stride)
                ) % cutlass.Int32(2) == cutlass.Int32(0)
                sort_descending = (
                    lane_idx // cutlass.Int32(sort_span)
                ) % cutlass.Int32(2) == cutlass.Int32(0)
                if self_precedes_partner != (lane_is_lower == sort_descending):
                    priority = partner_priority
                    lane_batch_idx = partner_batch_idx
                    num_tasks = partner_num_tasks
                sort_stride //= 2
        return lane_batch_idx, num_tasks

    @cute.kernel
    def kernel(
        self,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mTaskBatchIdx: cute.Tensor,
        mTaskQLocalBegin: cute.Tensor,
        batch: cutlass.Int32,
        num_task_slots: cutlass.Int32,
        mFragmentIndices: Optional[cute.Tensor],
    ):
        lane_idx = cute.arch.lane_idx()
        lpt_lane_batch_idx = lane_idx
        lpt_num_tasks = cutlass.Int32(0)
        if batch <= cutlass.Int32(cute.arch.WARP_SIZE - 1):
            lpt_lane_batch_idx, lpt_num_tasks = self._lpt_lane_metadata(
                mCuSeqlensQ,
                mCuSeqlensK,
                batch,
                mFragmentIndices,
            )

        task_cursor = cutlass.Int32(0)
        batch_group_begin = cutlass.Int32(0)
        while batch_group_begin < batch:
            lane_batch_idx = lpt_lane_batch_idx
            num_tasks = lpt_num_tasks
            if batch > cutlass.Int32(cute.arch.WARP_SIZE - 1):
                lane_batch_idx = batch_group_begin + lane_idx
                num_tasks = cutlass.Int32(0)
                if (
                    lane_idx < cutlass.Int32(cute.arch.WARP_SIZE - 1)
                    and lane_batch_idx < batch
                ):
                    seq_q = (
                        mCuSeqlensQ[lane_batch_idx + cutlass.Int32(1)]
                        - mCuSeqlensQ[lane_batch_idx]
                    )
                    num_tasks = cute.ceil_div(seq_q, self.q_per_cluster)

            cumulative = warp_prefix_sum(num_tasks, lane_idx)
            task_begin = task_cursor + cumulative - num_tasks
            for task_rank in cutlass.range(num_tasks, unroll=1):
                task_idx = task_begin + task_rank
                q_tile_rank = num_tasks - cutlass.Int32(1) - task_rank
                mTaskBatchIdx[task_idx] = lane_batch_idx
                mTaskQLocalBegin[task_idx] = q_tile_rank * cutlass.Int32(
                    self.q_per_cluster
                )

            task_cursor += cute.arch.shuffle_sync(
                cumulative,
                cute.arch.WARP_SIZE - 1,
            )
            if batch <= cutlass.Int32(cute.arch.WARP_SIZE - 1):
                batch_group_begin = batch
            else:
                batch_group_begin += cutlass.Int32(cute.arch.WARP_SIZE - 1)

        padding_task_idx = task_cursor + lane_idx
        while padding_task_idx < num_task_slots:
            mTaskBatchIdx[padding_task_idx] = cutlass.Int32(-1)
            mTaskQLocalBegin[padding_task_idx] = cutlass.Int32(0)
            padding_task_idx += cutlass.Int32(cute.arch.WARP_SIZE)


__all__ = ["M3IndexerScheduleSm100"]
