"""SM100/SM103 selected-block LSE gather for MiniMax-M3."""

import math

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math
import cuda.bindings.driver as cuda

_NUM_INDEX_HEADS = 4
_LOGICAL_Q_TILE = 64
_PHYSICAL_M_TILE = 256
_TOPK = 16
_LOG2_E = math.log2(math.e)
_LN_2 = math.log(2.0)


class M3IndexerLseSm100:
    """Gather selected block statistics and finalize ids and natural-log LSE."""

    def __init__(self, *, use_fp16_score: bool = False) -> None:
        self.num_index_heads = _NUM_INDEX_HEADS
        self.logical_q_tile = _LOGICAL_Q_TILE
        self.physical_m_tile = _PHYSICAL_M_TILE
        self.topk = _TOPK
        self.score_dtype = cutlass.Float16 if use_fp16_score else cutlass.Float32

    @cute.jit
    def __call__(
        self,
        mScore: cute.Tensor,
        mBlockSum: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mTaskBatchIdx: cute.Tensor,
        mTaskQLocalBegin: cute.Tensor,
        mTopkIdx: cute.Tensor,
        mSelectedLse: cute.Tensor,
        num_task_slots: cutlass.Int32,
        max_k_blocks: cutlass.Int32,
        stream: cuda.CUstream = None,
    ):
        score_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.score_dtype,
            num_bits_per_copy=self.score_dtype.width,
        )
        score_tiled_copy = cute.make_tiled_copy_tv(
            score_copy_atom,
            cute.make_layout((self.physical_m_tile,)),
            cute.make_layout((1,)),
        )
        block_sum_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float32,
            num_bits_per_copy=cutlass.Float32.width,
        )
        block_sum_tiled_copy = cute.make_tiled_copy_tv(
            block_sum_copy_atom,
            cute.make_layout((self.physical_m_tile,)),
            cute.make_layout((1,)),
        )
        output_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Int32,
            num_bits_per_copy=128,
        )
        lse_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float32,
            num_bits_per_copy=cutlass.Float32.width,
        )
        lse_tiled_copy = cute.make_tiled_copy_tv(
            lse_copy_atom,
            cute.make_layout(
                (self.num_index_heads, self.logical_q_tile),
                stride=(1, self.num_index_heads),
            ),
            cute.make_layout((1, 1)),
        )
        self.kernel(
            score_tiled_copy,
            block_sum_tiled_copy,
            output_copy_atom,
            lse_tiled_copy,
            mScore,
            mBlockSum,
            mCuSeqlensQ,
            mTaskBatchIdx,
            mTaskQLocalBegin,
            mTopkIdx,
            mSelectedLse,
            max_k_blocks,
        ).launch(
            grid=(num_task_slots, 1, 1),
            block=(self.physical_m_tile, 1, 1),
            stream=stream,
            min_blocks_per_mp=4,
        )

    @cute.jit
    def _load_stat(
        self,
        tile_storage_idx: cutlass.Int32,
        tidx: cutlass.Int32,
        stat_tiled_copy: cute.TiledCopy,
        mStat: cute.Tensor,
    ) -> cutlass.Float32:
        stat_iter = mStat.iterator + cute.crd2idx(
            (
                tile_storage_idx,
                cutlass.Int32(0),
                cutlass.Int32(0),
            ),
            mStat.layout,
        )
        gStat = cute.make_tensor(stat_iter, (self.physical_m_tile,))
        stat_thr_copy = stat_tiled_copy.get_slice(tidx)
        tRgStat = stat_thr_copy.partition_S(gStat)
        tRrStat = cute.make_rmem_tensor(tRgStat.layout, mStat.element_type)
        cute.copy(stat_tiled_copy, tRgStat, tRrStat)
        return tRrStat[0].to(cutlass.Float32)

    @cute.jit
    def _store_selected_lse(
        self,
        selected_lse: cutlass.Float32,
        tidx: cutlass.Int32,
        q_global_begin: cutlass.Int32,
        lse_tiled_copy: cute.TiledCopy,
        mSelectedLse: cute.Tensor,
    ) -> None:
        mSelectedLseTask = cute.domain_offset(
            (cutlass.Int32(0), q_global_begin),
            mSelectedLse,
        )
        gSelectedLse = cute.local_tile(
            mSelectedLseTask,
            (self.num_index_heads, self.logical_q_tile),
            (cutlass.Int32(0), cutlass.Int32(0)),
        )
        lse_thr_copy = lse_tiled_copy.get_slice(tidx)
        tRgSelectedLse = lse_thr_copy.partition_D(gSelectedLse)
        tRrSelectedLse = cute.make_rmem_tensor(
            tRgSelectedLse.layout,
            cutlass.Float32,
        )
        tRrSelectedLse[0] = selected_lse
        cute.copy(lse_tiled_copy, tRrSelectedLse, tRgSelectedLse)

    @cute.kernel
    def kernel(
        self,
        score_tiled_copy: cute.TiledCopy,
        block_sum_tiled_copy: cute.TiledCopy,
        output_copy_atom: cute.CopyAtom,
        lse_tiled_copy: cute.TiledCopy,
        mScore: cute.Tensor,
        mBlockSum: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mTaskBatchIdx: cute.Tensor,
        mTaskQLocalBegin: cute.Tensor,
        mTopkIdx: cute.Tensor,
        mSelectedLse: cute.Tensor,
        max_k_blocks: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        task_idx, _, _ = cute.arch.block_idx()
        lane_idx = cute.arch.lane_idx()
        q_in_tile = tidx // cutlass.Int32(self.num_index_heads)
        q_head = tidx - q_in_tile * cutlass.Int32(self.num_index_heads)

        batch_idx = cutlass.Int32(-1)
        q_local_begin = cutlass.Int32(0)
        if lane_idx == cutlass.Int32(0):
            batch_idx = mTaskBatchIdx[task_idx]
            q_local_begin = mTaskQLocalBegin[task_idx]
        batch_idx = cute.arch.shuffle_sync(batch_idx, 0)
        q_local_begin = cute.arch.shuffle_sync(q_local_begin, 0)

        q_start = cutlass.Int32(0)
        seq_q = cutlass.Int32(0)
        if lane_idx == cutlass.Int32(0) and batch_idx >= cutlass.Int32(0):
            q_start = mCuSeqlensQ[batch_idx]
            q_end = mCuSeqlensQ[batch_idx + cutlass.Int32(1)]
            seq_q = q_end - q_start
        q_start = cute.arch.shuffle_sync(q_start, 0)
        seq_q = cute.arch.shuffle_sync(seq_q, 0)

        q_local = q_local_begin + q_in_tile
        q_global = q_start + q_local
        q_valid = q_local < seq_q
        tile_storage_base = task_idx * max_k_blocks

        selected_max = -cutlass.Float32.inf
        selected_sum = cutlass.Float32(0.0)
        has_value = cutlass.Boolean(False)

        if q_valid:
            for vector_idx in cutlass.range_constexpr(self.topk // 4):
                slot_base = vector_idx * 4
                output_iter = mTopkIdx.iterator + cute.crd2idx(
                    (
                        q_head,
                        q_global,
                        cutlass.Int32(slot_base),
                    ),
                    mTopkIdx.layout,
                )
                output_ptr = cute.make_ptr(
                    mTopkIdx.element_type,
                    output_iter.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                gOutput = cute.make_tensor(output_ptr, (4,))
                rOutput = cute.make_rmem_tensor((4,), cutlass.Int32)
                output_tiled_copy = cute.make_cotiled_copy(
                    output_copy_atom,
                    cute.make_layout((1, 4)),
                    rOutput.layout,
                )
                output_thr_copy = output_tiled_copy.get_slice(0)
                tRgOutput = output_thr_copy.partition_S(gOutput)
                tRrOutput = output_thr_copy.partition_D(rOutput)
                cute.copy(output_copy_atom, tRgOutput, tRrOutput)

                for value_idx in cutlass.range_constexpr(4):
                    tile_storage_idx = rOutput[value_idx]
                    if tile_storage_idx >= cutlass.Int32(0):
                        score = self._load_stat(
                            tile_storage_idx,
                            tidx,
                            score_tiled_copy,
                            mScore,
                        )
                        block_sum = self._load_stat(
                            tile_storage_idx,
                            tidx,
                            block_sum_tiled_copy,
                            mBlockSum,
                        )
                        if not has_value:
                            selected_max = score
                            selected_sum = block_sum
                            has_value = cutlass.Boolean(True)
                        elif score > selected_max:
                            selected_sum = block_sum + selected_sum * cute_math.exp2(
                                (selected_max - score) * cutlass.Float32(_LOG2_E),
                                fastmath=True,
                            )
                            selected_max = score
                        else:
                            selected_sum = selected_sum + block_sum * cute_math.exp2(
                                (score - selected_max) * cutlass.Float32(_LOG2_E),
                                fastmath=True,
                            )

                        rOutput[value_idx] = tile_storage_idx - tile_storage_base

                tRrOutput = output_thr_copy.partition_S(rOutput)
                tRgOutput = output_thr_copy.partition_D(gOutput)
                cute.copy(output_copy_atom, tRrOutput, tRgOutput)

            selected_lse = -cutlass.Float32.inf
            if has_value and selected_sum > cutlass.Float32(0.0):
                selected_lse = selected_max + cute_math.log2(
                    selected_sum,
                    fastmath=True,
                ) * cutlass.Float32(_LN_2)
            self._store_selected_lse(
                selected_lse,
                tidx,
                q_start + q_local_begin,
                lse_tiled_copy,
                mSelectedLse,
            )


__all__ = ["M3IndexerLseSm100"]
