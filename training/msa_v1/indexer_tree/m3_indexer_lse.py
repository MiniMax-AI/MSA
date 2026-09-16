"""SM100/SM103 selected-block LSE gather for MiniMax-M3."""

import math
from typing import Optional

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

    def __init__(self, *, rebase_block_ids: bool = False) -> None:
        self.num_index_heads = _NUM_INDEX_HEADS
        self.logical_q_tile = _LOGICAL_Q_TILE
        self.physical_m_tile = _PHYSICAL_M_TILE
        self.topk = _TOPK
        self.rebase_block_ids = rebase_block_ids

    @cute.jit
    def __call__(
        self,
        mScore: cute.Tensor,
        mBlockSum: cute.Tensor,
        mPartialBlockIndices: cute.Tensor,
        mFullBlockIndices: cute.Tensor,
        mTopkIdx: cute.Tensor,
        mSelectedLse: cute.Tensor,
        mBlockBases: Optional[cute.Tensor],
        q_len: cutlass.Int32,
        stream: cuda.CUstream = None,
    ):
        stat_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float32,
            num_bits_per_copy=cutlass.Float32.width,
        )
        stat_tiled_copy = cute.make_tiled_copy_tv(
            stat_copy_atom,
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
            stat_tiled_copy,
            output_copy_atom,
            lse_tiled_copy,
            mScore,
            mBlockSum,
            mPartialBlockIndices,
            mFullBlockIndices,
            mTopkIdx,
            mSelectedLse,
            mBlockBases,
            q_len,
        ).launch(
            grid=(cute.ceil_div(q_len, self.logical_q_tile), 1, 1),
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
        tRrStat = cute.make_rmem_tensor(tRgStat.layout, cutlass.Float32)
        cute.copy(stat_tiled_copy, tRgStat, tRrStat)
        return tRrStat[0]

    @cute.jit
    def _store_selected_lse(
        self,
        selected_lse: cutlass.Float32,
        tidx: cutlass.Int32,
        q_tile: cutlass.Int32,
        lse_tiled_copy: cute.TiledCopy,
        mSelectedLse: cute.Tensor,
    ) -> None:
        gSelectedLse = cute.local_tile(
            mSelectedLse,
            (self.num_index_heads, self.logical_q_tile),
            (cutlass.Int32(0), q_tile),
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
        stat_tiled_copy: cute.TiledCopy,
        output_copy_atom: cute.CopyAtom,
        lse_tiled_copy: cute.TiledCopy,
        mScore: cute.Tensor,
        mBlockSum: cute.Tensor,
        mPartialBlockIndices: cute.Tensor,
        mFullBlockIndices: cute.Tensor,
        mTopkIdx: cute.Tensor,
        mSelectedLse: cute.Tensor,
        mBlockBases: Optional[cute.Tensor],
        q_len: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_tile, _, _ = cute.arch.block_idx()
        q_in_tile = tidx // cutlass.Int32(self.num_index_heads)
        q_head = tidx - q_in_tile * cutlass.Int32(self.num_index_heads)
        q_global = q_tile * cutlass.Int32(self.logical_q_tile) + q_in_tile
        q_valid = q_global < q_len
        num_partial_tiles = cutlass.Int32(cute.size(mPartialBlockIndices, mode=[1]))

        selected_max = -cutlass.Float32.inf
        selected_sum = cutlass.Float32(0.0)
        has_value = cutlass.Boolean(False)

        if q_valid:
            block_base = cutlass.Int32(0)
            if cutlass.const_expr(self.rebase_block_ids):
                block_base = mBlockBases[q_global]

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
                            stat_tiled_copy,
                            mScore,
                        )
                        block_sum = self._load_stat(
                            tile_storage_idx,
                            tidx,
                            stat_tiled_copy,
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

                        block_id = cutlass.Int32(0)
                        if tile_storage_idx < num_partial_tiles:
                            block_id = mPartialBlockIndices[
                                (cutlass.Int32(0), tile_storage_idx)
                            ]
                        else:
                            block_id = mFullBlockIndices[
                                (
                                    cutlass.Int32(0),
                                    tile_storage_idx - num_partial_tiles,
                                )
                            ]
                        if cutlass.const_expr(self.rebase_block_ids):
                            block_id -= block_base
                        rOutput[value_idx] = block_id

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
                q_tile,
                lse_tiled_copy,
                mSelectedLse,
            )


__all__ = ["M3IndexerLseSm100"]
