"""SM100/SM103 tile-major TopK and selected LSE for MiniMax-M3."""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math
from cutlass._mlir.dialects import arith, llvm
import cuda.bindings.driver as cuda

from msa_v1._common.indexer_topk_key import (
    fp16_from_topk_key,
    pack_fp16_topk_key,
)

_NUM_INDEX_HEADS = 4
_LOGICAL_Q_TILE = 64
_PHYSICAL_M_TILE = 256
_K_TILE = 128
_TOPK = 16
_DYNAMIC_TOPK = _TOPK - 1
_LOG2_E = math.log2(math.e)
_LN_2 = math.log(2.0)


def _float_as_uint32(value: cutlass.Float32) -> cutlass.Uint32:
    """Reinterpret one FP32 register as an unsigned ordered-key input."""

    return llvm.bitcast(cutlass.Uint32.mlir_type, value.ir_value())


def _uint32_as_float(value: cutlass.Uint32) -> cutlass.Float32:
    """Reinterpret one unsigned register as FP32."""

    return llvm.bitcast(cutlass.Float32.mlir_type, value.ir_value())


class M3IndexerTopkSm100:
    """Select exact TopK ids and merge their visible-token LSE statistics."""

    def __init__(
        self,
        *,
        deterministic: bool = False,
        small_plan: bool = False,
        gather_lse: bool = False,
        use_fp16_score: bool = False,
        rebase_block_ids: bool = False,
    ) -> None:
        self.num_index_heads = _NUM_INDEX_HEADS
        self.logical_q_tile = _LOGICAL_Q_TILE
        self.physical_m_tile = _PHYSICAL_M_TILE
        self.k_tile = _K_TILE
        self.topk = _TOPK
        self.dynamic_topk = _DYNAMIC_TOPK
        self.neg_inf = -1.0e30
        self.deterministic = deterministic
        self.small_plan = small_plan
        self.gather_lse = gather_lse
        self.use_fp16_score = use_fp16_score
        self.rebase_block_ids = rebase_block_ids
        self.score_dtype = cutlass.Float16 if use_fp16_score else cutlass.Float32

    @cute.jit
    def __call__(
        self,
        mScore: cute.Tensor,
        mBlockSum: cute.Tensor,
        mPartialOffsets: cute.Tensor,
        mPartialBlockIndices: cute.Tensor,
        mFullOffsets: cute.Tensor,
        mFullBlockIndices: cute.Tensor,
        mTopkIdx: cute.Tensor,
        mSelectedLse: cute.Tensor,
        mLocalBlockPositions: cute.Tensor,
        mBlockBases: Optional[cute.Tensor],
        q_len: cutlass.Int32,
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
            mPartialOffsets,
            mPartialBlockIndices,
            mFullOffsets,
            mFullBlockIndices,
            mTopkIdx,
            mSelectedLse,
            mLocalBlockPositions,
            mBlockBases,
            q_len,
        ).launch(
            grid=(
                cute.ceil_div(q_len, self.logical_q_tile) * cutlass.Int32(2)
                if cutlass.const_expr(self.gather_lse and not self.deterministic)
                else cute.ceil_div(q_len, self.logical_q_tile),
                1,
                1,
            ),
            block=(self.physical_m_tile, 1, 1),
            stream=stream,
            min_blocks_per_mp=3 if self.small_plan or self.deterministic else 4,
        )

    @cute.jit
    def _ordered_fp32_key(self, score: cutlass.Float32) -> cutlass.Uint32:
        """Map FP32 values to monotonically ordered unsigned keys."""

        if score != score:
            score = cutlass.Float32(self.neg_inf)
        if score == cutlass.Float32(0.0):
            score = cutlass.Float32(0.0)
        bits = cutlass.Uint32(_float_as_uint32(score))
        key = cutlass.Uint32(0)
        if bits & cutlass.Uint32(0x8000_0000):
            key = bits ^ cutlass.Uint32(0xFFFF_FFFF)
        else:
            key = bits | cutlass.Uint32(0x8000_0000)
        return key

    @cute.jit
    def _score_from_ordered_key(
        self,
        score_key: cutlass.Uint32,
    ) -> cutlass.Float32:
        """Recover the score represented by an ordered key as FP32."""

        if cutlass.const_expr(self.use_fp16_score):
            return fp16_from_topk_key(score_key).to(cutlass.Float32)

        bits = cutlass.Uint32(0)
        if score_key & cutlass.Uint32(0x8000_0000):
            bits = score_key & cutlass.Uint32(0x7FFF_FFFF)
        else:
            bits = score_key ^ cutlass.Uint32(0xFFFF_FFFF)
        return _uint32_as_float(bits)

    @cute.jit
    def _insert_topk_ordered(
        self,
        rScoreKeys: cute.Tensor,
        rBlockIds: cute.Tensor,
        rBlockSums: cute.Tensor,
        score_key: cutlass.Uint32,
        block_id: cutlass.Int32,
        block_sum: cutlass.Float32,
    ) -> None:
        """Insert a score-descending/id-ascending tuple into register Top15."""

        candidate_score_key = score_key
        candidate_block_id = block_id
        candidate_sum = block_sum
        for slot in cutlass.range_constexpr(self.dynamic_topk):
            old_score_key = rScoreKeys[slot]
            old_block_id = rBlockIds[slot]
            old_sum = rBlockSums[slot]
            take = candidate_score_key > old_score_key
            if candidate_score_key == old_score_key:
                take = candidate_block_id < old_block_id
            rScoreKeys[slot] = cutlass.Uint32(
                arith.select(
                    take.ir_value(),
                    candidate_score_key.ir_value(),
                    old_score_key.ir_value(),
                )
            )
            candidate_score_key = cutlass.Uint32(
                arith.select(
                    take.ir_value(),
                    old_score_key.ir_value(),
                    candidate_score_key.ir_value(),
                )
            )
            rBlockIds[slot] = cutlass.Int32(
                arith.select(
                    take.ir_value(),
                    candidate_block_id.ir_value(),
                    old_block_id.ir_value(),
                )
            )
            candidate_block_id = cutlass.Int32(
                arith.select(
                    take.ir_value(),
                    old_block_id.ir_value(),
                    candidate_block_id.ir_value(),
                )
            )
            rBlockSums[slot] = cutlass.Float32(
                arith.select(
                    take.ir_value(),
                    candidate_sum.ir_value(),
                    old_sum.ir_value(),
                )
            )
            candidate_sum = cutlass.Float32(
                arith.select(
                    take.ir_value(),
                    old_sum.ir_value(),
                    candidate_sum.ir_value(),
                )
            )

    @cute.jit
    def _insert_topk_ordered_index(
        self,
        rScoreKeys: cute.Tensor,
        rBlockIds: cute.Tensor,
        rStorageIndices: cute.Tensor,
        score_key: cutlass.Uint32,
        block_id: cutlass.Int32,
        storage_idx: cutlass.Int32,
    ) -> None:
        """Insert an ordered candidate while retaining its storage index."""

        candidate_score_key = score_key
        candidate_block_id = block_id
        candidate_storage_idx = storage_idx
        for slot in cutlass.range_constexpr(self.dynamic_topk):
            old_score_key = rScoreKeys[slot]
            old_block_id = rBlockIds[slot]
            old_storage_idx = rStorageIndices[slot]
            take = candidate_score_key > old_score_key
            if candidate_score_key == old_score_key:
                take = candidate_block_id < old_block_id
            rScoreKeys[slot] = cutlass.Uint32(
                arith.select(
                    take.ir_value(),
                    candidate_score_key.ir_value(),
                    old_score_key.ir_value(),
                )
            )
            candidate_score_key = cutlass.Uint32(
                arith.select(
                    take.ir_value(),
                    old_score_key.ir_value(),
                    candidate_score_key.ir_value(),
                )
            )
            rBlockIds[slot] = cutlass.Int32(
                arith.select(
                    take.ir_value(),
                    candidate_block_id.ir_value(),
                    old_block_id.ir_value(),
                )
            )
            candidate_block_id = cutlass.Int32(
                arith.select(
                    take.ir_value(),
                    old_block_id.ir_value(),
                    candidate_block_id.ir_value(),
                )
            )
            rStorageIndices[slot] = cutlass.Int32(
                arith.select(
                    take.ir_value(),
                    candidate_storage_idx.ir_value(),
                    old_storage_idx.ir_value(),
                )
            )
            candidate_storage_idx = cutlass.Int32(
                arith.select(
                    take.ir_value(),
                    old_storage_idx.ir_value(),
                    candidate_storage_idx.ir_value(),
                )
            )

    @cute.jit
    def _block_id(
        self,
        tile_idx: cutlass.Int32,
        partial_begin: cutlass.Int32,
        partial_count: cutlass.Int32,
        full_begin: cutlass.Int32,
        mPartialBlockIndices: cute.Tensor,
        mFullBlockIndices: cute.Tensor,
    ) -> cutlass.Int32:
        block_id = cutlass.Int32(0)
        if tile_idx < partial_count:
            block_id = mPartialBlockIndices[
                (cutlass.Int32(0), partial_begin + tile_idx)
            ]
        else:
            block_id = mFullBlockIndices[
                (
                    cutlass.Int32(0),
                    full_begin + tile_idx - partial_count,
                )
            ]
        return block_id

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
    def _load_score(
        self,
        tile_storage_idx: cutlass.Int32,
        tidx: cutlass.Int32,
        score_tiled_copy: cute.TiledCopy,
        mScore: cute.Tensor,
        block_id: cutlass.Int32,
    ) -> tuple[cutlass.Float32, cutlass.Uint32]:
        """Load a stored score and construct its compile-time-selected key."""

        stat_iter = mScore.iterator + cute.crd2idx(
            (
                tile_storage_idx,
                cutlass.Int32(0),
                cutlass.Int32(0),
            ),
            mScore.layout,
        )
        gScore = cute.make_tensor(stat_iter, (self.physical_m_tile,))
        score_thr_copy = score_tiled_copy.get_slice(tidx)
        tRgScore = score_thr_copy.partition_S(gScore)
        tRrScore = cute.make_rmem_tensor(tRgScore.layout, mScore.element_type)
        cute.copy(score_tiled_copy, tRgScore, tRrScore)
        if cutlass.const_expr(self.use_fp16_score):
            score_key = pack_fp16_topk_key(tRrScore[0], block_id)
        else:
            score_key = self._ordered_fp32_key(tRrScore[0])
        return tRrScore[0].to(cutlass.Float32), score_key

    @cute.jit
    def _is_valid_score(self, score: cutlass.Float32) -> cutlass.Boolean:
        """Return whether K1 produced a valid score for this row and tile."""

        if cutlass.const_expr(self.use_fp16_score):
            return score != -cutlass.Float32.inf
        return score != cutlass.Float32(self.neg_inf)

    @cute.jit
    def _consume_candidate_ordered(
        self,
        rScoreKeys: cute.Tensor,
        rBlockIds: cute.Tensor,
        rBlockSums: cute.Tensor,
        score: cutlass.Float32,
        score_key: cutlass.Uint32,
        block_sum: cutlass.Float32,
        block_id: cutlass.Int32,
        local_block: cutlass.Int32,
        nonlocal_count: cutlass.Int32,
        local_seen: cutlass.Boolean,
        local_score: cutlass.Float32,
        local_block_sum: cutlass.Float32,
    ) -> tuple[
        cutlass.Int32,
        cutlass.Boolean,
        cutlass.Float32,
        cutlass.Float32,
    ]:
        if block_id == local_block:
            if not local_seen:
                local_seen = cutlass.Boolean(True)
                local_score = score
                local_block_sum = block_sum
        else:
            worst_score_key = rScoreKeys[self.dynamic_topk - 1]
            worst_block_id = rBlockIds[self.dynamic_topk - 1]
            take = score_key > worst_score_key
            if score_key == worst_score_key:
                take = block_id < worst_block_id
            if take:
                self._insert_topk_ordered(
                    rScoreKeys,
                    rBlockIds,
                    rBlockSums,
                    score_key,
                    block_id,
                    block_sum,
                )
            nonlocal_count += cutlass.Int32(1)
        return nonlocal_count, local_seen, local_score, local_block_sum

    @cute.jit
    def _consume_candidate_ordered_index(
        self,
        rScoreKeys: cute.Tensor,
        rBlockIds: cute.Tensor,
        rStorageIndices: cute.Tensor,
        score_key: cutlass.Uint32,
        block_id: cutlass.Int32,
        storage_idx: cutlass.Int32,
        local_block: cutlass.Int32,
        nonlocal_count: cutlass.Int32,
        local_seen: cutlass.Boolean,
        local_storage_idx: cutlass.Int32,
    ) -> tuple[cutlass.Int32, cutlass.Boolean, cutlass.Int32]:
        """Maintain deterministic Top15 and the selected storage indices."""

        if block_id == local_block:
            if not local_seen:
                local_seen = cutlass.Boolean(True)
                local_storage_idx = storage_idx
        else:
            worst_score_key = rScoreKeys[self.dynamic_topk - 1]
            worst_block_id = rBlockIds[self.dynamic_topk - 1]
            take = score_key > worst_score_key
            if score_key == worst_score_key:
                take = block_id < worst_block_id
            if take:
                self._insert_topk_ordered_index(
                    rScoreKeys,
                    rBlockIds,
                    rStorageIndices,
                    score_key,
                    block_id,
                    storage_idx,
                )
            nonlocal_count += cutlass.Int32(1)
        return nonlocal_count, local_seen, local_storage_idx

    @cute.jit
    def _minimum_score_slot(
        self,
        rScoreKeys: cute.Tensor,
    ) -> tuple[cutlass.Uint32, cutlass.Int32]:
        """Find the lowest score key with a fixed-depth register reduction."""

        min_key_0 = rScoreKeys[0]
        min_slot_0 = cutlass.Int32(0)
        for slot in cutlass.range_constexpr(1, 4):
            key = rScoreKeys[slot]
            take = key < min_key_0
            min_key_0 = cutlass.Uint32(
                arith.select(
                    take.ir_value(),
                    key.ir_value(),
                    min_key_0.ir_value(),
                )
            )
            min_slot_0 = cutlass.Int32(
                arith.select(
                    take.ir_value(),
                    cutlass.Int32(slot).ir_value(),
                    min_slot_0.ir_value(),
                )
            )

        min_key_1 = rScoreKeys[4]
        min_slot_1 = cutlass.Int32(4)
        for slot in cutlass.range_constexpr(5, 8):
            key = rScoreKeys[slot]
            take = key < min_key_1
            min_key_1 = cutlass.Uint32(
                arith.select(
                    take.ir_value(),
                    key.ir_value(),
                    min_key_1.ir_value(),
                )
            )
            min_slot_1 = cutlass.Int32(
                arith.select(
                    take.ir_value(),
                    cutlass.Int32(slot).ir_value(),
                    min_slot_1.ir_value(),
                )
            )

        min_key_2 = rScoreKeys[8]
        min_slot_2 = cutlass.Int32(8)
        for slot in cutlass.range_constexpr(9, 12):
            key = rScoreKeys[slot]
            take = key < min_key_2
            min_key_2 = cutlass.Uint32(
                arith.select(
                    take.ir_value(),
                    key.ir_value(),
                    min_key_2.ir_value(),
                )
            )
            min_slot_2 = cutlass.Int32(
                arith.select(
                    take.ir_value(),
                    cutlass.Int32(slot).ir_value(),
                    min_slot_2.ir_value(),
                )
            )

        min_key_3 = rScoreKeys[12]
        min_slot_3 = cutlass.Int32(12)
        for slot in cutlass.range_constexpr(13, self.dynamic_topk):
            key = rScoreKeys[slot]
            take = key < min_key_3
            min_key_3 = cutlass.Uint32(
                arith.select(
                    take.ir_value(),
                    key.ir_value(),
                    min_key_3.ir_value(),
                )
            )
            min_slot_3 = cutlass.Int32(
                arith.select(
                    take.ir_value(),
                    cutlass.Int32(slot).ir_value(),
                    min_slot_3.ir_value(),
                )
            )

        take_1 = min_key_1 < min_key_0
        min_key_01 = cutlass.Uint32(
            arith.select(
                take_1.ir_value(),
                min_key_1.ir_value(),
                min_key_0.ir_value(),
            )
        )
        min_slot_01 = cutlass.Int32(
            arith.select(
                take_1.ir_value(),
                min_slot_1.ir_value(),
                min_slot_0.ir_value(),
            )
        )
        take_3 = min_key_3 < min_key_2
        min_key_23 = cutlass.Uint32(
            arith.select(
                take_3.ir_value(),
                min_key_3.ir_value(),
                min_key_2.ir_value(),
            )
        )
        min_slot_23 = cutlass.Int32(
            arith.select(
                take_3.ir_value(),
                min_slot_3.ir_value(),
                min_slot_2.ir_value(),
            )
        )
        take_23 = min_key_23 < min_key_01
        min_key = cutlass.Uint32(
            arith.select(
                take_23.ir_value(),
                min_key_23.ir_value(),
                min_key_01.ir_value(),
            )
        )
        min_slot = cutlass.Int32(
            arith.select(
                take_23.ir_value(),
                min_slot_23.ir_value(),
                min_slot_01.ir_value(),
            )
        )
        return min_key, min_slot

    @cute.jit
    def _consume_candidate_unordered_index(
        self,
        rScoreKeys: cute.Tensor,
        rStorageIndices: cute.Tensor,
        score_key: cutlass.Uint32,
        block_id: cutlass.Int32,
        storage_idx: cutlass.Int32,
        local_block: cutlass.Int32,
        worst_key: cutlass.Uint32,
        worst_slot: cutlass.Int32,
        nonlocal_count: cutlass.Int32,
        local_seen: cutlass.Boolean,
        local_storage_idx: cutlass.Int32,
    ) -> tuple[
        cutlass.Uint32,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Boolean,
        cutlass.Int32,
    ]:
        """Maintain an unordered Top15 reservoir of storage indices."""

        if block_id == local_block:
            if not local_seen:
                local_seen = cutlass.Boolean(True)
                local_storage_idx = storage_idx
        else:
            if score_key > worst_key:
                for slot in cutlass.range_constexpr(self.dynamic_topk):
                    replace = worst_slot == cutlass.Int32(slot)
                    old_key = rScoreKeys[slot]
                    old_storage_idx = rStorageIndices[slot]
                    rScoreKeys[slot] = cutlass.Uint32(
                        arith.select(
                            replace.ir_value(),
                            score_key.ir_value(),
                            old_key.ir_value(),
                        )
                    )
                    rStorageIndices[slot] = cutlass.Int32(
                        arith.select(
                            replace.ir_value(),
                            storage_idx.ir_value(),
                            old_storage_idx.ir_value(),
                        )
                    )
                worst_key, worst_slot = self._minimum_score_slot(rScoreKeys)
            nonlocal_count += cutlass.Int32(1)
        return (
            worst_key,
            worst_slot,
            nonlocal_count,
            local_seen,
            local_storage_idx,
        )

    @cute.jit
    def _consume_candidate_unordered_index_with_id(
        self,
        rScoreKeys: cute.Tensor,
        rBlockIds: cute.Tensor,
        rStorageIndices: cute.Tensor,
        score_key: cutlass.Uint32,
        block_id: cutlass.Int32,
        storage_idx: cutlass.Int32,
        local_block: cutlass.Int32,
        worst_key: cutlass.Uint32,
        worst_slot: cutlass.Int32,
        nonlocal_count: cutlass.Int32,
        local_storage_idx: cutlass.Int32,
    ) -> tuple[
        cutlass.Uint32,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Int32,
    ]:
        """Maintain Top15 ids while retaining selected storage indices."""

        if block_id == local_block:
            if local_storage_idx < cutlass.Int32(0):
                local_storage_idx = storage_idx
        else:
            if score_key > worst_key:
                for slot in cutlass.range_constexpr(self.dynamic_topk):
                    replace = worst_slot == cutlass.Int32(slot)
                    old_key = rScoreKeys[slot]
                    old_id = rBlockIds[slot]
                    old_storage_idx = rStorageIndices[slot]
                    rScoreKeys[slot] = cutlass.Uint32(
                        arith.select(
                            replace.ir_value(),
                            score_key.ir_value(),
                            old_key.ir_value(),
                        )
                    )
                    rBlockIds[slot] = cutlass.Int32(
                        arith.select(
                            replace.ir_value(),
                            block_id.ir_value(),
                            old_id.ir_value(),
                        )
                    )
                    rStorageIndices[slot] = cutlass.Int32(
                        arith.select(
                            replace.ir_value(),
                            storage_idx.ir_value(),
                            old_storage_idx.ir_value(),
                        )
                    )
                worst_key, worst_slot = self._minimum_score_slot(rScoreKeys)
            nonlocal_count += cutlass.Int32(1)
        return worst_key, worst_slot, nonlocal_count, local_storage_idx

    @cute.jit
    def _merge_unordered_index(
        self,
        rScoreKeys: cute.Tensor,
        rStorageIndices: cute.Tensor,
        score_key: cutlass.Uint32,
        storage_idx: cutlass.Int32,
        worst_key: cutlass.Uint32,
        worst_slot: cutlass.Int32,
    ) -> tuple[cutlass.Uint32, cutlass.Int32]:
        """Merge one pre-ranked candidate into an unordered Top15 reservoir."""

        if score_key > worst_key:
            for slot in cutlass.range_constexpr(self.dynamic_topk):
                replace = worst_slot == cutlass.Int32(slot)
                old_key = rScoreKeys[slot]
                old_storage_idx = rStorageIndices[slot]
                rScoreKeys[slot] = cutlass.Uint32(
                    arith.select(
                        replace.ir_value(),
                        score_key.ir_value(),
                        old_key.ir_value(),
                    )
                )
                rStorageIndices[slot] = cutlass.Int32(
                    arith.select(
                        replace.ir_value(),
                        storage_idx.ir_value(),
                        old_storage_idx.ir_value(),
                    )
                )
            worst_key, worst_slot = self._minimum_score_slot(rScoreKeys)
        return worst_key, worst_slot

    @cute.jit
    def _consume_candidate_unordered(
        self,
        rScoreKeys: cute.Tensor,
        rBlockIds: cute.Tensor,
        rBlockSums: cute.Tensor,
        score: cutlass.Float32,
        score_key: cutlass.Uint32,
        block_sum: cutlass.Float32,
        block_id: cutlass.Int32,
        local_block: cutlass.Int32,
        worst_key: cutlass.Uint32,
        worst_slot: cutlass.Int32,
        nonlocal_count: cutlass.Int32,
        local_seen: cutlass.Boolean,
        local_score: cutlass.Float32,
        local_block_sum: cutlass.Float32,
    ) -> tuple[
        cutlass.Uint32,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Boolean,
        cutlass.Float32,
        cutlass.Float32,
    ]:
        """Maintain an unordered exact Top15 score/statistics reservoir."""

        if block_id == local_block:
            if not local_seen:
                local_seen = cutlass.Boolean(True)
                local_score = score
                local_block_sum = block_sum
        else:
            if score_key > worst_key:
                for slot in cutlass.range_constexpr(self.dynamic_topk):
                    replace = worst_slot == cutlass.Int32(slot)
                    old_key = rScoreKeys[slot]
                    old_id = rBlockIds[slot]
                    old_sum = rBlockSums[slot]
                    rScoreKeys[slot] = cutlass.Uint32(
                        arith.select(
                            replace.ir_value(),
                            score_key.ir_value(),
                            old_key.ir_value(),
                        )
                    )
                    rBlockIds[slot] = cutlass.Int32(
                        arith.select(
                            replace.ir_value(),
                            block_id.ir_value(),
                            old_id.ir_value(),
                        )
                    )
                    rBlockSums[slot] = cutlass.Float32(
                        arith.select(
                            replace.ir_value(),
                            block_sum.ir_value(),
                            old_sum.ir_value(),
                        )
                    )
                worst_key, worst_slot = self._minimum_score_slot(rScoreKeys)
            nonlocal_count += cutlass.Int32(1)
        return (
            worst_key,
            worst_slot,
            nonlocal_count,
            local_seen,
            local_score,
            local_block_sum,
        )

    @cute.jit
    def _selected_lse(
        self,
        rScoreKeys: cute.Tensor,
        rBlockSums: cute.Tensor,
        nonlocal_count: cutlass.Int32,
        local_seen: cutlass.Boolean,
        local_score: cutlass.Float32,
        local_block_sum: cutlass.Float32,
    ) -> cutlass.Float32:
        """Merge selected block statistics from the register reservoir."""

        valid_nonlocal = nonlocal_count
        if valid_nonlocal > cutlass.Int32(self.dynamic_topk):
            valid_nonlocal = cutlass.Int32(self.dynamic_topk)

        selected_max = -cutlass.Float32.inf
        for slot in cutlass.range_constexpr(self.dynamic_topk):
            if cutlass.Int32(slot) < valid_nonlocal:
                score = self._score_from_ordered_key(rScoreKeys[slot])
                selected_max = cute.arch.fmax(selected_max, score)
        if local_seen:
            selected_max = cute.arch.fmax(selected_max, local_score)

        selected_sum = cutlass.Float32(0.0)
        for slot in cutlass.range_constexpr(self.dynamic_topk):
            if cutlass.Int32(slot) < valid_nonlocal:
                score = self._score_from_ordered_key(rScoreKeys[slot])
                selected_sum = selected_sum + rBlockSums[slot] * cute_math.exp2(
                    (score - selected_max) * cutlass.Float32(_LOG2_E),
                    fastmath=True,
                )
        if local_seen:
            selected_sum = selected_sum + local_block_sum * cute_math.exp2(
                (local_score - selected_max) * cutlass.Float32(_LOG2_E),
                fastmath=True,
            )

        selected_lse = -cutlass.Float32.inf
        if selected_sum > cutlass.Float32(0.0):
            selected_lse = selected_max + cute_math.log2(
                selected_sum,
                fastmath=True,
            ) * cutlass.Float32(_LN_2)
        return selected_lse

    @cute.jit
    def _selected_lse_from_storage_scalar(
        self,
        rScoreKeys: cute.Tensor,
        rStorageIndices: cute.Tensor,
        nonlocal_count: cutlass.Int32,
        local_storage_idx: cutlass.Int32,
        tidx: cutlass.Int32,
        score_tiled_copy: cute.TiledCopy,
        block_sum_tiled_copy: cute.TiledCopy,
        mScore: cute.Tensor,
        mBlockSum: cute.Tensor,
    ) -> cutlass.Float32:
        """Load statistics only for the final Top15 and merge LSE in FP32."""

        valid_nonlocal = nonlocal_count
        if valid_nonlocal > cutlass.Int32(self.dynamic_topk):
            valid_nonlocal = cutlass.Int32(self.dynamic_topk)
        local_seen = local_storage_idx >= cutlass.Int32(0)

        selected_max = -cutlass.Float32.inf
        for slot in cutlass.range_constexpr(self.dynamic_topk):
            if cutlass.Int32(slot) < valid_nonlocal:
                score = self._score_from_ordered_key(rScoreKeys[slot])
                selected_max = cute.arch.fmax(selected_max, score)
        local_score = -cutlass.Float32.inf
        if local_seen:
            local_score = self._load_stat(
                local_storage_idx,
                tidx,
                score_tiled_copy,
                mScore,
            )
            selected_max = cute.arch.fmax(selected_max, local_score)

        selected_sum = cutlass.Float32(0.0)
        for slot in cutlass.range_constexpr(self.dynamic_topk):
            if cutlass.Int32(slot) < valid_nonlocal:
                block_sum = self._load_stat(
                    rStorageIndices[slot],
                    tidx,
                    block_sum_tiled_copy,
                    mBlockSum,
                )
                score = self._score_from_ordered_key(rScoreKeys[slot])
                selected_sum += block_sum * cute_math.exp2(
                    (score - selected_max) * cutlass.Float32(_LOG2_E),
                    fastmath=True,
                )
        if local_seen:
            local_block_sum = self._load_stat(
                local_storage_idx,
                tidx,
                block_sum_tiled_copy,
                mBlockSum,
            )
            selected_sum += local_block_sum * cute_math.exp2(
                (local_score - selected_max) * cutlass.Float32(_LOG2_E),
                fastmath=True,
            )

        selected_lse = -cutlass.Float32.inf
        if selected_sum > cutlass.Float32(0.0):
            selected_lse = selected_max + cute_math.log2(
                selected_sum,
                fastmath=True,
            ) * cutlass.Float32(_LN_2)
        return selected_lse

    @cute.jit
    def _store_selected_lse(
        self,
        selected_lse: cutlass.Float32,
        tidx: cutlass.Int32,
        q_tile: cutlass.Int32,
        lse_tiled_copy: cute.TiledCopy,
        mSelectedLse: cute.Tensor,
    ) -> None:
        """Store one head-major LSE value through the CTA tiled copy."""

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

    @cute.jit
    def _store_selection(
        self,
        rSelected: cute.Tensor,
        nonlocal_count: cutlass.Int32,
        local_seen: cutlass.Boolean,
        local_selected: cutlass.Int32,
        q_global: cutlass.Int32,
        q_head: cutlass.Int32,
        output_copy_atom: cute.CopyAtom,
        mTopkIdx: cute.Tensor,
        mBlockBases: Optional[cute.Tensor],
        values_are_block_ids: cutlass.Constexpr[bool],
    ) -> None:
        """Store selected values followed by the local value."""

        valid_nonlocal = nonlocal_count
        if valid_nonlocal > cutlass.Int32(self.dynamic_topk):
            valid_nonlocal = cutlass.Int32(self.dynamic_topk)
        block_base = cutlass.Int32(0)
        if cutlass.const_expr(self.rebase_block_ids and values_are_block_ids):
            block_base = mBlockBases[q_global]
        for vector_idx in cutlass.range_constexpr(self.topk // 4):
            slot_base = vector_idx * 4
            rOutput = cute.make_rmem_tensor((4,), cutlass.Int32)
            for value_idx in cutlass.range_constexpr(4):
                slot = slot_base + value_idx
                output_id = cutlass.Int32(-1)
                if cutlass.Int32(slot) < valid_nonlocal:
                    output_id = rSelected[slot]
                elif local_seen and cutlass.Int32(slot) == valid_nonlocal:
                    output_id = local_selected
                if cutlass.const_expr(self.rebase_block_ids and values_are_block_ids):
                    if output_id >= cutlass.Int32(0):
                        output_id -= block_base
                rOutput[value_idx] = output_id

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
            output_tiled_copy = cute.make_cotiled_copy(
                output_copy_atom,
                cute.make_layout((1, 4)),
                rOutput.layout,
            )
            output_thr_copy = output_tiled_copy.get_slice(0)
            tRrOutput = output_thr_copy.partition_S(rOutput)
            tRgOutput = output_thr_copy.partition_D(gOutput)
            cute.copy(output_copy_atom, tRrOutput, tRgOutput)

    @cute.kernel
    def kernel(
        self,
        score_tiled_copy: cute.TiledCopy,
        block_sum_tiled_copy: cute.TiledCopy,
        output_copy_atom: cute.CopyAtom,
        lse_tiled_copy: cute.TiledCopy,
        mScore: cute.Tensor,
        mBlockSum: cute.Tensor,
        mPartialOffsets: cute.Tensor,
        mPartialBlockIndices: cute.Tensor,
        mFullOffsets: cute.Tensor,
        mFullBlockIndices: cute.Tensor,
        mTopkIdx: cute.Tensor,
        mSelectedLse: cute.Tensor,
        mLocalBlockPositions: cute.Tensor,
        mBlockBases: Optional[cute.Tensor],
        q_len: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        lane_idx = cute.arch.lane_idx()
        pair_lane = cutlass.Int32(0)
        stat_tidx = tidx
        if cutlass.const_expr(self.gather_lse and not self.deterministic):
            q_tile = block_idx // cutlass.Int32(2)
            task_half = block_idx - q_tile * cutlass.Int32(2)
            output_idx = tidx // cutlass.Int32(2)
            stat_tidx = (
                task_half * cutlass.Int32(self.physical_m_tile // 2)
                + output_idx
            )
            pair_lane = tidx - output_idx * cutlass.Int32(2)
            q_in_tile = (
                task_half * cutlass.Int32(self.logical_q_tile // 2)
                + output_idx // cutlass.Int32(self.num_index_heads)
            )
            q_head = output_idx - (
                output_idx // cutlass.Int32(self.num_index_heads)
            ) * cutlass.Int32(self.num_index_heads)
        else:
            q_tile = block_idx
            q_in_tile = tidx // cutlass.Int32(self.num_index_heads)
            q_head = tidx - q_in_tile * cutlass.Int32(self.num_index_heads)
        q_global = q_tile * cutlass.Int32(self.logical_q_tile) + q_in_tile
        q_valid = q_global < q_len

        partial_offset = cutlass.Int32(0)
        full_offset = cutlass.Int32(0)
        if lane_idx < cutlass.Int32(2):
            partial_offset = mPartialOffsets[(cutlass.Int32(0), q_tile + lane_idx)]
            full_offset = mFullOffsets[(cutlass.Int32(0), q_tile + lane_idx)]
        partial_begin = cute.arch.shuffle_sync(partial_offset, 0)
        partial_end = cute.arch.shuffle_sync(partial_offset, 1)
        full_begin = cute.arch.shuffle_sync(full_offset, 0)
        full_end = cute.arch.shuffle_sync(full_offset, 1)
        partial_count = partial_end - partial_begin
        full_count = full_end - full_begin
        num_tiles = partial_count + full_count
        num_partial_tiles = cutlass.Int32(cute.size(mPartialBlockIndices, mode=[1]))

        local_position = cutlass.Int32(0)
        if q_valid:
            local_position = mLocalBlockPositions[(cutlass.Int32(0), q_global)]
        local_block = local_position // cutlass.Int32(self.k_tile)

        if cutlass.const_expr(
            self.small_plan or (not self.gather_lse and self.deterministic)
        ):
            rScoreKeys = cute.make_rmem_tensor(
                (self.dynamic_topk,),
                cutlass.Uint32,
            )
            rBlockIds = cute.make_rmem_tensor(
                (self.dynamic_topk,),
                cutlass.Int32,
            )
            rScoreKeys.fill(cutlass.Uint32(0))
            rBlockIds.fill(cutlass.Int32(-1))
            nonlocal_count = cutlass.Int32(0)
            local_seen = cutlass.Boolean(False)
            if cutlass.const_expr(self.use_fp16_score):
                rStorageIndices = cute.make_rmem_tensor(
                    (self.dynamic_topk,),
                    cutlass.Int32,
                )
                rStorageIndices.fill(cutlass.Int32(-1))
                local_storage_idx = cutlass.Int32(-1)
            else:
                rBlockSums = cute.make_rmem_tensor(
                    (self.dynamic_topk,),
                    cutlass.Float32,
                )
                rBlockSums.fill(cutlass.Float32(0.0))
                local_score = cutlass.Float32(self.neg_inf)
                local_block_sum = cutlass.Float32(0.0)
            for tile_idx in cutlass.range(num_tiles, unroll=1):
                block_id = cutlass.Int32(0)
                if lane_idx == cutlass.Int32(0):
                    block_id = self._block_id(
                        tile_idx,
                        partial_begin,
                        partial_count,
                        full_begin,
                        mPartialBlockIndices,
                        mFullBlockIndices,
                    )
                block_id = cute.arch.shuffle_sync(block_id, 0)
                tile_storage_idx = partial_begin + tile_idx
                if tile_idx >= partial_count:
                    tile_storage_idx = (
                        num_partial_tiles + full_begin + tile_idx - partial_count
                    )
                score, score_key = self._load_score(
                    tile_storage_idx,
                    tidx,
                    score_tiled_copy,
                    mScore,
                    block_id,
                )
                if q_valid and self._is_valid_score(score):
                    if cutlass.const_expr(self.use_fp16_score):
                        (
                            nonlocal_count,
                            local_seen,
                            local_storage_idx,
                        ) = self._consume_candidate_ordered_index(
                            rScoreKeys,
                            rBlockIds,
                            rStorageIndices,
                            score_key,
                            block_id,
                            tile_storage_idx,
                            local_block,
                            nonlocal_count,
                            local_seen,
                            local_storage_idx,
                        )
                    else:
                        block_sum = self._load_stat(
                            tile_storage_idx,
                            tidx,
                            block_sum_tiled_copy,
                            mBlockSum,
                        )
                        (
                            nonlocal_count,
                            local_seen,
                            local_score,
                            local_block_sum,
                        ) = self._consume_candidate_ordered(
                            rScoreKeys,
                            rBlockIds,
                            rBlockSums,
                            score,
                            score_key,
                            block_sum,
                            block_id,
                            local_block,
                            nonlocal_count,
                            local_seen,
                            local_score,
                            local_block_sum,
                        )
            if q_valid:
                if cutlass.const_expr(self.use_fp16_score):
                    selected_lse = self._selected_lse_from_storage_scalar(
                        rScoreKeys,
                        rStorageIndices,
                        nonlocal_count,
                        local_storage_idx,
                        tidx,
                        score_tiled_copy,
                        block_sum_tiled_copy,
                        mScore,
                        mBlockSum,
                    )
                else:
                    selected_lse = self._selected_lse(
                        rScoreKeys,
                        rBlockSums,
                        nonlocal_count,
                        local_seen,
                        local_score,
                        local_block_sum,
                    )
                self._store_selection(
                    rBlockIds,
                    nonlocal_count,
                    local_seen,
                    local_block,
                    q_global,
                    q_head,
                    output_copy_atom,
                    mTopkIdx,
                    mBlockBases,
                    True,
                )
                self._store_selected_lse(
                    selected_lse,
                    tidx,
                    q_tile,
                    lse_tiled_copy,
                    mSelectedLse,
                )
        elif cutlass.const_expr(not self.gather_lse):
            rScoreKeys = cute.make_rmem_tensor(
                (self.dynamic_topk,),
                cutlass.Uint32,
            )
            rBlockIds = cute.make_rmem_tensor(
                (self.dynamic_topk,),
                cutlass.Int32,
            )
            rScoreKeys.fill(cutlass.Uint32(0))
            rBlockIds.fill(cutlass.Int32(-1))
            worst_key = cutlass.Uint32(0)
            worst_slot = cutlass.Int32(0)
            nonlocal_count = cutlass.Int32(0)
            local_seen = cutlass.Boolean(False)
            if cutlass.const_expr(self.use_fp16_score):
                rStorageIndices = cute.make_rmem_tensor(
                    (self.dynamic_topk,),
                    cutlass.Int32,
                )
                rStorageIndices.fill(cutlass.Int32(-1))
                local_storage_idx = cutlass.Int32(-1)
            else:
                rBlockSums = cute.make_rmem_tensor(
                    (self.dynamic_topk,),
                    cutlass.Float32,
                )
                rBlockSums.fill(cutlass.Float32(0.0))
                local_score = cutlass.Float32(self.neg_inf)
                local_block_sum = cutlass.Float32(0.0)
            for tile_idx in cutlass.range(num_tiles, unroll=1):
                block_id = cutlass.Int32(0)
                if lane_idx == cutlass.Int32(0):
                    block_id = self._block_id(
                        tile_idx,
                        partial_begin,
                        partial_count,
                        full_begin,
                        mPartialBlockIndices,
                        mFullBlockIndices,
                    )
                block_id = cute.arch.shuffle_sync(block_id, 0)
                tile_storage_idx = partial_begin + tile_idx
                if tile_idx >= partial_count:
                    tile_storage_idx = (
                        num_partial_tiles + full_begin + tile_idx - partial_count
                    )
                score, score_key = self._load_score(
                    tile_storage_idx,
                    tidx,
                    score_tiled_copy,
                    mScore,
                    block_id,
                )
                if q_valid and self._is_valid_score(score):
                    if cutlass.const_expr(self.use_fp16_score):
                        (
                            worst_key,
                            worst_slot,
                            nonlocal_count,
                            local_storage_idx,
                        ) = self._consume_candidate_unordered_index_with_id(
                            rScoreKeys,
                            rBlockIds,
                            rStorageIndices,
                            score_key,
                            block_id,
                            tile_storage_idx,
                            local_block,
                            worst_key,
                            worst_slot,
                            nonlocal_count,
                            local_storage_idx,
                        )
                    else:
                        block_sum = self._load_stat(
                            tile_storage_idx,
                            tidx,
                            block_sum_tiled_copy,
                            mBlockSum,
                        )
                        (
                            worst_key,
                            worst_slot,
                            nonlocal_count,
                            local_seen,
                            local_score,
                            local_block_sum,
                        ) = self._consume_candidate_unordered(
                            rScoreKeys,
                            rBlockIds,
                            rBlockSums,
                            score,
                            score_key,
                            block_sum,
                            block_id,
                            local_block,
                            worst_key,
                            worst_slot,
                            nonlocal_count,
                            local_seen,
                            local_score,
                            local_block_sum,
                        )
            if cutlass.const_expr(self.use_fp16_score):
                local_seen = local_storage_idx >= cutlass.Int32(0)
            if q_valid:
                if cutlass.const_expr(self.use_fp16_score):
                    selected_lse = self._selected_lse_from_storage_scalar(
                        rScoreKeys,
                        rStorageIndices,
                        nonlocal_count,
                        local_storage_idx,
                        tidx,
                        score_tiled_copy,
                        block_sum_tiled_copy,
                        mScore,
                        mBlockSum,
                    )
                else:
                    selected_lse = self._selected_lse(
                        rScoreKeys,
                        rBlockSums,
                        nonlocal_count,
                        local_seen,
                        local_score,
                        local_block_sum,
                    )
                self._store_selection(
                    rBlockIds,
                    nonlocal_count,
                    local_seen,
                    local_block,
                    q_global,
                    q_head,
                    output_copy_atom,
                    mTopkIdx,
                    mBlockBases,
                    True,
                )
                self._store_selected_lse(
                    selected_lse,
                    tidx,
                    q_tile,
                    lse_tiled_copy,
                    mSelectedLse,
                )
        elif cutlass.const_expr(self.deterministic):
            rScoreKeys = cute.make_rmem_tensor(
                (self.dynamic_topk,),
                cutlass.Uint32,
            )
            rBlockIds = cute.make_rmem_tensor(
                (self.dynamic_topk,),
                cutlass.Int32,
            )
            rStorageIndices = cute.make_rmem_tensor(
                (self.dynamic_topk,),
                cutlass.Int32,
            )
            rScoreKeys.fill(cutlass.Uint32(0))
            rBlockIds.fill(cutlass.Int32(-1))
            rStorageIndices.fill(cutlass.Int32(-1))
            nonlocal_count = cutlass.Int32(0)
            local_seen = cutlass.Boolean(False)
            local_storage_idx = cutlass.Int32(-1)
            for tile_idx in cutlass.range(num_tiles, unroll=1):
                block_id = cutlass.Int32(0)
                if lane_idx == cutlass.Int32(0):
                    block_id = self._block_id(
                        tile_idx,
                        partial_begin,
                        partial_count,
                        full_begin,
                        mPartialBlockIndices,
                        mFullBlockIndices,
                    )
                block_id = cute.arch.shuffle_sync(block_id, 0)
                tile_storage_idx = partial_begin + tile_idx
                if tile_idx >= partial_count:
                    tile_storage_idx = (
                        num_partial_tiles + full_begin + tile_idx - partial_count
                    )
                score, score_key = self._load_score(
                    tile_storage_idx,
                    tidx,
                    score_tiled_copy,
                    mScore,
                    block_id,
                )
                if q_valid and self._is_valid_score(score):
                    (
                        nonlocal_count,
                        local_seen,
                        local_storage_idx,
                    ) = self._consume_candidate_ordered_index(
                        rScoreKeys,
                        rBlockIds,
                        rStorageIndices,
                        score_key,
                        block_id,
                        tile_storage_idx,
                        local_block,
                        nonlocal_count,
                        local_seen,
                        local_storage_idx,
                    )
            if q_valid:
                if cutlass.const_expr(self.use_fp16_score):
                    selected_lse = self._selected_lse_from_storage_scalar(
                        rScoreKeys,
                        rStorageIndices,
                        nonlocal_count,
                        local_storage_idx,
                        tidx,
                        score_tiled_copy,
                        block_sum_tiled_copy,
                        mScore,
                        mBlockSum,
                    )
                    self._store_selection(
                        rBlockIds,
                        nonlocal_count,
                        local_seen,
                        local_block,
                        q_global,
                        q_head,
                        output_copy_atom,
                        mTopkIdx,
                        mBlockBases,
                        True,
                    )
                    self._store_selected_lse(
                        selected_lse,
                        tidx,
                        q_tile,
                        lse_tiled_copy,
                        mSelectedLse,
                    )
                else:
                    self._store_selection(
                        rStorageIndices,
                        nonlocal_count,
                        local_seen,
                        local_storage_idx,
                        q_global,
                        q_head,
                        output_copy_atom,
                        mTopkIdx,
                        mBlockBases,
                        False,
                    )
        else:
            rScoreKeys = cute.make_rmem_tensor(
                (self.dynamic_topk,),
                cutlass.Uint32,
            )
            rStorageIndices = cute.make_rmem_tensor(
                (self.dynamic_topk,),
                cutlass.Int32,
            )
            rScoreKeys.fill(cutlass.Uint32(0))
            rStorageIndices.fill(cutlass.Int32(-1))
            worst_key = cutlass.Uint32(0)
            worst_slot = cutlass.Int32(0)
            nonlocal_count = cutlass.Int32(0)
            local_seen = cutlass.Boolean(False)
            local_storage_idx = cutlass.Int32(-1)
            worker_tiles = (num_tiles + cutlass.Int32(1)) // cutlass.Int32(2)
            for worker_tile_idx in cutlass.range(worker_tiles, unroll=1):
                tile_idx = pair_lane + worker_tile_idx * cutlass.Int32(2)
                tile_valid = tile_idx < num_tiles
                block_id = cutlass.Int32(-1)
                if lane_idx < cutlass.Int32(2) and tile_valid:
                    block_id = self._block_id(
                        tile_idx,
                        partial_begin,
                        partial_count,
                        full_begin,
                        mPartialBlockIndices,
                        mFullBlockIndices,
                    )
                block_id = cute.arch.shuffle_sync(block_id, pair_lane)
                if tile_valid:
                    tile_storage_idx = partial_begin + tile_idx
                    if tile_idx >= partial_count:
                        tile_storage_idx = (
                            num_partial_tiles + full_begin + tile_idx - partial_count
                        )
                    score, score_key = self._load_score(
                        tile_storage_idx,
                        stat_tidx,
                        score_tiled_copy,
                        mScore,
                        block_id,
                    )
                    if q_valid and self._is_valid_score(score):
                        (
                            worst_key,
                            worst_slot,
                            nonlocal_count,
                            local_seen,
                            local_storage_idx,
                        ) = self._consume_candidate_unordered_index(
                            rScoreKeys,
                            rStorageIndices,
                            score_key,
                            block_id,
                            tile_storage_idx,
                            local_block,
                            worst_key,
                            worst_slot,
                            nonlocal_count,
                            local_seen,
                            local_storage_idx,
                        )

            partner_lane = lane_idx ^ cutlass.Int32(1)
            partner_count = cute.arch.shuffle_sync(nonlocal_count, partner_lane)
            partner_local_storage_idx = cute.arch.shuffle_sync(
                local_storage_idx,
                partner_lane,
            )
            for slot in cutlass.range_constexpr(self.dynamic_topk):
                partner_key = cute.arch.shuffle_sync(
                    rScoreKeys[slot],
                    partner_lane,
                )
                partner_storage_idx = cute.arch.shuffle_sync(
                    rStorageIndices[slot],
                    partner_lane,
                )
                if pair_lane == cutlass.Int32(0):
                    worst_key, worst_slot = self._merge_unordered_index(
                        rScoreKeys,
                        rStorageIndices,
                        partner_key,
                        partner_storage_idx,
                        worst_key,
                        worst_slot,
                    )

            if pair_lane == cutlass.Int32(0):
                nonlocal_count += partner_count
                if not local_seen and partner_local_storage_idx >= cutlass.Int32(0):
                    local_seen = cutlass.Boolean(True)
                    local_storage_idx = partner_local_storage_idx

            if q_valid and pair_lane == cutlass.Int32(0):
                if cutlass.const_expr(self.use_fp16_score):
                    selected_lse = self._selected_lse_from_storage_scalar(
                        rScoreKeys,
                        rStorageIndices,
                        nonlocal_count,
                        local_storage_idx,
                        stat_tidx,
                        score_tiled_copy,
                        block_sum_tiled_copy,
                        mScore,
                        mBlockSum,
                    )
                    for slot in cutlass.range_constexpr(self.dynamic_topk):
                        if cutlass.Int32(slot) < nonlocal_count:
                            rStorageIndices[slot] = cutlass.Int32(
                                (~rScoreKeys[slot]) & cutlass.Uint32(0xFFFF)
                            )
                    self._store_selection(
                        rStorageIndices,
                        nonlocal_count,
                        local_seen,
                        local_block,
                        q_global,
                        q_head,
                        output_copy_atom,
                        mTopkIdx,
                        mBlockBases,
                        True,
                    )
                    self._store_selected_lse(
                        selected_lse,
                        stat_tidx,
                        q_tile,
                        lse_tiled_copy,
                        mSelectedLse,
                    )
                else:
                    self._store_selection(
                        rStorageIndices,
                        nonlocal_count,
                        local_seen,
                        local_storage_idx,
                        q_global,
                        q_head,
                        output_copy_atom,
                        mTopkIdx,
                        mBlockBases,
                        False,
                    )


__all__ = ["M3IndexerTopkSm100"]
