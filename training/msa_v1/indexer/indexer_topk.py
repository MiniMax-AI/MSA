"""SM100/SM103 tile-major TopK and selected LSE for MiniMax-M3."""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math
from cutlass._mlir.dialects import arith, llvm
import cutlass.utils as utils
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
        self.score_dtype = cutlass.Float16 if use_fp16_score else cutlass.Float32
        self.selection_lanes = 4
        self.selection_ctas = 8
        self.heap_rows = (
            self.physical_m_tile // 2
            if self.gather_lse and not self.deterministic
            else 1
        )

    @cute.jit
    def __call__(
        self,
        mScore: cute.Tensor,
        mBlockSum: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mTaskBatchIdx: cute.Tensor,
        mTaskQLocalBegin: cute.Tensor,
        mTopkIdx: cute.Tensor,
        mSelectedLse: cute.Tensor,
        mFragmentIndices: Optional[cute.Tensor],
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

        @cute.struct
        class SharedStorage:
            heap_keys: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint32,
                    self.heap_rows * self.dynamic_topk,
                ],
                16,
            ]
            heap_indices: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32,
                    self.heap_rows * self.dynamic_topk,
                ],
                16,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            score_tiled_copy,
            block_sum_tiled_copy,
            output_copy_atom,
            lse_tiled_copy,
            mScore,
            mBlockSum,
            mCuSeqlensQ,
            mCuSeqlensK,
            mTaskBatchIdx,
            mTaskQLocalBegin,
            mTopkIdx,
            mSelectedLse,
            mFragmentIndices,
            max_k_blocks,
        ).launch(
            grid=(
                num_task_slots * cutlass.Int32(self.selection_ctas)
                if cutlass.const_expr(self.gather_lse and not self.deterministic)
                else num_task_slots,
                1,
                1,
            ),
            block=(
                self.physical_m_tile // 2
                if cutlass.const_expr(self.gather_lse and not self.deterministic)
                else self.physical_m_tile,
                1,
                1,
            ),
            stream=stream,
            min_blocks_per_mp=3 if self.small_plan or self.deterministic else 4,
            smem=self.shared_storage.size_in_bytes(),
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
    def _shared_heap_insert(
        self,
        sHeapKeys: cute.Tensor,
        sHeapIndices: cute.Tensor,
        tidx: cutlass.Int32,
        score_key: cutlass.Uint32,
        storage_idx: cutlass.Int32,
    ) -> None:
        """Insert one candidate into a shared-memory exact Top15 min-heap."""

        if score_key > sHeapKeys[(tidx, cutlass.Int32(0))]:
            sHeapKeys[(tidx, cutlass.Int32(0))] = score_key
            sHeapIndices[(tidx, cutlass.Int32(0))] = storage_idx
            node = cutlass.Int32(0)
            for _ in cutlass.range_constexpr(3):
                left = node * cutlass.Int32(2) + cutlass.Int32(1)
                if left < cutlass.Int32(self.dynamic_topk):
                    child = left
                    child_key = sHeapKeys[(tidx, left)]
                    right = left + cutlass.Int32(1)
                    if right < cutlass.Int32(self.dynamic_topk):
                        right_key = sHeapKeys[(tidx, right)]
                        if right_key < child_key:
                            child = right
                            child_key = right_key
                    node_key = sHeapKeys[(tidx, node)]
                    if child_key < node_key:
                        node_storage_idx = sHeapIndices[(tidx, node)]
                        child_storage_idx = sHeapIndices[(tidx, child)]
                        sHeapKeys[(tidx, node)] = child_key
                        sHeapIndices[(tidx, node)] = child_storage_idx
                        sHeapKeys[(tidx, child)] = node_key
                        sHeapIndices[(tidx, child)] = node_storage_idx
                        node = child

    @cute.jit
    def _shared_heapify(
        self,
        sHeapKeys: cute.Tensor,
        sHeapIndices: cute.Tensor,
        tidx: cutlass.Int32,
    ) -> None:
        """Build a Top15 min-heap from directly initialized shared slots."""

        for heap_node in cutlass.range_constexpr(7):
            node = cutlass.Int32(6 - heap_node)
            for _ in cutlass.range_constexpr(3):
                left = node * cutlass.Int32(2) + cutlass.Int32(1)
                if left < cutlass.Int32(self.dynamic_topk):
                    child = left
                    child_key = sHeapKeys[(tidx, left)]
                    right = left + cutlass.Int32(1)
                    if right < cutlass.Int32(self.dynamic_topk):
                        right_key = sHeapKeys[(tidx, right)]
                        if right_key < child_key:
                            child = right
                            child_key = right_key
                    node_key = sHeapKeys[(tidx, node)]
                    if child_key < node_key:
                        node_storage_idx = sHeapIndices[(tidx, node)]
                        child_storage_idx = sHeapIndices[(tidx, child)]
                        sHeapKeys[(tidx, node)] = child_key
                        sHeapIndices[(tidx, node)] = child_storage_idx
                        sHeapKeys[(tidx, child)] = node_key
                        sHeapIndices[(tidx, child)] = node_storage_idx
                        node = child

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
    def _consume_candidate_unordered_index(
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
        """Maintain Top15 while retaining only selected storage indices."""

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
    def _selected_lse_from_storage(
        self,
        rScoreKeys: cute.Tensor,
        rStorageIndices: cute.Tensor,
        nonlocal_count: cutlass.Int32,
        local_storage_idx: cutlass.Int32,
        selection_lane: cutlass.Int32,
        selection_group_base: cutlass.Int32,
        stat_tidx: cutlass.Int32,
        score_tiled_copy: cute.TiledCopy,
        block_sum_tiled_copy: cute.TiledCopy,
        mScore: cute.Tensor,
        mBlockSum: cute.Tensor,
    ) -> cutlass.Float32:
        """Gather selected statistics and merge FP32 LSE across four lanes."""

        valid_nonlocal = cute.arch.shuffle_sync(nonlocal_count, selection_group_base)
        if valid_nonlocal > cutlass.Int32(self.dynamic_topk):
            valid_nonlocal = cutlass.Int32(self.dynamic_topk)
        local_storage_idx = cute.arch.shuffle_sync(
            local_storage_idx,
            selection_group_base,
        )
        local_seen = local_storage_idx >= cutlass.Int32(0)

        lane_max = -cutlass.Float32.inf
        if selection_lane == cutlass.Int32(0) and local_seen:
            lane_max = self._load_stat(
                local_storage_idx,
                stat_tidx,
                score_tiled_copy,
                mScore,
            )
        for slot in cutlass.range_constexpr(self.dynamic_topk):
            score_key = cute.arch.shuffle_sync(
                rScoreKeys[slot],
                selection_group_base,
            )
            if (
                cutlass.Int32(slot) < valid_nonlocal
                and cutlass.Int32(slot % self.selection_lanes) == selection_lane
            ):
                score = self._score_from_ordered_key(score_key)
                lane_max = cute.arch.fmax(lane_max, score)

        selected_max = lane_max
        for partner_idx in cutlass.range_constexpr(1, self.selection_lanes):
            partner_max = cute.arch.shuffle_sync(
                lane_max,
                selection_group_base + cutlass.Int32(partner_idx),
            )
            if selection_lane == cutlass.Int32(0):
                selected_max = cute.arch.fmax(
                    selected_max,
                    partner_max,
                )
        selected_max = cute.arch.shuffle_sync(selected_max, selection_group_base)

        lane_sum = cutlass.Float32(0.0)
        for slot in cutlass.range_constexpr(self.dynamic_topk):
            storage_idx = cute.arch.shuffle_sync(
                rStorageIndices[slot],
                selection_group_base,
            )
            score_key = cute.arch.shuffle_sync(
                rScoreKeys[slot],
                selection_group_base,
            )
            if (
                cutlass.Int32(slot) < valid_nonlocal
                and cutlass.Int32(slot % self.selection_lanes) == selection_lane
            ):
                block_sum = self._load_stat(
                    storage_idx,
                    stat_tidx,
                    block_sum_tiled_copy,
                    mBlockSum,
                )
                score = self._score_from_ordered_key(score_key)
                lane_sum += block_sum * cute_math.exp2(
                    (score - selected_max) * cutlass.Float32(_LOG2_E),
                    fastmath=True,
                )
        if selection_lane == cutlass.Int32(0) and local_seen:
            local_score = self._load_stat(
                local_storage_idx,
                stat_tidx,
                score_tiled_copy,
                mScore,
            )
            local_block_sum = self._load_stat(
                local_storage_idx,
                stat_tidx,
                block_sum_tiled_copy,
                mBlockSum,
            )
            lane_sum += local_block_sum * cute_math.exp2(
                (local_score - selected_max) * cutlass.Float32(_LOG2_E),
                fastmath=True,
            )

        selected_lse = -cutlass.Float32.inf
        selected_sum = lane_sum
        for partner_idx in cutlass.range_constexpr(1, self.selection_lanes):
            partner_sum = cute.arch.shuffle_sync(
                lane_sum,
                selection_group_base + cutlass.Int32(partner_idx),
            )
            if selection_lane == cutlass.Int32(0):
                selected_sum += partner_sum
        if selection_lane == cutlass.Int32(0):
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
        q_global_begin: cutlass.Int32,
        lse_tiled_copy: cute.TiledCopy,
        mSelectedLse: cute.Tensor,
    ) -> None:
        """Store one head-major LSE value through the CTA tiled copy."""

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
    ) -> None:
        """Store selected values followed by the local value."""

        valid_nonlocal = nonlocal_count
        if valid_nonlocal > cutlass.Int32(self.dynamic_topk):
            valid_nonlocal = cutlass.Int32(self.dynamic_topk)
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
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mTaskBatchIdx: cute.Tensor,
        mTaskQLocalBegin: cute.Tensor,
        mTopkIdx: cute.Tensor,
        mSelectedLse: cute.Tensor,
        mFragmentIndices: Optional[cute.Tensor],
        max_k_blocks: cutlass.Int32,
    ):
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        sHeapKeys = storage.heap_keys.get_tensor(
            cute.make_layout(
                (self.heap_rows, self.dynamic_topk),
                stride=(1, self.heap_rows),
            )
        )
        sHeapIndices = storage.heap_indices.get_tensor(
            cute.make_layout(
                (self.heap_rows, self.dynamic_topk),
                stride=(1, self.heap_rows),
            )
        )
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        lane_idx = cute.arch.lane_idx()
        selection_lane = cutlass.Int32(0)
        stat_tidx = tidx
        if cutlass.const_expr(self.gather_lse and not self.deterministic):
            task_idx = block_idx // cutlass.Int32(self.selection_ctas)
            task_part = block_idx - task_idx * cutlass.Int32(
                self.selection_ctas
            )
            output_idx = tidx // cutlass.Int32(self.selection_lanes)
            stat_tidx = (
                task_part
                * cutlass.Int32(self.physical_m_tile // self.selection_ctas)
                + output_idx
            )
            selection_lane = tidx - output_idx * cutlass.Int32(
                self.selection_lanes
            )
            q_in_tile = (
                task_part
                * cutlass.Int32(self.logical_q_tile // self.selection_ctas)
                + output_idx // cutlass.Int32(self.num_index_heads)
            )
            q_head = output_idx - (
                output_idx // cutlass.Int32(self.num_index_heads)
            ) * cutlass.Int32(self.num_index_heads)
        else:
            task_idx = block_idx
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
        seq_k = cutlass.Int32(0)
        num_tiles = cutlass.Int32(0)
        if lane_idx == cutlass.Int32(0) and batch_idx >= cutlass.Int32(0):
            q_start = mCuSeqlensQ[batch_idx]
            q_end = mCuSeqlensQ[batch_idx + cutlass.Int32(1)]
            k_start_idx = (
                batch_idx
                if cutlass.const_expr(mFragmentIndices is None)
                else mFragmentIndices[batch_idx]
            )
            k_start = mCuSeqlensK[k_start_idx]
            k_end = mCuSeqlensK[batch_idx + cutlass.Int32(1)]
            seq_q = q_end - q_start
            seq_k = k_end - k_start
            q_local_end = q_local_begin + cutlass.Int32(self.logical_q_tile)
            if q_local_end > seq_q:
                q_local_end = seq_q
            last_visible = seq_k - seq_q + q_local_end
            if last_visible < cutlass.Int32(0):
                last_visible = cutlass.Int32(0)
            if last_visible > seq_k:
                last_visible = seq_k
            num_tiles = cute.ceil_div(last_visible, self.k_tile)
        q_start = cute.arch.shuffle_sync(q_start, 0)
        seq_q = cute.arch.shuffle_sync(seq_q, 0)
        seq_k = cute.arch.shuffle_sync(seq_k, 0)
        num_tiles = cute.arch.shuffle_sync(num_tiles, 0)

        q_local = q_local_begin + q_in_tile
        q_global = q_start + q_local
        q_valid = q_local < seq_q
        local_position = seq_k - seq_q + q_local
        local_block = local_position // cutlass.Int32(self.k_tile)
        tile_storage_base = task_idx * max_k_blocks

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
                block_id = tile_idx
                tile_storage_idx = tile_storage_base + tile_idx
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
                )
                self._store_selected_lse(
                    selected_lse,
                    tidx,
                    q_start + q_local_begin,
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
                block_id = tile_idx
                tile_storage_idx = tile_storage_base + tile_idx
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
                        ) = self._consume_candidate_unordered_index(
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
                )
                self._store_selected_lse(
                    selected_lse,
                    tidx,
                    q_start + q_local_begin,
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
                block_id = tile_idx
                tile_storage_idx = tile_storage_base + tile_idx
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
                self._store_selection(
                    rStorageIndices,
                    nonlocal_count,
                    local_seen,
                    local_storage_idx,
                    q_global,
                    q_head,
                    output_copy_atom,
                    mTopkIdx,
                )
        else:
            for slot in cutlass.range_constexpr(self.dynamic_topk):
                sHeapKeys[(tidx, cutlass.Int32(slot))] = cutlass.Uint32(0)
                sHeapIndices[(tidx, cutlass.Int32(slot))] = cutlass.Int32(-1)

            nonlocal_count = cutlass.Int32(0)
            local_seen = cutlass.Boolean(False)
            local_storage_idx = cutlass.Int32(-1)
            worker_tiles = (
                num_tiles
                + cutlass.Int32(self.selection_lanes - 1)
                - selection_lane
            ) // cutlass.Int32(self.selection_lanes)
            # Direct-fill the prefix, then build the exact lane-local min-heap.
            prefix_tiles = worker_tiles
            if prefix_tiles > cutlass.Int32(self.dynamic_topk):
                prefix_tiles = cutlass.Int32(self.dynamic_topk)
            for worker_tile_idx in cutlass.range(prefix_tiles, unroll=1):
                tile_idx = selection_lane + worker_tile_idx * cutlass.Int32(
                    self.selection_lanes
                )
                tile_storage_idx = tile_storage_base + tile_idx
                score, score_key = self._load_score(
                    tile_storage_idx,
                    stat_tidx,
                    score_tiled_copy,
                    mScore,
                    tile_idx,
                )
                if q_valid and self._is_valid_score(score):
                    if tile_idx == local_block:
                        if not local_seen:
                            local_seen = cutlass.Boolean(True)
                            local_storage_idx = tile_storage_idx
                    else:
                        sHeapKeys[(tidx, worker_tile_idx)] = score_key
                        sHeapIndices[(tidx, worker_tile_idx)] = tile_storage_idx
                        nonlocal_count += cutlass.Int32(1)

            self._shared_heapify(sHeapKeys, sHeapIndices, tidx)
            remaining_tiles = worker_tiles - prefix_tiles
            for remaining_tile_idx in cutlass.range(remaining_tiles, unroll=1):
                worker_tile_idx = prefix_tiles + remaining_tile_idx
                tile_idx = selection_lane + worker_tile_idx * cutlass.Int32(
                    self.selection_lanes
                )
                tile_storage_idx = tile_storage_base + tile_idx
                score, score_key = self._load_score(
                    tile_storage_idx,
                    stat_tidx,
                    score_tiled_copy,
                    mScore,
                    tile_idx,
                )
                if q_valid and self._is_valid_score(score):
                    if tile_idx == local_block:
                        if not local_seen:
                            local_seen = cutlass.Boolean(True)
                            local_storage_idx = tile_storage_idx
                    else:
                        self._shared_heap_insert(
                            sHeapKeys,
                            sHeapIndices,
                            tidx,
                            score_key,
                            tile_storage_idx,
                        )
                        nonlocal_count += cutlass.Int32(1)

            selection_group_base = lane_idx - selection_lane
            pair_base = (selection_lane // cutlass.Int32(2)) * cutlass.Int32(2)
            selection_group_tidx = tidx - selection_lane
            pair_leader_tidx = selection_group_tidx + pair_base
            pair_partner_tidx = pair_leader_tidx + cutlass.Int32(1)
            # Top15(A U B U C U D) equals
            # Top15(Top15(A U B) U Top15(C U D)).
            cute.arch.sync_warp()
            if selection_lane == pair_base:
                for slot in cutlass.range_constexpr(self.dynamic_topk):
                    partner_storage_idx = sHeapIndices[
                        (pair_partner_tidx, cutlass.Int32(slot))
                    ]
                    if partner_storage_idx >= cutlass.Int32(0):
                        self._shared_heap_insert(
                            sHeapKeys,
                            sHeapIndices,
                            pair_leader_tidx,
                            sHeapKeys[
                                (pair_partner_tidx, cutlass.Int32(slot))
                            ],
                            partner_storage_idx,
                        )
            cute.arch.sync_warp()
            if selection_lane == cutlass.Int32(0):
                for slot in cutlass.range_constexpr(self.dynamic_topk):
                    pair_23_storage_idx = sHeapIndices[
                        (selection_group_tidx + cutlass.Int32(2), cutlass.Int32(slot))
                    ]
                    if pair_23_storage_idx >= cutlass.Int32(0):
                        self._shared_heap_insert(
                            sHeapKeys,
                            sHeapIndices,
                            selection_group_tidx,
                            sHeapKeys[
                                (
                                    selection_group_tidx + cutlass.Int32(2),
                                    cutlass.Int32(slot),
                                )
                            ],
                            pair_23_storage_idx,
                        )
            cute.arch.sync_warp()

            pair_partner_lane = selection_group_base + pair_base + cutlass.Int32(1)
            partner_count = cute.arch.shuffle_sync(
                nonlocal_count,
                pair_partner_lane,
            )
            partner_local_storage_idx = cute.arch.shuffle_sync(
                local_storage_idx,
                pair_partner_lane,
            )
            if selection_lane == pair_base:
                nonlocal_count += partner_count
                if (
                    not local_seen
                    and partner_local_storage_idx >= cutlass.Int32(0)
                ):
                    local_seen = cutlass.Boolean(True)
                    local_storage_idx = partner_local_storage_idx

            pair_23_count = cute.arch.shuffle_sync(
                nonlocal_count,
                selection_group_base + cutlass.Int32(2),
            )
            pair_23_local_storage_idx = cute.arch.shuffle_sync(
                local_storage_idx,
                selection_group_base + cutlass.Int32(2),
            )
            if selection_lane == cutlass.Int32(0):
                nonlocal_count += pair_23_count
                if (
                    not local_seen
                    and pair_23_local_storage_idx >= cutlass.Int32(0)
                ):
                    local_seen = cutlass.Boolean(True)
                    local_storage_idx = pair_23_local_storage_idx

            # Only the selection-group leader materializes the final reservoir.
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
            if selection_lane == cutlass.Int32(0):
                if nonlocal_count >= cutlass.Int32(self.dynamic_topk):
                    for slot in cutlass.range_constexpr(self.dynamic_topk):
                        rScoreKeys[slot] = sHeapKeys[
                            (selection_group_tidx, cutlass.Int32(slot))
                        ]
                        rStorageIndices[slot] = sHeapIndices[
                            (selection_group_tidx, cutlass.Int32(slot))
                        ]
                else:
                    for slot in cutlass.range_constexpr(self.dynamic_topk):
                        heap_storage_idx = sHeapIndices[
                            (selection_group_tidx, cutlass.Int32(slot))
                        ]
                        if heap_storage_idx >= cutlass.Int32(0):
                            worst_key, worst_slot = self._merge_unordered_index(
                                rScoreKeys,
                                rStorageIndices,
                                sHeapKeys[(selection_group_tidx, cutlass.Int32(slot))],
                                heap_storage_idx,
                                worst_key,
                                worst_slot,
                            )

            selected_lse = self._selected_lse_from_storage(
                rScoreKeys,
                rStorageIndices,
                nonlocal_count,
                local_storage_idx,
                selection_lane,
                selection_group_base,
                stat_tidx,
                score_tiled_copy,
                block_sum_tiled_copy,
                mScore,
                mBlockSum,
            )
            if q_valid and selection_lane == cutlass.Int32(0):
                for slot in cutlass.range_constexpr(self.dynamic_topk):
                    if cutlass.Int32(slot) < nonlocal_count:
                        rStorageIndices[slot] -= tile_storage_base
                self._store_selection(
                    rStorageIndices,
                    nonlocal_count,
                    local_seen,
                    local_block,
                    q_global,
                    q_head,
                    output_copy_atom,
                    mTopkIdx,
                )
                self._store_selected_lse(
                    selected_lse,
                    stat_tidx,
                    q_start + q_local_begin,
                    lse_tiled_copy,
                    mSelectedLse,
                )


__all__ = ["M3IndexerTopkSm100"]
