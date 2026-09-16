"""SM100/SM103 arbitrary-mask plan compiler for the MiniMax-M3 indexer."""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
from cutlass import Int32, Int64, Uint32
import cutlass.cute as cute
import cutlass.utils as utils
import cuda.bindings.driver as cuda
import torch

from msa_v1._common.aot_cache import compile_or_load
from msa_v1._common.compile_utils import compile_with_timing
from msa_v1._common.utils import shr_u32

_NUM_INDEX_HEADS = 4
_PHYSICAL_M_TILE = 256
_LOGICAL_Q_TILE = _PHYSICAL_M_TILE // _NUM_INDEX_HEADS
_K_TILE = 128
_WARP_SIZE = 32
_CLASSIFY_THREADS = _PHYSICAL_M_TILE
_CLASSIFY_WARPS = _CLASSIFY_THREADS // _WARP_SIZE
_SCAN_THREADS = _PHYSICAL_M_TILE
_MATERIALIZE_THREADS = _LOGICAL_Q_TILE
# Prefix plans derive block IDs arithmetically; one column keeps the internal
# rank-2 tensor contract without allocating the unused dense scratch matrix.
_PREFIX_BLOCK_SCRATCH_COLUMNS = 1
# Largest sequence whose dense tile plan fits signed int32 offsets.
_MAX_SEQLEN = 4_194_240
_MAX_K_BLOCKS = (_MAX_SEQLEN + _K_TILE - 1) // _K_TILE
_MAX_K_BLOCK_WORDS = (_MAX_K_BLOCKS + _WARP_SIZE - 1) // _WARP_SIZE

_ERROR_INVALID_INTERVAL = 1

_CLASSIFY_COMPILE_CACHE: dict[tuple, object] = {}
_SCAN_COMPILE_CACHE: dict[tuple, object] = {}
_MATERIALIZE_COMPILE_CACHE: dict[tuple, object] = {}


@cute.jit
def _load_func_endpoint(
    mFunc: cute.Tensor,
    endpoint_idx: Int32,
    q_idx: Int32,
) -> Int32:
    """Load one scalar Func endpoint."""

    return mFunc[(endpoint_idx, q_idx)]


@dataclass(frozen=True)
class M3ArbitraryMaskPlan:
    """Head-broadcast FULL/PARTIAL plan consumed by the M3 indexer."""

    q_len: int
    k_len: int
    max_plan_tiles: int
    partial_offsets: torch.Tensor
    partial_block_indices: torch.Tensor
    partial_masks: torch.Tensor
    full_offsets: torch.Tensor
    full_block_indices: torch.Tensor
    local_block_positions: torch.Tensor

    @property
    def num_plan_tiles(self) -> int:
        return int(self.partial_block_indices.numel() + self.full_block_indices.numel())

    @property
    def seqlen(self) -> int:
        """Return the Q length for compatibility with equal-length callers."""

        return self.q_len


class M3PlanClassifySm100:
    """Classify Q/K tiles through runtime-decoded interval unions."""

    def __init__(self, is_prefix: bool = False) -> None:
        self.is_prefix = is_prefix
        self.logical_q_tile = _LOGICAL_Q_TILE
        self.k_tile = _K_TILE
        self.warp_size = _WARP_SIZE
        self.threads = _CLASSIFY_THREADS
        self.num_warps = _CLASSIFY_WARPS
        self.max_k_block_words = _MAX_K_BLOCK_WORDS

    @cute.jit
    def __call__(
        self,
        mFunc: cute.Tensor,
        mBlockScratch: cute.Tensor,
        mPartialCounts: cute.Tensor,
        mFullCounts: cute.Tensor,
        mWarpPartialCounts: cute.Tensor,
        mWarpFullCounts: cute.Tensor,
        mError: cute.Tensor,
        q_len: Int32,
        k_len: Int32,
        n_func: Int32,
        num_k_blocks: Int32,
        stream: cuda.CUstream = None,
    ):
        self.kernel(
            mFunc,
            mBlockScratch,
            mPartialCounts,
            mFullCounts,
            mWarpPartialCounts,
            mWarpFullCounts,
            mError,
            q_len,
            k_len,
            n_func,
            num_k_blocks,
        ).launch(
            grid=(cute.ceil_div(q_len, self.logical_q_tile), 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _set_candidate_range(
        self,
        sCandidate: cute.Tensor,
        block_begin: Int32,
        block_end: Int32,
    ) -> None:
        """OR one half-open K-block range into the CTA bitmap."""

        first_word = block_begin // Int32(32)
        last_word = (block_end - Int32(1)) // Int32(32)
        word_idx = first_word
        while word_idx <= last_word:
            lo = Int32(0)
            if word_idx == first_word:
                lo = block_begin - first_word * Int32(32)
            hi = Int32(32)
            if word_idx == last_word:
                hi = block_end - last_word * Int32(32)
            upper = shr_u32(
                Uint32(0xFFFF_FFFF),
                Uint32(Int32(32) - hi),
            )
            lower = shr_u32(
                Uint32(0xFFFF_FFFF),
                Uint32(Int32(32) - lo),
            )
            cute.arch.atomic_or(
                (sCandidate.iterator + word_idx).llvm_ptr,
                upper ^ lower,
                sem="relaxed",
                scope="cta",
            )
            word_idx += Int32(1)

    @cute.jit
    def _mark_error(
        self,
        sError: cute.Tensor,
        error: Uint32,
    ) -> None:
        cute.arch.atomic_or(
            sError.iterator.llvm_ptr,
            error,
            sem="relaxed",
            scope="cta",
        )

    @cute.jit
    def _build_candidate_bitmap(
        self,
        sCandidate: cute.Tensor,
        mFunc: cute.Tensor,
        sError: cute.Tensor,
        q_idx: Int32,
        k_len: Int32,
        n_func: Int32,
    ) -> None:
        previous_begin = Int32(-1)
        num_intervals = (n_func + Int32(1)) // Int32(2)
        for interval_idx in cutlass.range(num_intervals, unroll=1):
            token_begin = Int32(0)
            if interval_idx > Int32(0):
                token_begin = _load_func_endpoint(
                    mFunc,
                    interval_idx * Int32(2) - Int32(1),
                    q_idx,
                )
            token_end = _load_func_endpoint(
                mFunc,
                interval_idx * Int32(2),
                q_idx,
            )
            invalid = (
                (token_begin < Int32(0))
                | (token_end < Int32(0))
                | (token_begin > k_len)
                | (token_end > k_len)
                | (token_end < token_begin)
            )
            if token_end > token_begin and token_begin < previous_begin:
                invalid = cutlass.Boolean(True)
            if invalid:
                self._mark_error(sError, Uint32(_ERROR_INVALID_INTERVAL))

            safe_begin = token_begin
            if safe_begin < Int32(0):
                safe_begin = Int32(0)
            if safe_begin > k_len:
                safe_begin = k_len
            safe_end = token_end
            if safe_end < Int32(0):
                safe_end = Int32(0)
            if safe_end > k_len:
                safe_end = k_len
            if safe_end > safe_begin:
                if safe_begin > previous_begin:
                    previous_begin = safe_begin
                block_begin = safe_begin // Int32(self.k_tile)
                block_end = (
                    safe_end + Int32(self.k_tile - 1)
                ) // Int32(self.k_tile)
                self._set_candidate_range(
                    sCandidate,
                    block_begin,
                    block_end,
                )

    @cute.jit
    def _row_block_state(
        self,
        mFunc: cute.Tensor,
        q_idx: Int32,
        block_id: Int32,
        n_func: Int32,
    ) -> tuple[cutlass.Boolean, cutlass.Boolean]:
        """Evaluate visibility and complete coverage of one row/block pair."""

        block_begin = block_id * Int32(self.k_tile)
        block_end = block_begin + Int32(self.k_tile)
        covered_end = block_begin
        row_visible = cutlass.Boolean(False)
        num_intervals = (n_func + Int32(1)) // Int32(2)
        for interval_idx in cutlass.range(num_intervals, unroll=1):
            token_begin = Int32(0)
            if interval_idx > Int32(0):
                token_begin = _load_func_endpoint(
                    mFunc,
                    interval_idx * Int32(2) - Int32(1),
                    q_idx,
                )
            token_end = _load_func_endpoint(
                mFunc,
                interval_idx * Int32(2),
                q_idx,
            )
            lo = token_begin
            if lo < block_begin:
                lo = block_begin
            hi = token_end
            if hi > block_end:
                hi = block_end
            if hi > lo:
                row_visible = cutlass.Boolean(True)
                if lo <= covered_end and hi > covered_end:
                    covered_end = hi
        return row_visible, covered_end >= block_end

    @cute.kernel
    def kernel(
        self,
        mFunc: cute.Tensor,
        mBlockScratch: cute.Tensor,
        mPartialCounts: cute.Tensor,
        mFullCounts: cute.Tensor,
        mWarpPartialCounts: cute.Tensor,
        mWarpFullCounts: cute.Tensor,
        mError: cute.Tensor,
        q_len: Int32,
        k_len: Int32,
        n_func: Int32,
        num_k_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_tile, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        q_tile_begin = q_tile * Int32(self.logical_q_tile)
        build_q_idx = q_tile_begin + tidx
        builds_candidate = (tidx < Int32(self.logical_q_tile)) & (
            build_q_idx < q_len
        )
        num_words = (
            num_k_blocks + Int32(self.warp_size - 1)
        ) // Int32(self.warp_size)

        smem = utils.SmemAllocator()
        sCandidate = smem.allocate_tensor(
            element_type=Uint32,
            layout=cute.make_layout((self.max_k_block_words,)),
            byte_alignment=16,
        )
        sWarpPartialCounts = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.num_warps,)),
            byte_alignment=16,
        )
        sWarpFullCounts = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.num_warps,)),
            byte_alignment=16,
        )
        sError = smem.allocate_tensor(
            element_type=Uint32,
            layout=cute.make_layout((1,)),
            byte_alignment=4,
        )
        if tidx == Int32(0):
            sError[0] = Uint32(0)
        cute.arch.sync_threads()

        if cutlass.const_expr(self.is_prefix):
            safe_end = Int32(0)
            min_end = k_len
            if builds_candidate:
                token_end = _load_func_endpoint(mFunc, Int32(0), build_q_idx)
                invalid = (token_end < Int32(0)) | (token_end > k_len)
                if invalid:
                    self._mark_error(sError, Uint32(_ERROR_INVALID_INTERVAL))
                if token_end > Int32(0):
                    safe_end = token_end
                    if safe_end > k_len:
                        safe_end = k_len
                min_end = safe_end
            if tidx < Int32(self.logical_q_tile):
                sCandidate[tidx] = Uint32(safe_end)
                warp_max_end = cute.arch.warp_redux_sync(safe_end, "max")
                warp_min_end = cute.arch.warp_redux_sync(min_end, "min")
                if lane_idx == Int32(0):
                    sCandidate[
                        Int32(self.logical_q_tile) + warp_idx
                    ] = Uint32(warp_max_end)
                    sCandidate[
                        Int32(self.logical_q_tile + 2) + warp_idx
                    ] = Uint32(warp_min_end)
        else:
            word_idx = tidx
            while word_idx < num_words:
                sCandidate[word_idx] = Uint32(0)
                word_idx += Int32(self.threads)
            cute.arch.sync_threads()
            if builds_candidate:
                self._build_candidate_bitmap(
                    sCandidate,
                    mFunc,
                    sError,
                    build_q_idx,
                    k_len,
                    n_func,
                )

        cute.arch.sync_threads()

        partial_count = Int32(0)
        full_count = Int32(0)
        if cutlass.const_expr(self.is_prefix):
            max_end = Int32(sCandidate[Int32(self.logical_q_tile)])
            second_max_end = Int32(
                sCandidate[Int32(self.logical_q_tile + 1)]
            )
            if second_max_end > max_end:
                max_end = second_max_end
            min_end = Int32(sCandidate[Int32(self.logical_q_tile + 2)])
            second_min_end = Int32(
                sCandidate[Int32(self.logical_q_tile + 3)]
            )
            if second_min_end < min_end:
                min_end = second_min_end

            visible_block_end = (
                max_end + Int32(self.k_tile - 1)
            ) // Int32(self.k_tile)
            full_block_end = min_end // Int32(self.k_tile)
            blocks_per_warp = (
                visible_block_end + Int32(self.num_warps - 1)
            ) // Int32(self.num_warps)
            segment_begin = warp_idx * blocks_per_warp
            segment_end = segment_begin + blocks_per_warp
            if segment_end > visible_block_end:
                segment_end = visible_block_end

            full_end = segment_end
            if full_end > full_block_end:
                full_end = full_block_end
            if full_end > segment_begin:
                full_count = full_end - segment_begin

            partial_begin = segment_begin
            if partial_begin < full_block_end:
                partial_begin = full_block_end
            if segment_end > partial_begin:
                partial_count = segment_end - partial_begin
        else:
            words_per_warp = (
                num_words + Int32(self.num_warps - 1)
            ) // Int32(self.num_warps)
            warp_word_begin = warp_idx * words_per_warp
            warp_word_end = warp_word_begin + words_per_warp
            if warp_word_end > num_words:
                warp_word_end = num_words
            segment_begin = warp_word_begin * Int32(self.warp_size)
            segment_end = warp_word_end * Int32(self.warp_size)
            if segment_end > num_k_blocks:
                segment_end = num_k_blocks
            candidate_word_idx = warp_word_begin
            while candidate_word_idx < warp_word_end:
                candidate_word = sCandidate[candidate_word_idx]
                for bit_idx in cutlass.range_constexpr(self.warp_size):
                    block_id = (
                        candidate_word_idx * Int32(self.warp_size)
                        + Int32(bit_idx)
                    )
                    is_candidate = (block_id < num_k_blocks) & (
                        (candidate_word & Uint32(1 << bit_idx)) != Uint32(0)
                    )
                    if is_candidate:
                        row0_idx = q_tile_begin + lane_idx
                        row1_idx = row0_idx + Int32(self.warp_size)
                        row0_visible = cutlass.Boolean(False)
                        row0_full = cutlass.Boolean(True)
                        if row0_idx < q_len:
                            row0_visible, row0_full = self._row_block_state(
                                mFunc,
                                row0_idx,
                                block_id,
                                n_func,
                            )
                        row1_visible = cutlass.Boolean(False)
                        row1_full = cutlass.Boolean(True)
                        if row1_idx < q_len:
                            row1_visible, row1_full = self._row_block_state(
                                mFunc,
                                row1_idx,
                                block_id,
                                n_func,
                            )
                        row0_visible_bits = cute.arch.vote_ballot_sync(
                            row0_visible
                        )
                        row1_visible_bits = cute.arch.vote_ballot_sync(
                            row1_visible
                        )
                        row0_full_bits = cute.arch.vote_ballot_sync(row0_full)
                        row1_full_bits = cute.arch.vote_ballot_sync(row1_full)
                        if lane_idx == Int32(0):
                            any_visible = (
                                Uint32(row0_visible_bits)
                                | Uint32(row1_visible_bits)
                            ) != Uint32(0)
                            all_full = (
                                Uint32(row0_full_bits)
                                & Uint32(row1_full_bits)
                            ) == Uint32(0xFFFF_FFFF)
                            if all_full:
                                mBlockScratch[
                                    (q_tile, segment_begin + full_count)
                                ] = block_id
                                full_count += Int32(1)
                            elif any_visible:
                                mBlockScratch[
                                    (
                                        q_tile,
                                        segment_end - Int32(1) - partial_count,
                                    )
                                ] = block_id
                                partial_count += Int32(1)
                candidate_word_idx += Int32(1)

        if lane_idx == Int32(0):
            sWarpPartialCounts[warp_idx] = partial_count
            sWarpFullCounts[warp_idx] = full_count
            mWarpPartialCounts[(q_tile, warp_idx)] = partial_count
            mWarpFullCounts[(q_tile, warp_idx)] = full_count
        cute.arch.sync_threads()
        if tidx == Int32(0):
            total_partial = Int32(0)
            total_full = Int32(0)
            for owner_warp in cutlass.range_constexpr(self.num_warps):
                total_partial += sWarpPartialCounts[owner_warp]
                total_full += sWarpFullCounts[owner_warp]
            mPartialCounts[q_tile] = total_partial
            mFullCounts[q_tile] = total_full
            mError[q_tile] = sError[0]


class M3PlanScanSm100:
    """Build exact offsets and the D2H allocation header in one launch."""

    def __init__(self) -> None:
        self.threads = _SCAN_THREADS

    @cute.jit
    def __call__(
        self,
        mPartialCounts: cute.Tensor,
        mFullCounts: cute.Tensor,
        mError: cute.Tensor,
        mPartialOffsets: cute.Tensor,
        mFullOffsets: cute.Tensor,
        mHeader: cute.Tensor,
        num_q_tiles: Int32,
        stream: cuda.CUstream = None,
    ):
        self.kernel(
            mPartialCounts,
            mFullCounts,
            mError,
            mPartialOffsets,
            mFullOffsets,
            mHeader,
            num_q_tiles,
        ).launch(grid=(1, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        mPartialCounts: cute.Tensor,
        mFullCounts: cute.Tensor,
        mError: cute.Tensor,
        mPartialOffsets: cute.Tensor,
        mFullOffsets: cute.Tensor,
        mHeader: cute.Tensor,
        num_q_tiles: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        smem = utils.SmemAllocator()
        sPlanCounts = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.threads,)),
            byte_alignment=16,
        )
        partial_count = Int32(0)
        full_count = Int32(0)
        if tidx < num_q_tiles:
            partial_count = mPartialCounts[tidx]
            full_count = mFullCounts[tidx]
        sPlanCounts[tidx] = partial_count + full_count
        if tidx < num_q_tiles:
            partial_prefix = Int32(0)
            full_prefix = Int32(0)
            count_idx = Int32(0)
            while count_idx < tidx:
                partial_prefix += mPartialCounts[count_idx]
                full_prefix += mFullCounts[count_idx]
                count_idx += Int32(1)
            mPartialOffsets[(Int32(0), tidx)] = partial_prefix
            mFullOffsets[(Int32(0), tidx)] = full_prefix
            if tidx == num_q_tiles - Int32(1):
                total_partial = partial_prefix + partial_count
                total_full = full_prefix + full_count
                mPartialOffsets[(Int32(0), num_q_tiles)] = total_partial
                mFullOffsets[(Int32(0), num_q_tiles)] = total_full
                mHeader[0] = total_partial
                mHeader[1] = total_full
        cute.arch.sync_threads()
        if tidx == Int32(0):
            max_plan_tiles = Int32(0)
            error_flags = Uint32(0)
            count_idx = Int32(0)
            while count_idx < num_q_tiles:
                plan_tiles = sPlanCounts[count_idx]
                if plan_tiles > max_plan_tiles:
                    max_plan_tiles = plan_tiles
                error_flags = error_flags | mError[count_idx]
                count_idx += Int32(1)
            mHeader[2] = max_plan_tiles
            mHeader[3] = error_flags


class M3PlanMaterializeSm100:
    """Materialize exact lists and MMA-thread-bound R2P masks."""

    def __init__(self, is_prefix: bool = False) -> None:
        self.is_prefix = is_prefix
        self.logical_q_tile = _LOGICAL_Q_TILE
        self.k_tile = _K_TILE
        self.num_index_heads = _NUM_INDEX_HEADS
        self.warp_size = _WARP_SIZE
        self.num_classify_warps = _CLASSIFY_WARPS
        self.threads = _MATERIALIZE_THREADS

    @cute.jit
    def __call__(
        self,
        mFunc: cute.Tensor,
        mBlockScratch: cute.Tensor,
        mWarpPartialCounts: cute.Tensor,
        mWarpFullCounts: cute.Tensor,
        mPartialOffsets: cute.Tensor,
        mPartialBlockIndices: cute.Tensor,
        mPartialMasks: cute.Tensor,
        mFullOffsets: cute.Tensor,
        mFullBlockIndices: cute.Tensor,
        q_len: Int32,
        n_func: Int32,
        num_k_blocks: Int32,
        stream: cuda.CUstream = None,
    ):
        mask_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            Uint32,
            num_bits_per_copy=128,
        )
        self.kernel(
            mask_copy_atom,
            mFunc,
            mBlockScratch,
            mWarpPartialCounts,
            mWarpFullCounts,
            mPartialOffsets,
            mPartialBlockIndices,
            mPartialMasks,
            mFullOffsets,
            mFullBlockIndices,
            q_len,
            n_func,
            num_k_blocks,
        ).launch(
            grid=(cute.ceil_div(q_len, self.logical_q_tile), 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _make_row_mask(
        self,
        mFunc: cute.Tensor,
        q_idx: Int32,
        block_id: Int32,
        n_func: Int32,
        q_valid: cutlass.Boolean,
    ) -> cute.Tensor:
        block_base = block_id * Int32(self.k_tile)
        num_intervals = (n_func + Int32(1)) // Int32(2)
        rRowMask = cute.make_rmem_tensor((4,), Uint32)
        rRowMask.fill(Uint32(0))
        if q_valid:
            for word_idx in cutlass.range_constexpr(4):
                packed = Uint32(0)
                word_base = block_base + Int32(word_idx * 32)
                for interval_idx in cutlass.range(num_intervals, unroll=1):
                    token_begin = Int32(0)
                    if interval_idx > Int32(0):
                        token_begin = _load_func_endpoint(
                            mFunc,
                            interval_idx * Int32(2) - Int32(1),
                            q_idx,
                        )
                    token_end = _load_func_endpoint(
                        mFunc,
                        interval_idx * Int32(2),
                        q_idx,
                    )
                    lo = token_begin - word_base
                    hi = token_end - word_base
                    if lo < Int32(0):
                        lo = Int32(0)
                    if lo > Int32(32):
                        lo = Int32(32)
                    if hi < Int32(0):
                        hi = Int32(0)
                    if hi > Int32(32):
                        hi = Int32(32)
                    upper = shr_u32(
                        Uint32(0xFFFF_FFFF),
                        Uint32(Int32(32) - hi),
                    )
                    lower = shr_u32(
                        Uint32(0xFFFF_FFFF),
                        Uint32(Int32(32) - lo),
                    )
                    packed = packed | (upper ^ lower)
                rRowMask[word_idx] = packed
        return rRowMask

    @cute.jit
    def _store_row_mask(
        self,
        rRowMask: cute.Tensor,
        payload_idx: Int32,
        q_in_tile: Int32,
        mask_copy_atom: cute.CopyAtom,
        mPartialMasks: cute.Tensor,
    ) -> None:
        for head_idx in cutlass.range_constexpr(self.num_index_heads):
            physical_row = q_in_tile * Int32(self.num_index_heads) + Int32(head_idx)
            payload_stage = physical_row // Int32(128)
            payload_row = physical_row - payload_stage * Int32(128)
            mask_iter = mPartialMasks.iterator + cute.crd2idx(
                (
                    payload_idx,
                    payload_stage,
                    payload_row,
                    Int32(0),
                ),
                mPartialMasks.layout,
            )
            mask_ptr = cute.make_ptr(
                mPartialMasks.element_type,
                mask_iter.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            gRowMask = cute.make_tensor(mask_ptr, (4,))
            mask_tiled_copy = cute.make_cotiled_copy(
                mask_copy_atom,
                cute.make_layout((1, 4)),
                rRowMask.layout,
            )
            mask_thr_copy = mask_tiled_copy.get_slice(0)
            tRrMask = mask_thr_copy.partition_S(rRowMask)
            tRgMask = mask_thr_copy.partition_D(gRowMask)
            cute.copy(mask_copy_atom, tRrMask, tRgMask)

    @cute.jit
    def _warp_segment(
        self,
        owner_warp: Int32,
        num_k_blocks: Int32,
    ) -> tuple[Int32, Int32]:
        num_words = (
            num_k_blocks + Int32(self.warp_size - 1)
        ) // Int32(self.warp_size)
        words_per_warp = (
            num_words + Int32(self.num_classify_warps - 1)
        ) // Int32(self.num_classify_warps)
        segment_begin = owner_warp * words_per_warp * Int32(self.warp_size)
        segment_end = segment_begin + words_per_warp * Int32(self.warp_size)
        if segment_end > num_k_blocks:
            segment_end = num_k_blocks
        return segment_begin, segment_end

    @cute.kernel
    def kernel(
        self,
        mask_copy_atom: cute.CopyAtom,
        mFunc: cute.Tensor,
        mBlockScratch: cute.Tensor,
        mWarpPartialCounts: cute.Tensor,
        mWarpFullCounts: cute.Tensor,
        mPartialOffsets: cute.Tensor,
        mPartialBlockIndices: cute.Tensor,
        mPartialMasks: cute.Tensor,
        mFullOffsets: cute.Tensor,
        mFullBlockIndices: cute.Tensor,
        q_len: Int32,
        n_func: Int32,
        num_k_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_tile, _, _ = cute.arch.block_idx()
        q_idx = q_tile * Int32(self.logical_q_tile) + tidx
        q_valid = q_idx < q_len

        partial_begin = mPartialOffsets[(Int32(0), q_tile)]
        full_begin = mFullOffsets[(Int32(0), q_tile)]
        partial_warp_begin = Int32(0)
        full_warp_begin = Int32(0)
        for owner_warp in cutlass.range_constexpr(self.num_classify_warps):
            owner_warp_i32 = Int32(owner_warp)
            partial_count = mWarpPartialCounts[(q_tile, owner_warp_i32)]
            full_count = mWarpFullCounts[(q_tile, owner_warp_i32)]
            if cutlass.const_expr(self.is_prefix):
                segment_begin = partial_warp_begin + full_warp_begin
                segment_end = segment_begin + partial_count + full_count
            else:
                segment_begin, segment_end = self._warp_segment(
                    owner_warp_i32,
                    num_k_blocks,
                )
            partial_idx = Int32(0)
            while partial_idx < partial_count:
                if cutlass.const_expr(self.is_prefix):
                    block_id = segment_begin + full_count + partial_idx
                else:
                    block_id = mBlockScratch[
                        (q_tile, segment_end - Int32(1) - partial_idx)
                    ]
                payload_idx = partial_begin + partial_warp_begin + partial_idx
                if tidx == Int32(0):
                    mPartialBlockIndices[(Int32(0), payload_idx)] = block_id
                rRowMask = self._make_row_mask(
                    mFunc,
                    q_idx,
                    block_id,
                    n_func,
                    q_valid,
                )
                self._store_row_mask(
                    rRowMask,
                    payload_idx,
                    tidx,
                    mask_copy_atom,
                    mPartialMasks,
                )
                partial_idx += Int32(1)
            partial_warp_begin += partial_count

            full_idx = tidx
            while full_idx < full_count:
                if cutlass.const_expr(self.is_prefix):
                    block_id = segment_begin + full_idx
                else:
                    block_id = mBlockScratch[
                        (q_tile, segment_begin + full_idx)
                    ]
                output_idx = full_begin + full_warp_begin + full_idx
                mFullBlockIndices[(Int32(0), output_idx)] = block_id
                full_idx += Int32(self.threads)
            full_warp_begin += full_count


def _make_fake_tensor(
    dtype,
    shape,
    *,
    stride_order,
    assumed_align: int = 16,
):
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=stride_order,
        assumed_align=assumed_align,
    )


def _compile_classify(device: torch.device, n_func: int):
    capability = torch.cuda.get_device_capability(device)
    is_prefix = n_func == 1
    key = ("m3_plan_classify_sm100", capability, is_prefix)
    if key not in _CLASSIFY_COMPILE_CACHE:
        func_rows = cute.sym_int64()
        func_q_len = cute.sym_int64()
        num_q_tiles = cute.sym_int64()
        num_k_blocks = cute.sym_int64()
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compile_args = (
            M3PlanClassifySm100(is_prefix=is_prefix),
            _make_fake_tensor(
                Int32,
                (func_rows, func_q_len),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            _make_fake_tensor(
                Int32,
                (num_q_tiles, num_k_blocks),
                stride_order=(1, 0),
                assumed_align=16,
            ),
            _make_fake_tensor(
                Int32,
                (num_q_tiles,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                Int32,
                (num_q_tiles,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                Int32,
                (num_q_tiles, _CLASSIFY_WARPS),
                stride_order=(1, 0),
                assumed_align=16,
            ),
            _make_fake_tensor(
                Int32,
                (num_q_tiles, _CLASSIFY_WARPS),
                stride_order=(1, 0),
                assumed_align=16,
            ),
            _make_fake_tensor(
                Uint32,
                (num_q_tiles,),
                stride_order=(0,),
                assumed_align=4,
            ),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            fake_stream,
        )
        _CLASSIFY_COMPILE_CACHE[key] = compile_or_load(
            key,
            lambda: compile_with_timing(*compile_args, options="--enable-tvm-ffi"),
            log_prefix="m3_plan_classify",
        )
    return _CLASSIFY_COMPILE_CACHE[key]


def _compile_scan(device: torch.device):
    capability = torch.cuda.get_device_capability(device)
    key = ("m3_plan_scan_sm100", capability)
    if key not in _SCAN_COMPILE_CACHE:
        num_q_tiles = cute.sym_int64()
        offsets_len = cute.sym_int64()
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compile_args = (
            M3PlanScanSm100(),
            _make_fake_tensor(
                Int32,
                (num_q_tiles,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                Int32,
                (num_q_tiles,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                Uint32,
                (num_q_tiles,),
                stride_order=(0,),
                assumed_align=4,
            ),
            _make_fake_tensor(
                Int32,
                (1, offsets_len),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            _make_fake_tensor(
                Int32,
                (1, offsets_len),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            _make_fake_tensor(
                Int64,
                (4,),
                stride_order=(0,),
                assumed_align=8,
            ),
            Int32(1),
            fake_stream,
        )
        _SCAN_COMPILE_CACHE[key] = compile_or_load(
            key,
            lambda: compile_with_timing(*compile_args, options="--enable-tvm-ffi"),
            log_prefix="m3_plan_scan",
        )
    return _SCAN_COMPILE_CACHE[key]


def _compile_materialize(device: torch.device, n_func: int):
    capability = torch.cuda.get_device_capability(device)
    is_prefix = n_func == 1
    key = ("m3_plan_materialize_sm100", capability, is_prefix)
    if key not in _MATERIALIZE_COMPILE_CACHE:
        n_func = cute.sym_int64()
        func_q_len = cute.sym_int64()
        num_q_tiles = cute.sym_int64()
        num_k_blocks = cute.sym_int64()
        offsets_len = cute.sym_int64()
        partial_tiles = cute.sym_int64()
        full_tiles = cute.sym_int64()
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compile_args = (
            M3PlanMaterializeSm100(is_prefix=is_prefix),
            _make_fake_tensor(
                Int32,
                (n_func, func_q_len),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            _make_fake_tensor(
                Int32,
                (num_q_tiles, num_k_blocks),
                stride_order=(1, 0),
                assumed_align=16,
            ),
            _make_fake_tensor(
                Int32,
                (num_q_tiles, _CLASSIFY_WARPS),
                stride_order=(1, 0),
                assumed_align=16,
            ),
            _make_fake_tensor(
                Int32,
                (num_q_tiles, _CLASSIFY_WARPS),
                stride_order=(1, 0),
                assumed_align=16,
            ),
            _make_fake_tensor(
                Int32,
                (1, offsets_len),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            _make_fake_tensor(
                Int32,
                (1, partial_tiles),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            _make_fake_tensor(
                Uint32,
                (partial_tiles, 2, 128, 4),
                stride_order=(3, 2, 1, 0),
                assumed_align=16,
            ),
            _make_fake_tensor(
                Int32,
                (1, offsets_len),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            _make_fake_tensor(
                Int32,
                (1, full_tiles),
                stride_order=(1, 0),
                assumed_align=4,
            ),
            Int32(1),
            Int32(1),
            Int32(1),
            fake_stream,
        )
        _MATERIALIZE_COMPILE_CACHE[key] = compile_or_load(
            key,
            lambda: compile_with_timing(*compile_args, options="--enable-tvm-ffi"),
            log_prefix="m3_plan_materialize",
        )
    return _MATERIALIZE_COMPILE_CACHE[key]


def _exclusive_offsets(counts: torch.Tensor) -> torch.Tensor:
    offsets = torch.empty(
        (counts.numel() + 1,),
        dtype=torch.int32,
        device=counts.device,
    )
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
    return offsets


def _build_offsets_and_header(
    partial_counts: torch.Tensor,
    full_counts: torch.Tensor,
    error: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_q_tiles = int(partial_counts.numel())
    if num_q_tiles <= _SCAN_THREADS:
        offsets_shape = (1, num_q_tiles + 1)
        partial_offsets = torch.empty(
            offsets_shape,
            dtype=torch.int32,
            device=partial_counts.device,
        )
        full_offsets = torch.empty_like(partial_offsets)
        header = torch.empty((4,), dtype=torch.int64, device=partial_counts.device)
        _compile_scan(partial_counts.device)(
            partial_counts,
            full_counts,
            error,
            partial_offsets,
            full_offsets,
            header,
            Int32(num_q_tiles),
        )
        return partial_offsets, full_offsets, header

    partial_offsets_1d = _exclusive_offsets(partial_counts)
    full_offsets_1d = _exclusive_offsets(full_counts)
    plan_tile_counts = partial_counts + full_counts
    header = torch.stack(
        (
            partial_offsets_1d[-1].to(torch.int64),
            full_offsets_1d[-1].to(torch.int64),
            plan_tile_counts.max().to(torch.int64),
            error.to(torch.int64).max(),
        )
    )
    return (
        partial_offsets_1d.unsqueeze(0).contiguous(),
        full_offsets_1d.unsqueeze(0).contiguous(),
        header,
    )


def _validate_arbitrary_func(
    arbitrary_func: torch.Tensor,
    q_len: int,
    k_len: int,
) -> None:
    if not isinstance(q_len, int) or not isinstance(k_len, int):
        raise TypeError("q_len and k_len must be Python ints")
    if q_len <= 0 or q_len > _MAX_SEQLEN:
        raise ValueError(f"q_len must be in [1, {_MAX_SEQLEN}]")
    if k_len <= 0 or k_len > _MAX_SEQLEN:
        raise ValueError(f"k_len must be in [1, {_MAX_SEQLEN}]")
    if k_len < q_len:
        raise ValueError("k_len must be greater than or equal to q_len")
    if arbitrary_func.dtype != torch.int32:
        raise TypeError("arbitrary_func must be torch.int32")
    if not arbitrary_func.is_cuda:
        raise ValueError("arbitrary_func must be a CUDA tensor")
    if not arbitrary_func.is_contiguous():
        raise ValueError("arbitrary_func must be contiguous")
    if arbitrary_func.ndim != 4 or tuple(arbitrary_func.shape[:2]) != (1, 1):
        raise ValueError("arbitrary_func must have shape [1, 1, n_func, func_q_len]")
    n_func = int(arbitrary_func.shape[2])
    if n_func <= 0 or n_func % 2 == 0:
        raise ValueError("arbitrary_func.shape[2] must be a positive odd n_func")
    if int(arbitrary_func.shape[3]) < q_len + _PHYSICAL_M_TILE:
        raise ValueError(
            f"arbitrary_func.shape[3] must be at least q_len + {_PHYSICAL_M_TILE}"
        )
    capability = torch.cuda.get_device_capability(arbitrary_func.device)
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError("M3 arbitrary-mask plan requires an SM100 or SM103 GPU")


def compile_m3_arbitrary_mask_plan(
    arbitrary_func: torch.Tensor,
    q_len: int,
    k_len: int | None = None,
    local_block_positions: torch.Tensor | None = None,
) -> M3ArbitraryMaskPlan:
    """Compile one batch-1 Func tensor into an offline exact-size plan.

    This setup routine intentionally reads four device-side size/error scalars
    to the host before allocating the materialized tensors. It must not run in
    a training step or during CUDA Graph capture; callers cache and reuse its
    returned immutable plan.
    """

    if k_len is None:
        k_len = q_len
    _validate_arbitrary_func(arbitrary_func, q_len, k_len)
    device = arbitrary_func.device
    if local_block_positions is None:
        local_block_positions = (
            torch.arange(q_len, dtype=torch.int32, device=device) + k_len - q_len
        )
    elif (
        local_block_positions.dtype != torch.int32
        or local_block_positions.device != device
        or tuple(local_block_positions.shape) != (q_len,)
        or not local_block_positions.is_contiguous()
    ):
        raise ValueError(
            "local_block_positions must be contiguous int32 [q_len] on the Func device"
        )
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "compile_m3_arbitrary_mask_plan is an offline setup API and "
                "cannot run during CUDA Graph capture"
            )
    func_rows = arbitrary_func[0, 0]
    n_func = int(func_rows.shape[0])
    num_q_tiles = (q_len + _LOGICAL_Q_TILE - 1) // _LOGICAL_Q_TILE
    num_k_blocks = (k_len + _K_TILE - 1) // _K_TILE

    scratch_columns = (
        _PREFIX_BLOCK_SCRATCH_COLUMNS if n_func == 1 else num_k_blocks
    )
    block_scratch = torch.empty(
        (num_q_tiles, scratch_columns),
        dtype=torch.int32,
        device=device,
    )
    partial_counts = torch.empty(
        (num_q_tiles,),
        dtype=torch.int32,
        device=device,
    )
    full_counts = torch.empty(
        (num_q_tiles,),
        dtype=torch.int32,
        device=device,
    )
    warp_partial_counts = torch.empty(
        (num_q_tiles, _CLASSIFY_WARPS),
        dtype=torch.int32,
        device=device,
    )
    warp_full_counts = torch.empty_like(warp_partial_counts)
    error = torch.empty((num_q_tiles,), dtype=torch.uint32, device=device)

    _compile_classify(device, n_func)(
        func_rows,
        block_scratch,
        partial_counts,
        full_counts,
        warp_partial_counts,
        warp_full_counts,
        error,
        Int32(q_len),
        Int32(k_len),
        Int32(n_func),
        Int32(num_k_blocks),
    )

    partial_offsets, full_offsets, header = _build_offsets_and_header(
        partial_counts,
        full_counts,
        error,
    )
    num_partial, num_full, max_plan_tiles, error_value = (
        int(value) for value in header.cpu().tolist()
    )
    if error_value & _ERROR_INVALID_INTERVAL:
        raise ValueError(
            "arbitrary_func intervals must be clamped to [0, k_len], "
            "have begin <= end, and have ordered non-empty begins"
        )

    partial_block_indices = torch.empty(
        (1, num_partial),
        dtype=torch.int32,
        device=device,
    )
    partial_masks = torch.empty(
        (num_partial, 2, 128, 4),
        dtype=torch.uint32,
        device=device,
    )
    full_block_indices = torch.empty(
        (1, num_full),
        dtype=torch.int32,
        device=device,
    )

    if num_partial + num_full > 0:
        _compile_materialize(device, n_func)(
            func_rows,
            block_scratch,
            warp_partial_counts,
            warp_full_counts,
            partial_offsets,
            partial_block_indices,
            partial_masks,
            full_offsets,
            full_block_indices,
            Int32(q_len),
            Int32(n_func),
            Int32(num_k_blocks),
        )

    return M3ArbitraryMaskPlan(
        q_len=q_len,
        k_len=k_len,
        max_plan_tiles=max_plan_tiles,
        partial_offsets=partial_offsets,
        partial_block_indices=partial_block_indices,
        partial_masks=partial_masks,
        full_offsets=full_offsets,
        full_block_indices=full_block_indices,
        local_block_positions=local_block_positions.unsqueeze(0),
    )


__all__ = [
    "M3ArbitraryMaskPlan",
    "M3PlanClassifySm100",
    "M3PlanScanSm100",
    "M3PlanMaterializeSm100",
    "compile_m3_arbitrary_mask_plan",
]
