"""SM100/SM103 paged prefill indexer GEMM for MSA v1 inference."""

import enum
import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cuda.bindings.driver as cuda

from inference.msa_v1.attention.prefill._common.dsl.utils import fmax_reduce

_HEAD_DIM = 128
_PHYSICAL_M_TILE = 256
_K_TILE = 128
_SM_SCALE = 1.0 / math.sqrt(_HEAD_DIM)
M3_PAGED_DIRECT_SCORE_NUM_HEADS = (1, 2, 4)


@dsl_user_op
def _shr_u32(
    val: cutlass.Uint32, shift: cutlass.Uint32, *, loc=None, ip=None
) -> cutlass.Uint32:
    """Right-shift an unsigned value with PTX-defined clamping semantics."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Uint32(val).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(shift).ir_value(loc=loc, ip=ip),
            ],
            "shr.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class NamedBarrierIndexerSm100(enum.IntEnum):
    TmemPtr = enum.auto()
    Final = enum.auto()


class M3IndexerGemmSm100:
    """Compute direct FP32 block-max scores with a two-CTA tcgen05 pipeline."""

    def __init__(
        self,
        *,
        compute_capability: tuple[int, int],
        num_index_heads: int = 4,
        q_stages: int = 1,
        k_stages: int = 4,
        acc_stages: int = 4,
    ) -> None:
        if (q_stages, k_stages, acc_stages) != (1, 4, 4):
            raise ValueError("M3 K1 requires q/k/acc stages = 1/4/4")
        if num_index_heads not in M3_PAGED_DIRECT_SCORE_NUM_HEADS:
            raise ValueError("M3 indexer requires 1, 2, or 4 local index heads")

        self.num_index_heads = num_index_heads
        if compute_capability not in {(10, 0), (10, 3)}:
            raise ValueError("compute_capability must be SM100 or SM103")
        self.use_tmem_load_reduce = compute_capability == (10, 3)
        self.head_dim = _HEAD_DIM
        self.physical_m_tile = _PHYSICAL_M_TILE
        self.logical_q_tile = _PHYSICAL_M_TILE // num_index_heads
        self.k_tile = _K_TILE
        self.sm_scale = _SM_SCALE
        self.io_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.q_stages = q_stages
        self.k_stages = k_stages
        self.acc_stages = acc_stages

        self.cta_group_size = 2
        self.cluster_shape_mnk = (self.cta_group_size, 1, 1)
        self.mma_tiler_mnk = (
            self.physical_m_tile,
            self.k_tile,
            self.head_dim,
        )
        self.mma_inst_shape_mnk = (
            self.physical_m_tile,
            self.k_tile,
            16,
        )

        self.score_warp_ids = tuple(range(12))
        self.q_load_warp_id = 12
        self.k_load_warp_id = 13
        self.mma_warp_id = 14
        self.empty_warp_ids = (15,)
        self.score_threads = cute.arch.WARP_SIZE * len(self.score_warp_ids)
        self.score_consumer_warps = 4
        self.score_worker_groups = len(self.score_warp_ids) // self.score_consumer_warps
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.score_warp_ids,
                self.q_load_warp_id,
                self.k_load_warp_id,
                self.mma_warp_id,
                *self.empty_warp_ids,
            )
        )
        self.num_regs_score = 152
        self.num_regs_other = 32
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        self.neg_inf = -1.0e30

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mPageTable: cute.Tensor,
        mScore: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mTaskBatchIdx: cute.Tensor,
        mTaskQLocalBegin: cute.Tensor,
        mFragmentIndices: Optional[cute.Tensor],
        num_task_slots: cutlass.Int32,
        inv_lse_temperature: cutlass.Float32,
        stream: cuda.CUstream = None,
    ):
        op = tcgen05.MmaF16BF16Op(
            self.io_dtype,
            self.acc_dtype,
            self.mma_inst_shape_mnk,
            tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)

        q_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler_mnk,
            self.io_dtype,
            self.q_stages,
        )
        k_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler_mnk,
            self.io_dtype,
            self.k_stages,
        )
        cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (tiled_mma.thr_id,))

        tma_load_op = cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
        q_tma_atom, q_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(q_smem_layout, mode=[0, 1, 2]),
            self.mma_tiler_mnk,
            tiled_mma,
            cta_layout_vmnk.shape,
        )
        k_tma_atom, k_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(k_smem_layout, mode=[0, 1, 2]),
            self.mma_tiler_mnk,
            tiled_mma,
            cta_layout_vmnk.shape,
        )

        @cute.struct
        class SharedStorage:
            q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_stages * 2]
            k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.k_stages * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.acc_stages * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buffer: cutlass.Int32

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            q_tma_atom,
            q_tma_tensor,
            k_tma_atom,
            k_tma_tensor,
            cta_layout_vmnk,
            q_smem_layout,
            k_smem_layout,
            mScore,
            mCuSeqlensQ,
            mCuSeqlensK,
            mPageTable,
            mTaskBatchIdx,
            mTaskQLocalBegin,
            mFragmentIndices,
            inv_lse_temperature,
        ).launch(
            grid=(num_task_slots * self.cta_group_size, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _q_producer(
        self,
        q_pipe,
        q_tma_atom: cute.CopyAtom,
        tQgQ: cute.Tensor,
        tQsQ: cute.Tensor,
        q_mcast_mask: cutlass.Int16,
        num_tiles: cutlass.Int32,
    ):
        state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.q_stages
        )
        if num_tiles > cutlass.Int32(0):
            q_pipe.producer_acquire(state)
            cute.copy(
                q_tma_atom,
                tQgQ[(None, 0, 0)],
                tQsQ[(None, state.index)],
                tma_bar_ptr=q_pipe.producer_get_barrier(state),
                mcast_mask=q_mcast_mask,
            )
            state.advance()
            q_pipe.producer_tail(state)

    @cute.jit
    def _k_producer(
        self,
        k_pipe,
        k_tma_atom: cute.CopyAtom,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        mPageTable: cute.Tensor,
        batch_idx: cutlass.Int32,
        k_mcast_mask: cutlass.Int16,
        num_tiles: cutlass.Int32,
    ):
        state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.k_stages
        )
        for tile_idx in cutlass.range(num_tiles, unroll=1):
            k_pipe.producer_acquire(state)
            physical_page = mPageTable[batch_idx, tile_idx]
            cute.copy(
                k_tma_atom,
                tKgK[(None, 0, physical_page)],
                tKsK[(None, state.index)],
                tma_bar_ptr=k_pipe.producer_get_barrier(state),
                mcast_mask=k_mcast_mask,
            )
            state.advance()
        if num_tiles > cutlass.Int32(0):
            k_pipe.producer_tail(state)

    @cute.jit
    def _mma_producer(
        self,
        q_pipe,
        k_pipe,
        acc_pipe,
        tiled_mma: cute.TiledMma,
        tCrQ: cute.Tensor,
        tCrK: cute.Tensor,
        tCtAcc_staged: cute.Tensor,
        is_leader_cta: cutlass.Boolean,
        num_tiles: cutlass.Int32,
    ):
        q_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.q_stages
        )
        k_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.k_stages
        )
        acc_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.acc_stages
        )
        num_k_blocks = cute.size(tCrQ, mode=[2])

        if is_leader_cta and num_tiles > cutlass.Int32(0):
            q_pipe.consumer_wait(q_state)
            for _ in cutlass.range(num_tiles, unroll=1):
                k_pipe.consumer_wait(k_state)
                acc_pipe.producer_acquire(acc_state)
                tCtAcc = tCtAcc_staged[(None, None, None, acc_state.index)]
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrQ[(None, None, k_block_idx, 0)],
                        tCrK[(None, None, k_block_idx, k_state.index)],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                k_pipe.consumer_release(k_state)
                k_state.advance()
                acc_pipe.producer_commit(acc_state)
                acc_state.advance()
            q_pipe.consumer_release(q_state)
            q_state.advance()
            acc_pipe.producer_tail(acc_state)

    @cute.jit
    def _causal_mask_words(
        self,
        q_valid: cutlass.Boolean,
        q_local: cutlass.Int32,
        seq_q: cutlass.Int32,
        seq_k: cutlass.Int32,
        block_id: cutlass.Int32,
    ) -> cute.Tensor:
        """Return the visible-token prefix mask for one Q row and K block."""

        visible_in_block = cutlass.Int32(0)
        if q_valid:
            visible_end = seq_k - seq_q + q_local + cutlass.Int32(1)
            visible_in_block = visible_end - block_id * cutlass.Int32(self.k_tile)
            if visible_in_block < cutlass.Int32(0):
                visible_in_block = cutlass.Int32(0)
            if visible_in_block > cutlass.Int32(self.k_tile):
                visible_in_block = cutlass.Int32(self.k_tile)

        rMask = cute.make_rmem_tensor((4,), cutlass.Uint32)
        for word_idx in cutlass.range_constexpr(4):
            word_bits = visible_in_block - cutlass.Int32(word_idx * 32)
            if word_bits < cutlass.Int32(0):
                word_bits = cutlass.Int32(0)
            if word_bits > cutlass.Int32(32):
                word_bits = cutlass.Int32(32)
            rMask[word_idx] = _shr_u32(
                cutlass.Uint32(0xFFFF_FFFF),
                cutlass.Uint32(cutlass.Int32(32) - word_bits),
            )
        return rMask

    @cute.jit
    def _score_consumer(
        self,
        tile_worker_idx: cutlass.Int32,
        row_m: cutlass.Int32,
        q_global_begin: cutlass.Int32,
        q_local_begin: cutlass.Int32,
        seq_q: cutlass.Int32,
        seq_k: cutlass.Int32,
        num_full_tiles: cutlass.Int32,
        num_tiles: cutlass.Int32,
        inv_lse_temperature: cutlass.Float32,
        acc_pipe,
        tmem_tiled_copy: cute.TiledCopy,
        tmem_load_reduce_atom: cute.CopyAtom,
        tTR_tAcc_staged: cute.Tensor,
        tTR_rAcc: cute.Tensor,
        tTR_rBlockMax: cute.Tensor,
        mScore: cute.Tensor,
    ):
        """Consume interleaved accumulator stages across score warpgroups."""

        sm_scale_temperature = cutlass.Float32(self.sm_scale) * inv_lse_temperature
        q_in_tile = row_m // cutlass.Int32(self.num_index_heads)
        q_head = row_m - q_in_tile * cutlass.Int32(self.num_index_heads)
        q_local = q_local_begin + q_in_tile
        q_global = q_global_begin + q_in_tile
        q_valid = q_local < seq_q
        worker_full_tiles = (
            num_full_tiles
            + cutlass.Int32(self.score_worker_groups - 1)
            - tile_worker_idx
        ) // cutlass.Int32(self.score_worker_groups)

        for worker_tile_idx in cutlass.range(worker_full_tiles, unroll=1):
            tile_idx = tile_worker_idx + worker_tile_idx * cutlass.Int32(
                self.score_worker_groups
            )
            state = pipeline.PipelineState(
                self.acc_stages,
                tile_idx,
                tile_idx % cutlass.Int32(self.acc_stages),
                (tile_idx // cutlass.Int32(self.acc_stages)) & cutlass.Int32(1),
            )
            acc_pipe.consumer_wait(state)
            tTR_tAcc = tTR_tAcc_staged[(None, None, None, None, state.index)]
            if cutlass.const_expr(self.use_tmem_load_reduce):
                for chunk in cutlass.range_constexpr(4):
                    cute.copy_atom_call(
                        tmem_load_reduce_atom,
                        tTR_tAcc[(None, chunk, 0, 0)],
                        (
                            tTR_rAcc[(None, chunk, 0, 0)],
                            tTR_rBlockMax[(None, chunk)],
                        ),
                    )
            else:
                cute.copy(tmem_tiled_copy, tTR_tAcc, tTR_rAcc)
            cute.arch.fence_view_async_tmem_load()
            with cute.arch.elect_one():
                acc_pipe.consumer_release(state)

            if q_valid:
                if cutlass.const_expr(self.use_tmem_load_reduce):
                    block_max_01 = cute.arch.fmax(
                        tTR_rBlockMax[(0, 0)], tTR_rBlockMax[(0, 1)]
                    )
                    block_max_23 = cute.arch.fmax(
                        tTR_rBlockMax[(0, 2)], tTR_rBlockMax[(0, 3)]
                    )
                    block_max = cute.arch.fmax(block_max_01, block_max_23)
                else:
                    # SM100 has no ld.red; each lane owns a complete FP32 row.
                    block_max = fmax_reduce(tTR_rAcc.load())
                score = block_max * sm_scale_temperature
                mScore[q_head, q_global, tile_idx] = score

        first_boundary_tile = num_full_tiles
        boundary_remainder = (
            first_boundary_tile
            - tile_worker_idx
            + cutlass.Int32(self.score_worker_groups)
        ) % cutlass.Int32(self.score_worker_groups)
        first_worker_boundary = first_boundary_tile + (
            cutlass.Int32(self.score_worker_groups) - boundary_remainder
        ) % cutlass.Int32(self.score_worker_groups)
        worker_boundary_tiles = (
            num_tiles
            + cutlass.Int32(self.score_worker_groups - 1)
            - first_worker_boundary
        ) // cutlass.Int32(self.score_worker_groups)

        for worker_tile_idx in cutlass.range(worker_boundary_tiles, unroll=1):
            tile_idx = first_worker_boundary + worker_tile_idx * cutlass.Int32(
                self.score_worker_groups
            )
            state = pipeline.PipelineState(
                self.acc_stages,
                tile_idx,
                tile_idx % cutlass.Int32(self.acc_stages),
                (tile_idx // cutlass.Int32(self.acc_stages)) & cutlass.Int32(1),
            )
            acc_pipe.consumer_wait(state)
            tTR_tAcc = tTR_tAcc_staged[(None, None, None, None, state.index)]
            cute.copy(tmem_tiled_copy, tTR_tAcc, tTR_rAcc)
            cute.arch.fence_view_async_tmem_load()
            with cute.arch.elect_one():
                acc_pipe.consumer_release(state)

            if tile_idx < num_tiles:
                rMask = self._causal_mask_words(
                    q_valid,
                    q_local,
                    seq_q,
                    seq_k,
                    tile_idx,
                )
                mask_0 = rMask[0]
                mask_1 = rMask[1]
                mask_2 = rMask[2]
                mask_3 = rMask[3]
                row_visible = (mask_0 | mask_1 | mask_2 | mask_3) != cutlass.Uint32(0)
                block_max_0 = cutlass.Float32(self.neg_inf)
                block_max_1 = cutlass.Float32(self.neg_inf)
                block_max_2 = cutlass.Float32(self.neg_inf)
                block_max_3 = cutlass.Float32(self.neg_inf)
                if q_valid and row_visible:
                    for word_idx in cutlass.range_constexpr(4):
                        word = mask_0
                        if cutlass.const_expr(word_idx == 1):
                            word = mask_1
                        elif cutlass.const_expr(word_idx == 2):
                            word = mask_2
                        elif cutlass.const_expr(word_idx == 3):
                            word = mask_3
                        for bit_idx in cutlass.range_constexpr(32):
                            col = word_idx * 32 + bit_idx
                            if (word & cutlass.Uint32(1 << bit_idx)) != cutlass.Uint32(
                                0
                            ):
                                max_group = col & 3
                                if cutlass.const_expr(max_group == 0):
                                    block_max_0 = cute.arch.fmax(
                                        block_max_0,
                                        tTR_rAcc[col],
                                    )
                                elif cutlass.const_expr(max_group == 1):
                                    block_max_1 = cute.arch.fmax(
                                        block_max_1,
                                        tTR_rAcc[col],
                                    )
                                elif cutlass.const_expr(max_group == 2):
                                    block_max_2 = cute.arch.fmax(
                                        block_max_2,
                                        tTR_rAcc[col],
                                    )
                                else:
                                    block_max_3 = cute.arch.fmax(
                                        block_max_3,
                                        tTR_rAcc[col],
                                    )
                if q_valid:
                    score = cutlass.Float32(self.neg_inf)
                    if row_visible:
                        block_max_01 = cute.arch.fmax(block_max_0, block_max_1)
                        block_max_23 = cute.arch.fmax(block_max_2, block_max_3)
                        block_max = cute.arch.fmax(block_max_01, block_max_23)
                        score = block_max * sm_scale_temperature
                    mScore[q_head, q_global, tile_idx] = score

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        q_tma_atom: cute.CopyAtom,
        q_tma_tensor: cute.Tensor,
        k_tma_atom: cute.CopyAtom,
        k_tma_tensor: cute.Tensor,
        cta_layout_vmnk: cute.Layout,
        q_smem_layout: cute.ComposedLayout,
        k_smem_layout: cute.ComposedLayout,
        mScore: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mPageTable: cute.Tensor,
        mTaskBatchIdx: cute.Tensor,
        mTaskQLocalBegin: cute.Tensor,
        mFragmentIndices: Optional[cute.Tensor],
        inv_lse_temperature: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank)
        mma_tile_coord_v = bidx % cute.size(cta_layout_vmnk, mode=[0])
        is_leader_cta = mma_tile_coord_v == 0
        task_idx = bidx // cutlass.Int32(self.cta_group_size)

        batch_idx = cutlass.Int32(-1)
        q_local_begin = cutlass.Int32(0)
        if cute.arch.lane_idx() == cutlass.Int32(0):
            batch_idx = mTaskBatchIdx[task_idx]
            q_local_begin = mTaskQLocalBegin[task_idx]
        batch_idx = cute.arch.shuffle_sync(batch_idx, 0)
        q_local_begin = cute.arch.shuffle_sync(q_local_begin, 0)

        q_start = cutlass.Int32(0)
        k_start = cutlass.Int32(0)
        seq_q = cutlass.Int32(0)
        seq_k = cutlass.Int32(0)
        num_full_tiles = cutlass.Int32(0)
        num_tiles = cutlass.Int32(0)
        if cute.arch.lane_idx() == cutlass.Int32(0) and batch_idx >= cutlass.Int32(0):
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
            first_visible = seq_k - seq_q + q_local_begin + cutlass.Int32(1)
            last_visible = seq_k - seq_q + q_local_end
            if first_visible < cutlass.Int32(0):
                first_visible = cutlass.Int32(0)
            if last_visible < cutlass.Int32(0):
                last_visible = cutlass.Int32(0)
            if first_visible > seq_k:
                first_visible = seq_k
            if last_visible > seq_k:
                last_visible = seq_k
            num_full_tiles = first_visible // cutlass.Int32(self.k_tile)
            num_tiles = cute.ceil_div(last_visible, self.k_tile)

        q_start = cute.arch.shuffle_sync(q_start, 0)
        k_start = cute.arch.shuffle_sync(k_start, 0)
        seq_q = cute.arch.shuffle_sync(seq_q, 0)
        seq_k = cute.arch.shuffle_sync(seq_k, 0)
        num_full_tiles = cute.arch.shuffle_sync(num_full_tiles, 0)
        num_tiles = cute.arch.shuffle_sync(num_tiles, 0)

        if warp_idx == self.q_load_warp_id:
            cpasync.prefetch_descriptor(q_tma_atom)
        if warp_idx == self.k_load_warp_id:
            cpasync.prefetch_descriptor(k_tma_atom)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sQ = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=q_smem_layout.outer,
            byte_alignment=128,
            swizzle=q_smem_layout.inner,
        )
        sK = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=k_smem_layout.outer,
            byte_alignment=128,
            swizzle=k_smem_layout.inner,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierIndexerSm100.TmemPtr),
            num_threads=self.threads_per_cta,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buffer,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
        )

        thread_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        q_tma_bytes = (
            cute.size_in_bytes(
                self.io_dtype,
                cute.select(q_smem_layout, mode=[0, 1, 2]),
            )
            * self.cta_group_size
        )
        k_tma_bytes = (
            cute.size_in_bytes(
                self.io_dtype,
                cute.select(k_smem_layout, mode=[0, 1, 2]),
            )
            * self.cta_group_size
        )
        q_pipe = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stages,
            producer_group=thread_group,
            consumer_group=thread_group,
            tx_count=q_tma_bytes,
            barrier_storage=storage.q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        k_pipe = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stages,
            producer_group=thread_group,
            consumer_group=thread_group,
            tx_count=k_tma_bytes,
            barrier_storage=storage.k_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        acc_pipe = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stages,
            producer_group=thread_group,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.cta_group_size * self.score_consumer_warps,
            ),
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        q_row_base = (q_start + q_local_begin) * cutlass.Int32(self.num_index_heads)
        mQ_task = cute.domain_offset((q_row_base, 0), q_tma_tensor)
        gQ = cute.local_tile(
            mQ_task,
            cute.slice_(self.mma_tiler_mnk, (None, 0, None)),
            (None, None),
        )
        gK = cute.local_tile(
            k_tma_tensor,
            cute.slice_(self.mma_tiler_mnk, (0, None, None)),
            (None, 0, None),
        )
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgQ = thr_mma.partition_A(gQ)
        tCgK = thr_mma.partition_B(gK)
        tQsQ, tQgQ = cpasync.tma_partition(
            q_tma_atom,
            cta_coord_vmnk[2],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tCgQ, 0, 3),
        )
        tKsK, tKgK = cpasync.tma_partition(
            k_tma_atom,
            cta_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tCgK, 0, 3),
        )
        q_mcast_mask = cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, cta_coord_vmnk, mcast_mode=2
        )
        k_mcast_mask = cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, cta_coord_vmnk, mcast_mode=1
        )

        tCrQ = tiled_mma.make_fragment_A(sQ)
        tCrK = tiled_mma.make_fragment_B(sK)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler_mnk[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))

        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)
        tmem.allocate(self.tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc_staged = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        tmem_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), self.acc_dtype
        )
        if cutlass.const_expr(self.use_tmem_load_reduce):
            tmem_load_reduce_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tcgen05.copy.Repetition.x32,
                    redOp=tcgen05.TmemLoadRedOp.MAX,
                ),
                self.acc_dtype,
            )
        else:
            tmem_load_reduce_atom = tmem_atom
        tmem_tiled_copy = tcgen05.make_tmem_copy(
            tmem_atom, tCtAcc_staged[(None, None, None, 0)]
        )
        score_tid = tidx % cutlass.Int32(self.score_threads)
        tile_worker_idx = score_tid // cutlass.Int32(self.physical_m_tile // 2)
        tmem_tid = score_tid - tile_worker_idx * cutlass.Int32(
            self.physical_m_tile // 2
        )
        tmem_thr_copy = tmem_tiled_copy.get_slice(tmem_tid)
        tTR_tAcc_staged = tmem_thr_copy.partition_S(tCtAcc_staged)
        tCcC = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_mnk[:2]))
        tTR_cC = tmem_thr_copy.partition_D(tCcC)
        tTR_rAcc = cute.make_rmem_tensor(tTR_cC.shape, self.acc_dtype)
        tTR_rBlockMax = cute.make_rmem_tensor(
            cute.make_layout((1, 4), stride=(0, 1)), self.acc_dtype
        )

        if warp_idx <= self.score_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_score)
            row_m = tTR_cC[0][0]
            self._score_consumer(
                tile_worker_idx,
                row_m,
                q_start + q_local_begin,
                q_local_begin,
                seq_q,
                seq_k,
                num_full_tiles,
                num_tiles,
                inv_lse_temperature,
                acc_pipe,
                tmem_tiled_copy,
                tmem_load_reduce_atom,
                tTR_tAcc_staged,
                tTR_rAcc,
                tTR_rBlockMax,
                mScore,
            )
        elif warp_idx == self.q_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self._q_producer(
                q_pipe,
                q_tma_atom,
                tQgQ,
                tQsQ,
                q_mcast_mask,
                num_tiles,
            )
        elif warp_idx == self.k_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self._k_producer(
                k_pipe,
                k_tma_atom,
                tKgK,
                tKsK,
                mPageTable,
                batch_idx,
                k_mcast_mask,
                num_tiles,
            )
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self._mma_producer(
                q_pipe,
                k_pipe,
                acc_pipe,
                tiled_mma,
                tCrQ,
                tCrK,
                tCtAcc_staged,
                is_leader_cta,
                num_tiles,
            )
        else:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        tmem.relinquish_alloc_permit()
        pipeline.sync(barrier_id=int(NamedBarrierIndexerSm100.Final))
        tmem.free(tmem_ptr)


__all__ = ["M3IndexerGemmSm100", "M3_PAGED_DIRECT_SCORE_NUM_HEADS"]
