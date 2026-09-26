# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""CuTe DSL Q8KV8 paged decode indexer proxy scores for SM100/SM103."""

from __future__ import annotations

import enum

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.typing import Float32, Int32, Int64

import cuda.bindings.driver as cuda


class _NamedBarrier(enum.IntEnum):
    TmemPtr = enum.auto()
    Score = enum.auto()
    Final = enum.auto()


class Q8KV8DecodeIndexerSm100:
    """Compute direct-E4M3 page scores with balanced persistent workers.

    Every index head shares the single K head, so the eight MTP tokens of all
    ``num_heads`` heads form one MMA N tile of ``8 * num_heads`` query columns
    (``token * num_heads + head``) and each K page is loaded once per request.
    """

    supported_num_heads = (1, 2, 4)
    query_length = 8
    page_size = 128
    head_dim = 128
    k_chunk = head_dim
    chunks_per_page = head_dim // k_chunk
    m_tile = page_size
    k_stages = 6
    q_stages = chunks_per_page
    acc_stages = 4
    threads_per_warp = 32
    score_warp_begin = 3
    score_warps = 4
    score_threads = score_warps * threads_per_warp
    threads_per_cta = (score_warp_begin + score_warps) * threads_per_warp
    tmem_copy_threads = 128
    target_ctas_per_sm = 2

    def __init__(self, *, sm_count: int, num_heads: int) -> None:
        if sm_count <= 0:
            raise ValueError("sm_count must be positive")
        if num_heads not in self.supported_num_heads:
            raise ValueError(f"num_heads must be one of {self.supported_num_heads}")
        self.grid_ctas = sm_count * self.target_ctas_per_sm
        self.num_heads = num_heads
        self.num_queries = self.query_length * num_heads
        self.n_tile = self.num_queries

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mPageTable: cute.Tensor,
        mSeqLens: cute.Tensor,
        mOut: cute.Tensor,
        mSchedulerStorage: cute.Tensor,
        stream: cuda.CUstream = None,
    ) -> None:
        if cutlass.const_expr(
            mQ.element_type is not cutlass.Float8E4M3FN
            or mKCache.element_type is not cutlass.Float8E4M3FN
        ):
            raise TypeError("q and k_cache must be Float8E4M3FN")
        if cutlass.const_expr(
            mPageTable.element_type is not Int32 or mSeqLens.element_type is not Int32
        ):
            raise TypeError("page_table and seq_lens must be Int32")
        if cutlass.const_expr(mOut.element_type is not Float32):
            raise TypeError("out must be Float32")

        batch_size = mPageTable.shape[0]
        # vLLM pages may be padded, so tokens and pages keep the cache strides.
        mK_tdp = cute.make_tensor(
            mKCache.iterator,
            cute.select(mKCache.layout, mode=[1, 2, 0]),
        )
        mQ_nkb = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (self.num_queries, self.head_dim, batch_size),
                stride=(self.head_dim, 1, self.num_queries * self.head_dim),
            ),
        )
        mQ_kcr = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (
                    self.k_chunk,
                    self.chunks_per_page,
                    self.num_queries * batch_size,
                ),
                stride=(1, self.k_chunk, self.head_dim),
            ),
        )
        mPageTable_lb = cute.make_tensor(
            mPageTable.iterator,
            cute.select(mPageTable.layout, mode=[1, 0]),
        )
        mOut_pqb = cute.make_tensor(
            mOut.iterator,
            cute.select(mOut.layout, mode=[2, 1, 0]),
        )
        mScheduler = cute.make_tensor(
            cute.recast_ptr(mSchedulerStorage.iterator, dtype=Int32),
            cute.make_layout(batch_size + 1),
        )

        qk_tiler = (self.m_tile, self.n_tile, self.k_chunk)
        cta_group = tcgen05.CtaGroup.ONE
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            mKCache.element_type,
            mQ.element_type,
            utils.LayoutEnum.from_tensor(mK_tdp).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mQ_nkb).mma_major_mode(),
            Float32,
            cta_group,
            qk_tiler[:2],
            tcgen05.OperandSource.SMEM,
        )
        sK_layout = sm100_utils.make_smem_layout_a(
            tiled_mma,
            qk_tiler,
            mKCache.element_type,
            self.k_stages,
        )
        sQ_layout = sm100_utils.make_smem_layout_b(
            tiled_mma,
            qk_tiler,
            mQ.element_type,
            self.q_stages,
        )
        sK_tma_layout = cute.make_composed_layout(
            sK_layout.inner,
            0,
            cute.make_layout(
                (self.page_size, self.k_chunk, self.k_stages),
                stride=(
                    self.k_chunk,
                    1,
                    self.page_size * self.k_chunk,
                ),
            ),
        )
        sQ_tma_layout = cute.make_composed_layout(
            sQ_layout.inner,
            0,
            cute.make_layout(
                (
                    self.k_chunk,
                    self.chunks_per_page,
                    self.num_queries,
                ),
                stride=(
                    1,
                    self.num_queries * self.k_chunk,
                    self.k_chunk,
                ),
            ),
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            tma_load_op,
            mK_tdp,
            sK_tma_layout,
            (self.page_size, self.k_chunk),
        )
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            tma_load_op,
            mQ_kcr,
            sQ_tma_layout,
            (
                self.k_chunk,
                self.chunks_per_page,
                self.num_queries,
            ),
        )

        @cute.struct
        class SharedStorage:
            k_mbar_ptr: cute.struct.MemRange[Int64, self.k_stages * 2]
            q_mbar_ptr: cute.struct.MemRange[Int64, 2]
            acc_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stages * 2]
            tmem_holding_buf: Int32

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_Q,
            tma_tensor_Q,
            mPageTable_lb,
            mSeqLens,
            mOut_pqb,
            mScheduler,
            batch_size,
            sK_layout,
            sK_tma_layout,
            sQ_layout,
            sQ_tma_layout,
        ).launch(
            grid=(self.grid_ctas, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        mK_tdp: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        mQ_kcr: cute.Tensor,
        mPageTable_lb: cute.Tensor,
        mSeqLens: cute.Tensor,
        mOut_pqb: cute.Tensor,
        mScheduler: cute.Tensor,
        batch_size: Int32,
        sK_layout: cute.ComposedLayout,
        sK_tma_layout: cute.ComposedLayout,
        sQ_layout: cute.ComposedLayout,
        sQ_tma_layout: cute.ComposedLayout,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        worker_idx, _, _ = cute.arch.block_idx()

        total_pages = Int32(0)
        if lane_idx == Int32(0):
            total_pages = mScheduler[batch_size]
        total_pages = cute.arch.shuffle_sync(total_pages, 0)
        pages_per_worker = (
            total_pages + Int32(self.grid_ctas - 1)
        ) // Int32(self.grid_ctas)
        global_page_begin = worker_idx * pages_per_worker
        num_pages = Int32(0)
        if global_page_begin < total_pages:
            global_page_end = global_page_begin + pages_per_worker
            if global_page_end > total_pages:  # noqa: PLR1730
                global_page_end = total_pages
            num_pages = global_page_end - global_page_begin

        batch_idx = Int32(0)
        if lane_idx == Int32(0):
            batch_hi = batch_size
            while batch_idx < batch_hi:
                batch_mid = (batch_idx + batch_hi) // Int32(2)
                if mScheduler[batch_mid + 1] <= global_page_begin:
                    batch_idx = batch_mid + Int32(1)
                else:
                    batch_hi = batch_mid
        batch_idx = cute.arch.shuffle_sync(batch_idx, 0)
        logical_page_begin = global_page_begin - mScheduler[batch_idx]

        if warp_idx == Int32(1):
            cpasync.prefetch_descriptor(tma_atom_K)
        elif warp_idx == Int32(2):
            cpasync.prefetch_descriptor(tma_atom_Q)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sK = smem.allocate_tensor(
            element_type=cutlass.Float8E4M3FN,
            layout=sK_layout.outer,
            swizzle=sK_layout.inner,
            byte_alignment=128,
        )
        sK_tma = cute.make_tensor(sK.iterator, sK_tma_layout.outer)
        sQ = smem.allocate_tensor(
            element_type=cutlass.Float8E4M3FN,
            layout=sQ_layout.outer,
            swizzle=sQ_layout.inner,
            byte_alignment=128,
        )
        sQ_tma = cute.make_tensor(sQ.iterator, sQ_tma_layout.outer)
        sPartialMax = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout(
                (2, self.score_warps, self.num_queries),
                stride=(self.score_warps * self.num_queries, self.num_queries, 1),
            ),
            byte_alignment=16,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(_NamedBarrier.TmemPtr),
            num_threads=self.threads_per_cta,
        )
        score_barrier = pipeline.NamedBarrier(
            barrier_id=int(_NamedBarrier.Score),
            num_threads=self.score_threads,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
        )
        tmem.allocate(self.acc_stages * self.num_queries)

        k_producer, k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=self.page_size * self.k_chunk,
            barrier_storage=storage.k_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        q_producer, q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=self.num_queries * self.head_dim,
            barrier_storage=storage.q_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.tmem_copy_threads,
            ),
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        pipeline.pipeline_init_arrive(is_relaxed=True)

        gK = cute.flat_divide(mK_tdp, (self.page_size, self.k_chunk))
        tKsK, tKgK = cpasync.tma_partition(
            tma_atom_K,
            0,
            cute.make_layout(1),
            cute.group_modes(sK_tma, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        gQ = cute.flat_divide(
            mQ_kcr,
            (
                self.k_chunk,
                self.chunks_per_page,
                self.num_queries,
            ),
        )
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ_tma, 0, 3),
            cute.group_modes(gQ, 0, 3),
        )

        tCrK = tiled_mma.make_fragment_A(sK)
        tCrQ = tiled_mma.make_fragment_B(sQ)
        acc_shape = tiled_mma.partition_shape_C((self.m_tile, self.n_tile))
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))

        pipeline.pipeline_init_wait()
        thr_mma = tiled_mma.get_slice(0)
        tmem_ptr = tmem.retrieve_ptr(Float32)
        tCtAcc_staged = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(getattr(tcgen05.Repetition, f"x{self.num_queries}")),
            Float32,
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(
            tmem_load_atom,
            tCtAcc_staged[(None, None, None, 0)],
        )
        copy_tidx = tidx % Int32(self.tmem_copy_threads)
        thr_tmem_copy = tmem_tiled_copy.get_slice(copy_tidx)
        tTR_tAcc_staged = thr_tmem_copy.partition_S(tCtAcc_staged)
        cScores = cute.make_identity_tensor((self.m_tile, self.n_tile))
        tCcScores = thr_mma.partition_C(cScores)
        tTR_cScores = thr_tmem_copy.partition_D(tCcScores)

        if warp_idx == Int32(1):
            current_batch = batch_idx
            logical_page = logical_page_begin
            global_page = global_page_begin
            pages_remaining = num_pages
            while pages_remaining > Int32(0):
                segment_pages = mScheduler[current_batch + 1] - global_page
                if segment_pages > pages_remaining:  # noqa: PLR1730
                    segment_pages = pages_remaining
                for _ in cutlass.range(segment_pages, unroll=1):
                    physical_page = Int32(0)
                    if lane_idx == Int32(0):
                        physical_page = mPageTable_lb[
                            logical_page,
                            current_batch,
                        ]
                    physical_page = cute.arch.shuffle_sync(physical_page, 0)
                    for chunk in cutlass.range_constexpr(self.chunks_per_page):
                        k_empty = k_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_K,
                            tKgK[(None, 0, Int32(chunk), physical_page)],
                            tKsK[(None, k_empty.index)],
                            tma_bar_ptr=k_empty.barrier,
                        )
                    global_page += Int32(1)
                    logical_page += Int32(1)
                pages_remaining -= segment_pages
                while (
                    current_batch < batch_size - Int32(1)
                    and mScheduler[current_batch + 1] <= global_page
                ):
                    current_batch += Int32(1)
                    logical_page = global_page - mScheduler[current_batch]
            k_producer.tail()
        elif warp_idx == Int32(2):
            current_batch = batch_idx
            global_page = global_page_begin
            pages_remaining = num_pages
            while pages_remaining > Int32(0):
                segment_pages = mScheduler[current_batch + 1] - global_page
                if segment_pages > pages_remaining:  # noqa: PLR1730
                    segment_pages = pages_remaining
                q_empty = q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_Q,
                    tQgQ[(None, Int32(0), Int32(0), current_batch)],
                    tQsQ,
                    tma_bar_ptr=q_empty.barrier,
                )
                global_page += segment_pages
                pages_remaining -= segment_pages
                while (
                    current_batch < batch_size - Int32(1)
                    and mScheduler[current_batch + 1] <= global_page
                ):
                    current_batch += Int32(1)
            q_producer.tail()
        elif warp_idx == Int32(0):
            num_k_blocks = cute.size(tCrK, mode=[2])
            current_batch = batch_idx
            global_page = global_page_begin
            pages_remaining = num_pages
            while pages_remaining > Int32(0):
                segment_pages = mScheduler[current_batch + 1] - global_page
                if segment_pages > pages_remaining:  # noqa: PLR1730
                    segment_pages = pages_remaining
                q_full = q_consumer.wait_and_advance()
                for _ in cutlass.range(segment_pages, unroll=1):
                    acc_empty = acc_producer.acquire_and_advance()
                    tCtAcc = tCtAcc_staged[(None, None, None, acc_empty.index)]
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for chunk in cutlass.range_constexpr(self.chunks_per_page):
                        k_full = k_consumer.wait_and_advance()
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrK[
                                    (
                                        None,
                                        None,
                                        k_block_idx,
                                        k_full.index,
                                    )
                                ],
                                tCrQ[
                                    (
                                        None,
                                        None,
                                        k_block_idx,
                                        Int32(chunk),
                                    )
                                ],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        k_full.release()
                    acc_empty.commit()
                q_full.release()
                global_page += segment_pages
                pages_remaining -= segment_pages
                while (
                    current_batch < batch_size - Int32(1)
                    and mScheduler[current_batch + 1] <= global_page
                ):
                    current_batch += Int32(1)
            acc_producer.tail()

        rScores = cute.make_rmem_tensor(tTR_cScores.shape, Float32)
        if warp_idx >= Int32(self.score_warp_begin):
            score_warp_idx = warp_idx - Int32(self.score_warp_begin)
            rScoresFlat = cute.make_tensor(
                rScores.iterator,
                cute.make_layout(self.num_queries),
            )
            current_batch = batch_idx
            logical_page = logical_page_begin
            global_page = global_page_begin
            pages_remaining = num_pages
            local_block = Int32(0)
            if (
                num_pages > Int32(0)
                and score_warp_idx == Int32(0)
                and lane_idx < Int32(self.num_queries)
            ):
                seq_len = mSeqLens[current_batch]
                query_position = (
                    seq_len - Int32(self.query_length) + lane_idx // Int32(self.num_heads)
                )
                local_block = query_position // Int32(self.page_size)
            while pages_remaining > Int32(0):
                segment_pages = mScheduler[current_batch + 1] - global_page
                if segment_pages > pages_remaining:  # noqa: PLR1730
                    segment_pages = pages_remaining
                for _ in cutlass.range(segment_pages, unroll=1):
                    partial_stage = global_page % Int32(2)
                    acc_full = acc_consumer.wait_and_advance()
                    cute.copy(
                        tmem_tiled_copy,
                        tTR_tAcc_staged[
                            (None, None, None, None, acc_full.index)
                        ],
                        rScores,
                    )
                    cute.arch.fence_view_async_tmem_load()
                    for query_idx in cutlass.range_constexpr(self.num_queries):
                        partial_max = cute.arch.warp_redux_sync(
                            rScoresFlat[query_idx],
                            "fmax",
                        )
                        if lane_idx == Int32(0):
                            sPartialMax[
                                partial_stage,
                                score_warp_idx,
                                query_idx,
                            ] = partial_max
                    acc_full.release()
                    score_barrier.arrive_and_wait()

                    if score_warp_idx == Int32(0) and lane_idx < Int32(
                        self.num_queries
                    ):
                        query_idx = lane_idx
                        row_max = -Float32.inf
                        for partial_idx in cutlass.range_constexpr(
                            self.score_warps
                        ):
                            row_max = cute.arch.fmax(
                                row_max,
                                sPartialMax[
                                    partial_stage,
                                    partial_idx,
                                    query_idx,
                                ],
                            )
                        if logical_page < local_block:
                            mOut_pqb[
                                logical_page,
                                query_idx,
                                current_batch,
                            ] = row_max
                    global_page += Int32(1)
                    logical_page += Int32(1)
                pages_remaining -= segment_pages
                while (
                    current_batch < batch_size - Int32(1)
                    and mScheduler[current_batch + 1] <= global_page
                ):
                    current_batch += Int32(1)
                    logical_page = global_page - mScheduler[current_batch]
                if score_warp_idx == Int32(0) and lane_idx < Int32(
                    self.num_queries
                ):
                    seq_len = mSeqLens[current_batch]
                    query_position = (
                        seq_len
                        - Int32(self.query_length)
                        + lane_idx // Int32(self.num_heads)
                    )
                    local_block = query_position // Int32(self.page_size)

        tmem.relinquish_alloc_permit()
        pipeline.sync(barrier_id=int(_NamedBarrier.Final))
        tmem.free(tmem_ptr)


__all__ = ["Q8KV8DecodeIndexerSm100"]
