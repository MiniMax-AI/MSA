# Current scope:
# - Sparse Attention backward preprocess
# - CSR + varlen Q metadata only
# - SM100 delivery path
# - computes:
#     mdPsum   = rowsum(dO * O), unless logical-P backward replaces it
#     mLSElog2 = softmax_lse * log2(e)
#     mdQaccum = 0

import math
import operator
from typing import Type

import cutlass
from cutlass import Float32, Int32, Int64, const_expr
import cutlass.cute as cute
import cuda.bindings.driver as cuda

from msa_v1._common import copy_utils, utils
from msa_v1._common.cute_dsl_utils import ParamsBase, assume_tensor_aligned
from msa_v1._common.seqlen_info import SeqlenInfo
from msa_v1._common.tile_scheduler import (
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)


class SparseAttentionBackwardPreprocessSm100:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        *,
        tile_m: int = 128,
        num_threads: int = 256,
        use_atomic_dqaccum: bool = False,
        compute_dpsum: bool = True,
    ):
        if head_dim != 128:
            raise NotImplementedError(
                f"SparseAttentionBackwardPreprocessSm100 currently supports only D=128, got D={head_dim}"
            )
        self.dtype = dtype
        self.head_dim = 128
        self.qhead_per_kvhead = 16
        self.tile_m = tile_m
        self.num_threads = num_threads
        self.use_atomic_dqaccum = use_atomic_dqaccum
        self.compute_dpsum = compute_dpsum
        self.head_dim_atomic_padded = 128

    @staticmethod
    def can_implement(dtype, head_dim, tile_m, num_threads) -> bool:
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim != 128:
            return False
        if num_threads % 32 != 0:
            return False
        if num_threads < tile_m:
            return False
        return True

    def _setup_attributes(self):
        self.gmem_tiled_copy_O = copy_utils.tiled_copy_2d(
            self.dtype, self.head_dim, self.num_threads
        )
        self.gmem_tiled_copy_dQaccum_packed = copy_utils.tiled_copy_1d(
            Float32, self.num_threads, 1
        )
        self.gmem_tiled_copy_dQaccum_atomic = copy_utils.tiled_copy_2d(
            Float32, self.head_dim_atomic_padded, self.num_threads
        )

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,        # [total_q, H_q, D]
        mdO: cute.Tensor,       # [total_q, H_q, D]
        mPdPsum: cute.Tensor,   # [total_q_padded, H_q] fp32
        mLSE: cute.Tensor,      # [total_q, H_q] fp32
        mLSElog2: cute.Tensor,  # [total_q_padded, H_q] fp32
        mdQaccum: cute.Tensor,  # fp32 dQ workspace in the active varlen layout
        mCuSeqlensQ: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        if const_expr(not (mO.element_type == mdO.element_type)):
            raise TypeError("mO and mdO must have the same dtype")
        if const_expr(mO.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16/BFloat16 O tensors are supported")
        if const_expr(mPdPsum.element_type != Float32):
            raise TypeError("mPdPsum must be Float32")
        if const_expr(mLSE.element_type != Float32):
            raise TypeError("mLSE must be Float32")
        if const_expr(mLSElog2.element_type != Float32):
            raise TypeError("mLSElog2 must be Float32")
        if const_expr(mdQaccum.element_type != Float32):
            raise TypeError("mdQaccum must be Float32")

        mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum = [
            assume_tensor_aligned(t) for t in (mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum)
        ]
        self._setup_attributes()
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mO.shape[0], self.tile_m),
            num_head=mO.shape[1],
            num_batch=mCuSeqlensQ.shape[0] - 1,
            num_splits=1,
            seqlen_k=0,
            headdim=0,
            headdim_v=mO.shape[2],
            total_q=mO.shape[0],
            tile_shape_mn=(self.tile_m, 1),
            mCuSeqlensQ=mCuSeqlensQ,
        )
        tile_sched_params = SingleTileVarlenScheduler.to_underlying_arguments(tile_sched_args)
        grid = SingleTileVarlenScheduler.get_grid_shape(tile_sched_params)
        self.kernel_varlen(
            mO,
            mdO,
            mPdPsum,
            mLSE,
            mLSElog2,
            mdQaccum,
            mCuSeqlensQ,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_dQaccum_packed,
            self.gmem_tiled_copy_dQaccum_atomic,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel_varlen(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mPdPsum: cute.Tensor,
        mLSE: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdQaccum: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_dQaccum_packed: cute.TiledCopy,
        gmem_tiled_copy_dQaccum_atomic: cute.TiledCopy,
        tile_sched_params: ParamsBase,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tile_scheduler = SingleTileVarlenScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx

        if work_tile.is_valid_tile:
            seqlen = SeqlenInfo.create(
                batch_idx,
                mO.shape[0],
                cu_seqlens=mCuSeqlensQ,
                tile=self.tile_m,
            )
            mLSE_cur = cute.domain_offset((seqlen.offset,), mLSE[None, head_idx])
            mLSElog2_cur = cute.domain_offset((seqlen.offset_padded,), mLSElog2[None, head_idx])
            seqlen_q = seqlen.seqlen
            seqlen_limit = seqlen_q - m_block * self.tile_m
            if const_expr(self.compute_dpsum):
                mO_cur = seqlen.offset_batch(
                    mO, batch_idx, dim=0
                )[None, head_idx, None]
                mdO_cur = seqlen.offset_batch(
                    mdO, batch_idx, dim=0
                )[None, head_idx, None]
                mPdPsum_cur = cute.domain_offset(
                    (seqlen.offset_padded,), mPdPsum[None, head_idx]
                )

                blk_shape = (self.tile_m, self.head_dim)
                gO = cute.local_tile(mO_cur, blk_shape, (m_block, 0))
                gdO = cute.local_tile(mdO_cur, blk_shape, (m_block, 0))
                gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                tOgO = gmem_thr_copy_O.partition_S(gO)
                tOgdO = gmem_thr_copy_O.partition_S(gdO)
                cO = cute.make_identity_tensor(blk_shape)
                tOcO = gmem_thr_copy_O.partition_S(cO)
                t0OcO = gmem_thr_copy_O.get_slice(0).partition_S(cO)

                tOrO = cute.make_rmem_tensor_like(tOgO)
                tOrdO = cute.make_rmem_tensor_like(tOgdO)
                tOrO.fill(0.0)
                tOrdO.fill(0.0)

                for m in cutlass.range(
                    cute.size(tOrO.shape[1]), unroll_full=True
                ):
                    if t0OcO[0, m, 0][0] < seqlen_limit - tOcO[0][0]:
                        cute.copy(
                            gmem_tiled_copy_O,
                            tOgO[None, m, None],
                            tOrO[None, m, None],
                        )
                        cute.copy(
                            gmem_tiled_copy_O,
                            tOgdO[None, m, None],
                            tOrdO[None, m, None],
                        )

                pdpsum = (
                    tOrO.load().to(Float32) * tOrdO.load().to(Float32)
                ).reduce(
                    cute.ReductionOp.ADD,
                    init_val=0.0,
                    reduction_profile=(0, None, 1),
                )
                threads_per_row = (
                    gmem_tiled_copy_O.layout_src_tv_tiled[0].shape[0]
                )
                assert cute.arch.WARP_SIZE % threads_per_row == 0
                pdpsum = utils.warp_reduce(
                    pdpsum, operator.add, width=threads_per_row
                )
                PdP_sum = cute.make_rmem_tensor(
                    cute.size(tOrO, mode=[1]), Float32
                )
                PdP_sum.store(pdpsum)

                gPdPsum = cute.local_tile(
                    mPdPsum_cur, (self.tile_m,), (m_block,)
                )
                if tOcO[0, 0, 0][1] == 0:
                    for m in cutlass.range(
                        cute.size(PdP_sum), unroll_full=True
                    ):
                        row = tOcO[0, m, 0][0]
                        gPdPsum[row] = (
                            PdP_sum[m]
                            if row < seqlen_limit
                            else Float32(0.0)
                        )

            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
            gLSElog2 = cute.local_tile(mLSElog2_cur, (self.tile_m,), (m_block,))
            lse = -Float32.inf
            if tidx < seqlen_limit:
                lse = gLSE[tidx]
            LOG2_E = math.log2(math.e)
            if tidx < cute.round_up(seqlen_q, self.tile_m) - m_block * self.tile_m:
                gLSElog2[tidx] = lse * LOG2_E if lse != -Float32.inf else 0.0

            if const_expr(self.use_atomic_dqaccum):
                mdQaccum_cur = cute.domain_offset(
                    (seqlen.offset, 0), mdQaccum[None, head_idx, None]
                )
                gdQaccum = cute.local_tile(
                    mdQaccum_cur,
                    (self.tile_m, self.head_dim_atomic_padded),
                    (m_block, 0),
                )
                gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum_atomic.get_slice(tidx)
                tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)
                cdQ = cute.make_identity_tensor((self.tile_m, self.head_dim_atomic_padded))
                tdQc = gmem_thr_copy_dQaccum.partition_S(cdQ)
                t0dQc = gmem_tiled_copy_dQaccum_atomic.get_slice(0).partition_S(cdQ)
                zero = cute.make_rmem_tensor_like(tdQgdQaccum)
                zero.fill(0.0)
                for m in cutlass.range(cute.size(zero.shape[1]), unroll_full=True):
                    if t0dQc[0, m, 0][0] < seqlen_limit - tdQc[0][0]:
                        cute.copy(
                            gmem_tiled_copy_dQaccum_atomic,
                            zero[None, m, None],
                            tdQgdQaccum[None, m, None],
                        )
            else:
                head_kv_idx = head_idx // Int32(self.qhead_per_kvhead)
                local_head_idx = head_idx - head_kv_idx * Int32(self.qhead_per_kvhead)
                # mdQaccum is a flat fp32 workspace. For GQA16 varlen we
                # zero the full padded batch region, not just the first
                # token-major slice. Partition the batch chunk head-major so
                # every packed row in the active padded region is covered.
                packed_offset = (
                    Int64(seqlen.offset_padded) * Int64(self.qhead_per_kvhead)
                    + Int64(local_head_idx)
                    * Int64(cute.round_up(seqlen_q, self.tile_m))
                    + Int64(m_block) * Int64(self.tile_m)
                ) * Int64(self.head_dim)
                mdQaccum_cur = cute.domain_offset((packed_offset,), mdQaccum[head_kv_idx, None])
                gdQaccum = cute.local_tile(mdQaccum_cur, (self.tile_m * self.head_dim,), (0,))
                gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum_packed.get_slice(tidx)
                tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)
                zero = cute.make_rmem_tensor_like(tdQgdQaccum)
                zero.fill(0.0)
                cute.copy(gmem_tiled_copy_dQaccum_packed, zero, tdQgdQaccum)


class SparseAttentionBackwardDkvSplitZeroSm100:
    """Clear only FP32 dK/dV accumulator tiles with multiple CTA owners."""

    def __init__(
        self,
        head_dim: int = 128,
        tile_n: int = 128,
        num_threads: int = 256,
    ):
        if head_dim != 128 or tile_n != 128:
            raise NotImplementedError(
                "Selective dKV zero currently requires D=128 and tile_n=128"
            )
        if num_threads != 256:
            raise ValueError("Selective dKV zero expects 256 threads")
        self.head_dim = head_dim
        self.tile_n = tile_n
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mdKaccum: cute.Tensor,
        mdVaccum: cute.Tensor,
        mDkvSplitIndices: cute.Tensor,
        mDkvSplitCount: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        if const_expr(
            mdKaccum.element_type != Float32 or mdVaccum.element_type != Float32
        ):
            raise TypeError("dK/dV accumulators must be Float32")
        self.kernel(
            assume_tensor_aligned(mdKaccum),
            assume_tensor_aligned(mdVaccum),
            mDkvSplitIndices,
            mDkvSplitCount,
        ).launch(
            grid=(mDkvSplitIndices.shape[0], 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mdKaccum: cute.Tensor,
        mdVaccum: cute.Tensor,
        mDkvSplitIndices: cute.Tensor,
        mDkvSplitCount: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0]
        split_idx = cute.arch.block_idx()[0]
        if split_idx < mDkvSplitCount[Int32(0)]:
            tile_elements = Int32(self.tile_n * self.head_dim)
            head_stride = Int64(cute.size(mdKaccum.shape[1]))
            padded_kv_blocks = Int32(head_stride // Int64(self.tile_n * self.head_dim))
            flat_block = mDkvSplitIndices[split_idx]
            head_idx = flat_block // padded_kv_blocks
            block_idx = flat_block - head_idx * padded_kv_blocks
            tile_base = (
                Int64(head_idx) * head_stride
                + Int64(block_idx) * Int64(self.tile_n * self.head_dim)
            )
            zero = Float32(0.0)
            vector_idx = tidx
            while vector_idx < tile_elements // Int32(4):
                elem_offset = tile_base + Int64(vector_idx * Int32(4))
                dK_ptr = cute.make_ptr(
                    Float32,
                    mdKaccum.iterator.toint() + elem_offset * Int64(4),
                    mem_space=mdKaccum.iterator.memspace,
                    assumed_align=16,
                )
                dV_ptr = cute.make_ptr(
                    Float32,
                    mdVaccum.iterator.toint() + elem_offset * Int64(4),
                    mem_space=mdVaccum.iterator.memspace,
                    assumed_align=16,
                )
                copy_utils.stg_128(dK_ptr, zero, zero, zero, zero)
                copy_utils.stg_128(dV_ptr, zero, zero, zero, zero)
                vector_idx += Int32(self.num_threads)
