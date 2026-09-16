# Current scope of this postprocess:
# - Sparse Attention GQA16 dQ postprocess
# - CSR + varlen Q metadata only
# - SM100-only
# - expects the attention kernel to accumulate dQ in an FP32 workspace
#
# Postprocess job: scale mdQaccum by softmax_scale, convert fp32 -> output dtype,
# and scatter/store to natural mdQ layout.
from typing import Callable, Optional, Type

import cutlass
from cutlass import Float32, Int32, Int64, const_expr
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.utils import LayoutEnum
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import cuda.bindings.driver as cuda

from msa_v1._common import copy_utils
from msa_v1._common.cute_dsl_utils import ParamsBase, assume_tensor_aligned
from msa_v1._common.pack_gqa import PackGQA, pack_gqa_layout
from msa_v1._common.seqlen_info import SeqlenInfoQK
from msa_v1._common.tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)
from msa_v1._common.tma_utils import (
    stg128_fake_col_to_real_col,
    stg_64_bf16,
    stg_64_f16,
)


class SparseAttentionBackwardPostprocessSm100:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        tile_m: int = 128,
        num_threads: int = 128,
    ):
        """
        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size (packed: tile_m packs qhead Q rows)
        :type tile_m: int
        Packed mdQaccum uses the delivered varlen layout
        [H_kv, total_q_padded * qhead * D]. The grid iterates packed
        m_blocks × H_kv × B and scatters back to natural mdQ layout.
        """
        if head_dim != 128:
            raise NotImplementedError(
                f"SparseAttentionBackwardPostprocessSm100 currently supports only D=128, got D={head_dim}"
            )
        self.dtype = dtype
        self.tile_m = tile_m
        self.tile_hdim = 128
        self.check_hdim_oob = False
        self.num_threads = num_threads
        self.qhead_per_kvhead = 16
        assert num_threads == 128, "SparseAttentionBackwardPostprocessSm100 is tuned for 128 threads"

    @staticmethod
    def can_implement(dtype, head_dim, tile_m, num_threads) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size
        :type tile_m: int

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim != 128:
            return False
        if num_threads != 128:
            return False
        return True

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        universal_copy_bits = 128
        async_copy_elems_accum = universal_copy_bits // Float32.width
        atom_async_copy_accum = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            Float32,
            num_bits_per_copy=universal_copy_bits,
        )
        # We don't do bound checking for the gmem -> smem load so we just assert here.
        assert (self.tile_m * self.tile_hdim // async_copy_elems_accum) % self.num_threads == 0
        self.g2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            atom_async_copy_accum,
            cute.make_layout(self.num_threads),
            cute.make_layout(async_copy_elems_accum),
        )
        self.dQ_reduce_ncol = 32
        dQaccum_reduce_stage = self.tile_hdim // self.dQ_reduce_ncol
        self.s2r_tiled_copy_dQaccum = copy_utils.tiled_copy_1d(Float32, self.num_threads, 4)
        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.tile_hdim // dQaccum_reduce_stage, dQaccum_reduce_stage)
        )

        num_copy_elems = 128 // self.dtype.width
        threads_per_row = 128 // num_copy_elems
        self.gmem_tiled_copy_dQ = copy_utils.tiled_copy_2d(
            self.dtype, threads_per_row, self.num_threads, num_copy_elems
        )
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: dQ
        # ///////////////////////////////////////////////////////////////////////////////
        self.sdQ_layout = sm100_utils_basic.make_smem_layout_epi(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdim), 1
        )

    @cute.jit
    def __call__(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: cutlass.Float32,
        mCuSeqlensQ: cute.Tensor,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(mdQ.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mdQaccum is not None):
            if const_expr(mdQaccum.element_type not in [cutlass.Float32]):
                raise TypeError("dQaccum tensor must be Float32")

        mdQaccum, mdQ = [assume_tensor_aligned(t) for t in (mdQaccum, mdQ)]
        self._setup_attributes()

        smem_size = max(
            cute.size_in_bytes(cutlass.Float32, self.sdQaccum_layout),
            cute.size_in_bytes(self.dtype, self.sdQ_layout),
        )

        TileScheduler = SingleTileVarlenScheduler
        num_head = Int32(mdQaccum.shape[0])
        num_batch = mCuSeqlensQ.shape[0] - 1
        total_q = mdQ.shape[0] * self.qhead_per_kvhead
        num_block = cute.ceil_div(total_q, self.tile_m)

        tile_sched_args = TileSchedulerArguments(
            num_block=num_block,
            num_head=num_head,
            num_batch=num_batch,
            num_splits=1,
            seqlen_k=0,
            headdim=mdQ.shape[2],
            headdim_v=0,
            total_q=total_q,
            tile_shape_mn=(self.tile_m, 1),
            mCuSeqlensQ=mCuSeqlensQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # grid_dim: (m_block, num_head, batch_size)
        self.kernel(
            mdQaccum,
            mdQ,
            mCuSeqlensQ,
            scale,
            self.sdQaccum_layout,
            self.sdQ_layout,
            self.g2s_tiled_copy_dQaccum,
            self.s2r_tiled_copy_dQaccum,
            self.gmem_tiled_copy_dQ,
            tile_sched_params,
            TileScheduler,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        scale: cutlass.Float32,
        sdQaccum_layout: cute.Layout,
        sdQ_layout: cute.ComposedLayout,
        g2s_tiled_copy_dQaccum: cute.TiledCopy,
        s2r_tiled_copy_dQaccum: cute.TiledCopy,
        gmem_tiled_copy_dQ: cute.TiledCopy,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        sdQaccum = smem.allocate_tensor(cutlass.Float32, sdQaccum_layout, byte_alignment=1024)
        sdQaccum_flat = cute.make_tensor(sdQaccum.iterator, cute.make_layout(cute.size(sdQaccum)))
        sdQ = cute.make_tensor(
            cute.recast_ptr(sdQaccum.iterator, sdQ_layout.inner, dtype=self.dtype),
            sdQ_layout.outer,
        )[None, None, 0]

        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()

        m_block, head_idx, batch_idx, _ = work_tile.tile_idx

        if work_tile.is_valid_tile:
            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # ///////////////////////////////////////////////////////////////////////////////

            seqlen = SeqlenInfoQK.create(
                batch_idx,
                mdQ.shape[1],
                0,
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=None,
                tile_m=self.tile_m,
            )
            padded_offset_q = seqlen.padded_offset_q
            mdQ_batch = cute.domain_offset((seqlen.offset_q, None, None), mdQ)
            mdQ_T = cute.make_tensor(
                mdQ_batch.iterator, cute.select(mdQ_batch.layout, mode=[0, 2, 1])
            )  # (T, D, H_q)
            nheads_kv = mdQ.shape[1] // self.qhead_per_kvhead
            mdQ_packed = pack_gqa_layout(
                mdQ_T, self.qhead_per_kvhead, nheads_kv, head_idx=2
            )
            mdQ_cur = mdQ_packed[None, None, head_idx]
            packed_q_stride = Int64(self.qhead_per_kvhead * self.tile_hdim)
            mdQaccum_cur = cute.domain_offset(
                (Int64(padded_offset_q) * packed_q_stride,), mdQaccum[head_idx, None]
            )

            mdQaccum_cur_ptr = cute.make_ptr(
                dtype=mdQaccum_cur.element_type,
                value=mdQaccum_cur.iterator.toint(),
                mem_space=mdQaccum_cur.iterator.memspace,
                assumed_align=mdQaccum.iterator.alignment,
            )
            mdQaccum_cur = cute.make_tensor(mdQaccum_cur_ptr, mdQaccum_cur.layout)

            gdQaccum = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (m_block,))

            seqlen_q = seqlen.seqlen_q
            # Step 1: load dQaccum from gmem to smem
            g2s_thr_copy_dQaccum = g2s_tiled_copy_dQaccum.get_slice(tidx)
            tdQgdQaccum = g2s_thr_copy_dQaccum.partition_S(gdQaccum)
            tdQsdQaccumg2s = g2s_thr_copy_dQaccum.partition_D(sdQaccum_flat)
            cute.copy(g2s_tiled_copy_dQaccum, tdQgdQaccum, tdQsdQaccumg2s)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            # Step 2: load dQ from smem to rmem
            tile_shape = (self.tile_m, self.tile_hdim)
            # mdQaccum is token-major and stage-major within each token.
            # Each token-stage uses the internal R2S layout
            # [col_quad, h_in_kv, lane4]. Decode it back into the
            # row-major packed-tile fragment expected by the rest of
            # postprocess.
            q_per_tile = self.tile_m // self.qhead_per_kvhead
            num_stages = self.tile_hdim // self.dQ_reduce_ncol
            token_elems = self.qhead_per_kvhead * self.tile_hdim
            stage_elems = self.qhead_per_kvhead * self.dQ_reduce_ncol
            hidden_vec_elems = 128 // Float32.width
            hidden_chunks_per_stage = self.dQ_reduce_ncol // hidden_vec_elems
            hidden_chunk_elems = self.qhead_per_kvhead * hidden_vec_elems
            sdQaccum_sp = cute.make_tensor(
                sdQaccum.iterator,
                cute.make_layout(
                    (
                        q_per_tile,
                        num_stages,
                        hidden_chunks_per_stage,
                        self.qhead_per_kvhead,
                        hidden_vec_elems,
                    ),
                    stride=(
                        token_elems,
                        stage_elems,
                        hidden_chunk_elems,
                        hidden_vec_elems,
                        1,
                    ),
                ),
            )
            s2r_atom_sp = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
            )
            s2r_tiled_sp = cute.make_tiled_copy_tv(
                s2r_atom_sp,
                cute.make_layout((self.num_threads, 1)),
                cute.make_layout((1, 128 // Float32.width)),
            )
            thr_s2r_sp = s2r_tiled_sp.get_slice(tidx)
            tdQcdQ_sp = thr_s2r_sp.partition_S(cute.make_identity_tensor(tile_shape))
            acc = cute.make_rmem_tensor(tdQcdQ_sp.shape, Float32)
            for i in cutlass.range(cute.size(tdQcdQ_sp), unroll_full=True):
                coord = tdQcdQ_sp[i]
                row = coord[0]
                col = coord[1]
                tok = row // cutlass.Int32(self.qhead_per_kvhead)
                h_in_kv = row % cutlass.Int32(self.qhead_per_kvhead)
                stage = col // cutlass.Int32(self.dQ_reduce_ncol)
                col_in_stage = col % cutlass.Int32(self.dQ_reduce_ncol)
                col_quad = col_in_stage // cutlass.Int32(hidden_vec_elems)
                lane_in_quad = col_in_stage % cutlass.Int32(hidden_vec_elems)
                acc[i] = sdQaccum_sp[(tok, stage, col_quad, h_in_kv, lane_in_quad)]
            rdQ = cute.make_rmem_tensor_like(acc, self.dtype)
            rdQ.store((acc.load() * scale).to(self.dtype))

            # Step 3: copy dQ from register to smem
            cute.arch.barrier()  # make sure all threads have finished loading dQaccum
            thr_layout_r2s_dQ = cute.make_layout((self.num_threads, 1))
            val_layout_r2s_dQ = cute.make_layout((1, 128 // self.dtype.width))
            copy_atom_r2s_dQ = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype,
                num_bits_per_copy=128,
            )
            tiled_copy_r2s_dQ = cute.make_tiled_copy_tv(
                copy_atom_r2s_dQ, thr_layout_r2s_dQ, val_layout_r2s_dQ
            )
            thr_copy_r2s_dQ = tiled_copy_r2s_dQ.get_slice(tidx)
            cdQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
            taccdQcdQ_shape = thr_copy_r2s_dQ.partition_S(cdQ).shape
            taccdQrdQ = cute.make_tensor(rdQ.iterator, taccdQcdQ_shape)
            taccdQsdQ = thr_copy_r2s_dQ.partition_D(sdQ)
            cute.copy(thr_copy_r2s_dQ, taccdQrdQ, taccdQsdQ)

            # Step 4: Copy dQ from smem to register to prepare for coalesced write to gmem
            cute.arch.barrier()  # make sure all smem stores are done
            gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_slice(tidx)
            tdQsdQ = gmem_thr_copy_dQ.partition_D(sdQ)
            tdQrdQ = cute.make_rmem_tensor_like(tdQsdQ, self.dtype)
            # TODO: check OOB when reading from smem if kBlockM isn't evenly tiled
            cute.autovec_copy(tdQsdQ, tdQrdQ)

            # Step 5: copy dQ from register to gmem
            packer = PackGQA(
                m_block_size=self.tile_m,
                head_dim_padded=self.tile_hdim,
                check_hdim_oob=self.check_hdim_oob,
                qhead_per_kvhead=self.qhead_per_kvhead,
            )
            packer.store_O(mdQ_cur, tdQrdQ, gmem_tiled_copy_dQ, tidx, m_block, seqlen_q)


class SparseAttentionBackwardPostprocessAtomicDqSm100:
    """Postprocess the token-major FP32 atomic dQ workspace.

    Input mdQaccum is [total_q, H_q, D_padded] fp32 in STG128 fake-col layout.
    Output mdQ is [total_q, H_q, D] in real layout.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        head_dim: int,
        tile_m: int = 128,
        num_threads: int = 128,
    ):
        if head_dim != 128:
            raise NotImplementedError(
                f"SparseAttentionBackwardPostprocessAtomicDqSm100 currently supports only D=128, got D={head_dim}"
            )
        self.dtype = dtype
        self.head_dim = 128
        self.tile_m = tile_m
        self.num_threads = num_threads
        self.tile_hdim = 128
        self.rows_per_cta = min(tile_m, num_threads // 4)
        assert dtype in (cutlass.Float16, cutlass.BFloat16)
        assert num_threads % 4 == 0, "atomic dQ postprocess requires 4-thread row groups"

    @cute.jit
    def __call__(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: Float32,
        stream: cuda.CUstream = None,
    ):
        if const_expr(mdQaccum.element_type != Float32):
            raise TypeError("mdQaccum must be Float32")
        if const_expr(mdQ.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("mdQ must be Float16 or BFloat16")

        mdQaccum, mdQ = [assume_tensor_aligned(t) for t in (mdQaccum, mdQ)]
        assert cute.rank(mdQ.shape) == 3
        total_q = cute.size(mdQ.shape[0])
        nheads = cute.size(mdQ.shape[1])
        grid = (cute.ceil_div(total_q, self.rows_per_cta), nheads, 1)

        self.kernel(mdQaccum, mdQ, scale).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            smem=self.rows_per_cta * self.tile_hdim * Float32.width // 8,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, head_idx, batch_idx = cute.arch.block_idx()
        row_group = tidx // Int32(4)
        lane_in_row = tidx % Int32(4)
        q_idx = m_block * Int32(self.rows_per_cta) + row_group
        del batch_idx
        seq_len_q = cute.size(mdQ.shape[0])
        valid_q = q_idx < seq_len_q
        row_base = q_idx
        total_hq = Int32(cute.size(mdQ.shape[1]))
        sdQperm_ptr = cute.arch.get_dyn_smem(Float32)
        sdQperm = cute.make_tensor(
            sdQperm_ptr,
            cute.make_layout(
                (self.rows_per_cta, self.tile_hdim),
                stride=(self.tile_hdim, 1),
            ),
        )
        if valid_q:
            base_off = (
                Int64(row_base) * Int64(total_hq) + Int64(head_idx)
            ) * Int64(self.tile_hdim)
            src_memspace = mdQaccum.iterator.memspace
            for fake_group in cutlass.range_constexpr(self.tile_hdim // 16):
                fake_col = Int32(fake_group * 16) + lane_in_row * Int32(4)
                src_ptr = cute.make_ptr(
                    Float32,
                    mdQaccum.iterator.toint() + (base_off + Int64(fake_col)) * Int64(4),
                    mem_space=src_memspace,
                    assumed_align=16,
                )
                src = cute.make_tensor(src_ptr, cute.make_layout((4,), stride=(1,)))
                vals = src.load() * scale
                for v in cutlass.range_constexpr(4):
                    real_col = stg128_fake_col_to_real_col(fake_col + Int32(v))
                    sdQperm[row_group, real_col] = vals[v]

        cute.arch.sync_threads()

        if valid_q:
            base_off_out = (row_base * total_hq + head_idx) * Int32(self.head_dim)
            dst_memspace = mdQ.iterator.memspace
            for real_group in cutlass.range_constexpr(self.tile_hdim // 16):
                real_col = Int32(real_group * 16) + lane_in_row * Int32(4)
                if real_col < Int32(self.head_dim):
                    dst_ptr = cute.make_ptr(
                        mdQ.element_type,
                        mdQ.iterator.toint()
                        + (base_off_out + real_col) * Int32(mdQ.element_type.width // 8),
                        mem_space=dst_memspace,
                        assumed_align=8,
                    )
                    if const_expr(self.dtype == cutlass.BFloat16):
                        stg_64_bf16(
                            dst_ptr,
                            sdQperm[row_group, real_col + Int32(0)],
                            sdQperm[row_group, real_col + Int32(1)],
                            sdQperm[row_group, real_col + Int32(2)],
                            sdQperm[row_group, real_col + Int32(3)],
                        )
                    else:
                        stg_64_f16(
                            dst_ptr,
                            sdQperm[row_group, real_col + Int32(0)],
                            sdQperm[row_group, real_col + Int32(1)],
                            sdQperm[row_group, real_col + Int32(2)],
                            sdQperm[row_group, real_col + Int32(3)],
                        )


class SparseAttentionBackwardDkvPostprocessSm100:
    """Convert varlen fp32 dK/dV accumulators to the final token-major tensor."""

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        tile_n: int = 128,
        num_threads: int = 128,
    ):
        if head_dim != 128:
            raise NotImplementedError(
                "SparseAttentionBackwardDkvPostprocessSm100 currently supports only D=128, "
                f"got D={head_dim}"
            )
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            raise TypeError("Only Float16 or BFloat16 output is supported")
        if num_threads != 128:
            raise ValueError("SparseAttentionBackwardDkvPostprocessSm100 expects 128 threads")
        self.dtype = dtype
        self.head_dim = head_dim
        self.tile_n = tile_n
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mdKVaccum: cute.Tensor,
        mdKV: cute.Tensor,
        mDkvOwnerCounts: cute.Tensor,
        scale: Float32,
        mCuSeqlensK: cute.Tensor,
        mFragmentIndices: Optional[cute.Tensor],
        max_seqlen_k: Int32,
        stream: cuda.CUstream = None,
    ):
        if const_expr(mdKVaccum.element_type != cutlass.Float32):
            raise TypeError("dKV accum tensor must be Float32")
        if const_expr(mdKV.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 output is supported")
        mdKVaccum, mdKV = [assume_tensor_aligned(t) for t in (mdKVaccum, mdKV)]

        TileScheduler = SingleTileScheduler
        total_kv = mdKV.shape[0]
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(max_seqlen_k, self.tile_n),
            num_head=mdKV.shape[1],
            num_batch=mCuSeqlensK.shape[0] - 1,
            num_splits=1,
            seqlen_k=total_kv,
            headdim=mdKV.shape[2],
            headdim_v=0,
            total_q=total_kv,
            tile_shape_mn=(self.tile_n, 1),
            qhead_per_kvhead_packgqa=1,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid = TileScheduler.get_grid_shape(tile_sched_params)

        self.kernel(
            mdKVaccum,
            mdKV,
            mDkvOwnerCounts,
            scale,
            mCuSeqlensK,
            mFragmentIndices,
            max_seqlen_k,
            tile_sched_params,
            TileScheduler,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mdKVaccum: cute.Tensor,
        mdKV: cute.Tensor,
        mDkvOwnerCounts: cute.Tensor,
        scale: Float32,
        mCuSeqlensK: cute.Tensor,
        mFragmentIndices: Optional[cute.Tensor],
        max_seqlen_k: Int32,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx

        if work_tile.is_valid_tile:
            seqlen = SeqlenInfoQK.create(
                batch_idx,
                0,
                mdKV.shape[0],
                mCuSeqlensQ=None,
                mCuSeqlensK=mCuSeqlensK,
                mFragmentIndices=mFragmentIndices,
                tile_n=self.tile_n,
            )
            is_group_end = True
            if const_expr(mFragmentIndices is not None):
                batch = mFragmentIndices.shape[0]
                if batch_idx + Int32(1) < batch:
                    is_group_end = (
                        mFragmentIndices[batch_idx + Int32(1)]
                        != mFragmentIndices[batch_idx]
                    )
            seqlen_k = seqlen.seqlen_k if is_group_end else Int32(0)
            rows_left = seqlen_k - m_block * Int32(self.tile_n)
            valid_rows = cutlass.min(cutlass.max(rows_left, Int32(0)), Int32(self.tile_n))
            lanes_per_row = Int32(self.head_dim // 4)
            row_lane = tidx // lanes_per_row
            col = (tidx - row_lane * lanes_per_row) * Int32(4)
            # Byte-offset arithmetic must be Int64: at customer scale
            # (batch=4 * seqlen_k=512K, head_kv=4, head_dim=128) the address
            # offsets reach ~2^30 elements, and multiplying by the element
            # byte stride (2 for bf16, 4 for fp32) overflows Int32. Mirrors
            # the fix already applied to dQ postprocess in e51cf55.
            head_stride_accum = Int64(cute.size(mdKVaccum.shape[1]))
            total_heads = Int64(cute.size(mdKV.shape[1]))
            total_k = Int32(cute.size(mdKV.shape[0]))
            batch_count = mCuSeqlensK.shape[0] - Int32(1)
            accum_memspace = mdKVaccum.iterator.memspace
            dst_memspace = mdKV.iterator.memspace
            dst_elem_bytes = Int64(mdKV.element_type.width // 8)
            physical_block = (
                seqlen.padded_offset_k // Int32(self.tile_n) + m_block
            )
            # The grid uses the batch-wide maximum sequence length. Shorter
            # documents therefore receive empty tail tiles whose physical
            # block coordinate can be past the owner table. Do not read owner
            # metadata until this tile is known to contain logical KV rows.
            owner_count = Int32(0)
            if valid_rows > Int32(0):
                owner_count = mDkvOwnerCounts[head_idx, physical_block]

            if owner_count != Int32(1):
                for row_group in cutlass.range_constexpr(self.tile_n // 4):
                    row = Int32(row_group * 4) + row_lane
                    k_idx = m_block * Int32(self.tile_n) + row
                    dst_row = seqlen.offset_k + k_idx
                    dst_off = (
                        (Int64(dst_row) * total_heads + Int64(head_idx))
                        * Int64(self.head_dim)
                        + Int64(col)
                    )
                    if row < valid_rows:
                        vals = cute.make_rmem_tensor((4,), Float32)
                        vals.fill(0.0)
                        if owner_count > Int32(1):
                            accum_off = (
                                Int64(head_idx) * head_stride_accum
                                + Int64(seqlen.padded_offset_k + k_idx)
                                * Int64(self.head_dim)
                                + Int64(col)
                            )
                            src_ptr = cute.make_ptr(
                                Float32,
                                mdKVaccum.iterator.toint() + accum_off * Int64(4),
                                mem_space=accum_memspace,
                                assumed_align=16,
                            )
                            src = cute.make_tensor(
                                src_ptr, cute.make_layout((4,), stride=(1,))
                            )
                            vals.store(src.load() * scale)
                        dst_ptr = cute.make_ptr(
                            mdKV.element_type,
                            mdKV.iterator.toint() + dst_off * dst_elem_bytes,
                            mem_space=dst_memspace,
                            assumed_align=8,
                        )
                        if const_expr(self.dtype == cutlass.BFloat16):
                            stg_64_bf16(dst_ptr, vals[0], vals[1], vals[2], vals[3])
                        else:
                            stg_64_f16(dst_ptr, vals[0], vals[1], vals[2], vals[3])
                    elif batch_idx == batch_count - Int32(1) and dst_row < total_k:
                        dst_ptr = cute.make_ptr(
                            mdKV.element_type,
                            mdKV.iterator.toint() + dst_off * dst_elem_bytes,
                            mem_space=dst_memspace,
                            assumed_align=8,
                        )
                        zero = Float32(0.0)
                        if const_expr(self.dtype == cutlass.BFloat16):
                            stg_64_bf16(dst_ptr, zero, zero, zero, zero)
                        else:
                            stg_64_f16(dst_ptr, zero, zero, zero, zero)
