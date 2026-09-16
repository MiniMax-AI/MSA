# Current scope of this kernel:
# - SM100 backward attention kernel only.
# - delivery contract: head_dim=128 and qhead_per_kvhead=16 only.
# - single-CTA only.
# - non-local path with opt-in deterministic FP32 reductions.
# - k2q CSR sparse-only (`pack_gqa=True`).
# - sparse dQ uses TMA reduce-store.
#
# Explicitly unsupported in this cleaned branch:
# - multi-CTA variants
#
import math
from functools import partial
from typing import Callable, Optional

import cutlass
from cutlass import Float32, Int32, Int64, const_expr
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.pipeline import PipelineAsync
from cutlass.utils import LayoutEnum
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import cuda.bindings.driver as cuda

from msa_v1._common import activation, copy_utils, layout_utils, pipeline, utils
from msa_v1._common.barrier import ld_acquire, red_release
from msa_v1._common.blackwell_helpers import gemm_ptx_w_idx, gemm_w_idx
from msa_v1._common.block_info import BlockInfo
from msa_v1._common.cute_dsl_utils import (
    assume_tensor_aligned,
    exit_thread_if,
)
from msa_v1._common.mask import AttentionMask
from msa_v1._common.named_barrier import NamedBarrierBwdSm100
from msa_v1._common.pack_gqa import pack_gqa_layout
from msa_v1._common.seqlen_info import SeqlenInfoQK
from msa_v1._common.tile_scheduler import (
    SingleTileScheduler,
    TileSchedulerArguments,
)
from msa_v1.attention.bwd import qat


class SparseAttentionBackwardSm100(qat.SparseAttentionQatBackwardMixin):
    arch = 100

    def __init__(
        self,
        head_dim: int,
        qhead_per_kvhead: cutlass.Constexpr[int] = 16,
        tile_m: int = 128,
        tile_n: int = 128,
        pack_gqa: bool = False,
        is_causal: bool = False,
        use_prepare_scheduler: bool = True,
        sparse_attn_p_mode: bool = False,
        deterministic: bool = False,
    ):
        if head_dim != 128:
            raise NotImplementedError(
                f"SparseAttentionBackwardSm100 currently supports only D=128, got D={head_dim}"
            )
        if qhead_per_kvhead != 16:
            raise NotImplementedError(
                "SparseAttentionBackwardSm100 supports only qhead_per_kvhead=16"
            )
        self.tile_hdim = 128
        self.tile_hdimv = 128
        self.check_hdim_oob = False

        self.tile_m = tile_m
        self.tile_n = tile_n
        if sparse_attn_p_mode and (tile_m != 128 or tile_n != 128):
            raise NotImplementedError(
                "BF16 probability QAT requires a 128x128 backward score tile"
            )

        # Single-CTA sparse path.
        self.cta_group_size = 1
        self.is_persistent = False
        self.is_causal = is_causal
        self.pack_gqa = pack_gqa  # compound mode-0 Q packing for GQA16
        self.sparse_attn_p_mode = sparse_attn_p_mode
        self.deterministic = deterministic
        self.stage_p_stats = sparse_attn_p_mode
        if not use_prepare_scheduler:
            raise ValueError("SparseAttentionBackwardSm100 requires prepare scheduler")
        self.use_prepare_scheduler = True

        # CTA tiler (cta_group_size=1)
        self.cta_tiler = (tile_n, tile_m, self.tile_hdim)
        # S = K @ Q.T
        self.mma_tiler_kq = (tile_n, tile_m, self.tile_hdim)
        # dP = V @ dO.T
        self.mma_tiler_vdo = (tile_n, tile_m, self.tile_hdimv)
        # dV = P.T @ dO
        self.mma_tiler_pdo = (tile_n, self.tile_hdimv, tile_m)
        # dK = dS.T @ Q
        self.mma_tiler_dsq = (tile_n, self.tile_hdim, tile_m)
        # dQ = dS @ K
        self.mma_tiler_dsk = (tile_m, self.tile_hdim, tile_n)

        self.acc_dtype = Float32
        self.cluster_shape_mn = (1, 1)
        self.qhead_per_kvhead = qhead_per_kvhead

        # vec_size=4 (score_mod/aux_tensors stripped)
        self.vec_size: cutlass.Constexpr = 4

        # Performance configuration; these choices do not change the math.
        self.shuffle_LSE = False
        self.shuffle_dPsum = False
        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        # Warp 14 is repurposed as a second load producer for dO.
        self.load_do_warp_id = 14
        self.relay_warp_id = self.load_do_warp_id
        self.empty_warp_id = 15

        # 16 warps -> 512 threads
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.reduce_warp_ids,
                *self.compute_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.relay_warp_id,
                self.empty_warp_id,
            )
        )
        # NamedBarrier
        self.compute_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.Compute),
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )
        self.reduce_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.dQaccReduce),
            num_threads=len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )
        # Sparse two-warp Q/dO load: warp 13 produces shared metadata, warp 14
        # consumes it. Keep this to 64 threads so the rest of the CTA can
        # proceed independently.
        self.load_qdo_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.TmemPtr) + 1,
            num_threads=2 * cute.arch.WARP_SIZE,
        )
        # TMEM setup
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        self.tmem_S_offset = 0
        self.tmem_P_offset = 0  # overlap with S
        self.tmem_dV_offset = self.tmem_S_offset + self.tile_n
        self.tmem_dP_offset = self.tmem_dV_offset + self.tile_hdimv
        self.tmem_dQ_offset = self.tmem_dP_offset
        self.tmem_dK_offset = self.tmem_dP_offset + self.tile_m
        self.tmem_dS_offset = self.tmem_dP_offset  # overlap with dP

        if sparse_attn_p_mode:
            # Logical and quantized P are simultaneously live in the compute
            # warp groups. Shift registers from the lighter reduce/load roles
            # to keep that QAT-only fragment on chip.
            self.num_regs_reduce = 136
            self.num_regs_compute = 144
            self.num_regs_load = 88
        else:
            self.num_regs_reduce = 160
            self.num_regs_compute = 128
            self.num_regs_load = 96
        self.num_regs_mma = self.num_regs_load
        self.num_regs_empty = 24

        # 128-only path; historical hdim64/192 variants are intentionally unsupported.

        assert (
            self.num_regs_reduce
            + self.num_regs_compute * 2
            + max(self.num_regs_load, self.num_regs_mma)
            <= 512
        )
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.Q_stage = 2
        self.dO_stage = 1
        self.single_stage = 1
        # QAT stages row max and row scale in the former two-stage LSE footprint.
        self.LSE_stage = 1 if self.stage_p_stats else self.Q_stage
        self.num_lse_components = 2 if self.stage_p_stats else 1
        # dPsum_stage = dO_stage
        self.sdKVaccum_stage = 2
        self.dQ_reduce_ncol = 64
        self.sdQaccum_stage = 64 // self.dQ_reduce_ncol
        self.dQ_reduce_ncol_t2r = self.dQ_reduce_ncol
        assert self.tile_hdim % self.dQ_reduce_ncol == 0
        self.dQaccum_reduce_stage = self.tile_hdim // self.dQ_reduce_ncol
        self.dQaccum_reduce_stage_t2r = self.tile_hdim // self.dQ_reduce_ncol_t2r
        self.dK_reduce_ncol = 64
        self.dV_reduce_ncol = 64
        self.cta_group = tcgen05.CtaGroup.ONE

    def _get_tiled_mma(self):
        # S.T = K @ Q.T
        tiled_mma_S = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_kq[:2],
        )
        # dP.T = V @ dO.T
        tiled_mma_dP = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_vdo[:2],
        )
        # dV += P.T @ dO --> (K, MN) major
        tiled_mma_dV = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            OperandMajorMode.K,  # P_major_mode
            OperandMajorMode.MN,  # dO_major_mode
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_pdo[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        # dK += dS.T @ Q
        tiled_mma_dK = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            OperandMajorMode.K,  # dS_major_mode
            OperandMajorMode.MN,  # Q_major_mode
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dsq[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        # dQ = dS @ K
        tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
            self.k_dtype,
            OperandMajorMode.MN,  # dS_major_mode
            OperandMajorMode.MN,  # Kt_major_mode
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dsk[:2],
        )
        return tiled_mma_S, tiled_mma_dP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _setup_smem_layout(self):
        # S.T = K @ Q.T
        sK_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.k_dtype,
            1,
        )
        self.sK_layout = cute.slice_(sK_layout, (None, None, None, 0))
        self.sQ_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.q_dtype,
            self.Q_stage,
        )
        # dP.T = V @ dO.T
        sV_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.v_dtype,
            1,
        )
        self.sV_layout = cute.slice_(sV_layout, (None, None, None, 0))
        self.sdOt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dV += P.T @ dO
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            1,
        )
        self.tP_layout = cute.slice_(tP_layout, (None, None, None, 0))
        self.sdO_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dK += dS.T @ Q
        sdSt_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.sdSt_layout = cute.slice_(sdSt_layout, (None, None, None, 0))
        tdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.tdS_layout = cute.slice_(tdS_layout, (None, None, None, 0))
        self.sQt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.q_dtype,
            self.Q_stage,
        )
        # dQ = dS @ K
        sdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.ds_dtype,
            1,
        )
        self.sdS_layout = cute.slice_(sdS_layout, (None, None, None, 0))
        sKt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.k_dtype,
            1,
        )
        self.sKt_layout = cute.slice_(sKt_layout, (None, None, None, 0))

        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.dQ_reduce_ncol, self.sdQaccum_stage)
        )
        lse_rows = self.tile_m * self.num_lse_components
        self.sLSE_layout = cute.make_layout(
            shape=(lse_rows, self.LSE_stage),
            stride=(1, cute.round_up(lse_rows, 64)),
        )
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m, self.dO_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        self.num_epi_stages = max(1, (self.tile_hdim // 2) // self.dK_reduce_ncol)
        self.num_epi_stages_v = max(1, (self.tile_hdimv // 2) // self.dV_reduce_ncol)
        self.sdK_layout = cute.make_layout((self.tile_n * self.dK_reduce_ncol, 2))
        self.sdV_layout = cute.make_layout((self.tile_n * self.dV_reduce_ncol, 2))
        self.sdK_out_epi_tile = (self.tile_n, self.dK_reduce_ncol)
        self.sdV_out_epi_tile = (self.tile_n, self.dV_reduce_ncol)
        self.sdK_out_layout = sm100_utils_basic.make_smem_layout_epi(
            self.dk_out_dtype,
            LayoutEnum.ROW_MAJOR,
            self.sdK_out_epi_tile,
            2,
        )
        self.sdV_out_layout = sm100_utils_basic.make_smem_layout_epi(
            self.dv_out_dtype,
            LayoutEnum.ROW_MAJOR,
            self.sdV_out_epi_tile,
            2,
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdKaccum: cute.Tensor,
        mdVaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mDkvOwnerCounts: cute.Tensor,
        mDkvWriterRank: Optional[cute.Tensor],
        mDqSemaphore: Optional[cute.Tensor],
        mDkvSemaphore: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mFragmentIndices: Optional[cute.Tensor] = None,
        # k2q CSR sparse inputs:
        #   mQ_flat:      [B*Sq*H_q, D] bf16 — same storage as mQ reshaped such
        #                 that each row is one (b, s, h_q). TMA box = (qhead,
        #                 k_tile) loads qhead consecutive rows (= the qhead
        #                 heads for one (b, s, h_kv)) × k_tile D columns per
        #                 issue. Current attention path uses 16x64 subtiles.
        #   mdO_flat:     same shape as mQ_flat for dO scatter gather
        #   csr:
        #     mK2qIndices: [H_kv, nnz] int32 — batch-local q_idx payload
        #     mK2qCounts:  [H_kv, total_rows + 1] int32 — CSR row_ptr
        mQ_flat: Optional[cute.Tensor] = None,
        mdO_flat: Optional[cute.Tensor] = None,
        mK2qIndices: Optional[cute.Tensor] = None,
        mK2qCounts: Optional[cute.Tensor] = None,
        mK2qQSplitIndices: Optional[cute.Tensor] = None,
        mPQuantScale: Optional[cute.Tensor] = None,
        mSchedulerMetadata: Optional[cute.Tensor] = None,
        mWorkCount: Optional[cute.Tensor] = None,
        work_capacity: Int32 = Int32(0),
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.lse_dtype = mLSE.element_type
        self.dpsum_dtype = mdPsum.element_type
        self.dqaccum_dtype = mdQaccum.element_type
        self.dk_dtype = mdKaccum.element_type
        self.dv_dtype = mdVaccum.element_type
        self.dk_out_dtype = mdK.element_type
        self.dv_out_dtype = mdV.element_type
        self.ds_dtype = self.q_dtype

        if const_expr(mCuSeqlensQ is None or mCuSeqlensK is None):
            raise ValueError("sparse backward requires cu_seqlens_q and cu_seqlens_k")
        self.is_varlen_k = const_expr(True)
        self.is_varlen_q = const_expr(True)
        # Sparse-only attention path: CSR metadata and prepared work are required.
        if const_expr(mK2qIndices is None or mK2qCounts is None):
            raise ValueError("Sparse backward requires k2q_indices and k2q_counts")
        if const_expr(
            mSchedulerMetadata is None
            or mWorkCount is None
        ):
            raise ValueError(
                "Sparse backward prepare scheduler requires metadata and work count"
            )
        if const_expr(self.deterministic):
            if const_expr(
                mK2qQSplitIndices is None
                or mDkvWriterRank is None
                or mDqSemaphore is None
                or mDkvSemaphore is None
            ):
                raise ValueError(
                    "deterministic backward requires qsplit, writer-rank, and semaphore tensors"
                )
        assert (mQ_flat is not None and mdO_flat is not None and mK2qCounts is not None), (
            "k2q sparse backward requires mQ_flat, mdO_flat, mK2qIndices, mK2qCounts all non-None"
        )
        if const_expr(self.sparse_attn_p_mode):
            if const_expr(
                mK2qQSplitIndices is None
                or mPQuantScale is None
                or mPQuantScale.element_type != Float32
            ):
                raise ValueError(
                    "BF16 probability QAT backward requires qsplit metadata "
                    "and FP32 P-scale scratch"
                )
            mPQuantScale = assume_tensor_aligned(mPQuantScale)
        assert self.pack_gqa, "k2q sparse backward requires pack_gqa=True"
        assert self.tile_m % self.qhead_per_kvhead == 0, (
            "k2q sparse backward requires tile_m divisible by qhead_per_kvhead "
            f"(got tile_m={self.tile_m}, qhead_per_kvhead={self.qhead_per_kvhead})"
        )
        self.use_scalar_stats_load = False
        # Number of scattered Q positions represented by one packed MMA tile.
        self.q_per_tile: cutlass.Constexpr[int] = self.tile_m // self.qhead_per_kvhead
        self.sparse_load_meta_group_iters: cutlass.Constexpr[int] = 4
        assert self.dk_dtype.width == 32, "Must accumulate dK in float precision"
        assert self.dv_dtype.width == 32, "Must accumulate dV in float precision"
        if const_expr(self.dk_out_dtype not in (cutlass.BFloat16, cutlass.Float16)):
            raise TypeError("dK output must be BFloat16 or Float16")
        if const_expr(self.dv_out_dtype != self.dk_out_dtype):
            raise TypeError("dK and dV outputs must have the same dtype")

        mdQaccum, mdKaccum, mdVaccum, mdK, mdV = [
            assume_tensor_aligned(t)
            for t in (mdQaccum, mdKaccum, mdVaccum, mdK, mdV)
        ]
        mLSE = assume_tensor_aligned(mLSE)
        mdPsum = assume_tensor_aligned(mdPsum)

        # Sparse backward uses token-major varlen layouts throughout.
        QO_layout_transpose = [0, 2, 1]
        mQ, mdO = [layout_utils.select(t, mode=QO_layout_transpose) for t in (mQ, mdO)]

        KV_layout_transpose = [0, 2, 1]
        mK, mV = [layout_utils.select(t, mode=KV_layout_transpose) for t in (mK, mV)]
        mdK, mdV = [layout_utils.select(t, mode=KV_layout_transpose) for t in (mdK, mdV)]

        # LSE/dPsum stay token-major [total_q, H_q] and mdQaccum stays in the
        # delivered varlen workspace layout.
        LSE_dPsum_transpose = [1, 0]
        # GQA16 uses the packed varlen FP32 dQ workspace.
        dQaccum_transpose = [1, 0]
        mLSE, mdPsum = [
            layout_utils.select(t, mode=LSE_dPsum_transpose)
            for t in (mLSE, mdPsum)
        ]
        mdQaccum = layout_utils.select(mdQaccum, mode=dQaccum_transpose)

        mdKaccum, mdVaccum = [
            layout_utils.select(t, mode=[1, 0]) for t in (mdKaccum, mdVaccum)
        ]

        # PackGQA: fold qhead_per_kvhead into seqlen (compound mode 0).
        # Applied BEFORE dO_transpose because pack_gqa_layout always folds head into
        # mode 0 — it needs Sq at mode 0. mQ/mdO are (Sq, D, H_q, B) at this point.
        # mQ/mdO (head_idx=2): (Sq, D, H_q, B) -> ((qhead_per_kv, Sq), D, H_kv, B)
        # mLSE/mdPsum (head_idx=1): (Sq, H_q, B) -> ((qhead_per_kv, Sq), H_kv, B)
        # mdQaccum is packed as [H_kv, total_q_padded * 16 * D], which is
        # directly compatible with bulk TMA reduce.
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]  # H_kv after KV_layout_transpose
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mdO = pack_gqa_layout(mdO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)
            mdPsum = pack_gqa_layout(mdPsum, self.qhead_per_kvhead, nheads_kv, head_idx=1)

        # (s, h, n, b) --> (h, s, n, b) or (t, h, n) -> (h, t, b). Under pack_gqa,
        # mode 0 is compound ((qhead, Sq)) — the select still swaps outermost modes.
        dO_transpose = [1, 0, 2]
        mdO = layout_utils.select(mdO, mode=dO_transpose)

        self._setup_attributes()
        # Sparse dQ protocol: token-major workspace with token-stage hidden layout.
        self.sparse_dq_num_stages = self.dQaccum_reduce_stage
        self.sparse_dq_token_elems = self.qhead_per_kvhead * self.tile_hdim
        self.sparse_dq_stage_token_elems = self.qhead_per_kvhead * self.dQ_reduce_ncol
        self.sparse_dq_stage_tile_elems = self.q_per_tile * self.sparse_dq_stage_token_elems
        assert self.sparse_dq_num_stages * self.sparse_dq_stage_token_elems == (
            self.sparse_dq_token_elems
        )
        assert self.sparse_dq_stage_tile_elems == self.tile_m * self.dQ_reduce_ncol
        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()
        self._setup_smem_layout()

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma_S.thr_id.shape,),
        )
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_q_do_mcast = self.num_mcast_ctas_b > 1

        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
            tma_store_op,
            mdK,
            cute.select(self.sdK_out_layout, mode=[0, 1]),
            self.sdK_out_epi_tile,
            1,
        )
        tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
            tma_store_op,
            mdV,
            cute.select(self.sdV_out_layout, mode=[0, 1]),
            self.sdV_out_epi_tile,
            1,
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        # S.T = K @ Q.T
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        Q_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_S.thr_id
        )
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            Q_tma_op,
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        # dP.T = V @ dO.T
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV,
            cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            self.tiled_mma_dP,
            self.cluster_layout_vmnk.shape,
        )
        # dV = P.T @ dO
        dO_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_dV.thr_id
        )
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            dO_tma_op,
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_pdo,
            self.tiled_mma_dV,
            self.cluster_layout_vmnk.shape,
        )

        # Scatter-friendly TMA atoms on flat [B*Sq*H_q, D] mQ/mdO.
        # Box (qhead, k_tile=64) loads consecutive heads × 64 D per issue.
        tma_atom_Q_scatter = tma_tensor_Q_scatter = None
        tma_atom_dO_scatter = tma_tensor_dO_scatter = None
        self.k_tile: cutlass.Constexpr[int] = 64
        assert self.tile_hdim % self.k_tile == 0, "tile_hdim must be divisible by k_tile"
        self.k_subtiles: cutlass.Constexpr[int] = self.tile_hdim // self.k_tile
        self.num_subtiles_per_stage: cutlass.Constexpr[int] = self.q_per_tile * self.k_subtiles
        num_subtiles_total = self.Q_stage * self.num_subtiles_per_stage
        self.sQ_load_layout = sm100_utils_basic.make_smem_layout(
            OperandMajorMode.K,
            (self.qhead_per_kvhead, self.k_tile),
            self.q_dtype,
            num_subtiles_total,
        )
        self.sdO_load_layout = sm100_utils_basic.make_smem_layout(
            OperandMajorMode.K,
            (self.qhead_per_kvhead, self.k_tile),
            self.do_dtype,
            num_subtiles_total,
        )
        mQ_flat = assume_tensor_aligned(mQ_flat)
        mQ_flat_2d = cute.make_tensor(
            mQ_flat.iterator, cute.select(mQ_flat.layout, mode=[0, 1])
        )
        tma_atom_Q_scatter, tma_tensor_Q_scatter = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ_flat_2d,
            cute.select(self.sQ_load_layout, mode=[0, 1]),
            (self.qhead_per_kvhead, self.k_tile),
        )
        mdO_flat = assume_tensor_aligned(mdO_flat)
        mdO_flat_2d = cute.make_tensor(
            mdO_flat.iterator, cute.select(mdO_flat.layout, mode=[0, 1])
        )
        tma_atom_dO_scatter, tma_tensor_dO_scatter = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdO_flat_2d,
            cute.select(self.sdO_load_layout, mode=[0, 1]),
            (self.qhead_per_kvhead, self.k_tile),
        )
        self.q_subtile_bytes: cutlass.Constexpr[int] = cute.size_in_bytes(
            self.q_dtype, cute.select(self.sQ_load_layout, mode=[0, 1])
        )
        self.do_subtile_bytes: cutlass.Constexpr[int] = cute.size_in_bytes(
            self.do_dtype, cute.select(self.sdO_load_layout, mode=[0, 1])
        )
        self.tma_copy_bytes = {
            name: self.cta_group_size
            * cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = self.tile_m * self.dQ_reduce_ncol * Float32.width // 8
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.dK_reduce_ncol * Float32.width // 8
        self.tma_copy_bytes["dS"] = cute.size_in_bytes(self.ds_dtype, self.sdS_layout)

        TileScheduler = SingleTileScheduler
        packed_total_q = cute.size(mQ.shape[0])
        total_kv = mK.shape[0]
        head_dim = mK.shape[1]
        head_dim_v = mV.shape[1]
        tile_sched_args = TileSchedulerArguments(
            num_block=work_capacity,
            num_head=Int32(1),
            num_batch=Int32(1),
            num_splits=1,
            seqlen_k=total_kv,
            headdim=head_dim,
            headdim_v=head_dim_v,
            total_q=packed_total_q,
            tile_shape_mn=self.cta_tiler[:2],
            cluster_shape_mn=self.cluster_shape_mnk[:2],
            element_size=self.k_dtype.width // 8,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # Compute allocation sizes for shared buffers reused by the epilogues.
        # sdK aliases sQ. GQA16 sdV spans contiguous sV and sdO storage.
        sQ_alloc_bytes = max(
            cute.size_in_bytes(self.q_dtype, self.sQ_layout),
            cute.size_in_bytes(self.dk_dtype, self.sdK_layout),
            cute.size_in_bytes(self.dk_out_dtype, self.sdK_out_layout),
        )
        sdK_bytes = cute.size_in_bytes(self.dk_dtype, self.sdK_layout)
        sdV_bytes = cute.size_in_bytes(self.dv_dtype, self.sdV_layout)
        sdV_out_bytes = cute.size_in_bytes(self.dv_out_dtype, self.sdV_out_layout)
        sV_bytes = cute.size_in_bytes(self.v_dtype, self.sV_layout)
        sdO_bytes = cute.size_in_bytes(self.do_dtype, self.sdO_layout)
        sdO_alloc_bytes = sdO_bytes
        assert max(sdV_bytes, sdV_out_bytes) <= sV_bytes + sdO_alloc_bytes, (
            "sdV doesn't fit in contiguous sV and sdO storage allocations"
        )
        assert sdK_bytes <= sQ_alloc_bytes, "sdK doesn't fit in sQ storage allocation"

        @cute.struct
        class SharedStorage:
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
            dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
            LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.LSE_stage]
            dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
            S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
            dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
            dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
            dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.sdKVaccum_stage]
            dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: Int32
            sparse_load_qidx: cute.struct.Align[
                cute.struct.MemRange[
                    Int32,
                    2 * self.sparse_load_meta_group_iters * self.q_per_tile,
                ],
                128,
            ]
            sparse_load_rowbase: cute.struct.Align[
                cute.struct.MemRange[
                    Int32,
                    2 * self.sparse_load_meta_group_iters * self.q_per_tile,
                ],
                128,
            ]

            sQ: cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, sQ_alloc_bytes],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(self.sK_layout)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(self.sV_layout)],
                self.buffer_align_bytes,
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, sdO_alloc_bytes],
                self.buffer_align_bytes,
            ]
            sdS: cute.struct.Align[
                cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                128,
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                128,
            ]
            sdPsum: cute.struct.Align[
                cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                128,
            ]
            sdQaccum: cute.struct.Align[
                cute.struct.MemRange[
                    self.dqaccum_dtype,
                    cute.cosize(self.sdQaccum_layout),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        softmax_scale_log2 = softmax_scale * math.log2(math.e)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            mLSE,
            mdPsum,
            tma_tensor_dO,
            mdVaccum,
            mdKaccum,
            tma_tensor_dV,
            tma_tensor_dK,
            mdV,
            mdK,
            mDkvOwnerCounts,
            mDkvWriterRank,
            mDqSemaphore,
            mDkvSemaphore,
            mdQaccum,
            mCuSeqlensQ,
            mCuSeqlensK,
            mFragmentIndices,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dV,
            tma_atom_dK,
            self.sQ_layout,
            self.sQt_layout,
            self.sK_layout,
            self.sKt_layout,
            self.sV_layout,
            self.sLSE_layout,
            self.sdPsum_layout,
            self.sdO_layout,
            self.sdOt_layout,
            self.sdSt_layout,
            self.sdS_layout,
            self.sdQaccum_layout,
            self.sdK_layout,
            self.sdV_layout,
            self.sdK_out_layout,
            self.sdV_out_layout,
            self.tP_layout,
            self.tdS_layout,
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dV,
            self.tiled_mma_dK,
            self.tiled_mma_dQ,
            softmax_scale,
            softmax_scale_log2,
            # k2q CSR sparse metadata
            tma_atom_Q_scatter,
            tma_tensor_Q_scatter,
            tma_atom_dO_scatter,
            tma_tensor_dO_scatter,
            self.sQ_load_layout,
            self.sdO_load_layout,
            mK2qIndices,
            mK2qCounts,
            mK2qQSplitIndices,
            mPQuantScale,
            mSchedulerMetadata,
            mWorkCount,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk if cute.size(self.cluster_shape_mnk) > 1 else None,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        mdVaccum: cute.Tensor,
        mdKaccum: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        mdV_raw: cute.Tensor,
        mdK_raw: cute.Tensor,
        mDkvOwnerCounts: cute.Tensor,
        mDkvWriterRank: Optional[cute.Tensor],
        mDqSemaphore: Optional[cute.Tensor],
        mDkvSemaphore: Optional[cute.Tensor],
        mdQaccum: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mFragmentIndices: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        sQ_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sdPsum_layout: cute.Layout,
        sdO_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        sdK_layout: cute.ComposedLayout | cute.Layout,
        sdV_layout: cute.ComposedLayout | cute.Layout,
        sdK_out_layout: cute.ComposedLayout,
        sdV_out_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        # k2q CSR sparse: scatter TMA atoms + tensors on flat mQ/mdO, and per-kv Q indices.
        tma_atom_Q_scatter: Optional[cute.CopyAtom] = None,
        tma_tensor_Q_scatter: Optional[cute.Tensor] = None,
        tma_atom_dO_scatter: Optional[cute.CopyAtom] = None,
        tma_tensor_dO_scatter: Optional[cute.Tensor] = None,
        sQ_load_layout: Optional[cute.ComposedLayout] = None,
        sdO_load_layout: Optional[cute.ComposedLayout] = None,
        mK2qIndices: Optional[cute.Tensor] = None,
        mK2qCounts: Optional[cute.Tensor] = None,
        mK2qQSplitIndices: Optional[cute.Tensor] = None,
        mPQuantScale: Optional[cute.Tensor] = None,
        mSchedulerMetadata: Optional[cute.Tensor] = None,
        mWorkCount: Optional[cute.Tensor] = None,
    ):
        exit_thread_if(Int32(cute.arch.block_idx()[0] >= mWorkCount[Int32(0)]))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        mma_tile_coord_v = 0
        is_leader_cta = const_expr(True)

        # Prefetch TMA descriptors used by the load warps.
        if warp_idx == self.load_warp_id or warp_idx == self.load_do_warp_id:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_Q)
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)
                cpasync.prefetch_descriptor(tma_atom_dO)
                cpasync.prefetch_descriptor(tma_atom_dV)
                cpasync.prefetch_descriptor(tma_atom_dK)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_S.thr_id.shape,),
        )

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len((self.mma_warp_id, *self.compute_warp_ids, *self.reduce_warp_ids)),
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
        )

        # UMMA producers and AsyncThread consumers
        pipeline_producer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        pipeline_consumer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids) * self.cta_group_size
        )
        pipeline_S_P = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.S_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dKV = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dKV_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_consumer_group_MMA_AsyncThread_dQ = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.reduce_warp_ids) * self.cta_group_size,
        )  # Compute
        pipeline_dQ = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread_dQ,
            barrier_storage=storage.dQ_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # AsyncThread producers and UMMA consumers
        # Only 1 thread per warp will signal
        pipeline_PdS_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * self.cta_group_size,
        )  # Compute
        pipeline_PdS_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )  # MMA
        pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline_PdS_producer_group,
            consumer_group=pipeline_PdS_consumer_group,
            barrier_storage=storage.dS_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # TMA producer and UMMA consumers
        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.load_warp_id])
        )
        # The arrive count is the number of mcast size
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id]) * self.num_mcast_ctas_b
        )
        pipeline_consumer_group_compute = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * 1,
        )
        # LSE/dPsum pipeline type depends on load mechanism:
        # - pack_gqa=False: TMA bulk G2S → PipelineTmaAsync (bytes-driven via
        #   mbarrier tx_count, auto-signaled on TMA completion).
        # - pack_gqa=True: thread-level cp.async.cg → PipelineCpAsync (manages
        #   cp_async commit_group/wait_group internally, no tx_count semantics).
        #   Matches DSA sm100 bwd pattern (make_and_init_load_compute_LSE_pipeline).
        if const_expr(self.pack_gqa):
            # pack_gqa LSE/dPsum load = 32-thread cp.async (full load warp).
            # producer_group = WARP_SIZE = 32 threads.
            # consumer_group = WARP_SIZE * num_compute_warps = ALL compute threads
            # because consumer_release is called by all compute threads (not elect_one).
            # Match DSA sm100 bwd pattern (dsa_bwd_sm100.py make_and_init_load_compute_LSE_pipeline).
            # defer_sync=True: skip inline fence+block-sync (caller handles init sync).
            pipeline_cp_async_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, cute.arch.WARP_SIZE,
            )
            pipeline_cp_async_consumer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread,
                cute.arch.WARP_SIZE * len(self.compute_warp_ids),
            )
            pipeline_LSE = cutlass.pipeline.PipelineCpAsync.create(
                barrier_storage=storage.LSE_mbar_ptr.data_ptr(),
                num_stages=self.LSE_stage,
                producer_group=pipeline_cp_async_producer_group,
                consumer_group=pipeline_cp_async_consumer_group,
                defer_sync=True,
            )
            pipeline_dPsum = cutlass.pipeline.PipelineCpAsync.create(
                barrier_storage=storage.dPsum_mbar_ptr.data_ptr(),
                num_stages=self.dO_stage,
                producer_group=pipeline_cp_async_producer_group,
                consumer_group=pipeline_cp_async_consumer_group,
                defer_sync=True,
            )
        else:
            pipeline_LSE = cutlass.pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.LSE_mbar_ptr.data_ptr(),
                num_stages=self.LSE_stage,
                producer_group=pipeline_producer_group,
                consumer_group=pipeline_consumer_group_compute,
                tx_count=self.tma_copy_bytes["LSE"],
                defer_sync=True,
            )
            pipeline_dPsum = cutlass.pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.dPsum_mbar_ptr.data_ptr(),
                num_stages=self.dO_stage,
                producer_group=pipeline_producer_group,
                consumer_group=pipeline_consumer_group_compute,
                tx_count=self.tma_copy_bytes["dPsum"],
                defer_sync=True,
            )
        pipeline_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        pipeline_dO = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.dO_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=False,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype)
        sQt = cute.make_tensor(
            cute.recast_ptr(sQ.iterator, sQt_layout.inner, dtype=self.q_dtype),
            sQt_layout.outer,
        )
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sKt = cute.make_tensor(
            cute.recast_ptr(sK.iterator, sKt_layout.inner), sKt_layout.outer
        )
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdSt = storage.sdS.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        sdS = cute.make_tensor(cute.recast_ptr(sdSt.iterator, sdS_layout.inner), sdS_layout.outer)

        sdO = storage.sdO.get_tensor(
            sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_dtype
        )
        sdOt = cute.make_tensor(
            cute.recast_ptr(sdO.iterator, sdOt_layout.inner, dtype=self.do_dtype),
            sdOt_layout.outer,
        )

        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)
        sSparseLoadQIdx = storage.sparse_load_qidx.get_tensor(
            cute.make_layout((self.q_per_tile, self.sparse_load_meta_group_iters, 2))
        )
        sSparseLoadRowBase = storage.sparse_load_rowbase.get_tensor(
            cute.make_layout((self.q_per_tile, self.sparse_load_meta_group_iters, 2))
        )

        sQ_load = storage.sQ.get_tensor(
            sQ_load_layout.outer,
            swizzle=sQ_load_layout.inner,
            dtype=self.q_dtype,
        )
        sdO_load = storage.sdO.get_tensor(
            sdO_load_layout.outer,
            swizzle=sdO_load_layout.inner,
            dtype=self.do_dtype,
        )
        sdV = storage.sV.get_tensor(sdV_layout, dtype=self.dv_dtype)
        sdV_out = storage.sV.get_tensor(
            sdV_out_layout.outer,
            swizzle=sdV_out_layout.inner,
            dtype=self.dv_out_dtype,
        )
        sdK = storage.sQ.get_tensor(sdK_layout, dtype=self.dk_dtype)
        sdK_out = storage.sQ.get_tensor(
            sdK_out_layout.outer,
            swizzle=sdK_out_layout.inner,
            dtype=self.dk_out_dtype,
        )

        # The SharedStorage capacity checks above cover all aliasing views.
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        # The kernel allocates all 512 TMEM columns, so the base column is zero.
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        # S
        thr_mma_S = tiled_mma_S.get_slice(mma_tile_coord_v)
        Sacc_shape = thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])  # (M, N)
        tStS = thr_mma_S.make_fragment_C(Sacc_shape)
        # (MMA, MMA_M, MMA_N)
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)
        # dP
        thr_mma_dP = tiled_mma_dP.get_slice(mma_tile_coord_v)
        dPacc_shape = thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2])
        tdPtdP = thr_mma_dP.make_fragment_C(dPacc_shape)
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP.layout)
        # dV
        thr_mma_dV = tiled_mma_dV.get_slice(mma_tile_coord_v)
        dvacc_shape = thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dV.make_fragment_C(dvacc_shape)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)
        tP = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_P_offset, dtype=self.do_dtype), tP_layout.outer
        )
        # dK
        thr_mma_dK = tiled_mma_dK.get_slice(mma_tile_coord_v)
        dkacc_shape = thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
        tdKtdK = thr_mma_dK.make_fragment_C(dkacc_shape)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)
        tdS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_dS_offset, dtype=self.ds_dtype), tdS_layout.outer
        )
        # dQ
        thr_mma_dQ = tiled_mma_dQ.get_slice(mma_tile_coord_v)
        dQacc_shape = thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
        tdQtdQ = thr_mma_dQ.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tmem_ptr + self.tmem_dQ_offset, tdQtdQ.layout)

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n * self.cluster_shape_mnk[0],  # careful, this case is not very well-tested
            self.is_causal,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            # With pack_gqa, mQ.shape[0] is compound (qhead_per_kv, Sq); take mode[1] for Sq.
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mFragmentIndices=mFragmentIndices,
            tile_m=self.tile_m,
            tile_n=self.tile_n * self.cluster_shape_mnk[0],
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n * self.cta_group_size,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            swap_AB=True,
        )
        #  EMPTY
        # (15)
        if warp_idx == self.empty_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)

        #  RELAY / dO LOAD
        # (14)
        if warp_idx == self.relay_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                mQ,
                mK,
                mV,
                mdO,
                mLSE,
                mdPsum,
                sQ,
                sK,
                sV,
                sdO,
                sLSE,
                sdPsum,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_dO,
                pipeline_Q,
                pipeline_dO,
                pipeline_LSE,
                pipeline_dPsum,
                cluster_layout_vmnk,
                block_info,
                SeqlenInfoCls,
                # k2q CSR sparse scatter inputs
                tma_atom_Q_scatter,
                tma_tensor_Q_scatter,
                tma_atom_dO_scatter,
                tma_tensor_dO_scatter,
                sQ_load,
                sdO_load,
                sSparseLoadQIdx,
                sSparseLoadRowBase,
                mK2qIndices,
                mK2qCounts,
                mK2qQSplitIndices,
                mPQuantScale,
                mSchedulerMetadata,
                mWorkCount,
                should_load_Q=False,
                should_load_dO=True,
            )

        #  LOAD
        # (13)
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                mQ,
                mK,
                mV,
                mdO,
                mLSE,
                mdPsum,
                sQ,
                sK,
                sV,
                sdO,
                sLSE,
                sdPsum,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_dO,
                pipeline_Q,
                pipeline_dO,
                pipeline_LSE,
                pipeline_dPsum,
                cluster_layout_vmnk,
                block_info,
                SeqlenInfoCls,
                # k2q CSR sparse scatter inputs
                tma_atom_Q_scatter,
                tma_tensor_Q_scatter,
                tma_atom_dO_scatter,
                tma_tensor_dO_scatter,
                sQ_load,
                sdO_load,
                sSparseLoadQIdx,
                sSparseLoadRowBase,
                mK2qIndices,
                mK2qCounts,
                mK2qQSplitIndices,
                mPQuantScale,
                mSchedulerMetadata,
                mWorkCount,
                should_load_Q=True,
                should_load_dO=False,
            )

        #  MMA
        # (12)
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            # Alloc tmem buffer
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)

            self.mma(
                tiled_mma_S,
                tiled_mma_dP,
                tiled_mma_dV,
                tiled_mma_dK,
                tiled_mma_dQ,
                sQ,
                sQt,
                sK,
                sKt,
                sV,
                sdO,
                sdOt,
                tP,
                sdS,
                tdS,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                tdQtdQ,
                pipeline_Q,
                pipeline_dO,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                pipeline_dQ,
                block_info,
                SeqlenInfoCls,
                is_leader_cta,
                mK2qCounts,
                mSchedulerMetadata,
                mWorkCount,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # Compute
        # (4, 5, 6, 7, 8, 9, 10, 11) --> 8 warps
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_compute)  # 8 warps
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.compute_loop(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                sLSE,
                sdPsum,
                mdVaccum,
                mdKaccum,
                mdV,
                mdK,
                mdV_raw,
                mdK_raw,
                tma_atom_dV,
                tma_atom_dK,
                mDkvOwnerCounts,
                mDkvWriterRank,
                mDkvSemaphore,
                sdS,
                pipeline_LSE,
                pipeline_dPsum,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                softmax_scale,
                softmax_scale_log2,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                sdV,
                sdK,
                sdV_out,
                sdK_out,
                mK2qIndices,
                mK2qCounts,
                mK2qQSplitIndices,
                mPQuantScale,
                mSchedulerMetadata,
                mWorkCount,
                mFragmentIndices,
            )
            tmem_alloc_barrier.arrive()

        # Reduce
        # (0, 1, 2, 3) - dQ
        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.dq_acc_reduce(
                mdQaccum,
                sdQaccum,
                thr_mma_dQ,
                tdQtdQ,
                pipeline_dQ,
                SeqlenInfoCls,
                mK2qIndices,
                mK2qCounts,
                mK2qQSplitIndices,
                mDqSemaphore,
                mSchedulerMetadata,
                mWorkCount,
            )
            tmem_alloc_barrier.arrive()

        return

    @cute.jit
    def _resolve_sparse_work(
        self,
        work_idx: Int32,
        mK2qCounts: cute.Tensor,
        mSchedulerMetadata: Optional[cute.Tensor],
        mWorkCount: Optional[cute.Tensor],
    ) -> tuple[Int32, Int32, Int32, Int32, Int32]:
        n_block = Int32(0)
        head_idx = Int32(0)
        batch_idx = Int32(0)
        row_start = Int32(0)
        row_count = Int32(0)
        if work_idx < mWorkCount[Int32(0)]:
            head_idx = mSchedulerMetadata[work_idx, Int32(0)]
            row_linear = mSchedulerMetadata[work_idx, Int32(1)]
            q_begin = mSchedulerMetadata[work_idx, Int32(2)]
            row_count = mSchedulerMetadata[work_idx, Int32(3)]
            batch_idx = mSchedulerMetadata[work_idx, Int32(4)]
            n_block = mSchedulerMetadata[work_idx, Int32(5)]
            row_start = mK2qCounts[head_idx, row_linear] + q_begin
        return n_block, head_idx, batch_idx, row_start, row_count

    @cute.jit
    def _load_sparse_q_idx(
        self,
        mK2qIndices: cute.Tensor,
        head_idx: Int32,
        row_start: Int32,
        qi: Int32,
    ) -> Int32:
        return mK2qIndices[head_idx, row_start + qi]

    @cute.jit
    def load(
        self,
        thr_mma_S: cute.ThrMma,
        thr_mma_dP: cute.ThrMma,
        thr_mma_dV: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        cluster_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        # k2q CSR sparse scatter
        tma_atom_Q_scatter: Optional[cute.CopyAtom] = None,
        tma_tensor_Q_scatter: Optional[cute.Tensor] = None,
        tma_atom_dO_scatter: Optional[cute.CopyAtom] = None,
        tma_tensor_dO_scatter: Optional[cute.Tensor] = None,
        sQ_load: Optional[cute.Tensor] = None,
        sdO_load: Optional[cute.Tensor] = None,
        sSparseLoadQIdx: Optional[cute.Tensor] = None,
        sSparseLoadRowBase: Optional[cute.Tensor] = None,
        mK2qIndices: Optional[cute.Tensor] = None,
        mK2qCounts: Optional[cute.Tensor] = None,
        mK2qQSplitIndices: Optional[cute.Tensor] = None,
        mPQuantScale: Optional[cute.Tensor] = None,
        mSchedulerMetadata: Optional[cute.Tensor] = None,
        mWorkCount: Optional[cute.Tensor] = None,
        should_load_Q: bool = True,
        should_load_dO: bool = True,
    ):
        split_sparse_qdo_producers = const_expr(should_load_Q != should_load_dO)

        # Compute multicast mask for Q & dO buffer full
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        q_do_mcast_mask = None
        if const_expr(self.is_q_do_mcast):
            q_do_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        producer_state_Q_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_LSE = None
        if const_expr(self.stage_p_stats):
            producer_state_LSE = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.LSE_stage
            )
        producer_state_dO_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )
        work_idx = cute.arch.block_idx()[0]
        for _ in cutlass.range_constexpr(1):
            n_block, head_idx, batch_idx, row_start_c, k2q_count_c = (
                self._resolve_sparse_work(
                    work_idx,
                    mK2qCounts,
                    mSchedulerMetadata,
                    mWorkCount,
                )
            )
            row_start_sparse = row_start_c
            k2q_count = k2q_count_c
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )
            head_idx_kv = (
                head_idx
                if const_expr(self.pack_gqa)
                else head_idx // self.qhead_per_kvhead
            )
            n_block_cta_group = n_block // self.cta_group_size

            # GMEM tensors (varlen-aware)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
            mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
            mdO_offset = (
                (0, (None, seqlen.offset_q))
                if const_expr(self.pack_gqa)
                else (0, seqlen.offset_q)
            )
            mdO_cur = cute.domain_offset(mdO_offset, mdO[None, None, head_idx])
            if const_expr(self.pack_gqa):
                # For varlen PackGQA stats, select the KV head first and then
                # offset only the sequence submode of the compound packed row.
                # This avoids carrying the head mode through a nested
                # domain_offset, which can alias batch 0 rows for later
                # batches under the sparse stats load path.
                mLSE_cur = cute.domain_offset(
                    ((None, seqlen.padded_offset_q),),
                    mLSE[None, head_idx],
                )
                mdPsum_cur = cute.domain_offset(
                    ((None, seqlen.padded_offset_q),),
                    mdPsum[None, head_idx],
                )
            else:
                mLSE_cur = cute.domain_offset((seqlen.padded_offset_q,), mLSE[None, head_idx])
                mdPsum_cur = cute.domain_offset(
                    (seqlen.padded_offset_q,),
                    mdPsum[None, head_idx],
                )

            # (1) S.T = K @ Q.T
            gK = cute.local_tile(
                mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block_cta_group, 0)
            )
            tSgK = thr_mma_S.partition_A(gK)

            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0))
            tSgQ = thr_mma_S.partition_B(gQ)

            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                tSgK,
                sK,
                single_stage=True,
            )

            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tSgQ,
                dst_tensor=sQ,
                mcast_mask=q_do_mcast_mask,
            )
            load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)

            # (2) dP = V @ dO.T
            gV = cute.local_tile(
                mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block_cta_group, 0)
            )
            tdPgV = thr_mma_dP.partition_A(gV)

            load_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V,
                0,
                cute.make_layout(1),
                tdPgV,
                sV,
                single_stage=True,
            )

            # (3) dV += P.T @ dO
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdVgdO = thr_mma_dV.partition_B(gdO)
            load_dO, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tdVgdO,
                dst_tensor=sdO,
                mcast_mask=q_do_mcast_mask,
            )
            load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)

            # LSE/dPsum load: sparse PackGQA path uses thread-level cp.async.cg.
            # Extras used only when pack_gqa=True (const_expr-pruned otherwise).
            # cp.async.cg requires 128-bit (16-byte) granularity — each thread
            # copies 4 fp32 = 16 bytes = 128 bits. 32 threads × 4 = 128 = tile_m.
            # Within a tile's compound stride, each thread's 4-element window is
            # contiguous (inner stride = 1), so 16-byte alignment holds.
            # GQA16 provides four consecutive FP32 values per stats copy, so
            # cp.async.cg naturally satisfies its 16-byte alignment.
            async_stats_atom = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
                Float32,
                num_bits_per_copy=128,
            )
            load_warp_lane = cute.arch.thread_idx()[0] % cute.arch.WARP_SIZE

            # Scatter Q/dO copy fns. One flat mQ_flat 2D view is split into
            # k_subtiles (k_tile=64) along dim-1.
            gQ_k0 = cute.local_tile(
                tma_tensor_Q_scatter,
                (self.qhead_per_kvhead, self.k_tile),
                (None, 0),
            )
            gQ_k1 = cute.local_tile(
                tma_tensor_Q_scatter,
                (self.qhead_per_kvhead, self.k_tile),
                (None, 1 if self.k_subtiles > 1 else 0),
            )
            load_Q_fn_k0, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q_scatter, 0, cute.make_layout(1), gQ_k0, sQ_load
            )
            if const_expr(self.k_subtiles > 1):
                load_Q_fn_k1, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q_scatter, 0, cute.make_layout(1), gQ_k1, sQ_load
                )
            else:
                load_Q_fn_k1 = None
            gdO_k0 = cute.local_tile(
                tma_tensor_dO_scatter,
                (self.qhead_per_kvhead, self.k_tile),
                (None, 0),
            )
            gdO_k1 = cute.local_tile(
                tma_tensor_dO_scatter,
                (self.qhead_per_kvhead, self.k_tile),
                (None, 1 if self.k_subtiles > 1 else 0),
            )
            load_dO_fn_k0, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO_scatter, 0, cute.make_layout(1), gdO_k0, sdO_load
            )
            if const_expr(self.k_subtiles > 1):
                load_dO_fn_k1, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dO_scatter, 0, cute.make_layout(1), gdO_k1, sdO_load
                )
            else:
                load_dO_fn_k1 = None

            num_q_tiles = cute.ceil_div(k2q_count, self.q_per_tile)
            process_tile = num_q_tiles > Int32(0)

            if process_tile:
                # ========================================================
                # k2q CSR sparse: scatter Q/dO per-token TMA loads + scatter
                # LSE/dPsum per-token cp.async.cg. Iterates over num_q_tiles
                # = ceil_div(k2q_count, q_per_tile) instead of m_block range.
                # ========================================================
                nheads_kv = cute.size(mK.shape[2])
                # LSE/dPsum scatter gather partition:
                #   each thread loads 4 consecutive fp32 (=16B) via cp.async.cg.
                #   threads_per_token = qhead_per_kvhead / 4
                #   tok = lane // threads_per_token
                #   qhead_chunk = lane % threads_per_token
                #   elem_off = q_idx[tok] * H_q + qhead_chunk * 4
                nheads_q = nheads_kv * self.qhead_per_kvhead
                nheads_q_i32 = Int32(nheads_q)
                nheads_kv_i32 = Int32(nheads_kv)
                q_token_base = (
                    seqlen.offset_q
                    if const_expr(self.is_varlen_q)
                    else batch_idx * seqlen.seqlen_q
                )
                q_tile_batch_base = q_token_base * nheads_kv_i32 + head_idx_kv
                stats_src_lse = mLSE_cur
                stats_src_dPsum = mdPsum_cur
                stats_row_base_i32 = Int32(0)
                stats_row_stride_i32 = nheads_q_i32
                if const_expr(self.is_varlen_q):
                    stats_row_base_i32 = (
                        seqlen.padded_offset_q * nheads_q_i32
                        + head_idx_kv * Int32(self.qhead_per_kvhead)
                    )
                    stats_src_lse = mLSE
                    stats_src_dPsum = mdPsum
                if const_expr(not self.use_scalar_stats_load):
                    stats_threads_per_token = Int32(self.qhead_per_kvhead // 4)
                    tok_stats = load_warp_lane // stats_threads_per_token
                    qhead_chunk_stats = load_warp_lane % stats_threads_per_token

                if num_q_tiles > 0:
                    k2q_count_last = k2q_count - Int32(1)
                    meta_group_slot = Int32(0)
                    # Prologue: iter 0 loads K, Q, LSE, V, dO, and dPsum.
                    q_idx_tile_prologue = None
                    load_src_tile_prologue = None
                    # SMEM metadata is needed when Q and dO producers are
                    # split across warps.
                    use_smem_sparse_metadata = const_expr(
                        split_sparse_qdo_producers
                    )
                    if const_expr(should_load_Q or should_load_dO):
                        if const_expr(use_smem_sparse_metadata):
                            q_idx_tile_prologue = sSparseLoadQIdx[
                                None, Int32(0), Int32(0)
                            ]
                            load_src_tile_prologue = sSparseLoadRowBase[
                                None, Int32(0), Int32(0)
                            ]
                        else:
                            q_idx_tile_prologue = cute.make_rmem_tensor(self.q_per_tile, Int32)
                            load_src_tile_prologue = cute.make_rmem_tensor(
                                self.q_per_tile, Int32
                            )
                        q_idx_tile_smem = (
                            sSparseLoadQIdx[None, Int32(0), meta_group_slot]
                            if const_expr(split_sparse_qdo_producers)
                            else None
                        )
                        row_base_tile_smem = (
                            sSparseLoadRowBase[None, Int32(0), meta_group_slot]
                            if const_expr(split_sparse_qdo_producers)
                            else None
                        )
                        # Keep the prologue metadata flow aligned with the
                        # split-producer mainloop: when Q and dO use
                        # separate load warps, only the Q-side warp writes
                        # shared q_idx/row_base metadata and the dO-side
                        # warp only consumes it after the barrier.
                        if const_expr((not split_sparse_qdo_producers) or should_load_Q):
                            for tok in cutlass.range(self.q_per_tile, unroll_full=True):
                                qi_prologue = Int32(tok)
                                qi_clamped_prologue = cutlass.min(
                                    qi_prologue, k2q_count_last
                                )
                                q_idx_prologue = self._load_sparse_q_idx(
                                    mK2qIndices,
                                    head_idx_kv,
                                    row_start_sparse,
                                    qi_clamped_prologue,
                                )
                                q_idx_tile_prologue[tok] = q_idx_prologue
                                load_src = (
                                    q_tile_batch_base + q_idx_prologue * nheads_kv_i32
                                )
                                load_src_tile_prologue[tok] = load_src
                                if const_expr(split_sparse_qdo_producers):
                                    q_idx_tile_smem[tok] = q_idx_prologue
                                    row_base_tile_smem[tok] = load_src
                    elif const_expr(split_sparse_qdo_producers):
                        q_idx_tile_prologue = sSparseLoadQIdx[None, Int32(0), meta_group_slot]
                        load_src_tile_prologue = sSparseLoadRowBase[
                            None, Int32(0), meta_group_slot
                        ]
                    if const_expr(split_sparse_qdo_producers):
                        cute.arch.fence_view_async_shared()
                        self.load_qdo_sync_barrier.arrive_and_wait()
                    elif const_expr(use_smem_sparse_metadata):
                        cute.arch.sync_warp()
                    if const_expr(should_load_Q):
                        pipeline_Q.producer_acquire(
                            producer_state_Q_LSE,
                            extra_tx_count=self.tma_copy_bytes["K"],
                        )
                        mbar_Q_prologue = pipeline_Q.producer_get_barrier(
                            producer_state_Q_LSE
                        )
                        load_K(tma_bar_ptr=mbar_Q_prologue)
                        stage_base_prologue = (
                            producer_state_Q_LSE.index * self.num_subtiles_per_stage
                        )
                        for tok in cutlass.range_constexpr(self.q_per_tile):
                            load_Q_fn_k0(
                                src_idx=load_src_tile_prologue[tok],
                                dst_idx=stage_base_prologue + Int32(tok),
                                tma_bar_ptr=mbar_Q_prologue,
                            )
                            if const_expr(self.k_subtiles > 1):
                                load_Q_fn_k1(
                                    src_idx=load_src_tile_prologue[tok],
                                    dst_idx=stage_base_prologue
                                    + Int32(self.q_per_tile + tok),
                                    tma_bar_ptr=mbar_Q_prologue,
                                )
                        pipeline_Q.producer_commit(producer_state_Q_LSE)
                        if const_expr(self.stage_p_stats):
                            producer_state_Q_LSE.advance()

                        # LSE scatter gather (prologue, q_iter=0). Pointer
                        # cast with assumed_align=16 for cp.async.cg 128-bit.
                        if const_expr(self.stage_p_stats):
                            pipeline_LSE.producer_acquire(producer_state_LSE)
                            sLSE_stage_prologue = sLSE[None, producer_state_LSE.index]
                        else:
                            pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                            sLSE_stage_prologue = sLSE[None, producer_state_Q_LSE.index]
                        if const_expr(self.stage_p_stats):
                            self._load_sparse_p_stats_vec(
                                mPQuantScale,
                                mK2qQSplitIndices,
                                sLSE_stage_prologue,
                                head_idx_kv,
                                row_start_sparse,
                                Int32(0),
                                k2q_count,
                                seqlen.offset_q,
                                load_warp_lane,
                                async_stats_atom,
                            )
                        elif const_expr(self.use_scalar_stats_load):
                            self._load_sparse_stats_scalar(
                                stats_src_lse,
                                sLSE_stage_prologue,
                                q_idx_tile_prologue,
                                stats_row_base_i32,
                                stats_row_stride_i32,
                                load_warp_lane,
                            )
                        else:
                            self._load_sparse_stats_vec(
                                stats_src_lse,
                                sLSE_stage_prologue,
                                q_idx_tile_prologue,
                                stats_row_base_i32,
                                stats_row_stride_i32,
                                load_warp_lane,
                                tok_stats,
                                qhead_chunk_stats,
                                async_stats_atom,
                            )
                        cute.arch.sync_warp()
                        if const_expr(self.stage_p_stats):
                            pipeline_LSE.producer_commit(producer_state_LSE)
                            producer_state_LSE.advance()
                        else:
                            pipeline_LSE.producer_commit(producer_state_Q_LSE)
                            producer_state_Q_LSE.advance()

                    # dO scatter (+ V bundled). pipeline_dO base already
                    # includes dO bytes; add V as extra.
                    if const_expr(should_load_dO):
                        pipeline_dO.producer_acquire(
                            producer_state_dO_dPsum,
                            extra_tx_count=self.tma_copy_bytes["V"],
                        )
                        mbar_dO_prologue = pipeline_dO.producer_get_barrier(
                            producer_state_dO_dPsum
                        )
                        load_V(tma_bar_ptr=mbar_dO_prologue)
                        stage_base_dO_prologue = (
                            producer_state_dO_dPsum.index * self.num_subtiles_per_stage
                        )
                        for tok in cutlass.range_constexpr(self.q_per_tile):
                            load_dO_fn_k0(
                                src_idx=load_src_tile_prologue[tok],
                                dst_idx=stage_base_dO_prologue + Int32(tok),
                                tma_bar_ptr=mbar_dO_prologue,
                            )
                            if const_expr(self.k_subtiles > 1):
                                load_dO_fn_k1(
                                    src_idx=load_src_tile_prologue[tok],
                                    dst_idx=stage_base_dO_prologue
                                    + Int32(self.q_per_tile + tok),
                                    tma_bar_ptr=mbar_dO_prologue,
                                )
                        pipeline_dO.producer_commit(producer_state_dO_dPsum)
                        if const_expr(not should_load_Q):
                            # In dual-load mode, the dO warp maintains the
                            # dO pipeline state independently from the
                            # Q-side dPsum producer.
                            producer_state_dO_dPsum.advance()

                    # dPsum scatter gather (prologue, q_iter=0)
                    if const_expr(should_load_Q):
                        pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                        sdPsum_stage_prologue = sdPsum[
                            None, producer_state_dO_dPsum.index
                        ]
                        if const_expr(self.use_scalar_stats_load):
                            self._load_sparse_stats_scalar(
                                stats_src_dPsum,
                                sdPsum_stage_prologue,
                                q_idx_tile_prologue,
                                stats_row_base_i32,
                                stats_row_stride_i32,
                                load_warp_lane,
                            )
                        else:
                            self._load_sparse_stats_vec(
                                stats_src_dPsum,
                                sdPsum_stage_prologue,
                                q_idx_tile_prologue,
                                stats_row_base_i32,
                                stats_row_stride_i32,
                                load_warp_lane,
                                tok_stats,
                                qhead_chunk_stats,
                                async_stats_atom,
                            )
                        cute.arch.sync_warp()
                        pipeline_dPsum.producer_commit(producer_state_dO_dPsum)
                        producer_state_dO_dPsum.advance()

                    # Mainloop: process q_iter 1 through num_q_tiles - 1.
                    if const_expr(split_sparse_qdo_producers):
                        q_group_base = Int32(1)
                        while q_group_base < num_q_tiles:
                            meta_group_slot = Int32(1) - meta_group_slot
                            if const_expr(should_load_Q):
                                for q_group_off in cutlass.range_constexpr(
                                    self.sparse_load_meta_group_iters
                                ):
                                    q_iter_meta = q_group_base + Int32(q_group_off)
                                    q_iter_meta_c = cutlass.min(
                                        q_iter_meta, num_q_tiles - Int32(1)
                                    )
                                    q_idx_tile_smem = sSparseLoadQIdx[
                                        None, q_group_off, meta_group_slot
                                    ]
                                    row_base_tile_smem = sSparseLoadRowBase[
                                        None, q_group_off, meta_group_slot
                                    ]
                                    for tok in cutlass.range(self.q_per_tile, unroll_full=True):
                                        qi = (
                                            q_iter_meta_c * Int32(self.q_per_tile) + Int32(tok)
                                        )
                                        qi_clamped = cutlass.min(qi, k2q_count_last)
                                        q_idx = self._load_sparse_q_idx(
                                            mK2qIndices,
                                            head_idx_kv,
                                            row_start_sparse,
                                            qi_clamped,
                                        )
                                        load_src = q_tile_batch_base + q_idx * nheads_kv_i32
                                        q_idx_tile_smem[tok] = q_idx
                                        row_base_tile_smem[tok] = load_src
                            cute.arch.fence_view_async_shared()
                            self.load_qdo_sync_barrier.arrive_and_wait()

                            for q_group_off in cutlass.range_constexpr(
                                self.sparse_load_meta_group_iters
                            ):
                                q_iter = q_group_base + Int32(q_group_off)
                                if q_iter < num_q_tiles:
                                    if const_expr(should_load_Q):
                                        pipeline_Q.producer_acquire(producer_state_Q_LSE)
                                        mbar_Q = pipeline_Q.producer_get_barrier(
                                            producer_state_Q_LSE
                                        )
                                        stage_base = (
                                            producer_state_Q_LSE.index
                                            * self.num_subtiles_per_stage
                                        )
                                        for tok in cutlass.range_constexpr(self.q_per_tile):
                                            m_tile_idx_cur = sSparseLoadRowBase[
                                                tok, q_group_off, meta_group_slot
                                            ]
                                            load_Q_fn_k0(
                                                src_idx=m_tile_idx_cur,
                                                dst_idx=stage_base + Int32(tok),
                                                tma_bar_ptr=mbar_Q,
                                            )
                                            if const_expr(self.k_subtiles > 1):
                                                load_Q_fn_k1(
                                                    src_idx=m_tile_idx_cur,
                                                    dst_idx=stage_base
                                                    + Int32(self.q_per_tile + tok),
                                                    tma_bar_ptr=mbar_Q,
                                                )
                                        pipeline_Q.producer_commit(producer_state_Q_LSE)
                                        if const_expr(self.stage_p_stats):
                                            producer_state_Q_LSE.advance()

                                        if const_expr(self.stage_p_stats):
                                            pipeline_LSE.producer_acquire(producer_state_LSE)
                                            sLSE_stage = sLSE[None, producer_state_LSE.index]
                                        else:
                                            pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                                            sLSE_stage = sLSE[None, producer_state_Q_LSE.index]
                                        if const_expr(self.stage_p_stats):
                                            self._load_sparse_p_stats_vec(
                                                mPQuantScale,
                                                mK2qQSplitIndices,
                                                sLSE_stage,
                                                head_idx_kv,
                                                row_start_sparse,
                                                q_iter,
                                                k2q_count,
                                                seqlen.offset_q,
                                                load_warp_lane,
                                                async_stats_atom,
                                            )
                                        elif const_expr(self.use_scalar_stats_load):
                                            self._load_sparse_stats_scalar(
                                                stats_src_lse,
                                                sLSE_stage,
                                                sSparseLoadQIdx[
                                                    None, q_group_off, meta_group_slot
                                                ],
                                                stats_row_base_i32,
                                                stats_row_stride_i32,
                                                load_warp_lane,
                                            )
                                        else:
                                            self._load_sparse_stats_vec(
                                                stats_src_lse,
                                                sLSE_stage,
                                                sSparseLoadQIdx[
                                                    None, q_group_off, meta_group_slot
                                                ],
                                                stats_row_base_i32,
                                                stats_row_stride_i32,
                                                load_warp_lane,
                                                tok_stats,
                                                qhead_chunk_stats,
                                                async_stats_atom,
                                            )
                                        cute.arch.sync_warp()
                                        if const_expr(self.stage_p_stats):
                                            pipeline_LSE.producer_commit(producer_state_LSE)
                                            producer_state_LSE.advance()
                                        else:
                                            pipeline_LSE.producer_commit(producer_state_Q_LSE)
                                            producer_state_Q_LSE.advance()

                                    if const_expr(should_load_dO):
                                        pipeline_dO.producer_acquire(
                                            producer_state_dO_dPsum
                                        )
                                        mbar_dO = pipeline_dO.producer_get_barrier(
                                            producer_state_dO_dPsum
                                        )
                                        stage_base_dO = (
                                            producer_state_dO_dPsum.index
                                            * self.num_subtiles_per_stage
                                        )
                                        for tok in cutlass.range_constexpr(self.q_per_tile):
                                            m_tile_idx_cur = sSparseLoadRowBase[
                                                tok, q_group_off, meta_group_slot
                                            ]
                                            load_dO_fn_k0(
                                                src_idx=m_tile_idx_cur,
                                                dst_idx=stage_base_dO + Int32(tok),
                                                tma_bar_ptr=mbar_dO,
                                            )
                                            if const_expr(self.k_subtiles > 1):
                                                load_dO_fn_k1(
                                                    src_idx=m_tile_idx_cur,
                                                    dst_idx=stage_base_dO
                                                    + Int32(self.q_per_tile + tok),
                                                    tma_bar_ptr=mbar_dO,
                                                )
                                        pipeline_dO.producer_commit(producer_state_dO_dPsum)
                                        producer_state_dO_dPsum.advance()

                                    if const_expr(should_load_Q):
                                        pipeline_dPsum.producer_acquire(
                                            producer_state_dO_dPsum
                                        )
                                        sdPsum_stage = sdPsum[
                                            None, producer_state_dO_dPsum.index
                                        ]
                                        if const_expr(self.use_scalar_stats_load):
                                            self._load_sparse_stats_scalar(
                                                stats_src_dPsum,
                                                sdPsum_stage,
                                                sSparseLoadQIdx[
                                                    None, q_group_off, meta_group_slot
                                                ],
                                                stats_row_base_i32,
                                                stats_row_stride_i32,
                                                load_warp_lane,
                                            )
                                        else:
                                            self._load_sparse_stats_vec(
                                                stats_src_dPsum,
                                                sdPsum_stage,
                                                sSparseLoadQIdx[
                                                    None, q_group_off, meta_group_slot
                                                ],
                                                stats_row_base_i32,
                                                stats_row_stride_i32,
                                                load_warp_lane,
                                                tok_stats,
                                                qhead_chunk_stats,
                                                async_stats_atom,
                                            )
                                        cute.arch.sync_warp()
                                        pipeline_dPsum.producer_commit(
                                            producer_state_dO_dPsum
                                        )
                                        producer_state_dO_dPsum.advance()
                            q_group_base += Int32(self.sparse_load_meta_group_iters)
                    else:
                        for q_iter in cutlass.range(1, num_q_tiles, unroll=1):
                            if const_expr(use_smem_sparse_metadata):
                                q_idx_tile = sSparseLoadQIdx[None, Int32(0), Int32(0)]
                                load_src_tile = sSparseLoadRowBase[None, Int32(0), Int32(0)]
                            else:
                                q_idx_tile = cute.make_rmem_tensor(self.q_per_tile, Int32)
                                load_src_tile = cute.make_rmem_tensor(self.q_per_tile, Int32)
                            for tok in cutlass.range_constexpr(self.q_per_tile):
                                qi = q_iter * Int32(self.q_per_tile) + Int32(tok)
                                qi_clamped = cutlass.min(qi, k2q_count_last)
                                q_idx = self._load_sparse_q_idx(
                                    mK2qIndices,
                                    head_idx_kv,
                                    row_start_sparse,
                                    qi_clamped,
                                )
                                q_idx_tile[tok] = q_idx
                                load_src = q_tile_batch_base + q_idx * nheads_kv_i32
                                load_src_tile[tok] = load_src
                            if const_expr(use_smem_sparse_metadata):
                                cute.arch.sync_warp()
                            if const_expr(should_load_Q):
                                pipeline_Q.producer_acquire(producer_state_Q_LSE)
                                mbar_Q = pipeline_Q.producer_get_barrier(producer_state_Q_LSE)
                                stage_base = (
                                    producer_state_Q_LSE.index
                                    * self.num_subtiles_per_stage
                                )
                                for tok in cutlass.range_constexpr(self.q_per_tile):
                                    load_Q_fn_k0(
                                        src_idx=load_src_tile[tok],
                                        dst_idx=stage_base + Int32(tok),
                                        tma_bar_ptr=mbar_Q,
                                    )
                                    if const_expr(self.k_subtiles > 1):
                                        load_Q_fn_k1(
                                            src_idx=load_src_tile[tok],
                                            dst_idx=stage_base
                                            + Int32(self.q_per_tile + tok),
                                            tma_bar_ptr=mbar_Q,
                                        )
                                pipeline_Q.producer_commit(producer_state_Q_LSE)
                                if const_expr(self.stage_p_stats):
                                    producer_state_Q_LSE.advance()

                                if const_expr(self.stage_p_stats):
                                    pipeline_LSE.producer_acquire(producer_state_LSE)
                                    sLSE_stage = sLSE[None, producer_state_LSE.index]
                                else:
                                    pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                                    sLSE_stage = sLSE[None, producer_state_Q_LSE.index]
                                if const_expr(self.stage_p_stats):
                                    self._load_sparse_p_stats_vec(
                                        mPQuantScale,
                                        mK2qQSplitIndices,
                                        sLSE_stage,
                                        head_idx_kv,
                                        row_start_sparse,
                                        q_iter,
                                        k2q_count,
                                        seqlen.offset_q,
                                        load_warp_lane,
                                        async_stats_atom,
                                    )
                                elif const_expr(self.use_scalar_stats_load):
                                    self._load_sparse_stats_scalar(
                                        stats_src_lse,
                                        sLSE_stage,
                                        q_idx_tile,
                                        stats_row_base_i32,
                                        stats_row_stride_i32,
                                        load_warp_lane,
                                    )
                                else:
                                    self._load_sparse_stats_vec(
                                        stats_src_lse,
                                        sLSE_stage,
                                        q_idx_tile,
                                        stats_row_base_i32,
                                        stats_row_stride_i32,
                                        load_warp_lane,
                                        tok_stats,
                                        qhead_chunk_stats,
                                        async_stats_atom,
                                    )
                                cute.arch.sync_warp()
                                if const_expr(self.stage_p_stats):
                                    pipeline_LSE.producer_commit(producer_state_LSE)
                                    producer_state_LSE.advance()
                                else:
                                    pipeline_LSE.producer_commit(producer_state_Q_LSE)
                                    producer_state_Q_LSE.advance()

                            if const_expr(should_load_dO):
                                pipeline_dO.producer_acquire(producer_state_dO_dPsum)
                                mbar_dO = pipeline_dO.producer_get_barrier(
                                    producer_state_dO_dPsum
                                )
                                stage_base_dO = (
                                    producer_state_dO_dPsum.index
                                    * self.num_subtiles_per_stage
                                )
                                for tok in cutlass.range_constexpr(self.q_per_tile):
                                    load_dO_fn_k0(
                                        src_idx=load_src_tile[tok],
                                        dst_idx=stage_base_dO + Int32(tok),
                                        tma_bar_ptr=mbar_dO,
                                    )
                                    if const_expr(self.k_subtiles > 1):
                                        load_dO_fn_k1(
                                            src_idx=load_src_tile[tok],
                                            dst_idx=stage_base_dO
                                            + Int32(self.q_per_tile + tok),
                                            tma_bar_ptr=mbar_dO,
                                        )
                                pipeline_dO.producer_commit(producer_state_dO_dPsum)
                                if const_expr(not should_load_Q):
                                    producer_state_dO_dPsum.advance()

                            if const_expr(should_load_Q):
                                pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                                sdPsum_stage = sdPsum[None, producer_state_dO_dPsum.index]
                                if const_expr(self.use_scalar_stats_load):
                                    self._load_sparse_stats_scalar(
                                        stats_src_dPsum,
                                        sdPsum_stage,
                                        q_idx_tile,
                                        stats_row_base_i32,
                                        stats_row_stride_i32,
                                        load_warp_lane,
                                    )
                                else:
                                    self._load_sparse_stats_vec(
                                        stats_src_dPsum,
                                        sdPsum_stage,
                                        q_idx_tile,
                                        stats_row_base_i32,
                                        stats_row_stride_i32,
                                        load_warp_lane,
                                        tok_stats,
                                        qhead_chunk_stats,
                                        async_stats_atom,
                                    )
                                cute.arch.sync_warp()
                                pipeline_dPsum.producer_commit(producer_state_dO_dPsum)
                                producer_state_dO_dPsum.advance()
                if const_expr(split_sparse_qdo_producers):
                    if process_tile:
                        # The prologue-only sparse path reuses
                        # sSparseLoadQIdx/RowBase across work tiles without
                        # any later metadata-group barrier. Rendezvous the
                        # Q and dO producer warps here before the next work
                        # tile can overwrite that shared metadata.
                        self.load_qdo_sync_barrier.arrive_and_wait()
                if const_expr(should_load_Q):
                    if const_expr(self.stage_p_stats):
                        pipeline_Q.producer_tail(producer_state_Q_LSE)
                        pipeline_LSE.producer_tail(producer_state_LSE)
                    else:
                        pipeline_Q.producer_tail(producer_state_Q_LSE.clone())
                        pipeline_LSE.producer_tail(producer_state_Q_LSE.clone())
                    pipeline_dPsum.producer_tail(producer_state_dO_dPsum.clone())
                if const_expr(should_load_dO):
                    pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
    @cute.jit
    def mma(
        self,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        sQ: cute.Tensor,
        sQt: cute.Tensor,
        sK: cute.Tensor,
        sKt: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOt: cute.Tensor,
        tP: cute.Tensor,
        sdS: cute.Tensor,
        tdS: cute.Tensor,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdQtdQ: cute.Tensor,
        pipeline_Q: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_dQ: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        is_leader_cta: cutlass.Boolean,
        mK2qCounts: Optional[cute.Tensor] = None,
        mSchedulerMetadata: Optional[cute.Tensor] = None,
        mWorkCount: Optional[cute.Tensor] = None,
    ):
        # Keep SMEM/TMEM partitioning inside the MMA role to preserve the tuned
        # register live ranges after warp specialization.
        # S = K @ Q.T
        tSrK = tiled_mma_S.make_fragment_A(sK)
        tSrQ = tiled_mma_S.make_fragment_B(sQ)
        # dP = V @ dOt.T
        tdPrV = tiled_mma_dP.make_fragment_A(sV)
        tdPrdOt = tiled_mma_dP.make_fragment_B(sdOt)
        # dK = dS.T @ Q
        tdKrdS = tiled_mma_dK.make_fragment_A(tdS)
        tdKrQ = tiled_mma_dK.make_fragment_B(sQt)
        # dQ = dS @ K
        tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
        tdQrK = tiled_mma_dQ.make_fragment_B(sKt)
        # dV = P @ dO.T
        tdVrdO = tiled_mma_dV.make_fragment_B(sdO)
        tdVrP = tiled_mma_dV.make_fragment_A(tP)

        mma_qk_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_S,
            tStS,
            tSrK,
            tSrQ,
            sA=sK,
            sB=sQ,
            zero_init=True,
            cta_group=self.cta_group_size,
        )
        mma_dov_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dP,
            tdPtdP,
            tdPrV,
            tdPrdOt,
            sA=sV,
            sB=sdOt,
            zero_init=True,
            cta_group=self.cta_group_size,
        )
        mma_pdo_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dV,
            tdVtdV,
            tdVrP,
            tdVrdO,
            sA=None,
            sB=sdO,
            tA_addr=self.tmem_P_offset,
            cta_group=self.cta_group_size,
        )
        mma_dsk_fn = partial(
            gemm_w_idx,
            tiled_mma_dQ,
            tdQtdQ,
            tdQrdS,
            tdQrK,
            zero_init=True,
            num_unroll_groups=1,
        )
        mma_dsq_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dK,
            tdKtdK,
            tdKrdS,
            tdKrQ,
            sA=None,
            sB=sQt,
            tA_addr=self.tmem_dS_offset,
            cta_group=self.cta_group_size,
        )

        cta_group = pipeline_S_P.cta_group

        pipeline_Q_consumer = pipeline_Q.make_consumer()
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        producer_phase_S_P = Int32(1)
        producer_phase_dP_dQ = Int32(1)
        consumer_state_dS = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        producer_phase_dKV = Int32(1)
        work_idx = cute.arch.block_idx()[0]
        for _ in cutlass.range_constexpr(1):
            n_block, head_idx, batch_idx, _, k2q_count_mma = self._resolve_sparse_work(
                work_idx,
                mK2qCounts,
                mSchedulerMetadata,
                mWorkCount,
            )
            seqlen = SeqlenInfoCls(batch_idx)  # must be seqlen_k
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )
            num_q_tiles_mma = cute.ceil_div(k2q_count_mma, self.q_per_tile)
            m_block_max = m_block_min + num_q_tiles_mma

            block_iter_count = m_block_max - m_block_min
            process_tile = block_iter_count > Int32(0)

            if is_leader_cta and process_tile:
                accumulate_dK = False
                # Prologue
                # 1. S  = Q0 @ K.T
                # 2. dP = V @ dOt.T
                # 3. dV = P @ dO

                # 1) S = K @ Q
                handle_Q = pipeline_Q_consumer.wait_and_advance()
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_S_P)
                mma_qk_fn(B_idx=handle_Q.index)
                pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                # 2) dP = V @ dOt.T
                pipeline_dO.consumer_wait(consumer_state_dO)
                pipeline_dP.sync_object_empty.wait(0, producer_phase_dP_dQ)
                pipeline_dQ.sync_object_empty.wait(0, producer_phase_dP_dQ)
                mma_dov_fn(B_idx=consumer_state_dO.index)
                pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                producer_phase_S_P ^= 1
                producer_phase_dP_dQ ^= 1
                # 3) dV = P.T @ dO
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_S_P)
                mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
                pipeline_dO.consumer_release(consumer_state_dO)
                consumer_state_dO.advance()

                # Mainloop
                # 1. S  = K    @ Q.T
                # 2. dQ = dS   @ K
                # 3. dK = dS.T @ Q
                # 4. dP = V    @ dOt.T
                # 5. dV = P.T  @ dO

                main_loop_iters = m_block_max - m_block_min - 1

                handle_Q_next = handle_Q
                for _ in cutlass.range(main_loop_iters, unroll=1):
                    # (1) S.T = K @ Q.T
                    handle_Q_next = pipeline_Q_consumer.wait_and_advance()
                    mma_qk_fn(B_idx=handle_Q_next.index)
                    pipeline_S_P.sync_object_full.arrive(
                        0, pipeline_S_P.producer_mask, cta_group
                    )

                    # (2) dK += dS.T @ Q
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                    accumulate_dK = True
                    handle_Q.release()

                    # (3) dQ = dS @ K
                    mma_dsk_fn()
                    pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                    pipeline_dS.consumer_release(consumer_state_dS)
                    consumer_state_dS.advance()

                    # (4) dP = V @ dO.T
                    pipeline_dO.consumer_wait(consumer_state_dO)
                    pipeline_dQ.sync_object_empty.wait(0, producer_phase_dP_dQ)
                    mma_dov_fn(B_idx=consumer_state_dO.index)
                    pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                    # (5) dV += P.T @ dO
                    producer_phase_S_P ^= 1
                    producer_phase_dP_dQ ^= 1
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_S_P)
                    mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()

                    handle_Q = handle_Q_next

                # Publish the final dV completion. Compute drains this extra
                # S/P transaction separately from the dP/dQ transaction ring.
                pipeline_S_P.sync_object_full.arrive(
                    0, pipeline_S_P.producer_mask, cta_group
                )
                producer_phase_S_P ^= 1

                # Signal that dV is ready for the epilogue.
                pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

                # Tail: finish dK and dQ.
                # 1) dK += dS.T @ Q
                pipeline_dS.consumer_wait(consumer_state_dS)
                mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                # Signal that dK is ready for the epilogue.
                pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                producer_phase_dKV ^= 1

                # 2) dQ = dS @ K
                mma_dsk_fn()
                pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                handle_Q.release()
                pipeline_dS.consumer_release(consumer_state_dS)
                consumer_state_dS.advance()

                # Drain both dKV slots before TMEM teardown can overwrite the
                # accumulators.
                pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)


    @cute.jit
    def split_wg(
        self,
        t: cute.Tensor,
        wg_idx: cutlass.Int32,
        num_wg: cutlass.Constexpr[int],
    ):
        reduced_shape = cute.product_each(t.shape)
        rank = len(reduced_shape)
        if const_expr(reduced_shape[1] > 1):
            assert rank >= 2, "Need rank >= 2 for t in split_wg"
            t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1] // num_wg))
            coord = (None, (None, wg_idx)) + (None,) * (rank - 2)
        else:
            assert rank >= 3, "Need rank >= 3 for t in split_wg"
            if const_expr(rank == 3):
                t = cute.logical_divide(
                    t, (reduced_shape[0], reduced_shape[1], reduced_shape[2] // num_wg)
                )
                coord = (
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 3)
            else:
                t = cute.logical_divide(
                    t,
                    (
                        reduced_shape[0],
                        reduced_shape[1],
                        reduced_shape[2],
                        reduced_shape[3] // num_wg,
                    ),
                )
                coord = (
                    None,
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 4)
        return t[coord]

    @cute.jit
    def _load_sparse_q_idx_tile(
        self,
        q_idx_tile: cute.Tensor,
        head_idx: Int32,
        row_start: Int32,
        iter_idx: Int32,
        k2q_count: Int32,
        mK2qIndices: cute.Tensor,
    ):
        k2q_count_last = k2q_count - Int32(1)
        qi_base = iter_idx * Int32(self.q_per_tile)
        for tok in cutlass.range(self.q_per_tile, unroll_full=True):
            qi = qi_base + Int32(tok)
            qi_clamped = cutlass.min(qi, k2q_count_last)
            q_idx_tile[tok] = self._load_sparse_q_idx(
                mK2qIndices,
                head_idx,
                row_start,
                qi_clamped,
            )

    @cute.jit
    def _apply_sparse_gqa16_causal_mask(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        n_block: Int32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        valid_tok_count: Int32,
        q_idx_lane: Int32,
        masked_tok_count: Int32,
    ):
        kv_block_col_start = n_block * Int32(self.tile_n)
        causal_q_offset = seqlen_k - seqlen_q
        for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
            row_idx = tScS_t2r[i][1]
            tok_idx = row_idx // Int32(self.qhead_per_kvhead)
            kv_idx = kv_block_col_start + tScS_t2r[i][0]
            q_idx = cute.arch.shuffle_sync(q_idx_lane, tok_idx)
            acc_S[i] = -Float32.inf if tok_idx >= valid_tok_count else acc_S[i]
            acc_S[i] = -Float32.inf if kv_idx >= seqlen_k else acc_S[i]
            if tok_idx < masked_tok_count:
                acc_S[i] = (
                    -Float32.inf
                    if kv_idx > q_idx + causal_q_offset
                    else acc_S[i]
                )

    @cute.jit
    def _load_sparse_stats_vec(
        self,
        src_cur: cute.Tensor,
        dst_stage: cute.Tensor,
        q_idx_tile: cute.Tensor,
        packed_row_base: Int32,
        packed_stride_i32: Int32,
        load_warp_lane: Int32,
        tok_lane: Int32,
        qhead_chunk_lane: Int32,
        async_stats_atom: cute.CopyAtom,
    ):
        q_idx = q_idx_tile[tok_lane]
        src_off_bytes = (
            packed_row_base + q_idx * packed_stride_i32 + qhead_chunk_lane * Int32(4)
        ) * Int32(4)
        src_ptr = cute.make_ptr(
            Float32,
            src_cur.iterator.toint() + src_off_bytes,
            mem_space=src_cur.iterator.memspace,
            assumed_align=16,
        )
        src = cute.make_tensor(src_ptr, cute.make_layout((4,), stride=(1,)))
        dst_ptr = cute.make_ptr(
            Float32,
            dst_stage.iterator.toint() + load_warp_lane * Int32(16),
            mem_space=dst_stage.iterator.memspace,
            assumed_align=16,
        )
        dst = cute.make_tensor(dst_ptr, cute.make_layout((4,), stride=(1,)))
        cute.copy(async_stats_atom, src, dst)

    @cute.jit
    def _load_sparse_stats_scalar(
        self,
        src_cur: cute.Tensor,
        dst_stage: cute.Tensor,
        q_idx_tile: cute.Tensor,
        packed_row_base: Int32,
        packed_stride_i32: Int32,
        load_warp_lane: Int32,
    ):
        for row_iter_c in cutlass.range_constexpr(self.tile_m // cute.arch.WARP_SIZE):
            row_iter = Int32(row_iter_c)
            row_idx = load_warp_lane + row_iter * Int32(cute.arch.WARP_SIZE)
            tok_stats = row_idx // Int32(self.qhead_per_kvhead)
            h_stats = row_idx % Int32(self.qhead_per_kvhead)
            src_ptr = cute.make_ptr(
                Float32,
                src_cur.iterator.toint()
                + (
                    packed_row_base
                    + q_idx_tile[tok_stats] * packed_stride_i32
                    + h_stats
                )
                * Int32(4),
                mem_space=src_cur.iterator.memspace,
                assumed_align=4,
            )
            dst_ptr = cute.make_ptr(
                Float32,
                dst_stage.iterator.toint() + row_idx * Int32(4),
                mem_space=dst_stage.iterator.memspace,
                assumed_align=4,
            )
            cute.make_tensor(dst_ptr, cute.make_layout((1,), stride=(1,)))[0] = (
                cute.make_tensor(src_ptr, cute.make_layout((1,), stride=(1,)))[0]
            )

    @cute.jit
    def _load_sparse_stats_rowbase_direct(
        self,
        src_cur: cute.Tensor,
        dst_stage: cute.Tensor,
        row_base_tile: cute.Tensor,
        load_warp_lane: Int32,
        tok_lane: Int32,
    ):
        row_base = row_base_tile[tok_lane]
        src_ptr = cute.make_ptr(
            Float32,
            src_cur.iterator.toint() + row_base * Int32(4),
            mem_space=src_cur.iterator.memspace,
            assumed_align=16,
        )
        src = cute.make_tensor(src_ptr, cute.make_layout((4,), stride=(1,)))
        vals = src.load()
        dst_ptr = cute.make_ptr(
            Float32,
            dst_stage.iterator.toint() + load_warp_lane * Int32(16),
            mem_space=dst_stage.iterator.memspace,
            assumed_align=16,
        )
        dst = cute.make_tensor(dst_ptr, cute.make_layout((4,), stride=(1,)))
        for i in cutlass.range_constexpr(4):
            dst[Int32(i)] = vals[Int32(i)]

    @cute.jit
    def compute_loop(
        self,
        thr_mma_S: cute.ThrMma,
        thr_mma_dP: cute.ThrMma,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        mdVaccum: cute.Tensor,
        mdKaccum: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        mdV_raw: cute.Tensor,
        mdK_raw: cute.Tensor,
        tma_atom_dV: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        mDkvOwnerCounts: cute.Tensor,
        mDkvWriterRank: Optional[cute.Tensor],
        mDkvSemaphore: Optional[cute.Tensor],
        sdS: cute.Tensor,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        sdV: Optional[cute.Tensor],
        sdK: Optional[cute.Tensor],
        sdV_out: Optional[cute.Tensor],
        sdK_out: Optional[cute.Tensor],
        mK2qIndices: Optional[cute.Tensor] = None,
        mK2qCounts: Optional[cute.Tensor] = None,
        mK2qQSplitIndices: Optional[cute.Tensor] = None,
        mPQuantScale: Optional[cute.Tensor] = None,
        mSchedulerMetadata: Optional[cute.Tensor] = None,
        mWorkCount: Optional[cute.Tensor] = None,
        mFragmentIndices: Optional[cute.Tensor] = None,
    ):
        sLSE_2D = cute.make_tensor(
            sLSE.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.LSE_stage),
                stride=(
                    1,
                    0,
                    cute.round_up(self.tile_m * self.num_lse_components, 64),
                ),
            ),
        )
        sdPsum_2D = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.dO_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sLSE_2D = layout_utils.transpose_view(sLSE_2D)
        sdPsum_2D = layout_utils.transpose_view(sdPsum_2D)

        # Local thread index across the eight compute warps.
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        dp_idx = tidx % 128
        lane_idx = cute.arch.lane_idx()
        num_wg = len(self.compute_warp_ids) // 4  # 2

        tileP_f32_like = self.cta_tiler[1] // 32 * self.v_dtype.width
        # tStS has shape ((128, 128), 1, 1), tStP has shape ((128, 64), 1, 1)
        # tP overlap with tS
        tStP = cute.composition(tStS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        # Preserve the original TMEM base while reinterpreting the P layout.
        tStP = cute.make_tensor(tStS.iterator, tStP.layout)
        tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
        tScP = cute.composition(tScS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        # tdS overlap with tdP
        tdPtdS = cute.composition(tdPtdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)), Float32
        )

        # tmem -> rmem
        thr_copy_t2r = copy_utils.make_tmem_copy(tmem_load_atom, num_wg).get_slice(tidx)
        tStS_t2r = thr_copy_t2r.partition_S(tStS)  # (((32, 32), 1), 2, 1, 1)
        tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)
        tScS_t2r = thr_copy_t2r.partition_D(tScS)  # ((32, 1), 2, 1, 1)
        t0ScS_t2r = thr_copy_t2r.get_slice(0).partition_D(tScS)  # ((32, 1), 2, 1, 1)
        if const_expr(self.sparse_attn_p_mode):
            assert num_wg == 2, "BF16 probability QAT requires two compute warp groups"
            assert cute.size(tScS_t2r, mode=[0]) == 16
            assert cute.size(tScS_t2r, mode=[1]) == 4
        # ((32, 1), 2, 1, 1, STAGE)
        tSsLSE = thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))
        tSsdPsum = thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D))
        # rmem -> tmem
        thr_copy_r2t = copy_utils.make_tmem_copy(tmem_store_atom, num_wg).get_slice(tidx)
        tScP_r2t = thr_copy_r2t.partition_S(tScP)
        tStP_r2t = thr_copy_r2t.partition_D(tStP)
        tdPtdS_r2t = thr_copy_r2t.partition_D(tdPtdS)
        # RMEM -> SMEM
        copy_atom_r2s = sm100_utils_basic.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, self.ds_dtype, Float32, thr_copy_t2r
        )
        thr_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, thr_copy_t2r).get_slice(tidx)

        # Use the matching epilogue swizzle for the dS staging view.
        sdS_epi_layout = sm100_utils_basic.make_smem_layout_epi(
            self.ds_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_m), 1
        )
        sdS_layout = cute.slice_(sdS_epi_layout.outer, (None, None, 0))  # ((8,16), (64,2))
        # Group the layout into one mode to match the tiled-copy destination.
        sdS_layout = cute.make_layout((sdS_layout.shape,), stride=(sdS_layout.stride,))
        sdS_epi = cute.make_tensor(sdS.iterator, sdS_layout)

        consumer_state_dKV = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 2
        )
        consumer_state_S_P = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        consumer_state_dP = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        producer_state_dS = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, 1
        )
        consumer_state_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.LSE_stage
        )
        consumer_state_dPsum = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        work_idx = cute.arch.block_idx()[0]
        for _ in cutlass.range_constexpr(1):
            n_block, head_idx, batch_idx, row_start_c, k2q_count_c = (
                self._resolve_sparse_work(
                    work_idx,
                    mK2qCounts,
                    mSchedulerMetadata,
                    mWorkCount,
                )
            )
            seqlen = SeqlenInfoCls(batch_idx)
            has_later_fragment = Int32(0)
            if const_expr(mFragmentIndices is not None):
                fragment_idx = mFragmentIndices[batch_idx]
                if batch_idx + Int32(1) < mFragmentIndices.shape[0]:
                    if mFragmentIndices[batch_idx + Int32(1)] == fragment_idx:
                        has_later_fragment = Int32(1)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )
            causal_q_offset = seqlen.seqlen_k - seqlen.seqlen_q
            diag_q_count = Int32(0)
            masked_q_tiles = Int32(0)
            num_q_tiles_c = cute.ceil_div(k2q_count_c, self.q_per_tile)
            m_block_max = m_block_min + num_q_tiles_c
            if const_expr(self.is_causal):
                if lane_idx == 0:
                    q_block_end = (n_block // self.cluster_shape_mnk[0] + Int32(1)) * Int32(
                        self.tile_n
                    )
                    max_probe = cutlass.min(k2q_count_c, Int32(self.tile_n))
                    # CSR q-indices are sorted, so the causal region is a prefix.
                    lo = Int32(0)
                    hi = max_probe
                    for _ in cutlass.range_constexpr(
                        int(math.log2(self.tile_n)) + 1
                    ):
                        if lo < hi:
                            mid = (lo + hi) // Int32(2)
                            q_idx = self._load_sparse_q_idx(
                                mK2qIndices,
                                head_idx,
                                row_start_c,
                                mid,
                            )
                            if q_idx + causal_q_offset < q_block_end:
                                lo = mid + Int32(1)
                            else:
                                hi = mid
                    diag_q_count = lo
                diag_q_count = utils.shuffle_sync(diag_q_count, 0)
                masked_q_tiles = (
                    diag_q_count + Int32(self.q_per_tile - 1)
                ) // Int32(self.q_per_tile)
            mask = AttentionMaskCls(seqlen)
            n_block_for_cluster = n_block // self.cta_group_size
            # Sparse backward is always varlen, so sequence-length masking is required.
            mask_fn = partial(
                mask.apply_mask_sm100_transposed,
                tScS_t2r=tScS_t2r,
                t0ScS_t2r=t0ScS_t2r,
                n_block=n_block_for_cluster,
                mask_seqlen=True,
                mask_causal=self.is_causal,
            )

            loop_count = m_block_max - m_block_min
            prefetch_LSE = False
            process_tile = loop_count > Int32(0)

            # Mainloop: iterate m_block_min..m_block_max directly
            for iter_idx in cutlass.range(loop_count, unroll=1):
                m_block = m_block_min + iter_idx
                is_full_block = False
                sparse_valid_tok_count = cutlass.min(
                    Int32(self.q_per_tile),
                    k2q_count_c - iter_idx * Int32(self.q_per_tile),
                )
                # Prefetch 1 stage of LSE
                pipeline_LSE.consumer_wait(consumer_state_LSE)
                if const_expr(not self.sparse_attn_p_mode):
                    tSrLSE_s2r = cute.make_rmem_tensor(
                        tScS_t2r[None, 0, 0, 0].shape, Float32
                    )
                    if const_expr(prefetch_LSE and not self.shuffle_LSE):
                        cute.autovec_copy(
                            tSsLSE[None, 0, 0, 0, consumer_state_LSE.index],
                            tSrLSE_s2r,
                        )

                pipeline_S_P.consumer_wait(consumer_state_S_P)
                # Load S from TMEM to RMEM.
                tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, Float32)
                cute.copy(thr_copy_t2r, tStS_t2r, tSrS_t2r)

                # Apply causal and sequence-length masks.
                # pack_gqa: tile_m is packed rows; compare against packed seqlen
                # PackGQA compares against the packed GQA16 sequence length.
                seqlen_q_for_m = (
                    seqlen.seqlen_q * self.qhead_per_kvhead
                    if const_expr(self.pack_gqa)
                    else seqlen.seqlen_q
                )
                check_m_boundary = (m_block + 1) * self.tile_m > seqlen_q_for_m
                if const_expr(self.is_causal):
                    if iter_idx < masked_q_tiles:
                        masked_tok_count = cutlass.min(
                            Int32(self.q_per_tile),
                            diag_q_count - iter_idx * Int32(self.q_per_tile),
                        )
                        if const_expr(not self.sparse_attn_p_mode):
                            q_idx_lane = Int32(0)
                            if lane_idx < Int32(self.q_per_tile):
                                qi = (
                                    iter_idx * Int32(self.q_per_tile)
                                    + lane_idx
                                )
                                qi = cutlass.min(qi, k2q_count_c - Int32(1))
                                q_idx_lane = self._load_sparse_q_idx(
                                    mK2qIndices,
                                    head_idx,
                                    row_start_c,
                                    qi,
                                )
                            self._apply_sparse_gqa16_causal_mask(
                                tSrS_t2r,
                                tScS_t2r,
                                n_block_for_cluster,
                                seqlen.seqlen_q,
                                seqlen.seqlen_k,
                                sparse_valid_tok_count,
                                q_idx_lane,
                                masked_tok_count,
                            )
                        else:
                            q_idx_tile = cute.make_rmem_tensor(
                                self.q_per_tile, Int32
                            )
                            self._load_sparse_q_idx_tile(
                                q_idx_tile,
                                head_idx,
                                row_start_c,
                                iter_idx,
                                k2q_count_c,
                                mK2qIndices,
                            )
                            mask_fn(
                                tSrS_t2r,
                                m_block=m_block,
                                is_full_block=is_full_block,
                                check_m_boundary=check_m_boundary,
                                valid_tok_count=sparse_valid_tok_count,
                                q_idx_tile=q_idx_tile,
                                masked_tok_count=masked_tok_count,
                            )
                    else:
                        mask_fn(
                            tSrS_t2r,
                            m_block=m_block,
                            is_full_block=is_full_block,
                            check_m_boundary=check_m_boundary,
                            valid_tok_count=sparse_valid_tok_count,
                        )
                else:
                    mask_fn(
                        tSrS_t2r,
                        m_block=m_block,
                        is_full_block=is_full_block,
                        check_m_boundary=check_m_boundary,
                        valid_tok_count=sparse_valid_tok_count,
                    )
                num_stages = cute.size(tScS_t2r, mode=[1])
                # ---------------------------------------------
                # P = exp(S - LSE)
                # ---------------------------------------------
                tSrP_r2t_f32 = cute.make_rmem_tensor(tScP_r2t.shape, Float32)  # 64
                tSrP_r2t = cute.recast_tensor(tSrP_r2t_f32, self.q_dtype)
                for stage in cutlass.range_constexpr(num_stages):
                    tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                    tSsLSE_cur = tSsLSE[None, stage, 0, 0, consumer_state_LSE.index]
                    if const_expr(self.sparse_attn_p_mode):
                        qat.reconstruct_attention_ste_probabilities(
                            tScS_t2r[None, stage, 0, 0],
                            tSrS_cur,
                            tSrP_r2t[None, stage, 0, 0],
                            tSsLSE_cur,
                            mPQuantScale,
                            mK2qQSplitIndices,
                            head_idx,
                            row_start_c,
                            iter_idx,
                            k2q_count_c,
                            seqlen.offset_q,
                            softmax_scale_log2,
                            self.stage_p_stats,
                            self.q_per_tile,
                            self.qhead_per_kvhead,
                        )
                    else:
                        if const_expr(not self.shuffle_LSE):
                            if const_expr(stage > 0 or not prefetch_LSE):
                                cute.autovec_copy(tSsLSE_cur, tSrLSE_s2r)
                            tSrLSE = tSrLSE_s2r
                        else:
                            tSrLSE = tSsLSE_cur[lane_idx]
                        for v in cutlass.range_constexpr(cute.size(tSrS_t2r, mode=[0]) // 2):
                            if const_expr(not self.shuffle_LSE):
                                lse_pair = (tSrLSE[2 * v], tSrLSE[2 * v + 1])
                            else:
                                lse_pair = (
                                    utils.shuffle_sync(tSrLSE, offset=2 * v),
                                    utils.shuffle_sync(tSrLSE, offset=2 * v + 1),
                                )
                            tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                                ((tSrS_cur[2 * v], tSrS_cur[2 * v + 1])),
                                (softmax_scale_log2, softmax_scale_log2),
                                (-lse_pair[0], -lse_pair[1]),
                            )
                            tSrS_cur[2 * v] = cute.math.exp2(tSrS_cur[2 * v], fastmath=True)
                            tSrS_cur[2 * v + 1] = cute.math.exp2(tSrS_cur[2 * v + 1], fastmath=True)
                        utils.cvt_f16(tSrS_cur, tSrP_r2t[None, stage, 0, 0])
                    if const_expr(stage == 0):
                        cute.arch.fence_view_async_tmem_load()
                        # Prevent P writes from racing another warp's S reads.
                        self.compute_sync_barrier.arrive_and_wait()
                    cute.copy(
                        thr_copy_r2t,
                        tSrP_r2t_f32[None, stage, None, None],
                        tStP_r2t[None, stage, None, None],
                    )

                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_shared()
                self.compute_sync_barrier.arrive_and_wait()
                # Signal completion after all compute warps finish storing P.
                with cute.arch.elect_one():
                    pipeline_S_P.consumer_release(consumer_state_S_P)
                consumer_state_S_P.advance()
                pipeline_LSE.consumer_release(consumer_state_LSE)
                consumer_state_LSE.advance()

                # dS.T = P.T * (dP.T - D)
                pipeline_dPsum.consumer_wait(consumer_state_dPsum)
                pipeline_dP.consumer_wait(consumer_state_dP)
                for stage_pair in cutlass.range_constexpr(num_stages // 2):
                    stage_base = stage_pair * 2
                    tdPrdP_t2r_0 = cute.make_rmem_tensor(
                        tScS_t2r[None, 0, None, None].shape, Float32
                    )
                    tdPrdP_t2r_1 = cute.make_rmem_tensor(
                        tScS_t2r[None, 0, None, None].shape, Float32
                    )
                    cute.copy(
                        thr_copy_t2r,
                        tdPtdP_t2r[None, stage_base, None, None],
                        tdPrdP_t2r_0,
                    )
                    cute.copy(
                        thr_copy_t2r,
                        tdPtdP_t2r[None, stage_base + 1, None, None],
                        tdPrdP_t2r_1,
                    )
                    cute.arch.fence_view_async_tmem_load()
                    self.compute_sync_barrier.arrive_and_wait()
                    for stage_in_pair in cutlass.range_constexpr(2):
                        stage = stage_base + stage_in_pair
                        tdPrdP_t2r = tdPrdP_t2r_0
                        if const_expr(stage_in_pair == 1):
                            tdPrdP_t2r = tdPrdP_t2r_1
                        tdPrdP_cur = tdPrdP_t2r[None, 0, 0]
                        tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                        tSsdPsum_cur = tSsdPsum[
                            None, stage, 0, 0, consumer_state_dPsum.index
                        ]
                        if const_expr(not self.shuffle_dPsum):
                            tSrdPsum = cute.make_rmem_tensor_like(tSsdPsum_cur, Float32)
                            cute.autovec_copy(tSsdPsum_cur, tSrdPsum)
                        else:
                            dpsum_lane = cutlass.min(
                                lane_idx,
                                Int32(cute.size(tSsdPsum_cur) - 1),
                            )
                            tSrdPsum = tSsdPsum_cur[dpsum_lane]
                        for v in cutlass.range_constexpr(
                            cute.size(tdPrdP_t2r, mode=[0]) // 2
                        ):
                            if const_expr(not self.shuffle_dPsum):
                                dPsum_pair = (tSrdPsum[2 * v], tSrdPsum[2 * v + 1])
                            else:
                                dPsum_pair = (
                                    utils.shuffle_sync(tSrdPsum, offset=2 * v),
                                    utils.shuffle_sync(tSrdPsum, offset=2 * v + 1),
                                )
                            tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = (
                                activation.sub_packed_f32x2(
                                    (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                                    dPsum_pair,
                                )
                            )
                            tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = (
                                cute.arch.mul_packed_f32x2(
                                    (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                                    (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                                )
                            )

                        tdPrdS_cvt = cute.make_rmem_tensor_like(tdPrdP_cur, self.ds_dtype)
                        utils.cvt_f16(tdPrdP_cur, tdPrdS_cvt)
                        if const_expr(stage == 0):
                            pipeline_dS.producer_acquire(producer_state_dS)

                        # Stage dS in TMEM for the dK MMA.
                        tdPrdS_r2t_f32 = cute.recast_tensor(tdPrdS_cvt, Float32)
                        cute.copy(
                            thr_copy_r2t,
                            tdPrdS_r2t_f32,
                            tdPtdS_r2t[None, stage, 0, 0],
                        )

                        tRS_sdS = thr_copy_r2s.partition_D(sdS_epi)
                        cute.arch.sync_warp()
                        tdPrdS_store = cute.logical_divide(
                            tdPrdS_cvt, cute.make_layout(8)
                        )
                        tRS_sdS_store = cute.logical_divide(
                            tRS_sdS[None, stage], cute.make_layout(8)
                        )
                        for store_iter in cutlass.range_constexpr(
                            cute.size(tdPrdS_store, mode=[1])
                        ):
                            cute.autovec_copy(
                                tdPrdS_store[None, store_iter],
                                tRS_sdS_store[None, store_iter],
                            )

                cute.arch.fence_view_async_tmem_store()

                consumer_state_dP.advance()

                cute.arch.fence_view_async_shared()
                self.compute_sync_barrier.arrive_and_wait()
                # The compute barrier makes the following single-thread signal safe.
                pipeline_dPsum.consumer_release(consumer_state_dPsum)
                consumer_state_dPsum.advance()
                with cute.arch.elect_one():
                    pipeline_dS.producer_commit(producer_state_dS)
                producer_state_dS.advance()

            if process_tile:
                # The MMA warp publishes one final S/P completion after the
                # last dV. Drain it without advancing the dP consumer ring.
                pipeline_S_P.consumer_wait(consumer_state_S_P)
                with cute.arch.elect_one():
                    pipeline_S_P.consumer_release(consumer_state_S_P)
                consumer_state_S_P.advance()

            # Epilogue
            # Run epilogue if we processed any m_blocks for this n_block
            if process_tile:
                physical_block_idx = (
                    seqlen.padded_offset_k // Int32(self.tile_n) + n_block
                )
                owner_count = mDkvOwnerCounts[head_idx, physical_block_idx]
                dkv_writer_rank = Int32(-1)
                if const_expr(self.deterministic):
                    dkv_writer_rank = mDkvWriterRank[work_idx]
                next_owner_count = Int32(0)
                next_physical_block = (
                    seqlen.offset_k
                    + seqlen.seqlen_k
                    + (batch_idx + Int32(1)) * Int32(self.tile_n)
                ) // Int32(self.tile_n)
                if next_physical_block < mDkvOwnerCounts.shape[1]:
                    next_owner_count = mDkvOwnerCounts[
                        head_idx, next_physical_block
                    ]
                consumer_state_dKV = self.epilogue_dkv(
                    dp_idx,
                    batch_idx,
                    head_idx,
                    n_block,
                    seqlen,
                    thr_mma_dV,
                    tdVtdV,
                    mdVaccum,
                    mdV,
                    mdV_raw,
                    tma_atom_dV,
                    sdV,
                    sdV_out,
                    pipeline_dKV,
                    consumer_state_dKV,
                    physical_block_idx,
                    dkv_writer_rank,
                    mDkvSemaphore,
                    owner_count,
                    next_owner_count,
                    has_later_fragment,
                    Float32(1.0),
                    int(NamedBarrierBwdSm100.EpilogueWG1),
                    "V",
                )
                consumer_state_dKV = self.epilogue_dkv(
                    dp_idx,
                    batch_idx,
                    head_idx,
                    n_block,
                    seqlen,
                    thr_mma_dK,
                    tdKtdK,
                    mdKaccum,
                    mdK,
                    mdK_raw,
                    tma_atom_dK,
                    sdK,
                    sdK_out,
                    pipeline_dKV,
                    consumer_state_dKV,
                    physical_block_idx,
                    dkv_writer_rank,
                    mDkvSemaphore,
                    owner_count,
                    next_owner_count,
                    has_later_fragment,
                    softmax_scale,
                    int(NamedBarrierBwdSm100.EpilogueWG1),
                    "K",
                )


    @cute.jit
    def dq_acc_reduce(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        thr_mma_dQ: cute.ThrMma,
        tdQtdQ: cute.Tensor,
        pipeline_dQ: PipelineAsync,
        SeqlenInfoCls: Callable,
        mK2qIndices: Optional[cute.Tensor] = None,
        mK2qCounts: Optional[cute.Tensor] = None,
        mK2qQSplitIndices: Optional[cute.Tensor] = None,
        mDqSemaphore: Optional[cute.Tensor] = None,
        mSchedulerMetadata: Optional[cute.Tensor] = None,
        mWorkCount: Optional[cute.Tensor] = None,
    ):
        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_reduce_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx() % len(self.reduce_warp_ids))
        is_tma_warp = warp_idx == 0
        tdQcdQ = thr_mma_dQ.partition_C(cute.make_identity_tensor(self.mma_tiler_dsk[:2]))
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol_t2r)), Float32
        )
        thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ).get_slice(tidx)
        tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ)
        tdQcdQ_tensor = cute.make_tensor(tdQcdQ.iterator, tdQcdQ.layout)
        tdQcdQ_t2r = thr_copy_t2r.partition_D(tdQcdQ_tensor)
        tdQrdQ_t2r_shape = tdQcdQ_t2r.shape
        assert cute.size(tdQrdQ_t2r_shape, mode=[1]) == self.dQaccum_reduce_stage_t2r, (
            "dQaccum t2r reduce stage mismatch"
        )

        thr_copy_dQaccum_r2s = copy_utils.tiled_copy_1d(
            self.dqaccum_dtype, num_reduce_threads, num_copy_elems=128 // self.dqaccum_dtype.width
        ).get_slice(tidx)
        tdQsdQ = thr_copy_dQaccum_r2s.partition_D(sdQaccum)

        read_flag = const_expr(True)

        dQ_consumer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        dQ_tma_store_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.sdQaccum_stage
        )
        work_idx = cute.arch.block_idx()[0]
        for _ in cutlass.range_constexpr(1):
            n_block, head_idx, batch_idx, row_start_dq, k2q_count_dq = (
                self._resolve_sparse_work(
                    work_idx,
                    mK2qCounts,
                    mSchedulerMetadata,
                    mWorkCount,
                )
            )
            seqlen = SeqlenInfoCls(batch_idx)
            packed_q_stride = Int64(self.qhead_per_kvhead * self.tile_hdim)
            mdQaccum_cur = cute.domain_offset(
                (Int64(seqlen.padded_offset_q) * packed_q_stride,), mdQaccum[None, head_idx]
            )

            # Each iter_idx is a sparse q tile. The scatter bulk reduce is
            # per-token, while reduction storage is cut along head_dim.
            num_q_tiles_dq = cute.ceil_div(k2q_count_dq, self.q_per_tile)
            loop_count = num_q_tiles_dq
            process_tile = num_q_tiles_dq > Int32(0)
            token_elems = self.sparse_dq_token_elems
            stage_token_elems = self.sparse_dq_stage_token_elems
            stage_tile_elems = self.sparse_dq_stage_tile_elems
            hidden_vec_elems = 128 // self.dqaccum_dtype.width
            hidden_chunks_per_stage = self.dQ_reduce_ncol // hidden_vec_elems
            hidden_chunk_elems = self.qhead_per_kvhead * hidden_vec_elems
            num_reduce_warps = len(self.reduce_warp_ids)
            tokens_per_reduce_warp = (
                self.q_per_tile + num_reduce_warps - 1
            ) // num_reduce_warps

            # dQ accumulator reduction mainloop.
            for iter_idx in cutlass.range(loop_count, unroll=1):
                pipeline_dQ.consumer_wait(dQ_consumer_state)
                tdQrdQ_t2r = cute.make_rmem_tensor(tdQrdQ_t2r_shape, Float32)
                cute.copy(thr_copy_t2r, tdQtdQ_t2r, tdQrdQ_t2r)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_dQ.consumer_release(dQ_consumer_state)
                dQ_consumer_state.advance()

                tdQrdQ_shape = (
                    self.dQ_reduce_ncol,
                    self.tile_hdim // self.cta_group_size // self.dQ_reduce_ncol,
                )
                tdQrdQ = cute.make_tensor(tdQrdQ_t2r.iterator, tdQrdQ_shape)

                for stage in cutlass.range_constexpr(cute.size(tdQrdQ_t2r, mode=[1])):
                    smem_idx_dq = dQ_tma_store_producer_state.index
                    stage_idx_dq = Int32(stage)
                    tdQsdQ_r2s = tdQsdQ[None, None, smem_idx_dq]
                    # Transpose the hidden-chunk-major thread layout into
                    # token-major SMEM while preserving float4 stores.
                    token_idx_r2s = tidx // Int32(hidden_chunks_per_stage)
                    token_major_offset = token_idx_r2s * Int32(
                        stage_token_elems - hidden_chunk_elems
                    )
                    token_major_layout = cute.make_layout(
                        ((hidden_vec_elems, 1), hidden_chunks_per_stage),
                        stride=((1, 0), hidden_chunk_elems),
                    )
                    tdQsdQ_token_major = cute.make_tensor(
                        tdQsdQ_r2s.iterator + token_major_offset,
                        token_major_layout,
                    )
                    tdQrdQ_token_major = cute.make_tensor(
                        tdQrdQ[None, stage].iterator,
                        token_major_layout.shape,
                    )
                    cute.autovec_copy(tdQrdQ_token_major, tdQsdQ_token_major)
                    cute.arch.fence_view_async_shared()
                    self.reduce_sync_barrier.arrive_and_wait()
                    with cute.arch.elect_one():
                        qi_base_scatter = iter_idx * Int32(self.q_per_tile)
                        tok_begin_scatter = Int32(warp_idx) * Int32(tokens_per_reduce_warp)
                        smem_stage_base_scatter = (
                            sdQaccum.iterator + smem_idx_dq * Int32(stage_tile_elems)
                        )
                        stage_token_off_scatter = Int64(stage_idx_dq) * Int64(stage_token_elems)
                        for tok_local in cutlass.range_constexpr(tokens_per_reduce_warp):
                            tok_scatter = tok_begin_scatter + Int32(tok_local)
                            qi_scatter = qi_base_scatter + Int32(tok_scatter)
                            if tok_scatter < Int32(self.q_per_tile) and qi_scatter < k2q_count_dq:
                                q_idx_scatter = self._load_sparse_q_idx(
                                    mK2qIndices,
                                    head_idx,
                                    row_start_dq,
                                    qi_scatter,
                                )
                                gmem_token_ptr_scatter = (
                                    mdQaccum_cur.iterator
                                    + Int64(q_idx_scatter) * Int64(token_elems)
                                    + stage_token_off_scatter
                                )
                                if const_expr(self.deterministic):
                                    packed_qsplit = mK2qQSplitIndices[
                                        head_idx,
                                        row_start_dq + qi_scatter,
                                    ]
                                    writer_rank = (
                                        packed_qsplit >> Int32(24)
                                    ) & Int32(0xFF)
                                    q_global_scatter = (
                                        seqlen.offset_q + q_idx_scatter
                                    )
                                    lock_ptr = mDqSemaphore[
                                        head_idx,
                                        q_global_scatter,
                                        None,
                                    ].iterator + stage_idx_dq
                                    observed_rank = Int32(-1)
                                    while observed_rank != writer_rank:
                                        observed_rank = ld_acquire(lock_ptr)
                                    copy_utils.cpasync_reduce_bulk_add_f32(
                                        smem_stage_base_scatter
                                        + Int32(tok_scatter)
                                        * Int32(stage_token_elems),
                                        gmem_token_ptr_scatter,
                                        Int32(stage_token_elems * 4),
                                    )
                                    cute.arch.cp_async_bulk_commit_group()
                                    cute.arch.cp_async_bulk_wait_group(
                                        0, read=read_flag
                                    )
                                    red_release(lock_ptr, 1)
                                else:
                                    copy_utils.cpasync_reduce_bulk_add_f32(
                                        smem_stage_base_scatter
                                        + Int32(tok_scatter)
                                        * Int32(stage_token_elems),
                                        gmem_token_ptr_scatter,
                                        Int32(stage_token_elems * 4),
                                    )
                        if const_expr(not self.deterministic):
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(
                                self.sdQaccum_stage - 1, read=read_flag
                            )
                    self.reduce_sync_barrier.arrive_and_wait()
                    dQ_tma_store_producer_state.advance()

            if process_tile:
                if is_tma_warp:
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                self.reduce_sync_barrier.arrive_and_wait()

        cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def epilogue_dkv(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        seqlen,
        thr_mma: cute.ThrMma,
        tdKVtdKV: cute.Tensor,
        mdKVaccum: cute.Tensor,
        mdKV: cute.Tensor,
        mdKV_raw: cute.Tensor,
        tma_atom_dKV: cute.CopyAtom,
        sdKVaccum: cute.Tensor,
        sdKVout: cute.Tensor,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        physical_block_idx: Int32,
        writer_rank: Int32,
        mDkvSemaphore: Optional[cute.Tensor],
        owner_count: Int32,
        next_owner_count: Int32,
        has_later_fragment: Int32,
        scale: Float32,
        barrier_id: Int32,
        K_or_V: cutlass.Constexpr[str],
    ) -> cutlass.pipeline.PipelineState:
        assert K_or_V in ("K", "V")
        tile_hdim = self.tile_hdim if const_expr(K_or_V == "K") else self.tile_hdimv
        out_dtype = self.dk_out_dtype if const_expr(K_or_V == "K") else self.dv_out_dtype
        num_epi_stages = self.num_epi_stages if const_expr(K_or_V == "K") else self.num_epi_stages_v
        reduce_ncol = self.dK_reduce_ncol if const_expr(K_or_V == "K") else self.dV_reduce_ncol
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0
        kv_lock_kind = Int32(0)
        if const_expr(K_or_V == "K"):
            kv_lock_kind = Int32(1)
        if const_expr(self.deterministic):
            if owner_count > Int32(1):
                lock_ptr = mDkvSemaphore[
                    head_idx,
                    physical_block_idx,
                    kv_lock_kind,
                    None,
                ].iterator + wg_idx
                if leader_warp:
                    with cute.arch.elect_one():
                        observed_rank = Int32(-1)
                        while observed_rank != writer_rank:
                            observed_rank = ld_acquire(lock_ptr)
                cute.arch.barrier(
                    barrier_id=barrier_id + wg_idx,
                    number_of_threads=128,
                )

        cta_group_tile_n = const_expr(self.tile_n * self.cta_group_size)
        sdKVaccum = sdKVaccum[None, wg_idx]
        sdKVout = sdKVout[None, None, wg_idx]

        head_idx_kv = (
            head_idx
            if const_expr(self.pack_gqa)
            else head_idx // self.qhead_per_kvhead
        )
        mdKVaccum_cur = seqlen.offset_batch_K(
            mdKVaccum,
            batch_idx,
            dim=2,
            padded=True,
            multiple=tile_hdim,
        )[None, head_idx_kv]  # (seqlen * hdim)
        gdKVaccum = cute.local_tile(
            mdKVaccum_cur, (self.tile_n * tile_hdim,), (n_block,)
        )  # (tile_n * hdim)
        mdKV_cur = seqlen.offset_batch_K(mdKV, batch_idx, dim=3)[
            None, None, head_idx_kv
        ]
        gdKV_p = cute.local_tile(
            mdKV_cur, (self.tile_n, tile_hdim), (n_block, 0)
        )
        gdKV = self.split_wg(gdKV_p, wg_idx, num_wg)
        gdKV_epi = cute.local_tile(
            gdKV,
            (self.tile_n, reduce_ncol),
            (0, None),
        )
        is_full_kv_block = (
            n_block * Int32(self.tile_n) + Int32(self.tile_n) <= seqlen.seqlen_k
        )
        tdKVsdKVout, tdKVgdKV = cpasync.tma_partition(
            tma_atom_dKV,
            0,
            cute.make_layout(1),
            cute.group_modes(sdKVout, 0, 2),
            cute.group_modes(gdKV_epi, 0, 2),
        )
        assert cute.size(tdKVgdKV.shape[1]) == num_epi_stages

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(reduce_ncol)), Float32
        )

        read_flag = const_expr(True)

        pipeline_dKV.consumer_wait(consumer_state_dKV)

        for epi_stage in cutlass.range_constexpr(num_epi_stages):
            # TMEM -> RMEM -- setup
            thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(tidx)
            tdKVtdKV_t2r_p = thr_copy_t2r.partition_S(tdKVtdKV)
            tdKVtdKV_t2r = self.split_wg(tdKVtdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVtdKV_t2r = tdKVtdKV_t2r[None, epi_stage]

            cdKV = cute.make_identity_tensor((cta_group_tile_n, tile_hdim))
            tdKVcdKV = thr_mma.partition_C(cdKV)
            tdKVcdKV_t2r_p = thr_copy_t2r.partition_D(tdKVcdKV)
            tdKVcdKV_t2r = self.split_wg(tdKVcdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVcdKV_t2r = tdKVcdKV_t2r[None, epi_stage]

            tdKVrdKV_t2r = cute.make_rmem_tensor(tdKVcdKV_t2r.shape, Float32)

            assert cute.size(tdKVrdKV_t2r) == cute.size(tdKVtdKV_t2r) // cute.arch.WARP_SIZE, (
                "RMEM<->TMEM fragment size mismatch"
            )

            # TMEM -> RMEM -- copy and fence
            cute.copy(thr_copy_t2r, tdKVtdKV_t2r, tdKVrdKV_t2r)
            cute.arch.fence_view_async_tmem_load()

            # Split owners retain FP32 partials for reduction. Single-owner
            # blocks stage the final dtype. Sequence tails use guarded stores
            # so reused staging storage cannot leak to the output.
            sdKVaccum_stage = cute.make_tensor(
                sdKVaccum.iterator,
                cute.make_layout(
                    (self.tile_n, reduce_ncol),
                    stride=(reduce_ncol, 1),
                ),
            )
            for i in cutlass.range(cute.size(tdKVrdKV_t2r.shape), unroll_full=True):
                coord = tdKVcdKV_t2r[i]
                row = coord[0]
                col = coord[1]
                col_in_stage = col % Int32(reduce_ncol)
                value = tdKVrdKV_t2r[i]
                if owner_count > Int32(1):
                    sdKVaccum_stage[row, col_in_stage] = value
                else:
                    if const_expr(K_or_V == "K"):
                        value *= scale
                    sdKVout[row, col_in_stage] = out_dtype(value)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

            # SMEM -> GMEM. Sequence tails use guarded stores so invalid rows
            # cannot expose reused SMEM data.
            col_stage_start = (
                wg_idx * Int32(tile_hdim // num_wg)
                + Int32(epi_stage * reduce_ncol)
            )
            if owner_count == Int32(1) and not is_full_kv_block:
                valid_rows = seqlen.seqlen_k - n_block * Int32(self.tile_n)
                for vector_iter in cutlass.range_constexpr(
                    self.tile_n * reduce_ncol // (128 * 4)
                ):
                    vector_idx = tidx + Int32(vector_iter * 128)
                    row = vector_idx // Int32(reduce_ncol // 4)
                    col = (vector_idx - row * Int32(reduce_ncol // 4)) * Int32(4)
                    if (
                        (row < valid_rows)
                        | (next_owner_count == Int32(0))
                        | (has_later_fragment != Int32(0))
                    ):
                        values = cute.make_rmem_tensor((4,), Float32)
                        values.fill(0.0)
                        if row < valid_rows:
                            for value_idx in cutlass.range_constexpr(4):
                                values[value_idx] = Float32(
                                    sdKVout[row, col + Int32(value_idx)]
                                )
                        dst_row = (
                            seqlen.offset_k
                            + n_block * Int32(self.tile_n)
                            + row
                        )
                        if dst_row < mdKV_raw.shape[0]:
                            dst_offset = (
                                (Int64(dst_row)
                                * Int64(cute.size(mdKV_raw.shape[2]))
                                + Int64(head_idx_kv))
                                * Int64(tile_hdim)
                                + Int64(col_stage_start + col)
                            )
                            dst = cute.make_ptr(
                                out_dtype,
                                mdKV_raw.iterator.toint()
                                + dst_offset * Int64(out_dtype.width // 8),
                                mem_space=mdKV_raw.iterator.memspace,
                                assumed_align=8,
                            )
                            if const_expr(out_dtype == cutlass.BFloat16):
                                copy_utils.stg_64_bf16(
                                    dst, values[0], values[1], values[2], values[3]
                                )
                            else:
                                copy_utils.stg_64_f16(
                                    dst, values[0], values[1], values[2], values[3]
                                )
                if leader_warp:
                    cute.arch.barrier_arrive(
                        barrier_id=barrier_id + wg_idx,
                        number_of_threads=128 + cute.arch.WARP_SIZE,
                    )
            elif leader_warp:
                with cute.arch.elect_one():
                    if owner_count > Int32(1):
                        for row in cutlass.range(self.tile_n, unroll_full=True):
                            copy_utils.cpasync_reduce_bulk_add_f32(
                                sdKVaccum.iterator + Int32(row * reduce_ncol),
                                gdKVaccum.iterator
                                + Int32(row * tile_hdim)
                                + col_stage_start,
                                reduce_ncol * Float32.width // 8,
                            )
                    else:
                        cute.copy(
                            tma_atom_dKV,
                            tdKVsdKVout,
                            tdKVgdKV[None, epi_stage],
                        )
                cute.arch.cp_async_bulk_commit_group()
                if const_expr(epi_stage < num_epi_stages - 1):
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                cute.arch.barrier_arrive(
                    barrier_id=barrier_id + wg_idx,
                    number_of_threads=128 + cute.arch.WARP_SIZE,
                )

            # Barrier since all warps need to wait for SMEM to be freed
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
            )

        if leader_warp:
            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            # The final bulk wait is issued only by the leader warp. Make the
            # whole epilogue warp-group rendezvous here before releasing the
            # dKV pipeline state so the next work tile cannot reuse the SMEM
            # staging buffer while the last store group is still draining.
            cute.arch.barrier_arrive(
                barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
            )
        cute.arch.barrier(
            barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
        )
        if const_expr(self.deterministic):
            if owner_count > Int32(1) and leader_warp:
                lock_ptr = mDkvSemaphore[
                    head_idx,
                    physical_block_idx,
                    kv_lock_kind,
                    None,
                ].iterator + wg_idx
                with cute.arch.elect_one():
                    red_release(lock_ptr, 1)
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV
