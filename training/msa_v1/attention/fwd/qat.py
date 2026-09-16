"""Logical-probability QAT helpers for the SM100 sparse attention kernels."""

import math
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05

from msa_v1._common import blackwell_helpers as sm100_helpers
from msa_v1._common import copy_utils, utils
from msa_v1._common import mma_sm100_desc as sm100_desc
from msa_v1._common.mask import AttentionMask
from msa_v1._common.seqlen_info import SeqlenInfoQK
from msa_v1._common.softmax import LOG2_FP8_PROBABILITY_SCALE


class SparseAttentionQatForwardMixin:
    """Probability scaling and logical-probability QAT device helpers."""

    @cute.jit
    def _probability_max_log2(self, row_max: Float32, scale_log2: Float32):
        """Return the negative exponent bias, including the P448 offset."""
        if const_expr(self.scale_fp8_p):
            result, _ = cute.arch.fma_packed_f32x2(
                (row_max, row_max),
                (scale_log2, scale_log2),
                (-Float32(LOG2_FP8_PROBABILITY_SCALE), -Float32(LOG2_FP8_PROBABILITY_SCALE)),
            )
            return result
        else:
            return row_max * scale_log2

    @cute.jit
    def load_qdo_tma(
        self,
        tma_atom_Q,
        tma_atom_DO,
        mQ_2d: cute.Tensor,
        mDO_2d: cute.Tensor,
        mK2qQSplitIndices: cute.Tensor,
        sQIdxMeta: cute.Tensor,
        sQ_load: cute.Tensor,
        sDO_load: cute.Tensor,
        pipeline_q,
        pipeline_do,
        load_wg_barrier,
        num_q_groups: Int32,
        count_raw: Int32,
        has_work: Int32,
        head_kv_idx: Int32,
        row_start: Int32,
        q_batch_offset: Int32,
        num_heads_kv: Int32,
    ):
        """Stage matching Q and dO tiles for independent QK/dP UMMAs."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        warp_idx_in_wg = warp_idx - Int32(self.q_load_warp_base)
        if const_expr(self.q_dtype == cutlass.Float8E4M3FN):
            gQ = cute.local_tile(
                mQ_2d, (self.qheadperkv, self.head_dim), (None, 0)
            )
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), gQ, sQ_load
            )
        else:
            gQ_k0 = cute.local_tile(
                mQ_2d, (self.qheadperkv, self.k_tile), (None, 0)
            )
            gQ_k1 = cute.local_tile(
                mQ_2d, (self.qheadperkv, self.k_tile), (None, 1)
            )
            load_Q_k0, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), gQ_k0, sQ_load
            )
            load_Q_k1, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), gQ_k1, sQ_load
            )
        gDO_k0 = cute.local_tile(
            mDO_2d, (self.qheadperkv, self.k_tile), (None, 0)
        )
        gDO_k1 = cute.local_tile(
            mDO_2d, (self.qheadperkv, self.k_tile), (None, 1)
        )
        load_DO_k0, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_DO, 0, cute.make_layout(1), gDO_k0, sDO_load
        )
        load_DO_k1, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_DO, 0, cute.make_layout(1), gDO_k1, sDO_load
        )
        q_oob_m_idx = mQ_2d.shape[0] // Int32(self.qheadperkv)
        tokens_per_warp: cutlass.Constexpr[int] = (
            self.q_tokens_per_group + self.num_q_load_warps - 1
        ) // self.num_q_load_warps

        if has_work:
            for qi_group in cutlass.range(num_q_groups, unroll=1):
                q_slot = qi_group % Int32(self.q_stage)
                q_phase = (qi_group // Int32(self.q_stage)) & Int32(1)
                q_producer_phase = q_phase ^ Int32(1)
                do_slot = qi_group % Int32(self.do_stage)
                do_phase = (qi_group // Int32(self.do_stage)) & Int32(1)
                do_producer_phase = do_phase ^ Int32(1)
                if warp_idx_in_wg == Int32(0):
                    pipeline_q.producer_acquire_w_index_phase(
                        q_slot, q_producer_phase
                    )
                    pipeline_do.producer_acquire_w_index_phase(
                        do_slot, do_producer_phase
                    )
                load_wg_barrier.arrive_and_wait()

                q_mbar_ptr = pipeline_q.sync_object_full.get_barrier(q_slot)
                do_mbar_ptr = pipeline_do.sync_object_full.get_barrier(do_slot)
                q_subtiles_per_token = (
                    1
                    if const_expr(self.q_dtype == cutlass.Float8E4M3FN)
                    else self.k_stages
                )
                q_sub_stage_base = q_slot * Int32(
                    self.q_tokens_per_group * q_subtiles_per_token
                )
                do_sub_stage_base = do_slot * Int32(
                    self.q_tokens_per_group * self.k_stages
                )
                qidx_meta_slot = (
                    qi_group & Int32(self.qidx_meta_stages - 1)
                ) * Int32(self.q_tokens_per_group)
                if (
                    warp_idx_in_wg == Int32(0)
                    and lane_idx < Int32(self.q_tokens_per_group)
                ):
                    tok_idx = lane_idx
                    qi = qi_group * Int32(self.q_tokens_per_group) + tok_idx
                    if qi < count_raw:
                        sQIdxMeta[qidx_meta_slot + tok_idx] = (
                            self._load_qsplit_idx(
                                mK2qQSplitIndices,
                                head_kv_idx,
                                row_start,
                                qi,
                            )
                        )
                    else:
                        sQIdxMeta[qidx_meta_slot + tok_idx] = Int32(0)
                    cute.arch.mbarrier_arrive(q_mbar_ptr)
                load_wg_barrier.arrive_and_wait()

                for qi_slot in cutlass.range_constexpr(tokens_per_warp):
                    tok_idx = (
                        warp_idx_in_wg * Int32(tokens_per_warp)
                        + Int32(qi_slot)
                    )
                    if tok_idx < Int32(self.q_tokens_per_group):
                        qi = qi_group * Int32(self.q_tokens_per_group) + tok_idx
                        m_tile_idx = q_oob_m_idx
                        if qi < count_raw:
                            qsplit = sQIdxMeta[qidx_meta_slot + tok_idx]
                            q_idx = self._decode_q_idx_from_qsplit(qsplit)
                            m_tile_idx = (
                                (q_batch_offset + q_idx) * num_heads_kv
                                + head_kv_idx
                            )
                        if const_expr(self.q_dtype == cutlass.Float8E4M3FN):
                            load_Q(
                                src_idx=m_tile_idx,
                                dst_idx=q_sub_stage_base + tok_idx,
                                tma_bar_ptr=q_mbar_ptr,
                            )
                        else:
                            load_Q_k0(
                                src_idx=m_tile_idx,
                                dst_idx=q_sub_stage_base + tok_idx,
                                tma_bar_ptr=q_mbar_ptr,
                            )
                            load_Q_k1(
                                src_idx=m_tile_idx,
                                dst_idx=(
                                    q_sub_stage_base
                                    + Int32(self.q_tokens_per_group)
                                    + tok_idx
                                ),
                                tma_bar_ptr=q_mbar_ptr,
                            )
                        load_DO_k0(
                            src_idx=m_tile_idx,
                            dst_idx=do_sub_stage_base + tok_idx,
                            tma_bar_ptr=do_mbar_ptr,
                        )
                        load_DO_k1(
                            src_idx=m_tile_idx,
                            dst_idx=(
                                do_sub_stage_base
                                + Int32(self.q_tokens_per_group)
                                + tok_idx
                            ),
                            tma_bar_ptr=do_mbar_ptr,
                        )
                load_wg_barrier.arrive_and_wait()

            if warp_idx_in_wg == Int32(0):
                next_q_slot = num_q_groups % Int32(self.q_stage)
                next_q_phase = (
                    (num_q_groups // Int32(self.q_stage)) & Int32(1)
                ) ^ Int32(1)
                pipeline_q.producer_acquire_w_index_phase(
                    next_q_slot, next_q_phase
                )
                next_do_slot = num_q_groups % Int32(self.do_stage)
                next_do_phase = (
                    (num_q_groups // Int32(self.do_stage)) & Int32(1)
                ) ^ Int32(1)
                pipeline_do.producer_acquire_w_index_phase(
                    next_do_slot, next_do_phase
                )

    @cute.jit
    def mma_dpsum(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_dp: cute.TiledMma,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sQ: cute.Tensor,
        sDO: cute.Tensor,
        pipeline_q,
        pipeline_do,
        pipeline_s,
        pipeline_dp,
        mbar_k_ptr,
        mbar_v_ptr,
        num_q_groups: Int32,
        has_work: Int32,
    ):
        """Issue independent QK and dO @ V.T tiles into paired TMEM rings."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == Int32(self.mma_warp_id) and has_work:
            tSrQ = tiled_mma_qk.make_fragment_A(sQ)
            tSrK = tiled_mma_qk.make_fragment_B(sK)
            tSrDO = tiled_mma_dp.make_fragment_A(sDO)
            tSrV = tiled_mma_dp.make_fragment_B(sV)
            tSrQ0 = tSrQ[None, None, None, 0]
            tSrK0 = tSrK[None, None, None, 0]
            tSrDO0 = tSrDO[None, None, None, 0]
            tSrV0 = tSrV[None, None, None, 0]

            q_smem_base = sm100_desc.smem_desc_base_from_tensor(
                sQ, sm100_desc.Major.K
            )
            do_smem_base = sm100_desc.smem_desc_base_from_tensor(
                sDO, sm100_desc.Major.K
            )
            k_smem_base = sm100_desc.smem_desc_base_from_tensor(
                sK, sm100_desc.Major.K
            )
            v_smem_base = sm100_desc.smem_desc_base_from_tensor(
                sV, sm100_desc.Major.K
            )
            k_smem_start = sm100_desc.make_smem_desc_start_addr(
                sK[None, None, None, 0].iterator
            )
            v_smem_start = sm100_desc.make_smem_desc_start_addr(
                sV[None, None, None, 0].iterator
            )
            q_smem_start = sm100_desc.make_smem_desc_start_addr(
                sQ[None, None, None, self.q_stage - 1].iterator
            )
            do_smem_start = sm100_desc.make_smem_desc_start_addr(
                sDO[None, None, None, self.do_stage - 1].iterator
            )
            sm100_helpers.declare_ptx_smem_desc(
                q_smem_start,
                q_smem_base,
                tSrQ0.layout,
                var_name_prefix="dpsum_q_desc",
            )
            sm100_helpers.declare_ptx_smem_desc(
                do_smem_start,
                do_smem_base,
                tSrDO0.layout,
                var_name_prefix="dpsum_do_desc",
            )
            sm100_helpers.declare_ptx_idesc(
                tiled_mma_qk.op, var_name="dpsum_qk_idesc"
            )
            sm100_helpers.declare_ptx_idesc(
                tiled_mma_dp.op, var_name="dpsum_dp_idesc"
            )
            q_stage_stride = (
                sQ.layout.stride[-1] * sQ.element_type.width // 8
            ) >> 4
            do_stage_stride = (
                sDO.layout.stride[-1] * sDO.element_type.width // 8
            ) >> 4
            q_wrap_offset = -(self.q_stage - 1) * q_stage_stride
            do_wrap_offset = -(self.do_stage - 1) * do_stage_stride

            gemm_qk_s0_wrap = partial(
                sm100_helpers.gemm_ptx_precomputed_varname,
                Int32(self.tmem_s_offset),
                smem_desc_base_b=k_smem_base,
                tCrB_layout=tSrK0.layout,
                smem_var_name_prefix="dpsum_q_desc",
                idesc_var_name="dpsum_qk_idesc",
                smem_offset=q_wrap_offset,
                zero_init=True,
                cta_group=self.cta_group_size,
                mma_kind=self.qk_mma_kind,
            )
            gemm_qk_s1_advance = partial(
                sm100_helpers.gemm_ptx_precomputed_varname,
                Int32(self.tmem_stage_stride + self.tmem_s_offset),
                smem_desc_base_b=k_smem_base,
                tCrB_layout=tSrK0.layout,
                smem_var_name_prefix="dpsum_q_desc",
                idesc_var_name="dpsum_qk_idesc",
                smem_offset=q_stage_stride,
                zero_init=True,
                cta_group=self.cta_group_size,
                mma_kind=self.qk_mma_kind,
            )
            gemm_dp_s0_wrap = partial(
                sm100_helpers.gemm_ptx_precomputed_varname,
                Int32(self.tmem_o_offset),
                smem_desc_base_b=v_smem_base,
                tCrB_layout=tSrV0.layout,
                smem_var_name_prefix="dpsum_do_desc",
                idesc_var_name="dpsum_dp_idesc",
                smem_offset=do_wrap_offset,
                zero_init=True,
                cta_group=self.cta_group_size,
                mma_kind=self.pv_mma_kind,
            )
            gemm_dp_s1_advance = partial(
                sm100_helpers.gemm_ptx_precomputed_varname,
                Int32(self.tmem_o_offset + self.tmem_o_stage_stride),
                smem_desc_base_b=v_smem_base,
                tCrB_layout=tSrV0.layout,
                smem_var_name_prefix="dpsum_do_desc",
                idesc_var_name="dpsum_dp_idesc",
                smem_offset=do_stage_stride,
                zero_init=True,
                cta_group=self.cta_group_size,
                mma_kind=self.pv_mma_kind,
            )

            cute.arch.mbarrier_wait(mbar_k_ptr, 0)
            cute.arch.mbarrier_wait(mbar_v_ptr, 0)
            for qi_group in cutlass.range(num_q_groups, unroll=1):
                slot = qi_group & Int32(1)
                phase = (qi_group // Int32(2)) & Int32(1)
                pipeline_q.consumer_wait_w_index_phase(slot, phase)
                pipeline_do.consumer_wait_w_index_phase(slot, phase)
                pipeline_s.producer_acquire_w_index_phase(
                    slot, phase ^ Int32(1)
                )
                pipeline_dp.producer_acquire_w_index_phase(
                    slot, phase ^ Int32(1)
                )
                if slot == Int32(0):
                    gemm_qk_s0_wrap(smem_desc_start_b=k_smem_start)
                else:
                    gemm_qk_s1_advance(smem_desc_start_b=k_smem_start)
                pipeline_s.producer_commit_w_index(slot)
                if slot == Int32(0):
                    gemm_dp_s0_wrap(smem_desc_start_b=v_smem_start)
                else:
                    gemm_dp_s1_advance(smem_desc_start_b=v_smem_start)
                pipeline_dp.producer_commit_w_index(slot)
                pipeline_q.consumer_release_w_index(slot)
                pipeline_do.consumer_release_w_index(slot)


    @cute.jit
    def compute_dpsum(
        self,
        stage: cutlass.Constexpr[int],
        tiled_mma_qk: cute.TiledMma,
        tStS: cute.Tensor,
        tDPtDP: cute.Tensor,
        sQIdxMeta: cute.Tensor,
        pipeline_s,
        pipeline_dp,
        mLSE: cute.Tensor,
        mdPsumPartial: cute.Tensor,
        mPQuantScale: cute.Tensor,
        softmax_scale_log2: Float32,
        kv_block_idx: Int32,
        kv_valid_cols: Int32,
        diag_q_count: Int32,
        num_q_groups: Int32,
        count_raw: Int32,
        has_work: Int32,
        causal_q_offset: Int32,
        batch_idx: Int32,
        head_kv_idx: Int32,
        seq_len_q: Int32,
        q_batch_offset: Int32,
    ):
        """Reduce exact logical D from paired score and dP TMEM tiles."""
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx_in_wg = warp_idx % Int32(self.warps_per_group)
        group_tidx = (
            warp_idx_in_wg * Int32(cute.arch.WARP_SIZE)
            + tidx % Int32(cute.arch.WARP_SIZE)
        )
        thr0_qk = tiled_mma_qk.get_slice(0)
        tScS = thr0_qk.partition_C(
            cute.make_identity_tensor(self.mma_tiler_qk[:2])
        )
        tScS = tScS[(None, None), 0, 0]
        tSAcc = tStS[(None, None), 0, 0, stage]
        tDPAcc = tDPtDP[(None, None), 0, 0, stage]
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            Float32,
        )
        thr_s_load = tcgen05.make_tmem_copy(
            tmem_load_atom, tSAcc
        ).get_slice(group_tidx)
        thr_dp_load = tcgen05.make_tmem_copy(
            tmem_load_atom, tDPAcc
        ).get_slice(group_tidx)
        tStS_t2r = thr_s_load.partition_S(tSAcc)
        tScS_t2r = thr_s_load.partition_D(tScS)
        tDPtdP_t2r = thr_dp_load.partition_S(tDPAcc)
        tScdP_t2r = thr_dp_load.partition_D(tScS)

        if has_work:
            kv_block_col_start = Int32(0)
            if const_expr(self.causal):
                kv_block_col_start = kv_block_idx * Int32(self.n_block_size)
            num_stage_groups = (
                num_q_groups + Int32(1 - stage)
            ) // Int32(2)
            for qi_iter in cutlass.range(num_stage_groups, unroll=1):
                qi_group = qi_iter * Int32(2) + Int32(stage)
                phase = qi_iter & Int32(1)
                slot = Int32(stage)
                pipeline_s.consumer_wait_w_index_phase(slot, phase)

                tSrS = cute.make_rmem_tensor(tScS_t2r.shape, Float32)
                cute.copy(thr_s_load, tStS_t2r, tSrS)
                cute.arch.fence_view_async_tmem_load()
                pipeline_s.consumer_release_w_index(slot)

                seqlen_info = SeqlenInfoQK(
                    Int32(0),
                    Int32(0),
                    Int32(0),
                    Int32(0),
                    seq_len_q,
                    seq_len_q + causal_q_offset,
                    False,
                    False,
                    False,
                    False,
                )
                mask = AttentionMask(
                    self.m_block_size,
                    self.n_block_size,
                    seqlen_info,
                )
                qidx_meta_slot = (
                    qi_group & Int32(self.qidx_meta_stages - 1)
                ) * Int32(self.q_tokens_per_group)
                if const_expr(self.causal):
                    qi_group_start = (
                        qi_group * Int32(self.q_tokens_per_group)
                    )
                    masked_tok_count = cutlass.max(
                        Int32(0),
                        cutlass.min(
                            Int32(self.q_tokens_per_group),
                            diag_q_count - qi_group_start,
                        ),
                    )
                    if masked_tok_count > Int32(0):
                        tok_idx = group_tidx // Int32(self.qheadperkv)
                        q_idx_mask = self._decode_q_idx_from_qsplit(
                            sQIdxMeta[qidx_meta_slot + tok_idx]
                        )
                        mask.apply_mask_sm100(
                            tSrS,
                            tScS_t2r,
                            m_block=Int32(0),
                            n_block=Int32(0),
                            mask_seqlen=True,
                            mask_causal=True,
                            row_idx=q_idx_mask,
                            kv_valid_cols=kv_valid_cols,
                            kv_block_col_start=kv_block_col_start,
                        )
                    else:
                        mask.apply_mask_sm100(
                            tSrS,
                            tScS_t2r,
                            m_block=Int32(0),
                            n_block=Int32(0),
                            mask_seqlen=True,
                            mask_causal=False,
                            kv_valid_cols=kv_valid_cols,
                        )
                else:
                    mask.apply_mask_sm100(
                        tSrS,
                        tScS_t2r,
                        m_block=Int32(0),
                        n_block=Int32(0),
                        mask_seqlen=True,
                        mask_causal=False,
                        kv_valid_cols=kv_valid_cols,
                    )

                row_max = utils.fmax_reduce(tSrS.load(), arch=100)
                row_max_safe = (
                    row_max if row_max != -Float32.inf else Float32(0.0)
                )
                row_max_scaled = self._probability_max_log2(
                    row_max_safe, softmax_scale_log2
                )
                tok = group_tidx // Int32(self.qheadperkv)
                head_in_kv = group_tidx - tok * Int32(self.qheadperkv)
                qi = qi_group * Int32(self.q_tokens_per_group) + tok
                qsplit = sQIdxMeta[qidx_meta_slot + tok]
                q_idx = self._decode_q_idx_from_qsplit(qsplit)
                split = self._decode_split_idx_from_qsplit(qsplit)
                q_abs = q_batch_offset + q_idx
                head_abs = (
                    head_kv_idx * Int32(self.qheadperkv) + head_in_kv
                )
                global_lse_log2 = (
                    mLSE[q_abs, head_abs] * Float32(math.log2(math.e))
                )
                valid_scale = (
                    row_max != -Float32.inf
                    and row_max == row_max
                    and global_lse_log2 != -Float32.inf
                    and global_lse_log2 == global_lse_log2
                )
                row_scale = Float32(0.0)
                if valid_scale:
                    row_scale = cute.math.exp2(
                        row_max_scaled - global_lse_log2,
                        fastmath=True,
                    )

                # Materialize logical P in place while the independent dP MMA
                # is still running. This overlaps mask/exp2 latency with the
                # tensor-core work instead of serializing the two paths.
                for value_idx in cutlass.range_constexpr(
                    0, cute.size(tSrS), 2
                ):
                    p0, p1 = cute.arch.fma_packed_f32x2(
                        (tSrS[value_idx], tSrS[value_idx + 1]),
                        (softmax_scale_log2, softmax_scale_log2),
                        (-row_max_scaled, -row_max_scaled),
                    )
                    p0 = cute.math.exp2(p0, fastmath=True)
                    p1 = cute.math.exp2(p1, fastmath=True)
                    p0, p1 = cute.arch.mul_packed_f32x2(
                        (p0, p1), (row_scale, row_scale)
                    )
                    tSrS[value_idx] = p0
                    tSrS[value_idx + 1] = p1

                pipeline_dp.consumer_wait_w_index_phase(slot, phase)
                tRrdP = cute.make_rmem_tensor(tScdP_t2r.shape, Float32)
                cute.copy(thr_dp_load, tDPtdP_t2r, tRrdP)
                cute.arch.fence_view_async_tmem_load()
                pipeline_dp.consumer_release_w_index(slot)

                dpsum = Float32(0.0)
                for value_idx in cutlass.range_constexpr(
                    0, cute.size(tSrS), 2
                ):
                    pdp0, pdp1 = cute.arch.mul_packed_f32x2(
                        (tSrS[value_idx], tSrS[value_idx + 1]),
                        (tRrdP[value_idx], tRrdP[value_idx + 1]),
                    )
                    dpsum += pdp0 + pdp1

                if qi < count_raw:
                    stored_row_max_scaled = Float32(0.0)
                    if row_max != -Float32.inf and row_max == row_max:
                        stored_row_max_scaled = (
                            self._probability_max_log2(row_max, softmax_scale_log2)
                        )
                    mPQuantScale[Int32(0), split, q_abs, head_abs] = (
                        stored_row_max_scaled
                    )
                    mPQuantScale[Int32(1), split, q_abs, head_abs] = row_scale
                    q_padded_batch_offset = (
                        q_batch_offset
                        + batch_idx * Int32(self.m_block_size)
                    ) // Int32(self.m_block_size) * Int32(self.m_block_size)
                    mdPsumPartial[
                        split,
                        q_padded_batch_offset + q_idx,
                        head_abs,
                    ] = dpsum


    @cute.jit
    def _load_dout_pair(
        self,
        sDO: cute.Tensor,
        row: Int32,
        real_col: Int32,
    ):
        ptr = cute.make_ptr(
            self.o_dtype,
            sDO.iterator.toint()
            + Int64(
                row * Int32(self.dout_smem_row_stride) + real_col
            ) * Int64(self.o_dtype.width // 8),
            mem_space=sDO.iterator.memspace,
            assumed_align=4,
        )
        return cute.make_tensor(
            ptr,
            cute.make_layout((2,), stride=(1,)),
        ).load().to(Float32)

    @cute.jit
    def _stage_dout(
        self,
        mdO: cute.Tensor,
        sDO: cute.Tensor,
        sQIdx_slot: cute.Tensor,
        qi_group: Int32,
        group_tidx: Int32,
        warp_idx_in_wg: Int32,
        count_raw: Int32,
        q_batch_offset: Int32,
        head_kv_idx: Int32,
        head_q: Int32,
    ):
        lane = group_tidx % Int32(cute.arch.WARP_SIZE)
        lane_in_row = lane % Int32(2)
        row_in_wave = warp_idx_in_wg * Int32(16) + lane // Int32(2)
        dout_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            self.o_dtype,
            num_bits_per_copy=128,
        )
        for row_iter in cutlass.range_constexpr(2):
            load_row = row_in_wave + Int32(row_iter * 64)
            load_tok = load_row // Int32(self.qheadperkv)
            load_row_in_tok = (
                load_row - load_tok * Int32(self.qheadperkv)
            )
            load_qi = qi_group * Int32(self.q_tokens_per_group) + load_tok
            if load_qi < count_raw:
                load_q_idx = sQIdx_slot[load_tok]
                load_h_abs = (
                    head_kv_idx * Int32(self.qheadperkv) + load_row_in_tok
                )
                load_flat_row = (
                    Int64(q_batch_offset + load_q_idx) * Int64(head_q)
                    + Int64(load_h_abs)
                )
                for chunk in cutlass.range_constexpr(self.head_dim // 16):
                    load_col = Int32(chunk * 16) + lane_in_row * Int32(8)
                    gDout_ptr = cute.make_ptr(
                        self.o_dtype,
                        mdO.iterator.toint()
                        + (
                            load_flat_row * Int64(self.head_dim)
                            + Int64(load_col)
                        ) * Int64(self.o_dtype.width // 8),
                        mem_space=mdO.iterator.memspace,
                        assumed_align=16,
                    )
                    gDout = cute.make_tensor(
                        gDout_ptr,
                        cute.make_layout((8,), stride=(1,)),
                    )
                    sDout_ptr = cute.make_ptr(
                        self.o_dtype,
                        sDO.iterator.toint()
                        + Int64(
                            load_row * Int32(self.dout_smem_row_stride)
                            + load_col
                        ) * Int64(self.o_dtype.width // 8),
                        mem_space=sDO.iterator.memspace,
                        assumed_align=16,
                    )
                    sDout = cute.make_tensor(
                        sDout_ptr,
                        cute.make_layout((8,), stride=(1,)),
                    )
                    cute.copy(dout_copy_atom, gDout, sDout)
        cute.arch.cp_async_commit_group()
