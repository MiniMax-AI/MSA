"""SM100 sparse attention forward kernel.

This kernel implements the delivered Sparse Attention / Sparse Page Attention
forward contract:
- CSR sparse metadata (`k2q_row_ptr`, `k2q_q_indices`)
- varlen Q metadata via `cu_seqlens_q`
- Sparse Attention with flat varlen K/V
- Sparse Page Attention with paged K/V
"""

import math
import operator
from dataclasses import dataclass
from functools import partial
from typing import ClassVar, Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as cutlass_pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from msa_v1._common import blackwell_helpers as sm100_helpers
from msa_v1._common import copy_utils, pipeline, utils
from msa_v1._common import mma_sm100_desc as sm100_desc
from msa_v1._common.cute_dsl_utils import (
    ParamsBase,
    assume_tensor_aligned,
    exit_thread_if,
)
from msa_v1._common.mask import AttentionMask
from msa_v1._common.named_barrier import NamedBarrierFwdSm100
from msa_v1._common.paged_kv import PagedKVManager
from msa_v1._common.seqlen_info import SeqlenInfoQK
from msa_v1._common.softmax import SoftmaxSm100
from msa_v1._common.tma_utils import (
    TMA_CACHE_EVICT_LAST,
    make_16x256b_tensor_mn_view,
    prefetch_tma_desc_raw,
    real_col_to_stg128_half_fake_col,
    stg_128_bf16_cs,
    stg_128_f16_cs,
    tma_gather4_cached,
    tma_gather4_prefetch,
)
from msa_v1.attention.fwd.qat import SparseAttentionQatForwardMixin


@dataclass
class _ForwardRowMetadata(ParamsBase):
    """Compact CTA metadata shared by the forward warp roles."""

    _storage: cute.Tensor

    num_work_fields: ClassVar[int] = 9
    num_role_fields: ClassVar[int] = 6
    num_fields: ClassVar[int] = num_work_fields + num_role_fields

    def write_work_info(
        self,
        batch_idx: Int32,
        kv_block_idx: Int32,
        q_begin: Int32,
        q_count: Int32,
        kv_columns: Int32,
        q_batch_offset: Int32,
        kv_batch_offset: Int32,
        kv_head_idx: Int32,
    ) -> None:
        self._storage[0] = batch_idx
        self._storage[1] = kv_block_idx
        self._storage[2] = q_begin
        self._storage[3] = q_count
        self._storage[4] = kv_columns
        self._storage[5] = q_batch_offset
        self._storage[6] = kv_batch_offset
        self._storage[8] = kv_head_idx

    def write_causal_q_offset(self, causal_q_offset: Int32) -> None:
        self._storage[7] = causal_q_offset

    def write_role_validity(self, is_valid: Int32) -> None:
        # Keep one predicate slot per role to isolate setmaxnreg live ranges.
        self._storage[9] = is_valid
        self._storage[10] = is_valid
        self._storage[11] = is_valid
        self._storage[12] = is_valid
        self._storage[13] = is_valid
        self._storage[14] = is_valid

    def batch_idx(self) -> Int32:
        return self._storage[0]

    def kv_block_idx(self) -> Int32:
        return self._storage[1]

    def q_begin(self) -> Int32:
        return self._storage[2]

    def q_count(self) -> Int32:
        return self._storage[3]

    def kv_columns(self) -> Int32:
        return self._storage[4]

    def q_batch_offset(self) -> Int32:
        return self._storage[5]

    def kv_batch_offset(self) -> Int32:
        return self._storage[6]

    def causal_q_offset(self) -> Int32:
        return self._storage[7]

    def kv_head_idx(self) -> Int32:
        return self._storage[8]

    def q_load_active(self) -> Int32:
        return self._storage[9]

    def kv_load_active(self) -> Int32:
        return self._storage[10]

    def mma_active(self) -> Int32:
        return self._storage[11]

    def softmax0_active(self) -> Int32:
        return self._storage[12]

    def softmax1_active(self) -> Int32:
        return self._storage[13]

    def epilogue_active(self) -> Int32:
        return self._storage[14]


class SparseAttentionForwardSm100(SparseAttentionQatForwardMixin):
    """SM100 sparse attention forward kernel."""

    k_tile = 64  # UTCMMA bf16 K-tile (matches sparse_fwd_utcmma.py)

    def __init__(
        self,
        head_dim: int = 128,
        qheadperkv: int = 16,
        m_block_size: int = 128,
        n_block_size: int = 128,
        paged_kv: bool = False,
        page_size: Optional[int] = None,
        has_seqused_k: bool = False,
        causal: bool = False,
        use_prepare_scheduler: bool = True,
        qk_dtype=None,
        pv_dtype=None,
        sparse_attn_p_mode: bool = False,
        dpsum_only: bool = False,
        raw_partial_stats: bool = False,
    ):
        if head_dim != 128:
            raise NotImplementedError(
                f"SparseAttentionForwardSm100 currently supports only D=128, got D={head_dim}"
            )
        self.head_dim = 128
        self.qheadperkv = qheadperkv
        self.use_q_gather4 = qheadperkv in (4, 2, 1)
        if qheadperkv not in (16, 8, 4, 2, 1):
            raise ValueError(
                "SparseAttentionForwardSm100 supports qheadperkv in "
                f"{{1, 2, 4, 8, 16}}, got {qheadperkv}"
            )
        self.tokens_per_gather4 = 4 // qheadperkv if self.use_q_gather4 else 0
        self.m_block_size = m_block_size          # 128 packed Q heads
        self.n_block_size = n_block_size          # 128 KV-block width
        self.paged_kv = paged_kv
        self.page_size = page_size
        self.has_seqused_k = has_seqused_k
        self.causal = causal
        self.qk_dtype_param = qk_dtype
        self.pv_dtype_param = pv_dtype
        self.store_raw_partial = raw_partial_stats
        if not use_prepare_scheduler:
            raise ValueError("SparseAttentionForwardSm100 requires prepare scheduler")
        if self.paged_kv:
            if page_size is None:
                raise ValueError("page_size must be provided when paged_kv=True")
            if page_size != n_block_size:
                raise ValueError(
                    f"page_size ({page_size}) must equal blk_kv ({n_block_size})"
                )
        else:
            self.page_size = n_block_size
        self.q_tokens_per_group = m_block_size // qheadperkv   # 8

        self.mma_tiler_qk = (m_block_size, n_block_size, self.head_dim)
        self.mma_tiler_pv = (m_block_size, self.head_dim, n_block_size)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32

        # Pipeline configuration. FP8 specializes the Q ring to three stages
        # in __call__; BF16 keeps the lower-overhead two-stage ring.
        self.q_stage = 2
        self.s_stage = 2
        self.o_stage = 2
        self.kv_stage = 1
        # Sparse q_idx metadata ring bridging load -> epilogue. Sized larger
        # than the in-flight group distance so epilogue can reuse q_idx
        # without rereading mK2qIndices.
        self.qidx_meta_stages = 16

        self.k_stages = 2
        self.q_stage_stride_bytes = m_block_size * self.head_dim * 2
        self.k_tile_stride_bytes = m_block_size * self.k_tile * 2
        # Shift adjacent BF16 dO rows by four banks. This is cheaper for the
        # epilogue's sparse pair loads than the dense-tile MMA swizzle.
        self.dout_smem_row_stride = self.head_dim + 8

        # The raw target follows the CUDA role split: two softmax WGs,
        # four Q-load warps, one MMA warp, separate K/V TMA warps, and one
        # idle donor warp. The legacy and QAT paths keep their correction WG
        # topology; Q-load warps 13/14 issue K/V before gathering Q.
        self.warps_per_group = 4
        self.softmax0_warp_base = 0
        self.softmax1_warp_base = self.softmax0_warp_base + self.warps_per_group
        self.correction_warp_base = (
            self.softmax1_warp_base + self.warps_per_group
        )
        self.mma_warp_id = self.correction_warp_base + self.warps_per_group
        self.q_load_warp_base = (
            self.correction_warp_base
            if self.store_raw_partial
            else self.mma_warp_id + 1
        )
        self.kv_load_warp_base = self.mma_warp_id + 1
        self.num_kv_load_warps = 2
        self.num_q_load_warps = 4 if self.store_raw_partial else 3
        self.idle_warp_id = 15
        self.total_warps = 16
        self.threads_per_cta = cute.arch.WARP_SIZE * self.total_warps  # 512

        # TMEM layout follows FA SM100:
        #   S0/S1: [0:128], [128:256]
        #   O0/O1: [256:384], [384:512] for hdim_v=128
        #   P dtype follows the PV operand policy and is packed into each S tile.
        tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        self.tmem_s_offset = 0
        self.tmem_stage_stride = n_block_size
        self.tmem_o_stage_stride = self.head_dim
        self.tmem_o_offset = self.s_stage * n_block_size
        self.tmem_s_to_p_offset = n_block_size // 2
        self.tmem_p_offset = self.tmem_s_offset + self.tmem_s_to_p_offset
        raw_tmem_total = self.tmem_o_offset + self.o_stage * self.tmem_o_stage_stride
        # SM100 TMEM allocation requires a power-of-two column count.  The
        # 128-wide path naturally uses 512 columns; 64-wide KV blocks use 384
        # columns and must round the allocation up while keeping the same
        # logical offsets.
        self.tmem_total = 1 << (raw_tmem_total - 1).bit_length()
        assert self.tmem_total <= tmem_alloc_cols

        # A single full-P handoff avoids the second barrier protocol. For one
        # resident 128-key block, the extra synchronization costs more than
        # overlapping the final P slice with the first PV issue.
        self.split_P_arrive = 64

        # Register allocation per role.  The causal hdim128 split gives the
        # epilogue enough room for partial-O/LSE address generation while the
        # two softmax WGs still have enough registers to avoid S/P spills.
        self.dpsum_direct = dpsum_only and qheadperkv == 16
        if self.store_raw_partial:
            self.num_regs_softmax = 176
            self.num_regs_load = 112
            self.num_regs_mma = 48
            self.num_regs_kv_load = 48
            self.num_regs_idle = 48
            self.num_regs_correction = 48
        else:
            self.num_regs_softmax = 176 if causal else 192
            self.num_regs_correction = 96 if causal else 80
            if self.dpsum_direct:
                # Direct logical-D keeps one score row plus a 32-column dP
                # slice live in each softmax thread. The correction WG is
                # unused, so move its budget to the active consumer WGs.
                self.num_regs_softmax = 208
                self.num_regs_correction = 24
            elif dpsum_only:
                # Logical-D carries four row accumulators across both output
                # passes. Shift registers from softmax to the correction WG.
                self.num_regs_softmax -= 8
                self.num_regs_correction += 16
            self.num_regs_other = (
                512 - self.num_regs_softmax * 2 - self.num_regs_correction
            )
            self.num_regs_mma = self.num_regs_other
            self.num_regs_load = self.num_regs_other
            self.num_regs_kv_load = self.num_regs_other
            self.num_regs_idle = self.num_regs_other
        self.ex2_emu_freq = 16 if causal else 0
        self.ex2_emu_start_frg = 1
        self.buffer_align_bytes = 1024
        self.sparse_attn_p_mode = sparse_attn_p_mode
        self.dpsum_only = dpsum_only
        # The qh16 training path uses a dedicated logical-D pipeline.  It
        # computes QK and dO @ V.T as independent UMMAs, then reduces P * dP
        # directly in the softmax warpgroup.  Smaller GQA ratios retain the
        # established PV + dO epilogue until their gather4 producer has an
        # equivalent dual-input pipeline.
        self.do_stage = 2
        if dpsum_only and (paged_kv or sparse_attn_p_mode):
            raise ValueError(
                "dpsum_only requires non-paged logical-P forward"
            )

        # Blackwell config.
        self.cta_group_size = 1
        self.cluster_shape_mnk = (1, 1, 1)

    @cute.jit
    def _batch_q_offset(
        self,
        batch_idx: Int32,
        mCuSeqlensQ,
    ) -> Int32:
        return mCuSeqlensQ[batch_idx]

    @cute.jit
    def _logical_seqlen_k(
        self,
        batch_idx: Int32,
        mPageTable,
        mSeqUsedK,
        mCuSeqlensK,
        mFragmentIndices: Optional[cute.Tensor],
    ) -> Int32:
        if const_expr(self.has_seqused_k):
            return mSeqUsedK[batch_idx]
        if const_expr(self.paged_kv):
            return Int32(mPageTable.shape[1]) * Int32(self.page_size)
        k_start_idx = (
            batch_idx
            if const_expr(mFragmentIndices is None)
            else mFragmentIndices[batch_idx]
        )
        return mCuSeqlensK[batch_idx + Int32(1)] - mCuSeqlensK[k_start_idx]

    @cute.jit
    def _k_batch_offset(
        self,
        batch_idx: Int32,
        mCuSeqlensK: cute.Tensor,
        mFragmentIndices: Optional[cute.Tensor],
    ) -> Int32:
        k_start_idx = (
            batch_idx
            if const_expr(mFragmentIndices is None)
            else mFragmentIndices[batch_idx]
        )
        return mCuSeqlensK[k_start_idx]

    @cute.jit
    def _valid_cols_in_block(
        self,
        batch_idx: Int32,
        kv_block_idx: Int32,
        mPageTable,
        mSeqUsedK,
        mCuSeqlensK,
        mFragmentIndices: Optional[cute.Tensor],
    ) -> Int32:
        seqlen_k = self._logical_seqlen_k(
            batch_idx,
            mPageTable,
            mSeqUsedK,
            mCuSeqlensK,
            mFragmentIndices,
        )
        block_start = kv_block_idx * Int32(self.n_block_size)
        remaining = seqlen_k - block_start
        remaining = cutlass.max(remaining, Int32(0))
        return cutlass.min(remaining, Int32(self.n_block_size))

    @cute.jit
    def _load_q_idx(
        self,
        mK2qIndices: cute.Tensor,
        head_kv_idx: Int32,
        row_start: Int32,
        qi: Int32,
    ) -> Int32:
        return mK2qIndices[head_kv_idx, row_start + qi]

    @cute.jit
    def _load_qsplit_idx(
        self,
        mK2qQSplitIndices: cute.Tensor,
        head_kv_idx: Int32,
        row_start: Int32,
        qi: Int32,
    ) -> Int32:
        return mK2qQSplitIndices[head_kv_idx, row_start + qi]

    @cute.jit
    def _decode_q_idx_from_qsplit(self, qsplit: Int32) -> Int32:
        return qsplit & Int32(0x00FF_FFFF)

    @cute.jit
    def _decode_split_idx_from_qsplit(self, qsplit: Int32) -> Int32:
        return (qsplit >> Int32(24)) & Int32(0xFF)

    @cute.jit
    def _lower_bound_q_idx(
        self,
        mK2qIndices: cute.Tensor,
        head_kv_idx: Int32,
        row_start: Int32,
        count: Int32,
        q_value: Int32,
    ) -> Int32:
        left = Int32(0)
        right = count
        # k2q_q_indices is sorted by q_idx within each CSR row. A fixed
        # 32-step loop covers int32-sized rows and keeps this CTA-level.
        for _ in cutlass.range(32, unroll=1):
            if left < right:
                mid = (left + right) // Int32(2)
                q_idx = self._load_q_idx(
                    mK2qIndices,
                    head_kv_idx,
                    row_start,
                    mid,
                )
                if q_idx < q_value:
                    left = mid + Int32(1)
                else:
                    right = mid
        return left

    # ------------------------------------------------------------------
    # Host-side: TMA descriptors, SMEM layout, launch
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        mK: cute.Tensor,            # Sparse Attention: [total_k, head_kv, dim] / Sparse Page Attention: prepared paged KV tensor
        mV: cute.Tensor,            # Sparse Attention: [total_k, head_kv, dim] / Sparse Page Attention: prepared paged KV tensor
        mK2qIndices: cute.Tensor,   # csr payload: [head_kv, nnz]
        mK2qQSplitIndices: cute.Tensor,  # csr payload: [head_kv, nnz] packed q_idx/split slot
        mK2qCounts: cute.Tensor,    # csr row_ptr: [head_kv, total_rows + 1]
        mSchedulerMetadata: Optional[cute.Tensor],
        mWorkCount: Optional[cute.Tensor],
        mO_partial: cute.Tensor,    # Partial output in the configured storage dtype
        mLSE_partial: cute.Tensor,  # fp32 LSE partial or raw (scaled_max_log2, row_sum)
        mLSE_temperature_partial: Optional[cute.Tensor],  # fp32 temperature-scaled LSE_partial
        mPQuantScale: Optional[cute.Tensor],  # [2, topK, total_q, head_q] FP32 QAT scratch
        mQ_flat: cute.Tensor,       # [batch*Sq*head_q, dim], pre-flattened
        mQ_gather4_desc: Optional[cute.Tensor],  # [128] uint8 tensor map for gather4 Q load
        mPageTable,
        mSeqUsedK,
        mCuSeqlensQ,
        mCuSeqlensK,
        mFragmentIndices: Optional[cute.Tensor],
        softmax_scale: Float32,
        lse_temperature_scale: Float32,
        num_heads_kv: Int32,
        seq_len_q: Int32,
        work_capacity: Int32,
        stream=None,
    ):
        self.q_dtype = mQ_flat.element_type
        self.q_stage = (
            4
            if const_expr(self.store_raw_partial)
            else (
                3
                if const_expr(
                    self.q_dtype == cutlass.Float8E4M3FN
                    and not self.dpsum_only
                )
                else 2
            )
        )
        self.k_input_dtype = mK.element_type
        self.v_input_dtype = mV.element_type
        self.qk_dtype = (
            self.q_dtype if const_expr(self.qk_dtype_param is None) else self.qk_dtype_param
        )
        if const_expr(self.pv_dtype_param is None):
            legacy_fp8_kv_cache = (
                self.q_dtype == cutlass.BFloat16
                and self.k_input_dtype == cutlass.Float8E4M3FN
                and self.v_input_dtype == cutlass.Float8E4M3FN
            )
            self.pv_dtype = cutlass.BFloat16 if legacy_fp8_kv_cache else self.v_input_dtype
        else:
            self.pv_dtype = self.pv_dtype_param
        self.k_dtype = self.qk_dtype
        self.v_dtype = self.pv_dtype
        self.p_dtype = self.pv_dtype
        self.scale_fp8_p = const_expr(
            self.p_dtype == cutlass.Float8E4M3FN
            or self.sparse_attn_p_mode
            or self.dpsum_only
        )
        if const_expr(self.q_dtype not in [cutlass.BFloat16, cutlass.Float8E4M3FN]):
            raise TypeError(f"Unsupported Q/K/V dtype: {self.q_dtype}")
        if const_expr(self.qk_dtype not in [cutlass.BFloat16, cutlass.Float8E4M3FN]):
            raise TypeError(f"Unsupported qk_dtype: {self.qk_dtype}")
        if const_expr(self.pv_dtype not in [cutlass.BFloat16, cutlass.Float8E4M3FN]):
            raise TypeError(f"Unsupported pv_dtype: {self.pv_dtype}")
        if const_expr(self.q_dtype != self.qk_dtype):
            raise TypeError("Q storage dtype must match qk_dtype")
        if const_expr(
            self.k_input_dtype != self.k_dtype
            and not (self.k_input_dtype == cutlass.Float8E4M3FN and self.k_dtype == cutlass.BFloat16)
        ):
            raise TypeError("Only FP8 K -> BF16 QK staging is supported")
        if const_expr(
            self.v_input_dtype != self.v_dtype
            and not (self.v_input_dtype == cutlass.Float8E4M3FN and self.v_dtype == cutlass.BFloat16)
        ):
            raise TypeError("Only FP8 V -> BF16 PV staging is supported")
        self.k_fp8_to_bf16 = (
            self.k_input_dtype == cutlass.Float8E4M3FN
            and self.k_dtype == cutlass.BFloat16
        )
        self.v_fp8_to_bf16 = (
            self.v_input_dtype == cutlass.Float8E4M3FN
            and self.v_dtype == cutlass.BFloat16
        )
        self.kv_fp8_to_bf16 = self.k_fp8_to_bf16 or self.v_fp8_to_bf16
        self.qk_mma_kind = "f8f6f4" if const_expr(self.qk_dtype.width == 8) else "f16"
        self.pv_mma_kind = "f8f6f4" if const_expr(self.pv_dtype.width == 8) else "f16"
        elem_bytes = const_expr(self.q_dtype.width // 8)
        self.q_stage_stride_bytes = self.m_block_size * self.head_dim * elem_bytes
        self.k_tile_stride_bytes = self.m_block_size * self.k_tile * elem_bytes
        p_cols_as_fp32 = const_expr(self.n_block_size * self.p_dtype.width // Float32.width)
        self.tmem_s_to_p_offset = self.n_block_size - p_cols_as_fp32
        self.tmem_p_offset = self.tmem_s_offset + self.tmem_s_to_p_offset
        self.o_dtype = mO_partial.element_type
        if const_expr(
            self.o_dtype not in [cutlass.BFloat16, cutlass.Float16]
        ):
            raise TypeError(
                f"O_partial dtype must be BFloat16 or Float16, got {self.o_dtype}"
            )
        if const_expr(self.dpsum_only):
            qk_dtype_supported = (
                (
                    self.q_dtype == cutlass.BFloat16
                    and self.k_dtype == cutlass.BFloat16
                )
                or (
                    self.q_dtype == cutlass.Float8E4M3FN
                    and self.k_dtype == cutlass.Float8E4M3FN
                )
            )
            if const_expr(
                not qk_dtype_supported
                or self.v_dtype != cutlass.BFloat16
                or self.o_dtype != cutlass.BFloat16
            ):
                raise TypeError(
                    "dpsum_only requires matching BF16 or E4M3 Q/K and "
                    "BF16 V/dO"
                )
            if const_expr(
                mLSE_partial.element_type != Float32
                or mLSE_temperature_partial is None
                or mLSE_temperature_partial.element_type != Float32
                or mPQuantScale is None
                or mPQuantScale.element_type != Float32
            ):
                raise TypeError(
                    "dpsum_only requires FP32 LSE, dPsum, and P-scale scratch"
                )
        else:
            if const_expr(mLSE_partial.element_type != Float32):
                raise TypeError("LSE partial or raw stats tensor must be Float32")
            if const_expr(len(mLSE_partial.shape) not in [3, 4]):
                raise ValueError(
                    "Partial stats tensor must have shape "
                    "[split, total_q, head] or [split, total_q, head, 2]"
                )
            if const_expr(
                self.store_raw_partial and len(mLSE_partial.shape) != 4
            ):
                raise ValueError("Raw partial stats must have rank 4")
            if const_expr(
                not self.store_raw_partial and len(mLSE_partial.shape) != 3
            ):
                raise ValueError(
                    "Normalized partial LSE must have shape "
                    "[split, total_q, head]"
                )
            if const_expr(
                self.store_raw_partial
                and mLSE_temperature_partial is not None
            ):
                raise ValueError(
                    "Raw partial stats do not support independent temperature LSE"
                )
        mK, mV = [assume_tensor_aligned(t) for t in (mK, mV)]

        if const_expr(not self.paged_kv):
            # Flat varlen K/V use CUTE-managed TMA descriptors, matching FA:
            # K: [total_k, h, d] -> [total_k, d, h].
            # V: [total_k, h, d] -> [d, total_k, h] for MN-major PV.
            layout_t = [0, 2, 1]
            mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=layout_t))
            mV_kv = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=layout_t))
            mV = (
                mV_kv
                if const_expr(self.dpsum_direct)
                else cute.make_tensor(
                    mV_kv.iterator,
                    cute.select(mV_kv.layout, mode=[1, 0, 2]),
                )
            )
        else:
            # Sparse Page Attention with page-sized blocks can use the blocked
            # paged TMA layout directly. Host input is [page, head, token, dim].
            layout_t = [2, 3, 1, 0]
            mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=layout_t))
            mV_kv = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=layout_t))
            # V: (s,d,h,b) -> (d,s,h,b) for MN-major
            mV = cute.make_tensor(mV_kv.iterator, cute.select(mV_kv.layout, mode=[1, 0, 2, 3]))

        # ------------------------------------------------------------------
        #  UTCMMA TiledMma: QK^T and PV
        # ------------------------------------------------------------------
        cta_group = tcgen05.CtaGroup.ONE
        tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        if const_expr(self.dpsum_direct):
            tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
                self.v_dtype,
                OperandMajorMode.K,
                OperandMajorMode.K,
                Float32,
                cta_group,
                self.mma_tiler_qk[:2],
            )
        else:
            tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
                self.v_dtype,
                OperandMajorMode.K,
                OperandMajorMode.MN,
                Float32,
                cta_group,
                self.mma_tiler_pv[:2],
                tcgen05.OperandSource.TMEM,
            )

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,))

        # ------------------------------------------------------------------
        #  SMEM layouts: sQ/sK/sV only. O_partial is written directly from
        #  registers to GMEM in the epilogue.
        # ------------------------------------------------------------------
        total_q_stages = self.q_stage
        sQ_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, total_q_stages)
        q_load_tile = (
            self.head_dim
            if const_expr(self.q_dtype == cutlass.Float8E4M3FN)
            else self.k_tile
        )
        q_load_subtiles_per_token = const_expr(self.head_dim // q_load_tile)
        num_subtiles_total = (
            total_q_stages * self.q_tokens_per_group * q_load_subtiles_per_token
        )
        sQ_load_layout = sm100_utils.make_smem_layout(
            OperandMajorMode.K,
            (self.qheadperkv, q_load_tile),
            self.q_dtype,
            num_subtiles_total,
        )
        sDO_layout = cute.make_layout(
            (self.m_block_size, self.head_dim),
            stride=(self.dout_smem_row_stride, 1),
        )
        sDO_mma_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_pv,
            self.mma_tiler_qk,
            self.o_dtype,
            self.do_stage,
        )
        do_load_subtiles_per_token = self.head_dim // self.k_tile
        sDO_load_layout = sm100_utils.make_smem_layout(
            OperandMajorMode.K,
            (self.qheadperkv, self.k_tile),
            self.o_dtype,
            self.do_stage
            * self.q_tokens_per_group
            * do_load_subtiles_per_token,
        )
        sK_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage)
        sV_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage)
        sK_fp8_layout = cute.append(
            cute.make_layout(
                (self.n_block_size, self.head_dim),
                stride=(self.head_dim, 1),
            ),
            cute.make_layout((1,)),
        )
        sV_fp8_layout = cute.append(
            cute.make_layout(
                (self.head_dim, self.n_block_size),
                stride=(1, self.head_dim),
            ),
            cute.make_layout((1,)),
        )
        # P SMEM layout metadata (no actual SMEM allocation — P lives in TMEM,
        # overlaying the S region; this layout is only used to compute the PV
        # A-operand TMEM descriptor shape at the MMA issue site.)
        tP_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.p_dtype, self.s_stage)

        # ------------------------------------------------------------------
        #  TMA atoms
        # ------------------------------------------------------------------
        q_tma_bytes = cute.size_in_bytes(
            self.q_dtype, cute.select(sQ_layout, mode=[0, 1, 2]))
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        if const_expr(self.k_fp8_to_bf16):
            tma_atom_K, mK = cpasync.make_tiled_tma_atom(
                tma_load_op,
                mK,
                cute.select(sK_fp8_layout, mode=[0, 1]),
                (self.n_block_size, self.head_dim),
            )
        else:
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op, mK, cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk, tiled_mma_qk, cta_layout_vmnk.shape)
        if const_expr(self.v_fp8_to_bf16):
            tma_atom_V, mV = cpasync.make_tiled_tma_atom(
                tma_load_op,
                mV,
                cute.select(sV_fp8_layout, mode=[0, 1]),
                (self.head_dim, self.n_block_size),
            )
        else:
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op, mV, cute.select(sV_layout, mode=[0, 1, 2]),
                self.mma_tiler_pv, tiled_mma_pv, cta_layout_vmnk.shape)

        # Q per-sub-tile TMA atom: bf16 uses two 64-element halves; fp8 uses
        # one 128-element row because 128 fp8 elements occupy the same 128B
        # swizzle span as 64 bf16 elements.
        mQ_flat = assume_tensor_aligned(mQ_flat)
        mQ_2d = cute.make_tensor(
            mQ_flat.iterator, cute.select(mQ_flat.layout, mode=[0, 1]))
        if const_expr(self.use_q_gather4):
            # Placeholder atom for unified kernel signature. Small-GQA Q load
            # uses raw gather4 and keeps mQ_2d as a plain row-major GMEM tensor.
            tma_atom_Q = tma_atom_V
        else:
            tma_atom_Q, mQ_2d = cpasync.make_tiled_tma_atom(
                tma_load_op, mQ_2d,
                cute.select(sQ_load_layout, mode=[0, 1]),
                (self.qheadperkv, q_load_tile))
        if const_expr(self.dpsum_direct):
            mDO_2d = cute.make_tensor(
                mO_partial.iterator,
                cute.select(mO_partial.layout, mode=[0, 1]),
            )
            tma_atom_DO, mDO_2d = cpasync.make_tiled_tma_atom(
                tma_load_op,
                mDO_2d,
                cute.select(sDO_load_layout, mode=[0, 1]),
                (self.qheadperkv, self.k_tile),
            )
            do_subtile_bytes = cute.size_in_bytes(
                self.o_dtype,
                cute.select(sDO_load_layout, mode=[0, 1]),
            )
            do_tma_bytes = (
                self.q_tokens_per_group
                * do_load_subtiles_per_token
                * do_subtile_bytes
            )
        else:
            mDO_2d = mQ_2d
            tma_atom_DO = tma_atom_Q
            do_tma_bytes = q_tma_bytes

        softmax_scale_log2 = softmax_scale * Float32(math.log2(math.e))
        lse_temperature_scale_log2 = softmax_scale_log2 * lse_temperature_scale

        # ------------------------------------------------------------------
        #  SharedStorage — lean: just the mbars and tiles we actually use.
        #
        #  Mbarriers (all storage rings stay below the 64-per-CTA limit):
        #    mbar_kv           [2]  one-shot K/V load handshake (full + empty)
        #    mbar_q            [q_stage * 2]  Q producer/consumer ring
        #    mbar_s            [2]  QK UTCMMA -> softmax (full + empty)
        #    mbar_o            [2]  PV UTCMMA -> epilogue (full + empty)
        #    mbar_p            [s_stage * 2]  softmax full-P arrive -> PV
        #    mbar_p_lastsplit  [s_stage * 2]  reserved helper-compatible ring
        #    mbar_sm_stats     [s_stage * 2]  softmax row_sum/row_max
        #                      publish -> epilogue consumer read.  In lean
        #                      1-WG topology the producer and consumer are
        #                      the same 128 WG_C threads, but we keep the
        #                      barrier for structural parity with FA so the
        #                      softmax body reads identically.
        # ------------------------------------------------------------------
        @cute.struct
        class SharedStorage:
            mbar_k: cute.struct.MemRange[Int64, 2]
            mbar_v: cute.struct.MemRange[Int64, 2]
            if const_expr(self.k_fp8_to_bf16):
                mbar_k_tma: cute.struct.MemRange[Int64, 2]
            if const_expr(self.v_fp8_to_bf16):
                mbar_v_tma: cute.struct.MemRange[Int64, 2]
            mbar_q: cute.struct.MemRange[Int64, self.q_stage * 2]
            if const_expr(self.dpsum_direct):
                mbar_do: cute.struct.MemRange[Int64, self.do_stage * 2]
            mbar_s: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_p: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_p_lastsplit: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_o: cute.struct.MemRange[Int64, self.o_stage * 2]
            if const_expr(not self.store_raw_partial):
                mbar_sm_stats: cute.struct.MemRange[Int64, self.o_stage * 2]
            tmem_dealloc_mbar_ptr: Int64
            tmem_holding_buf: Int32
            if const_expr(not self.store_raw_partial):
                # Legacy normalized-partial stats handoff.
                sScale: cute.struct.MemRange[
                    Float32, self.o_stage * self.m_block_size * 2]
                sScaleTemperature: cute.struct.MemRange[
                    Float32, self.o_stage * self.m_block_size]
                sSplitIdx: cute.struct.MemRange[
                    Int32, self.o_stage * self.q_tokens_per_group]
                sQIdx: cute.struct.MemRange[
                    Int32, self.o_stage * self.q_tokens_per_group]
            # Prefix length of q_idx-sorted row entries that may need causal
            # masking for this KV block. This is CTA-level metadata, not a
            # token-count cap.
            sDiagQCount: cute.struct.MemRange[Int32, 1]
            # Compact CTA metadata published by tidx 0. `_ForwardRowMetadata`
            # owns the physical layout and exposes semantic accessors.
            sRowMeta: cute.struct.MemRange[
                Int32, _ForwardRowMetadata.num_fields
            ]
            sPagedKvIdx: cute.struct.MemRange[Int32, 1]
            # Packed per-edge q/split metadata:
            #   low 24 bits = q_idx, high 8 bits = split slot.
            sQIdxMeta: cute.struct.MemRange[
                Int32, self.qidx_meta_stages * self.q_tokens_per_group]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(sV_layout)],
                self.buffer_align_bytes]
            if const_expr(self.k_fp8_to_bf16):
                sKFp8: cute.struct.Align[
                    cute.struct.MemRange[
                        self.k_input_dtype, cute.cosize(sK_fp8_layout)
                    ],
                    self.buffer_align_bytes]
            if const_expr(self.v_fp8_to_bf16):
                sVFp8: cute.struct.Align[
                    cute.struct.MemRange[
                        self.v_input_dtype, cute.cosize(sV_fp8_layout)
                    ],
                    self.buffer_align_bytes]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)],
                self.buffer_align_bytes]
            if const_expr(self.dpsum_only):
                if const_expr(self.dpsum_direct):
                    sDO: cute.struct.Align[
                        cute.struct.MemRange[
                            cutlass.BFloat16,
                            cute.cosize(sDO_mma_layout),
                        ],
                        self.buffer_align_bytes,
                    ]
                else:
                    sDO: cute.struct.Align[
                        cute.struct.MemRange[
                            cutlass.BFloat16,
                            self.m_block_size * self.dout_smem_row_stride,
                        ],
                        16,
                    ]
        self.shared_storage = SharedStorage
        num_ctas = work_capacity

        self.kernel(
            mK, mV, mK2qIndices, mK2qQSplitIndices, mK2qCounts,
            mSchedulerMetadata, mWorkCount,
            mO_partial, mLSE_partial, mLSE_temperature_partial, mPQuantScale,
            mQ_2d, mQ_gather4_desc, mDO_2d,
            mPageTable, mSeqUsedK, mCuSeqlensQ, mCuSeqlensK,
            mFragmentIndices,
            softmax_scale_log2, lse_temperature_scale_log2, lse_temperature_scale,
            sQ_layout, sQ_load_layout, sDO_layout, sDO_mma_layout,
            sDO_load_layout, sK_layout, sV_layout,
            sK_fp8_layout, sV_fp8_layout, tP_layout,
            tma_atom_K, tma_atom_V, tma_atom_Q, tma_atom_DO,
            tiled_mma_qk, tiled_mma_pv,
            q_tma_bytes, do_tma_bytes,
            num_heads_kv, seq_len_q,
        ).launch(
            grid=(num_ctas,),
            block=[self.threads_per_cta, 1, 1],
            smem=max(SharedStorage.size_in_bytes(), 49152),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # ------------------------------------------------------------------
    # Device-side: kernel entry, dispatch by warpgroup
    # ------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        # Runtime tensors
        tma_K: cute.Tensor,
        tma_V: cute.Tensor,
        mK2qIndices: cute.Tensor,
        mK2qQSplitIndices: cute.Tensor,
        mK2qCounts: cute.Tensor,
        mSchedulerMetadata: Optional[cute.Tensor],
        mWorkCount: Optional[cute.Tensor],
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mLSE_temperature_partial: Optional[cute.Tensor],
        mPQuantScale: Optional[cute.Tensor],
        mQ_2d: cute.Tensor,
        mQ_gather4_desc: Optional[cute.Tensor],
        mDO_2d: cute.Tensor,
        mPageTable,
        mSeqUsedK,
        mCuSeqlensQ,
        mCuSeqlensK,
        mFragmentIndices: Optional[cute.Tensor],
        # Scalars
        softmax_scale_log2: Float32,
        lse_temperature_scale_log2: Float32,
        lse_temperature_scale: Float32,
        # Layouts
        sQ_layout: cute.ComposedLayout,
        sQ_load_layout: cute.ComposedLayout,
        sDO_layout: cute.Layout,
        sDO_mma_layout: cute.ComposedLayout,
        sDO_load_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sK_fp8_layout: cute.Layout,
        sV_fp8_layout: cute.Layout,
        tP_layout: cute.ComposedLayout,
        # TMA atoms
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_DO: cute.CopyAtom,
        # MMA
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        # Transfer sizes
        q_tma_bytes: cutlass.Constexpr[int],
        do_tma_bytes: cutlass.Constexpr[int],
        # Iteration bounds
        num_heads_kv: Int32,
        seq_len_q: Int32,
    ):
        # ------------------------------------------------------------------
        #  Thread / warp identity, CTA coordinate
        # ------------------------------------------------------------------
        bidx, _, _ = cute.arch.block_idx()
        exit_thread_if(Int32(bidx >= mWorkCount[Int32(0)]))
        row_linear = Int32(0)
        head_kv_idx = Int32(0)
        batch_idx = Int32(0)
        kv_block_idx = Int32(0)
        work_q_begin = Int32(0)
        work_q_count = Int32(0)
        cta_valid_work = True
        work_idx = bidx
        cta_valid_work = work_idx < mWorkCount[Int32(0)]
        if cta_valid_work:
            head_kv_idx = mSchedulerMetadata[work_idx, Int32(0)]
            row_linear = mSchedulerMetadata[work_idx, Int32(1)]
            work_q_begin = mSchedulerMetadata[work_idx, Int32(2)]
            work_q_count = mSchedulerMetadata[work_idx, Int32(3)]
            batch_idx = mSchedulerMetadata[work_idx, Int32(4)]
            kv_block_idx = mSchedulerMetadata[work_idx, Int32(5)]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0]
        head_q = num_heads_kv * Int32(self.qheadperkv)
        paged_kv_manager = (
            PagedKVManager.create(
                mPageTable,
                page_size=self.page_size,
                n_block_size=self.n_block_size,
            )
            if const_expr(self.paged_kv)
            else None
        )

        # Prefetch TMA descriptors (warp 0 once).
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(not self.use_q_gather4):
                cpasync.prefetch_descriptor(tma_atom_Q)
            else:
                with cute.arch.elect_one():
                    prefetch_tma_desc_raw(mQ_gather4_desc.iterator)
            if const_expr(self.dpsum_direct):
                cpasync.prefetch_descriptor(tma_atom_DO)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,))

        # ------------------------------------------------------------------
        #  SMEM allocation (all warps — same SharedStorage type from __call__)
        # ------------------------------------------------------------------
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        if const_expr(self.k_fp8_to_bf16):
            sKFp8 = storage.sKFp8.get_tensor(sK_fp8_layout)
            mbar_k_tma_ptr = storage.mbar_k_tma.data_ptr()
        if const_expr(self.v_fp8_to_bf16):
            sVFp8 = storage.sVFp8.get_tensor(sV_fp8_layout)
            mbar_v_tma_ptr = storage.mbar_v_tma.data_ptr()
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sQ_load = storage.sQ.get_tensor(sQ_load_layout.outer, swizzle=sQ_load_layout.inner)
        if const_expr(self.dpsum_only):
            if const_expr(self.dpsum_direct):
                sDO = storage.sDO.get_tensor(
                    sDO_mma_layout.outer,
                    swizzle=sDO_mma_layout.inner,
                )
                sDO_load = storage.sDO.get_tensor(
                    sDO_load_layout.outer,
                    swizzle=sDO_load_layout.inner,
                )
            else:
                sDO = storage.sDO.get_tensor(sDO_layout)
        sScale = None
        sScaleTemperature = None
        sSplitIdx = None
        sQIdx = None
        if const_expr(not self.store_raw_partial):
            sScale = storage.sScale.get_tensor(
                cute.make_layout(self.o_stage * self.m_block_size * 2))
            sScaleTemperature = storage.sScaleTemperature.get_tensor(
                cute.make_layout(self.o_stage * self.m_block_size))
            sSplitIdx = storage.sSplitIdx.get_tensor(
                cute.make_layout((self.o_stage * self.q_tokens_per_group,)))
            sQIdx = storage.sQIdx.get_tensor(
                cute.make_layout((self.o_stage * self.q_tokens_per_group,)))
        sDiagQCount = storage.sDiagQCount.get_tensor(cute.make_layout((1,)))
        sRowMeta = storage.sRowMeta.get_tensor(
            cute.make_layout((_ForwardRowMetadata.num_fields,))
        )
        row_metadata = _ForwardRowMetadata(sRowMeta)
        sPagedKvIdx = storage.sPagedKvIdx.get_tensor(cute.make_layout((1,)))
        sQIdxMeta = storage.sQIdxMeta.get_tensor(
            cute.make_layout((self.qidx_meta_stages * self.q_tokens_per_group,)))
        mbar_k_ptr = storage.mbar_k.data_ptr()
        mbar_v_ptr = storage.mbar_v.data_ptr()

        # ------------------------------------------------------------------
        #  TMEM allocator — allocator warp 0 serves the whole CTA.
        # ------------------------------------------------------------------
        tmem_alloc_warps: cutlass.Constexpr[int] = (
            self.warps_per_group * (2 if self.store_raw_partial else 3) + 1
        )
        tmem_alloc_threads = cute.arch.WARP_SIZE * tmem_alloc_warps
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=tmem_alloc_threads,
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # ------------------------------------------------------------------
        #  Warp-specialized pipelines.
        # ------------------------------------------------------------------
        ThreadCooperativeGroup = partial(
            cutlass_pipeline.CooperativeGroup, cutlass_pipeline.Agent.Thread)
        q_tma_threads = ThreadCooperativeGroup(
            1 + self.q_tokens_per_group
        )
        do_tma_threads = ThreadCooperativeGroup(1)
        mma_thread = ThreadCooperativeGroup(1)
        softmax_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * self.warps_per_group)
        epilogue_threads = softmax_threads

        pipeline_q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_q.data_ptr(),
            num_stages=self.q_stage,
            producer_group=q_tma_threads,
            consumer_group=mma_thread,
            tx_count=q_tma_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        if const_expr(self.dpsum_direct):
            pipeline_do = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.mbar_do.data_ptr(),
                num_stages=self.do_stage,
                producer_group=do_tma_threads,
                consumer_group=mma_thread,
                tx_count=do_tma_bytes,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        pipeline_s = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_s.data_ptr(),
            num_stages=self.s_stage,
            producer_group=mma_thread,
            consumer_group=softmax_threads,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_p.data_ptr(),
            num_stages=self.s_stage,
            producer_group=softmax_threads,
            consumer_group=mma_thread,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_p_lastsplit.data_ptr(),
            num_stages=self.s_stage,
            producer_group=softmax_threads,
            consumer_group=mma_thread,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_o = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_o.data_ptr(),
            num_stages=self.o_stage,
            producer_group=mma_thread,
            consumer_group=epilogue_threads,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_sm_stats = None
        if const_expr(not self.store_raw_partial):
            pipeline_sm_stats = pipeline.PipelineAsync.create(
                barrier_storage=storage.mbar_sm_stats.data_ptr(),
                num_stages=self.o_stage,
                producer_group=softmax_threads,
                consumer_group=epilogue_threads,
                defer_sync=True,
            )

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)
        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS_base = thr_mma_qk.make_fragment_C(qk_acc_shape)
        tStS = cute.make_tensor(
            tStS_base.iterator,
            cute.append(
                tStS_base.layout,
                cute.make_layout(
                    (self.s_stage,), stride=(self.tmem_stage_stride,)),
            ),
        )
        tDPtDP = cute.make_tensor(
            tStS_base.iterator + self.tmem_o_offset,
            cute.append(
                tStS_base.layout,
                cute.make_layout(
                    (self.o_stage,), stride=(self.tmem_o_stage_stride,)
                ),
            ),
        )
        if const_expr(self.dpsum_direct):
            tOrP = tStS
        else:
            tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
            tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
            tP_width_ratio = Float32.width // self.v_dtype.width
            tP_stage_stride = self.tmem_stage_stride * tP_width_ratio
            tOrP = cute.make_tensor(
                tOrP.iterator + self.tmem_p_offset * tP_width_ratio,
                cute.append(
                    tOrP.layout,
                    cute.make_layout(
                        (self.s_stage,), stride=(tP_stage_stride,)
                    ),
                ),
            )
        tmem_cols = self.tmem_total

        load_wg_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.LoadWG),
            num_threads=cute.arch.WARP_SIZE * self.num_q_load_warps,
        )
        if const_expr(self.kv_fp8_to_bf16):
            kv_load_barrier = pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierFwdSm100.KvLoad),
                num_threads=cute.arch.WARP_SIZE * self.num_kv_load_warps,
            )
        if const_expr(self.k_fp8_to_bf16):
            kv_dequant_k_barrier = pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierFwdSm100.KvDequantK),
                num_threads=cute.arch.WARP_SIZE * self.warps_per_group,
            )
        if const_expr(self.v_fp8_to_bf16):
            kv_dequant_v_barrier = pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierFwdSm100.KvDequantV),
                num_threads=cute.arch.WARP_SIZE * self.warps_per_group,
            )
        sm_stats_barrier = None
        epilogue_barrier = None
        if const_expr(not self.store_raw_partial):
            sm_stats_barrier = pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0),
                num_threads=cute.arch.WARP_SIZE * 2,
            )
            epilogue_barrier = pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierFwdSm100.StoreEpilogue),
                num_threads=cute.arch.WARP_SIZE * self.warps_per_group,
            )

        k_tma_bytes = cute.size_in_bytes(
            self.k_input_dtype,
            cute.select(sK_fp8_layout, mode=[0, 1])
            if const_expr(self.k_fp8_to_bf16)
            else cute.select(sK_layout, mode=[0, 1, 2]))
        v_tma_bytes = cute.size_in_bytes(
            self.v_input_dtype,
            cute.select(sV_fp8_layout, mode=[0, 1])
            if const_expr(self.v_fp8_to_bf16)
            else cute.select(sV_layout, mode=[0, 1, 2]))

        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        # ------------------------------------------------------------------
        #  Work count: how many Q tokens reference this CTA's KV block
        # ------------------------------------------------------------------
        if tidx == 0:
            row_batch_idx = batch_idx
            row_kv_block_idx = kv_block_idx
            base_row_start = mK2qCounts[head_kv_idx, row_linear]
            row_start = base_row_start + work_q_begin
            count_raw = work_q_count
            kv_valid_cols = (
                self._valid_cols_in_block(
                    row_batch_idx,
                    row_kv_block_idx,
                    mPageTable,
                    mSeqUsedK,
                    mCuSeqlensK,
                    mFragmentIndices,
                )
            )
            q_batch_offset = self._batch_q_offset(
                row_batch_idx, mCuSeqlensQ
            )
            k_batch_offset = (
                Int32(0)
                if const_expr(self.paged_kv)
                else self._k_batch_offset(
                    row_batch_idx,
                    mCuSeqlensK,
                    mFragmentIndices,
                )
            )
            row_metadata.write_work_info(
                row_batch_idx,
                row_kv_block_idx,
                row_start,
                count_raw,
                kv_valid_cols,
                q_batch_offset,
                k_batch_offset,
                head_kv_idx,
            )
            role_valid = Int32(1) if cta_valid_work else Int32(0)
            row_metadata.write_role_validity(role_valid)
            causal_q_offset = Int32(0)
            if const_expr(self.causal):
                seqlen_q = (
                    mCuSeqlensQ[row_batch_idx + Int32(1)]
                    - q_batch_offset
                )
                seqlen_k = self._logical_seqlen_k(
                    row_batch_idx,
                    mPageTable,
                    mSeqUsedK,
                    mCuSeqlensK,
                    mFragmentIndices,
                )
                causal_q_offset = seqlen_k - seqlen_q
            row_metadata.write_causal_q_offset(causal_q_offset)
            if const_expr(self.paged_kv):
                sPagedKvIdx[0] = paged_kv_manager.physical_block_index(
                    row_batch_idx, row_kv_block_idx
                )
            cute.arch.mbarrier_init(mbar_k_ptr, 1)
            cute.arch.mbarrier_init(mbar_v_ptr, 1)
            if const_expr(self.k_fp8_to_bf16):
                cute.arch.mbarrier_init(mbar_k_tma_ptr, 1)
                cute.arch.mbarrier_expect_tx(
                    mbar_k_tma_ptr, k_tma_bytes
                )
            else:
                cute.arch.mbarrier_expect_tx(mbar_k_ptr, k_tma_bytes)
            if const_expr(self.v_fp8_to_bf16):
                cute.arch.mbarrier_init(mbar_v_tma_ptr, 1)
                cute.arch.mbarrier_expect_tx(
                    mbar_v_tma_ptr, v_tma_bytes
                )
            else:
                cute.arch.mbarrier_expect_tx(mbar_v_ptr, v_tma_bytes)
            diag_q_count = Int32(0)
            if const_expr(self.causal):
                row_has_visible_cols = (
                    (count_raw > Int32(0))
                    & (kv_valid_cols > Int32(0))
                )
                if row_has_visible_cols:
                    kv_valid_end = (
                        row_kv_block_idx * Int32(self.n_block_size)
                        + kv_valid_cols
                    )
                    q_threshold = kv_valid_end - causal_q_offset
                    diag_q_count = self._lower_bound_q_idx(
                        mK2qIndices,
                        head_kv_idx,
                        row_start,
                        count_raw,
                        q_threshold,
                    )
            sDiagQCount[0] = diag_q_count
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        if const_expr(self.store_raw_partial):
            if warp_idx == Int32(self.idle_warp_id):
                cute.arch.setmaxregister_decrease(self.num_regs_idle)
        q_load_thread_base = Int32(self.q_load_warp_base * cute.arch.WARP_SIZE)
        q_load_thread_end = Int32(
            (self.q_load_warp_base + self.num_q_load_warps) * cute.arch.WARP_SIZE
        )
        is_q_load_thread = tidx >= q_load_thread_base and tidx < q_load_thread_end
        if is_q_load_thread and row_metadata.q_load_active() != Int32(0):
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            head_kv_idx_load = row_metadata.kv_head_idx()
            # K/V must be issued before these warps enter the bounded Q
            # producer ring. Otherwise the MMA warp can wait for K while the
            # Q producers wait for a consumer and form a circular wait.
            if (
                const_expr(not self.store_raw_partial)
                and warp_idx < Int32(
                    self.kv_load_warp_base + self.num_kv_load_warps
                )
            ):
                kv_block_idx_load = row_metadata.kv_block_idx()
                k_batch_offset_load = row_metadata.kv_batch_offset()
                has_work_kv_load = row_metadata.q_count() > Int32(0)
                if const_expr(self.kv_fp8_to_bf16):
                    self.load_kv_maybe_cast(
                        tma_atom_K,
                        tma_atom_V,
                        tma_K,
                        tma_V,
                        sPagedKvIdx,
                        sK,
                        sV,
                        sKFp8 if const_expr(self.k_fp8_to_bf16) else None,
                        sVFp8 if const_expr(self.v_fp8_to_bf16) else None,
                        tiled_mma_qk,
                        tiled_mma_pv,
                        mbar_k_ptr,
                        mbar_v_ptr,
                        mbar_k_tma_ptr
                        if const_expr(self.k_fp8_to_bf16)
                        else None,
                        mbar_v_tma_ptr
                        if const_expr(self.v_fp8_to_bf16)
                        else None,
                        kv_load_barrier,
                        has_work_kv_load,
                        head_kv_idx_load,
                        kv_block_idx_load,
                        k_batch_offset_load,
                    )
                else:
                    self.load_kv(
                        tma_atom_K,
                        tma_atom_V,
                        tma_K,
                        tma_V,
                        sPagedKvIdx,
                        sK,
                        sV,
                        tiled_mma_qk,
                        tiled_mma_pv,
                        mbar_k_ptr,
                        mbar_v_ptr,
                        has_work_kv_load,
                        head_kv_idx_load,
                        kv_block_idx_load,
                        k_batch_offset_load,
                    )
            row_start_load = row_metadata.q_begin()
            count_raw_load = row_metadata.q_count()
            q_batch_offset_load = row_metadata.q_batch_offset()
            # Do not gate on KV validity here; sparse entries past seqused_k
            # still need the all-masked path to produce neutral partials.
            has_work_load = count_raw_load > Int32(0)
            num_q_groups_load = (
                count_raw_load + Int32(self.q_tokens_per_group - 1)
            ) // Int32(self.q_tokens_per_group)
            if const_expr(self.dpsum_direct):
                self.load_qdo_tma(
                    tma_atom_Q,
                    tma_atom_DO,
                    mQ_2d,
                    mDO_2d,
                    mK2qQSplitIndices,
                    sQIdxMeta,
                    sQ_load,
                    sDO_load,
                    pipeline_q,
                    pipeline_do,
                    load_wg_barrier,
                    num_q_groups_load,
                    count_raw_load,
                    has_work_load,
                    head_kv_idx,
                    row_start_load,
                    q_batch_offset_load,
                    num_heads_kv,
                )
            elif const_expr(self.use_q_gather4):
                self.load_q_gather4(
                    mQ_2d,
                    mQ_gather4_desc,
                    mK2qQSplitIndices,
                    sQIdxMeta,
                    sQ,
                    pipeline_q,
                    load_wg_barrier,
                    num_q_groups_load,
                    count_raw_load,
                    has_work_load,
                    head_kv_idx_load,
                    row_start_load,
                    q_batch_offset_load,
                    num_heads_kv,
                )
            else:
                self.load_q_tma(
                    tma_atom_Q,
                    mQ_2d,
                    mK2qQSplitIndices,
                    sQIdxMeta,
                    sQ_load,
                    pipeline_q,
                    load_wg_barrier,
                    num_q_groups_load,
                    count_raw_load,
                    has_work_load,
                    head_kv_idx_load,
                    row_start_load,
                    q_batch_offset_load,
                    num_heads_kv,
                )

        if const_expr(self.store_raw_partial):
            is_kv_load_warp = (
                warp_idx >= Int32(self.kv_load_warp_base)
                and warp_idx < Int32(
                    self.kv_load_warp_base + self.num_kv_load_warps
                )
            )
            if is_kv_load_warp and row_metadata.kv_load_active() != Int32(0):
                cute.arch.setmaxregister_decrease(self.num_regs_kv_load)
                head_kv_idx_load = row_metadata.kv_head_idx()
                kv_block_idx_load = row_metadata.kv_block_idx()
                k_batch_offset_load = row_metadata.kv_batch_offset()
                has_work_kv_load = row_metadata.q_count() > Int32(0)
                if const_expr(self.kv_fp8_to_bf16):
                    self.load_kv_maybe_cast(
                        tma_atom_K,
                        tma_atom_V,
                        tma_K,
                        tma_V,
                        sPagedKvIdx,
                        sK,
                        sV,
                        sKFp8 if const_expr(self.k_fp8_to_bf16) else None,
                        sVFp8 if const_expr(self.v_fp8_to_bf16) else None,
                        tiled_mma_qk,
                        tiled_mma_pv,
                        mbar_k_ptr,
                        mbar_v_ptr,
                        mbar_k_tma_ptr
                        if const_expr(self.k_fp8_to_bf16)
                        else None,
                        mbar_v_tma_ptr
                        if const_expr(self.v_fp8_to_bf16)
                        else None,
                        kv_load_barrier,
                        has_work_kv_load,
                        head_kv_idx_load,
                        kv_block_idx_load,
                        k_batch_offset_load,
                    )
                else:
                    self.load_kv(
                        tma_atom_K,
                        tma_atom_V,
                        tma_K,
                        tma_V,
                        sPagedKvIdx,
                        sK,
                        sV,
                        tiled_mma_qk,
                        tiled_mma_pv,
                        mbar_k_ptr,
                        mbar_v_ptr,
                        has_work_kv_load,
                        head_kv_idx_load,
                        kv_block_idx_load,
                        k_batch_offset_load,
                    )

        if (
            warp_idx == Int32(self.mma_warp_id)
            and row_metadata.mma_active() != Int32(0)
        ):
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            count_raw_mma = row_metadata.q_count()
            has_work_mma = count_raw_mma > Int32(0)
            num_q_groups_mma = (
                count_raw_mma + Int32(self.q_tokens_per_group - 1)
            ) // Int32(self.q_tokens_per_group)
            tmem.allocate(tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            if const_expr(self.dpsum_direct):
                self.mma_dpsum(
                    tiled_mma_qk,
                    tiled_mma_pv,
                    sK,
                    sV,
                    sQ,
                    sDO,
                    pipeline_q,
                    pipeline_do,
                    pipeline_s,
                    pipeline_o,
                    mbar_k_ptr,
                    mbar_v_ptr,
                    num_q_groups_mma,
                    has_work_mma,
                )
            else:
                self.mma(
                    tiled_mma_qk, tiled_mma_pv,
                    tOrP,
                    sK, sV, sQ,
                    pipeline_q,
                    pipeline_s, pipeline_p, pipeline_p_lastsplit,
                    pipeline_o,
                    mbar_k_ptr, mbar_v_ptr,
                    num_q_groups_mma, has_work_mma,
                )
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr, num_columns=tmem_cols)
            cute.arch.griddepcontrol_launch_dependents()

        if (
            warp_idx >= Int32(self.softmax0_warp_base)
            and warp_idx < Int32(self.softmax1_warp_base)
            and row_metadata.softmax0_active() != Int32(0)
        ):
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            kv_block_idx_softmax = row_metadata.kv_block_idx()
            count_raw_softmax = row_metadata.q_count()
            kv_valid_cols_softmax = row_metadata.kv_columns()
            causal_q_offset_softmax = row_metadata.causal_q_offset()
            head_kv_idx_softmax = row_metadata.kv_head_idx()
            has_work_softmax = count_raw_softmax > Int32(0)
            num_q_groups_softmax = (
                count_raw_softmax + Int32(self.q_tokens_per_group - 1)
            ) // Int32(self.q_tokens_per_group)
            diag_q_count_softmax = sDiagQCount[0]
            if const_expr(self.k_fp8_to_bf16):
                self.convert_fp8_kv_to_bf16_smem(
                    sKFp8,
                    sK,
                    mbar_k_tma_ptr,
                    mbar_k_ptr,
                    kv_dequant_k_barrier,
                    has_work_softmax,
                    self.softmax0_warp_base,
                    self.warps_per_group,
                    False,
                )
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            if const_expr(self.dpsum_direct):
                self.compute_dpsum(
                    0,
                    tiled_mma_qk,
                    tStS,
                    tDPtDP,
                    sQIdxMeta,
                    pipeline_s,
                    pipeline_o,
                    mLSE_partial,
                    mLSE_temperature_partial,
                    mPQuantScale,
                    softmax_scale_log2,
                    kv_block_idx_softmax,
                    kv_valid_cols_softmax,
                    diag_q_count_softmax,
                    num_q_groups_softmax,
                    count_raw_softmax,
                    has_work_softmax,
                    causal_q_offset_softmax,
                    row_metadata.batch_idx(),
                    head_kv_idx_softmax,
                    seq_len_q,
                    row_metadata.q_batch_offset(),
                )
            else:
                self.softmax(
                    0,
                    tiled_mma_qk,
                    tiled_mma_pv,
                    tStS,
                    sScale,
                    sScaleTemperature,
                    sQIdxMeta,
                    pipeline_s, pipeline_p, pipeline_p_lastsplit,
                    pipeline_o, pipeline_sm_stats,
                    sm_stats_barrier,
                    mO_partial, mLSE_partial, mLSE_temperature_partial,
                    softmax_scale_log2,
                    lse_temperature_scale,
                    kv_block_idx_softmax,
                    kv_valid_cols_softmax,
                    diag_q_count_softmax,
                    num_q_groups_softmax, count_raw_softmax, has_work_softmax,
                    causal_q_offset_softmax,
                    head_kv_idx_softmax,
                    seq_len_q,
                    head_q,
                    row_metadata.q_batch_offset(),
                    mQ_2d,
                )
            tmem_alloc_barrier.arrive()

        if (
            warp_idx >= Int32(self.softmax1_warp_base)
            and warp_idx < Int32(self.correction_warp_base)
            and row_metadata.softmax1_active() != Int32(0)
        ):
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            kv_block_idx_softmax = row_metadata.kv_block_idx()
            count_raw_softmax = row_metadata.q_count()
            kv_valid_cols_softmax = row_metadata.kv_columns()
            causal_q_offset_softmax = row_metadata.causal_q_offset()
            head_kv_idx_softmax = row_metadata.kv_head_idx()
            has_work_softmax = count_raw_softmax > Int32(0)
            num_q_groups_softmax = (
                count_raw_softmax + Int32(self.q_tokens_per_group - 1)
            ) // Int32(self.q_tokens_per_group)
            diag_q_count_softmax = sDiagQCount[0]
            if const_expr(self.v_fp8_to_bf16):
                self.convert_fp8_kv_to_bf16_smem(
                    sVFp8,
                    sV,
                    mbar_v_tma_ptr,
                    mbar_v_ptr,
                    kv_dequant_v_barrier,
                    has_work_softmax,
                    self.softmax1_warp_base,
                    self.warps_per_group,
                    True,
                )
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            if const_expr(self.dpsum_direct):
                self.compute_dpsum(
                    1,
                    tiled_mma_qk,
                    tStS,
                    tDPtDP,
                    sQIdxMeta,
                    pipeline_s,
                    pipeline_o,
                    mLSE_partial,
                    mLSE_temperature_partial,
                    mPQuantScale,
                    softmax_scale_log2,
                    kv_block_idx_softmax,
                    kv_valid_cols_softmax,
                    diag_q_count_softmax,
                    num_q_groups_softmax,
                    count_raw_softmax,
                    has_work_softmax,
                    causal_q_offset_softmax,
                    row_metadata.batch_idx(),
                    head_kv_idx_softmax,
                    seq_len_q,
                    row_metadata.q_batch_offset(),
                )
            else:
                self.softmax(
                    1,
                    tiled_mma_qk,
                    tiled_mma_pv,
                    tStS,
                    sScale,
                    sScaleTemperature,
                    sQIdxMeta,
                    pipeline_s, pipeline_p, pipeline_p_lastsplit,
                    pipeline_o, pipeline_sm_stats,
                    sm_stats_barrier,
                    mO_partial, mLSE_partial, mLSE_temperature_partial,
                    softmax_scale_log2,
                    lse_temperature_scale,
                    kv_block_idx_softmax,
                    kv_valid_cols_softmax,
                    diag_q_count_softmax,
                    num_q_groups_softmax, count_raw_softmax, has_work_softmax,
                    causal_q_offset_softmax,
                    head_kv_idx_softmax,
                    seq_len_q,
                    head_q,
                    row_metadata.q_batch_offset(),
                    mQ_2d,
                )
            tmem_alloc_barrier.arrive()

        if const_expr(not self.store_raw_partial):
            if (
                warp_idx >= Int32(self.correction_warp_base)
                and warp_idx < Int32(self.mma_warp_id)
                and row_metadata.epilogue_active() != Int32(0)
            ):
                cute.arch.setmaxregister_decrease(self.num_regs_correction)
                head_kv_idx_correction = row_metadata.kv_head_idx()
                count_raw_correction = row_metadata.q_count()
                has_work_correction = count_raw_correction > Int32(0)
                num_q_groups_correction = (
                    count_raw_correction + Int32(self.q_tokens_per_group - 1)
                ) // Int32(self.q_tokens_per_group)
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
                if const_expr(not self.dpsum_direct):
                    self.epilogue(
                        tiled_mma_pv,
                        sScale,
                        sScaleTemperature,
                        sSplitIdx,
                        sQIdx,
                        sQIdxMeta,
                        sDO if const_expr(self.dpsum_only) else None,
                        pipeline_o,
                        pipeline_sm_stats,
                        epilogue_barrier,
                        mO_partial,
                        mLSE_partial,
                        mLSE_temperature_partial,
                        mPQuantScale,
                        softmax_scale_log2,
                        lse_temperature_scale_log2,
                        count_raw_correction,
                        has_work_correction,
                        row_metadata.batch_idx(),
                        head_kv_idx_correction,
                        head_q,
                        row_metadata.q_batch_offset(),
                        mQ_2d,
                        num_q_groups_correction,
                    )
                tmem_alloc_barrier.arrive()

    # ------------------------------------------------------------------
    # Warp-specialized helpers
    # ------------------------------------------------------------------

    @cute.jit
    def _convert_fp8x16_to_bf16x16(
        self,
        src: cute.Tensor,
        dst: cute.Tensor,
    ):
        src_i32 = cute.recast_tensor(src, cutlass.Int32)
        dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
        for word_idx in cutlass.range_constexpr(4):
            (
                dst_i32[word_idx * 2],
                dst_i32[word_idx * 2 + 1],
            ) = utils.cvt_fp8x4_e4m3_bf16x4(src_i32[word_idx])

    @cute.jit
    def _convert_fp8_kv_to_bf16_smem(
        self,
        sFp8: cute.Tensor,
        sBf16: cute.Tensor,
        lane: Int32,
        warp_idx_in_wg: Int32,
        num_dequant_warps: cutlass.Constexpr[int],
        is_v: cutlass.Constexpr[bool],
    ):
        elems_per_load: cutlass.Constexpr[int] = 16
        elems_per_store: cutlass.Constexpr[int] = 8
        chunks_per_row: cutlass.Constexpr[int] = self.head_dim // elems_per_load
        r_fp8 = cute.make_rmem_tensor((elems_per_load,), cutlass.Float8E4M3FN)
        r_bf16 = cute.make_rmem_tensor((elems_per_load,), cutlass.BFloat16)
        total_tasks: cutlass.Constexpr[int] = self.n_block_size * chunks_per_row
        task_stride: cutlass.Constexpr[int] = num_dequant_warps * cute.arch.WARP_SIZE
        task = warp_idx_in_wg * Int32(cute.arch.WARP_SIZE) + lane
        for task_idx in cutlass.range(task, total_tasks, task_stride, unroll=1):
            row = task_idx // Int32(chunks_per_row)
            chunk = task_idx - row * Int32(chunks_per_row)
            col = chunk * Int32(elems_per_load)
            smem_offset = row * Int32(self.head_dim) + col
            s_fp8_ptr = cute.make_ptr(
                cutlass.Float8E4M3FN,
                sFp8.iterator.toint() + Int64(smem_offset),
                mem_space=sFp8.iterator.memspace,
                assumed_align=elems_per_load,
            )
            s_fp8_vec = cute.make_tensor(
                s_fp8_ptr,
                cute.make_layout(elems_per_load),
            )
            cute.autovec_copy(s_fp8_vec, r_fp8)
            self._convert_fp8x16_to_bf16x16(r_fp8, r_bf16)
            if const_expr(is_v):
                sBf16_view = sBf16[(None, row % Int32(16)), 0, row // Int32(16), 0]
                sBf16_vec = cute.local_tile(sBf16_view, (elems_per_load,), (chunk,))
            else:
                sBf16_vec = sBf16[
                    (row, None),
                    0,
                    (chunk % Int32(4), chunk // Int32(4)),
                    0,
                ]
            r_tiles = cute.logical_divide(r_bf16, cute.make_layout(elems_per_store))
            s_tiles = cute.logical_divide(sBf16_vec, cute.make_layout(elems_per_store))
            for v in cutlass.range_constexpr(elems_per_load // elems_per_store):
                cute.autovec_copy(r_tiles[None, v], s_tiles[None, v])
        cute.arch.fence_view_async_shared()

    @cute.jit
    def load_kv_maybe_cast(
        self,
        tma_atom_K,
        tma_atom_V,
        tma_K: cute.Tensor,
        tma_V: cute.Tensor,
        sPagedKvIdx: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sKFp8: Optional[cute.Tensor],
        sVFp8: Optional[cute.Tensor],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mbar_k_ptr,
        mbar_v_ptr,
        mbar_k_tma_ptr: Optional[cutlass.Pointer],
        mbar_v_tma_ptr: Optional[cutlass.Pointer],
        kv_load_barrier,
        has_work: Int32,
        head_kv_idx: Int32,
        kv_block_idx: Int32,
        k_batch_offset: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx_in_wg = warp_idx - Int32(self.kv_load_warp_base)

        if has_work:
            if warp_idx_in_wg == Int32(0):
                if const_expr(self.k_fp8_to_bf16):
                    if const_expr(self.paged_kv):
                        mK_cur = tma_K[None, None, head_kv_idx, sPagedKvIdx[0]]
                        gK = cute.local_tile(
                            mK_cur,
                            (self.n_block_size, self.head_dim),
                            (None, 0),
                        )
                        src_idx = Int32(0)
                    else:
                        mK_cur = cute.domain_offset(
                            (k_batch_offset, 0),
                            tma_K[None, None, head_kv_idx],
                        )
                        gK = cute.local_tile(
                            mK_cur,
                            (self.n_block_size, self.head_dim),
                            (None, 0),
                        )
                        src_idx = kv_block_idx
                    load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_K,
                        0,
                        cute.make_layout(1),
                        gK,
                        sKFp8,
                    )
                    load_K_fn(
                        src_idx=src_idx,
                        dst_idx=0,
                        tma_bar_ptr=mbar_k_tma_ptr,
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mbar_k_tma_ptr)
                else:
                    thr_mma_qk = tiled_mma_qk.get_slice(0)
                    if const_expr(self.paged_kv):
                        mK_cur = tma_K[None, None, head_kv_idx, None]
                        gK = cute.local_tile(
                            mK_cur,
                            cute.select(self.mma_tiler_qk, mode=[1, 2]),
                            (None, 0, None),
                        )
                    else:
                        mK_cur = cute.domain_offset(
                            (k_batch_offset, 0),
                            tma_K[None, None, head_kv_idx],
                        )
                        gK = cute.local_tile(
                            mK_cur,
                            cute.select(self.mma_tiler_qk, mode=[1, 2]),
                            (None, 0),
                        )
                    tSgK = thr_mma_qk.partition_B(gK)
                    tKsK, tKgK = cpasync.tma_partition(
                        tma_atom_K,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sK, 0, 3),
                        cute.group_modes(tSgK, 0, 3),
                    )
                    gmem_k_idx = (
                        sPagedKvIdx[0] if const_expr(self.paged_kv) else kv_block_idx
                    )
                    cute.copy(
                        tma_atom_K,
                        tKgK[(None, 0, gmem_k_idx)]
                        if const_expr(self.paged_kv)
                        else tKgK[(None, gmem_k_idx)],
                        tKsK[(None, 0)],
                        tma_bar_ptr=mbar_k_ptr,
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mbar_k_ptr)

            if warp_idx_in_wg == Int32(1):
                if const_expr(self.v_fp8_to_bf16):
                    if const_expr(self.paged_kv):
                        mV_cur = tma_V[None, None, head_kv_idx, sPagedKvIdx[0]]
                        gV = cute.local_tile(
                            mV_cur,
                            (self.head_dim, self.n_block_size),
                            (0, None),
                        )
                        src_idx = Int32(0)
                    else:
                        mV_cur = cute.domain_offset(
                            (0, k_batch_offset),
                            tma_V[None, None, head_kv_idx],
                        )
                        gV = cute.local_tile(
                            mV_cur,
                            (self.head_dim, self.n_block_size),
                            (0, None),
                        )
                        src_idx = kv_block_idx
                    load_V_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_V,
                        0,
                        cute.make_layout(1),
                        gV,
                        sVFp8,
                    )
                    load_V_fn(
                        src_idx=src_idx,
                        dst_idx=0,
                        tma_bar_ptr=mbar_v_tma_ptr,
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mbar_v_tma_ptr)
                else:
                    thr_mma_pv = tiled_mma_pv.get_slice(0)
                    if const_expr(self.paged_kv):
                        mV_cur = tma_V[None, None, head_kv_idx, None]
                        gV = cute.local_tile(
                            mV_cur,
                            cute.select(self.mma_tiler_pv, mode=[1, 2]),
                            (0, None, None),
                        )
                    else:
                        mV_cur = cute.domain_offset(
                            (0, k_batch_offset),
                            tma_V[None, None, head_kv_idx],
                        )
                        gV = cute.local_tile(
                            mV_cur,
                            cute.select(self.mma_tiler_pv, mode=[1, 2]),
                            (0, None),
                        )
                    tOgV = thr_mma_pv.partition_B(gV)
                    tVsV, tVgV = cpasync.tma_partition(
                        tma_atom_V,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sV, 0, 3),
                        cute.group_modes(tOgV, 0, 3),
                    )
                    gmem_v_idx = (
                        sPagedKvIdx[0] if const_expr(self.paged_kv) else kv_block_idx
                    )
                    cute.copy(
                        tma_atom_V,
                        tVgV[(None, 0, gmem_v_idx)]
                        if const_expr(self.paged_kv)
                        else tVgV[(None, gmem_v_idx)],
                        tVsV[(None, 0)],
                        tma_bar_ptr=mbar_v_ptr,
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mbar_v_ptr)

            kv_load_barrier.arrive_and_wait()

    @cute.jit
    def convert_fp8_kv_to_bf16_smem(
        self,
        sFp8: cute.Tensor,
        sBf16: cute.Tensor,
        mbar_tma_ptr,
        mbar_ready_ptr,
        dequant_barrier,
        has_work: Int32,
        dequant_warp_base: cutlass.Constexpr[int],
        num_dequant_warps: cutlass.Constexpr[int],
        is_v: cutlass.Constexpr[bool],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx_in_wg = warp_idx - Int32(dequant_warp_base)
        lane = cute.arch.lane_idx()
        if has_work:
            cute.arch.mbarrier_wait(mbar_tma_ptr, 0)
            self._convert_fp8_kv_to_bf16_smem(
                sFp8,
                sBf16,
                lane,
                warp_idx_in_wg,
                num_dequant_warps,
                is_v,
            )
            dequant_barrier.arrive_and_wait()
            if warp_idx_in_wg == Int32(0):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mbar_ready_ptr)

    @cute.jit
    def load_kv(
        self,
        tma_atom_K,
        tma_atom_V,
        tma_K: cute.Tensor,
        tma_V: cute.Tensor,
        sPagedKvIdx: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mbar_k_ptr,
        mbar_v_ptr,
        has_work: Int32,
        head_kv_idx: Int32,
        kv_block_idx: Int32,
        k_batch_offset: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx_in_wg = warp_idx - Int32(self.kv_load_warp_base)

        if has_work:
            if warp_idx_in_wg == Int32(0):
                thr_mma_qk = tiled_mma_qk.get_slice(0)
                if const_expr(self.paged_kv):
                    mK_cur = tma_K[None, None, head_kv_idx, None]
                    gK = cute.local_tile(
                        mK_cur,
                        cute.select(self.mma_tiler_qk, mode=[1, 2]),
                        (None, 0, None),
                    )
                else:
                    mK_cur = cute.domain_offset(
                        (k_batch_offset, 0),
                        tma_K[None, None, head_kv_idx],
                    )
                    gK = cute.local_tile(
                        mK_cur,
                        cute.select(self.mma_tiler_qk, mode=[1, 2]),
                        (None, 0),
                    )
                tSgK = thr_mma_qk.partition_B(gK)
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK, 0, 3),
                )
                gmem_k_idx = (
                    sPagedKvIdx[0] if const_expr(self.paged_kv) else kv_block_idx
                )
                cute.copy(
                    tma_atom_K,
                    tKgK[(None, 0, gmem_k_idx)]
                    if const_expr(self.paged_kv)
                    else tKgK[(None, gmem_k_idx)],
                    tKsK[(None, 0)],
                    tma_bar_ptr=mbar_k_ptr,
                )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mbar_k_ptr)

            if warp_idx_in_wg == Int32(1):
                thr_mma_pv = tiled_mma_pv.get_slice(0)
                if const_expr(self.dpsum_direct):
                    mV_cur = cute.domain_offset(
                        (k_batch_offset, 0),
                        tma_V[None, None, head_kv_idx],
                    )
                    gV = cute.local_tile(
                        mV_cur,
                        cute.select(self.mma_tiler_qk, mode=[1, 2]),
                        (None, 0),
                    )
                elif const_expr(self.paged_kv):
                    mV_cur = tma_V[None, None, head_kv_idx, None]
                    gV = cute.local_tile(
                        mV_cur,
                        cute.select(self.mma_tiler_pv, mode=[1, 2]),
                        (0, None, None),
                    )
                else:
                    mV_cur = cute.domain_offset(
                        (0, k_batch_offset),
                        tma_V[None, None, head_kv_idx],
                    )
                    gV = cute.local_tile(
                        mV_cur,
                        cute.select(self.mma_tiler_pv, mode=[1, 2]),
                        (0, None),
                    )
                tOgV = thr_mma_pv.partition_B(gV)
                tVsV, tVgV = cpasync.tma_partition(
                    tma_atom_V,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tOgV, 0, 3),
                )
                gmem_v_idx = (
                    sPagedKvIdx[0] if const_expr(self.paged_kv) else kv_block_idx
                )
                cute.copy(
                    tma_atom_V,
                    tVgV[(None, 0, gmem_v_idx)]
                    if const_expr(self.paged_kv)
                    else tVgV[(None, gmem_v_idx)],
                    tVsV[(None, 0)],
                    tma_bar_ptr=mbar_v_ptr,
                )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mbar_v_ptr)

    @cute.jit
    def load_q_gather4(
        self,
        mQ_2d: cute.Tensor,
        mQ_gather4_desc: cute.Tensor,
        mK2qQSplitIndices: cute.Tensor,
        sQIdxMeta: cute.Tensor,
        sQ: cute.Tensor,
        pipeline_q,
        load_wg_barrier,
        num_q_groups: Int32,
        count_raw: Int32,
        has_work: Int32,
        head_kv_idx: Int32,
        row_start: Int32,
        q_batch_offset: Int32,
        num_heads_kv: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        q_load_thread_base = Int32(self.q_load_warp_base * cute.arch.WARP_SIZE)
        group_tidx = tidx - q_load_thread_base
        producer_warp_idx_in_wg = cute.arch.make_warp_uniform(
            group_tidx // Int32(cute.arch.WARP_SIZE)
        )
        q_oob_m_idx = mQ_2d.shape[0] // Int32(self.qheadperkv)
        # Three load warps do not evenly divide the 32 gather4 operations.
        # Round up and guard the tail so all 128 Q rows contribute their TMA
        # transaction bytes; dropping the final two gathers stalls mbar_q.
        num_gathers: cutlass.Constexpr[int] = self.m_block_size // 4
        gathers_per_warp: cutlass.Constexpr[int] = (
            num_gathers + self.num_q_load_warps - 1
        ) // self.num_q_load_warps

        if has_work:
            for qi_group in cutlass.range(num_q_groups, unroll=1):
                slot = qi_group % Int32(self.q_stage)
                phase = (qi_group // Int32(self.q_stage)) & Int32(1)
                producer_phase = phase ^ Int32(1)
                if producer_warp_idx_in_wg == Int32(0):
                    pipeline_q.producer_acquire_w_index_phase(
                        slot, producer_phase)
                load_wg_barrier.arrive_and_wait()

                group_tidx = tidx - q_load_thread_base
                warp_idx_in_wg = cute.arch.make_warp_uniform(
                    group_tidx // Int32(cute.arch.WARP_SIZE)
                )
                lane_idx = group_tidx % Int32(cute.arch.WARP_SIZE)
                mbar_ptr = pipeline_q.sync_object_full.get_barrier(slot)
                qidx_meta_slot = (qi_group
                                  & Int32(self.qidx_meta_stages - 1)) * Int32(
                                      self.q_tokens_per_group)

                meta_iters: cutlass.Constexpr[int] = (
                    (self.q_tokens_per_group
                     + self.num_q_load_warps * cute.arch.WARP_SIZE - 1)
                    // (self.num_q_load_warps * cute.arch.WARP_SIZE)
                )
                for meta_iter in cutlass.range_constexpr(meta_iters):
                    tok_idx_g4 = (
                        (Int32(meta_iter) * Int32(self.num_q_load_warps)
                         + warp_idx_in_wg)
                        * Int32(cute.arch.WARP_SIZE)
                        + lane_idx
                    )
                    if tok_idx_g4 < Int32(self.q_tokens_per_group):
                        qi = qi_group * Int32(self.q_tokens_per_group) + tok_idx_g4
                        if qi < count_raw:
                            sQIdxMeta[qidx_meta_slot + tok_idx_g4] = self._load_qsplit_idx(
                                mK2qQSplitIndices, head_kv_idx, row_start, qi
                            )
                        else:
                            sQIdxMeta[qidx_meta_slot + tok_idx_g4] = Int32(0)
                        cute.arch.mbarrier_arrive(mbar_ptr)
                load_wg_barrier.arrive_and_wait()

                with cute.arch.elect_one():
                    q_desc_ptr = mQ_gather4_desc.iterator
                    sQ_ptr = sQ.iterator
                    for gather_slot in cutlass.range_constexpr(gathers_per_warp):
                        gather_idx = (
                            Int32(gather_slot) * Int32(self.num_q_load_warps)
                            + warp_idx_in_wg
                        )
                        if gather_idx < Int32(num_gathers):
                            tok_base = gather_idx * Int32(self.tokens_per_gather4)
                            if const_expr(self.qheadperkv == 1):
                                qi0 = qi_group * Int32(self.q_tokens_per_group) + tok_base
                                qi1 = qi0 + Int32(1)
                                qi2 = qi0 + Int32(2)
                                qi3 = qi0 + Int32(3)
                                row0 = q_oob_m_idx
                                row1 = q_oob_m_idx
                                row2 = q_oob_m_idx
                                row3 = q_oob_m_idx
                                if qi0 < count_raw:
                                    q_idx0 = self._decode_q_idx_from_qsplit(
                                        sQIdxMeta[qidx_meta_slot + tok_base])
                                    row0 = (q_batch_offset + q_idx0) * num_heads_kv + head_kv_idx
                                if qi1 < count_raw:
                                    q_idx1 = self._decode_q_idx_from_qsplit(
                                        sQIdxMeta[qidx_meta_slot + tok_base + Int32(1)])
                                    row1 = (q_batch_offset + q_idx1) * num_heads_kv + head_kv_idx
                                if qi2 < count_raw:
                                    q_idx2 = self._decode_q_idx_from_qsplit(
                                        sQIdxMeta[qidx_meta_slot + tok_base + Int32(2)])
                                    row2 = (q_batch_offset + q_idx2) * num_heads_kv + head_kv_idx
                                if qi3 < count_raw:
                                    q_idx3 = self._decode_q_idx_from_qsplit(
                                        sQIdxMeta[qidx_meta_slot + tok_base + Int32(3)])
                                    row3 = (q_batch_offset + q_idx3) * num_heads_kv + head_kv_idx
                            elif const_expr(self.qheadperkv == 2):
                                qi0 = qi_group * Int32(self.q_tokens_per_group) + tok_base
                                qi1 = qi0 + Int32(1)
                                row_base0 = q_oob_m_idx * Int32(self.qheadperkv)
                                row_base1 = q_oob_m_idx * Int32(self.qheadperkv)
                                if qi0 < count_raw:
                                    q_idx0 = self._decode_q_idx_from_qsplit(
                                        sQIdxMeta[qidx_meta_slot + tok_base])
                                    row_base0 = (
                                        (q_batch_offset + q_idx0) * num_heads_kv
                                        + head_kv_idx
                                    ) * Int32(self.qheadperkv)
                                if qi1 < count_raw:
                                    q_idx1 = self._decode_q_idx_from_qsplit(
                                        sQIdxMeta[qidx_meta_slot + tok_base + Int32(1)])
                                    row_base1 = (
                                        (q_batch_offset + q_idx1) * num_heads_kv
                                        + head_kv_idx
                                    ) * Int32(self.qheadperkv)
                                row0 = row_base0
                                row1 = row_base0 + Int32(1)
                                row2 = row_base1
                                row3 = row_base1 + Int32(1)
                            else:
                                qi0 = qi_group * Int32(self.q_tokens_per_group) + tok_base
                                row_base0 = q_oob_m_idx * Int32(self.qheadperkv)
                                if qi0 < count_raw:
                                    q_idx0 = self._decode_q_idx_from_qsplit(
                                        sQIdxMeta[qidx_meta_slot + tok_base])
                                    row_base0 = (
                                        (q_batch_offset + q_idx0) * num_heads_kv
                                        + head_kv_idx
                                    ) * Int32(self.qheadperkv)
                                row0 = row_base0
                                row1 = row_base0 + Int32(1)
                                row2 = row_base0 + Int32(2)
                                row3 = row_base0 + Int32(3)
                            group_byte_off = gather_idx * Int32(
                                4 * self.k_tile * (self.q_dtype.width // 8)
                            )
                            if const_expr(self.q_dtype == cutlass.Float8E4M3FN):
                                stage_byte_off = slot * Int32(self.q_stage_stride_bytes)
                                full_group_byte_off = gather_idx * Int32(
                                    4 * self.head_dim * (self.q_dtype.width // 8)
                                )
                                tma_gather4_cached(
                                    sQ_ptr,
                                    stage_byte_off + full_group_byte_off,
                                    q_desc_ptr,
                                    Int32(0),
                                    row0,
                                    row1,
                                    row2,
                                    row3,
                                    mbar_ptr,
                                    TMA_CACHE_EVICT_LAST,
                                )
                            else:
                                for ks_c in cutlass.range_constexpr(self.k_stages):
                                    stage_idx = slot * Int32(self.k_stages) + Int32(ks_c)
                                    stage_byte_off = stage_idx * Int32(self.k_tile_stride_bytes)
                                    if const_expr(ks_c + 1 < self.k_stages):
                                        tma_gather4_prefetch(
                                            q_desc_ptr,
                                            Int32((ks_c + 1) * self.k_tile),
                                            row0,
                                            row1,
                                            row2,
                                            row3,
                                            TMA_CACHE_EVICT_LAST,
                                        )
                                    tma_gather4_cached(
                                        sQ_ptr,
                                        stage_byte_off + group_byte_off,
                                        q_desc_ptr,
                                        Int32(ks_c * self.k_tile),
                                        row0,
                                        row1,
                                        row2,
                                        row3,
                                        mbar_ptr,
                                        TMA_CACHE_EVICT_LAST,
                                    )
                load_wg_barrier.arrive_and_wait()

            if producer_warp_idx_in_wg == Int32(0):
                next_slot = num_q_groups % Int32(self.q_stage)
                next_phase = (
                    (num_q_groups // Int32(self.q_stage)) & Int32(1)
                ) ^ Int32(1)
                pipeline_q.producer_acquire_w_index_phase(next_slot, next_phase)

    @cute.jit
    def load_q_tma(
        self,
        tma_atom_Q,
        mQ_2d: cute.Tensor,
        mK2qQSplitIndices: cute.Tensor,
        sQIdxMeta: cute.Tensor,
        sQ_load: cute.Tensor,
        pipeline_q,
        load_wg_barrier,
        num_q_groups: Int32,
        count_raw: Int32,
        has_work: Int32,
        head_kv_idx: Int32,
        row_start: Int32,
        q_batch_offset: Int32,
        num_heads_kv: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        warp_idx_in_wg = warp_idx - Int32(self.q_load_warp_base)
        if const_expr(self.q_dtype == cutlass.Float8E4M3FN):
            gQ_full = cute.local_tile(
                mQ_2d, (self.qheadperkv, self.head_dim), (None, 0))
            load_Q_fn_full, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), gQ_full, sQ_load)
            load_Q_fn_k0, load_Q_fn_k1 = None, None
        else:
            gQ_k0 = cute.local_tile(mQ_2d, (self.qheadperkv, self.k_tile), (None, 0))
            gQ_k1 = cute.local_tile(mQ_2d, (self.qheadperkv, self.k_tile), (None, 1))
            load_Q_fn_k0, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), gQ_k0, sQ_load)
            load_Q_fn_k1, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), gQ_k1, sQ_load)
            load_Q_fn_full = None
        q_oob_m_idx = mQ_2d.shape[0] // Int32(self.qheadperkv)
        tokens_per_warp: cutlass.Constexpr[int] = (
            (self.q_tokens_per_group + self.num_q_load_warps - 1)
            // self.num_q_load_warps
        )

        if has_work:
            for qi_group in cutlass.range(num_q_groups, unroll=1):
                slot = qi_group % Int32(self.q_stage)
                phase = (qi_group // Int32(self.q_stage)) & Int32(1)
                producer_phase = phase ^ Int32(1)
                if warp_idx_in_wg == Int32(0):
                    pipeline_q.producer_acquire_w_index_phase(
                        slot, producer_phase)

                mbar_ptr = pipeline_q.sync_object_full.get_barrier(slot)
                q_load_subtiles_per_token = (
                    1 if const_expr(self.q_dtype == cutlass.Float8E4M3FN)
                    else self.k_stages
                )
                sub_stage_base = slot * Int32(
                    self.q_tokens_per_group * q_load_subtiles_per_token)
                qidx_meta_slot = (qi_group
                                  & Int32(self.qidx_meta_stages - 1)) * Int32(
                                      self.q_tokens_per_group)
                if (
                    warp_idx_in_wg == Int32(0)
                    and lane_idx < Int32(self.q_tokens_per_group)
                ):
                    tok_idx = lane_idx
                    qi = (
                        qi_group * Int32(self.q_tokens_per_group)
                        + tok_idx
                    )
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
                    cute.arch.mbarrier_arrive(mbar_ptr)
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
                            load_Q_fn_full(
                                src_idx=m_tile_idx,
                                dst_idx=sub_stage_base + tok_idx,
                                tma_bar_ptr=mbar_ptr,
                            )
                        else:
                            load_Q_fn_k0(
                                src_idx=m_tile_idx,
                                dst_idx=sub_stage_base + tok_idx,
                                tma_bar_ptr=mbar_ptr,
                            )
                            load_Q_fn_k1(
                                src_idx=m_tile_idx,
                                dst_idx=(
                                    sub_stage_base
                                    + Int32(self.q_tokens_per_group)
                                    + tok_idx
                                ),
                                tma_bar_ptr=mbar_ptr,
                            )

            if warp_idx_in_wg == Int32(0):
                next_slot = num_q_groups % Int32(self.q_stage)
                next_phase = (
                    (num_q_groups // Int32(self.q_stage)) & Int32(1)
                ) ^ Int32(1)
                pipeline_q.producer_acquire_w_index_phase(next_slot, next_phase)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tOrP: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sQ: cute.Tensor,
        pipeline_q,
        pipeline_s,
        pipeline_p,
        pipeline_p_lastsplit,
        pipeline_o,
        mbar_k_ptr,
        mbar_v_ptr,
        num_q_groups: Int32,
        has_work: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        is_mma_warp = warp_idx == Int32(self.mma_warp_id)

        if is_mma_warp:
            if has_work:
                tSrQ = tiled_mma_qk.make_fragment_A(sQ)
                tSrK = tiled_mma_qk.make_fragment_B(sK)
                tSrQ0 = tSrQ[None, None, None, 0]
                tSrK0 = tSrK[None, None, None, 0]
                tOrV = tiled_mma_pv.make_fragment_B(sV)
                tOrV0 = tOrV[None, None, None, 0]
                sV0 = sV[None, None, None, 0]
                pv_mma_op = tiled_mma_pv.op
                qk_mma_op = tiled_mma_qk.op
                use_scoped_q_desc = const_expr(
                    self.store_raw_partial and self.q_stage == 4
                )
                q_smem_base = sm100_desc.smem_desc_base_from_tensor(
                    sQ, sm100_desc.Major.K)
                k_smem_base = sm100_desc.smem_desc_base_from_tensor(
                    sK, sm100_desc.Major.K)
                k_smem_start = sm100_desc.make_smem_desc_start_addr(
                    sK[None, None, None, 0].iterator)
                q_smem_start_0 = sm100_desc.make_smem_desc_start_addr(
                    sQ[None, None, None, 0].iterator)
                sQ_stage_stride = (
                    sQ.layout.stride[-1] * sQ.element_type.width // 8) >> 4
                if const_expr(self.q_stage == 1):
                    sQ_stage_stride = 0
                if const_expr(use_scoped_q_desc):
                    qk_idesc = const_expr(
                        sm100_desc.mma_op_to_idesc(qk_mma_op)
                    )
                    gemm_qk_s0_scoped = partial(
                        sm100_helpers.gemm_ptx_precomputed,
                        acc_tmem_addr=Int32(self.tmem_s_offset),
                        idesc=qk_idesc,
                        smem_desc_base_a=q_smem_base,
                        smem_desc_base_b=k_smem_base,
                        tCrA_layout=tSrQ0.layout,
                        tCrB_layout=tSrK0.layout,
                        zero_init=True,
                        cta_group=self.cta_group_size,
                    )
                    gemm_qk_s1_scoped = partial(
                        sm100_helpers.gemm_ptx_precomputed,
                        acc_tmem_addr=Int32(
                            self.tmem_stage_stride + self.tmem_s_offset
                        ),
                        idesc=qk_idesc,
                        smem_desc_base_a=q_smem_base,
                        smem_desc_base_b=k_smem_base,
                        tCrA_layout=tSrQ0.layout,
                        tCrB_layout=tSrK0.layout,
                        zero_init=True,
                        cta_group=self.cta_group_size,
                    )
                else:
                    q_smem_start = sm100_desc.make_smem_desc_start_addr(
                        sQ[None, None, None, self.q_stage - 1].iterator
                    )
                    sm100_helpers.declare_ptx_smem_desc(
                        q_smem_start,
                        q_smem_base,
                        tSrQ0.layout,
                        var_name_prefix="lean_q_desc",
                    )
                    sm100_helpers.declare_ptx_idesc(
                        qk_mma_op, var_name="lean_qk_idesc"
                    )
                    q_wrap_offset = -(self.q_stage - 1) * sQ_stage_stride
                    q_advance_offset = sQ_stage_stride
                    gemm_qk_s0_wrap = partial(
                        sm100_helpers.gemm_ptx_precomputed_varname,
                        Int32(self.tmem_s_offset),
                        smem_desc_base_b=k_smem_base,
                        tCrB_layout=tSrK0.layout,
                        smem_var_name_prefix="lean_q_desc",
                        idesc_var_name="lean_qk_idesc",
                        smem_offset=q_wrap_offset,
                        zero_init=True,
                        cta_group=self.cta_group_size,
                        mma_kind=self.qk_mma_kind,
                    )
                    gemm_qk_s0_advance = partial(
                        sm100_helpers.gemm_ptx_precomputed_varname,
                        Int32(self.tmem_s_offset),
                        smem_desc_base_b=k_smem_base,
                        tCrB_layout=tSrK0.layout,
                        smem_var_name_prefix="lean_q_desc",
                        idesc_var_name="lean_qk_idesc",
                        smem_offset=q_advance_offset,
                        zero_init=True,
                        cta_group=self.cta_group_size,
                        mma_kind=self.qk_mma_kind,
                    )
                    gemm_qk_s1_wrap = partial(
                        sm100_helpers.gemm_ptx_precomputed_varname,
                        Int32(self.tmem_stage_stride + self.tmem_s_offset),
                        smem_desc_base_b=k_smem_base,
                        tCrB_layout=tSrK0.layout,
                        smem_var_name_prefix="lean_q_desc",
                        idesc_var_name="lean_qk_idesc",
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
                        smem_var_name_prefix="lean_q_desc",
                        idesc_var_name="lean_qk_idesc",
                        smem_offset=q_advance_offset,
                        zero_init=True,
                        cta_group=self.cta_group_size,
                        mma_kind=self.qk_mma_kind,
                    )
                gemm_pv_0 = partial(
                    sm100_helpers.gemm_ptx_partial,
                    pv_mma_op,
                    Int32(self.tmem_o_offset),
                    tOrP[None, None, None, 0],
                    sA=None,
                    split_arrive=(
                        self.split_P_arrive if self.split_P_arrive > 0 else None),
                    tA_addr=Int32(self.tmem_p_offset),
                    cta_group=self.cta_group_size,
                    mma_kind=self.pv_mma_kind,
                )
                gemm_pv_1 = partial(
                    sm100_helpers.gemm_ptx_partial,
                    pv_mma_op,
                    Int32(self.tmem_o_offset + self.tmem_o_stage_stride),
                    tOrP[None, None, None, 1],
                    sA=None,
                    split_arrive=(
                        self.split_P_arrive if self.split_P_arrive > 0 else None),
                    tA_addr=Int32(self.tmem_stage_stride + self.tmem_p_offset),
                    cta_group=self.cta_group_size,
                    mma_kind=self.pv_mma_kind,
                )

                cute.arch.mbarrier_wait(mbar_k_ptr, 0)
                # Issue order:
                #   Q0K, Q1K, P0V, Q2K, P1V, Q3K, ...
                # This reuses each slot as soon as its previous PV drains,
                # instead of batching both PVs after both QKs of a pair.
                # The schedule is still 2-slot safe:
                #   - QK(qi) consumes slot qi&1
                #   - PV(qi-2) frees the same slot before QK(qi) reuses it
                #   - phases still toggle every 2 groups per slot

                # Prologue: issue up to the first two QK tiles. Q slots come
                # from the q_stage ring; S slots remain a 2-slot ring.
                pipeline_q.consumer_wait_w_index_phase(Int32(0), Int32(0))
                pipeline_s.producer_acquire_w_index_phase(Int32(0), Int32(1))
                if const_expr(use_scoped_q_desc):
                    gemm_qk_s0_scoped(
                        smem_desc_start_a=q_smem_start_0,
                        smem_desc_start_b=k_smem_start,
                    )
                else:
                    gemm_qk_s0_wrap(smem_desc_start_b=k_smem_start)
                pipeline_s.producer_commit_w_index(Int32(0))
                pipeline_q.consumer_release_w_index(Int32(0))

                if num_q_groups > Int32(1):
                    pipeline_q.consumer_wait_w_index_phase(Int32(1), Int32(0))
                    pipeline_s.producer_acquire_w_index_phase(Int32(1), Int32(1))
                    if const_expr(use_scoped_q_desc):
                        gemm_qk_s1_scoped(
                            smem_desc_start_a=(
                                q_smem_start_0 + Int32(sQ_stage_stride)
                            ),
                            smem_desc_start_b=k_smem_start,
                        )
                    else:
                        gemm_qk_s1_advance(smem_desc_start_b=k_smem_start)
                    pipeline_s.producer_commit_w_index(Int32(1))
                    pipeline_q.consumer_release_w_index(Int32(1))

                cute.arch.mbarrier_wait(mbar_v_ptr, 0)

                # Steady-state: for qi >= 2, reuse the S/P slot as a BLK128-
                # style handoff. The MMA warp waits for softmax to release
                # the slot after P is visible, issues PV, then immediately
                # reuses the same acquired slot for the next QK.
                for qi in cutlass.range(Int32(2), num_q_groups, unroll=1):
                    pv_qi = qi - Int32(2)
                    pv_slot = pv_qi & Int32(1)
                    pv_phase = (pv_qi // Int32(2)) & Int32(1)
                    pipeline_p.consumer_wait_w_index_phase(pv_slot, pv_phase)
                    pipeline_o.producer_acquire_w_index_phase(
                        pv_slot, pv_phase ^ Int32(1))
                    if pv_slot == Int32(0):
                        gemm_pv_0(
                            tCrB=tOrV0,
                            sB=sV0,
                            mbar_ptr=(
                                pipeline_p_lastsplit.sync_object_full.get_barrier(pv_slot)
                                if self.split_P_arrive > 0 else None),
                            mbar_phase=(pv_phase if self.split_P_arrive > 0 else None),
                            zero_init=True,
                        )
                    else:
                        gemm_pv_1(
                            tCrB=tOrV0,
                            sB=sV0,
                            mbar_ptr=(
                                pipeline_p_lastsplit.sync_object_full.get_barrier(pv_slot)
                                if self.split_P_arrive > 0 else None),
                            mbar_phase=(pv_phase if self.split_P_arrive > 0 else None),
                            zero_init=True,
                        )
                    pipeline_o.producer_commit_w_index(pv_slot)
                    if cutlass.const_expr(self.split_P_arrive > 0):
                        pipeline_p_lastsplit.consumer_release_w_index(pv_slot)
                    pipeline_p.consumer_release_w_index(pv_slot)

                    q_slot = qi % Int32(self.q_stage)
                    q_phase = (qi // Int32(self.q_stage)) & Int32(1)
                    s_slot = qi & Int32(1)
                    s_phase = (qi // Int32(2)) & Int32(1)
                    pipeline_q.consumer_wait_w_index_phase(q_slot, q_phase)
                    pipeline_s.producer_acquire_w_index_phase(
                        s_slot, s_phase ^ Int32(1))
                    if const_expr(use_scoped_q_desc):
                        q_smem_start_stage = (
                            q_smem_start_0
                            + q_slot * Int32(sQ_stage_stride)
                        )
                        if s_slot == Int32(0):
                            gemm_qk_s0_scoped(
                                smem_desc_start_a=q_smem_start_stage,
                                smem_desc_start_b=k_smem_start,
                            )
                        else:
                            gemm_qk_s1_scoped(
                                smem_desc_start_a=q_smem_start_stage,
                                smem_desc_start_b=k_smem_start,
                            )
                    else:
                        if s_slot == Int32(0):
                            if q_slot == Int32(0):
                                gemm_qk_s0_wrap(smem_desc_start_b=k_smem_start)
                            else:
                                gemm_qk_s0_advance(smem_desc_start_b=k_smem_start)
                        else:
                            if q_slot == Int32(0):
                                gemm_qk_s1_wrap(smem_desc_start_b=k_smem_start)
                            else:
                                gemm_qk_s1_advance(smem_desc_start_b=k_smem_start)
                    pipeline_s.producer_commit_w_index(s_slot)
                    pipeline_q.consumer_release_w_index(q_slot)

                # Drain the remaining one or two PV tiles.
                drain_begin = Int32(0) if num_q_groups == Int32(1) else num_q_groups - Int32(2)
                for pv_qi in cutlass.range(drain_begin, num_q_groups, unroll=1):
                    pv_slot = pv_qi & Int32(1)
                    pv_phase = (pv_qi // Int32(2)) & Int32(1)
                    pipeline_p.consumer_wait_w_index_phase(pv_slot, pv_phase)
                    pipeline_o.producer_acquire_w_index_phase(
                        pv_slot, pv_phase ^ Int32(1))
                    if pv_slot == Int32(0):
                        gemm_pv_0(
                            tCrB=tOrV0,
                            sB=sV0,
                            mbar_ptr=(
                                pipeline_p_lastsplit.sync_object_full.get_barrier(pv_slot)
                                if self.split_P_arrive > 0 else None),
                            mbar_phase=(pv_phase if self.split_P_arrive > 0 else None),
                            zero_init=True,
                        )
                    else:
                        gemm_pv_1(
                            tCrB=tOrV0,
                            sB=sV0,
                            mbar_ptr=(
                                pipeline_p_lastsplit.sync_object_full.get_barrier(pv_slot)
                                if self.split_P_arrive > 0 else None),
                            mbar_phase=(pv_phase if self.split_P_arrive > 0 else None),
                            zero_init=True,
                        )
                    pipeline_o.producer_commit_w_index(pv_slot)
                    if cutlass.const_expr(self.split_P_arrive > 0):
                        pipeline_p_lastsplit.consumer_release_w_index(pv_slot)
                    pipeline_p.consumer_release_w_index(pv_slot)

    @cute.jit
    def _softmax_step(
        self,
        slot: cutlass.Constexpr[int],
        s_consumer_phase: Int32,
        p_producer_phase: Int32,
        sm_stats_producer_phase: Int32,
        softmax: SoftmaxSm100,
        sScale: cute.Tensor,
        sScaleTemperature: cute.Tensor,
        pipeline_s,
        pipeline_p,
        pipeline_p_lastsplit,
        pipeline_sm_stats,
        sm_stats_barrier,
        stats_barrier_idx: Int32,
        thr_tmem_load,
        thr_tmem_store,
        tStS_t2r: cute.Tensor,
        tStP_r2t: cute.Tensor,
        tScS_t2r: cute.Tensor,
        tScP_shape,
        sQIdxMeta: cute.Tensor,
        qidx_meta_slot: Int32,
        group_tidx: Int32,
        masked_tok_count: Int32,
        kv_block_col_start: Int32,
        seq_len_q: Int32,
        causal_q_offset: Int32,
        kv_valid_cols: Int32,
        lse_temperature_scale: Float32,
        return_temperature_lse: cutlass.Constexpr[bool],
        apply_causal_mask: cutlass.Constexpr[bool] = False,
        signal_stats_barrier: cutlass.Constexpr[bool] = True,
        publish_stats: cutlass.Constexpr[bool] = True,
    ):
        slot_rt = Int32(slot)

        pipeline_s.consumer_wait_w_index_phase(slot_rt, s_consumer_phase)

        tSrS_t2r = cute.make_rmem_tensor(
            tScS_t2r.shape,
            self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)

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
        if const_expr(self.causal and apply_causal_mask):
            need_causal_mask = masked_tok_count > Int32(0)
            if need_causal_mask:
                tok_idx = group_tidx // Int32(self.qheadperkv)
                q_idx = self._decode_q_idx_from_qsplit(
                    sQIdxMeta[qidx_meta_slot + tok_idx])
                mask.apply_mask_sm100(
                    tSrS_t2r,
                    tScS_t2r,
                    m_block=Int32(0),
                    n_block=Int32(0),
                    mask_seqlen=True,
                    mask_causal=True,
                    row_idx=q_idx,
                    kv_valid_cols=kv_valid_cols,
                    kv_block_col_start=kv_block_col_start,
                )
            else:
                mask.apply_mask_sm100(
                    tSrS_t2r,
                    tScS_t2r,
                    m_block=Int32(0),
                    n_block=Int32(0),
                    mask_seqlen=True,
                    mask_causal=False,
                    kv_valid_cols=kv_valid_cols,
                )
        else:
            mask.apply_mask_sm100(
                tSrS_t2r,
                tScS_t2r,
                m_block=Int32(0),
                n_block=Int32(0),
                mask_seqlen=True,
                mask_causal=False,
                kv_valid_cols=kv_valid_cols,
            )

        # Each sparse CTA computes exactly one KV block for the current Q group,
        # so full-tile softmax is always the first and only online-softmax step.
        row_max, _ = softmax.update_row_max(tSrS_t2r.load(), True)
        if const_expr(return_temperature_lse and self.scale_fp8_p):
            tSrS_temperature = cute.make_rmem_tensor_like(tSrS_t2r)
            tSrS_temperature.store(tSrS_t2r.load())
            softmax.scale_subtract_rowmax(tSrS_temperature, row_max)
        softmax.scale_subtract_rowmax(
            tSrS_t2r, row_max, fp8_probability=self.scale_fp8_p
        )
        if const_expr(return_temperature_lse):
            lse_temperature_row_sum = softmax.compute_scaled_exp2_row_sum(
                tSrS_temperature if const_expr(self.scale_fp8_p) else tSrS_t2r,
                lse_temperature_scale,
            )

        if cutlass.const_expr(self.split_P_arrive > 0):
            # This full barrier is the late-P handoff consumed inside
            # gemm_ptx_partial after its early PV k-slices are issued.
            pipeline_p_lastsplit.producer_acquire_w_index_phase(
                slot_rt, p_producer_phase)
        pipeline_p.producer_acquire_w_index_phase(slot_rt, p_producer_phase)
        tSrP_r2t_f32 = cute.make_rmem_tensor(
            thr_tmem_store.partition_S(
                cute.make_identity_tensor(tScP_shape)).shape,
            Float32)
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.p_dtype),
            tSrS_t2r.layout)
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=0 if const_expr(self.scale_fp8_p) else self.ex2_emu_freq,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
            sparse_attn_p_mode=(
                self.sparse_attn_p_mode
                and self.q_dtype == cutlass.BFloat16
                and not self.paged_kv
            ),
        )

        for k in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(
                thr_tmem_store,
                tSrP_r2t_f32[None, None, k],
                tStP_r2t[None, None, k])
            if cutlass.const_expr(self.split_P_arrive > 0):
                split_idx = (
                    cute.size(tStP_r2t.shape[2])
                    * self.split_P_arrive // self.n_block_size)
                if cutlass.const_expr(k + 1 == split_idx):
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_p.producer_commit_w_index(slot_rt)
        cute.arch.fence_view_async_tmem_store()
        if cutlass.const_expr(self.split_P_arrive == 0):
            pipeline_p.producer_commit_w_index(slot_rt)
        else:
            pipeline_p_lastsplit.producer_commit_w_index(slot_rt)
        if const_expr(publish_stats):
            pipeline_sm_stats.producer_acquire_w_index_phase(
                slot_rt, sm_stats_producer_phase)
        if const_expr(not self.dpsum_only):
            softmax.update_row_sum(tSrS_t2r.load(), Float32(0.0), True)
        del tSrS_t2r
        if const_expr(publish_stats):
            sScale_slot = cute.make_tensor(
                sScale.iterator + slot_rt * Int32(self.m_block_size * 2),
                cute.make_layout(self.m_block_size * 2),
            )
            if const_expr(not self.dpsum_only):
                sScale_slot[group_tidx] = softmax.row_sum[0]
            sScale_slot[group_tidx + Int32(self.m_block_size)] = (
                softmax.row_max[0])
            if const_expr(return_temperature_lse):
                sScale_temperature_slot = cute.make_tensor(
                    sScaleTemperature.iterator + slot_rt * Int32(self.m_block_size),
                    cute.make_layout(self.m_block_size),
                )
                sScale_temperature_slot[group_tidx] = lse_temperature_row_sum
            cute.arch.fence_view_async_shared()
            pipeline_sm_stats.producer_commit_w_index(slot_rt)

        if const_expr(publish_stats and signal_stats_barrier):
            sm_stats_barrier.arrive_w_index(index=stats_barrier_idx)
        pipeline_s.consumer_release_w_index(slot_rt)

    @cute.jit
    def softmax(
        self,
        stage: cutlass.Constexpr[int],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tStS: cute.Tensor,
        sScale: cute.Tensor,
        sScaleTemperature: cute.Tensor,
        sQIdxMeta: cute.Tensor,
        pipeline_s,
        pipeline_p,
        pipeline_p_lastsplit,
        pipeline_o,
        pipeline_sm_stats,
        sm_stats_barrier,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mLSE_temperature_partial: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        lse_temperature_scale: Float32,
        kv_block_idx: Int32,
        kv_valid_cols: Int32,
        diag_q_count: Int32,
        num_q_groups: Int32,
        count_raw: Int32,
        has_work: Int32,
        causal_q_offset: Int32,
        head_kv_idx: Int32,
        seq_len_q: Int32,
        head_q: Int32,
        q_batch_offset: Int32,
        mQ_2d: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx_in_wg = warp_idx % Int32(self.warps_per_group)
        group_tidx = (
            warp_idx_in_wg * Int32(cute.arch.WARP_SIZE)
            + tidx % Int32(cute.arch.WARP_SIZE)
        )
        stats_barrier_idx = (
            Int32(stage) * Int32(self.warps_per_group) + warp_idx_in_wg)

        thr0_qk = tiled_mma_qk.get_slice(0)
        tScS = thr0_qk.partition_C(
            cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]
        cta_qk_tiler = (
            self.mma_tiler_qk[0] // thr0_qk.thr_id.shape,
            self.mma_tiler_qk[1])
        tilePlikeFP32 = (
            self.mma_tiler_qk[1] // Float32.width * self.p_dtype.width)
        tScP_shape = (cta_qk_tiler[0], tilePlikeFP32)
        tSAcc = tStS[(None, None), 0, 0, stage]

        softmax = SoftmaxSm100.create(softmax_scale_log2)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype)
        thr_tmem_load = tcgen05.make_tmem_copy(
            tmem_load_atom, tSAcc).get_slice(group_tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        tStP_layout = cute.composition(
            tSAcc.layout,
            cute.make_layout((self.m_block_size, tilePlikeFP32)))
        tStP = cute.make_tensor(
            tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)
        # P-store Repetition is dtype-aware: each PV MMA K-segment is
        # ``32 / (p_dtype.width / 8)`` fp8/bf16 columns wide, which equals
        # ``32 * Float32.width / p_dtype.width`` packed fp32 TMEM columns
        # ``// (p_dtype.width / 8)``. Concretely, R=16 packs two bf16 PV K
        # segments per chunk (shape[2]=4 ⇒ 3/4 publish boundary aligns),
        # while fp8 (PV K=32 fp8 ⇒ 8 fp32 cols) needs R=8 so
        # shape[2]=4 and split_idx=3 publishes exactly 24 fp32 cols
        # (= 96 fp8 cols = 3 PV K segments) at the early-arrive edge.
        store_rep = const_expr(
            8 if self.p_dtype == cutlass.Float8E4M3FN else 16
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(store_rep)),
            Float32)
        thr_tmem_store = tcgen05.make_tmem_copy(
            tmem_store_atom, tStP).get_slice(group_tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)

        if const_expr(self.store_raw_partial):
            thr0_pv = tiled_mma_pv.get_slice(0)
            pv_acc_shape = thr0_pv.partition_shape_C(self.mma_tiler_pv[:2])
            tOtO_base = thr0_pv.make_fragment_C(pv_acc_shape)
            raw_o_tile_size = 64
            tOcO = thr0_pv.partition_C(
                cute.make_identity_tensor(self.mma_tiler_pv[:2])
            )
            tOcO_i = cute.logical_divide(
                tOcO,
                cute.make_layout((self.m_block_size, raw_o_tile_size)),
            )
            o_tmem_copy_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
                self.pv_acc_dtype,
            )

        if has_work:
            kv_block_col_start = Int32(0)
            if const_expr(self.causal):
                kv_block_col_start = kv_block_idx * Int32(self.n_block_size)

            num_stage_groups = (
                num_q_groups + Int32(1 - stage)) // Int32(2)
            for qi_iter in cutlass.range(num_stage_groups, unroll=1):
                qi_group = qi_iter * Int32(2) + Int32(stage)
                phase = qi_iter & Int32(1)
                producer_phase = phase ^ Int32(1)
                qidx_meta_slot = (
                    (qi_group & Int32(self.qidx_meta_stages - 1))
                    * Int32(self.q_tokens_per_group)
                )

                softmax.reset()

                if const_expr(self.causal):
                    qi_group_start = qi_group * Int32(self.q_tokens_per_group)
                    masked_tok_count = cutlass.max(
                        Int32(0),
                        cutlass.min(
                            Int32(self.q_tokens_per_group),
                            diag_q_count - qi_group_start,
                        ),
                    )
                    self._softmax_step(
                        stage,
                        phase,
                        producer_phase,
                        producer_phase,
                        softmax,
                        sScale,
                        sScaleTemperature,
                        pipeline_s,
                        pipeline_p,
                        pipeline_p_lastsplit,
                        pipeline_sm_stats,
                        sm_stats_barrier,
                        stats_barrier_idx,
                        thr_tmem_load,
                        thr_tmem_store,
                        tStS_t2r,
                        tStP_r2t,
                        tScS_t2r,
                        tScP_shape,
                        sQIdxMeta,
                        qidx_meta_slot,
                        group_tidx,
                        masked_tok_count,
                        kv_block_col_start,
                        seq_len_q,
                        causal_q_offset,
                        kv_valid_cols,
                        lse_temperature_scale,
                        const_expr(
                            mLSE_temperature_partial is not None
                            and not self.dpsum_only
                        ),
                        True,
                        False,
                        const_expr(not self.store_raw_partial),
                    )
                else:
                    self._softmax_step(
                        stage,
                        phase,
                        producer_phase,
                        producer_phase,
                        softmax,
                        sScale,
                        sScaleTemperature,
                        pipeline_s,
                        pipeline_p,
                        pipeline_p_lastsplit,
                        pipeline_sm_stats,
                        sm_stats_barrier,
                        stats_barrier_idx,
                        thr_tmem_load,
                        thr_tmem_store,
                        tStS_t2r,
                        tStP_r2t,
                        tScS_t2r,
                        tScP_shape,
                        sQIdxMeta,
                        qidx_meta_slot,
                        group_tidx,
                        Int32(0),
                        kv_block_col_start,
                        seq_len_q,
                        Int32(0),
                        kv_valid_cols,
                        lse_temperature_scale,
                        const_expr(
                            mLSE_temperature_partial is not None
                            and not self.dpsum_only
                        ),
                        False,
                        False,
                        const_expr(not self.store_raw_partial),
                    )

                if const_expr(self.store_raw_partial):
                    self._store_raw_partial_step(
                        qi_group,
                        group_tidx,
                        tOtO_base,
                        tOcO_i,
                        o_tmem_copy_atom,
                        sQIdxMeta,
                        pipeline_o,
                        mO_partial,
                        mLSE_partial,
                        softmax.row_sum[0],
                        softmax.row_max[0],
                        softmax_scale_log2,
                        count_raw,
                        head_kv_idx,
                        head_q,
                        q_batch_offset,
                        mQ_2d.shape[0] // head_q,
                    )

    @cute.jit
    def _store_raw_partial_step(
        self,
        qi_group: Int32,
        group_tidx: Int32,
        tOtO_base: cute.Tensor,
        tOcO_i: cute.Tensor,
        o_tmem_copy_atom,
        sQIdxMeta: cute.Tensor,
        pipeline_o,
        mO_partial: cute.Tensor,
        mPartialStats: cute.Tensor,
        row_sum: Float32,
        row_max: Float32,
        softmax_scale_log2: Float32,
        count_raw: Int32,
        head_kv_idx: Int32,
        head_q: Int32,
        q_batch_offset: Int32,
        total_q: Int32,
    ):
        """Store raw PV accumulators and FP32 stats from the softmax WG."""
        slot = qi_group & Int32(1)
        phase = (qi_group // Int32(2)) & Int32(1)
        stage_base = slot * Int32(self.tmem_o_stage_stride)
        raw_o_tile_size = 64
        qidx_meta_slot = (
            qi_group & Int32(self.qidx_meta_stages - 1)
        ) * Int32(self.q_tokens_per_group)

        # Each output warp owns two Q tokens. Load their packed metadata once
        # and broadcast it across the warp, matching the CUDA raw epilogue.
        # Keeping this outside the vector-store loops avoids one shared load
        # for every 128-bit output vector.
        lane_idx = cute.arch.lane_idx()
        warp_idx_in_wg = group_tidx // Int32(cute.arch.WARP_SIZE)
        output_token_base = warp_idx_in_wg * Int32(2)
        output_qsplit_0 = Int32(0)
        output_qsplit_1 = Int32(0)
        if lane_idx == Int32(0):
            output_qsplit_0 = sQIdxMeta[
                qidx_meta_slot + output_token_base
            ]
            output_qsplit_1 = sQIdxMeta[
                qidx_meta_slot + output_token_base + Int32(1)
            ]
        output_qsplit_0 = cute.arch.shuffle_sync(
            output_qsplit_0, Int32(0)
        )
        output_qsplit_1 = cute.arch.shuffle_sync(
            output_qsplit_1, Int32(0)
        )

        pipeline_o.consumer_wait_w_index_phase(slot, phase)
        tOtO = cute.make_tensor(
            tOtO_base.iterator + stage_base + Int32(self.tmem_o_offset),
            tOtO_base.layout,
        )
        for col_pass_idx in cutlass.range(Int32(2), unroll=2):
            col_pass = col_pass_idx * Int32(raw_o_tile_size)
            tOtO_pass_ptr = cute.make_ptr(
                self.pv_acc_dtype,
                tOtO.iterator.toint() + col_pass,
                cute.AddressSpace.tmem,
                assumed_align=8,
            )
            tOtO_pass = cute.make_tensor(tOtO_pass_ptr, tOtO.layout)
            tOtO_pass_i = cute.logical_divide(
                tOtO_pass,
                cute.make_layout((self.m_block_size, raw_o_tile_size)),
            )
            tiled_tmem_load_pass = tcgen05.make_tmem_copy(
                o_tmem_copy_atom, tOtO_pass_i[(None, None), 0]
            )
            thr_tmem_load_pass = tiled_tmem_load_pass.get_slice(group_tidx)
            tOtO_t2r_pass = thr_tmem_load_pass.partition_S(
                tOtO_pass_i[(None, None), None]
            )
            tOcO_t2r_pass = thr_tmem_load_pass.partition_D(
                tOcO_i[(None, None), None]
            )
            tOtO_t2r_i = tOtO_t2r_pass[None, None, None, 0]
            tOcO_t2r_i = tOcO_t2r_pass[None, None, None, 0]
            tOrO_frg = cute.make_rmem_tensor_like(
                tOcO_t2r_i, self.pv_acc_dtype
            )
            cute.copy(tiled_tmem_load_pass, tOtO_t2r_i, tOrO_frg)

            tOrO_mn = make_16x256b_tensor_mn_view(tOrO_frg)
            tOrO_mn = cute.make_tensor(
                tOrO_mn.iterator,
                cute.select(tOrO_mn.layout, mode=[0, 1]),
            )
            tOcO_mn = make_16x256b_tensor_mn_view(tOcO_t2r_i)
            tOcO_mn = cute.make_tensor(
                tOcO_mn.iterator,
                cute.select(tOcO_mn.layout, mode=[0, 1]),
            )
            num_rows = cute.size(tOrO_mn, mode=[0])
            num_cols = cute.size(tOrO_mn, mode=[1])

            for r in cutlass.range_constexpr(num_rows):
                assert num_cols % 8 == 0
                for c8 in cutlass.range_constexpr(num_cols // 8):
                    c_base = Int32(c8) * Int32(8)
                    row_col = tOcO_mn[r, c_base]
                    row = row_col[0]
                    col = row_col[1] + col_pass
                    if row < Int32(self.m_block_size):
                        tok = row // Int32(self.qheadperkv)
                        row_in_tok = row - tok * Int32(self.qheadperkv)
                        qi = qi_group * Int32(self.q_tokens_per_group) + tok
                        if qi < count_raw:
                            qsplit = (
                                output_qsplit_0
                                if tok == output_token_base
                                else output_qsplit_1
                            )
                            q_idx = self._decode_q_idx_from_qsplit(qsplit)
                            split = self._decode_split_idx_from_qsplit(qsplit)
                            q_abs = q_batch_offset + q_idx
                            flat_row = (
                                Int64(split) * Int64(total_q) * Int64(head_q)
                                + Int64(q_abs) * Int64(head_q)
                                + Int64(head_kv_idx) * Int64(self.qheadperkv)
                                + Int64(row_in_tok)
                            )
                            row_base_ptr = flat_row * Int64(self.head_dim)
                            fake_col = real_col_to_stg128_half_fake_col(col)
                            ptr = (
                                mO_partial.iterator
                                + row_base_ptr
                                + Int64(fake_col)
                            )
                            self._store_o_partial_vec8_half(
                                ptr,
                                tOrO_mn[r, c_base],
                                tOrO_mn[r, c_base + Int32(1)],
                                tOrO_mn[r, c_base + Int32(2)],
                                tOrO_mn[r, c_base + Int32(3)],
                                tOrO_mn[r, c_base + Int32(4)],
                                tOrO_mn[r, c_base + Int32(5)],
                                tOrO_mn[r, c_base + Int32(6)],
                                tOrO_mn[r, c_base + Int32(7)],
                            )
        cute.arch.fence_view_async_tmem_load()

        stats_valid = (
            row_sum > Float32(0.0)
            and row_sum < Float32.inf
            and row_max > -Float32.inf
            and row_max < Float32.inf
            and row_sum == row_sum
            and row_max == row_max
        )
        scaled_max_log2 = (
            self._probability_max_log2(row_max, softmax_scale_log2)
            if stats_valid
            else -Float32.inf
        )
        row_sum_store = row_sum if stats_valid else Float32(0.0)
        lane_idx = cute.arch.lane_idx()
        pair_lane = lane_idx | Int32(1)
        paired_scaled_max_log2 = cute.arch.shuffle_sync(
            scaled_max_log2, pair_lane
        )
        paired_row_sum = cute.arch.shuffle_sync(row_sum_store, pair_lane)
        if group_tidx % Int32(2) == Int32(0):
            tok = group_tidx // Int32(self.qheadperkv)
            qi = qi_group * Int32(self.q_tokens_per_group) + tok
            if qi < count_raw:
                qsplit = (
                    output_qsplit_0
                    if tok == output_token_base
                    else output_qsplit_1
                )
                q_idx = self._decode_q_idx_from_qsplit(qsplit)
                split = self._decode_split_idx_from_qsplit(qsplit)
                q_abs = q_batch_offset + q_idx
                h_abs = (
                    head_kv_idx * Int32(self.qheadperkv)
                    + group_tidx % Int32(self.qheadperkv)
                )
                stats_offset = (
                    (Int64(split) * Int64(total_q) + Int64(q_abs))
                    * Int64(head_q)
                    + Int64(h_abs)
                ) * Int64(2)
                copy_utils.stg_128_cs(
                    mPartialStats.iterator + stats_offset,
                    scaled_max_log2,
                    row_sum_store,
                    paired_scaled_max_log2,
                    paired_row_sum,
                )

        pipeline_o.consumer_release_w_index(slot)

    @cute.jit
    def epilogue(
        self,
        tiled_mma_pv: cute.TiledMma,
        sScale: cute.Tensor,
        sScaleTemperature: cute.Tensor,
        sSplitIdx: cute.Tensor,
        sQIdx: cute.Tensor,
        sQIdxMeta: cute.Tensor,
        sDO: Optional[cute.Tensor],
        pipeline_o,
        pipeline_sm_stats,
        epilogue_barrier,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mLSE_temperature_partial: Optional[cute.Tensor],
        mPQuantScale: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        lse_temperature_scale_log2: Float32,
        count_raw: Int32,
        has_work: Int32,
        batch_idx: Int32,
        head_kv_idx: Int32,
        head_q: Int32,
        q_batch_offset: Int32,
        mQ_2d: cute.Tensor,
        num_q_groups: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx_in_wg = warp_idx - Int32(self.correction_warp_base)
        group_tidx = (
            warp_idx_in_wg * Int32(cute.arch.WARP_SIZE)
            + tidx % Int32(cute.arch.WARP_SIZE)
        )
        total_q = mQ_2d.shape[0] // head_q
        thr0_pv = tiled_mma_pv.get_slice(0)
        pv_acc_shape = thr0_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO_base = thr0_pv.make_fragment_C(pv_acc_shape)
        corr_tile_size = 64
        tOcO = thr0_pv.partition_C(
            cute.make_identity_tensor(self.mma_tiler_pv[:2])
        )
        tOcO_i = cute.logical_divide(
            tOcO,
            cute.make_layout((self.m_block_size, corr_tile_size)),
        )
        o_tmem_copy_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
            self.pv_acc_dtype,
        )
        if has_work:
            for qi_group in cutlass.range(num_q_groups, unroll=1):
                softmax_stage = qi_group & Int32(1)
                self._epilogue_step(
                    qi_group,
                    group_tidx,
                    warp_idx_in_wg,
                    tOtO_base,
                    tOcO_i,
                    o_tmem_copy_atom,
                    sScale,
                    sScaleTemperature,
                    sSplitIdx,
                    sQIdx,
                    sQIdxMeta,
                    sDO,
                    pipeline_o,
                    pipeline_sm_stats,
                    epilogue_barrier,
                    mO_partial,
                    mLSE_partial,
                    mLSE_temperature_partial,
                    mPQuantScale,
                    softmax_scale_log2,
                    lse_temperature_scale_log2,
                    count_raw,
                    batch_idx,
                    head_kv_idx,
                    head_q,
                    q_batch_offset,
                    total_q,
                    True,
                    softmax_stage,
                )
    @cute.jit
    def _store_o_partial_vec8_half(
        self,
        ptr: cute.Pointer,
        v0: Float32,
        v1: Float32,
        v2: Float32,
        v3: Float32,
        v4: Float32,
        v5: Float32,
        v6: Float32,
        v7: Float32,
    ):
        if cutlass.const_expr(self.o_dtype is cutlass.BFloat16):
            stg_128_bf16_cs(ptr, v0, v1, v2, v3, v4, v5, v6, v7)
        else:
            stg_128_f16_cs(ptr, v0, v1, v2, v3, v4, v5, v6, v7)

    @cute.jit
    def _epilogue_step(
        self,
        qi_group: Int32,
        group_tidx: Int32,
        warp_idx_in_wg: Int32,
        tOtO_base: cute.Tensor,
        tOcO_i: cute.Tensor,
        o_tmem_copy_atom,
        sScale: cute.Tensor,
        sScaleTemperature: cute.Tensor,
        sSplitIdx: cute.Tensor,
        sQIdx: cute.Tensor,
        sQIdxMeta: cute.Tensor,
        sDO: Optional[cute.Tensor],
        pipeline_o,
        pipeline_sm_stats,
        epilogue_barrier,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mLSE_temperature_partial: Optional[cute.Tensor],
        mPQuantScale: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        lse_temperature_scale_log2: Float32,
        count_raw: Int32,
        batch_idx: Int32,
        head_kv_idx: Int32,
        head_q: Int32,
        q_batch_offset: Int32,
        total_q: Int32,
        use_stats_barrier: cutlass.Constexpr[bool],
        softmax_stage: cutlass.Constexpr[int],
    ):
        slot = qi_group & Int32(1)
        phase = (qi_group // Int32(2)) & Int32(1)
        stage_base = slot * Int32(self.tmem_o_stage_stride)
        corr_tile_size = 64
        sScale_slot = cute.make_tensor(
            sScale.iterator + slot * Int32(self.m_block_size * 2),
            cute.make_layout(self.m_block_size * 2),
        )
        sScale_temperature_slot = cute.make_tensor(
            sScaleTemperature.iterator + slot * Int32(self.m_block_size),
            cute.make_layout(self.m_block_size),
        )
        sSplitIdx_slot = cute.make_tensor(
            sSplitIdx.iterator + slot * Int32(self.q_tokens_per_group),
            cute.make_layout((self.q_tokens_per_group,)),
        )
        sQIdx_slot = cute.make_tensor(
            sQIdx.iterator + slot * Int32(self.q_tokens_per_group),
            cute.make_layout((self.q_tokens_per_group,)),
        )
        qidx_meta_slot = (qi_group
                          & Int32(self.qidx_meta_stages - 1)) * Int32(
                              self.q_tokens_per_group)

        if const_expr(self.dpsum_only):
            # Softmax stats are published with P, before the PV MMA completes.
            # Start the independent dO/LSE loads as soon as those stats are
            # ready so their GMEM latency overlaps the PV tensor-core work.
            if const_expr(use_stats_barrier):
                pipeline_sm_stats.consumer_wait_w_index_phase(slot, phase)
        else:
            pipeline_o.consumer_wait_w_index_phase(slot, phase)
            if const_expr(use_stats_barrier):
                pipeline_sm_stats.consumer_wait_w_index_phase(slot, phase)

        if const_expr(self.q_dtype != cutlass.Float8E4M3FN):
            if group_tidx < Int32(self.q_tokens_per_group):
                tok = group_tidx
                qi = qi_group * Int32(self.q_tokens_per_group) + tok
                if qi < count_raw:
                    qsplit = sQIdxMeta[qidx_meta_slot + tok]
                    q_idx = self._decode_q_idx_from_qsplit(qsplit)
                    sQIdx_slot[tok] = q_idx
                    sSplitIdx_slot[tok] = self._decode_split_idx_from_qsplit(qsplit)
            epilogue_barrier.arrive_and_wait_w_index(index=Int32(softmax_stage))

        if const_expr(self.dpsum_only):
            self._stage_dout(
                mO_partial,
                sDO,
                sQIdx_slot,
                qi_group,
                group_tidx,
                warp_idx_in_wg,
                count_raw,
                q_batch_offset,
                head_kv_idx,
                head_q,
            )

            tok_scale = group_tidx // Int32(self.qheadperkv)
            h_scale = group_tidx % Int32(self.qheadperkv)
            qi_scale = qi_group * Int32(self.q_tokens_per_group) + tok_scale
            block_to_global_scale = Float32(0.0)
            if qi_scale < count_raw:
                q_idx_scale = sQIdx_slot[tok_scale]
                q_abs_scale = q_batch_offset + q_idx_scale
                h_abs_scale = (
                    head_kv_idx * Int32(self.qheadperkv) + h_scale
                )
                row_max_scale = sScale_slot[
                    group_tidx + Int32(self.m_block_size)
                ]
                global_lse_log2 = (
                    mLSE_partial[q_abs_scale, h_abs_scale]
                    * Float32(math.log2(math.e))
                )
                valid_scale = (
                    row_max_scale != -Float32.inf
                    and row_max_scale == row_max_scale
                    and global_lse_log2 != -Float32.inf
                    and global_lse_log2 == global_lse_log2
                )
                if valid_scale:
                    block_to_global_scale = cute.math.exp2(
                        self._probability_max_log2(row_max_scale, softmax_scale_log2)
                        - global_lse_log2,
                        fastmath=True,
                    )
            # The row sum is no longer needed in this specialization. Reuse
            # its slot for the block-to-global probability scale consumed by
            # every vector fragment of this row.
            sScale_slot[group_tidx] = block_to_global_scale
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            epilogue_barrier.arrive_and_wait_w_index(
                index=Int32(softmax_stage)
            )

            pipeline_o.consumer_wait_w_index_phase(slot, phase)

        if const_expr(self.dpsum_only):
            # The 16x256b fragment gives each thread four logical rows, with
            # every row distributed across four adjacent lanes. Accumulate
            # both 64-column passes by logical row before the warp reduction,
            # matching the row-owned backward preprocess reduction.
            dpsum_acc = cute.make_rmem_tensor((4,), Float32)
            dpsum_rows = cute.make_rmem_tensor((4,), Int32)
            dpsum_acc.fill(0.0)

        tOtO = cute.make_tensor(
            tOtO_base.iterator + stage_base + Int32(self.tmem_o_offset),
            tOtO_base.layout)
        for col_pass_idx in cutlass.range(Int32(2), unroll=2):
            col_pass = col_pass_idx * Int32(corr_tile_size)
            tOtO_pass_ptr = cute.make_ptr(
                self.pv_acc_dtype,
                tOtO.iterator.toint() + col_pass,
                cute.AddressSpace.tmem,
                assumed_align=8,
            )
            tOtO_pass = cute.make_tensor(tOtO_pass_ptr, tOtO.layout)
            tOtO_pass_i = cute.logical_divide(
                tOtO_pass,
                cute.make_layout((self.m_block_size, corr_tile_size)))
            tiled_tmem_load_pass = tcgen05.make_tmem_copy(
                o_tmem_copy_atom, tOtO_pass_i[(None, None), 0])
            thr_tmem_load_pass = tiled_tmem_load_pass.get_slice(group_tidx)
            tOtO_t2r_pass = thr_tmem_load_pass.partition_S(
                tOtO_pass_i[(None, None), None])
            tOcO_t2r_pass = thr_tmem_load_pass.partition_D(
                tOcO_i[(None, None), None])
            tOtO_t2r_i = tOtO_t2r_pass[None, None, None, 0]
            tOcO_t2r_i = tOcO_t2r_pass[None, None, None, 0]
            tOrO_frg = cute.make_rmem_tensor_like(
                tOcO_t2r_i, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load_pass, tOtO_t2r_i, tOrO_frg)

            tOrO_mn = make_16x256b_tensor_mn_view(tOrO_frg)
            tOrO_mn = cute.make_tensor(
                tOrO_mn.iterator,
                cute.select(tOrO_mn.layout, mode=[0, 1]))
            tOcO_mn = make_16x256b_tensor_mn_view(tOcO_t2r_i)
            tOcO_mn = cute.make_tensor(
                tOcO_mn.iterator,
                cute.select(tOcO_mn.layout, mode=[0, 1]))
            num_rows = cute.size(tOrO_mn, mode=[0])
            num_cols = cute.size(tOrO_mn, mode=[1])
            if const_expr(self.dpsum_only):
                assert num_rows == 2

            for r in cutlass.range_constexpr(num_rows):
                assert num_cols % 8 == 0, (
                    "half O_partial STG.128 requires the epilogue "
                    "TMEM fragment column count to be a multiple of 8"
                )
                for c8 in cutlass.range_constexpr(num_cols // 8):
                    c_base = Int32(c8) * Int32(8)
                    row_col = tOcO_mn[r, c_base]
                    row = row_col[0]
                    col = row_col[1] + col_pass
                    if const_expr(self.dpsum_only):
                        dpsum_slot = r + (c8 // 2) * 2
                        dpsum_rows[dpsum_slot] = row
                    if row < Int32(self.m_block_size):
                        tok = row // Int32(self.qheadperkv)
                        row_in_tok = row - tok * Int32(self.qheadperkv)
                        qi = (qi_group
                              * Int32(self.q_tokens_per_group)
                              + tok)
                        if qi < count_raw:
                            if const_expr(self.dpsum_only):
                                q_idx = sQIdx_slot[tok]
                            elif const_expr(
                                self.q_dtype == cutlass.Float8E4M3FN
                            ):
                                qsplit = sQIdxMeta[qidx_meta_slot + tok]
                                q_idx = self._decode_q_idx_from_qsplit(qsplit)
                                split = self._decode_split_idx_from_qsplit(qsplit)
                            else:
                                q_idx = sQIdx_slot[tok]
                                split = sSplitIdx_slot[tok]
                            q_abs = q_batch_offset + q_idx
                            if const_expr(self.dpsum_only):
                                # The 8-value TMEM fragment follows the
                                # STG.128 fake-column order: four contiguous
                                # BF16 pairs separated by 8 real columns.
                                # Load those pairs in the matching order so
                                # the dot product uses the real dO columns.
                                dpsum = Float32(0.0)
                                for pair in cutlass.range_constexpr(0, 8, 2):
                                    pair_coord = tOcO_mn[
                                        r, c_base + Int32(pair)
                                    ]
                                    dout_pair = self._load_dout_pair(
                                        sDO,
                                        row,
                                        pair_coord[1] + col_pass,
                                    )
                                    o_pair = (
                                        tOrO_mn[r, c_base + Int32(pair)],
                                        tOrO_mn[
                                            r, c_base + Int32(pair + 1)
                                        ],
                                    )
                                    o_pair = cute.arch.mul_packed_f32x2(
                                        o_pair,
                                        (dout_pair[0], dout_pair[1]),
                                    )
                                    dpsum += o_pair[0] + o_pair[1]
                                dpsum *= sScale_slot[row]
                                dpsum_acc[dpsum_slot] += dpsum
                            else:
                                o0 = tOrO_mn[r, c_base]
                                o1 = tOrO_mn[r, c_base + Int32(1)]
                                o2 = tOrO_mn[r, c_base + Int32(2)]
                                o3 = tOrO_mn[r, c_base + Int32(3)]
                                o4 = tOrO_mn[r, c_base + Int32(4)]
                                o5 = tOrO_mn[r, c_base + Int32(5)]
                                o6 = tOrO_mn[r, c_base + Int32(6)]
                                o7 = tOrO_mn[r, c_base + Int32(7)]
                                flat_row = (
                                    Int64(split)
                                    * Int64(total_q) * Int64(head_q)
                                    + Int64(q_abs) * Int64(head_q)
                                    + Int64(head_kv_idx)
                                    * Int64(self.qheadperkv)
                                    + Int64(row_in_tok)
                                )
                                row_base_ptr = (
                                    flat_row * Int64(self.head_dim)
                                )
                                if const_expr(not self.store_raw_partial):
                                    row_sum_val = sScale_slot[row]
                                    is_zero_or_nan = (
                                        row_sum_val == Float32(0.0)
                                        or row_sum_val != row_sum_val
                                    )
                                    row_scale = cute.arch.rcp_approx(
                                        row_sum_val
                                        if not is_zero_or_nan
                                        else Float32(1.0)
                                    )
                                    scale_pair = (row_scale, row_scale)
                                    o0, o1 = cute.arch.mul_packed_f32x2(
                                        (o0, o1), scale_pair
                                    )
                                    o2, o3 = cute.arch.mul_packed_f32x2(
                                        (o2, o3), scale_pair
                                    )
                                    o4, o5 = cute.arch.mul_packed_f32x2(
                                        (o4, o5), scale_pair
                                    )
                                    o6, o7 = cute.arch.mul_packed_f32x2(
                                        (o6, o7), scale_pair
                                    )
                                fake_col = real_col_to_stg128_half_fake_col(col)
                                ptr = (
                                    mO_partial.iterator
                                    + row_base_ptr
                                    + Int64(fake_col)
                                )
                                self._store_o_partial_vec8_half(
                                    ptr,
                                    o0,
                                    o1,
                                    o2,
                                    o3,
                                    o4,
                                    o5,
                                    o6,
                                    o7,
                                )
        cute.arch.fence_view_async_tmem_load()

        if const_expr(self.dpsum_only):
            lane = group_tidx % Int32(cute.arch.WARP_SIZE)
            for r in cutlass.range_constexpr(4):
                dpsum = utils.warp_reduce(
                    dpsum_acc[r], operator.add, width=4
                )
                row = dpsum_rows[r]
                tok = row // Int32(self.qheadperkv)
                row_in_tok = row - tok * Int32(self.qheadperkv)
                qi = qi_group * Int32(self.q_tokens_per_group) + tok
                if lane % Int32(4) == Int32(0) and qi < count_raw:
                    q_idx = sQIdx_slot[tok]
                    split = sSplitIdx_slot[tok]
                    h_abs = (
                        head_kv_idx * Int32(self.qheadperkv) + row_in_tok
                    )
                    q_abs = q_batch_offset + q_idx
                    row_max = sScale_slot[
                        row + Int32(self.m_block_size)
                    ]
                    row_max_scaled = Float32(0.0)
                    if row_max != -Float32.inf and row_max == row_max:
                        row_max_scaled = self._probability_max_log2(
                            row_max, softmax_scale_log2
                        )
                    mPQuantScale[Int32(0), split, q_abs, h_abs] = (
                        row_max_scaled
                    )
                    mPQuantScale[Int32(1), split, q_abs, h_abs] = (
                        sScale_slot[row]
                    )
                    q_padded_batch_offset = (
                        q_batch_offset + batch_idx * Int32(self.m_block_size)
                    ) // Int32(self.m_block_size) * Int32(self.m_block_size)
                    if const_expr(
                        cute.rank(mLSE_temperature_partial) == 3
                    ):
                        mLSE_temperature_partial[
                            split,
                            q_padded_batch_offset + q_idx,
                            h_abs,
                        ] = dpsum
                    else:
                        dpsum_ptr = utils.elem_pointer(
                            mLSE_temperature_partial,
                            (q_padded_batch_offset + q_idx, h_abs),
                        )
                        copy_utils.atomic_add_fp32(dpsum, dpsum_ptr)

        if const_expr(not self.dpsum_only):
            tok_local = Int32(group_tidx) // Int32(self.qheadperkv)
            h_local = Int32(group_tidx) % Int32(self.qheadperkv)
            qi_lse = qi_group * Int32(self.q_tokens_per_group) + tok_local
            if qi_lse < count_raw:
                row_sum_val = sScale_slot[group_tidx]
                row_max_val = sScale_slot[group_tidx + Int32(self.m_block_size)]
                is_invalid = (
                    row_sum_val <= Float32(0.0)
                    or row_sum_val != row_sum_val
                    or row_max_val != row_max_val
                )
                if const_expr(self.q_dtype == cutlass.Float8E4M3FN):
                    qsplit_lse = sQIdxMeta[qidx_meta_slot + tok_local]
                    q_idx_lse = self._decode_q_idx_from_qsplit(qsplit_lse)
                    split_lse = self._decode_split_idx_from_qsplit(qsplit_lse)
                else:
                    q_idx_lse = sQIdx_slot[tok_local]
                    split_lse = sSplitIdx_slot[tok_local]
                h_abs = head_kv_idx * Int32(self.qheadperkv) + h_local
                q_abs_lse = q_batch_offset + q_idx_lse
                LN2 = Float32(math.log(2.0))
                if const_expr(self.store_raw_partial):
                    scaled_max_log2 = (
                        self._probability_max_log2(row_max_val, softmax_scale_log2)
                        if not is_invalid
                        else -Float32.inf
                    )
                    mLSE_partial[split_lse, q_abs_lse, h_abs, 0] = (
                        scaled_max_log2
                    )
                    mLSE_partial[split_lse, q_abs_lse, h_abs, 1] = (
                        row_sum_val if not is_invalid else Float32(0.0)
                    )
                else:
                    lse_cur = (
                        (
                            self._probability_max_log2(row_max_val, softmax_scale_log2)
                            + cute.math.log2(row_sum_val, fastmath=True)
                        )
                        * LN2
                        if not is_invalid
                        else -Float32.inf
                    )
                    mLSE_partial[split_lse, q_abs_lse, h_abs] = lse_cur
                if const_expr(mLSE_temperature_partial is not None):
                    row_sum_temperature_val = sScale_temperature_slot[group_tidx]
                    is_temperature_zero_or_nan = (
                        row_sum_temperature_val == Float32(0.0)
                        or row_sum_temperature_val != row_sum_temperature_val
                    )
                    lse_temperature_cur = (
                        (
                            row_max_val * lse_temperature_scale_log2
                            + cute.math.log2(
                                row_sum_temperature_val,
                                fastmath=True,
                            )
                        )
                        * LN2
                        if not is_temperature_zero_or_nan
                        else -Float32.inf
                    )
                    mLSE_temperature_partial[split_lse, q_abs_lse, h_abs] = (
                        lse_temperature_cur
                    )
        epilogue_barrier.arrive_and_wait_w_index(index=Int32(softmax_stage))

        pipeline_sm_stats.consumer_release_w_index(slot)
        pipeline_o.consumer_release_w_index(slot)
