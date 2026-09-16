"""KV-owner fused sparse KL backward kernel for MSA v1 on SM100/SM103."""

import math
from dataclasses import dataclass
from typing import Optional

import cutlass
from cutlass import Boolean, Float32, Int32, Int64, const_expr
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
from cuda.bindings import driver as cuda

from msa_v1._common import copy_utils, layout_utils, pipeline
from msa_v1._common.barrier import ld_acquire, red_release
from msa_v1._common.blackwell_helpers import gemm_ptx_w_idx
from msa_v1._common.cute_dsl_utils import (
    ParamsBase,
    assume_tensor_aligned,
    exit_thread_if,
)
from msa_v1._common.tile_scheduler import WorkTileInfo
from msa_v1._common.tma_utils import (
    extract_tma_desc_ptr,
    prefetch_tma_desc_raw,
    tma_gather4,
)
from msa_v1.attention.prepare_scheduler import (
    KL_VALID_ROWS_MASK,
    KL_WRITER_RANK_MASK,
    KL_WRITER_RANK_SHIFT,
)


@dataclass
class _SparseKlClcState(ParamsBase):
    """Single-stage CLC state with a minimal live register footprint."""

    _mbar_ptr: cute.Pointer
    _response_ptr: cute.Pointer
    _phase: Int32
    _empty_mbar_offset: int
    _response_bytes: int

    @staticmethod
    def create(
        *,
        mbar_ptr: cute.Pointer,
        response_ptr: cute.Pointer,
        empty_mbar_offset: int,
        response_bytes: int,
    ) -> "_SparseKlClcState":
        return _SparseKlClcState(
            mbar_ptr,
            response_ptr,
            Int32(0),
            empty_mbar_offset,
            response_bytes,
        )

    def get_current_work(self):
        m_idx, n_idx, l_idx, is_valid = cute.arch.clc_response(
            self._response_ptr
        )
        cute.arch.fence_proxy("async.shared", space="cta")
        return WorkTileInfo(
            (m_idx, n_idx, l_idx, Int32(0)), is_valid
        )

    def prefetch_next_work_no_advance(self, *, loc=None, ip=None) -> None:
        self.prefetch_next_work_at_phase(
            self._phase, loc=loc, ip=ip
        )

    def prefetch_next_work_at_phase(
        self,
        phase: Int32,
        *,
        loc=None,
        ip=None,
    ) -> None:
        cute.arch.mbarrier_wait(
            self._mbar_ptr + Int32(self._empty_mbar_offset),
            phase ^ Int32(1),
            loc=loc,
            ip=ip,
        )
        with cute.arch.elect_one(loc=loc, ip=ip):
            cute.arch.mbarrier_arrive_and_expect_tx(
                self._mbar_ptr,
                self._response_bytes,
                Int32(0),
                loc=loc,
                ip=ip,
            )
            cute.arch.issue_clc_query(
                self._mbar_ptr,
                self._response_ptr,
                loc=loc,
                ip=ip,
            )

    def consumer_wait_at_phase(
        self,
        phase: Int32,
        *,
        loc=None,
        ip=None,
    ) -> None:
        cute.arch.mbarrier_wait(
            self._mbar_ptr, phase, loc=loc, ip=ip
        )

    def consumer_release_no_advance(self, *, loc=None, ip=None) -> None:
        with cute.arch.elect_one(loc=loc, ip=ip):
            cute.arch.mbarrier_arrive(
                self._mbar_ptr + Int32(self._empty_mbar_offset),
                Int32(0),
                loc=loc,
                ip=ip,
            )

    def producer_tail(self, *, loc=None, ip=None) -> None:
        cute.arch.mbarrier_wait(
            self._mbar_ptr + Int32(self._empty_mbar_offset),
            self._phase ^ Int32(1),
            loc=loc,
            ip=ip,
        )


@dataclass
class _SparseKlWorkState(ParamsBase):
    """Single-stage broadcast from the scheduler warp to data roles."""

    _mbar_ptr: cute.Pointer
    _response_ptr: cute.Pointer
    _phase: Int32
    _empty_mbar_offset: int

    @staticmethod
    def create(
        *,
        mbar_ptr: cute.Pointer,
        response_ptr: cute.Pointer,
        empty_mbar_offset: int,
    ) -> "_SparseKlWorkState":
        return _SparseKlWorkState(
            mbar_ptr,
            response_ptr,
            Int32(0),
            empty_mbar_offset,
        )

    def producer_acquire(self, *, loc=None, ip=None) -> None:
        cute.arch.mbarrier_wait(
            self._mbar_ptr + Int32(self._empty_mbar_offset),
            self._phase ^ Int32(1),
            loc=loc,
            ip=ip,
        )

    def producer_commit(
        self,
        work: WorkTileInfo,
        *,
        loc=None,
        ip=None,
    ) -> None:
        with cute.arch.elect_one(loc=loc, ip=ip):
            self._response_ptr[Int32(0)] = cutlass.select_(
                work.is_valid_tile,
                work.tile_idx[0],
                Int32(-1),
            )
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(
                self._mbar_ptr,
                Int32(0),
                loc=loc,
                ip=ip,
            )
        self._phase ^= Int32(1)

    def consumer_get(self, *, loc=None, ip=None) -> WorkTileInfo:
        cute.arch.mbarrier_wait(
            self._mbar_ptr,
            self._phase,
            loc=loc,
            ip=ip,
        )
        work_idx = self._response_ptr[Int32(0)]
        work = WorkTileInfo(
            (work_idx, Int32(0), Int32(0), Int32(0)),
            Boolean(work_idx >= Int32(0)),
        )
        with cute.arch.elect_one(loc=loc, ip=ip):
            cute.arch.mbarrier_arrive(
                self._mbar_ptr + Int32(self._empty_mbar_offset),
                Int32(0),
                loc=loc,
                ip=ip,
            )
        self._phase ^= Int32(1)
        return work

    def producer_tail(self, *, loc=None, ip=None) -> None:
        cute.arch.mbarrier_wait(
            self._mbar_ptr + Int32(self._empty_mbar_offset),
            self._phase ^ Int32(1),
            loc=loc,
            ip=ip,
        )


@dataclass
class _SparseKlClcScheduler(ParamsBase):
    """CLC scheduler over the device-bounded compact worklist."""

    _clc: _SparseKlClcState
    _work: _SparseKlWorkState
    _work_count: cute.Tensor

    @staticmethod
    def create(
        clc: _SparseKlClcState,
        work: _SparseKlWorkState,
        mWorkCount: cute.Tensor,
    ) -> "_SparseKlClcScheduler":
        return _SparseKlClcScheduler(clc, work, mWorkCount)

    @cute.jit
    def _map_clc_work(self, work) -> WorkTileInfo:
        record_idx = work.tile_idx[0]
        has_record = Boolean(record_idx < self._work_count[Int32(0)])
        return WorkTileInfo(
            (record_idx, Int32(0), Int32(0), Int32(0)),
            Boolean(work.is_valid_tile) & has_record,
        )

    @cute.jit
    def initial_work_tile_info(self) -> WorkTileInfo:
        work_idx = _block_idx_x_remat()
        return WorkTileInfo(
            (work_idx, Int32(0), Int32(0), Int32(0)),
            Boolean(work_idx < self._work_count[Int32(0)]),
        )

    @cute.jit
    def get_current_work(self) -> WorkTileInfo:
        return self._map_clc_work(self._clc.get_current_work())

    @cute.jit
    def prefetch_next_work(
        self,
        work: WorkTileInfo,
        *,
        loc=None,
        ip=None,
    ) -> None:
        self._clc.prefetch_next_work_no_advance(loc=loc, ip=ip)

    @cute.jit
    def advance_to_next_work(
        self,
        work: WorkTileInfo,
        *,
        loc=None,
        ip=None,
    ) -> WorkTileInfo:
        return self._work.consumer_get(loc=loc, ip=ip)

    @cute.jit
    def scheduler_advance_to_next_work(
        self,
        work: WorkTileInfo,
        *,
        loc=None,
        ip=None,
    ) -> WorkTileInfo:
        phase = self._clc._phase
        self._clc.consumer_wait_at_phase(phase, loc=loc, ip=ip)
        clc_work = self._clc.get_current_work()
        next_work = self._map_clc_work(clc_work)
        self._clc.consumer_release_no_advance(loc=loc, ip=ip)
        phase ^= Int32(1)
        skip_work = Boolean(clc_work.is_valid_tile) & Boolean(
            not next_work.is_valid_tile
        )
        while skip_work:
            self._clc.prefetch_next_work_at_phase(
                phase, loc=loc, ip=ip
            )
            self._clc.consumer_wait_at_phase(phase, loc=loc, ip=ip)
            clc_work = self._clc.get_current_work()
            next_work = self._map_clc_work(clc_work)
            self._clc.consumer_release_no_advance(loc=loc, ip=ip)
            phase ^= Int32(1)
            skip_work = Boolean(clc_work.is_valid_tile) & Boolean(
                not next_work.is_valid_tile
            )
        self._clc._phase = phase
        self._work.producer_acquire(loc=loc, ip=ip)
        self._work.producer_commit(next_work, loc=loc, ip=ip)
        return next_work

    def complete_current_work(self, *, loc=None, ip=None) -> None:
        """The KL role barriers already complete the current work item."""

    def producer_tail(self, *, loc=None, ip=None) -> None:
        self._clc.producer_tail(loc=loc, ip=ip)
        self._work.producer_tail(loc=loc, ip=ip)


@dataclass
class _SparseKlSingleWorkScheduler(ParamsBase):
    """One ordered work item per CTA for deterministic execution."""

    _work_count: cute.Tensor

    @staticmethod
    def create(mWorkCount: cute.Tensor) -> "_SparseKlSingleWorkScheduler":
        return _SparseKlSingleWorkScheduler(mWorkCount)

    @cute.jit
    def _invalid_work(self) -> WorkTileInfo:
        return WorkTileInfo(
            (Int32(-1), Int32(0), Int32(0), Int32(0)),
            Boolean(False),
        )

    @cute.jit
    def initial_work_tile_info(self) -> WorkTileInfo:
        work_idx = _block_idx_x_remat()
        return WorkTileInfo(
            (work_idx, Int32(0), Int32(0), Int32(0)),
            Boolean(work_idx < self._work_count[Int32(0)]),
        )

    @cute.jit
    def advance_to_next_work(
        self,
        work: WorkTileInfo,
        *,
        loc=None,
        ip=None,
    ) -> WorkTileInfo:
        return self._invalid_work()


@dsl_user_op
def _block_idx_x_remat(*, loc=None, ip=None) -> Int32:
    """Read CTA X locally so the value does not span all warp roles."""

    return Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %ctaid.x;",
            "=r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _atomic_add_fp32x4_if(
    predicate: Int32,
    value_0: Float32,
    value_1: Float32,
    value_2: Float32,
    value_3: Float32,
    gmem_ptr: cute.Pointer,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue one predicated vector FP32 global reduction."""

    llvm.inline_asm(
        None,
        [
            gmem_ptr.toint(loc=loc, ip=ip).ir_value(),
            Float32(value_0).ir_value(loc=loc, ip=ip),
            Float32(value_1).ir_value(loc=loc, ip=ip),
            Float32(value_2).ir_value(loc=loc, ip=ip),
            Float32(value_3).ir_value(loc=loc, ip=ip),
            Int32(predicate).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .v4 .f32 abcd;\n\t"
        "setp.ne.b32 p, $5, 0;\n\t"
        "mov.f32 abcd.x, $1;\n\t"
        "mov.f32 abcd.y, $2;\n\t"
        "mov.f32 abcd.z, $3;\n\t"
        "mov.f32 abcd.w, $4;\n\t"
        "@p red.global.add.v4.f32 [$0], abcd;\n\t"
        "}\n",
        "l,f,f,f,f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class SparseKlLossBackwardSm100:
    """Persistent true-varlen sparse KL backward for SM100 GPUs."""

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        acc_dtype: type[cutlass.Numeric],
        deterministic: bool = False,
    ) -> None:
        assert dtype == cutlass.BFloat16
        assert acc_dtype == Float32
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        self.deterministic = deterministic
        self.dki_smem_dtype = Float32

        self.teacher_heads = 64
        self.index_heads = 4
        self.teacher_heads_per_index = 16
        self.head_dim = 128
        self.block_size = 128
        self.q_per_macro = 16
        self.q_per_teacher_tile = 8

        self.q_stage = 2
        self.metadata_stage = 4
        self.dqi_stage = 2
        self.score_stage = 1
        self.ds_stage = 2
        self.k_stage = 1
        self.ki_stage = 1
        self.dki_stage = 1
        self.store_stage = 1
        self.clc_stage = 1

        self.q_halves = 2
        self.q_k_subtiles = 2
        self.q_subtiles_per_half = (
            self.q_per_teacher_tile * self.q_k_subtiles
        )
        self.q_subtiles_per_stage = self.q_halves * self.q_subtiles_per_half
        self.q_subtiles_total = self.q_stage * self.q_subtiles_per_stage
        self.qi_gather_rows = 1
        self.qi_gather_cols = self.head_dim // 2
        self.qi_k_subtiles = self.head_dim // self.qi_gather_cols
        self.qi_rows_per_subtile = 8
        self.qi_rows_per_gather = 4
        self.qi_gathers_per_subtile = (
            self.qi_rows_per_subtile // self.qi_rows_per_gather
        )
        self.qi_q_groups = self.q_per_macro // self.qi_rows_per_subtile
        self.qi_gathers_per_k_subtile = (
            self.qi_q_groups * self.qi_gathers_per_subtile
        )
        self.qi_gathers_per_stage = (
            self.qi_k_subtiles * self.qi_gathers_per_k_subtile
        )
        self.qi_subtiles_per_stage = self.qi_q_groups * self.qi_k_subtiles
        self.qi_subtiles_total = self.q_stage * self.qi_subtiles_per_stage

        self.teacher_tiler = (128, 128, 128)
        self.student_tiler = (128, 16, 128)
        self.dqi_tiler = (128, 16, 128)
        self.dki_tiler = (128, 64, 16)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mnk = (1, 1, 1)

        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.metadata_warp_id = 14
        self.clc_scheduler_warp_id = 15
        self.threads_per_cta = 16 * cute.arch.WARP_SIZE
        self.num_regs_reduce = 136
        self.num_regs_compute = 144
        self.num_regs_mma = 88
        self.num_regs_load = 88
        self.num_regs_metadata = 40
        self.num_regs_scheduler = 32
        self.compute_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=12,
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )
        self.reduce_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=13,
            num_threads=len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )

        self.buffer_align_bytes = 1024
        self.tmem_alloc_cols = 512
        self.tmem_teacher_offset = 0
        self.tmem_student_offset = 256
        self.tmem_ds_offset = 272
        self.tmem_dqi_offset = 288
        self.tmem_dki_offset = 320
        assert self.tmem_dki_offset + 2 * 64 <= self.tmem_alloc_cols

    def _setup_mma(self) -> None:
        self.tiled_mma_teacher = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.teacher_tiler[:2],
        )
        self.tiled_mma_student = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.student_tiler[:2],
        )
        self.tiled_mma_dqi = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            OperandMajorMode.MN,
            OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.dqi_tiler[:2],
        )
        self.tiled_mma_dki = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.dki_tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        self.tiled_mma_teacher_chunk = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            (128, 16),
        )
        self.tiled_mma_student_chunk = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            (128, 8),
        )
        self.tiled_mma_dki_chunk = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            (128, 32),
        )

    def _setup_smem_layouts(self) -> None:
        self.sK_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma_teacher,
            self.teacher_tiler,
            self.dtype,
            self.k_stage,
        )
        self.sQ_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma_teacher,
            self.teacher_tiler,
            self.dtype,
            self.q_stage * self.q_halves,
        )
        self.sQ_load_layout = sm100_utils.make_smem_layout(
            OperandMajorMode.K,
            (self.teacher_heads_per_index, 64),
            self.dtype,
            self.q_subtiles_total,
        )
        self.sKI_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma_student,
            self.student_tiler,
            self.dtype,
            self.ki_stage,
        )
        self.sQI_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma_student,
            self.student_tiler,
            self.dtype,
            self.q_stage,
        )
        self.sQI_load_layout = sm100_utils.make_smem_layout(
            OperandMajorMode.K,
            (self.qi_rows_per_subtile, 64),
            self.dtype,
            self.qi_subtiles_total,
        )
        self.sKIt_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma_dqi,
            self.dqi_tiler,
            self.dtype,
            self.ki_stage,
        )
        self.sdS_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma_dqi,
            self.dqi_tiler,
            self.dtype,
            self.ds_stage,
        )
        self.tdS_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma_dki,
            self.dki_tiler,
            self.dtype,
            self.ds_stage,
        )
        sQIt_base_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma_dki,
            self.dki_tiler,
            self.dtype,
            1,
        )
        self.sQIt_layout = cute.make_composed_layout(
            sQIt_base_layout.inner,
            0,
            cute.append(
                cute.select(sQIt_base_layout.outer, mode=[0, 1, 2]),
                cute.make_layout(
                    (self.q_stage,),
                    stride=(self.q_per_macro * self.head_dim,),
                ),
            ),
        )
        self.sK_sparse_layout = cute.make_composed_layout(
            self.sK_layout.inner,
            0,
            cute.make_layout(
                (
                    self.block_size,
                    (64, self.head_dim // 64),
                    self.k_stage,
                ),
                stride=(
                    64,
                    (1, self.block_size * 64),
                    self.block_size * self.head_dim,
                ),
            ),
        )
        self.sKI_sparse_layout = cute.make_composed_layout(
            self.sKI_layout.inner,
            0,
            cute.make_layout(
                (
                    self.block_size,
                    (64, self.head_dim // 64),
                    self.ki_stage,
                ),
                stride=(
                    64,
                    (1, self.block_size * 64),
                    self.block_size * self.head_dim,
                ),
            ),
        )
        self.sdKI_layout = cute.make_layout((128, 128), stride=(128, 1))

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mTeacherLse: cute.Tensor,
        mQI: cute.Tensor,
        mKI: cute.Tensor,
        mIndexerLse: cute.Tensor,
        mPhysicalQIndices: cute.Tensor,
        mPhysicalValidRows: cute.Tensor,
        mPhysicalRowPtr: cute.Tensor,
        mWorklist: cute.Tensor,
        mWorkCount: cute.Tensor,
        mDkiOwnerCounts: cute.Tensor,
        mDkiWriterRank: Optional[cute.Tensor],
        mDqiSemaphore: Optional[cute.Tensor],
        mDkiSemaphore: Optional[cute.Tensor],
        mDQI: cute.Tensor,
        mDKIAccum: cute.Tensor,
        softmax_scale: Float32,
        indexer_softmax_scale: Float32,
        grad_scale: Float32,
        stream: cuda.CUstream,
    ) -> None:
        if const_expr(
            not (
                mQ.element_type
                == mK.element_type
                == mQI.element_type
                == mKI.element_type
                == self.dtype
            )
        ):
            raise TypeError("Q, K, QI, and KI must be BF16")
        if const_expr(
            not (
                mTeacherLse.element_type
                == mIndexerLse.element_type
                == mDQI.element_type
                == Float32
            )
        ):
            raise TypeError("LSE and dQI accumulator tensors must be FP32")
        if const_expr(mDKIAccum.element_type != Float32):
            raise TypeError("raw dKI accumulator must be FP32")
        if const_expr(
            not (
                mPhysicalQIndices.element_type
                == mPhysicalValidRows.element_type
                == mPhysicalRowPtr.element_type
                == mDkiOwnerCounts.element_type
                == Int32
            )
        ):
            raise TypeError("Sparse metadata must be INT32")
        if const_expr(mWorklist.element_type != Int32):
            raise TypeError("worklist must be INT32")
        if const_expr(mWorkCount.element_type != Int32):
            raise TypeError("work count must be INT32")
        if const_expr(self.deterministic):
            if const_expr(
                mDkiWriterRank is None
                or mDqiSemaphore is None
                or mDkiSemaphore is None
            ):
                raise ValueError(
                    "deterministic KL backward requires writer-rank and semaphore tensors"
                )
            if const_expr(
                mDkiWriterRank.element_type != Int32
                or mDqiSemaphore.element_type != Int32
                or mDkiSemaphore.element_type != Int32
            ):
                raise TypeError(
                    "deterministic KL writer-rank and semaphore tensors must be INT32"
                )

        mDQI, mDKIAccum = [
            assume_tensor_aligned(t) for t in (mDQI, mDKIAccum)
        ]
        self._setup_mma()
        self._setup_smem_layouts()

        mQ_flat = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (mQ.shape[0] * self.teacher_heads, self.head_dim),
                stride=(self.head_dim, 1),
            ),
        )
        mQI_flat = cute.make_tensor(
            mQI.iterator,
            cute.make_layout(
                (mQI.shape[0] * self.index_heads, self.head_dim),
                stride=(self.head_dim, 1),
            ),
        )
        mK_tma = layout_utils.select(mK, mode=[0, 2, 1])
        mKI_tma = layout_utils.select(mKI, mode=[0, 2, 1])
        mDKIAccum_2d = cute.make_tensor(
            mDKIAccum.iterator,
            cute.make_layout(
                (mDKIAccum.shape[0], self.head_dim),
                stride=(self.head_dim, 1),
            ),
        )
        # Single-owner tiles use the first BF16 half of each padded FP32 tile.
        # Keeping the same padded block stride prevents packed sequence overlap.
        mDKIWorkspace_2d = cute.make_tensor(
            cute.recast_ptr(mDKIAccum.iterator, dtype=self.dtype),
            cute.make_layout(
                (mDKIAccum.shape[0], self.head_dim * 2),
                stride=(self.head_dim * 2, 1),
            ),
        )
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            tma_load_op,
            mK_tma,
            self.sK_sparse_layout,
            (self.block_size, self.head_dim),
        )
        tma_atom_KI, tma_tensor_KI = cpasync.make_tiled_tma_atom(
            tma_load_op,
            mKI_tma,
            self.sKI_sparse_layout,
            (self.block_size, self.head_dim),
        )
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ_flat,
            cute.select(self.sQ_load_layout, mode=[0, 1]),
            (self.teacher_heads_per_index, 64),
        )
        sQI_gather4_layout = cute.make_composed_layout(
            self.sQI_load_layout.inner,
            0,
            cute.make_layout(
                (self.qi_gather_rows, self.qi_gather_cols),
                stride=(self.qi_gather_cols, 1),
            ),
        )
        tma_atom_QI, _ = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQI_flat,
            sQI_gather4_layout,
            (self.qi_gather_rows, self.qi_gather_cols),
        )
        tma_store_dki_op = cpasync.CopyReduceBulkTensorTileS2GOp()
        tma_atom_DKIAccum, tma_tensor_DKIAccum = cpasync.make_tiled_tma_atom(
            tma_store_dki_op,
            mDKIAccum_2d,
            self.sdKI_layout,
            (self.block_size, self.head_dim),
        )
        tma_atom_DKIWorkspace, tma_tensor_DKIWorkspace = (
            cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mDKIWorkspace_2d,
                self.sdKI_layout,
                (self.block_size, self.head_dim),
            )
        )
        self.tma_copy_bytes_q = (
            self.q_halves
            * self.q_per_teacher_tile
            * self.teacher_heads_per_index
            * self.head_dim
            * self.dtype.width
            // 8
        )
        self.tma_copy_bytes_qi = (
            self.q_per_macro * self.head_dim * self.dtype.width // 8
        )
        self.tma_copy_bytes_k = self.block_size * self.head_dim * self.dtype.width // 8
        self.tma_copy_bytes_ki = self.tma_copy_bytes_k

        self.sQ_storage_elems = max(
            cute.cosize(self.sQ_layout), cute.cosize(self.sQ_load_layout)
        )
        self.sK_storage_elems = max(
            cute.cosize(self.sK_layout), cute.cosize(self.sK_sparse_layout)
        )
        self.sQI_storage_elems = max(
            cute.cosize(self.sQI_layout),
            cute.cosize(self.sQI_load_layout),
            cute.cosize(self.sQIt_layout),
        )
        self.sKI_storage_elems = max(
            cute.cosize(self.sKI_layout),
            cute.cosize(self.sKIt_layout),
            cute.cosize(self.sKI_sparse_layout),
        )
        self.sdS_storage_elems = max(
            cute.cosize(self.sdS_layout), cute.cosize(self.tdS_layout)
        )

        @cute.struct
        class SharedStorage:
            Q_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.q_stage]
            K_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.k_stage]
            QI_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.q_stage]
            KI_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.ki_stage]
            score_lo_mbar_ptr: cute.struct.MemRange[
                Int64, 2 * self.score_stage
            ]
            score_hi_mbar_ptr: cute.struct.MemRange[
                Int64, 2 * self.score_stage
            ]
            dS_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.ds_stage]
            dQI_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.dqi_stage]
            dKI_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.dki_stage]
            metadata_mbar_ptr: cute.struct.MemRange[
                Int64, 2 * self.metadata_stage
            ]
            store_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.store_stage]
            clc_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.clc_stage]
            clc_response: cute.struct.MemRange[Int32, 4 * self.clc_stage]
            work_mbar_ptr: cute.struct.MemRange[Int64, 2]
            work_response: cute.struct.MemRange[Int32, 1]
            tmem_holding_buf: Int32
            tmem_dealloc_mbar_ptr: Int64
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.dtype, self.sQ_storage_elems],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.dtype, self.sK_storage_elems],
                self.buffer_align_bytes,
            ]
            sQI: cute.struct.Align[
                cute.struct.MemRange[self.dtype, self.sQI_storage_elems],
                self.buffer_align_bytes,
            ]
            sKI: cute.struct.Align[
                cute.struct.MemRange[self.dtype, self.sKI_storage_elems],
                self.buffer_align_bytes,
            ]
            sdS: cute.struct.Align[
                cute.struct.MemRange[self.dtype, self.sdS_storage_elems], 128
            ]
            sTeacherLse: cute.struct.Align[
                cute.struct.MemRange[
                    Float32,
                    self.q_per_macro
                    * self.teacher_heads_per_index
                    * self.metadata_stage,
                ],
                128,
            ]
            sIndexerLse: cute.struct.Align[
                cute.struct.MemRange[
                    Float32, self.q_per_macro * self.metadata_stage
                ],
                64,
            ]
            sQIdx: cute.struct.Align[
                cute.struct.MemRange[
                    Int32, self.q_per_macro * self.metadata_stage
                ],
                64,
            ]
            sValidRows: cute.struct.Align[
                cute.struct.MemRange[
                    Int32, self.q_per_macro * self.metadata_stage
                ],
                64,
            ]
            sWriterRank: cute.struct.Align[
                cute.struct.MemRange[
                    Int32, self.q_per_macro * self.metadata_stage
                ],
                64,
            ]
            sWorkMeta: cute.struct.Align[
                cute.struct.MemRange[Int32, 3 * self.metadata_stage], 32
            ]

        self.shared_storage = SharedStorage
        assert SharedStorage.size_in_bytes() <= 232448

        total_rows = mWorklist.shape[0]
        # The host-known capacity defines the asynchronous launch domain. CLC
        # dynamically cancels pending CTAs, while the device work count rejects
        # the first response outside the compact worklist before any new work
        # enters the data pipeline.
        grid_dim = (total_rows, 1, 1)
        log2_e = Float32(math.log2(math.e))

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            mTeacherLse,
            tma_tensor_KI,
            mIndexerLse,
            mPhysicalQIndices,
            mPhysicalValidRows,
            mPhysicalRowPtr,
            mWorklist,
            mWorkCount,
            mDkiOwnerCounts,
            mDkiWriterRank,
            mDqiSemaphore,
            mDkiSemaphore,
            mDQI,
            tma_tensor_DKIAccum,
            tma_tensor_DKIWorkspace,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_QI,
            tma_atom_KI,
            tma_atom_DKIAccum,
            tma_atom_DKIWorkspace,
            self.sQ_layout,
            self.sQ_load_layout,
            self.sK_layout,
            self.sK_sparse_layout,
            self.sQI_layout,
            self.sQI_load_layout,
            self.sQIt_layout,
            self.sKI_layout,
            self.sKI_sparse_layout,
            self.sKIt_layout,
            self.sdS_layout,
            self.tdS_layout,
            self.sdKI_layout,
            self.tiled_mma_teacher,
            self.tiled_mma_student,
            self.tiled_mma_dqi,
            self.tiled_mma_dki,
            self.tiled_mma_teacher_chunk,
            self.tiled_mma_student_chunk,
            self.tiled_mma_dki_chunk,
            SharedStorage,
            softmax_scale * log2_e,
            indexer_softmax_scale * log2_e,
            log2_e,
            grad_scale,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mTeacherLse: cute.Tensor,
        mKI: cute.Tensor,
        mIndexerLse: cute.Tensor,
        mPhysicalQIndices: cute.Tensor,
        mPhysicalValidRows: cute.Tensor,
        mPhysicalRowPtr: cute.Tensor,
        mWorklist: cute.Tensor,
        mWorkCount: cute.Tensor,
        mDkiOwnerCounts: cute.Tensor,
        mDkiWriterRank: Optional[cute.Tensor],
        mDqiSemaphore: Optional[cute.Tensor],
        mDkiSemaphore: Optional[cute.Tensor],
        mDQI: cute.Tensor,
        mDKIAccum: cute.Tensor,
        mDKIWorkspace: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_QI: cute.CopyAtom,
        tma_atom_KI: cute.CopyAtom,
        tma_atom_DKIAccum: cute.CopyAtom,
        tma_atom_DKIWorkspace: cute.CopyAtom,
        sQ_layout: cute.ComposedLayout,
        sQ_load_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sK_sparse_layout: cute.ComposedLayout,
        sQI_layout: cute.ComposedLayout,
        sQI_load_layout: cute.ComposedLayout,
        sQIt_layout: cute.ComposedLayout,
        sKI_layout: cute.ComposedLayout,
        sKI_sparse_layout: cute.ComposedLayout,
        sKIt_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        sdKI_layout: cute.Layout,
        tiled_mma_teacher: cute.TiledMma,
        tiled_mma_student: cute.TiledMma,
        tiled_mma_dqi: cute.TiledMma,
        tiled_mma_dki: cute.TiledMma,
        tiled_mma_teacher_chunk: cute.TiledMma,
        tiled_mma_student_chunk: cute.TiledMma,
        tiled_mma_dki_chunk: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        softmax_scale_log2: Float32,
        indexer_softmax_scale_log2: Float32,
        log2_e: Float32,
        grad_scale: Float32,
    ) -> None:
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        exit_thread_if(
            Int32(cute.arch.block_idx()[0] >= mWorkCount[Int32(0)])
        )
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=14,
            num_threads=(
                len(self.compute_warp_ids) + len(self.reduce_warp_ids) + 1
            )
            * cute.arch.WARP_SIZE,
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=False,
        )

        producer_tma = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, 1
        )
        consumer_mma = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, 1
        )
        producer_metadata = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, cute.arch.WARP_SIZE
        )
        consumer_reduce_threads = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )
        consumer_compute_warps = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids)
        )
        consumer_compute_half_warps = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids) // 2
        )
        producer_compute_half_threads = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) // 2 * cute.arch.WARP_SIZE,
        )
        consumer_load_threads = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, cute.arch.WARP_SIZE
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_teacher.thr_id.shape,),
        )

        clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()
        work_mbar_ptr = storage.work_mbar_ptr.data_ptr()
        if warp_idx == Int32(0):
            with cute.arch.elect_one():
                # Only the scheduler warp consumes raw CLC responses. It
                # filters capacity-tail responses before broadcasting work to
                # the remaining specialized warps.
                cute.arch.mbarrier_init(clc_mbar_ptr, Int32(1))
                cute.arch.mbarrier_init(
                    clc_mbar_ptr + Int32(self.clc_stage),
                    Int32(1),
                )
                cute.arch.mbarrier_init(work_mbar_ptr, Int32(1))
                cute.arch.mbarrier_init(
                    work_mbar_ptr + Int32(1),
                    Int32(
                        self.threads_per_cta // cute.arch.WARP_SIZE - 1
                    ),
                )

        pipeline_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.q_stage,
            producer_group=producer_tma,
            consumer_group=consumer_mma,
            tx_count=self.tma_copy_bytes_q,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_K = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.K_mbar_ptr.data_ptr(),
            num_stages=self.k_stage,
            producer_group=producer_tma,
            consumer_group=consumer_mma,
            tx_count=self.tma_copy_bytes_k,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_QI = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.QI_mbar_ptr.data_ptr(),
            num_stages=self.q_stage,
            producer_group=producer_tma,
            consumer_group=consumer_mma,
            tx_count=self.tma_copy_bytes_qi,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_KI = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.KI_mbar_ptr.data_ptr(),
            num_stages=self.ki_stage,
            producer_group=producer_tma,
            consumer_group=consumer_mma,
            tx_count=self.tma_copy_bytes_ki,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_score_lo = cutlass.pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.score_lo_mbar_ptr.data_ptr(),
            num_stages=self.score_stage,
            producer_group=consumer_mma,
            consumer_group=consumer_compute_half_warps,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_score_hi = cutlass.pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.score_hi_mbar_ptr.data_ptr(),
            num_stages=self.score_stage,
            producer_group=consumer_mma,
            consumer_group=consumer_compute_half_warps,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        score_lo_mbar_ptr = storage.score_lo_mbar_ptr.data_ptr()
        pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.dS_mbar_ptr.data_ptr(),
            num_stages=self.ds_stage,
            producer_group=consumer_compute_warps,
            consumer_group=consumer_mma,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dQI = cutlass.pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.dQI_mbar_ptr.data_ptr(),
            num_stages=self.dqi_stage,
            producer_group=consumer_mma,
            consumer_group=cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, len(self.reduce_warp_ids)
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dKI = cutlass.pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.dKI_mbar_ptr.data_ptr(),
            num_stages=self.dki_stage,
            producer_group=consumer_mma,
            consumer_group=consumer_compute_half_warps,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_metadata = pipeline.PipelineAsync.create(
            barrier_storage=storage.metadata_mbar_ptr.data_ptr(),
            num_stages=self.metadata_stage,
            producer_group=producer_metadata,
            consumer_group=consumer_reduce_threads,
        )
        pipeline_store = pipeline.PipelineAsync.create(
            barrier_storage=storage.store_mbar_ptr.data_ptr(),
            num_stages=self.store_stage,
            producer_group=producer_compute_half_threads,
            consumer_group=consumer_load_threads,
        )
        pipeline_init_arrive(
            cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True
        )
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
        if const_expr(self.deterministic):
            tile_scheduler = _SparseKlSingleWorkScheduler.create(mWorkCount)
        else:
            clc = _SparseKlClcState.create(
                mbar_ptr=clc_mbar_ptr,
                response_ptr=storage.clc_response.data_ptr(),
                empty_mbar_offset=self.clc_stage,
                response_bytes=4 * self.clc_stage * (Int32.width // 8),
            )
            work = _SparseKlWorkState.create(
                mbar_ptr=work_mbar_ptr,
                response_ptr=storage.work_response.data_ptr(),
                empty_mbar_offset=1,
            )
            tile_scheduler = _SparseKlClcScheduler.create(
                clc,
                work,
                mWorkCount,
            )
        sQ = storage.sQ.get_tensor(
            sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.dtype
        )
        sQ_load = storage.sQ.get_tensor(
            sQ_load_layout.outer,
            swizzle=sQ_load_layout.inner,
            dtype=self.dtype,
        )
        sK = storage.sK.get_tensor(
            sK_layout.outer, swizzle=sK_layout.inner, dtype=self.dtype
        )
        sK_sparse = cute.make_tensor(sK.iterator, sK_sparse_layout.outer)
        sQI = storage.sQI.get_tensor(
            sQI_layout.outer, swizzle=sQI_layout.inner, dtype=self.dtype
        )
        sQI_load = storage.sQI.get_tensor(
            sQI_load_layout.outer,
            swizzle=sQI_load_layout.inner,
            dtype=self.dtype,
        )
        sQIt = cute.make_tensor(
            cute.recast_ptr(
                storage.sQI.data_ptr(), sQIt_layout.inner, dtype=self.dtype
            ),
            sQIt_layout.outer,
        )
        sQItHi = cute.make_tensor(
            cute.recast_ptr(
                storage.sQI.data_ptr() + self.q_per_macro * 64,
                sQIt_layout.inner,
                dtype=self.dtype,
            ),
            sQIt_layout.outer,
        )
        sKI = storage.sKI.get_tensor(
            sKI_layout.outer, swizzle=sKI_layout.inner, dtype=self.dtype
        )
        sKI_sparse = cute.make_tensor(sKI.iterator, sKI_sparse_layout.outer)
        sKIt = cute.make_tensor(
            cute.recast_ptr(
                storage.sKI.data_ptr(), sKIt_layout.inner, dtype=self.dtype
            ),
            sKIt_layout.outer,
        )
        sdS = storage.sdS.get_tensor(
            sdS_layout.outer, swizzle=sdS_layout.inner, dtype=self.dtype
        )
        sdS_logical = cute.composition(
            sdS,
            cute.make_ordered_layout(
                (self.q_per_macro, self.block_size, self.ds_stage),
                order=(0, 1, 2),
            ),
        )
        tdS = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    Float32,
                    self.tmem_ds_offset,
                    mem_space=cute.AddressSpace.tmem,
                    assumed_align=16,
                ),
                dtype=self.dtype,
            ),
            tdS_layout.outer,
        )
        sdKIAccum = cute.make_tensor(
            cute.recast_ptr(storage.sQ.data_ptr(), dtype=self.dki_smem_dtype),
            sdKI_layout,
        )
        sdKI = cute.make_tensor(
            cute.recast_ptr(storage.sQ.data_ptr(), dtype=self.dtype),
            sdKI_layout,
        )
        sTeacherLse = storage.sTeacherLse.get_tensor(
            cute.make_layout(
                (
                    self.q_per_macro * self.teacher_heads_per_index,
                    self.metadata_stage,
                )
            )
        )
        sIndexerLse = storage.sIndexerLse.get_tensor(
            cute.make_layout((self.q_per_macro, self.metadata_stage))
        )
        sQIdx = storage.sQIdx.get_tensor(
            cute.make_layout((self.q_per_macro, self.metadata_stage))
        )
        sValidRows = storage.sValidRows.get_tensor(
            cute.make_layout((self.q_per_macro, self.metadata_stage))
        )
        sWriterRank = storage.sWriterRank.get_tensor(
            cute.make_layout((self.q_per_macro, self.metadata_stage))
        )
        sWorkMeta = storage.sWorkMeta.get_tensor(
            cute.make_layout((3, self.metadata_stage))
        )

        tmem_ptr = cute.make_ptr(
            Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16
        )
        thr_teacher = tiled_mma_teacher.get_slice(0)
        teacher_shape = thr_teacher.partition_shape_C(self.teacher_tiler[:2])
        tTeacher = thr_teacher.make_fragment_C(cute.append(teacher_shape, 2))
        tTeacher = cute.make_tensor(
            tmem_ptr + self.tmem_teacher_offset, tTeacher.layout
        )
        thr_student = tiled_mma_student.get_slice(0)
        student_shape = thr_student.partition_shape_C(self.student_tiler[:2])
        tStudent = thr_student.make_fragment_C(student_shape)
        tStudent = cute.make_tensor(
            tmem_ptr + self.tmem_student_offset, tStudent.layout
        )
        thr_dqi = tiled_mma_dqi.get_slice(0)
        dqi_shape = thr_dqi.partition_shape_C(self.dqi_tiler[:2])
        tdQI = thr_dqi.make_fragment_C(cute.append(dqi_shape, self.dqi_stage))
        tdQI = cute.make_tensor(tmem_ptr + self.tmem_dqi_offset, tdQI.layout)
        thr_dki = tiled_mma_dki.get_slice(0)
        dki_shape = thr_dki.partition_shape_C(self.dki_tiler[:2])
        tdKI = thr_dki.make_fragment_C(cute.append(dki_shape, 2))
        tdKI = cute.make_tensor(tmem_ptr + self.tmem_dki_offset, tdKI.layout)

        if warp_idx == self.clc_scheduler_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_scheduler)
            if const_expr(not self.deterministic):
                self.schedule(tile_scheduler)

        if warp_idx == self.metadata_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_metadata)
            self.produce_metadata(
                mTeacherLse,
                mIndexerLse,
                mPhysicalQIndices,
                mPhysicalValidRows,
                mPhysicalRowPtr,
                mWorklist,
                sTeacherLse,
                sIndexerLse,
                sQIdx,
                sValidRows,
                sWriterRank,
                sWorkMeta,
                pipeline_metadata,
                tile_scheduler,
                log2_e,
            )

        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load_and_store(
                mQ,
                mK,
                mKI,
                mDKIAccum,
                mDKIWorkspace,
                mDkiOwnerCounts,
                mDkiWriterRank,
                mDkiSemaphore,
                mWorklist,
                sQ_load,
                sK_sparse,
                sQI_load,
                sKI_sparse,
                sdKIAccum,
                sdKI,
                sQIdx,
                sWorkMeta,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_QI,
                tma_atom_KI,
                tma_atom_DKIAccum,
                tma_atom_DKIWorkspace,
                pipeline_Q,
                pipeline_K,
                pipeline_QI,
                pipeline_KI,
                pipeline_metadata,
                pipeline_store,
                tile_scheduler,
            )

        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            self.mma(
                tiled_mma_teacher,
                tiled_mma_student,
                tiled_mma_dqi,
                tiled_mma_dki,
                sQ,
                sK,
                sQI,
                sQIt,
                sQItHi,
                sKI,
                sKIt,
                sdS,
                tdS,
                tTeacher,
                tStudent,
                tdQI,
                tdKI,
                mPhysicalRowPtr,
                mWorklist,
                pipeline_Q,
                pipeline_K,
                pipeline_QI,
                pipeline_KI,
                pipeline_score_lo,
                pipeline_score_hi,
                pipeline_dS,
                pipeline_dQI,
                pipeline_dKI,
                pipeline_metadata,
                tile_scheduler,
            )
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            allocated_tmem_ptr = tmem.retrieve_ptr(Float32)
            cute.arch.dealloc_tmem(
                allocated_tmem_ptr, self.tmem_alloc_cols, is_two_cta=False
            )

        if (
            warp_idx >= self.compute_warp_ids[0]
            and warp_idx <= self.compute_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem.wait_for_alloc()
            self.compute_loop(
                tiled_mma_student,
                tiled_mma_teacher_chunk,
                tiled_mma_student_chunk,
                tiled_mma_dki_chunk,
                tmem_ptr,
                sdS_logical,
                sdKIAccum,
                sdKI,
                sTeacherLse,
                sIndexerLse,
                sValidRows,
                mWorklist,
                mDkiOwnerCounts,
                score_lo_mbar_ptr,
                pipeline_dS,
                pipeline_dKI,
                pipeline_metadata,
                pipeline_store,
                tile_scheduler,
                softmax_scale_log2,
                indexer_softmax_scale_log2,
                grad_scale,
            )
            tmem_alloc_barrier.arrive()

        if (
            warp_idx >= self.reduce_warp_ids[0]
            and warp_idx <= self.reduce_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            tmem.wait_for_alloc()
            self.dqi_acc_reduce(
                tiled_mma_dqi,
                tdQI,
                mDQI,
                mDqiSemaphore,
                mWorklist,
                sQIdx,
                sWriterRank,
                sWorkMeta,
                pipeline_dQI,
                pipeline_metadata,
                tile_scheduler,
            )
            tmem_alloc_barrier.arrive()

    @cute.jit
    def schedule(
        self,
        tile_scheduler,
    ) -> None:
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            tile_scheduler.prefetch_next_work(work_tile)
            work_tile = tile_scheduler.scheduler_advance_to_next_work(
                work_tile
            )
        tile_scheduler.producer_tail()

    @cute.jit
    def _work_physical_block(
        self,
        mWorklist: cute.Tensor,
        work_idx: Int32,
    ) -> Int32:
        return mWorklist[work_idx, Int32(0)]

    @cute.jit
    def _macro_info(
        self,
        mPhysicalRowPtr: cute.Tensor,
        mWorklist: cute.Tensor,
        work_idx: Int32,
        macro_idx: Int32,
    ) -> tuple[Int32, Int32, Int32]:
        physical_block = self._work_physical_block(mWorklist, work_idx)
        remaining = mWorklist[work_idx, Int32(2)] + macro_idx
        head = Int32(0)
        row_start = Int32(0)
        q_count = Int32(0)
        found = Boolean(False)
        for head_idx in cutlass.range_constexpr(self.index_heads):
            begin = mPhysicalRowPtr[head_idx, physical_block]
            end = mPhysicalRowPtr[head_idx, physical_block + Int32(1)]
            row_count = end - begin
            head_macros = cute.ceil_div(row_count, self.q_per_macro)
            if not found:
                if remaining < head_macros:
                    head = Int32(head_idx)
                    row_start = begin + remaining * Int32(self.q_per_macro)
                    q_count = cutlass.min(
                        Int32(self.q_per_macro),
                        end - row_start,
                    )
                    found = Boolean(True)
                else:
                    remaining -= head_macros
        return head, row_start, q_count

    @cute.jit
    def _work_has_q(
        self,
        mWorklist: cute.Tensor,
        work_idx: Int32,
    ) -> Boolean:
        return mWorklist[work_idx, Int32(3)] > Int32(0)

    @cute.jit
    def produce_metadata(
        self,
        mTeacherLse: cute.Tensor,
        mIndexerLse: cute.Tensor,
        mPhysicalQIndices: cute.Tensor,
        mPhysicalValidRows: cute.Tensor,
        mPhysicalRowPtr: cute.Tensor,
        mWorklist: cute.Tensor,
        sTeacherLse: cute.Tensor,
        sIndexerLse: cute.Tensor,
        sQIdx: cute.Tensor,
        sValidRows: cute.Tensor,
        sWriterRank: cute.Tensor,
        sWorkMeta: cute.Tensor,
        pipeline_metadata,
        tile_scheduler,
        log2_e: Float32,
    ) -> None:
        lane = cute.arch.thread_idx()[0] % cute.arch.WARP_SIZE
        producer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.metadata_stage
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_idx = work_tile.tile_idx[0]
            k_offset = mWorklist[work_idx, Int32(1)]
            num_macros = mWorklist[work_idx, Int32(3)]
            for macro in cutlass.range(num_macros, unroll=1):
                head, row_start, q_count = self._macro_info(
                    mPhysicalRowPtr,
                    mWorklist,
                    work_idx,
                    macro,
                )
                pipeline_metadata.producer_acquire(producer_state)
                stage = producer_state.index
                if lane < Int32(self.q_per_macro):
                    q_slot = lane
                    valid_q = q_slot < q_count
                    q_global = Int32(0)
                    packed_valid_rows = Int32(0)
                    if valid_q:
                        q_global = mPhysicalQIndices[
                            head, row_start + q_slot
                        ]
                        packed_valid_rows = mPhysicalValidRows[
                            head, row_start + q_slot
                        ]
                    sQIdx[q_slot, stage] = q_global
                    sValidRows[q_slot, stage] = (
                        packed_valid_rows & Int32(KL_VALID_ROWS_MASK)
                    )
                    sWriterRank[q_slot, stage] = (
                        packed_valid_rows >> Int32(KL_WRITER_RANK_SHIFT)
                    ) & Int32(KL_WRITER_RANK_MASK)
                    sIndexerLse[q_slot, stage] = (
                        mIndexerLse[head, q_global] * log2_e
                        if valid_q
                        else Float32(0.0)
                    )
                # Publish lane-owned metadata before cross-lane shared reads.
                cute.arch.sync_warp()
                for pass_i in cutlass.range_constexpr(2):
                    q_slot = (
                        Int32(pass_i * 8) + lane // Int32(4)
                    )
                    teacher_chunk = lane % Int32(4)
                    teacher_in_head = teacher_chunk * Int32(4)
                    valid_q = q_slot < q_count
                    q_global = sQIdx[q_slot, stage]
                    teacher_head = (
                        head * Int32(self.teacher_heads_per_index)
                        + teacher_in_head
                    )
                    teacher_lse_ptr = cute.make_ptr(
                        Float32,
                        mTeacherLse.iterator.toint()
                        + Int64(
                            q_global * Int32(self.teacher_heads)
                            + teacher_head
                        )
                        * Int64(4),
                        mem_space=mTeacherLse.iterator.memspace,
                        assumed_align=16,
                    )
                    gTeacherLse = cute.make_tensor(
                        teacher_lse_ptr, cute.make_layout((4,), stride=(1,))
                    )
                    tTeacherLse = cute.make_rmem_tensor((4,), Float32)
                    if valid_q:
                        tTeacherLse.store(gTeacherLse.load() * log2_e)
                    else:
                        tTeacherLse.fill(Float32(0.0))
                    linear = (
                        q_slot * Int32(self.teacher_heads_per_index)
                        + teacher_in_head
                    )
                    teacher_lse_smem_offset = cute.crd2idx(
                        (linear, stage), sTeacherLse.layout
                    )
                    sTeacherLseVec = cute.make_tensor(
                        sTeacherLse.iterator + teacher_lse_smem_offset,
                        cute.make_layout((4,), stride=(1,)),
                    )
                    cute.autovec_copy(tTeacherLse, sTeacherLseVec)
                if lane == Int32(0):
                    sWorkMeta[0, stage] = k_offset
                    sWorkMeta[1, stage] = head
                    sWorkMeta[2, stage] = q_count
                cute.arch.fence_view_async_shared()
                pipeline_metadata.producer_commit(producer_state)
                producer_state.advance()
            work_tile = tile_scheduler.advance_to_next_work(work_tile)
        pipeline_metadata.producer_tail(producer_state)

    @cute.jit
    def _load_qi_stage(
        self,
        qi_gather4_desc_ptr: cute.Pointer,
        sQI: cute.Tensor,
        sQIdx: cute.Tensor,
        sWorkMeta: cute.Tensor,
        metadata_stage: Int32,
        qi_stage: Int32,
        qi_mbar_ptr: cute.Pointer,
    ) -> None:
        head = sWorkMeta[1, metadata_stage]
        elem_bytes = self.dtype.width // 8
        subtile_elems = self.qi_rows_per_subtile * self.qi_gather_cols
        group_elems = self.qi_rows_per_gather * self.qi_gather_cols
        with cute.arch.elect_one():
            for gather_i in cutlass.range(self.qi_gathers_per_stage, unroll=1):
                d_half = gather_i // Int32(self.qi_gathers_per_k_subtile)
                gather_in_half = (
                    gather_i
                    - d_half * Int32(self.qi_gathers_per_k_subtile)
                )
                q_group = gather_in_half // Int32(self.qi_gathers_per_subtile)
                q4 = (
                    gather_in_half
                    - q_group * Int32(self.qi_gathers_per_subtile)
                )
                substage = (
                    qi_stage * Int32(self.qi_subtiles_per_stage)
                    + d_half * Int32(self.qi_q_groups)
                    + q_group
                )
                q_slot = (
                    q_group * Int32(self.qi_rows_per_subtile)
                    + q4 * Int32(self.qi_rows_per_gather)
                )
                row0 = (
                    sQIdx[q_slot, metadata_stage]
                    * Int32(self.index_heads)
                    + head
                )
                row1 = (
                    sQIdx[q_slot + Int32(1), metadata_stage]
                    * Int32(self.index_heads)
                    + head
                )
                row2 = (
                    sQIdx[q_slot + Int32(2), metadata_stage]
                    * Int32(self.index_heads)
                    + head
                )
                row3 = (
                    sQIdx[q_slot + Int32(3), metadata_stage]
                    * Int32(self.index_heads)
                    + head
                )
                dst_byte_offset = (
                    substage * Int32(subtile_elems)
                    + q4 * Int32(group_elems)
                ) * Int32(elem_bytes)
                tma_gather4(
                    sQI.iterator,
                    dst_byte_offset,
                    qi_gather4_desc_ptr,
                    d_half * Int32(self.qi_gather_cols),
                    row0,
                    row1,
                    row2,
                    row3,
                    qi_mbar_ptr,
                )

    @cute.jit
    def load_and_store(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mKI: cute.Tensor,
        mDKIAccum: cute.Tensor,
        mDKIWorkspace: cute.Tensor,
        mDkiOwnerCounts: cute.Tensor,
        mDkiWriterRank: Optional[cute.Tensor],
        mDkiSemaphore: Optional[cute.Tensor],
        mWorklist: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sQI: cute.Tensor,
        sKI: cute.Tensor,
        sdKIAccum: cute.Tensor,
        sdKI: cute.Tensor,
        sQIdx: cute.Tensor,
        sWorkMeta: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_QI: cute.CopyAtom,
        tma_atom_KI: cute.CopyAtom,
        tma_atom_DKIAccum: cute.CopyAtom,
        tma_atom_DKIWorkspace: cute.CopyAtom,
        pipeline_Q,
        pipeline_K,
        pipeline_QI,
        pipeline_KI,
        pipeline_metadata,
        pipeline_store,
        tile_scheduler,
    ) -> None:
        producer_q = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.q_stage
        )
        producer_k = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.k_stage
        )
        producer_qi = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.q_stage
        )
        producer_ki = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.ki_stage
        )
        consumer_meta = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.metadata_stage
        )
        consumer_store = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.store_stage
        )
        gQ_k0 = cute.local_tile(
            mQ, (self.teacher_heads_per_index, 64), (None, 0)
        )
        gQ_k1 = cute.local_tile(
            mQ, (self.teacher_heads_per_index, 64), (None, 1)
        )
        load_Q_k0, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_Q, 0, cute.make_layout(1), gQ_k0, sQ
        )
        load_Q_k1, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_Q, 0, cute.make_layout(1), gQ_k1, sQ
        )
        qi_gather4_desc_ptr = extract_tma_desc_ptr(tma_atom_QI)
        with cute.arch.elect_one():
            prefetch_tma_desc_raw(qi_gather4_desc_ptr)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_idx = work_tile.tile_idx[0]
            num_macros = mWorklist[work_idx, Int32(3)]
            previous_head = Int32(-1)
            for macro in cutlass.range(num_macros, unroll=1):
                pipeline_metadata.consumer_wait(consumer_meta)
                meta_stage = consumer_meta.index
                k_offset = sWorkMeta[0, meta_stage]
                head = sWorkMeta[1, meta_stage]
                if head != previous_head:
                    mK_cur = cute.domain_offset(
                        (k_offset, 0), mK[None, None, head]
                    )
                    gK_tiles = cute.flat_divide(
                        mK_cur, (self.block_size, self.head_dim)
                    )
                    tKsK, tKgK = cpasync.tma_partition(
                        tma_atom_K,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sK, 0, 2),
                        cute.group_modes(gK_tiles, 0, 2),
                    )
                    pipeline_K.producer_acquire(producer_k)
                    cute.copy(
                        tma_atom_K,
                        tKgK[None, Int32(0), Int32(0)],
                        tKsK[None, producer_k.index],
                        tma_bar_ptr=pipeline_K.producer_get_barrier(producer_k),
                    )
                    pipeline_K.producer_commit(producer_k)
                    producer_k.advance()
                if macro == Int32(0):
                    mKI_cur = cute.domain_offset(
                        (k_offset, 0), mKI[None, None, Int32(0)]
                    )
                    gKI_tiles = cute.flat_divide(
                        mKI_cur, (self.block_size, self.head_dim)
                    )
                    tKIsKI, tKIgKI = cpasync.tma_partition(
                        tma_atom_KI,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sKI, 0, 2),
                        cute.group_modes(gKI_tiles, 0, 2),
                    )
                    pipeline_KI.producer_acquire(producer_ki)
                    cute.copy(
                        tma_atom_KI,
                        tKIgKI[None, Int32(0), Int32(0)],
                        tKIsKI[None, producer_ki.index],
                        tma_bar_ptr=pipeline_KI.producer_get_barrier(producer_ki),
                    )
                    pipeline_KI.producer_commit(producer_ki)
                    producer_ki.advance()

                pipeline_Q.producer_acquire(producer_q)
                q_mbar = pipeline_Q.producer_get_barrier(producer_q)
                q_stage_base = producer_q.index * Int32(
                    self.q_subtiles_per_stage
                )
                for half_i in cutlass.range_constexpr(self.q_halves):
                    for q_i in cutlass.range_constexpr(
                        self.q_per_teacher_tile
                    ):
                        q_slot = Int32(
                            half_i * self.q_per_teacher_tile + q_i
                        )
                        q_global = sQIdx[q_slot, meta_stage]
                        src_tile = q_global * Int32(self.index_heads) + head
                        dst_base = q_stage_base + Int32(
                            half_i * self.q_subtiles_per_half
                        )
                        load_Q_k0(
                            src_idx=src_tile,
                            dst_idx=dst_base + Int32(q_i),
                            tma_bar_ptr=q_mbar,
                        )
                        load_Q_k1(
                            src_idx=src_tile,
                            dst_idx=(
                                dst_base
                                + Int32(self.q_per_teacher_tile + q_i)
                            ),
                            tma_bar_ptr=q_mbar,
                        )
                pipeline_Q.producer_commit(producer_q)
                producer_q.advance()

                pipeline_QI.producer_acquire(producer_qi)
                self._load_qi_stage(
                    qi_gather4_desc_ptr,
                    sQI,
                    sQIdx,
                    sWorkMeta,
                    meta_stage,
                    producer_qi.index,
                    pipeline_QI.producer_get_barrier(producer_qi),
                )
                pipeline_QI.producer_commit(producer_qi)
                producer_qi.advance()
                consumer_meta.advance()
                previous_head = head

            if self._work_has_q(mWorklist, work_idx):
                pipeline_store.consumer_wait(consumer_store)
                cute.arch.fence_view_async_shared()
                physical_block = self._work_physical_block(mWorklist, work_idx)
                dki_offset = physical_block * Int32(self.block_size)
                owner_count = mDkiOwnerCounts[physical_block]
                if const_expr(self.deterministic):
                    if owner_count > Int32(1):
                        lock_ptr = mDkiSemaphore.iterator + physical_block
                        with cute.arch.elect_one():
                            writer_rank = mDkiWriterRank[work_idx]
                            observed_rank = Int32(-1)
                            while observed_rank != writer_rank:
                                observed_rank = ld_acquire(lock_ptr)
                        cute.arch.sync_warp()
                if owner_count == Int32(1):
                    mDKIWorkspace_cur = cute.domain_offset(
                        (dki_offset, 0), mDKIWorkspace
                    )
                    gDKIWorkspace_tiles = cute.flat_divide(
                        mDKIWorkspace_cur,
                        (self.block_size, self.head_dim),
                    )
                    tDKIsDKI, tDKIgDKI = cpasync.tma_partition(
                        tma_atom_DKIWorkspace,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sdKI, 0, 2),
                        cute.group_modes(gDKIWorkspace_tiles, 0, 2),
                    )
                    cute.copy(
                        tma_atom_DKIWorkspace,
                        tDKIsDKI,
                        tDKIgDKI[None, Int32(0), Int32(0)],
                    )
                else:
                    mDKIAccum_cur = cute.domain_offset(
                        (dki_offset, 0), mDKIAccum
                    )
                    gDKIAccum_tiles = cute.flat_divide(
                        mDKIAccum_cur, (self.block_size, self.head_dim)
                    )
                    tDKIsDKI, tDKIgDKI = cpasync.tma_partition(
                        tma_atom_DKIAccum,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sdKIAccum, 0, 2),
                        cute.group_modes(gDKIAccum_tiles, 0, 2),
                    )
                    cute.copy(
                        tma_atom_DKIAccum,
                        tDKIsDKI,
                        tDKIgDKI[None, Int32(0), Int32(0)],
                    )
                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_commit_group()
                    if const_expr(self.deterministic):
                        if owner_count > Int32(1):
                            cute.arch.cp_async_bulk_wait_group(0)
                        else:
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                    else:
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.sync_warp()
                if const_expr(self.deterministic):
                    if owner_count > Int32(1):
                        lock_ptr = mDkiSemaphore.iterator + physical_block
                        with cute.arch.elect_one():
                            red_release(lock_ptr, 1)
                        cute.arch.sync_warp()
                pipeline_store.consumer_release(consumer_store)
                consumer_store.advance()
            work_tile = tile_scheduler.advance_to_next_work(work_tile)

        # The per-work read wait only protects SMEM reuse. Drain the global
        # writes before the dKI postprocess can consume the accumulator.
        with cute.arch.elect_one():
            cute.arch.cp_async_bulk_wait_group(0)
        cute.arch.sync_warp()
        pipeline_Q.producer_tail(producer_q)
        pipeline_K.producer_tail(producer_k)
        pipeline_QI.producer_tail(producer_qi)
        pipeline_KI.producer_tail(producer_ki)

    @cute.jit
    def mma(
        self,
        tiled_mma_teacher: cute.TiledMma,
        tiled_mma_student: cute.TiledMma,
        tiled_mma_dqi: cute.TiledMma,
        tiled_mma_dki: cute.TiledMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sQI: cute.Tensor,
        sQIt: cute.Tensor,
        sQItHi: cute.Tensor,
        sKI: cute.Tensor,
        sKIt: cute.Tensor,
        sdS: cute.Tensor,
        tdS: cute.Tensor,
        tTeacher: cute.Tensor,
        tStudent: cute.Tensor,
        tdQI: cute.Tensor,
        tdKI: cute.Tensor,
        mPhysicalRowPtr: cute.Tensor,
        mWorklist: cute.Tensor,
        pipeline_Q,
        pipeline_K,
        pipeline_QI,
        pipeline_KI,
        pipeline_score_lo,
        pipeline_score_hi,
        pipeline_dS,
        pipeline_dQI,
        pipeline_dKI,
        pipeline_metadata,
        tile_scheduler,
    ) -> None:
        tTrK = tiled_mma_teacher.make_fragment_A(sK)
        tTrQ = tiled_mma_teacher.make_fragment_B(sQ)
        tSrKI = tiled_mma_student.make_fragment_A(sKI)
        tSrQI = tiled_mma_student.make_fragment_B(sQI)
        tdQIrKI = tiled_mma_dqi.make_fragment_A(sKIt)
        tdQIrdS = tiled_mma_dqi.make_fragment_B(sdS)
        tdKIrdS = tiled_mma_dki.make_fragment_A(tdS)
        tdKIrQI = tiled_mma_dki.make_fragment_B(sQIt)
        tdKIrQIHi = tiled_mma_dki.make_fragment_B(sQItHi)

        consumer_q = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.q_stage
        )
        consumer_k = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.k_stage
        )
        consumer_qi = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.q_stage
        )
        release_qi = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.q_stage
        )
        consumer_ki = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.ki_stage
        )
        producer_score_lo = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.score_stage
        )
        producer_score_hi = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.score_stage
        )
        consumer_ds = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.ds_stage
        )
        producer_dqi = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dqi_stage
        )
        producer_dki = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dki_stage
        )
        consumer_meta = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.metadata_stage
        )

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_idx = work_tile.tile_idx[0]
            has_dki = Boolean(False)
            num_macros = mWorklist[work_idx, Int32(3)]
            previous_head = Int32(-1)
            for macro in cutlass.range(num_macros + Int32(1), unroll=1):
                if macro < num_macros:
                    pipeline_metadata.consumer_wait(consumer_meta)
                    pipeline_Q.consumer_wait(consumer_q)
                    pipeline_QI.consumer_wait(consumer_qi)
                    head, _, _ = self._macro_info(
                        mPhysicalRowPtr,
                        mWorklist,
                        work_idx,
                        macro,
                    )
                    if head != previous_head:
                        pipeline_K.consumer_wait(consumer_k)
                    if macro == Int32(0):
                        pipeline_KI.consumer_wait(consumer_ki)

                    pipeline_score_lo.producer_acquire(producer_score_lo)
                    pipeline_score_hi.producer_acquire(producer_score_hi)
                    gemm_ptx_w_idx(
                        tiled_mma_student,
                        tStudent,
                        tSrKI,
                        tSrQI,
                        sA=sKI,
                        sB=sQI,
                        A_idx=consumer_ki.index,
                        B_idx=consumer_qi.index,
                        zero_init=True,
                        cta_group=1,
                    )
                    for half_i in cutlass.range_constexpr(2):
                        tTeacherHalf = tTeacher[
                            (None, None, None, Int32(half_i))
                        ]
                        q_b_idx = (
                            consumer_q.index * Int32(self.q_halves)
                            + Int32(half_i)
                        )
                        gemm_ptx_w_idx(
                            tiled_mma_teacher,
                            tTeacherHalf,
                            tTrK,
                            tTrQ,
                            sA=sK,
                            sB=sQ,
                            A_idx=consumer_k.index,
                            B_idx=q_b_idx,
                            zero_init=True,
                            cta_group=1,
                        )
                        cute.arch.fence_view_async_tmem_store()
                        if const_expr(half_i == 0):
                            pipeline_score_lo.producer_commit(
                                producer_score_lo
                            )
                            producer_score_lo.advance()
                        else:
                            pipeline_score_hi.producer_commit(
                                producer_score_hi
                            )
                            producer_score_hi.advance()
                    pipeline_Q.consumer_release(consumer_q)
                    consumer_q.advance()
                    consumer_qi.advance()
                    consumer_meta.advance()
                    release_k = Boolean(macro + Int32(1) == num_macros)
                    if macro + Int32(1) < num_macros:
                        next_head, _, _ = self._macro_info(
                            mPhysicalRowPtr,
                            mWorklist,
                            work_idx,
                            macro + Int32(1),
                        )
                        release_k = Boolean(next_head != head)
                    if release_k:
                        pipeline_K.consumer_release(consumer_k)
                        consumer_k.advance()
                    previous_head = head

                # Keep score one record ahead of the gradient UMMA.
                if macro > Int32(0):
                    consumer_ds, producer_dqi = self._mma_gradient_record(
                        tiled_mma_dqi,
                        tiled_mma_dki,
                        tdQI,
                        tdKI,
                        tdQIrKI,
                        tdQIrdS,
                        tdKIrdS,
                        tdKIrQI,
                        tdKIrQIHi,
                        sdS,
                        sKIt,
                        sQIt,
                        sQItHi,
                        consumer_ds,
                        producer_dqi,
                        consumer_ki.index,
                        release_qi.index,
                        not has_dki,
                        pipeline_dS,
                        pipeline_dQI,
                    )
                    has_dki = Boolean(True)
                    pipeline_QI.consumer_release(release_qi)
                    release_qi.advance()
            if has_dki:
                pipeline_KI.consumer_release(consumer_ki)
                consumer_ki.advance()
                pipeline_dKI.producer_acquire(producer_dki)
                cute.arch.fence_view_async_tmem_store()
                pipeline_dKI.producer_commit(producer_dki)
                producer_dki.advance()
            work_tile = tile_scheduler.advance_to_next_work(work_tile)

    @cute.jit
    def _mma_gradient_record(
        self,
        tiled_mma_dqi: cute.TiledMma,
        tiled_mma_dki: cute.TiledMma,
        tdQI: cute.Tensor,
        tdKI: cute.Tensor,
        tdQIrKI: cute.Tensor,
        tdQIrdS: cute.Tensor,
        tdKIrdS: cute.Tensor,
        tdKIrQI: cute.Tensor,
        tdKIrQIHi: cute.Tensor,
        sdS: cute.Tensor,
        sKIt: cute.Tensor,
        sQIt: cute.Tensor,
        sQItHi: cute.Tensor,
        consumer_ds,
        producer_dqi,
        ki_stage: Int32,
        qi_stage: Int32,
        zero_dki: Boolean,
        pipeline_dS,
        pipeline_dQI,
    ):
        pipeline_dS.consumer_wait(consumer_ds)
        pipeline_dQI.producer_acquire(producer_dqi)
        tdQICur = tdQI[(None, None, None, producer_dqi.index)]
        gemm_ptx_w_idx(
            tiled_mma_dqi,
            tdQICur,
            tdQIrKI,
            tdQIrdS,
            sA=sKIt,
            sB=sdS,
            A_idx=ki_stage,
            B_idx=consumer_ds.index,
            zero_init=True,
            cta_group=1,
        )
        for half_i in cutlass.range_constexpr(2):
            tdKIHalf = tdKI[(None, None, None, Int32(half_i))]
            gemm_ptx_w_idx(
                tiled_mma_dki,
                tdKIHalf,
                tdKIrdS,
                tdKIrQI if half_i == 0 else tdKIrQIHi,
                sA=None,
                sB=sQIt if half_i == 0 else sQItHi,
                A_idx=consumer_ds.index,
                B_idx=qi_stage,
                zero_init=zero_dki,
                tA_addr=self.tmem_ds_offset + consumer_ds.index * Int32(8),
                cta_group=1,
            )
        cute.arch.fence_view_async_tmem_store()
        pipeline_dQI.producer_commit(producer_dqi)
        producer_dqi.advance()
        pipeline_dS.consumer_release(consumer_ds)
        consumer_ds.advance()
        return consumer_ds, producer_dqi

    @cute.jit
    def _compute_gradient_record(
        self,
        tiled_mma_student: cute.TiledMma,
        tiled_mma_teacher_chunk: cute.TiledMma,
        tiled_mma_student_chunk: cute.TiledMma,
        tmem_ptr: cute.Pointer,
        sdS: cute.Tensor,
        sTeacherLse: cute.Tensor,
        sIndexerLse: cute.Tensor,
        sValidRows: cute.Tensor,
        metadata_stage: Int32,
        ds_stage: Int32,
        softmax_scale_log2: Float32,
        indexer_softmax_scale_log2: Float32,
    ) -> None:
        tidx = cute.arch.thread_idx()[0] - Int32(
            self.compute_warp_ids[0] * cute.arch.WARP_SIZE
        )
        wg_idx = tidx // Int32(128)
        dp_idx = tidx - wg_idx * Int32(128)

        thr_teacher = tiled_mma_teacher_chunk.get_slice(0)
        teacher_shape = thr_teacher.partition_shape_C((128, 16))
        tTeacherChunk = thr_teacher.make_fragment_C(teacher_shape)
        tTeacherChunk = cute.make_tensor(
            tmem_ptr + self.tmem_teacher_offset + wg_idx * Int32(128),
            tTeacherChunk.layout,
        )
        cTeacher = cute.make_identity_tensor((128, 16))
        tTeacherCoord = thr_teacher.partition_C(cTeacher)
        teacher_load_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), Float32
        )
        teacher_tiled_copy = tcgen05.make_tmem_copy(
            teacher_load_atom, tTeacherChunk
        )
        teacher_thr_copy = teacher_tiled_copy.get_slice(dp_idx)
        tTeacherCopyCoord = teacher_thr_copy.partition_D(tTeacherCoord)

        thr_student = tiled_mma_student_chunk.get_slice(0)
        student_shape = thr_student.partition_shape_C((128, 8))
        tStudentChunk = thr_student.make_fragment_C(student_shape)
        tStudentChunk = cute.make_tensor(
            tmem_ptr + self.tmem_student_offset + wg_idx * Int32(8),
            tStudentChunk.layout,
        )
        cStudent = cute.make_identity_tensor((128, 8))
        tStudentCoord = thr_student.partition_C(cStudent)
        student_load_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x8), Float32
        )
        student_tiled_copy = tcgen05.make_tmem_copy(
            student_load_atom, tStudentChunk
        )
        student_thr_copy = student_tiled_copy.get_slice(dp_idx)
        tStudentCopyCoord = student_thr_copy.partition_D(tStudentCoord)
        tStudentTmem = student_thr_copy.partition_S(tStudentChunk)
        tStudentRegs = cute.make_rmem_tensor(
            tStudentCopyCoord.shape, Float32
        )
        cute.copy(student_tiled_copy, tStudentTmem, tStudentRegs)
        cute.arch.fence_view_async_tmem_load()

        tdSBase = tiled_mma_student.get_slice(0).make_fragment_C(
            tiled_mma_student.get_slice(0).partition_shape_C((128, 16))
        )
        tdSPacked = cute.composition(
            tdSBase,
            (cute.make_layout((128, 8)), 1, 1),
        )
        tdSPacked = cute.make_tensor(
            tmem_ptr + self.tmem_ds_offset + ds_stage * Int32(8),
            tdSPacked.layout,
        )
        cStudentFull = tiled_mma_student.get_slice(0).partition_C(
            cute.make_identity_tensor((128, 16))
        )
        cdSPacked = cute.composition(
            cStudentFull,
            (cute.make_layout((128, 8)), 1, 1),
        )
        ds_store_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition.x4), Float32
        )
        ds_tiled_store = copy_utils.make_tmem_copy(ds_store_atom, 2)
        virtual_tidx = wg_idx * Int32(128) + dp_idx
        ds_thr_store = ds_tiled_store.get_slice(virtual_tidx)
        tdSCoord = ds_thr_store.partition_S(cdSPacked)
        tdSTmem = ds_thr_store.partition_D(tdSPacked)
        tdSReg = cute.make_rmem_tensor(8, self.dtype)

        key_idx = cute.get(tStudentCopyCoord[0], mode=[0])
        for q_i in cutlass.range_constexpr(8):
            tTeacherQ = cute.make_tensor(
                tTeacherChunk.iterator + Int32(q_i * 16),
                tTeacherChunk.layout,
            )
            tTeacherTmem = teacher_thr_copy.partition_S(tTeacherQ)
            tTeacherRegs = cute.make_rmem_tensor(
                tTeacherCopyCoord.shape, Float32
            )
            cute.copy(teacher_tiled_copy, tTeacherTmem, tTeacherRegs)
            cute.arch.fence_view_async_tmem_load()
            q_slot = wg_idx * Int32(8) + Int32(q_i)
            p_teacher_pairs = [
                (Float32(0.0), Float32(0.0)),
                (Float32(0.0), Float32(0.0)),
            ]
            for teacher_i in cutlass.range_constexpr(0, 16, 8):
                for pair_i in cutlass.range_constexpr(4):
                    index = teacher_i + pair_i * 2
                    accum_i = pair_i % 2
                    lse_0 = sTeacherLse[
                        q_slot * Int32(16) + Int32(index), metadata_stage
                    ]
                    lse_1 = sTeacherLse[
                        q_slot * Int32(16) + Int32(index + 1), metadata_stage
                    ]
                    score_0, score_1 = cute.arch.fma_packed_f32x2(
                        (tTeacherRegs[index], tTeacherRegs[index + 1]),
                        (softmax_scale_log2, softmax_scale_log2),
                        (-lse_0, -lse_1),
                    )
                    p_0 = cute.math.exp2(score_0, fastmath=True)
                    p_1 = cute.math.exp2(score_1, fastmath=True)
                    p_teacher_pairs[accum_i] = cute.arch.add_packed_f32x2(
                        p_teacher_pairs[accum_i], (p_0, p_1)
                    )
            p_teacher_pairs[0] = cute.arch.add_packed_f32x2(
                p_teacher_pairs[0], p_teacher_pairs[1]
            )
            p_teacher = p_teacher_pairs[0][0] + p_teacher_pairs[0][1]
            p_student = cute.math.exp2(
                tStudentRegs[q_i] * indexer_softmax_scale_log2
                - sIndexerLse[q_slot, metadata_stage],
                fastmath=True,
            )
            ds = p_student - p_teacher * Float32(1.0 / 16.0)
            ds = (
                ds
                if key_idx < sValidRows[q_slot, metadata_stage]
                else Float32(0.0)
            )
            tdSReg[q_i] = ds.to(self.dtype)

        sdSQ = sdS[(None, key_idx, ds_stage)]
        sdSQ = cute.zipped_divide(sdSQ, (8,))[None, wg_idx]
        tdSSmem = cute.make_tensor(tdSReg.iterator, sdSQ.shape)
        cute.autovec_copy(tdSSmem, sdSQ)

        tdSPackedReg = cute.recast_tensor(tdSReg, Float32)
        tdSTmemSrc = cute.make_tensor(tdSPackedReg.iterator, tdSCoord.shape)
        cute.copy(ds_tiled_store, tdSTmemSrc, tdSTmem)

    @cute.jit
    def _store_dki_to_smem(
        self,
        tiled_mma_dki_chunk: cute.TiledMma,
        sdKIAccum: cute.Tensor,
        sdKI: cute.Tensor,
        direct_store: Boolean,
        grad_scale: Float32,
    ) -> None:
        dp_idx = cute.arch.thread_idx()[0] % Int32(128)
        thr_chunk = tiled_mma_dki_chunk.get_slice(0)
        chunk_shape = thr_chunk.partition_shape_C((128, 32))
        tDKIChunk = thr_chunk.make_fragment_C(chunk_shape)
        cDKI = thr_chunk.partition_C(cute.make_identity_tensor((128, 32)))
        load_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), Float32
        )
        for dim_half in cutlass.range_constexpr(2):
            for pass_i in cutlass.range_constexpr(2):
                tDKICur = cute.make_tensor(
                    cute.make_ptr(
                        Float32,
                        self.tmem_dki_offset
                        + Int32(dim_half * 64 + pass_i * 32),
                        mem_space=cute.AddressSpace.tmem,
                        assumed_align=16,
                    ),
                    tDKIChunk.layout,
                )
                tiled_copy_dki = tcgen05.make_tmem_copy(load_atom, tDKICur)
                thr_copy_dki = tiled_copy_dki.get_slice(dp_idx)
                tDKITmem = thr_copy_dki.partition_S(tDKICur)
                tDKICoord = thr_copy_dki.partition_D(cDKI)
                tDKI = cute.make_rmem_tensor(tDKICoord.shape, Float32)
                cute.copy(tiled_copy_dki, tDKITmem, tDKI)
                cute.arch.fence_view_async_tmem_load()
                key_idx = cute.get(tDKICoord[0], mode=[0])
                for i in cutlass.range_constexpr(32):
                    dim = Int32(dim_half * 64 + pass_i * 32 + i)
                    if direct_store:
                        sdKI[key_idx, dim] = (
                            tDKI[i] * grad_scale
                        ).to(self.dtype)
                    else:
                        sdKIAccum[key_idx, dim] = tDKI[i]

    @cute.jit
    def compute_loop(
        self,
        tiled_mma_student: cute.TiledMma,
        tiled_mma_teacher_chunk: cute.TiledMma,
        tiled_mma_student_chunk: cute.TiledMma,
        tiled_mma_dki_chunk: cute.TiledMma,
        tmem_ptr: cute.Pointer,
        sdS: cute.Tensor,
        sdKIAccum: cute.Tensor,
        sdKI: cute.Tensor,
        sTeacherLse: cute.Tensor,
        sIndexerLse: cute.Tensor,
        sValidRows: cute.Tensor,
        mWorklist: cute.Tensor,
        mDkiOwnerCounts: cute.Tensor,
        score_lo_mbar_ptr: cute.Pointer,
        pipeline_dS,
        pipeline_dKI,
        pipeline_metadata,
        pipeline_store,
        tile_scheduler,
        softmax_scale_log2: Float32,
        indexer_softmax_scale_log2: Float32,
        grad_scale: Float32,
    ) -> None:
        score_mbar_offset = cutlass.select_(
            cute.arch.warp_idx() > Int32(self.compute_warp_ids[3]),
            Int32(2 * self.score_stage),
            Int32(0),
        )
        score_mbar_ptr = score_lo_mbar_ptr + Int32(score_mbar_offset)
        consumer_score = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.score_stage
        )
        producer_ds = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.ds_stage
        )
        consumer_meta = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.metadata_stage
        )
        consumer_dki = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dki_stage
        )
        producer_store = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.store_stage
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_idx = work_tile.tile_idx[0]
            num_macros = mWorklist[work_idx, Int32(3)]
            for _ in cutlass.range(num_macros, unroll=1):
                pipeline_metadata.consumer_wait(consumer_meta)
                cute.arch.mbarrier_wait(
                    score_mbar_ptr + consumer_score.index,
                    consumer_score.phase,
                )
                pipeline_dS.producer_acquire(producer_ds)
                self._compute_gradient_record(
                    tiled_mma_student,
                    tiled_mma_teacher_chunk,
                    tiled_mma_student_chunk,
                    tmem_ptr,
                    sdS,
                    sTeacherLse,
                    sIndexerLse,
                    sValidRows,
                    consumer_meta.index,
                    producer_ds.index,
                    softmax_scale_log2,
                    indexer_softmax_scale_log2,
                )
                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_shared()
                self.compute_sync_barrier.arrive_and_wait()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(
                        score_mbar_ptr
                        + Int32(self.score_stage)
                        + consumer_score.index
                    )
                    pipeline_dS.producer_commit(producer_ds)
                consumer_score.advance()
                producer_ds.advance()
                consumer_meta.advance()
            if (
                cute.arch.warp_idx() <= Int32(self.compute_warp_ids[3])
                and self._work_has_q(mWorklist, work_idx)
            ):
                pipeline_dKI.consumer_wait(consumer_dki)
                pipeline_store.producer_acquire(producer_store)
                physical_block = self._work_physical_block(
                    mWorklist, work_idx
                )
                self._store_dki_to_smem(
                    tiled_mma_dki_chunk,
                    sdKIAccum,
                    sdKI,
                    Boolean(mDkiOwnerCounts[physical_block] == Int32(1)),
                    grad_scale,
                )
                cute.arch.fence_view_async_shared()
                self.reduce_sync_barrier.arrive_and_wait()
                pipeline_store.producer_commit(producer_store)
                with cute.arch.elect_one():
                    pipeline_dKI.consumer_release(consumer_dki)
                consumer_dki.advance()
                producer_store.advance()
            work_tile = tile_scheduler.advance_to_next_work(work_tile)

    @cute.jit
    def dqi_acc_reduce(
        self,
        tiled_mma_dqi: cute.TiledMma,
        tdQI: cute.Tensor,
        mDQI: cute.Tensor,
        mDqiSemaphore: Optional[cute.Tensor],
        mWorklist: cute.Tensor,
        sQIdx: cute.Tensor,
        sWriterRank: cute.Tensor,
        sWorkMeta: cute.Tensor,
        pipeline_dQI,
        pipeline_metadata,
        tile_scheduler,
    ) -> None:
        tidx = cute.arch.thread_idx()[0]
        reducer_tidx = tidx % Int32(128)
        lane_in_warp = tidx % cute.arch.WARP_SIZE
        reduce_warp_idx = tidx // cute.arch.WARP_SIZE
        lane_in_quad = lane_in_warp % Int32(4)
        lane_in_16 = lane_in_warp % Int32(16)
        fake_group = lane_in_16 // Int32(4)
        thr_dqi = tiled_mma_dqi.get_slice(0)
        cDQI = thr_dqi.partition_C(cute.make_identity_tensor((128, 16)))
        load_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), Float32
        )
        consumer_dqi = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dqi_stage
        )
        consumer_meta = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.metadata_stage
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_idx = work_tile.tile_idx[0]
            num_macros = mWorklist[work_idx, Int32(3)]
            for _ in cutlass.range(num_macros, unroll=1):
                pipeline_metadata.consumer_wait(consumer_meta)
                pipeline_dQI.consumer_wait(consumer_dqi)
                tdQICur = tdQI[(None, None, None, consumer_dqi.index)]
                tiled_copy_dqi = tcgen05.make_tmem_copy(load_atom, tdQICur)
                thr_copy_dqi = tiled_copy_dqi.get_slice(reducer_tidx)
                tDQITmem = thr_copy_dqi.partition_S(tdQICur)
                tDQICoord = thr_copy_dqi.partition_D(cDQI)
                tDQI = cute.make_rmem_tensor(tDQICoord.shape, Float32)
                cute.copy(tiled_copy_dqi, tDQITmem, tDQI)
                cute.arch.fence_view_async_tmem_load()
                dim = cute.get(tDQICoord[0], mode=[0])
                fake_col = (
                    (dim // Int32(16)) * Int32(16) + fake_group * Int32(4)
                )
                head = sWorkMeta[1, consumer_meta.index]
                q_count = sWorkMeta[2, consumer_meta.index]
                for q_i in cutlass.range_constexpr(self.q_per_macro):
                    value = tDQI[q_i]
                    real_col_0 = copy_utils.stg128_fake_col_to_real_col(
                        fake_col + Int32(0)
                    )
                    real_col_1 = copy_utils.stg128_fake_col_to_real_col(
                        fake_col + Int32(1)
                    )
                    real_col_2 = copy_utils.stg128_fake_col_to_real_col(
                        fake_col + Int32(2)
                    )
                    real_col_3 = copy_utils.stg128_fake_col_to_real_col(
                        fake_col + Int32(3)
                    )
                    value_0 = cute.arch.shuffle_sync(
                        value, real_col_0 % Int32(cute.arch.WARP_SIZE)
                    )
                    value_1 = cute.arch.shuffle_sync(
                        value, real_col_1 % Int32(cute.arch.WARP_SIZE)
                    )
                    value_2 = cute.arch.shuffle_sync(
                        value, real_col_2 % Int32(cute.arch.WARP_SIZE)
                    )
                    value_3 = cute.arch.shuffle_sync(
                        value, real_col_3 % Int32(cute.arch.WARP_SIZE)
                    )
                    issuer = Int32(q_i % 4)
                    q_global = sQIdx[Int32(q_i), consumer_meta.index]
                    element_offset = (
                        (
                            Int64(q_global) * Int64(self.index_heads)
                            + Int64(head)
                        )
                        * Int64(self.head_dim)
                        + Int64(fake_col)
                    )
                    ptr = cute.make_ptr(
                        Float32,
                        mDQI.iterator.toint() + element_offset * Int64(4),
                        mem_space=mDQI.iterator.memspace,
                        assumed_align=16,
                    )
                    valid_q = Int32(q_i) < q_count
                    if const_expr(self.deterministic):
                        if valid_q:
                            lock_ptr = mDqiSemaphore[
                                head,
                                q_global,
                                None,
                            ].iterator + reduce_warp_idx
                            with cute.arch.elect_one():
                                writer_rank = sWriterRank[
                                    Int32(q_i), consumer_meta.index
                                ]
                                observed_rank = Int32(-1)
                                while observed_rank != writer_rank:
                                    observed_rank = ld_acquire(lock_ptr)
                            cute.arch.sync_warp()
                    _atomic_add_fp32x4_if(
                        Int32(
                            lane_in_quad == issuer
                            and valid_q
                        ),
                        value_0,
                        value_1,
                        value_2,
                        value_3,
                        ptr,
                    )
                    if const_expr(self.deterministic):
                        if valid_q:
                            cute.arch.sync_warp()
                            lock_ptr = mDqiSemaphore[
                                head,
                                q_global,
                                None,
                            ].iterator + reduce_warp_idx
                            with cute.arch.elect_one():
                                red_release(lock_ptr, 1)
                            cute.arch.sync_warp()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_dQI.consumer_release(consumer_dqi)
                pipeline_metadata.consumer_release(consumer_meta)
                consumer_dqi.advance()
                consumer_meta.advance()
            work_tile = tile_scheduler.advance_to_next_work(work_tile)


__all__ = ["SparseKlLossBackwardSm100"]
