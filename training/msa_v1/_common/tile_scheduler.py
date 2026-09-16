from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional, Protocol, Tuple, runtime_checkable

try:
    from typing import override
except ImportError:  # Python < 3.12
    from typing_extensions import override

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass._mlir import ir
from cutlass.cute import FastDivmodDivisor
from cutlass.pipeline import PipelineClcFetchAsync, PipelineState
from cutlass.utils import (
    ClcDynamicPersistentTileScheduler,
    ClcDynamicPersistentTileSchedulerParams,
)

import msa_v1._common.utils as utils
from msa_v1._common.cute_dsl_utils import ParamsBase
from msa_v1._common.fast_math import clz


class SchedulingMode(IntEnum):
    NONE = auto()
    STATIC = auto()
    DYNAMIC = auto()
    CLC = auto()


@dataclass
class ClcState(ParamsBase):
    """Owns the runtime state shared by CLC-capable tile schedulers.

    `SparseAttentionForwardSm100` constructs this state because it owns the CLC
    response buffer, mbarrier storage, and launch geometry needed to initialize
    the hardware scheduler and async pipeline. Individual tile schedulers then
    consume this state and map the returned hardware work tiles into their own
    logical `WorkTileInfo` coordinates.

    To add CLC support to a scheduler:
    - implement `clc_problem_shape(params)` so the kernel can create the hardware scheduler
    - accept `clc: ClcState | None` in `create(...)` / `__init__`
    - map `clc.initial_work_tile_info()` and `clc.get_current_work()` into scheduler coordinates
    """

    _hw_scheduler: ClcDynamicPersistentTileScheduler
    _pipeline: PipelineClcFetchAsync
    _consumer_state: PipelineState
    _producer_state: PipelineState

    @staticmethod
    def create(
        *,
        hw_scheduler: ClcDynamicPersistentTileScheduler,
        pipeline: PipelineClcFetchAsync,
        consumer_state: PipelineState,
        producer_state: PipelineState,
    ) -> "ClcState":
        return ClcState(hw_scheduler, pipeline, consumer_state, producer_state)

    def initial_work_tile_info(self):
        return self._hw_scheduler.initial_work_tile_info()

    def get_current_work(self):
        return self._hw_scheduler.get_current_work()

    def prefetch_next_work(self, *, loc=None, ip=None):
        self._pipeline.producer_acquire(self._producer_state, loc=loc, ip=ip)
        mbarrier_addr = self._pipeline.producer_get_barrier(self._producer_state, loc=loc, ip=ip)
        self._hw_scheduler.advance_to_next_work(mbarrier_addr, loc=loc, ip=ip)
        self._producer_state.advance(loc=loc, ip=ip)

    def consumer_wait(self, *, loc=None, ip=None):
        self._pipeline.consumer_wait(self._consumer_state, loc=loc, ip=ip)

    def consumer_release(self, *, loc=None, ip=None):
        self._pipeline.consumer_release(self._consumer_state, loc=loc, ip=ip)
        self._consumer_state.advance(loc=loc, ip=ip)

    def producer_tail(self, *, loc=None, ip=None):
        self._pipeline.producer_tail(self._producer_state, loc=loc, ip=ip)


@dataclass
class DeviceQueueState(ParamsBase):
    """Shared-memory handoff between one scheduler warp and all CTA warps."""

    _work_info: cute.Tensor
    _sync_barrier: cutlass.pipeline.NamedBarrier

    @staticmethod
    def create(
        *,
        work_info: cute.Tensor,
        sync_barrier: cutlass.pipeline.NamedBarrier,
    ) -> "DeviceQueueState":
        return DeviceQueueState(work_info, sync_barrier)

    def sync(self, *, loc=None, ip=None):
        self._sync_barrier.arrive_and_wait(loc=loc, ip=ip)



class WorkTileInfo(cutlass.utils.WorkTileInfo):
    """Altered WorkTileInfo which includes four axes: (block, head, batch, split)"""

    @override
    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "WorkTileInfo":
        new_tile_idx = cutlass.new_from_mlir_values(self._tile_idx, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(self._is_valid_tile, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)


@runtime_checkable
class TileSchedulerProtocol(Protocol):
    """Protocol defining the interface all tile schedulers must implement.

    Schedulers are responsible for:
    1. Coordinate mapping: linear tile index -> (m_block, head, batch, split)
    2. Work distribution: how to get the next tile (static grid-stride vs CLC dynamic)
    """

    def get_current_work(self) -> WorkTileInfo:
        """Get the current work tile coordinates."""
        ...

    def initial_work_tile_info(self) -> WorkTileInfo:
        """Get the initial work tile for this CTA."""
        ...

    def advance_to_next_work(self, *, loc=None, ip=None):
        """Consumer-side advance: move to next tile and return it.

        For static schedulers: grid-stride increment + get_current_work.
        For CLC schedulers: consumer wait + get_current_work + consumer release + state advance.
        """
        ...

    def complete_current_work(self, *, loc=None, ip=None):
        """Rendezvous all CTA roles before assigning the next work tile."""
        ...

    def prefetch_next_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        """Producer-side prefetch of next work tile.

        Only the scheduler warp calls this method. Dynamic schedulers return the
        work they published; static schedulers may return their current work.
        """
        ...

    def producer_tail(self, *, loc=None, ip=None) -> None:
        """Producer-side cleanup after the last tile.

        No-op for static schedulers. For CLC schedulers: pipeline producer_tail.
        """
        ...


@dataclass
class TileSchedulerArguments(ParamsBase):
    num_block: Int32
    num_head: Int32
    num_batch: Int32
    num_splits: Int32
    seqlen_k: Int32
    headdim: Int32
    headdim_v: Int32
    total_q: Int32
    tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
    cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
    mCuSeqlensQ: Optional[cute.Tensor] = None
    mSeqUsedQ: Optional[cute.Tensor] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    element_size: cutlass.Constexpr[int] = 2
    is_persistent: cutlass.Constexpr[bool] = False
    lpt: cutlass.Constexpr[bool] = False
    is_split_kv: cutlass.Constexpr[bool] = False
    head_swizzle: cutlass.Constexpr[bool] = False
    use_cluster_idx: cutlass.Constexpr[bool] = False
    use_global_q_direct: cutlass.Constexpr[bool] = False
    clc_sync_after_work: cutlass.Constexpr[bool] = False


class SingleTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        num_splits: Int32
        num_splits_divmod: FastDivmodDivisor
        is_split_kv: cutlass.Constexpr[bool] = False
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
        use_cluster_idx: cutlass.Constexpr[bool] = False

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileScheduler.Params":
            return SingleTileScheduler.Params(
                args.num_block,
                args.num_head,
                args.num_batch,
                args.num_splits,
                FastDivmodDivisor(args.num_splits),
                args.is_split_kv,
                args.cluster_shape_mn,
                args.use_cluster_idx,
            )

    def __init__(self, params: Params, blk_coord: cute.Coord, *, loc=None, ip=None):
        self.params = params
        self._blk_coord = blk_coord
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert scheduling_mode == SchedulingMode.STATIC, (
            f"SingleTileScheduler only supports STATIC, got {scheduling_mode!r}"
        )
        return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(
        params: Params, clc: ClcState | None = None, *, loc=None, ip=None
    ) -> "SingleTileScheduler":
        if const_expr(cute.size(params.cluster_shape_mn) == 1 or not params.use_cluster_idx):
            blk_coord = cute.arch.block_idx()
        else:
            blk_coord = cute.arch.cluster_idx()
        return SingleTileScheduler(params, blk_coord, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        # TODO: this hard-codes the fact that we only use cluster = (1, 1) or (2, 1)
        assert params.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
        if const_expr(params.use_cluster_idx):
            # Grid must have num_block * cluster_m physical blocks so that there are num_block clusters
            grid_x = params.num_block * params.cluster_shape_mn[0]
        else:
            grid_x = cute.round_up(params.num_block, params.cluster_shape_mn[0])
        return (
            grid_x,
            params.num_head * params.num_splits,
            params.num_batch,
        )

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        block_idx, head_idx, batch_idx = self._blk_coord
        if const_expr(self.params.is_split_kv):
            head_idx, split_idx = divmod(head_idx, self.params.num_splits_divmod)
        else:
            split_idx = Int32(0)
        return WorkTileInfo(
            (block_idx, head_idx, batch_idx, split_idx),
            self._is_first_block,
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._blk_coord]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._blk_coord], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileScheduler(*(tuple(obj_list)), loc=self._loc)


class StaticPersistentTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block_cluster_divmod: FastDivmodDivisor
        num_head_divmod: FastDivmodDivisor
        total_blocks_cluster: Int32
        cluster_shape_m: cutlass.Constexpr[int] = 1

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "StaticPersistentTileScheduler.Params":
            num_block_cluster = cute.ceil_div(args.num_block, cute.size(args.cluster_shape_mn))
            total_blocks_cluster = num_block_cluster * args.num_head * args.num_batch
            return StaticPersistentTileScheduler.Params(
                FastDivmodDivisor(num_block_cluster),
                FastDivmodDivisor(args.num_head),
                total_blocks_cluster,
                cluster_shape_m=args.cluster_shape_mn[0],
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert scheduling_mode == SchedulingMode.STATIC, (
            f"StaticPersistentTileScheduler only supports STATIC, got {scheduling_mode!r}"
        )
        return StaticPersistentTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(
        params: Params, clc: ClcState | None = None, *, loc=None, ip=None
    ) -> "StaticPersistentTileScheduler":
        if const_expr(cute.size(params.cluster_shape_m) == 1):
            tile_idx = cute.arch.block_idx()[0]
        else:
            tile_idx = cute.arch.cluster_idx()[0]
        return StaticPersistentTileScheduler(params, tile_idx, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        usable_SM_count=0,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        hardware_info = cutlass.utils.HardwareInfo()
        cluster_shape_m = int(params.cluster_shape_m)
        if usable_SM_count > 0:
            sm_count = usable_SM_count
        else:
            sm_count = hardware_info.get_device_multiprocessor_count()
        max_ctas = (sm_count // cluster_shape_m) * cluster_shape_m
        max_ctas = max(max_ctas, cluster_shape_m)
        grid_x = cutlass.min(max_ctas, params.total_blocks_cluster * cluster_shape_m)
        return (grid_x, Int32(1), Int32(1))

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        hn_idx, block_idx = divmod(self._tile_idx, self.params.num_block_cluster_divmod)
        batch_idx, head_idx = divmod(hn_idx, self.params.num_head_divmod)
        is_valid = self._tile_idx < self.params.total_blocks_cluster
        return WorkTileInfo(
            (Int32(block_idx), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.cluster_shape_m == 1):
            self._tile_idx += cute.arch.grid_dim()[0]
        else:
            self._tile_idx += cute.arch.cluster_dim()[0]
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.params, self._tile_idx],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return StaticPersistentTileScheduler(*(tuple(obj_list)), loc=self._loc)


class DeviceQueuePersistentTileScheduler:
    """Resident CTA scheduler with device-side dynamic work distribution.

    The first resident wave maps directly from CTA id to the compact worklist.
    One scheduler warp per CTA then obtains additional work through a global
    ticket counter and broadcasts it through a one-stage shared-memory
    pipeline. A retirement counter lets the final active CTA reset both queue
    fields before kernel completion, so repeated launches and CUDA Graph replay
    do not require a host action or a separate reset kernel.
    """

    NEXT_TICKET = 0
    RETIRED_CTA = 1
    STATE_FIELDS = 2

    @dataclass
    class Params(ParamsBase):
        work_capacity: Int32

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "DeviceQueuePersistentTileScheduler.Params":
            return DeviceQueuePersistentTileScheduler.Params(args.num_block)

    def __init__(
        self,
        params: Params,
        mWorkCount: cute.Tensor,
        mQueueState: cute.Tensor,
        state: DeviceQueueState,
        work_idx: Int32,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self.mWorkCount = mWorkCount
        self.mQueueState = mQueueState
        self.state = state
        self._work_idx = work_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.DYNAMIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert scheduling_mode == SchedulingMode.DYNAMIC, (
            "DeviceQueuePersistentTileScheduler only supports DYNAMIC"
        )
        return DeviceQueuePersistentTileScheduler.Params.create(
            args, loc=loc, ip=ip
        )

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        mWorkCount: cute.Tensor,
        mQueueState: cute.Tensor,
        state: DeviceQueueState,
        *,
        loc=None,
        ip=None,
    ) -> "DeviceQueuePersistentTileScheduler":
        return DeviceQueuePersistentTileScheduler(
            params,
            mWorkCount,
            mQueueState,
            state,
            cute.arch.block_idx()[0],
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        usable_SM_count=0,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = (
            usable_SM_count
            if usable_SM_count > 0
            else hardware_info.get_device_multiprocessor_count()
        )
        resident_ctas = max(sm_count, 1)
        return (
            cutlass.min(resident_ctas, params.work_capacity),
            Int32(1),
            Int32(1),
        )

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        return WorkTileInfo(
            (self._work_idx, Int32(0), Int32(0), Int32(0)),
            self._work_idx < self.mWorkCount[Int32(0)],
        )

    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self.get_current_work(loc=loc, ip=ip)

    @cute.jit
    def prefetch_next_work(self, *, loc=None, ip=None):
        self.state.sync(loc=loc, ip=ip)
        next_work_idx = Int32(0)
        if cute.arch.lane_idx() == Int32(0):
            ticket = cute.arch.atomic_add(
                (self.mQueueState.iterator + Int32(self.NEXT_TICKET)).llvm_ptr,
                Int32(1),
                sem="relaxed",
                scope="gpu",
            )
            next_work_idx = cute.arch.grid_dim()[0] + ticket
        next_work_idx = cute.arch.shuffle_sync(next_work_idx, 0)
        with cute.arch.elect_one():
            self.state._work_info[Int32(0)] = next_work_idx
        self.state.sync(loc=loc, ip=ip)
        self._work_idx = next_work_idx
        return self.get_current_work(loc=loc, ip=ip)

    @cute.jit
    def complete_current_work(self, *, loc=None, ip=None):
        self.state.sync(loc=loc, ip=ip)

    @cute.jit
    def advance_to_next_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        self.state.sync(loc=loc, ip=ip)
        self._work_idx = self.state._work_info[Int32(0)]
        return self.get_current_work(loc=loc, ip=ip)

    @cute.jit
    def producer_tail(self, *, loc=None, ip=None):
        if cute.arch.lane_idx() == Int32(0):
            retired = cute.arch.atomic_add(
                (self.mQueueState.iterator + Int32(self.RETIRED_CTA)).llvm_ptr,
                Int32(1),
                sem="relaxed",
                scope="gpu",
            )
            active_ctas = cutlass.min(
                cute.arch.grid_dim()[0],
                self.mWorkCount[Int32(0)],
            )
            if retired + Int32(1) == active_ctas:
                self.mQueueState[Int32(self.NEXT_TICKET)] = Int32(0)
                self.mQueueState[Int32(self.RETIRED_CTA)] = Int32(0)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.params,
            self.mWorkCount,
            self.mQueueState,
            self.state,
            self._work_idx,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self.params,
                self.mWorkCount,
                self.mQueueState,
                self.state,
                self._work_idx,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return DeviceQueuePersistentTileScheduler(
            *(tuple(obj_list)), loc=self._loc, ip=self._ip
        )



class SingleTileLPTScheduler:
    @dataclass
    class Params(ParamsBase):
        total_blocks: Int32
        num_splits: Int32
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        l2_minor: Int32
        num_head_divmod: FastDivmodDivisor
        l2_minor_divmod: FastDivmodDivisor
        l2_major_divmod: FastDivmodDivisor
        l2_minor_residual_divmod: FastDivmodDivisor
        num_hb_quotient: Int32
        num_splits_divmod: FastDivmodDivisor
        is_split_kv: cutlass.Constexpr[bool] = False
        cluster_shape_m: cutlass.Constexpr[int] = 1
        scheduling_mode: cutlass.Constexpr[SchedulingMode] = SchedulingMode.STATIC
        lpt: cutlass.Constexpr[bool] = True
        use_cluster_idx: cutlass.Constexpr[bool] = True

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments,
            *,
            scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
            loc=None,
            ip=None,
        ) -> "SingleTileLPTScheduler.Params":
            assert scheduling_mode in (SchedulingMode.STATIC, SchedulingMode.CLC), (
                f"Only STATIC and CLC are supported, got {scheduling_mode!r}"
            )
            size_one_kv_head = args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size
            size_one_head = size_one_kv_head
            size_l2 = 50 * 1024 * 1024  # 40 MB for K & V
            # Swizzle is the size of each "section". Round swizzle to a power of 2
            # Need to be careful about the case where only one head will fit
            # swizzle is how many heads can fit in L2
            # Seems faster if swizzle is a power of 2
            log2_floor = lambda n: 31 - clz(n)
            swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(size_l2 // size_one_head))
            # If we're in the last section (called residual), we don't want to divide by
            # swizzle. Instead we want to divide by the remainder.
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            return SingleTileLPTScheduler.Params(
                total_blocks=args.num_block * args.num_head * args.num_batch,
                num_block=args.num_block,
                num_head=args.num_head,
                num_batch=args.num_batch,
                l2_minor=Int32(swizzle),
                num_head_divmod=FastDivmodDivisor(args.num_head),
                l2_minor_divmod=FastDivmodDivisor(swizzle),
                l2_major_divmod=FastDivmodDivisor(swizzle * args.num_block),
                l2_minor_residual_divmod=FastDivmodDivisor(max(num_hb_remainder, 1)),
                num_hb_quotient=Int32(num_hb_quotient),
                num_splits=args.num_splits,
                num_splits_divmod=FastDivmodDivisor(args.num_splits),
                is_split_kv=args.is_split_kv,
                cluster_shape_m=args.cluster_shape_mn[0],
                scheduling_mode=scheduling_mode,
                lpt=args.lpt,
                use_cluster_idx=args.use_cluster_idx,
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        split_idx: Int32,
        clc: ClcState | None = None,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
        self._split_idx = split_idx
        self.clc = clc
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        return SingleTileLPTScheduler.Params.create(
            args, scheduling_mode=scheduling_mode, loc=loc, ip=ip
        )

    @staticmethod
    def _clc_grid_shape(params: Params):
        num_batch_splits = (
            params.num_batch * params.num_splits
            if const_expr(params.is_split_kv)
            else params.num_batch
        )
        return (
            cute.round_up(params.num_block, params.cluster_shape_m),
            params.num_head,
            num_batch_splits,
        )

    @staticmethod
    @cute.jit
    def clc_problem_shape(params: Params):
        return ClcDynamicPersistentTileSchedulerParams(
            problem_shape_ntile_mnl=SingleTileLPTScheduler._clc_grid_shape(params),
            cluster_shape_mnk=(params.cluster_shape_m, 1, 1),
        )

    @staticmethod
    @cute.jit
    def create(
        params: Params, clc: ClcState | None = None, *, loc=None, ip=None
    ) -> "SingleTileLPTScheduler":
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            return SingleTileLPTScheduler(
                params, cute.arch.block_idx()[0], Int32(0), clc, loc=loc, ip=ip
            )
        tile_idx, split_idx, _ = cute.arch.block_idx()
        return SingleTileLPTScheduler(params, tile_idx, split_idx, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            return SingleTileLPTScheduler._clc_grid_shape(params)
        return (params.total_blocks, params.num_splits, Int32(1))

    @cute.jit
    def clc_work_to_coords(self, work) -> WorkTileInfo:
        """Convert CLC response (block, head, batch_split) to WorkTileInfo.

        CLC returns raw grid coordinates — no L2 swizzle (hardware decides order).
        We only apply cluster division, optional LPT block reversal, and split_kv unpacking.
        """
        block_idx = work.tile_idx[0]
        if const_expr(self.params.cluster_shape_m > 1):
            block_idx = block_idx // self.params.cluster_shape_m
        if const_expr(self.params.lpt):
            # Longest-processing-time-first: reverse block order
            if const_expr(self.params.cluster_shape_m > 1 and not self.params.use_cluster_idx):
                num_block = self.params.num_block // self.params.cluster_shape_m
            else:
                num_block = self.params.num_block
            block_idx = num_block - 1 - block_idx
        split_idx = Int32(0)
        if const_expr(self.params.is_split_kv):
            batch_idx, split_idx = divmod(work.tile_idx[2], self.params.num_splits_divmod)
        else:
            batch_idx = work.tile_idx[2]
        if const_expr(self.params.cluster_shape_m > 1 and not self.params.use_cluster_idx):
            bidx_in_cluster = cute.arch.block_in_cluster_idx()
            block_idx = block_idx * self.params.cluster_shape_m + bidx_in_cluster[0]
        return WorkTileInfo(
            (Int32(block_idx), Int32(work.tile_idx[1]), Int32(batch_idx), Int32(split_idx)),
            work.is_valid_tile,
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            work = self.clc.get_current_work()
            self._tile_idx = work.tile_idx[0]
            return self.clc_work_to_coords(work)
        # Static path: L2-swizzled coordinate mapping
        params = self.params
        # Implement LPT scheduling coordinate calculation
        bidhb, l2_mod = divmod(self._tile_idx, params.l2_major_divmod)
        # If we're in the last section (called residual), we don't want to divide by
        # swizzle. Instead we want to divide by the remainder.
        block, bidhb_residual = 0, 0
        if bidhb < params.num_hb_quotient:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_divmod)
        else:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_residual_divmod)
        bidhb_actual = bidhb * params.l2_minor + bidhb_residual
        batch_idx, head_idx = divmod(bidhb_actual, params.num_head_divmod)
        # Longest-processing-time-first
        if const_expr(params.lpt):
            block = params.num_block - 1 - block
        is_valid = self._tile_idx < params.total_blocks
        return WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(self._split_idx)), is_valid
        )

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            work = self.clc.initial_work_tile_info()
            self._tile_idx = work.tile_idx[0]
            return self.clc_work_to_coords(work)
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.prefetch_next_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.consumer_wait(loc=loc, ip=ip)
            work = self.get_current_work()
            self.clc.consumer_release(loc=loc, ip=ip)
            return work
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._tile_idx = self.params.total_blocks
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.producer_tail(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self.clc]
        for obj in objs:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self.clc]
        for obj, n_items in zip(objs, self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*obj_list, loc=self._loc)


class SingleTileLPTBwdScheduler:
    @dataclass
    class Params(ParamsBase):
        total_blocks: Int32
        num_block: Int32
        l2_minor: Int32
        num_head_divmod: FastDivmodDivisor
        l2_minor_divmod: FastDivmodDivisor
        l2_major_divmod: FastDivmodDivisor
        l2_minor_residual_divmod: FastDivmodDivisor
        num_hb_quotient: Int32
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
        spt: cutlass.Constexpr[bool] = True

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileLPTBwdScheduler.Params":
            size_l2 = 50 * 1024 * 1024
            size_one_qdo_head = args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size
            size_one_dqaccum_head = args.seqlen_k * (args.headdim) * 4
            size_one_head = size_one_qdo_head + size_one_dqaccum_head
            log2_floor = lambda n: 31 - clz(n)
            swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(size_l2 // size_one_head))
            # If we're in the last section (called residual), we don't want to divide by
            # swizzle. Instead we want to divide by the remainder.
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            num_block = cute.ceil_div(args.num_block, args.cluster_shape_mn[0])
            return SingleTileLPTBwdScheduler.Params(
                total_blocks=(num_block * args.cluster_shape_mn[0])
                * args.num_head
                * args.num_batch,
                num_block=num_block,
                l2_minor=Int32(swizzle),
                num_head_divmod=FastDivmodDivisor(args.num_head),
                l2_minor_divmod=FastDivmodDivisor(swizzle),
                l2_major_divmod=FastDivmodDivisor(swizzle * num_block),
                l2_minor_residual_divmod=FastDivmodDivisor(
                    max(num_hb_remainder, 1)
                ),  # don't divide by 0
                num_hb_quotient=Int32(num_hb_quotient),
                cluster_shape_mn=args.cluster_shape_mn,
                spt=args.lpt,
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert scheduling_mode == SchedulingMode.STATIC, (
            f"SingleTileLPTBwdScheduler only supports STATIC, got {scheduling_mode!r}"
        )
        return SingleTileLPTBwdScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileLPTBwdScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return SingleTileLPTBwdScheduler(params, tile_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        return (params.total_blocks, Int32(1), Int32(1))

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        cluster_idx = self._tile_idx // self.params.cluster_shape_mn[0]
        params = self.params
        # Implement LPT scheduling coordinate calculation
        bidhb, l2_mod = divmod(cluster_idx, params.l2_major_divmod)
        # If we're in the last section (called residual), we don't want to divide by
        # swizzle. Instead we want to divide by the remainder.
        block, bidhb_residual = 0, 0
        if bidhb < params.num_hb_quotient:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_divmod)
        else:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_residual_divmod)
        bidhb_actual = bidhb * params.l2_minor + bidhb_residual
        batch_idx, head_idx = divmod(bidhb_actual, params.num_head_divmod)
        if cutlass.const_expr(params.spt):
            block = params.num_block - 1 - block
        if cutlass.const_expr(params.cluster_shape_mn[0] > 1):
            bidx_in_cluster = cute.arch.block_in_cluster_idx()
            block = block * params.cluster_shape_mn[0] + bidx_in_cluster[0]
        is_valid = self._tile_idx < params.total_blocks
        return WorkTileInfo((Int32(block), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._tile_idx = self.params.total_blocks
        return self.get_current_work()

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._tile_idx], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)


class SingleTileVarlenScheduler:
    @dataclass
    class Params(ParamsBase):
        num_head: Int32
        num_batch: Int32
        total_q: Int32
        num_head_divmod: FastDivmodDivisor
        total_q_divmod: FastDivmodDivisor
        num_splits: Int32
        max_kvblock_in_l2: Int32
        tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
        mCuSeqlensQ: Optional[cute.Tensor] = None
        mSeqUsedQ: Optional[cute.Tensor] = None
        qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
        lpt: cutlass.Constexpr[bool] = False
        is_split_kv: cutlass.Constexpr[bool] = False
        head_swizzle: cutlass.Constexpr[bool] = False
        cluster_shape_m: cutlass.Constexpr[int] = 1
        use_global_q_direct: cutlass.Constexpr[bool] = False
        clc_sync_after_work: cutlass.Constexpr[bool] = False
        is_persistent: cutlass.Constexpr[bool] = False
        scheduling_mode: cutlass.Constexpr[SchedulingMode] = SchedulingMode.STATIC

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments,
            *,
            scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
            loc=None,
            ip=None,
        ) -> "SingleTileVarlenScheduler.Params":
            assert scheduling_mode in (SchedulingMode.STATIC, SchedulingMode.CLC), (
                f"Only STATIC and CLC are supported, got {scheduling_mode!r}"
            )
            size_l2 = 50 * 1024 * 1024  # 50 MB for K & V
            # if backward, this is qdo block size
            kv_block_size = (
                (args.headdim + args.headdim_v) * args.element_size * args.tile_shape_mn[1]
            )
            # if backward, add dqaccum block size to calculate swizzle
            if args.head_swizzle:
                kv_block_size += args.headdim * 4 * args.tile_shape_mn[1]
            max_kvblock_in_l2 = size_l2 // kv_block_size
            assert args.mCuSeqlensQ is not None or args.mSeqUsedQ is not None, (
                "At least one of mCuSeqlensQ or mSeqUsedQ must be provided"
            )
            assert args.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
            # TODO: Support varlen CLC with cluster_shape_m > 1 by refactoring the
            # flattened-tile decode so cluster unpacking semantics are explicit.
            assert scheduling_mode != SchedulingMode.CLC or args.cluster_shape_mn[0] == 1, (
                "Varlen CLC currently requires cluster_shape_mn[0] == 1"
            )
            return SingleTileVarlenScheduler.Params(
                num_head=args.num_head,
                num_batch=args.num_batch,
                total_q=args.total_q,
                num_head_divmod=FastDivmodDivisor(args.num_head),
                total_q_divmod=FastDivmodDivisor(
                    cutlass.max(args.total_q, Int32(1))
                ),
                num_splits=args.num_splits,
                max_kvblock_in_l2=max_kvblock_in_l2,
                tile_shape_mn=args.tile_shape_mn,
                mCuSeqlensQ=args.mCuSeqlensQ,
                mSeqUsedQ=args.mSeqUsedQ,
                qhead_per_kvhead_packgqa=args.qhead_per_kvhead_packgqa,
                lpt=args.lpt,
                is_split_kv=args.is_split_kv,
                head_swizzle=args.head_swizzle,
                cluster_shape_m=args.cluster_shape_mn[0],
                use_global_q_direct=args.use_global_q_direct,
                clc_sync_after_work=args.clc_sync_after_work,
                is_persistent=args.is_persistent,
                scheduling_mode=scheduling_mode,
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        split_idx: Int32,
        clc: ClcState | None = None,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
        self._split_idx = split_idx
        self._is_first_block = True
        self.clc = clc
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        return SingleTileVarlenScheduler.Params.create(
            args, scheduling_mode=scheduling_mode, loc=loc, ip=ip
        )

    @staticmethod
    @cute.jit
    def clc_problem_shape(params: Params):
        return ClcDynamicPersistentTileSchedulerParams(
            problem_shape_ntile_mnl=SingleTileVarlenScheduler.get_grid_shape(params),
            cluster_shape_mnk=(1, 1, 1),
        )

    @staticmethod
    @cute.jit
    def create(
        params: Params, clc: ClcState | None = None, *, loc=None, ip=None
    ) -> "SingleTileVarlenScheduler":
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            block_idx = cute.arch.block_idx()
            split_idx = Int32(0)
            if const_expr(params.is_split_kv):
                split_idx = block_idx[1]
            return SingleTileVarlenScheduler(
                params,
                block_idx[0],
                split_idx,
                clc,
                loc=loc,
                ip=ip,
            )
        tile_idx, split_idx, _ = cute.arch.block_idx()
        return SingleTileVarlenScheduler(params, tile_idx, split_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        if const_expr(params.use_global_q_direct):
            total_blocks_max = params.total_q * params.cluster_shape_m
        else:
            total_blocks_max = (
                params.total_q
                + params.num_batch * (params.cluster_shape_m * params.tile_shape_mn[0] - 1)
            ) // params.tile_shape_mn[0]
            # Round down to nearest multiple of cluster since odd excess is always padding.
            total_blocks_max = total_blocks_max // params.cluster_shape_m * params.cluster_shape_m
        grid_x = total_blocks_max * params.num_head
        if const_expr(
            params.is_persistent
            and params.scheduling_mode == SchedulingMode.STATIC
        ):
            hardware_info = cutlass.utils.HardwareInfo()
            sm_count = hardware_info.get_device_multiprocessor_count()
            grid_x = cutlass.min(grid_x, sm_count)
        return (grid_x, params.num_splits, Int32(1))

    @cute.jit
    def _get_num_m_blocks(self, lane: Int32, bidb_start: Int32) -> Int32:
        params = self.params
        batch_idx = lane + bidb_start
        if cutlass.const_expr(params.mSeqUsedQ is not None):
            seqlen = Int32(0)
            if batch_idx < params.num_batch:
                seqlen = params.mSeqUsedQ[batch_idx]
        else:
            assert params.mCuSeqlensQ is not None
            cur_cu_seqlen = Int32(0)
            if batch_idx <= params.num_batch:
                cur_cu_seqlen = params.mCuSeqlensQ[batch_idx]
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen = next_cu_seqlen - cur_cu_seqlen
        if cutlass.const_expr(params.qhead_per_kvhead_packgqa > 1):
            seqlen *= params.qhead_per_kvhead_packgqa
        return (
            cute.ceil_div(cute.ceil_div(seqlen, params.tile_shape_mn[0]), params.cluster_shape_m)
            if batch_idx < params.num_batch and lane < cute.arch.WARP_SIZE - 1
            else Int32(0)
        )

    @cute.jit
    def _varlen_coord_map(self) -> WorkTileInfo:
        """Map self._tile_idx to (block, head, batch) via warp-level prefix sums."""
        params = self.params
        lane_idx = cute.arch.lane_idx()
        num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=0)
        num_m_blocks_cumulative = utils.warp_prefix_sum(num_m_blocks, lane_idx)
        # Total number of blocks for the next 31 batches
        m_blocks_in_group = cute.arch.shuffle_sync(num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1)
        # Same for all lanes
        group_end_tile = m_blocks_in_group * params.num_head
        block, head_idx, batch_idx = Int32(0), Int32(0), Int32(0)
        next_tile_idx = self._tile_idx // params.cluster_shape_m
        while group_end_tile <= next_tile_idx:
            batch_idx += cute.arch.WARP_SIZE - 1
            if batch_idx >= params.num_batch:
                batch_idx = Int32(params.num_batch)
                group_end_tile = next_tile_idx + 1
            else:
                num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=batch_idx)
                num_m_blocks_cumulative = utils.warp_prefix_sum(num_m_blocks, lane_idx)
                m_blocks_in_group = cute.arch.shuffle_sync(
                    num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1
                )
                group_end_tile += m_blocks_in_group * params.num_head
        is_valid = False
        if batch_idx >= params.num_batch:
            block, head_idx, batch_idx = Int32(0), Int32(0), Int32(params.num_batch)
        else:
            group_start_tile = group_end_tile - m_blocks_in_group * params.num_head
            # The next problem to process is the first one that does not have ending tile position
            # that is greater than or equal to tile index.
            batch_idx_in_group = cute.arch.popc(
                cute.arch.vote_ballot_sync(
                    group_start_tile + num_m_blocks_cumulative * params.num_head <= next_tile_idx
                )
            )
            batch_idx += batch_idx_in_group
            num_m_blocks_prev_lane = (
                0
                if batch_idx_in_group == 0
                else cute.arch.shuffle_sync(num_m_blocks_cumulative, batch_idx_in_group - 1)
            )
            num_m_blocks = cute.arch.shuffle_sync(num_m_blocks, batch_idx_in_group)
            mh_block = next_tile_idx - group_start_tile - num_m_blocks_prev_lane * params.num_head
            if cutlass.const_expr(params.lpt or params.head_swizzle):
                # This is a version of the SingleTileLPTScheduler, complicated by the fact that
                # the seqlen can vary per batch.
                # TODO: is there any case where num_m_blocks is 0?
                # TODO: by right we should read the seqlen_kv but we're assuming seqlen_q == seqlen_k here
                num_n_blocks = (
                    num_m_blocks
                    * params.tile_shape_mn[0]
                    * params.cluster_shape_m
                    // params.qhead_per_kvhead_packgqa
                    // params.tile_shape_mn[1]
                )
                # Seems faster to have this be a power of 2
                nheads_in_l2 = (
                    16
                    if num_n_blocks * 16 <= params.max_kvblock_in_l2
                    else (
                        8
                        if num_n_blocks * 8 <= params.max_kvblock_in_l2
                        else (
                            4
                            if num_n_blocks * 4 <= params.max_kvblock_in_l2
                            else (2 if num_n_blocks * 2 <= params.max_kvblock_in_l2 else 1)
                        )
                    )
                )
                nheads_in_l2 = min(nheads_in_l2, params.num_head)
                mh_in_l2 = nheads_in_l2 * num_m_blocks
                section_idx = mh_block // mh_in_l2
                l2_mod = mh_block - section_idx * mh_in_l2
                # Deal with tail section
                nheads_in_this_section = (
                    nheads_in_l2
                    if nheads_in_l2 * (section_idx + 1) <= params.num_head
                    else params.num_head - section_idx * nheads_in_l2
                )
                block = l2_mod // nheads_in_this_section
                head_idx_residual = l2_mod - block * nheads_in_this_section
                head_idx = section_idx * nheads_in_l2 + head_idx_residual
                if cutlass.const_expr(params.lpt):
                    block = num_m_blocks - 1 - block
            else:
                head_idx = mh_block // num_m_blocks
                block = mh_block - head_idx * num_m_blocks
            is_valid = self._is_first_block and batch_idx < params.num_batch
            if cutlass.const_expr(params.cluster_shape_m > 1):
                bidx_in_cluster = cute.arch.block_in_cluster_idx()
                block = block * params.cluster_shape_m + bidx_in_cluster[0]
        split_idx = self._split_idx if const_expr(params.is_split_kv) else Int32(0)
        return WorkTileInfo((Int32(block), Int32(head_idx), Int32(batch_idx), split_idx), is_valid)

    @cute.jit
    def _direct_global_q_coord_map(self) -> WorkTileInfo:
        """Map flattened global-q work directly through cu_seqlens."""
        params = self.params
        assert params.mCuSeqlensQ is not None, "Direct global-q mapping requires cu_seqlens_q"
        logical_tile_idx = self._tile_idx // params.cluster_shape_m
        global_q, head_idx = divmod(logical_tile_idx, params.num_head_divmod)
        batch_begin = Int32(0)
        batch_end = params.num_batch
        batch_guess, _ = divmod(
            global_q * params.num_batch,
            params.total_q_divmod,
        )
        batch_guess = cutlass.min(
            batch_guess,
            params.num_batch - Int32(1),
        )
        guess_start = params.mCuSeqlensQ[batch_guess]
        guess_end = params.mCuSeqlensQ[batch_guess + Int32(1)]
        guess_matches = guess_start <= global_q and global_q < guess_end
        if guess_matches:
            batch_begin = batch_guess
            batch_end = batch_guess
        elif global_q < guess_start:
            batch_end = batch_guess
        else:
            batch_begin = batch_guess + Int32(1)
        while batch_begin < batch_end:
            batch_mid = batch_begin + (batch_end - batch_begin) // Int32(2)
            if params.mCuSeqlensQ[batch_mid + Int32(1)] <= global_q:
                batch_begin = batch_mid + Int32(1)
            else:
                batch_end = batch_mid
        batch_idx = batch_begin
        q_local = Int32(0)
        if batch_idx < params.num_batch:
            q_local = global_q - params.mCuSeqlensQ[batch_idx]
            if cutlass.const_expr(params.lpt):
                q_local = (
                    params.mCuSeqlensQ[batch_idx + Int32(1)]
                    - params.mCuSeqlensQ[batch_idx]
                    - Int32(1)
                    - q_local
                )
        is_valid = global_q < params.total_q and batch_idx < params.num_batch
        if cutlass.const_expr(not params.is_persistent):
            is_valid = self._is_first_block and is_valid
        split_idx = Int32(0)
        return WorkTileInfo((q_local, head_idx, batch_idx, split_idx), is_valid)

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            clc_work = self.clc.get_current_work()
            # Default to grid_dim (one past last valid flat index) so _varlen_coord_map
            # returns is_valid=False when CLC is exhausted. CLC tile_idx is garbage when
            # invalid, so we can't trust it. Local-then-assign avoids CuTe DSL structural
            # mismatch on self inside the runtime if.
            new_tile_idx = cute.arch.grid_dim()[0]
            new_split_idx = Int32(0)
            if clc_work.is_valid_tile:
                new_tile_idx = clc_work.tile_idx[0]
                if const_expr(self.params.is_split_kv):
                    new_split_idx = clc_work.tile_idx[1]
            self._tile_idx = new_tile_idx
            self._split_idx = new_split_idx
        if const_expr(self.params.use_global_q_direct):
            return self._direct_global_q_coord_map()
        return self._varlen_coord_map()

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            clc_work = self.clc.initial_work_tile_info()
            # See get_current_work for why grid_dim and local-then-assign.
            new_tile_idx = cute.arch.grid_dim()[0]
            new_split_idx = Int32(0)
            if clc_work.is_valid_tile:
                new_tile_idx = clc_work.tile_idx[0]
                if const_expr(self.params.is_split_kv):
                    new_split_idx = clc_work.tile_idx[1]
            self._tile_idx = new_tile_idx
            self._split_idx = new_split_idx
        if const_expr(self.params.use_global_q_direct):
            return self._direct_global_q_coord_map()
        return self._varlen_coord_map()

    def prefetch_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.prefetch_next_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.consumer_wait(loc=loc, ip=ip)
            work = self.get_current_work()
            self.clc.consumer_release(loc=loc, ip=ip)
            if const_expr(self.params.clc_sync_after_work):
                cute.arch.barrier()
            return work
        if const_expr(self.params.is_persistent):
            self._tile_idx += cute.arch.grid_dim()[0]
        self._is_first_block = False
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.producer_tail(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self.clc]
        for obj in objs:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self.clc]
        for obj, n_items in zip(objs, self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*obj_list, loc=self._loc)


class Sm100FmhaStaticTileSchedulerParams:
    """Parameters for the SM100 FMHA static tile scheduler."""

    def __init__(
        self,
        is_persistent: bool,
        problem_shape_mbh: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        self.is_persistent = is_persistent
        self.problem_shape_mbh = problem_shape_mbh
        self._loc = loc
        self._ip = ip

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.problem_shape_mbh]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.problem_shape_mbh], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return Sm100FmhaStaticTileSchedulerParams(
            self.is_persistent, *(tuple(obj_list)), loc=self._loc
        )


class Sm100FmhaStaticTileScheduler:
    """Static scheduler for SM100 FMHA kernels over an (M, B, H) problem."""

    def __init__(
        self,
        params: Sm100FmhaStaticTileSchedulerParams,
        current_work_linear_idx: Int32,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        self._params = params
        self._blk_coord = blk_coord
        self._grid_shape = grid_shape
        self._is_persistent = params.is_persistent
        self._current_work_linear_idx = current_work_linear_idx
        self._problem_shape_mbh = cute.make_layout(params.problem_shape_mbh, loc=loc, ip=ip)
        self._num_blocks = cute.size(self._problem_shape_mbh, loc=loc, ip=ip)
        self._is_first_block = True
        self.num_persistent_sm = cute.size(grid_shape, loc=loc, ip=ip)
        self._loc = loc
        self._ip = ip

    @staticmethod
    def get_grid_shape(
        params: Sm100FmhaStaticTileSchedulerParams,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        if params.is_persistent:
            hardware_info = cutlass.utils.HardwareInfo()
            sm_count = hardware_info.get_device_multiprocessor_count()
            return (
                cutlass.min(sm_count, cute.size(params.problem_shape_mbh, loc=loc, ip=ip)),
                1,
                1,
            )
        return params.problem_shape_mbh

    @staticmethod
    def check_valid_work_for_seqlen_q(
        q_tiler: int,
        current_idx: Int32,
        seqlen_q: Int32,
    ) -> cutlass.Boolean:
        return current_idx * q_tiler < seqlen_q

    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        is_valid = (
            self._current_work_linear_idx < self._num_blocks
            if self._is_persistent
            else self._is_first_block
        )
        if self._is_persistent:
            blk_coord = self._problem_shape_mbh.get_hier_coord(
                self._current_work_linear_idx, loc=loc, ip=ip
            )
        else:
            blk_coord = self._blk_coord
        return WorkTileInfo((blk_coord[0], 0, (blk_coord[1], blk_coord[2])), is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
        if self._is_persistent:
            self._current_work_linear_idx += advance_count * self.num_persistent_sm
        self._is_first_block = False
        return self.get_current_work()

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self._params, self._current_work_linear_idx, self._blk_coord, self._grid_shape]:
            obj_values = cutlass.extract_mlir_values(obj)
            values.extend(obj_values)
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        new_objs = []
        for obj, n_items in zip(
            [self._params, self._current_work_linear_idx, self._blk_coord, self._grid_shape],
            self._values_pos,
        ):
            new_objs.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return Sm100FmhaStaticTileScheduler(new_objs[0], new_objs[1], new_objs[2], new_objs[3])


def compute_sm100_fmha_grid(
    o_shape: cute.Shape,
    cta_tiler: Tuple[int, int, int],
    is_persistent: bool,
) -> Tuple[Sm100FmhaStaticTileSchedulerParams, Tuple[int, int, int]]:
    """Compute static SM100 FMHA scheduler parameters."""
    tile_sched_params = Sm100FmhaStaticTileSchedulerParams(
        is_persistent,
        (
            cute.ceil_div(cute.size(o_shape[0]), cta_tiler[0]),
            cute.size(o_shape[2][0]),
            cute.size(o_shape[2][1]),
        ),
    )
    grid = Sm100FmhaStaticTileScheduler.get_grid_shape(tile_sched_params)
    return tile_sched_params, grid


class Sm100FmhaClcDynamicTileSchedulerParams:
    """Parameters for the SM100 FMHA CLC dynamic tile scheduler."""

    def __init__(
        self,
        problem_shape_mbh: cute.Shape,
        cluster_shape_mnk: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        self.problem_shape_mbh = problem_shape_mbh
        self._cluster_shape_mnk = cluster_shape_mnk
        self.cluster_shape_mn = cluster_shape_mnk[:2]
        self._loc = loc
        self._ip = ip
        self.problem_shape_ntile_mnl = (
            problem_shape_mbh[0],
            1,
            problem_shape_mbh[1] * problem_shape_mbh[2],
        )
        self.problem_layout_ncluster_mnl = cute.make_layout(
            cute.ceil_div(self.problem_shape_ntile_mnl, cluster_shape_mnk[:2]),
            loc=loc,
            ip=ip,
        )

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.problem_shape_mbh, self._cluster_shape_mnk]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        values_copy = list(values)
        for obj, n_items in zip(
            [self.problem_shape_mbh, self._cluster_shape_mnk],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values_copy[:n_items]))
            values_copy = values_copy[n_items:]
        return Sm100FmhaClcDynamicTileSchedulerParams(*(tuple(obj_list)), loc=self._loc)

    def get_grid_shape(self, *, loc=None, ip=None) -> Tuple[int, int, int]:
        return cute.round_up(self.problem_shape_ntile_mnl, self._cluster_shape_mnk)

    def clc_hw_params(self) -> ClcDynamicPersistentTileSchedulerParams:
        return ClcDynamicPersistentTileSchedulerParams(
            problem_shape_ntile_mnl=self.problem_shape_ntile_mnl,
            cluster_shape_mnk=self._cluster_shape_mnk,
        )


class Sm100FmhaClcDynamicTileScheduler:
    """CLC dynamic scheduler for SM100 FMHA kernels over an (M, B, H) problem."""

    def __init__(
        self,
        params: Sm100FmhaClcDynamicTileSchedulerParams,
        cta_id_in_cluster: cute.Coord,
        num_tiles_executed: Int32,
        clc_response_ptr: cute.Pointer,
        block_idx: Tuple,
        clc: ClcState = None,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self.cta_id_in_cluster = cta_id_in_cluster
        self._num_tiles_executed = num_tiles_executed
        self._clc_response_ptr = clc_response_ptr
        self._block_idx = block_idx
        self.clc = clc
        self._loc = loc
        self._ip = ip

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.cta_id_in_cluster)
        values.extend(cutlass.extract_mlir_values(self._num_tiles_executed))
        values.extend(cutlass.extract_mlir_values(self._clc_response_ptr))
        values.extend(cutlass.extract_mlir_values(self._block_idx))
        if self.clc is not None:
            values.extend(cutlass.extract_mlir_values(self.clc))
        return values

    def __new_from_mlir_values__(self, values):
        new_cta_id_in_cluster = cutlass.new_from_mlir_values(self.cta_id_in_cluster, values[0:3])
        new_num_tiles_executed = cutlass.new_from_mlir_values(self._num_tiles_executed, [values[3]])
        new_clc_response_ptr = cutlass.new_from_mlir_values(self._clc_response_ptr, [values[4]])
        new_block_idx = cutlass.new_from_mlir_values(self._block_idx, values[5:8])
        new_clc = None
        if self.clc is not None:
            new_clc = cutlass.new_from_mlir_values(self.clc, values[8:])
        return Sm100FmhaClcDynamicTileScheduler(
            self.params,
            new_cta_id_in_cluster,
            new_num_tiles_executed,
            new_clc_response_ptr,
            new_block_idx,
            new_clc,
        )

    @staticmethod
    def create(
        params: Sm100FmhaClcDynamicTileSchedulerParams,
        block_idx: Tuple,
        grid_dim: Tuple,
        clc_response_ptr: cute.Pointer,
        clc: ClcState = None,
        *,
        loc=None,
        ip=None,
    ):
        del grid_dim
        bidx, bidy, _ = block_idx
        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )
        return Sm100FmhaClcDynamicTileScheduler(
            params,
            cta_id_in_cluster,
            Int32(0),
            clc_response_ptr,
            block_idx,
            clc,
        )

    @staticmethod
    def get_grid_shape(
        params: Sm100FmhaClcDynamicTileSchedulerParams,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[int, int, int]:
        return params.get_grid_shape(loc=loc, ip=ip)

    def work_tile_info_from_clc_response(self, result_addr: cute.Pointer, *, loc=None, ip=None):
        m_idx, n_idx, l_idx, vld = cute.arch.clc_response(result_addr, loc=loc, ip=ip)
        del n_idx
        cute.arch.fence_proxy("async.shared", space="cta")
        hid = l_idx % self.params.problem_shape_mbh[2]
        bid = l_idx // self.params.problem_shape_mbh[2]
        cta_idx_in_cluster, _, _ = self.cta_id_in_cluster
        return WorkTileInfo((m_idx + cta_idx_in_cluster, 0, (bid, hid)), vld)

    def get_current_work(self, *, loc=None, ip=None):
        return self.work_tile_info_from_clc_response(self._clc_response_ptr, loc=loc, ip=ip)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        bidx, _, bidz = self._block_idx
        hid = bidz % self.params.problem_shape_mbh[2]
        bid = bidz // self.params.problem_shape_mbh[2]
        return WorkTileInfo((bidx, 0, (bid, hid)), True)

    def advance_to_next_work(self, *, loc=None, ip=None):
        self.clc.consumer_wait(loc=loc, ip=ip)
        work = self.get_current_work(loc=loc, ip=ip)
        self.clc.consumer_release(loc=loc, ip=ip)
        self._num_tiles_executed += Int32(1)
        return work

    def prefetch_next_work(self, *, loc=None, ip=None):
        self.clc.prefetch_next_work(loc=loc, ip=ip)

    def producer_tail(self, *, loc=None, ip=None):
        self.clc.producer_tail(loc=loc, ip=ip)

    @property
    def num_tiles_executed(self) -> Int32:
        return self._num_tiles_executed


def compute_sm100_fmha_grid_clc(
    o_shape: cute.Shape,
    cta_tiler: Tuple[int, int, int],
    cluster_shape_mnk: Tuple[int, int, int],
) -> Tuple[Sm100FmhaClcDynamicTileSchedulerParams, Tuple[int, int, int]]:
    """Compute CLC dynamic SM100 FMHA scheduler parameters."""
    problem_shape_mbh = (
        cute.ceil_div(cute.size(o_shape[0]), cta_tiler[0]),
        cute.size(o_shape[2][0]),
        cute.size(o_shape[2][1]),
    )
    tile_sched_params = Sm100FmhaClcDynamicTileSchedulerParams(problem_shape_mbh, cluster_shape_mnk)
    grid = Sm100FmhaClcDynamicTileScheduler.get_grid_shape(tile_sched_params)
    return tile_sched_params, grid


def make_sm100_thread_cooperative_group(size: int):
    return cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, size)


SM100_TMEM_CAPACITY_COLUMNS = 512
