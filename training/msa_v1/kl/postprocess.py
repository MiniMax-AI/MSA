"""Vectorized BF16 gradient postprocess kernels for MSA v1 KL backward."""

from __future__ import annotations

from typing import Optional

import cutlass
from cutlass import Boolean, Float32, Int32, Int64, const_expr
import cutlass.cute as cute
import cuda.bindings.driver as cuda

from msa_v1._common import copy_utils
from msa_v1._common.cute_dsl_utils import assume_tensor_aligned


class SparseKlDkiSplitZeroSm100:
    """Clear only physical FP32 dKI tiles with multiple work owners."""

    def __init__(self, num_threads: int = 256, max_ctas: int = 160) -> None:
        if num_threads != 256:
            raise ValueError("SparseKlDkiSplitZeroSm100 expects 256 threads")
        self.num_threads = num_threads
        self.max_ctas = max_ctas
        self.block_size = 128
        self.head_dim = 128

    @cute.jit
    def __call__(
        self,
        mDKIAccum: cute.Tensor,
        mSplitIndices: cute.Tensor,
        mSplitCount: cute.Tensor,
        stream: cuda.CUstream = None,
    ) -> None:
        if const_expr(mDKIAccum.element_type != Float32):
            raise TypeError("dKI accumulator must be Float32")
        mDKIAccum = assume_tensor_aligned(mDKIAccum)
        self.kernel(mDKIAccum, mSplitIndices, mSplitCount).launch(
            grid=(cutlass.min(mSplitIndices.shape[0], self.max_ctas), 1, 1),
            block=(self.num_threads, 1, 1),
            smem=0,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mDKIAccum: cute.Tensor,
        mSplitIndices: cute.Tensor,
        mSplitCount: cute.Tensor,
    ) -> None:
        tidx = cute.arch.thread_idx()[0]
        cta_idx = cute.arch.block_idx()[0]
        grid_dim = cute.arch.grid_dim()[0]
        split_count = cutlass.min(
            mSplitCount[Int32(0)], Int32(mSplitIndices.shape[0])
        )
        split_idx = cta_idx
        tile_vectors = Int32(self.block_size * self.head_dim // 4)
        zero = Float32(0.0)
        while split_idx < split_count:
            physical_block = mSplitIndices[split_idx]
            tile_base = Int64(physical_block) * Int64(
                self.block_size * self.head_dim
            )
            vector_idx = tidx
            while vector_idx < tile_vectors:
                dst_ptr = cute.make_ptr(
                    Float32,
                    mDKIAccum.iterator.toint()
                    + (tile_base + Int64(vector_idx * Int32(4))) * Int64(4),
                    mem_space=mDKIAccum.iterator.memspace,
                    assumed_align=16,
                )
                copy_utils.stg_128(dst_ptr, zero, zero, zero, zero)
                vector_idx += Int32(self.num_threads)
            split_idx += grid_dim


class SparseKlDqiPostprocessSm100:
    """Convert row-local fake-D FP32 dQI accumulation to natural BF16."""

    def __init__(self, num_threads: int = 128) -> None:
        if num_threads != 128:
            raise ValueError("SparseKlDqiPostprocessSm100 expects 128 threads")
        self.num_threads = num_threads
        self.threads_per_row = 16
        self.rows_per_cta = num_threads // self.threads_per_row
        self.head_dim = 128

    @cute.jit
    def __call__(
        self,
        mDQIAccum: cute.Tensor,
        mDQI: cute.Tensor,
        grad_scale: Float32,
        stream: cuda.CUstream = None,
    ) -> None:
        if const_expr(mDQIAccum.element_type != Float32):
            raise TypeError("dQI accumulator must be Float32")
        if const_expr(mDQI.element_type != cutlass.BFloat16):
            raise TypeError("dQI output must be BFloat16")
        if const_expr(cute.rank(mDQIAccum.shape) != 3 or cute.rank(mDQI.shape) != 3):
            raise ValueError("dQI tensors must be rank 3")

        mDQIAccum, mDQI = [assume_tensor_aligned(t) for t in (mDQIAccum, mDQI)]
        total_q = cute.size(mDQI.shape[0])
        num_heads = cute.size(mDQI.shape[1])
        self.kernel(mDQIAccum, mDQI, grad_scale).launch(
            grid=(cute.ceil_div(total_q, self.rows_per_cta), num_heads, 1),
            block=[self.num_threads, 1, 1],
            smem=0,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mDQIAccum: cute.Tensor,
        mDQI: cute.Tensor,
        grad_scale: Float32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_q, head_idx, _ = cute.arch.block_idx()
        row_in_cta = tidx // Int32(self.threads_per_row)
        lane_in_row = tidx % Int32(self.threads_per_row)
        q_idx = block_q * Int32(self.rows_per_cta) + row_in_cta
        total_q = Int32(cute.size(mDQI.shape[0]))

        if q_idx < total_q:
            num_heads = Int64(cute.size(mDQI.shape[1]))
            row_offset = (
                Int64(q_idx) * num_heads + Int64(head_idx)
            ) * Int64(self.head_dim)
            src_memspace = mDQIAccum.iterator.memspace
            fake_col_lo = lane_in_row * Int32(4)
            fake_col_hi = fake_col_lo + Int32(64)
            src_ptr_lo = cute.make_ptr(
                Float32,
                mDQIAccum.iterator.toint()
                + (row_offset + Int64(fake_col_lo)) * Int64(4),
                mem_space=src_memspace,
                assumed_align=16,
            )
            src_ptr_hi = cute.make_ptr(
                Float32,
                mDQIAccum.iterator.toint()
                + (row_offset + Int64(fake_col_hi)) * Int64(4),
                mem_space=src_memspace,
                assumed_align=16,
            )
            values_lo = cute.make_tensor(
                src_ptr_lo, cute.make_layout((4,), stride=(1,))
            ).load()
            values_hi = cute.make_tensor(
                src_ptr_hi, cute.make_layout((4,), stride=(1,))
            ).load()

            # Four fake vectors form two natural 8-element output vectors.
            natural_group = lane_in_row
            chunk = natural_group // Int32(2)
            half = natural_group % Int32(2)
            warp_lane = tidx % Int32(cute.arch.WARP_SIZE)
            row_lane_base = (warp_lane // Int32(16)) * Int32(16)
            source_base = row_lane_base + (chunk % Int32(4)) * Int32(4)
            row_mask = Int32(0xFFFF) << row_lane_base
            output = cute.make_rmem_tensor((8,), Float32)
            for rank in cutlass.range_constexpr(4):
                source_lane = source_base + Int32(rank)
                lo_0 = cute.arch.shuffle_sync(
                    values_lo[0], source_lane, mask=row_mask
                )
                lo_1 = cute.arch.shuffle_sync(
                    values_lo[1], source_lane, mask=row_mask
                )
                lo_2 = cute.arch.shuffle_sync(
                    values_lo[2], source_lane, mask=row_mask
                )
                lo_3 = cute.arch.shuffle_sync(
                    values_lo[3], source_lane, mask=row_mask
                )
                hi_0 = cute.arch.shuffle_sync(
                    values_hi[0], source_lane, mask=row_mask
                )
                hi_1 = cute.arch.shuffle_sync(
                    values_hi[1], source_lane, mask=row_mask
                )
                hi_2 = cute.arch.shuffle_sync(
                    values_hi[2], source_lane, mask=row_mask
                )
                hi_3 = cute.arch.shuffle_sync(
                    values_hi[3], source_lane, mask=row_mask
                )
                if chunk < Int32(4):
                    if half == Int32(0):
                        output[rank * 2] = lo_0 * grad_scale
                        output[rank * 2 + 1] = lo_1 * grad_scale
                    else:
                        output[rank * 2] = lo_2 * grad_scale
                        output[rank * 2 + 1] = lo_3 * grad_scale
                else:
                    if half == Int32(0):
                        output[rank * 2] = hi_0 * grad_scale
                        output[rank * 2 + 1] = hi_1 * grad_scale
                    else:
                        output[rank * 2] = hi_2 * grad_scale
                        output[rank * 2 + 1] = hi_3 * grad_scale

            real_col = natural_group * Int32(8)
            dst_ptr = cute.make_ptr(
                cutlass.BFloat16,
                mDQI.iterator.toint()
                + (row_offset + Int64(real_col)) * Int64(2),
                mem_space=mDQI.iterator.memspace,
                assumed_align=16,
            )
            copy_utils.stg_128_bf16(
                dst_ptr,
                output[0],
                output[1],
                output[2],
                output[3],
                output[4],
                output[5],
                output[6],
                output[7],
            )


class SparseKlDkiPostprocessSm100:
    """Finalize padded FP32 dKI or generate zero-owner BF16 output."""

    def __init__(self, num_threads: int = 128) -> None:
        if num_threads != 128:
            raise ValueError("SparseKlDkiPostprocessSm100 expects 128 threads")
        self.num_threads = num_threads
        self.threads_per_row = 16
        self.rows_per_cta = num_threads // self.threads_per_row
        self.block_size = 128
        self.head_dim = 128

    @cute.jit
    def __call__(
        self,
        mDKIAccum: cute.Tensor,
        mDKI: cute.Tensor,
        mDkiOwnerCounts: cute.Tensor,
        grad_scale: Float32,
        mCuSeqlensK: cute.Tensor,
        mFragmentIndices: Optional[cute.Tensor],
        max_seqlen_k: Int32,
        stream: cuda.CUstream = None,
    ) -> None:
        if const_expr(mDKIAccum.element_type != Float32):
            raise TypeError("dKI accumulator must be Float32")
        if const_expr(mDKI.element_type != cutlass.BFloat16):
            raise TypeError("dKI output must be BFloat16")
        if const_expr(cute.rank(mDKIAccum.shape) != 3 or cute.rank(mDKI.shape) != 3):
            raise ValueError("dKI tensors must be rank 3")

        mDKIAccum, mDKI = [assume_tensor_aligned(t) for t in (mDKIAccum, mDKI)]
        self.kernel(
            mDKIAccum,
            mDKI,
            mDkiOwnerCounts,
            grad_scale,
            mCuSeqlensK,
            mFragmentIndices,
        ).launch(
            grid=(
                cute.ceil_div(max_seqlen_k, self.block_size),
                mCuSeqlensK.shape[0] - 1,
                1,
            ),
            block=[self.num_threads, 1, 1],
            smem=0,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mDKIAccum: cute.Tensor,
        mDKI: cute.Tensor,
        mDkiOwnerCounts: cute.Tensor,
        grad_scale: Float32,
        mCuSeqlensK: cute.Tensor,
        mFragmentIndices: Optional[cute.Tensor],
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_k, batch_idx, _ = cute.arch.block_idx()
        row_in_cta = tidx // Int32(self.threads_per_row)
        lane_in_row = tidx % Int32(self.threads_per_row)
        physical_batch_idx = (
            batch_idx
            if const_expr(mFragmentIndices is None)
            else mFragmentIndices[batch_idx]
        )
        is_group_end = Boolean(True)
        if const_expr(mFragmentIndices is not None):
            batch = mFragmentIndices.shape[0]
            if batch_idx + Int32(1) < batch:
                is_group_end = (
                    mFragmentIndices[batch_idx + Int32(1)]
                    != physical_batch_idx
                )
        physical_offset_k = mCuSeqlensK[physical_batch_idx]
        seqlen_k = Int32(0)
        if is_group_end:
            seqlen_k = mCuSeqlensK[batch_idx + Int32(1)] - physical_offset_k
        rows_left = seqlen_k - block_k * Int32(self.block_size)
        valid_rows = cutlass.min(
            cutlass.max(rows_left, Int32(0)), Int32(self.block_size)
        )
        if valid_rows > Int32(0):
            padded_offset_k = (
                physical_offset_k
                + physical_batch_idx * Int32(self.block_size)
            ) // Int32(self.block_size) * Int32(self.block_size)
            physical_block = (
                padded_offset_k // Int32(self.block_size) + block_k
            )
            owner_count = mDkiOwnerCounts[physical_block]
            real_col = lane_in_row * Int32(8)
            zero = Float32(0.0)
            for row_pass in cutlass.range_constexpr(16):
                row = Int32(row_pass * self.rows_per_cta) + row_in_cta
                if row < valid_rows:
                    accum_row = padded_offset_k + block_k * Int32(
                        self.block_size
                    ) + row
                    output_row = physical_offset_k + block_k * Int32(
                        self.block_size
                    ) + row
                    values = cute.make_rmem_tensor((8,), Float32)
                    values.fill(zero)
                    if owner_count == Int32(1):
                        workspace_row_offset = Int64(
                            accum_row * Int32(self.head_dim * 2)
                        )
                        src_ptr = cute.make_ptr(
                            cutlass.BFloat16,
                            mDKIAccum.iterator.toint()
                            + (
                                workspace_row_offset + Int64(real_col)
                            )
                            * Int64(2),
                            mem_space=mDKIAccum.iterator.memspace,
                            assumed_align=16,
                        )
                        src = cute.make_tensor(
                            src_ptr, cute.make_layout((8,), stride=(1,))
                        ).load()
                        for value_idx in cutlass.range_constexpr(8):
                            values[value_idx] = src[value_idx].to(Float32)
                    elif owner_count > Int32(1):
                        row_offset = Int64(accum_row * Int32(self.head_dim))
                        src_ptr = cute.make_ptr(
                            Float32,
                            mDKIAccum.iterator.toint()
                            + (row_offset + Int64(real_col)) * Int64(4),
                            mem_space=mDKIAccum.iterator.memspace,
                            assumed_align=16,
                        )
                        src_lo = cute.make_tensor(
                            src_ptr, cute.make_layout((4,), stride=(1,))
                        ).load()
                        src_hi = cute.make_tensor(
                            src_ptr + 4, cute.make_layout((4,), stride=(1,))
                        ).load()
                        for value_idx in cutlass.range_constexpr(4):
                            values[value_idx] = src_lo[value_idx] * grad_scale
                            values[value_idx + 4] = (
                                src_hi[value_idx] * grad_scale
                            )
                    dst_offset = Int64(output_row * Int32(self.head_dim))
                    dst_ptr = cute.make_ptr(
                        cutlass.BFloat16,
                        mDKI.iterator.toint()
                        + (dst_offset + Int64(real_col)) * Int64(2),
                        mem_space=mDKI.iterator.memspace,
                        assumed_align=16,
                    )
                    copy_utils.stg_128_bf16(
                        dst_ptr,
                        values[0], values[1], values[2], values[3],
                        values[4], values[5], values[6], values[7],
                    )


__all__ = [
    "SparseKlDkiSplitZeroSm100",
    "SparseKlDkiPostprocessSm100",
    "SparseKlDqiPostprocessSm100",
]
