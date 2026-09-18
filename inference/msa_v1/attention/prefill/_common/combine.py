"""Sparse forward combine kernel and public launcher.

The epilogue converts the partial-output STG.128 fake layout to the real
output-column order.
"""

import math
from functools import partial
from typing import Optional, Type

import cutlass
from cutlass import Boolean, Float32, Int32, Int64, const_expr
import cutlass.cute as cute
from cutlass.cutlass_dsl import BaseDSL
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync
import cuda.bindings.driver as cuda
import torch

from inference.msa_v1.attention.prefill._common.dsl import utils
from inference.msa_v1.aot_cache import save_aot, try_load_aot
from inference.msa_v1.attention.prefill._common.dsl.compile_utils import compile_with_timing
from inference.msa_v1.attention.prefill._common.dsl.cute_dsl_utils import assume_tensor_aligned, torch2cute_dtype_map
from inference.msa_v1.attention.prefill._common.dsl.pack_gqa import PackGQAComb
from inference.msa_v1.attention.prefill._common.dsl.seqlen_info import SeqlenInfo
from inference.msa_v1.attention.prefill._common.dsl.tma_utils import stg128_half_fake_col_to_real_col


class SparseAttentionForwardCombine:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        head_dim: int,
        tile_m: int = 8,
        k_block_size: int = 64,
        topk: int = 16,
        num_threads: int = 256,
        stages: int = 4,
        use_pdl: bool = False,
        min_blocks_per_mp: int = 0,
        raw_partial_stats: bool = False,
    ):
        """
        Forward combine kernel for split attention computation.

        :param dtype: output data type
        :param dtype_partial: partial accumulation data type
        :param head_dim: head dimension
        :param tile_m: m block size
        :param k_block_size: k block size
        :param topk: exact number of split partials
        :param num_threads: number of threads
        :param stages: number of pipeline stages
        """
        self.dtype = dtype
        self.dtype_partial = dtype_partial
        self.head_dim = head_dim
        self.tile_m = tile_m
        self.k_block_size = k_block_size
        self.topk = topk
        self.num_threads = num_threads
        self.is_even_k = head_dim % k_block_size == 0
        self.stages = stages
        arch = BaseDSL._get_dsl().get_arch_enum()
        rubin_arch = getattr(arch.__class__, "sm_107", None)
        self.is_rubin = rubin_arch is not None and arch.is_family_of(rubin_arch)
        self.use_pdl = use_pdl
        self.min_blocks_per_mp = min_blocks_per_mp
        self.raw_partial_stats = raw_partial_stats
        self.use_tma_partial = (
            raw_partial_stats
            and dtype == cutlass.BFloat16
            and dtype_partial == cutlass.BFloat16
            and head_dim == 128
            and tile_m in (32, 64)
            and k_block_size == 128
            and topk == 16
            and num_threads == tile_m * 4
            and stages == 2
        )

    @staticmethod
    def can_implement(
        dtype,
        dtype_partial,
        head_dim,
        tile_m,
        k_block_size,
        topk,
        num_threads,
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters."""
        if dtype not in [cutlass.Float16, cutlass.BFloat16, cutlass.Float32]:
            return False
        if dtype_partial not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if tile_m % 8 != 0:
            return False
        if topk > 256:
            return False
        if (tile_m * topk) % num_threads != 0:
            return False
        return True

    def _setup_attributes(self):
        # GMEM copy setup for O partial
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype_partial.width
        assert self.k_block_size % async_copy_elems == 0

        k_block_gmem = (
            128 if self.k_block_size % 128 == 0 else (64 if self.k_block_size % 64 == 0 else 32)
        )
        gmem_threads_per_row = k_block_gmem // async_copy_elems
        assert self.num_threads % gmem_threads_per_row == 0

        # Async copy atom for O partial load
        atom_async_copy_partial = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype_partial,
            num_bits_per_copy=universal_copy_bits,
        )
        tOpartial_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        vOpartial_layout = cute.make_layout((1, async_copy_elems))
        self.gmem_tiled_copy_O_partial = cute.make_tiled_copy_tv(
            atom_async_copy_partial, tOpartial_layout, vOpartial_layout
        )

        # GMEM copy setup for final O (use universal copy for store).
        # Keep output stores independent from the partial-load copy atom.
        output_copy_elems = universal_copy_bits // self.dtype.width
        assert self.k_block_size % output_copy_elems == 0
        gmem_threads_per_row_o = k_block_gmem // output_copy_elems
        assert self.num_threads % gmem_threads_per_row_o == 0
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tO_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row_o, gmem_threads_per_row_o),
            order=(1, 0),
        )
        vO_layout = cute.make_layout((1, output_copy_elems))
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy,
            tO_layout,
            vO_layout,
        )
        # LSE copy setup with async copy (alignment = 1)
        lse_copy_bits = Float32.width  # 1 element per copy, width is in bits
        m_block_smem = (
            128
            if self.tile_m % 128 == 0
            else (
                64
                if self.tile_m % 64 == 0
                else (32 if self.tile_m % 32 == 0 else (16 if self.tile_m % 16 == 0 else 8))
            )
        )
        gmem_threads_per_row_lse = m_block_smem
        assert self.num_threads % gmem_threads_per_row_lse == 0

        # Async copy atom for LSE load
        atom_async_copy_lse = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            Float32,
            num_bits_per_copy=lse_copy_bits,
        )
        tLSE_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row_lse, gmem_threads_per_row_lse),
            order=(1, 0),
        )
        vLSE_layout = cute.make_layout(1)
        self.gmem_tiled_copy_LSE = cute.make_tiled_copy_tv(
            atom_async_copy_lse, tLSE_layout, vLSE_layout
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory
        # ///////////////////////////////////////////////////////////////////////////////

        # Shared memory to register copy for LSE
        self.smem_threads_per_col_lse = self.num_threads // m_block_smem
        assert 32 % self.smem_threads_per_col_lse == 0  # Must divide warp size

        s2r_layout_atom_lse = cute.make_ordered_layout(
            (self.smem_threads_per_col_lse, self.num_threads // self.smem_threads_per_col_lse),
            order=(0, 1),
        )
        self.s2r_tiled_copy_LSE = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32),
            s2r_layout_atom_lse,
            cute.make_layout(1),
        )

        # LSE shared memory layout with swizzling to avoid bank conflicts
        # This works for kBlockMSmem = 8, 16, 32, 64, 128, no bank conflicts
        if const_expr(m_block_smem == 8):
            smem_lse_swizzle = cute.make_swizzle(5, 0, 5)
        elif const_expr(m_block_smem == 16):
            smem_lse_swizzle = cute.make_swizzle(4, 0, 4)
        else:
            smem_lse_swizzle = cute.make_swizzle(3, 2, 3)
        lse_atom_splits = min(self.topk, 8)
        smem_layout_atom_lse = cute.make_composed_layout(
            smem_lse_swizzle,
            0,
            cute.make_ordered_layout((lse_atom_splits, m_block_smem), order=(1, 0)),
        )
        self.smem_layout_lse = cute.tile_to_shape(
            smem_layout_atom_lse, (self.topk, self.tile_m), (0, 1)
        )

        # O_partial staging layout. The target raw-BF16 specialization uses a
        # row-major tile that is shared by the TMA and consumer-thread views.
        if const_expr(self.use_tma_partial):
            self.smem_layout_o = cute.make_ordered_layout(
                (self.tile_m, self.k_block_size, self.stages),
                order=(1, 0, 2),
            )
            self.smem_layout_o_tma = cute.make_ordered_layout(
                (self.k_block_size, self.tile_m, self.stages),
                order=(0, 1, 2),
            )
        else:
            smem_layout_atom_o = _get_cpasync_smem_layout_atom(
                self.dtype_partial, self.k_block_size
            )
            self.smem_layout_o = cute.tile_to_shape(
                smem_layout_atom_o,
                (self.tile_m, self.k_block_size, self.stages),
                (0, 1, 2),
            )
            self.smem_layout_o_tma = None

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor] = None,
        mLSE_temperature_partial: Optional[cute.Tensor] = None,
        mLSE_temperature: Optional[cute.Tensor] = None,
        cu_seqlens: Optional[cute.Tensor] = None,
        seqused: Optional[cute.Tensor] = None,
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,
        varlen_batch_idx: Optional[cute.Tensor] = None,
        semaphore_to_reset: Optional[cute.Tensor] = None,
        mSplitCounts: Optional[cute.Tensor] = None,
        mOutputScale: Optional[cute.Tensor] = None,
        qhead_per_kvhead: Int32 = Int32(1),
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # Type checking
        if const_expr(not (mO_partial.element_type == self.dtype_partial)):
            raise TypeError("O partial tensor must match dtype_partial")
        if const_expr(not (mO.element_type == self.dtype)):
            raise TypeError("O tensor must match dtype")
        if const_expr(mLSE_partial.element_type not in [Float32]):
            raise TypeError("LSE partial tensor must be Float32")
        if const_expr(mLSE is not None and mLSE.element_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(
            mLSE_temperature_partial is not None
            and mLSE_temperature_partial.element_type not in [Float32]
        ):
            raise TypeError("temperature LSE partial tensor must be Float32")
        if const_expr(mLSE_temperature is not None and mLSE_temperature.element_type not in [Float32]):
            raise TypeError("temperature LSE tensor must be Float32")
        if const_expr((mLSE_temperature_partial is None) != (mLSE_temperature is None)):
            raise ValueError(
                "temperature LSE partial and output tensors must either both be provided or both be None"
            )

        # Shape validation - input tensors are in user format, need to be converted to kernel format
        if const_expr(len(mO_partial.shape) not in [4, 5]):
            raise ValueError(
                "O partial tensor must have 4 or 5 dimensions: (num_splits, batch, seqlen, nheads, headdim) or (num_splits, total_q, nheads, headdim)"
            )
        if const_expr(self.raw_partial_stats):
            if const_expr(len(mLSE_partial.shape) != 4):
                raise ValueError(
                    "Raw partial stats must have shape "
                    "(num_splits, total_q, nheads, 2)"
                )
            if const_expr(cu_seqlens is None):
                raise ValueError("Raw partial stats currently require the varlen combine path")
            if const_expr(mLSE_temperature_partial is not None):
                raise ValueError("Raw partial stats do not support independent temperature LSE")
        elif const_expr(len(mLSE_partial.shape) not in [3, 4]):
            raise ValueError(
                "LSE partial tensor must have 3 or 4 dimensions: (num_splits, batch, seqlen, nheads) or (num_splits, total_q, nheads)"
            )
        if const_expr(len(mO.shape) not in [3, 4]):
            raise ValueError(
                "O tensor must have 3 or 4 dimensions: (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim)"
            )
        if const_expr(mLSE is not None and len(mLSE.shape) not in [2, 3]):
            raise ValueError(
                "LSE tensor must have 2 or 3 dimensions: (batch, seqlen, nheads) or (total_q, nheads)"
            )
        if const_expr(mLSE_temperature_partial is not None and len(mLSE_temperature_partial.shape) not in [3, 4]):
            raise ValueError(
                "temperature LSE partial tensor must have 3 or 4 dimensions: "
                "(num_splits, batch, seqlen, nheads) or (num_splits, total_q, nheads)"
            )
        if const_expr(mLSE_temperature is not None and len(mLSE_temperature.shape) not in [2, 3]):
            raise ValueError(
                "temperature LSE tensor must have 2 or 3 dimensions: "
                "(batch, seqlen, nheads) or (total_q, nheads)"
            )
        if const_expr(mSplitCounts is not None):
            if const_expr(mSplitCounts.element_type not in [Int32]):
                raise TypeError("split_counts tensor must be Int32")
            if const_expr(cu_seqlens is not None):
                if const_expr(len(mSplitCounts.shape) != 2):
                    raise ValueError("varlen split_counts tensor must have shape (total_q, nheads_kv)")
            elif const_expr(len(mSplitCounts.shape) != 3):
                raise ValueError("batched split_counts tensor must have shape (batch, seqlen, nheads_kv)")
        if const_expr(mOutputScale is not None and mOutputScale.element_type not in [Float32]):
            raise TypeError("output_scale tensor must be Float32")

        mO_partial, mO = [assume_tensor_aligned(t) for t in (mO_partial, mO)]
        # (num_splits, b, seqlen, h, d) -> (seqlen, d, num_splits, h, b)
        # or (num_splits, total_q, h, d) -> (total_q, d, num_splits, h)
        O_partial_layout_transpose = (
            [2, 4, 0, 3, 1] if const_expr(cu_seqlens is None) else [1, 3, 0, 2]
        )
        # (b, seqlen, h, d) -> (seqlen, d, h, b) or (total_q, h, d) -> (total_q, d, h)
        mO_partial = cute.make_tensor(
            mO_partial.iterator, cute.select(mO_partial.layout, mode=O_partial_layout_transpose)
        )
        O_layout_transpose = [1, 3, 2, 0] if const_expr(cu_seqlens is None) else [0, 2, 1]
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        # (num_splits, b, h, seqlen) -> (seqlen, num_splits, h, b)
        # Input is pre-transposed: [topK, B, Hq, Sq] with Sq innermost for K2-friendly reads.
        # or (num_splits, total_q, h) -> (total_q, num_splits, h)
        LSE_partial_layout_transpose = [3, 0, 2, 1] if const_expr(cu_seqlens is None) else [1, 0, 2]
        mRawStats = None
        mRowSum_partial = None
        if const_expr(self.raw_partial_stats):
            # [split, q, head, stat] -> [q, split, head, stat]. Keep the
            # interleaved pair ABI while exposing scalar component views to K2.
            mPartialStats = cute.make_tensor(
                mLSE_partial.iterator,
                cute.select(mLSE_partial.layout, mode=[1, 0, 2, 3]),
            )
            mRawStats = mPartialStats
            mLSE_partial = mPartialStats[None, None, None, 0]
            mRowSum_partial = mPartialStats[None, None, None, 1]
        else:
            mLSE_partial = cute.make_tensor(
                mLSE_partial.iterator,
                cute.select(mLSE_partial.layout, mode=LSE_partial_layout_transpose),
            )
        # (b, seqlen, h) -> (seqlen, h, b) or (total_q, h) -> (total_q, h)
        LSE_layout_transpose = [1, 2, 0] if const_expr(cu_seqlens is None) else [0, 1]
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if mLSE is not None
            else None
        )
        mLSE_temperature_partial = (
            cute.make_tensor(
                mLSE_temperature_partial.iterator,
                cute.select(mLSE_temperature_partial.layout, mode=LSE_partial_layout_transpose),
            )
            if mLSE_temperature_partial is not None
            else None
        )
        mLSE_temperature = (
            cute.make_tensor(
                mLSE_temperature.iterator,
                cute.select(mLSE_temperature.layout, mode=LSE_layout_transpose),
            )
            if mLSE_temperature is not None
            else None
        )

        # Determine if we have variable length sequences
        varlen = const_expr(cu_seqlens is not None or seqused is not None)

        self._setup_attributes()

        # Flatten [split, q, head] into one TMA row coordinate, matching the
        # CUDA combine descriptor. A separate split mode would require a
        # D * total_q * heads stride; that stride reaches exactly 2**32
        # elements for the 512K target and wraps them in a CuTe DSL descriptor
        # lowering. The flattened row extent remains representable and keeps
        # every 32x128 transfer contiguous.
        tma_atom_O_partial = None
        mO_partial_tma = None
        if const_expr(self.use_tma_partial):
            total_flat = mO_partial.shape[0] * mO_partial.shape[3]
            mO_partial_tma = cute.make_tensor(
                mO_partial.iterator,
                cute.make_layout(
                    (self.k_block_size, total_flat * self.topk),
                    stride=(
                        1,
                        self.k_block_size,
                    ),
                ),
            )
            tma_atom_O_partial, mO_partial_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mO_partial_tma,
                cute.select(self.smem_layout_o_tma, mode=[0, 1]),
                (self.k_block_size, self.tile_m),
            )

        raw_stats_copy_atom = None
        if const_expr(self.use_tma_partial):
            raw_stats_copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                Float32,
                num_bits_per_copy=64,
            )

        smem_layout_lse_aux = (
            cute.make_layout((1,), stride=(1,))
            if const_expr(self.use_tma_partial)
            else self.smem_layout_lse
        )

        # BF16/F16 output uses the register-shuffle epilogue and does not need
        # the legacy shared-memory permutation buffer. Keep a one-element
        # placeholder so one SharedStorage type can serve all specializations.
        if const_expr(self.dtype in [cutlass.Float16, cutlass.BFloat16]):
            smem_layout_perm = cute.make_layout((1,), stride=(1,))
        else:
            smem_layout_perm = cute.make_ordered_layout(
                (self.tile_m, self.k_block_size), order=(1, 0)
            )

        @cute.struct
        class SharedStorage:
            sLSE: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128
            ]
            sLSETemperature: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(smem_layout_lse_aux)], 128
            ]
            sMaxValidSplit: cute.struct.Align[cute.struct.MemRange[Int32, self.tile_m], 128]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.dtype_partial, cute.cosize(self.smem_layout_o)], 128
            ]
            sO_perm: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout_perm)], 128
            ]
            sOTmaMbar: cute.struct.Align[
                cute.struct.MemRange[Int64, self.stages], 128
            ]
        smem_size = SharedStorage.size_in_bytes()

        @cute.struct
        class TmaRawSharedStorage:
            sOTmaMbar: cute.struct.Align[
                cute.struct.MemRange[Int64, self.stages], 128
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128
            ]
            sMaxValidSplit: cute.struct.Align[
                cute.struct.MemRange[Int32, self.tile_m], 128
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[
                    self.dtype_partial,
                    cute.cosize(self.smem_layout_o),
                ],
                128,
            ]
        tma_raw_smem_size = TmaRawSharedStorage.size_in_bytes()

        # Flatten concatenated varlen rows into one scheduling domain. Each
        # combine row is independent, so this preserves the varlen layout
        # while avoiding one total_q-sized grid per batch.
        flatten_varlen = const_expr(
            cu_seqlens is not None
            and seqused is None
            and varlen_batch_idx is None
        )

        # Grid: (ceil(rows * heads / tile_m), ceil(dim / k_block), domains)
        # Head separated from rows enables future TMA on contiguous row tiles.
        seqlen = mO_partial.shape[0]
        num_head = mO_partial.shape[3]
        batch_size = (
            Int32(1)
            if const_expr(flatten_varlen)
            else (
                mO_partial.shape[4]
                if const_expr(cu_seqlens is None)
                else Int32(cu_seqlens.shape[0] - 1)
            )
        )

        seqlen_divmod = FastDivmodDivisor(seqlen)
        head_divmod = FastDivmodDivisor(num_head)

        grid_dim = (
            cute.ceil_div(seqlen * num_head, self.tile_m),
            cute.ceil_div(self.head_dim, self.k_block_size),
            batch_size,
        )

        if const_expr(self.use_tma_partial):
            if const_expr(not flatten_varlen):
                raise ValueError("TMA raw combine requires flattened varlen tensors")
            self.kernel_tma_raw(
                mO_partial_tma,
                mRawStats,
                mO,
                mLSE,
                semaphore_to_reset,
                mSplitCounts,
                mOutputScale,
                qhead_per_kvhead,
                tma_atom_O_partial,
                raw_stats_copy_atom,
                TmaRawSharedStorage,
                self.smem_layout_lse,
                self.smem_layout_o,
                self.smem_layout_o_tma,
                self.gmem_tiled_copy_O_partial,
                self.gmem_tiled_copy_O,
                head_divmod,
                self.use_pdl,
            ).launch(
                grid=grid_dim,
                block=[self.num_threads, 1, 1],
                smem=tma_raw_smem_size,
                stream=stream,
                min_blocks_per_mp=self.min_blocks_per_mp,
                use_pdl=self.use_pdl,
            )
        else:
            self.kernel(
                mO_partial,
                mO_partial_tma,
                mLSE_partial,
                mRowSum_partial,
                mRawStats,
                mO,
                mLSE,
                mLSE_temperature_partial,
                mLSE_temperature,
                cu_seqlens,
                seqused,
                num_splits_dynamic_ptr,
                varlen_batch_idx,
                semaphore_to_reset,
                mSplitCounts,
                mOutputScale,
                qhead_per_kvhead,
                tma_atom_O_partial,
                raw_stats_copy_atom,
                SharedStorage,
                self.smem_layout_lse,
                smem_layout_lse_aux,
                self.smem_layout_o,
                self.smem_layout_o_tma,
                smem_layout_perm,
                self.gmem_tiled_copy_O_partial,
                self.gmem_tiled_copy_O,
                self.gmem_tiled_copy_LSE,
                self.s2r_tiled_copy_LSE,
                seqlen_divmod,
                head_divmod,
                self.use_pdl,
                varlen,
                flatten_varlen,
            ).launch(
                grid=grid_dim,
                block=[self.num_threads, 1, 1],
                smem=smem_size,
                stream=stream,
                min_blocks_per_mp=self.min_blocks_per_mp,
                use_pdl=self.use_pdl,
            )

    @cute.jit
    def decode_flat_row_idx(
        self,
        idx: Int32,
        head_divmod: FastDivmodDivisor,
    ):
        """Decode flattened tile rows under the H_q-innermost contract."""
        q_idx_local, head_idx = divmod(idx, head_divmod)
        return q_idx_local, head_idx

    @cute.kernel
    def kernel_tma_raw(
        self,
        mO_partial_tma: cute.Tensor,
        mRawStats: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        semaphore_to_reset: Optional[cute.Tensor],
        mSplitCounts: Optional[cute.Tensor],
        mOutputScale: Optional[cute.Tensor],
        qhead_per_kvhead: Int32,
        tma_atom_O_partial: cute.CopyAtom,
        raw_stats_copy_atom: cute.CopyAtom,
        SharedStorage: cutlass.Constexpr,
        smem_layout_lse: cute.Layout | cute.ComposedLayout,
        smem_layout_o: cute.Layout | cute.ComposedLayout,
        smem_layout_o_tma: cute.Layout,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        head_divmod: FastDivmodDivisor,
        use_pdl: cutlass.Constexpr[bool],
    ):
        """CUDA-shaped raw-stat combine specialization for the SM100 target."""
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        m_block, k_block, maybe_virtual_batch = cute.arch.block_idx()

        if const_expr(use_pdl and semaphore_to_reset is not None):
            if (
                tidx == 0
                and m_block == cute.arch.grid_dim()[0] - 1
                and k_block == cute.arch.grid_dim()[1] - 1
                and maybe_virtual_batch == cute.arch.grid_dim()[2] - 1
            ):
                cute.arch.griddepcontrol_wait()
                semaphore_to_reset[0] = 0
        if const_expr(use_pdl):
            cute.arch.griddepcontrol_wait()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sLSE = storage.sLSE.get_tensor(smem_layout_lse)
        sMaxValidSplit = storage.sMaxValidSplit.get_tensor((self.tile_m,))
        sO = storage.sO.get_tensor(smem_layout_o)
        sO_tma = storage.sO.get_tensor(smem_layout_o_tma)
        sO_tma_mbar_ptr = storage.sOTmaMbar.data_ptr()

        total_flat = mRawStats.shape[0] * mRawStats.shape[2]
        tile_begin = m_block * self.tile_m
        output_scale = Float32(1.0)
        if const_expr(mOutputScale is not None):
            output_scale = mOutputScale[0]

        # Initialize the two transaction barriers and overlap the initial TMA
        # loads with raw-stat reduction, matching the CUDA schedule.
        gO_partial_tma = cute.local_tile(
            mO_partial_tma,
            (self.k_block_size, self.tile_m),
            (k_block, None),
        )
        tOsO_tma, tOgO_tma = cpasync.tma_partition(
            tma_atom_O_partial,
            0,
            cute.make_layout(1),
            cute.group_modes(sO_tma, 0, 2),
            cute.group_modes(gO_partial_tma, 0, 2),
        )
        o_tma_copy_bytes = const_expr(
            cute.size_in_bytes(
                self.dtype_partial,
                cute.select(smem_layout_o_tma, mode=[0, 1]),
            )
        )
        row_tiles_per_split = total_flat // Int32(self.tile_m)
        if tidx == 0:
            for stage in cutlass.range(self.stages, unroll_full=True):
                cute.arch.mbarrier_init(sO_tma_mbar_ptr + stage, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if warp_idx == 0:
            for stage in cutlass.range(self.stages, unroll_full=True):
                o_tma_barrier = sO_tma_mbar_ptr + stage
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        o_tma_barrier,
                        o_tma_copy_bytes,
                    )
                cute.copy(
                    tma_atom_O_partial,
                    tOgO_tma[
                        None,
                        m_block + Int32(stage) * row_tiles_per_split,
                    ],
                    tOsO_tma[None, stage],
                    tma_bar_ptr=o_tma_barrier,
                )

        # Four adjacent lanes own one row and four interleaved split pairs.
        reduce_lane = tidx % Int32(4)
        reduce_row = tidx // Int32(4)
        reduce_flat = tile_begin + reduce_row
        row_valid = reduce_flat < total_flat
        m_idx = Int32(0)
        head_idx = Int32(0)
        row_split_count = Int32(0)
        if row_valid:
            m_idx, head_idx = self.decode_flat_row_idx(
                reduce_flat,
                head_divmod,
            )
            row_split_count = (
                mSplitCounts[m_idx, head_idx // qhead_per_kvhead]
                if const_expr(mSplitCounts is not None)
                else Int32(self.topk)
            )

        local_page_max = cute.make_rmem_tensor((4,), Float32)
        local_page_sum = cute.make_rmem_tensor((4,), Float32)
        raw_stats = cute.make_rmem_tensor((2,), Float32)
        local_max = -Float32.inf
        local_max_valid = Int32(-1)
        for item in cutlass.range(4, unroll_full=True):
            split = reduce_lane + item * Int32(4)
            raw_stats[0] = -Float32.inf
            raw_stats[1] = Float32(0.0)
            if row_valid and split < row_split_count:
                cute.copy(
                    raw_stats_copy_atom,
                    mRawStats[m_idx, split, head_idx, None],
                    raw_stats,
                )
            stats_valid = (
                raw_stats[1] > Float32(0.0)
                and raw_stats[1] < Float32.inf
                and raw_stats[0] > -Float32.inf
                and raw_stats[0] < Float32.inf
                and raw_stats[0] == raw_stats[0]
                and raw_stats[1] == raw_stats[1]
            )
            page_max = raw_stats[0] if stats_valid else -Float32.inf
            page_sum = raw_stats[1] if stats_valid else Float32(0.0)
            local_page_max[item] = page_max
            local_page_sum[item] = page_sum
            local_max = max(local_max, page_max)
            if stats_valid:
                local_max_valid = split

        stats_max = cute.arch.warp_reduction_max(
            local_max,
            threads_in_group=4,
        )
        max_valid_split = cute.arch.warp_reduction_max(
            local_max_valid,
            threads_in_group=4,
        )
        stats_max_safe = (
            stats_max if max_valid_split >= Int32(0) else Float32(0.0)
        )
        stats_sum = Float32(0.0)
        for item in cutlass.range(4, unroll_full=True):
            split_scale = cute.math.exp2(
                local_page_max[item] - stats_max_safe,
                fastmath=True,
            )
            local_page_max[item] = split_scale
            stats_sum += split_scale * local_page_sum[item]
        stats_sum = cute.arch.warp_reduction_sum(
            stats_sum,
            threads_in_group=4,
        )
        stats_sum_valid = (
            max_valid_split >= Int32(0)
            and stats_sum > Float32(0.0)
            and stats_sum < Float32.inf
            and stats_sum == stats_sum
        )
        inv_sum = Float32(0.0)
        final_lse = -Float32.inf
        if stats_sum_valid:
            inv_sum = Float32(1.0) / stats_sum
            final_lse = (
                stats_max + cute.math.log2(stats_sum, fastmath=True)
            ) * Float32(math.log(2.0))
        for item in cutlass.range(4, unroll_full=True):
            split = reduce_lane + item * Int32(4)
            sLSE[split, reduce_row] = local_page_max[item] * inv_sum
        if reduce_lane == Int32(0):
            sMaxValidSplit[reduce_row] = max_valid_split
            if const_expr(mLSE is not None):
                if row_valid:
                    mLSE[m_idx, head_idx] = final_lse

        cute.arch.sync_threads()
        block_max_valid_split = Int32(-1)
        if const_expr(self.tile_m == 32):
            if tidx < Int32(self.tile_m):
                block_max_valid_split = cute.arch.warp_reduction_max(
                    sMaxValidSplit[tidx],
                    threads_in_group=self.tile_m,
                )
                if tidx == 0:
                    sMaxValidSplit[0] = block_max_valid_split
        else:
            if tidx < Int32(32):
                block_max_valid_split = cute.arch.warp_reduction_max(
                    max(
                        sMaxValidSplit[tidx],
                        sMaxValidSplit[tidx + Int32(32)],
                    ),
                    threads_in_group=32,
                )
                if tidx == 0:
                    sMaxValidSplit[0] = block_max_valid_split
        cute.arch.sync_threads()
        block_max_valid_split = sMaxValidSplit[0]

        cO = cute.make_identity_tensor((self.tile_m, self.k_block_size))
        partial_copy = gmem_tiled_copy_O_partial.get_slice(tidx)
        tOcO = partial_copy.partition_D(cO)
        tOsO_partial = partial_copy.partition_D(sO)
        num_rows = const_expr(cute.size(tOcO, mode=[1]))
        tOrO_partial_row = cute.make_rmem_tensor_like(
            tOsO_partial[None, 0, None, 0]
        )
        tOrO = cute.make_rmem_tensor_like(
            tOsO_partial[None, None, None, 0],
            Float32,
        )
        tOrO.fill(0.0)

        for split in cutlass.range(block_max_valid_split + 1, unroll=4):
            stage = split % Int32(self.stages)
            phase = (split // Int32(self.stages)) & Int32(1)
            cute.arch.mbarrier_wait(sO_tma_mbar_ptr + stage, phase)
            for row in cutlass.range(num_rows, unroll_full=True):
                cute.autovec_copy(
                    tOsO_partial[None, row, None, stage],
                    tOrO_partial_row,
                )
                weight = sLSE[split, tOcO[0, row, 0][0]]
                if weight != Float32(0.0):
                    tOrO[None, row, None].store(
                        tOrO[None, row, None].load()
                        + weight
                        * tOrO_partial_row.load().to(Float32)
                    )

            cute.arch.sync_threads()
            split_to_load = split + Int32(self.stages)
            if warp_idx == 0 and split_to_load <= block_max_valid_split:
                o_tma_barrier = sO_tma_mbar_ptr + stage
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        o_tma_barrier,
                        o_tma_copy_bytes,
                    )
                cute.copy(
                    tma_atom_O_partial,
                    tOgO_tma[
                        None,
                        m_block + split_to_load * row_tiles_per_split,
                    ],
                    tOsO_tma[None, stage],
                    tma_bar_ptr=o_tma_barrier,
                )

        # Transpose the four 32-bit BF16 pairs within each four-lane group,
        # then retain one coalesced 128-bit output store per row.
        output_copy = gmem_tiled_copy_O.get_slice(tidx)
        tOcO_store = output_copy.partition_D(cO)
        rO_row = cute.make_rmem_tensor(
            tOcO_store[None, 0, 0].shape,
            self.dtype,
        )
        elems_per_store = const_expr(
            cute.size(gmem_tiled_copy_O.layout_tv_tiled[1])
        )
        num_store_rows = const_expr(cute.size(tOcO_store, mode=[1]))
        lane_in_pair_group = tidx % Int32(4)
        for row in cutlass.range(num_store_rows, unroll_full=True):
            for k in cutlass.range(
                cute.size(tOcO_store, mode=[2]),
                unroll_full=True,
            ):
                o0, o1 = tOrO[0, row, k], tOrO[1, row, k]
                o2, o3 = tOrO[2, row, k], tOrO[3, row, k]
                o4, o5 = tOrO[4, row, k], tOrO[5, row, k]
                o6, o7 = tOrO[6, row, k], tOrO[7, row, k]
                if const_expr(mOutputScale is not None):
                    o0, o1 = cute.arch.mul_packed_f32x2(
                        (o0, o1),
                        (output_scale, output_scale),
                    )
                    o2, o3 = cute.arch.mul_packed_f32x2(
                        (o2, o3),
                        (output_scale, output_scale),
                    )
                    o4, o5 = cute.arch.mul_packed_f32x2(
                        (o4, o5),
                        (output_scale, output_scale),
                    )
                    o6, o7 = cute.arch.mul_packed_f32x2(
                        (o6, o7),
                        (output_scale, output_scale),
                    )
                word0 = utils.cvt_f16x2_f32(o0, o1, self.dtype)
                word1 = utils.cvt_f16x2_f32(o2, o3, self.dtype)
                word2 = utils.cvt_f16x2_f32(o4, o5, self.dtype)
                word3 = utils.cvt_f16x2_f32(o6, o7, self.dtype)

                partner0 = cute.arch.shuffle_sync_bfly(word1, offset=1)
                partner1 = cute.arch.shuffle_sync_bfly(word0, offset=1)
                partner2 = cute.arch.shuffle_sync_bfly(word3, offset=1)
                partner3 = cute.arch.shuffle_sync_bfly(word2, offset=1)
                stage0, stage1 = word0, word1
                stage2, stage3 = word2, word3
                if lane_in_pair_group % Int32(2) != Int32(0):
                    stage0, stage2 = partner0, partner2
                if lane_in_pair_group % Int32(2) != Int32(1):
                    stage1, stage3 = partner1, partner3

                partner0 = cute.arch.shuffle_sync_bfly(stage2, offset=2)
                partner1 = cute.arch.shuffle_sync_bfly(stage3, offset=2)
                partner2 = cute.arch.shuffle_sync_bfly(stage0, offset=2)
                partner3 = cute.arch.shuffle_sync_bfly(stage1, offset=2)
                out0, out1 = stage0, stage1
                out2, out3 = stage2, stage3
                if lane_in_pair_group // Int32(2) != Int32(0):
                    out0, out1 = partner0, partner1
                if lane_in_pair_group // Int32(2) != Int32(1):
                    out2, out3 = partner2, partner3

                rO_pairs = cute.recast_tensor(rO_row, cutlass.Int32)
                rO_pairs[0] = out0
                rO_pairs[1] = out1
                rO_pairs[2] = out2
                rO_pairs[3] = out3
                row_local = tOcO_store[0, row, 0][0]
                flat_idx = tile_begin + row_local
                if flat_idx < total_flat:
                    m_idx, head_idx = self.decode_flat_row_idx(
                        flat_idx,
                        head_divmod,
                    )
                    mO_cur_copy = cute.tiled_divide(
                        mO[m_idx, None, head_idx],
                        (elems_per_store,),
                    )
                    k_idx = tOcO_store[0, 0, k][1] // elems_per_store
                    cute.copy(
                        output_copy,
                        rO_row,
                        mO_cur_copy[None, k_idx],
                    )

    @cute.kernel
    def kernel(
        self,
        mO_partial: cute.Tensor,
        mO_partial_tma: Optional[cute.Tensor],
        mLSE_partial: cute.Tensor,
        mRowSum_partial: Optional[cute.Tensor],
        mRawStats: Optional[cute.Tensor],
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSE_temperature_partial: Optional[cute.Tensor],
        mLSE_temperature: Optional[cute.Tensor],
        cu_seqlens: Optional[cute.Tensor],
        seqused: Optional[cute.Tensor],
        num_splits_dynamic_ptr: Optional[cute.Tensor],
        varlen_batch_idx: Optional[cute.Tensor],
        semaphore_to_reset: Optional[cute.Tensor],
        mSplitCounts: Optional[cute.Tensor],
        mOutputScale: Optional[cute.Tensor],
        qhead_per_kvhead: Int32,
        tma_atom_O_partial: Optional[cute.CopyAtom],
        raw_stats_copy_atom: Optional[cute.CopyAtom],
        SharedStorage: cutlass.Constexpr,
        smem_layout_lse: cute.Layout | cute.ComposedLayout,
        smem_layout_lse_aux: cute.Layout | cute.ComposedLayout,
        smem_layout_o: cute.Layout | cute.ComposedLayout,
        smem_layout_o_tma: Optional[cute.Layout],
        smem_layout_perm: cute.Layout,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        s2r_tiled_copy_LSE: cute.TiledCopy,
        seqlen_divmod: FastDivmodDivisor,
        head_divmod: FastDivmodDivisor,
        use_pdl: cutlass.Constexpr[bool],
        varlen: cutlass.Constexpr[bool],
        flatten_varlen: cutlass.Constexpr[bool],
    ):
        # Thread and block indices
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        m_block, k_block, maybe_virtual_batch = cute.arch.block_idx()

        if const_expr(flatten_varlen):
            batch_idx = Int32(0)
        else:
            batch_idx = (
                varlen_batch_idx[maybe_virtual_batch]
                if const_expr(varlen_batch_idx is not None)
                else maybe_virtual_batch
            )

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sLSE = storage.sLSE.get_tensor(smem_layout_lse)
        sLSE_temperature = storage.sLSETemperature.get_tensor(smem_layout_lse_aux)
        sMaxValidSplit = storage.sMaxValidSplit.get_tensor((self.tile_m,))
        sO = storage.sO.get_tensor(smem_layout_o)
        sO_tma = None
        sO_tma_mbar_ptr = None
        if const_expr(self.use_tma_partial):
            sO_tma = storage.sO.get_tensor(smem_layout_o_tma)
            sO_tma_mbar_ptr = storage.sOTmaMbar.data_ptr()
        sO_perm_buf = storage.sO_perm.get_tensor(smem_layout_perm)

        # Handle semaphore reset — wait for dependent grids first
        if const_expr(use_pdl and semaphore_to_reset is not None):
            if (
                tidx == 0
                and m_block == cute.arch.grid_dim()[0] - 1
                and k_block == cute.arch.grid_dim()[1] - 1
                and maybe_virtual_batch == cute.arch.grid_dim()[2] - 1
            ):
                cute.arch.griddepcontrol_wait()
                semaphore_to_reset[0] = 0

        if const_expr(num_splits_dynamic_ptr is not None):
            raise ValueError("K2 combine requires compile-time exact topK")
        num_splits = Int32(self.topk)
        # Concatenated varlen tensors already have total_q as their leading
        # extent. Other modes retain SeqlenInfo-based per-batch slicing.
        if const_expr(flatten_varlen):
            seqlen = Int32(mO_partial.shape[0])
            offset = Int32(0)
        else:
            seqlen_info = SeqlenInfo.create(
                batch_idx=batch_idx,
                seqlen_static=mO_partial.shape[0],
                cu_seqlens=cu_seqlens,
                seqused=seqused,
                # Don't need tile size because offset_padded is unused.
            )
            seqlen, offset = seqlen_info.seqlen, seqlen_info.offset

        num_head = mO_partial.shape[3]
        max_idx = seqlen * num_head
        output_scale = Float32(1.0)
        if const_expr(mOutputScale is not None):
            output_scale = mOutputScale[0]

        if const_expr(not varlen) or m_block * self.tile_m < max_idx:
            # Wait for dependent grids (e.g., the main attention kernel that produces O_partial/LSE_partial)
            if const_expr(use_pdl):
                cute.arch.griddepcontrol_wait()

            # ===============================
            # Step 1: Load LSE_partial from gmem to shared memory
            # ===============================
            # `cLSE` (identity tensor for row/split coord tracking) is reused
            # later in steps 4-5, so it must be defined on both branches.
            cLSE = cute.make_identity_tensor((self.topk, self.tile_m))
            # Reshape mLSE_partial to PackGQA packed layout and delegate the
            # tile load to PackGQAComb.load_LSE. The packed form folds (H_q, Sq)
            # into one compound dim with H_q innermost (stride 1), so thread
            # rows that vary along h_pos produce one-sector coalesced reads.
            # Non-varlen path only — varlen keeps the original inline loop.
            if const_expr(not varlen):
                mLSE_partial_cur = seqlen_info.offset_batch(mLSE_partial, batch_idx, dim=3)
                # mLSE_partial_cur: (H_q, topK, Sq) — after initial transpose
                # [3,0,2,1] on [topK,B,Sq,H_q] and dropping B.
                # Reorder to (H_q, Sq, topK) then group modes 0..1 for packed dim:
                mLSE_partial_reord = cute.make_tensor(
                    mLSE_partial_cur.iterator,
                    cute.select(mLSE_partial_cur.layout, mode=[0, 2, 1]),
                )
                mLSE_partial_packed = cute.group_modes(mLSE_partial_reord, 0, 2)
                # shape ((H_q, Sq), topK) with H_q innermost.
                packgqa = PackGQAComb(
                    m_block_size=self.tile_m,
                    head_dim_padded=0,       # unused for LSE load
                    check_hdim_oob=False,    # unused for LSE load
                    qhead_per_kvhead=1,      # unused; num_heads_divmod is passed explicitly
                )
                packgqa.load_LSE(
                    mLSE_partial_packed,
                    sLSE,
                    self.topk,
                    gmem_tiled_copy_LSE,
                    tidx,
                    m_block,
                    num_splits,
                    seqlen,
                    head_divmod,
                    mSplitCounts,
                    batch_idx,
                    qhead_per_kvhead,
                )
                if const_expr(mLSE_temperature_partial is not None):
                    mLSE_temperature_partial_cur = seqlen_info.offset_batch(
                        mLSE_temperature_partial, batch_idx, dim=3)
                    mLSE_temperature_partial_reord = cute.make_tensor(
                        mLSE_temperature_partial_cur.iterator,
                        cute.select(mLSE_temperature_partial_cur.layout, mode=[0, 2, 1]),
                    )
                    mLSE_temperature_partial_packed = cute.group_modes(
                        mLSE_temperature_partial_reord, 0, 2)
                    packgqa.load_LSE(
                        mLSE_temperature_partial_packed,
                        sLSE_temperature,
                        self.topk,
                        gmem_tiled_copy_LSE,
                        tidx,
                        m_block,
                        num_splits,
                        seqlen,
                        head_divmod,
                        mSplitCounts,
                        batch_idx,
                        qhead_per_kvhead,
                    )
            else:
                # Varlen path keeps the same H_q-innermost flat-row contract:
                # after transpose [1, 0, 2], mLSE_partial_cur is
                # (q_local, split, head).
                # mSplitCounts is the authoritative valid-split count per
                # packed (q_abs, kv_head); masked splits stay at -inf and
                # therefore drop out of the final kernel LSE_out reduction.
                if const_expr(flatten_varlen):
                    mLSE_partial_cur = mLSE_partial
                    if const_expr(self.raw_partial_stats):
                        mRowSum_partial_cur = mRowSum_partial
                else:
                    mLSE_partial_cur = seqlen_info.offset_batch(
                        mLSE_partial, batch_idx, dim=3
                    )
                    if const_expr(self.raw_partial_stats):
                        mRowSum_partial_cur = seqlen_info.offset_batch(
                            mRowSum_partial, batch_idx, dim=3
                        )
                mLSE_partial_copy = cute.tiled_divide(mLSE_partial_cur, (1,))
                if const_expr(self.raw_partial_stats):
                    mRowSum_partial_copy = cute.tiled_divide(mRowSum_partial_cur, (1,))
                gmem_thr_copy_LSE = gmem_tiled_copy_LSE.get_slice(tidx)
                tLSEsLSE = gmem_thr_copy_LSE.partition_D(sLSE)
                tLSEsLSE_temperature = gmem_thr_copy_LSE.partition_D(sLSE_temperature)
                tLSEcLSE = gmem_thr_copy_LSE.partition_S(cLSE)
                if const_expr(mLSE_temperature_partial is not None):
                    if const_expr(flatten_varlen):
                        mLSE_temperature_partial_cur = mLSE_temperature_partial
                    else:
                        mLSE_temperature_partial_cur = seqlen_info.offset_batch(
                            mLSE_temperature_partial, batch_idx, dim=3
                        )
                    mLSE_temperature_partial_copy = cute.tiled_divide(
                        mLSE_temperature_partial_cur, (1,))

                for m in cutlass.range(cute.size(tLSEcLSE, mode=[2]), unroll_full=True):
                    mi = tLSEcLSE[0, 0, m][1]
                    idx = m_block * self.tile_m + mi
                    if idx < max_idx:
                        m_idx, head_idx = self.decode_flat_row_idx(idx, head_divmod)
                        row_count = (
                            mSplitCounts[offset + m_idx, head_idx // qhead_per_kvhead]
                            if const_expr(mSplitCounts is not None)
                            else num_splits
                        )
                        mLSE_partial_cur_copy = mLSE_partial_copy[None, m_idx, None, head_idx]
                        if const_expr(self.raw_partial_stats):
                            mRowSum_partial_cur_copy = (
                                mRowSum_partial_copy[None, m_idx, None, head_idx]
                            )
                        if const_expr(mLSE_temperature_partial is not None):
                            mLSE_temperature_partial_cur_copy = (
                                mLSE_temperature_partial_copy[None, m_idx, None, head_idx])
                        for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                            si = tLSEcLSE[0, s, 0][0]
                            if si < num_splits and si < row_count:
                                cute.copy(
                                    gmem_thr_copy_LSE,
                                    mLSE_partial_cur_copy[None, si],
                                    tLSEsLSE[None, s, m],
                                )
                                if const_expr(self.raw_partial_stats):
                                    cute.copy(
                                        gmem_thr_copy_LSE,
                                        mRowSum_partial_cur_copy[None, si],
                                        tLSEsLSE_temperature[None, s, m],
                                    )
                                if const_expr(mLSE_temperature_partial is not None):
                                    cute.copy(
                                        gmem_thr_copy_LSE,
                                        mLSE_temperature_partial_cur_copy[None, si],
                                        tLSEsLSE_temperature[None, s, m],
                                    )
                            else:
                                tLSEsLSE[None, s, m].fill(-Float32.inf)
                                if const_expr(self.raw_partial_stats):
                                    tLSEsLSE_temperature[None, s, m].fill(Float32(0.0))
                                if const_expr(mLSE_temperature_partial is not None):
                                    tLSEsLSE_temperature[None, s, m].fill(-Float32.inf)
                    else:
                        for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                            tLSEsLSE[None, s, m].fill(-Float32.inf)
                            if const_expr(self.raw_partial_stats):
                                tLSEsLSE_temperature[None, s, m].fill(Float32(0.0))
                            if const_expr(mLSE_temperature_partial is not None):
                                tLSEsLSE_temperature[None, s, m].fill(-Float32.inf)
            cute.arch.cp_async_commit_group()

            # ===============================
            # Step 2: Load O_partial for pipeline stages
            # ===============================

            gmem_thr_copy_O_partial = gmem_tiled_copy_O_partial.get_slice(tidx)
            cO = cute.make_identity_tensor((self.tile_m, self.k_block_size))
            tOcO = gmem_thr_copy_O_partial.partition_D(cO)
            tOsO_partial = gmem_thr_copy_O_partial.partition_D(sO)

            # Precompute per-row values for flattened (q_local, head) tiles.
            num_rows = const_expr(cute.size(tOcO, mode=[1]))
            tOmidx = cute.make_rmem_tensor(num_rows, cutlass.Int32)
            tOhidx = cute.make_rmem_tensor(num_rows, cutlass.Int32)
            if const_expr(not self.use_tma_partial):
                tOSplitCount = cute.make_rmem_tensor(num_rows, cutlass.Int32)
                tOrOptr = cute.make_rmem_tensor(num_rows, cutlass.Int64)
                if const_expr(flatten_varlen):
                    mO_partial_cur = mO_partial
                else:
                    mO_partial_cur = seqlen_info.offset_batch(
                        mO_partial, batch_idx, dim=4
                    )
            for m in cutlass.range(num_rows, unroll_full=True):
                mi = tOcO[0, m, 0][0]  # m coordinate in tile
                idx = m_block * self.tile_m + mi
                if idx >= max_idx:
                    tOhidx[m] = -1
                    tOmidx[m] = 0
                    if const_expr(not self.use_tma_partial):
                        tOSplitCount[m] = 0
                        tOrOptr[m] = cutlass.Int64(0)
                else:
                    tOmidx[m], tOhidx[m] = self.decode_flat_row_idx(idx, head_divmod)
                    if const_expr(not self.use_tma_partial):
                        if const_expr(mSplitCounts is None):
                            tOSplitCount[m] = num_splits
                        elif const_expr(cu_seqlens is None):
                            tOSplitCount[m] = mSplitCounts[
                                batch_idx, tOmidx[m], tOhidx[m] // qhead_per_kvhead
                            ]
                        else:
                            tOSplitCount[m] = mSplitCounts[
                                offset + tOmidx[m], tOhidx[m] // qhead_per_kvhead
                            ]
                        tOrOptr[m] = utils.elem_pointer(
                            mO_partial_cur,
                            (tOmidx[m], k_block * self.k_block_size, 0, tOhidx[m]),
                        ).toint()

            tOpO = None
            if const_expr(not self.is_even_k):
                tOpO = cute.make_rmem_tensor(cute.size(tOcO, mode=[2]), Boolean)
                for k in cutlass.range(cute.size(tOpO), unroll_full=True):
                    tOpO[k] = tOcO[0, 0, k][1] < mO_partial.shape[1] - k_block * self.k_block_size
            if const_expr(self.use_tma_partial):
                gO_partial_tma = cute.local_tile(
                    mO_partial_tma,
                    (self.k_block_size, self.tile_m),
                    (k_block, m_block, None),
                )
                tOsO_tma, tOgO_tma = cpasync.tma_partition(
                    tma_atom_O_partial,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sO_tma, 0, 2),
                    cute.group_modes(gO_partial_tma, 0, 2),
                )
                o_tma_copy_bytes = const_expr(
                    cute.size_in_bytes(
                        self.dtype_partial,
                        cute.select(smem_layout_o_tma, mode=[0, 1]),
                    )
                )
                if tidx == 0:
                    for stage in cutlass.range(self.stages, unroll_full=True):
                        cute.arch.mbarrier_init(sO_tma_mbar_ptr + stage, 1)
                    cute.arch.mbarrier_init_fence()
                cute.arch.sync_threads()
                if warp_idx == 0:
                    for stage in cutlass.range(self.stages, unroll_full=True):
                        o_tma_barrier = sO_tma_mbar_ptr + stage
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                o_tma_barrier,
                                o_tma_copy_bytes,
                            )
                        cute.copy(
                            tma_atom_O_partial,
                            tOgO_tma[None, stage],
                            tOsO_tma[None, stage],
                            tma_bar_ptr=o_tma_barrier,
                        )
            else:
                load_O_partial = partial(
                    self.load_O_partial,
                    gmem_tiled_copy_O_partial,
                    tOrOptr,
                    tOsO_partial,
                    tOhidx,
                    tOSplitCount,
                    tOpO,
                    tOcO,
                    mO_partial_cur.layout,
                )

                # Load the initial cp.async stages of O_partial.
                for stage in cutlass.range(self.stages - 1, unroll_full=True):
                    if stage < num_splits:
                        load_O_partial(stage, stage)
                    cute.arch.cp_async_commit_group()

            # ===============================
            # Step 3: Load and transpose LSE from smem to registers
            # ===============================

            # Wait for LSE and initial O partial stages to complete
            if const_expr(self.use_tma_partial):
                cute.arch.cp_async_wait_group(0)
            else:
                cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.sync_threads()

            s2r_thr_copy_LSE = s2r_tiled_copy_LSE.get_slice(tidx)
            ts2rsLSE = s2r_thr_copy_LSE.partition_S(sLSE)
            ts2rrLSE = cute.make_rmem_tensor_like(ts2rsLSE)
            cute.copy(s2r_tiled_copy_LSE, ts2rsLSE, ts2rrLSE)
            if const_expr(
                self.raw_partial_stats or mLSE_temperature_partial is not None
            ):
                ts2rsLSE_temperature = s2r_thr_copy_LSE.partition_S(sLSE_temperature)
                ts2rrLSE_temperature = cute.make_rmem_tensor_like(ts2rsLSE_temperature)
                cute.copy(
                    s2r_tiled_copy_LSE,
                    ts2rsLSE_temperature,
                    ts2rrLSE_temperature,
                )

            # ===============================
            # Step 4: Compute final LSE along split dimension
            # ===============================

            final_lse = cute.make_rmem_tensor(cute.size(ts2rrLSE, mode=[2]), Float32)
            ts2rcLSE = s2r_thr_copy_LSE.partition_D(cLSE)
            # We compute the max valid split for each row to short-circuit the computation later
            max_valid_split = cute.make_rmem_tensor(cute.size(ts2rrLSE, mode=[2]), Int32)
            assert cute.size(ts2rrLSE, mode=[0]) == 1
            # Compute max, scales, and final LSE for each row. Invalid splits
            # have already been filled with -inf, so Step 5 can write the
            # kernel-native LSE_out directly.
            for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                threads_per_col = const_expr(self.smem_threads_per_col_lse)
                if const_expr(self.raw_partial_stats):
                    # Sanitize the raw pair before reductions so NaN/Inf cannot
                    # contaminate a neighboring split in the thread group.
                    max_valid_idx = Int32(-1)
                    for s in cutlass.range(
                        cute.size(ts2rrLSE, mode=[1]), unroll_full=True
                    ):
                        scaled_max_log2 = ts2rrLSE[0, s, m]
                        row_sum = ts2rrLSE_temperature[0, s, m]
                        stats_valid = (
                            row_sum > Float32(0.0)
                            and row_sum < Float32.inf
                            and scaled_max_log2 > -Float32.inf
                            and scaled_max_log2 < Float32.inf
                            and row_sum == row_sum
                            and scaled_max_log2 == scaled_max_log2
                        )
                        ts2rrLSE[0, s, m] = (
                            scaled_max_log2 if stats_valid else -Float32.inf
                        )
                        ts2rrLSE_temperature[0, s, m] = (
                            row_sum if stats_valid else Float32(0.0)
                        )
                        if stats_valid:
                            max_valid_idx = ts2rcLSE[0, s, 0][0]

                    stats_max = cute.arch.warp_reduction_max(
                        ts2rrLSE[None, None, m]
                        .load()
                        .reduce(
                            cute.ReductionOp.MAX,
                            init_val=-Float32.inf,
                            reduction_profile=0,
                        ),
                        threads_in_group=threads_per_col,
                    )
                    max_valid_split[m] = cute.arch.warp_reduction_max(
                        max_valid_idx, threads_in_group=threads_per_col
                    )
                    stats_max_safe = (
                        stats_max
                        if max_valid_split[m] >= Int32(0)
                        else Float32(0.0)
                    )
                    stats_sum = Float32(0.0)
                    for s in cutlass.range(
                        cute.size(ts2rrLSE, mode=[1]), unroll_full=True
                    ):
                        row_sum = ts2rrLSE_temperature[0, s, m]
                        split_scale = Float32(0.0)
                        if row_sum > Float32(0.0):
                            split_scale = cute.math.exp2(
                                ts2rrLSE[0, s, m] - stats_max_safe,
                                fastmath=True,
                            )
                        stats_sum += split_scale * row_sum
                        # Raw partial-O already includes the local row sum, so
                        # its combine weight excludes row_sum from the numerator.
                        ts2rrLSE[0, s, m] = split_scale
                    stats_sum = cute.arch.warp_reduction_sum(
                        stats_sum, threads_in_group=threads_per_col
                    )
                    stats_sum_valid = (
                        max_valid_split[m] >= Int32(0)
                        and stats_sum > Float32(0.0)
                        and stats_sum < Float32.inf
                        and stats_sum == stats_sum
                    )
                    inv_sum = Float32(0.0)
                    if stats_sum_valid:
                        inv_sum = Float32(1.0) / stats_sum
                        final_lse[m] = (
                            stats_max
                            + cute.math.log2(stats_sum, fastmath=True)
                        ) * Float32(math.log(2.0))
                    else:
                        final_lse[m] = -Float32.inf
                    ts2rrLSE[None, None, m].store(
                        ts2rrLSE[None, None, m].load() * inv_sum
                    )
                else:
                    # Find max normalized-partial LSE value across splits.
                    lse_max = cute.arch.warp_reduction_max(
                        ts2rrLSE[None, None, m]
                        .load()
                        .reduce(
                            cute.ReductionOp.MAX,
                            init_val=-Float32.inf,
                            reduction_profile=0,
                        ),
                        threads_in_group=threads_per_col,
                    )
                    max_valid_idx = -1
                    for s in cutlass.range(
                        cute.size(ts2rrLSE, mode=[1]), unroll_full=True
                    ):
                        if ts2rrLSE[0, s, m] != -Float32.inf:
                            max_valid_idx = ts2rcLSE[0, s, 0][0]
                    max_valid_split[m] = cute.arch.warp_reduction_max(
                        max_valid_idx, threads_in_group=threads_per_col
                    )
                    lse_max_cur = Float32(0.0) if lse_max == -Float32.inf else lse_max
                    LOG2_E = math.log2(math.e)
                    lse_sum_cur = Float32(0.0)
                    for s in cutlass.range(
                        cute.size(ts2rrLSE, mode=[1]), unroll_full=True
                    ):
                        scale = cute.math.exp2(
                            ts2rrLSE[0, s, m] * LOG2_E
                            - lse_max_cur * LOG2_E,
                            fastmath=True,
                        )
                        lse_sum_cur += scale
                        ts2rrLSE[0, s, m] = scale
                    lse_sum_cur = cute.arch.warp_reduction_sum(
                        lse_sum_cur, threads_in_group=threads_per_col
                    )
                    inv_sum = Float32(0.0)
                    if (
                        max_valid_split[m] < 0
                        or lse_sum_cur == Float32(0.0)
                        or lse_sum_cur != lse_sum_cur
                    ):
                        final_lse[m] = -Float32.inf
                    else:
                        final_lse[m] = (
                            cute.math.log(lse_sum_cur, fastmath=True) + lse_max
                        )
                        inv_sum = Float32(1.0) / lse_sum_cur
                    ts2rrLSE[None, None, m].store(
                        ts2rrLSE[None, None, m].load() * inv_sum
                    )
            # Store the scales exp(lse - lse_logsum) back to smem
            cute.copy(s2r_tiled_copy_LSE, ts2rrLSE, ts2rsLSE)

            if const_expr(mLSE_temperature_partial is not None):
                final_lse_temperature = cute.make_rmem_tensor(
                    cute.size(ts2rrLSE_temperature, mode=[2]), Float32)
                for m in cutlass.range(cute.size(ts2rrLSE_temperature, mode=[2]), unroll_full=True):
                    threads_per_col = const_expr(self.smem_threads_per_col_lse)
                    lse_temperature_max = cute.arch.warp_reduction_max(
                        ts2rrLSE_temperature[None, None, m]
                        .load()
                        .reduce(
                            cute.ReductionOp.MAX,
                            init_val=-Float32.inf,
                            reduction_profile=0,
                        ),
                        threads_in_group=threads_per_col,
                    )
                    lse_temperature_max_cur = (
                        0.0 if lse_temperature_max == -Float32.inf else lse_temperature_max
                    )
                    LOG2_E = math.log2(math.e)
                    lse_temperature_sum_cur = 0.0
                    for s in cutlass.range(
                        cute.size(ts2rrLSE_temperature, mode=[1]), unroll_full=True):
                        scale = cute.math.exp2(
                            ts2rrLSE_temperature[0, s, m] * LOG2_E
                            - (lse_temperature_max_cur * LOG2_E),
                            fastmath=True,
                        )
                        lse_temperature_sum_cur += scale
                    lse_temperature_sum_cur = cute.arch.warp_reduction_sum(
                        lse_temperature_sum_cur, threads_in_group=threads_per_col
                    )
                    if (
                        max_valid_split[m] < 0
                        or lse_temperature_sum_cur == 0.0
                        or lse_temperature_sum_cur != lse_temperature_sum_cur
                    ):
                        final_lse_temperature[m] = -Float32.inf
                    else:
                        final_lse_temperature[m] = (
                            cute.math.log(lse_temperature_sum_cur, fastmath=True)
                            + lse_temperature_max
                        )

            # Store max valid split to smem
            for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                if ts2rcLSE[0, 0, m][0] == 0:  # Only thread responsible for s=0 writes
                    mi = ts2rcLSE[0, 0, m][1]
                    if mi < self.tile_m:
                        sMaxValidSplit[mi] = max_valid_split[m]

            # ===============================
            # Step 5: Store final LSE to gmem
            # This writeback is the authoritative LSE_out returned by the
            # public Sparse Attention / Sparse Page Attention interface.
            # ===============================

            if const_expr(mLSE is not None):
                if const_expr(cu_seqlens is None):
                    mLSE_cur = mLSE[None, None, batch_idx]
                else:
                    mLSE_cur = cute.domain_offset((offset, 0), mLSE)
                if const_expr(mLSE_temperature is not None):
                    if const_expr(cu_seqlens is None):
                        mLSE_temperature_cur = mLSE_temperature[None, None, batch_idx]
                    else:
                        mLSE_temperature_cur = cute.domain_offset(
                            (offset, 0), mLSE_temperature)
                if k_block == 0:  # Only first k_block writes LSE when mLSE is provided
                    for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                        if ts2rcLSE[0, 0, m][0] == 0:  # Only thread responsible for s=0 writes
                            mi = ts2rcLSE[0, 0, m][1]
                            idx = m_block * self.tile_m + mi
                            if idx < max_idx:
                                m_idx, head_idx = self.decode_flat_row_idx(idx, head_divmod)
                                mLSE_cur[m_idx, head_idx] = final_lse[m]
                                if const_expr(mLSE_temperature is not None):
                                    mLSE_temperature_cur[m_idx, head_idx] = (
                                        final_lse_temperature[m])

            # ===============================
            # Step 6: Read O_partial and accumulate final O
            # ===============================

            cute.arch.sync_threads()

            if const_expr(self.use_tma_partial):
                # TMA stages are CTA-wide resources, so every consumer thread
                # must execute the same split count and barrier sequence.
                block_max_valid_split = Int32(-1)
                if tidx < Int32(self.tile_m):
                    block_max_valid_split = cute.arch.warp_reduction_max(
                        sMaxValidSplit[tidx], threads_in_group=self.tile_m
                    )
                    if tidx == 0:
                        sMaxValidSplit[0] = block_max_valid_split
                cute.arch.sync_threads()
                thr_max_valid_split = sMaxValidSplit[0]
            else:
                # cp.async loads are thread-private, so each thread can stop at
                # the largest valid split among its own rows.
                thr_max_valid_split = sMaxValidSplit[tOcO[0, 0, 0][0]]
                for m in cutlass.range(
                    1, cute.size(tOcO, mode=[1]), unroll_full=True
                ):
                    thr_max_valid_split = max(
                        thr_max_valid_split,
                        sMaxValidSplit[tOcO[0, m, 0][0]],
                    )

            tOrO_partial = cute.make_rmem_tensor_like(tOsO_partial[None, None, None, 0])
            tOrO = cute.make_rmem_tensor_like(tOrO_partial, Float32)
            tOrO.fill(0.0)

            stage_load = Int32(self.stages - 1)
            stage_compute = Int32(0)

            # Main accumulation loop
            for s in cutlass.range(thr_max_valid_split + 1, unroll=4):
                if const_expr(self.use_tma_partial):
                    stage_compute = s % Int32(self.stages)
                    stage_phase = (s // Int32(self.stages)) & Int32(1)
                    cute.arch.mbarrier_wait(
                        sO_tma_mbar_ptr + stage_compute,
                        stage_phase,
                    )

                # Get scales for this split
                scale = cute.make_rmem_tensor(num_rows, Float32)
                for m in cutlass.range(num_rows, unroll_full=True):
                    scale[m] = sLSE[s, tOcO[0, m, 0][0]]

                if const_expr(self.use_tma_partial):
                    cute.autovec_copy(
                        tOsO_partial[None, None, None, stage_compute],
                        tOrO_partial,
                    )
                else:
                    # Load the next thread-private cp.async stage.
                    split_to_load = s + self.stages - 1
                    if split_to_load <= thr_max_valid_split:
                        load_O_partial(split_to_load, stage_load)
                    cute.arch.cp_async_commit_group()
                    stage_load = (
                        0 if stage_load == self.stages - 1 else stage_load + 1
                    )
                    cute.arch.cp_async_wait_group(self.stages - 1)
                    cute.autovec_copy(
                        tOsO_partial[None, None, None, stage_compute],
                        tOrO_partial,
                    )
                    stage_compute = (
                        0
                        if stage_compute == self.stages - 1
                        else stage_compute + 1
                    )

                # Accumulate scaled partial results
                for m in cutlass.range(num_rows, unroll_full=True):
                    if tOhidx[m] >= 0 and scale[m] > 0.0:
                        tOrO[None, m, None].store(
                            tOrO[None, m, None].load()
                            + scale[m] * tOrO_partial[None, m, None].load().to(Float32)
                        )

                if const_expr(self.use_tma_partial):
                    # Match CUDA's CTA-wide stage handoff: no consumer may
                    # release a stage while another warp is still reading it.
                    cute.arch.sync_threads()
                    split_to_load = s + self.stages
                    if warp_idx == 0 and split_to_load <= thr_max_valid_split:
                        o_tma_barrier = sO_tma_mbar_ptr + stage_compute
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                o_tma_barrier,
                                o_tma_copy_bytes,
                            )
                        cute.copy(
                            tma_atom_O_partial,
                            tOgO_tma[None, split_to_load],
                            tOsO_tma[None, stage_compute],
                            tma_bar_ptr=o_tma_barrier,
                        )

            # Flush any outstanding async-copy groups before the local Step-7
            # permutation buffer is read on the tail of the kernel.
            if const_expr(not self.use_tma_partial):
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # ===============================
            # Step 7: Write final O to gmem (fake→real via SMEM)
            # ===============================

            if const_expr(cu_seqlens is None):
                mO_cur = mO[None, None, None, batch_idx]
            elif const_expr(flatten_varlen):
                mO_cur = mO
            else:
                mO_cur = cute.domain_offset((offset, 0, 0), mO)
            mO_cur = utils.domain_offset_aligned((0, k_block * self.k_block_size, 0), mO_cur)
            num_vals = const_expr(cute.size(tOcO, mode=[0]))
            if const_expr(not use_pdl):
                # Direct / standalone calls don't participate in the K1->K2
                # dependency chain. Use a simple per-element real-column store
                # path here to keep mixed-shape launches stable.
                for m in cutlass.range(num_rows, unroll_full=True):
                    if tOhidx[m] >= 0:
                        for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                            if const_expr(self.is_even_k) or tOpO[k]:
                                for v in cutlass.range(num_vals, unroll_full=True):
                                    fake_col = tOcO[v, 0, k][1]
                                    real_col = stg128_half_fake_col_to_real_col(fake_col)
                                    o_val = tOrO[v, m, k]
                                    if const_expr(mOutputScale is not None):
                                        o_val = o_val * output_scale
                                    mO_cur[tOmidx[m], real_col, tOhidx[m]] = o_val.to(self.dtype)
            elif const_expr(self.dtype in [cutlass.BFloat16, cutlass.Float16]):
                # The BF16/F16 fake layout is a 4x4 transpose of 32-bit
                # pairs inside each 32-column group. Transpose the pair
                # registers with two XOR-shuffle stages, then retain the
                # original coalesced 128-bit GMEM store.
                assert num_vals == 8
                gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                tOcO_store = gmem_thr_copy_O.partition_D(cO)
                rO = cute.make_rmem_tensor(tOcO_store.shape, self.dtype)
                elems_per_store = const_expr(
                    cute.size(gmem_tiled_copy_O.layout_tv_tiled[1])
                )
                num_store_rows = const_expr(cute.size(tOcO_store, mode=[1]))
                assert num_store_rows == num_rows
                lane_in_pair_group = tidx % Int32(4)

                for m in cutlass.range(num_store_rows, unroll_full=True):
                    for k in cutlass.range(
                        cute.size(tOcO_store, mode=[2]), unroll_full=True
                    ):
                        o0, o1 = tOrO[0, m, k], tOrO[1, m, k]
                        o2, o3 = tOrO[2, m, k], tOrO[3, m, k]
                        o4, o5 = tOrO[4, m, k], tOrO[5, m, k]
                        o6, o7 = tOrO[6, m, k], tOrO[7, m, k]
                        if const_expr(mOutputScale is not None):
                            o0, o1 = cute.arch.mul_packed_f32x2(
                                (o0, o1), (output_scale, output_scale)
                            )
                            o2, o3 = cute.arch.mul_packed_f32x2(
                                (o2, o3), (output_scale, output_scale)
                            )
                            o4, o5 = cute.arch.mul_packed_f32x2(
                                (o4, o5), (output_scale, output_scale)
                            )
                            o6, o7 = cute.arch.mul_packed_f32x2(
                                (o6, o7), (output_scale, output_scale)
                            )
                        word0 = utils.cvt_f16x2_f32(o0, o1, self.dtype)
                        word1 = utils.cvt_f16x2_f32(o2, o3, self.dtype)
                        word2 = utils.cvt_f16x2_f32(o4, o5, self.dtype)
                        word3 = utils.cvt_f16x2_f32(o6, o7, self.dtype)

                        partner0 = cute.arch.shuffle_sync_bfly(word1, offset=1)
                        partner1 = cute.arch.shuffle_sync_bfly(word0, offset=1)
                        partner2 = cute.arch.shuffle_sync_bfly(word3, offset=1)
                        partner3 = cute.arch.shuffle_sync_bfly(word2, offset=1)
                        stage0, stage1 = word0, word1
                        stage2, stage3 = word2, word3
                        if lane_in_pair_group % Int32(2) != Int32(0):
                            stage0, stage2 = partner0, partner2
                        if lane_in_pair_group % Int32(2) != Int32(1):
                            stage1, stage3 = partner1, partner3

                        partner0 = cute.arch.shuffle_sync_bfly(stage2, offset=2)
                        partner1 = cute.arch.shuffle_sync_bfly(stage3, offset=2)
                        partner2 = cute.arch.shuffle_sync_bfly(stage0, offset=2)
                        partner3 = cute.arch.shuffle_sync_bfly(stage1, offset=2)
                        out0, out1 = stage0, stage1
                        out2, out3 = stage2, stage3
                        if lane_in_pair_group // Int32(2) != Int32(0):
                            out0, out1 = partner0, partner1
                        if lane_in_pair_group // Int32(2) != Int32(1):
                            out2, out3 = partner2, partner3

                        rO_pairs = cute.recast_tensor(
                            rO[None, m, k], cutlass.Int32
                        )
                        rO_pairs[0] = out0
                        rO_pairs[1] = out1
                        rO_pairs[2] = out2
                        rO_pairs[3] = out3

                for m in cutlass.range(num_store_rows, unroll_full=True):
                    row_local = tOcO_store[0, m, 0][0]
                    idx = m_block * self.tile_m + row_local
                    if idx < max_idx:
                        m_idx, head_idx = self.decode_flat_row_idx(
                            idx, head_divmod
                        )
                        mO_cur_copy = cute.tiled_divide(
                            mO_cur[m_idx, None, head_idx], (elems_per_store,)
                        )
                        for k in cutlass.range(
                            cute.size(tOcO_store, mode=[2]), unroll_full=True
                        ):
                            k_idx = (
                                tOcO_store[0, 0, k][1] // elems_per_store
                            )
                            cute.copy(
                                gmem_thr_copy_O,
                                rO[None, m, k],
                                mO_cur_copy[None, k_idx],
                            )
            else:
                # 7a: fp32 accumulator -> output dtype SMEM with fake→real
                # permutation. The dedicated permutation buffer stays separate
                # from the O_partial pipeline staging buffer.
                sO_perm = sO_perm_buf

                if const_expr(self.dtype in [cutlass.BFloat16, cutlass.Float16]):
                    # O_partial uses a dtype-specific STG.128 fake layout, but
                    # sO_perm is in the final O dtype. For all supported fake
                    # layouts, adjacent fake pairs map to adjacent real columns,
                    # so write the final BF16/F16 O pair as one 32-bit SMEM store.
                    assert num_vals % 2 == 0
                    r2s_o_pair_atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        cutlass.Int32,
                        num_bits_per_copy=32,
                    )
                    rO_pair_word = cute.make_rmem_tensor((1,), cutlass.Int32)
                    sO_perm_i32_base = cute.make_ptr(
                        dtype=cutlass.Int32,
                        value=sO_perm.iterator.toint(),
                        mem_space=sO_perm.iterator.memspace,
                        assumed_align=4,
                    )
                    sO_perm_i32_row_stride = Int32((self.k_block_size + 16) // 2)
                    for m in cutlass.range(num_rows, unroll_full=True):
                        row_local = tOcO[0, m, 0][0]
                        if tOhidx[m] >= 0:
                            for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                                for v_pair in cutlass.range(num_vals // 2, unroll_full=True):
                                    v = v_pair * 2
                                    fake_col = tOcO[v, 0, k][1]
                                    real_col = stg128_half_fake_col_to_real_col(fake_col)
                                    o0 = tOrO[v, m, k]
                                    o1 = tOrO[v + 1, m, k]
                                    if const_expr(mOutputScale is not None):
                                        o0, o1 = cute.arch.mul_packed_f32x2(
                                            (o0, o1),
                                            (output_scale, output_scale),
                                        )
                                    rO_pair_word[0] = utils.cvt_f16x2_f32(o0, o1, self.dtype)
                                    smem_pair_ptr = cute.make_ptr(
                                        dtype=cutlass.Int32,
                                        value=(
                                            sO_perm_i32_base.toint()
                                            + Int64(
                                                row_local * sO_perm_i32_row_stride
                                                + real_col // Int32(2)
                                            )
                                            * Int64(4)
                                        ),
                                        mem_space=sO_perm.iterator.memspace,
                                        assumed_align=4,
                                    )
                                    sO_pair = cute.make_tensor(
                                        smem_pair_ptr,
                                        cute.make_layout((1,), stride=(1,)),
                                    )
                                    cute.copy(r2s_o_pair_atom, rO_pair_word, sO_pair)
                else:
                    # 7a: iterate over ALL val elements in mode[0].
                    # tOcO[v, m, k][1] gives different fake_col for each v.
                    r2s_o_scalar_atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.dtype,
                        num_bits_per_copy=self.dtype.width,
                    )
                    rO_scalar = cute.make_rmem_tensor((1,), self.dtype)
                    for m in cutlass.range(num_rows, unroll_full=True):
                        row_local = tOcO[0, m, 0][0]
                        if tOhidx[m] >= 0:
                            for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                                for v in cutlass.range(num_vals, unroll_full=True):
                                    fake_col = tOcO[v, 0, k][1]
                                    real_col = stg128_half_fake_col_to_real_col(fake_col)
                                    o_val = tOrO[v, m, k]
                                    if const_expr(mOutputScale is not None):
                                        o_val = o_val * output_scale
                                    rO_scalar[0] = o_val.to(self.dtype)
                                    smem_ptr = utils.elem_pointer(sO_perm, (row_local, real_col))
                                    smem_scalar_ptr = cute.make_ptr(
                                        dtype=self.dtype,
                                        value=smem_ptr.toint(),
                                        mem_space=sO_perm.iterator.memspace,
                                        assumed_align=self.dtype.width // 8,
                                    )
                                    sO_scalar = cute.make_tensor(
                                        smem_scalar_ptr,
                                        cute.make_layout((1,), stride=(1,)),
                                    )
                                    cute.copy(r2s_o_scalar_atom, rO_scalar, sO_scalar)

                cute.arch.sync_threads()

                # 7b: SMEM (real order, output dtype) → GMEM
                gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                tOcO_store = gmem_thr_copy_O.partition_D(cO)
                tOsO_store = gmem_thr_copy_O.partition_D(sO_perm)
                rO = cute.make_rmem_tensor(tOcO_store.shape, self.dtype)
                elems_per_store = const_expr(cute.size(gmem_tiled_copy_O.layout_tv_tiled[1]))
                num_store_rows = const_expr(cute.size(tOcO_store, mode=[1]))
                tOpO_store = None
                if const_expr(not self.is_even_k):
                    tOpO_store = cute.make_rmem_tensor(cute.size(tOcO_store, mode=[2]), Boolean)
                    for k in cutlass.range(cute.size(tOpO_store), unroll_full=True):
                        tOpO_store[k] = (
                            tOcO_store[0, 0, k][1]
                            < mO_partial.shape[1] - k_block * self.k_block_size
                        )

                # Read output dtype from SMEM (now in real column order).
                for m in cutlass.range(num_store_rows, unroll_full=True):
                    for k in cutlass.range(cute.size(tOcO_store, mode=[2]), unroll_full=True):
                        if const_expr(self.is_even_k) or tOpO_store[k]:
                            cute.autovec_copy(tOsO_store[None, m, k], rO[None, m, k])

                # Write bf16 to GMEM using gmem_tiled_copy_O (same as original FA Step 7)
                for m in cutlass.range(num_store_rows, unroll_full=True):
                    row_local = tOcO_store[0, m, 0][0]
                    idx = m_block * self.tile_m + row_local
                    if idx < max_idx:
                        m_idx, head_idx = self.decode_flat_row_idx(idx, head_divmod)
                        mO_cur_copy = cute.tiled_divide(
                            mO_cur[m_idx, None, head_idx], (elems_per_store,)
                        )
                        for k in cutlass.range(cute.size(tOcO_store, mode=[2]), unroll_full=True):
                            k_idx = tOcO_store[0, 0, k][1] // elems_per_store
                            if const_expr(self.is_even_k) or tOpO_store[k]:
                                cute.copy(gmem_thr_copy_O, rO[None, m, k], mO_cur_copy[None, k_idx])

    @cute.jit
    def load_O_partial(
        self,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        tOrOptr: cute.Tensor,
        tOsO_partial: cute.Tensor,
        tOhidx: cute.Tensor,
        tOSplitCount: cute.Tensor,
        tOpO: Optional[cute.Tensor],
        tOcO: cute.Tensor,
        mO_cur_partial_layout: cute.Layout,
        split: Int32,
        stage: Int32,
    ) -> None:
        elems_per_load = const_expr(cute.size(gmem_tiled_copy_O_partial.layout_tv_tiled[1]))
        tOsO_partial_cur = tOsO_partial[None, None, None, stage]
        for m in cutlass.range(cute.size(tOcO, [1]), unroll_full=True):
            if tOhidx[m] >= 0:
                o_gmem_ptr = cute.make_ptr(
                    tOsO_partial.element_type,
                    tOrOptr[m],
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                mO_partial_cur = cute.make_tensor(
                    o_gmem_ptr, cute.slice_(mO_cur_partial_layout, (0, None, None, 0))
                )
                mO_partial_cur_copy = cute.tiled_divide(mO_partial_cur, (elems_per_load,))
                for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                    k_idx = tOcO[0, 0, k][1] // elems_per_load
                    if split < tOSplitCount[m] and (const_expr(tOpO is None) or tOpO[k]):
                        cute.copy(
                            gmem_tiled_copy_O_partial,
                            mO_partial_cur_copy[None, k_idx, split],
                            tOsO_partial_cur[None, m, k],
                        )
                    else:
                        tOsO_partial_cur[None, m, k].fill(0)


def _get_cutlass_dtype(torch_dtype: torch.dtype):
    if torch_dtype not in torch2cute_dtype_map:
        raise TypeError(f"Unsupported dtype: {torch_dtype}")
    return torch2cute_dtype_map[torch_dtype]


_combine_compile_cache = {}


def _get_cpasync_smem_layout_atom(dtype: Type[cutlass.Numeric], k_dim: int) -> cute.ComposedLayout:
    dtype_byte = const_expr(dtype.width // 8)
    bytes_per_row = const_expr(k_dim * dtype_byte)
    smem_k_block_size = (
        const_expr(
            128
            if bytes_per_row % 128 == 0
            else (64 if bytes_per_row % 64 == 0 else (32 if bytes_per_row % 32 == 0 else 16))
        )
        // dtype_byte
    )
    swizzle_bits = (
        4
        if smem_k_block_size == 128
        else (3 if smem_k_block_size == 64 else (2 if smem_k_block_size == 32 else 1))
    )
    swizzle_base = 2 if dtype_byte == 4 else (3 if dtype_byte == 2 else 4)
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0,
        cute.make_ordered_layout(
            (8 if const_expr(k_dim % 32 == 0) else 16, smem_k_block_size),
            order=(1, 0),
        ),
    )


def combine(
    o_partial_fake,
    lse_partial,
    o_out,
    lse_out,
    *,
    lse_temperature_partial=None,
    lse_temperature_out=None,
    cu_seqlens=None,
    seqused=None,
    split_counts=None,
    output_scale=None,
    use_pdl=False,
    raw_partial_stats=False,
):
    """K2: merge sparse forward split partials into the final output.

    STG.128 fake-layout handling remains an internal implementation detail.
    When lse_out is provided, the kernel writes the final authoritative
    log-sum-exp for each query row/head directly into that tensor.

    Args:
        o_partial_fake:
            Batched: [num_splits, batch, Sq, head_q, dim]
            Varlen:  [num_splits, total_q, head_q, dim]
        lse_partial:
            Batched: [num_splits, batch, Sq, head_q]
            Varlen:  [num_splits, total_q, head_q]
        o_out:
            Batched: [batch, Sq, head_q, dim]
            Varlen:  [total_q, head_q, dim]
        lse_out:
            Batched: [batch, Sq, head_q]
            Varlen:  [total_q, head_q]
        lse_temperature_partial:
            Optional temperature-scaled LSE partial with the same shape as
            lse_partial.
        lse_temperature_out:
            Optional temperature-scaled final LSE with the same shape as
            lse_out.
        cu_seqlens: Optional [batch + 1] int32 for varlen-Q combine.
        seqused: Optional [batch] int32 effective lengths for combine.
        split_counts: Optional int32 rowwise valid split counts prepared from
            q2k metadata. Batched: [batch, seqlen, head_kv]. Varlen:
            [total_q, head_kv].
        output_scale: Optional fp32 tensor with at least one element. When
            provided, the final O accumulator is multiplied once before store.
        use_pdl: When True, wait on PDL dependencies from the producer K1
            kernel. When False, launch without PDL waits.
        raw_partial_stats: Interpret lse_partial as interleaved FP32
            (scaled_max_log2, row_sum) pairs and combine raw partial-O.
    """
    D = o_partial_fake.shape[-1]
    num_splits = o_partial_fake.shape[0]
    raw_partial_stats = bool(raw_partial_stats)
    return_temperature_lse = (
        lse_temperature_partial is not None or lse_temperature_out is not None
    )
    if (lse_temperature_partial is None) != (lse_temperature_out is None):
        raise ValueError(
            "lse_temperature_partial and lse_temperature_out must either both be provided or both be None"
        )
    if raw_partial_stats:
        expected_stats_shape = (*o_partial_fake.shape[:-1], 2)
        if tuple(lse_partial.shape) != tuple(expected_stats_shape):
            raise ValueError(
                "raw lse_partial must have shape "
                f"{expected_stats_shape}, got {tuple(lse_partial.shape)}"
            )
        if cu_seqlens is None or o_partial_fake.ndim != 4:
            raise ValueError("raw_partial_stats currently requires varlen partial tensors")
        if return_temperature_lse:
            raise ValueError("raw_partial_stats does not support independent temperature LSE")
    else:
        expected_lse_shape = o_partial_fake.shape[:-1]
        if tuple(lse_partial.shape) != tuple(expected_lse_shape):
            raise ValueError(
                "lse_partial shape must match o_partial_fake without head_dim, "
                f"got {tuple(lse_partial.shape)} vs {tuple(expected_lse_shape)}"
            )
    if lse_partial.dtype != torch.float32:
        raise TypeError(f"lse_partial must be torch.float32, got {lse_partial.dtype}")
    if lse_temperature_partial is not None and lse_temperature_partial.shape != lse_partial.shape:
        raise ValueError(
            "lse_temperature_partial must have the same shape as lse_partial, "
            f"got {lse_temperature_partial.shape} vs {lse_partial.shape}"
        )
    if lse_temperature_out is not None:
        if lse_out is None:
            raise ValueError("lse_temperature_out requires lse_out")
        if lse_temperature_out.shape != lse_out.shape:
            raise ValueError(
                "lse_temperature_out must have the same shape as lse_out, "
                f"got {lse_temperature_out.shape} vs {lse_out.shape}"
            )
        if lse_temperature_out.dtype != torch.float32 or lse_temperature_partial.dtype != torch.float32:
            raise TypeError("temperature LSE tensors must be torch.float32")

    if o_partial_fake.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            "o_partial_fake dtype must be torch.bfloat16 or torch.float16, "
            f"got {o_partial_fake.dtype}"
        )
    partial_dtype = _get_cutlass_dtype(o_partial_fake.dtype)
    out_dtype = _get_cutlass_dtype(o_out.dtype)
    if output_scale is not None:
        if output_scale.dtype != torch.float32:
            raise TypeError(f"output_scale must be torch.float32, got {output_scale.dtype}")
        if output_scale.numel() < 1:
            raise ValueError("output_scale must contain at least one element")
        if output_scale.device != o_out.device:
            raise ValueError("output_scale must be on the same device as o_out")
        output_scale = output_scale.contiguous()
    if split_counts is not None:
        if split_counts.dtype != torch.int32:
            raise TypeError(f"split_counts must be torch.int32, got {split_counts.dtype}")
        if o_out.ndim == 4:
            if split_counts.ndim != 3:
                raise ValueError(
                    f"batched split_counts must have shape [batch, seqlen, head_kv], got {split_counts.shape}"
                )
            if split_counts.shape[:2] != o_out.shape[:2]:
                raise ValueError(
                    f"split_counts shape {split_counts.shape} must match batch/seqlen of o_out {o_out.shape}"
                )
        else:
            if cu_seqlens is None:
                raise ValueError("split_counts with varlen output requires cu_seqlens")
            if split_counts.ndim != 2:
                raise ValueError(
                    f"varlen split_counts must have shape [total_q, head_kv], got {split_counts.shape}"
                )
            if split_counts.shape[0] != o_out.shape[0]:
                raise ValueError(
                    f"split_counts total_q ({split_counts.shape[0]}) must match o_out total_q "
                    f"({o_out.shape[0]})"
                )
        if o_out.shape[-2] % split_counts.shape[-1] != 0:
            raise ValueError(
                f"o_out heads ({o_out.shape[-2]}) must be divisible by split_counts heads ({split_counts.shape[-1]})"
            )
        qheadperkv = o_out.shape[-2] // split_counts.shape[-1]
    else:
        qheadperkv = 1
    if cu_seqlens is not None:
        if cu_seqlens.dtype != torch.int32:
            raise TypeError(f"cu_seqlens must be torch.int32, got {cu_seqlens.dtype}")
        if cu_seqlens.ndim != 1:
            raise ValueError(f"cu_seqlens must be rank-1, got {cu_seqlens.shape}")
        if not cu_seqlens.is_contiguous():
            raise ValueError("cu_seqlens must be contiguous")
    if seqused is not None:
        if seqused.dtype != torch.int32:
            raise TypeError(f"seqused must be torch.int32, got {seqused.dtype}")
        if seqused.ndim != 1:
            raise ValueError(f"seqused must be rank-1, got {seqused.shape}")
        if not seqused.is_contiguous():
            raise ValueError("seqused must be contiguous")

    k_block_size = 128 if D > 64 else 64
    tile_m = 64 if raw_partial_stats else 32
    num_threads = tile_m * 4
    has_cu_seqlens = cu_seqlens is not None
    has_seqused = seqused is not None
    has_lse = lse_out is not None
    has_split_counts = split_counts is not None
    has_output_scale = output_scale is not None
    min_blocks_per_mp = 3 if has_output_scale and use_pdl else 0

    capability = torch.cuda.get_device_capability(o_out.device)
    stages = 4 if capability == (10, 7) else 2
    key = (
        "combine",
        capability,
        stages,
        D,
        k_block_size,
        tile_m,
        num_threads,
        num_splits,
        partial_dtype,
        out_dtype,
        has_cu_seqlens,
        has_seqused,
        has_lse,
        bool(return_temperature_lse),
        has_split_counts,
        has_output_scale,
        use_pdl,
        min_blocks_per_mp,
        raw_partial_stats,
    )
    if key not in _combine_compile_cache:
        loaded = try_load_aot(key)
        if loaded is not None:
            _combine_compile_cache[key] = loaded
        else:
            from inference.msa_v1.attention.prefill._common.dsl.compile_utils import make_fake_tensor

            kernel = SparseAttentionForwardCombine(
                dtype=out_dtype,
                dtype_partial=partial_dtype,
                head_dim=D,
                tile_m=tile_m,
                k_block_size=k_block_size,
                topk=num_splits,
                num_threads=num_threads,
                use_pdl=use_pdl,
                min_blocks_per_mp=min_blocks_per_mp,
                stages=stages,
                raw_partial_stats=raw_partial_stats,
            )
            div = 128 // partial_dtype.width
            if has_cu_seqlens:
                total_q, nheads = (cute.sym_int64() for _ in range(2))
                mO_partial = make_fake_tensor(
                    partial_dtype, (num_splits, total_q, nheads, D), divisibility=div
                )
                mLSE_partial = make_fake_tensor(
                    Float32,
                    (
                        (num_splits, total_q, nheads, 2)
                        if raw_partial_stats
                        else (num_splits, total_q, nheads)
                    ),
                    divisibility=2 if raw_partial_stats else 1,
                    leading_dim=3 if raw_partial_stats else 2,
                )
                mO = make_fake_tensor(
                    out_dtype, (total_q, nheads, D), divisibility=128 // out_dtype.width
                )
                mLSE = (
                    make_fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=1)
                    if has_lse
                    else None
                )
                mLSE_temperature_partial = (
                    make_fake_tensor(
                        Float32, (num_splits, total_q, nheads), divisibility=1, leading_dim=2
                    )
                    if return_temperature_lse
                    else None
                )
                mLSE_temperature = (
                    make_fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=1)
                    if return_temperature_lse
                    else None
                )
            else:
                batch, sq, nheads = (cute.sym_int64() for _ in range(3))
                mO_partial = make_fake_tensor(
                    partial_dtype, (num_splits, batch, sq, nheads, D), divisibility=div
                )
                mLSE_partial = make_fake_tensor(
                    Float32, (num_splits, batch, sq, nheads), divisibility=1, leading_dim=3
                )
                mO = make_fake_tensor(
                    out_dtype, (batch, sq, nheads, D), divisibility=128 // out_dtype.width
                )
                mLSE = (
                    make_fake_tensor(Float32, (batch, sq, nheads), divisibility=1, leading_dim=2)
                    if has_lse
                    else None
                )
                mLSE_temperature_partial = (
                    make_fake_tensor(
                        Float32, (num_splits, batch, sq, nheads), divisibility=1, leading_dim=3
                    )
                    if return_temperature_lse
                    else None
                )
                mLSE_temperature = (
                    make_fake_tensor(Float32, (batch, sq, nheads), divisibility=1, leading_dim=2)
                    if return_temperature_lse
                    else None
                )
            if not has_split_counts:
                mSplitCounts = None
            elif has_cu_seqlens:
                total_q_ctr, nheads_kv = (cute.sym_int64() for _ in range(2))
                mSplitCounts = make_fake_tensor(
                    Int32, (total_q_ctr, nheads_kv), divisibility=1, leading_dim=1
                )
            else:
                nheads_kv = cute.sym_int64()
                mSplitCounts = make_fake_tensor(
                    Int32, (batch, sq, nheads_kv), divisibility=1, leading_dim=2
                )
            mOutputScale = (
                make_fake_tensor(Float32, (cute.sym_int64(),), divisibility=1, leading_dim=0)
                if has_output_scale
                else None
            )
            stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

            _combine_compile_cache[key] = compile_with_timing(
                kernel,
                mO_partial,
                mLSE_partial,
                mO,
                mLSE,
                mLSE_temperature_partial,
                mLSE_temperature,
                None
                if cu_seqlens is None
                else make_fake_tensor(Int32, (cute.sym_int64(),), divisibility=1, leading_dim=0),
                None
                if seqused is None
                else make_fake_tensor(Int32, (cute.sym_int64(),), divisibility=1, leading_dim=0),
                None,
                None,
                None,
                mSplitCounts,
                mOutputScale,
                Int32(qheadperkv),
                stream,
                options="--enable-tvm-ffi",
            )
            save_aot(key, _combine_compile_cache[key])

    with torch.cuda.nvtx.range("K2_Combine"):
        _combine_compile_cache[key](
            o_partial_fake,
            lse_partial,
            o_out,
            lse_out,
            lse_temperature_partial,
            lse_temperature_out,
            cu_seqlens,
            seqused,
            None,
            None,
            None,
            split_counts,
            output_scale,
            qheadperkv,
        )
