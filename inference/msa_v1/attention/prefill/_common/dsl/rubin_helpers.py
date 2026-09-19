# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Rubin-specific helpers for manually issued dense FP8 UTCMMA operations."""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr
from cutlass._mlir.dialects import llvm

from . import mma_sm100_desc as sm100_desc


def make_instr_desc(
    a_type,
    b_type,
    c_type,
    M: int,
    N: int,
    K: int,
    a_major: sm100_desc.Major,
    b_major: sm100_desc.Major,
    a_neg: sm100_desc.ScaleIn = sm100_desc.ScaleIn.One,
    b_neg: sm100_desc.ScaleIn = sm100_desc.ScaleIn.One,
    max_shift: sm100_desc.MaxShift = sm100_desc.MaxShift.NoShift,
) -> int:
    """Build a dense SM107 FP8 UTCMMA instruction descriptor.

    Unlike SM100, SM107 encodes the FP8 instruction K dimension explicitly:
    bit 29 selects K32 (0) or K64 (1).
    """
    if a_type not in (cutlass.Float8E4M3FN, cutlass.Float8E5M2):
        raise TypeError("SM107 descriptor requires an FP8 A operand")
    if b_type not in (cutlass.Float8E4M3FN, cutlass.Float8E5M2):
        raise TypeError("SM107 descriptor requires an FP8 B operand")
    if M not in (64, 128, 256):
        raise ValueError("SM107 M must be 64, 128 or 256")
    if N < 8 or N > 256 or (N & 7):
        raise ValueError("SM107 N must be a multiple of 8 in the range 8…256")
    if K not in (32, 64):
        raise ValueError("SM107 dense FP8 K must be 32 or 64")

    a_fmt = int(sm100_desc.to_UMMA_format(a_type))
    b_fmt = int(sm100_desc.to_UMMA_format(b_type))
    c_fmt = int(sm100_desc.to_C_format(c_type))
    m_dim = M >> 5
    n_dim = N >> 3
    k_size = K >> 6

    # fmt: off
    desc = 0
    desc |= (c_fmt          & 0x3) << 4   # c_format
    desc |= (a_fmt          & 0x7) << 7   # a_format
    desc |= (b_fmt          & 0x7) << 10  # b_format
    desc |= (int(a_neg)     & 0x1) << 13  # a_negate
    desc |= (int(b_neg)     & 0x1) << 14  # b_negate
    desc |= (int(a_major)   & 0x1) << 15  # a_major
    desc |= (int(b_major)   & 0x1) << 16  # b_major
    desc |= (n_dim          & 0x3F) << 17 # n_dim
    desc |= (m_dim          & 0xF) << 25  # m_dim
    desc |= (k_size         & 0x1) << 29  # FP8 K32/K64 selector
    desc |= (int(max_shift) & 0x3) << 30  # max_shift
    # fmt: on

    return desc & 0xFFFF_FFFF


def mma_op_to_idesc(op: cute.nvgpu.tcgen05.mma.MmaOp) -> int:
    return make_instr_desc(
        op.a_dtype,
        op.b_dtype,
        op.acc_dtype,
        op.shape_mnk[0],
        op.shape_mnk[1],
        op.shape_mnk[2],
        sm100_desc.Major.K
        if op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
        else sm100_desc.Major.MN,
        sm100_desc.Major.K
        if op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
        else sm100_desc.Major.MN,
    )


def i64_to_i32x2(i: int) -> Tuple[int, int]:
    return i & 0xFFFF_FFFF, (i >> 32) & 0xFFFF_FFFF


@cute.jit
def gemm_ptx_partial(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: Int32,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    mbar_ptr: Optional[cutlass.Pointer] = None,
    mbar_phase: Optional[Int32] = None,
    split_arrive: Optional[int] = None,
    zero_init: bool | Boolean = False,
    tA_addr: Optional[Int32] = None,
    cta_group: int = 1,
    mma_kind: str = "f8f6f4",
) -> None:
    """Issue SM107 FP8 MMA slices with an optional barrier between slices."""
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else tCrA.layout
    sB_layout = sB.layout
    idesc: int = const_expr(mma_op_to_idesc(op))
    if const_expr(not is_ts):
        sA_swizzle = sA.iterator.type.swizzle_type
        smem_desc_base_a: int = const_expr(
            sm100_desc.make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                sm100_desc.Major.K
                if const_expr(
                    op.a_major_mode
                    == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
                )
                else sm100_desc.Major.MN,
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    sB_swizzle = sB.iterator.type.swizzle_type
    smem_desc_base_b: int = const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            sm100_desc.Major.K
            if const_expr(
                op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
            )
            else sm100_desc.Major.MN,
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = const_expr(smem_desc_b_hi)

    tCrA_layout = (
        tCrA.layout
        if const_expr(not is_ts)
        else cute.recast_layout(32, tCrA.element_type.width, tCrA.layout)
    )
    offset_a = [
        cute.crd2idx((0, 0, k), tCrA_layout)
        for k in range(cute.size(tCrA.shape[2]))
    ]
    offset_b = [
        cute.crd2idx((0, 0, k), tCrB.layout)
        for k in range(cute.size(tCrB.shape[2]))
    ]

    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(
            smem_desc_base_a_lo
            | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator)
        )
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = Int32(
        smem_desc_base_b_lo
        | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator)
    )
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        assert mbar_ptr is None, "mbar_ptr must be None when a_src is not TMEM"
        llvm.inline_asm(
            None,
            [
                Int32(
                    cute.arch.make_warp_uniform(smem_desc_start_a_lo)
                ).ir_value(),
                Int32(
                    cute.arch.make_warp_uniform(smem_desc_start_b_lo)
                ).ir_value(),
                Int32(not zero_init).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo_start, smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            "mov.b32 tmem_acc, $3;\n\t"
            "mov.b32 smem_desc_a_lo_start, $0;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            "mov.b64 smem_desc_a, "
            "{smem_desc_a_lo_start, smem_desc_a_hi};\n\t"
            "mov.b64 smem_desc_b, "
            "{smem_desc_b_lo_start, smem_desc_b_hi};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{mma_kind} "
            f"[tmem_acc], smem_desc_a, smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    "add.u32 smem_desc_a_lo, smem_desc_a_lo_start, "
                    f"{hex(offset_a[k])};\n\t"
                    "add.u32 smem_desc_b_lo, smem_desc_b_lo_start, "
                    f"{hex(offset_b[k])};\n\t"
                    "mov.b64 smem_desc_a, "
                    "{smem_desc_a_lo, smem_desc_a_hi};\n\t"
                    "mov.b64 smem_desc_b, "
                    "{smem_desc_b_lo, smem_desc_b_hi};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{mma_kind} "
                    "[tmem_acc], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        # The TMEM fragment iterator does not preserve its address here.
        tA_addr = (
            tCrA[None, None, 0].iterator.toint() if tA_addr is None else tA_addr
        )
        input_args = [
            Int32(cute.arch.make_warp_uniform(tA_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(not zero_init).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ]
        if const_expr(mbar_ptr is not None):
            assert mbar_phase is not None, (
                "mbar_phase must be provided when mbar_ptr is not None"
            )
            assert split_arrive is not None, (
                "split_arrive must be provided when mbar_ptr is not None"
            )
            split_arrive_idx = split_arrive // op.shape_mnk[2]
            input_args.append(mbar_ptr.toint().ir_value())
            input_args.append(Int32(mbar_phase).ir_value())
            mbar_wait_str = (
                ".reg .pred P1; \n\t"
                "LAB_WAIT: \n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, "
                "[$4], $5, 10000000; \n\t"
                "@P1 bra DONE; \n\t"
                "bra     LAB_WAIT; \n\t"
                "DONE: \n\t"
            )
        else:
            mbar_wait_str = ""
        llvm.inline_asm(
            None,
            input_args,
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            "mov.b32 tmem_acc, $3;\n\t"
            "mov.b32 tmem_a, $0;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            "mov.b64 smem_desc_b, "
            "{smem_desc_b_lo_start, smem_desc_b_hi};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{mma_kind} "
            f"[tmem_acc], [tmem_a], smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    "add.u32 smem_desc_b_lo, smem_desc_b_lo_start, "
                    f"{hex(offset_b[k])};\n\t"
                    "mov.b64 smem_desc_b, "
                    "{smem_desc_b_lo, smem_desc_b_hi};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{mma_kind} "
                    f"[tmem_acc], [tmem_a + {hex(offset_a[k])}], "
                    "smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(
                    1,
                    cute.size(tCrA.shape[2])
                    if const_expr(mbar_ptr is None)
                    else split_arrive_idx,
                )
            )
            + mbar_wait_str
            + (
                "".join(
                    (
                        "add.u32 smem_desc_b_lo, smem_desc_b_lo_start, "
                        f"{hex(offset_b[k])};\n\t"
                        "mov.b64 smem_desc_b, "
                        "{smem_desc_b_lo, smem_desc_b_hi};\n\t"
                        f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{mma_kind} "
                        f"[tmem_acc], [tmem_a + {hex(offset_a[k])}], "
                        "smem_desc_b, idesc, 1;\n\t"
                    )
                    for k in range(split_arrive_idx, cute.size(tCrA.shape[2]))
                )
                if const_expr(mbar_ptr is not None)
                else ""
            )
            + "}\n",
            "r,r,r,r" if const_expr(mbar_ptr is None) else "r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def declare_ptx_idesc(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    var_name: str = "idesc",
) -> None:
    idesc = const_expr(mma_op_to_idesc(op))
    llvm.inline_asm(
        None,
        [],
        f".reg .b32 {var_name};\n\t"
        f"mov.b32 {var_name}, {hex(idesc)};\n\t",
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
