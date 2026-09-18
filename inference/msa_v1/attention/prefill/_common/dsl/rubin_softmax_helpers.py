# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Rubin packed-FP16 softmax helpers.

Datapath (FP8 Q/K, FP8 P/V only):

    t  = fma_f32x2(score, scale_log2, -row_max * scale_log2)   # cute.arch
    x  = cvt_packed_f16x2_f32x2(t)                             # cvt.rn.f16x2.f32
    p  = ex2_packed_f16x2(x)                                   # ex2.approx.f16x2
    p  = mul_packed_f16x2(p, 448)                              # mul.f16x2, MSA P448 scale
    P  = cvt_f16x4_to_f8x4(p)                                  # cvt.rn.satfinite.e4m3x2.f16x2
    sum = add_packed_f32x2_f16x2_f32x2(p, sum)                 # add.f32x2.f16x2.f32x2, FP32 accumulate

MSA applies the FP8 probability scale (448) *after* the exponential rather than
folding log2(448) into the FP32 bias as the FP32 path does: the FP16 rounding
of the exponent argument is ~16x coarser at |x| ~ 8.8 than near zero, and the
single-token row (x = 0) must produce exactly 448 so the LSE is exactly zero.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float16, Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir as _mlir_ir
from cutlass._mlir.dialects import llvm, vector


@dsl_user_op
def _f32x2_pair_to_i64(a: Float32, b: Float32, *, loc=None, ip=None):
    """Pack 2 Float32 values into a single .b64 (Int64) via vector<2xf32>."""
    vec_ty = _mlir_ir.VectorType.get([2], Float32.mlir_type, loc=loc)
    vec = vector.from_elements(
        vec_ty,
        [
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
        ],
        loc=loc,
        ip=ip,
    )
    return Int64(llvm.bitcast(Int64.mlir_type, vec, loc=loc, ip=ip))


@dsl_user_op
def _i64_to_f32x2_pair(val_i64, *, loc=None, ip=None):
    """Unpack .b64 (Int64) into 2 Float32 via vector<2xf32>."""
    vec_ty = _mlir_ir.VectorType.get([2], Float32.mlir_type, loc=loc)
    vec = llvm.bitcast(
        vec_ty,
        Int64(val_i64).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return (
        Float32(vector.extract(vec, [], [0], loc=loc, ip=ip)),
        Float32(vector.extract(vec, [], [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _f16x2_pair_to_i32(a: Float16, b: Float16, *, loc=None, ip=None):
    """Pack 2 Float16 values into a single .b32 (Int32) via vector<2xf16>."""
    vec_ty = _mlir_ir.VectorType.get([2], Float16.mlir_type, loc=loc)
    vec = vector.from_elements(
        vec_ty,
        [
            Float16(a).ir_value(loc=loc, ip=ip),
            Float16(b).ir_value(loc=loc, ip=ip),
        ],
        loc=loc,
        ip=ip,
    )
    return Int32(llvm.bitcast(Int32.mlir_type, vec, loc=loc, ip=ip))


@dsl_user_op
def _i32_to_f16x2_pair(val_i32, *, loc=None, ip=None):
    """Unpack .b32 (Int32) into 2 Float16 via vector<2xf16>."""
    vec_ty = _mlir_ir.VectorType.get([2], Float16.mlir_type, loc=loc)
    vec = llvm.bitcast(
        vec_ty,
        Int32(val_i32).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return (
        Float16(vector.extract(vec, [], [0], loc=loc, ip=ip)),
        Float16(vector.extract(vec, [], [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def cvt_packed_f16x2_f32x2(src_a, *, loc=None, ip=None):
    """Round two FP32 values into one packed FP16 pair."""
    a0, a1 = src_a
    res_i32 = llvm.inline_asm(
        T.i32(),
        [
            Float32(a0).ir_value(loc=loc, ip=ip),
            Float32(a1).ir_value(loc=loc, ip=ip),
        ],
        "cvt.rn.f16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return _i32_to_f16x2_pair(Int32(res_i32), loc=loc, ip=ip)


@dsl_user_op
def mul_packed_f16x2(src_a, src_b, *, loc=None, ip=None):
    """Multiply 2 packed FP16 pairs and return a packed FP16 pair."""
    a0, a1 = src_a
    b0, b1 = src_b
    a_i32 = _f16x2_pair_to_i32(a0, a1, loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    b_i32 = _f16x2_pair_to_i32(b0, b1, loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    res_i32 = llvm.inline_asm(
        T.i32(),
        [a_i32, b_i32],
        "mul.f16x2 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return _i32_to_f16x2_pair(Int32(res_i32), loc=loc, ip=ip)


@dsl_user_op
def ex2_packed_f16x2(src_a, *, loc=None, ip=None):
    """Compute approximate base-2 exp on a packed FP16 pair."""
    a0, a1 = src_a
    a_i32 = _f16x2_pair_to_i32(a0, a1, loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    res_i32 = llvm.inline_asm(
        T.i32(),
        [a_i32],
        "ex2.approx.f16x2 $0, $1;",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return _i32_to_f16x2_pair(Int32(res_i32), loc=loc, ip=ip)


@dsl_user_op
def add_packed_f32x2_f16x2_f32x2(src_a, src_b, *, loc=None, ip=None):
    """Add packed FP16 and FP32 pairs and return a packed FP32 pair."""
    a0, a1 = src_a  # Float16
    b0, b1 = src_b  # Float32
    a_i32 = _f16x2_pair_to_i32(a0, a1, loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    b_i64 = _f32x2_pair_to_i64(b0, b1, loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    res_i64 = llvm.inline_asm(
        T.i64(),
        [a_i32, b_i64],
        "add.f32x2.f16x2.f32x2 $0, $1, $2;",
        "=l,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return _i64_to_f32x2_pair(Int64(res_i64), loc=loc, ip=ip)


@cute.jit
def cvt_f16x4_to_f8x4_pack_i32(fp16x4_tensor, fp8_type, *, loc=None, ip=None):
    """Convert an FP16x4 tensor to a packed FP8x4 Int32 value."""
    fp16x4 = fp16x4_tensor.load()
    src_vec = fp16x4.ir_value(loc=loc, ip=ip) if hasattr(fp16x4, "ir_value") else fp16x4
    # Pack adjacent fp16 pairs into two .b32 registers (f16x2).
    h0 = Float16(vector.extract(src_vec, [], [0]))
    h1 = Float16(vector.extract(src_vec, [], [1]))
    h2 = Float16(vector.extract(src_vec, [], [2]))
    h3 = Float16(vector.extract(src_vec, [], [3]))
    lo_i32 = _f16x2_pair_to_i32(h0, h1, loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    hi_i32 = _f16x2_pair_to_i32(h2, h3, loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    cvt_instruction = ""
    if cutlass.const_expr(fp8_type == cutlass.Float8E4M3FN):
        cvt_instruction = "cvt.rn.satfinite.e4m3x2.f16x2"
    else:
        assert False, "Unsupported fp8 element type"
    asm_tmpl = (
        "{\n"
        "  .reg .b16 lo;\n"
        "  .reg .b16 hi;\n"
        f"  {cvt_instruction} lo, $1;\n"
        f"  {cvt_instruction} hi, $2;\n"
        "  mov.b32 $0, {lo, hi};\n"
        "}"
    )
    packed_i32 = llvm.inline_asm(
        T.i32(),
        [lo_i32, hi_i32],
        asm_tmpl,
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return packed_i32


@cute.jit
def cvt_f16x4_to_f8x4(fp16x4_tensor, fp8x4_tensor, *, loc=None, ip=None):
    """Store an FP16x4 tensor into an FP8x4 tensor."""
    packed_i32 = cvt_f16x4_to_f8x4_pack_i32(fp16x4_tensor, fp8x4_tensor.element_type)
    fp8x4_i32 = cute.recast_tensor(fp8x4_tensor, cutlass.Int32)
    fp8x4_i32[0] = cutlass.Int32(packed_i32)
    return


__all__ = [
    "cvt_packed_f16x2_f32x2",
    "mul_packed_f16x2",
    "ex2_packed_f16x2",
    "add_packed_f32x2_f16x2_f32x2",
    "cvt_f16x4_to_f8x4_pack_i32",
    "cvt_f16x4_to_f8x4",
]
