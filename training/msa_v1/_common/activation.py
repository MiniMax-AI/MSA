"""Activation helpers shared by legacy SM100 kernels."""

from functools import partial

import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm

sub_packed_f32x2 = partial(
    cute.arch.calc_packed_f32x2_op,
    src_c=None,
    calc_func=nvvm.sub_packed_f32x2,
)


__all__ = ["sub_packed_f32x2"]
