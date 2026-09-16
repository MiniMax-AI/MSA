"""CuTe DSL ordered-key helpers for MSA v1 FP16 indexer scores."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm


def _half_as_uint16(value: cutlass.Float16) -> cutlass.Uint16:
    """Return the IEEE-754 bits of an FP16 register."""

    return cutlass.Uint16(llvm.bitcast(cutlass.Uint16.mlir_type, value.ir_value()))


def _uint16_as_half(value: cutlass.Uint16) -> cutlass.Float16:
    """Reinterpret one unsigned 16-bit register as FP16."""

    return cutlass.Float16(llvm.bitcast(cutlass.Float16.mlir_type, value.ir_value()))


@cute.jit
def pack_fp16_topk_key(
    score: cutlass.Float16,
    block_id: cutlass.Int32,
) -> cutlass.Uint32:
    """Pack score-descending/id-ascending order into one unsigned key."""

    bits = _half_as_uint16(score)
    magnitude = bits & cutlass.Uint16(0x7FFF)
    if magnitude > cutlass.Uint16(0x7C00):
        bits = cutlass.Uint16(0xFC00)
    elif magnitude == cutlass.Uint16(0):
        bits = cutlass.Uint16(0)

    ordered = cutlass.Uint16(0)
    if bits & cutlass.Uint16(0x8000):
        ordered = bits ^ cutlass.Uint16(0xFFFF)
    else:
        ordered = bits | cutlass.Uint16(0x8000)

    id_key = (~cutlass.Uint32(block_id)) & cutlass.Uint32(0xFFFF)
    return (cutlass.Uint32(ordered) << cutlass.Uint32(16)) | id_key


@cute.jit
def fp16_from_topk_key(key: cutlass.Uint32) -> cutlass.Float16:
    """Recover the canonical FP16 score encoded in a packed TopK key."""

    ordered = cutlass.Uint16(key >> cutlass.Uint32(16))
    bits = cutlass.Uint16(0)
    if ordered & cutlass.Uint16(0x8000):
        bits = ordered & cutlass.Uint16(0x7FFF)
    else:
        bits = ordered ^ cutlass.Uint16(0xFFFF)
    return _uint16_as_half(bits)


__all__ = ["fp16_from_topk_key", "pack_fp16_topk_key"]
