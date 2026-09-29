// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"

#if !defined(MINIMAX_MSA_Q8KV4_HAS_QMUL4)
#error "The Q8KV4 JIT must define MINIMAX_MSA_Q8KV4_HAS_QMUL4"
#endif

namespace fmha_sm100::decode_q8kv4::sm100::common {

#if MINIMAX_MSA_Q8KV4_HAS_QMUL4
CUTLASS_DEVICE
void nvfp4_e2m1x4_mul_e4m3x4(uint32_t &dst, uint16_t packed_e2m1x4, uint32_t scale_e4m3x4) {
  asm volatile("mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, %1, %2;"
               : "=r"(dst)
               : "h"(packed_e2m1x4), "r"(scale_e4m3x4));
}
#endif

} // namespace fmha_sm100::decode_q8kv4::sm100::common
