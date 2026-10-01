// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"

#if !defined(__CUDACC_VER_MAJOR__) || \
    (__CUDACC_VER_MAJOR__ < 13) || \
    (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ < 4)
#error "Q8KV4 prefill attention QMUL4 requires CUDA Toolkit 13.4 or newer"
#endif

namespace fmha_sm100::prefill_q8kv4::sm100::common {

CUTLASS_DEVICE void nvfp4_to_e4m3x4(
    uint32_t& output, uint16_t packed_fp4, uint32_t scale_e4m3x4) {
  asm volatile(
      "mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, %1, %2;"
      : "=r"(output)
      : "h"(packed_fp4), "r"(scale_e4m3x4));
}

}  // namespace fmha_sm100::prefill_q8kv4::sm100::common
