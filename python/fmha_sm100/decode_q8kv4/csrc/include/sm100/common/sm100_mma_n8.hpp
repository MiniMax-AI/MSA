// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "cute/arch/mma_sm100_umma.hpp"

namespace cutlass::fmha::collective {

// Public tcgen05 supports M128 N8; the upstream TS atom still requires N >= 16.
// Keep descriptor construction in CUTLASS and bridge only this instruction shape.
struct Sm100MmaF8TsM128N8 {
  CUTE_HOST_DEVICE static void fma(uint32_t tmem_a, uint64_t desc_b, uint32_t tmem_c,
                                   uint32_t accumulate, uint64_t instr_desc) {
#if defined(CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t const zero_mask = 0;
      asm volatile("{\n\t"
                   ".reg .pred p;\n\t"
                   "setp.ne.b32 p, %4, 0;\n\t"
                   "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, "
                   "{%5, %5, %5, %5}, p;\n\t"
                   "}\n"
                   :
                   : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(instr_desc >> 32)),
                     "r"(accumulate), "r"(zero_mask));
    }
#else
    CUTE_INVALID_CONTROL_PATH("M128 N8 FP8 MMA requires tcgen05 f8f6f4 support.");
#endif
  }
};

} // namespace cutlass::fmha::collective
