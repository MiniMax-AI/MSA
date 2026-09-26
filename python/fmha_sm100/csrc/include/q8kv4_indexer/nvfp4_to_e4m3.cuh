// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "cute/arch/util.hpp"

// QMUL4 is public PTX from CUDA 13.4 and only for the arch-specific SM100/SM103
// targets. The family target and older toolkits use the exact FP16 path.
#if defined(__CUDA_ARCH__) &&                                                                      \
    (defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)) &&                \
    (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
#define Q8KV4_INDEXER_HAS_QMUL4 1
#else
#define Q8KV4_INDEXER_HAS_QMUL4 0
#endif

namespace q8kv4_indexer::detail {

#if Q8KV4_INDEXER_HAS_QMUL4
CUTE_DEVICE uint32_t nvfp4_to_e4m3x4(uint16_t packed_fp4, uint32_t scale_e4m3x4) {
  uint32_t output;
  asm volatile("{ .reg .b16 value;\n"
               "  mov.b16 value, %1;\n"
               "  mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, value, %2; }\n"
               : "=r"(output)
               : "h"(packed_fp4), "r"(scale_e4m3x4));
  return output;
}
#else
CUTE_DEVICE void nvfp4x8_to_f16x2x4(uint32_t &output0, uint32_t &output1, uint32_t &output2,
                                    uint32_t &output3, uint32_t packed_fp4x8,
                                    uint16_t scale_e4m3x2) {
  asm volatile("{\n"
               ".reg .b32 scale_f16x2;\n"
               ".reg .b8 e2m1_0, e2m1_1, e2m1_2, e2m1_3;\n"
               ".reg .b32 f16_0, f16_1, f16_2, f16_3;\n"
               ".reg .b16 e4m3_0, e4m3_1, e4m3_2, e4m3_3;\n"
               "cvt.rn.f16x2.e4m3x2 scale_f16x2, %5;\n"
               "mov.b32 {e2m1_0, e2m1_1, e2m1_2, e2m1_3}, %4;\n"
               "cvt.rn.f16x2.e2m1x2 f16_0, e2m1_0;\n"
               "cvt.rn.f16x2.e2m1x2 f16_1, e2m1_1;\n"
               "cvt.rn.f16x2.e2m1x2 f16_2, e2m1_2;\n"
               "cvt.rn.f16x2.e2m1x2 f16_3, e2m1_3;\n"
               "mul.rn.f16x2 f16_0, f16_0, scale_f16x2;\n"
               "mul.rn.f16x2 f16_1, f16_1, scale_f16x2;\n"
               "mul.rn.f16x2 f16_2, f16_2, scale_f16x2;\n"
               "mul.rn.f16x2 f16_3, f16_3, scale_f16x2;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e4m3_0, f16_0;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e4m3_1, f16_1;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e4m3_2, f16_2;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 e4m3_3, f16_3;\n"
               "cvt.rn.f16x2.e4m3x2 %0, e4m3_0;\n"
               "cvt.rn.f16x2.e4m3x2 %1, e4m3_1;\n"
               "cvt.rn.f16x2.e4m3x2 %2, e4m3_2;\n"
               "cvt.rn.f16x2.e4m3x2 %3, e4m3_3;\n"
               "}\n"
               : "=&r"(output0), "=&r"(output1), "=&r"(output2), "=&r"(output3)
               : "r"(packed_fp4x8), "h"(scale_e4m3x2));
}
#endif

} // namespace q8kv4_indexer::detail
