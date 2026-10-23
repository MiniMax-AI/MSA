#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"

#if !defined(MINIMAX_MSA_Q8KV4_HAS_QMUL4)
#error "The Q8KV4 prefill JIT must define MINIMAX_MSA_Q8KV4_HAS_QMUL4"
#endif

#if MINIMAX_MSA_Q8KV4_HAS_QMUL4 &&                                                                 \
    (!defined(__CUDACC_VER_MAJOR__) || (__CUDACC_VER_MAJOR__ < 13) ||                              \
     (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ < 4))
#error "Q8KV4 prefill attention QMUL4 requires CUDA Toolkit 13.4 or newer"
#endif

namespace minimax::msa_v1::attention::prefill::q8kv4::sm100::common {

// Dequantizes eight consecutive E2M1 values (one 32-bit word, element i in nibble i) that
// share one E4M3 scale into eight E4M3 values: `output_lo` holds elements 0-3 and
// `output_hi` elements 4-7, each byte i being element i. `scale_word` comes from
// `make_dequant_scale_word` so the per-element scale broadcast happens once per group.

#if MINIMAX_MSA_Q8KV4_HAS_QMUL4

CUTLASS_DEVICE uint32_t make_dequant_scale_word(uint8_t scale_e4m3) {
  return static_cast<uint32_t>(scale_e4m3) * 0x01010101u;
}

CUTLASS_DEVICE void nvfp4_to_e4m3x4(uint32_t &output, uint16_t packed_fp4, uint32_t scale_e4m3x4) {
  asm volatile("mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, %1, %2;"
               : "=r"(output)
               : "h"(packed_fp4), "r"(scale_e4m3x4));
}

CUTLASS_DEVICE void dequant_e2m1x8_uniform_scale(uint32_t &output_lo, uint32_t &output_hi,
                                                 uint32_t packed_fp4, uint32_t scale_word) {
  nvfp4_to_e4m3x4(output_lo, static_cast<uint16_t>(packed_fp4), scale_word);
  nvfp4_to_e4m3x4(output_hi, static_cast<uint16_t>(packed_fp4 >> 16), scale_word);
}

#else

// Architectures without QMUL4 (SM107) multiply in packed FP16. An E2M1 x E4M3
// product has at most five significant bits and stays within the FP16 normal
// range, so the FP16 product is exact and the single rounding happens in the
// final E4M3 conversion, exactly as in QMUL4.
CUTLASS_DEVICE uint32_t make_dequant_scale_word(uint8_t scale_e4m3) {
  uint32_t scale_f16x2;
  asm("{\n"
      ".reg .b16 scale_pair;\n"
      "mov.b16 scale_pair, %1;\n"
      "cvt.rn.f16x2.e4m3x2 %0, scale_pair;\n"
      "}\n"
      : "=r"(scale_f16x2)
      : "h"(static_cast<uint16_t>(scale_e4m3 * 0x0101u)));
  return scale_f16x2;
}

CUTLASS_DEVICE void dequant_e2m1x8_uniform_scale(uint32_t &output_lo, uint32_t &output_hi,
                                                 uint32_t packed_fp4, uint32_t scale_f16x2) {
  asm volatile("{\n"
               ".reg .b8 e0, e1, e2, e3;\n"
               ".reg .b32 f0, f1, f2, f3;\n"
               ".reg .b16 q0, q1, q2, q3;\n"
               "mov.b32 {e0, e1, e2, e3}, %2;\n"
               "cvt.rn.f16x2.e2m1x2 f0, e0;\n"
               "cvt.rn.f16x2.e2m1x2 f1, e1;\n"
               "cvt.rn.f16x2.e2m1x2 f2, e2;\n"
               "cvt.rn.f16x2.e2m1x2 f3, e3;\n"
               "mul.rn.f16x2 f0, f0, %3;\n"
               "mul.rn.f16x2 f1, f1, %3;\n"
               "mul.rn.f16x2 f2, f2, %3;\n"
               "mul.rn.f16x2 f3, f3, %3;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 q0, f0;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 q1, f1;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 q2, f2;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 q3, f3;\n"
               "mov.b32 %0, {q0, q1};\n"
               "mov.b32 %1, {q2, q3};\n"
               "}\n"
               : "=&r"(output_lo), "=&r"(output_hi)
               : "r"(packed_fp4), "r"(scale_f16x2));
}

#endif

} // namespace minimax::msa_v1::attention::prefill::q8kv4::sm100::common
