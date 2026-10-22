#pragma once
#include "include/sm100/common/numeric.cuh"
#include "include/sm100/common/shared_tc.cuh"

namespace mega_moe::routed_tc {
// Routed down keeps all twelve activation scale groups after the per-buffer weight
// scale banks. N8 uses two columns per group; two groups share one
// four-column native scale copy. Weight-scale bank reuse is ordered by the
// producer's empty wait, which already carries the prior tensor completion.
__device__ inline void issue_down_tile(uint8_t const *p, uint8_t const *b, uint32_t base, int tile,
                                       int slot, uint32_t sfb_base) {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
  for (int kp = 0; kp < 3; ++kp) {
    uint64_t const da = shared_tc::value_desc(p + kp * 16384);
    uint64_t const db = shared_tc::scale_desc(b + kp * 128);
    constexpr uint32_t descriptor = (1u << 7) | (1u << 10) | (1u << 17) | (1u << 27);
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      uint64_t const a_desc = da + g * 2, b_desc = db + g * 2;
      uint32_t const sfa = base + 96 + slot * 48 + kp * 16 + g * 4,
                     sfb = base + sfb_base + kp * 8 + g * 2;
      asm volatile("{ .reg .pred acc; setp.ne.b32 acc,%6,0; "
                   "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
                   "[%0],%1,%2,%3,[%4],[%5],acc; }" ::"r"(base + tile * 8),
                   "l"(a_desc), "l"(b_desc), "r"(descriptor), "r"(sfa), "r"(sfb),
                   "r"(int(kp != 0 || g != 0))
                   : "memory");
    }
  }
}
} // namespace mega_moe::routed_tc
