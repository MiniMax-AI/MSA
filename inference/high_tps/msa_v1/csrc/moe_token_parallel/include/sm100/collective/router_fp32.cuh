#pragma once
#include "include/sm100/collective/op_router_pack.cuh"

namespace mega_moe::token_parallel {
// Router job: 16 experts x 8 tokens over one K512 split, full FP32 FMA
// products and FP32 accumulation of the exact BF16 token values. Warp w owns
// the K64 slice [64w, 64w+64) of the split (warps 0-3 page 0, 4-7 page 1);
// lane (e8 = lane/4, kq = lane%4) owns experts e8 and e8+8 at K positions
// kq + 4j. The fixed reduction order is: sequential j per lane, the kq pair
// tree within the warp, then the eight warp partials in warp order.
//
// The job depends on no stage of this invocation except the generation tag of
// its records, so its consumer warps run it ahead of the CTA startup join:
// every record-storing thread takes the same relaxed completion-word snapshot
// as the preamble (ready + 1) itself, and the FP32 compute starts as soon as
// the pages land.
__device__ inline void op_router_prejoin(uint8_t *smem, Args const &a, int job, int tid) {
  uint32_t const epoch = tid < 128 ? load_relaxed_gpu(a.ready[a.rank]) + 1u : 0u;
  // Producer warp 0 has initialized both page barriers and the retire barrier.
  asm volatile("bar.sync 10, 288;" ::: "memory");
  int const warp = tid / 32, lane = tid & 31, e8 = lane / 4, kq = lane & 3;
  int const stage = warp / 4, kbase = (warp & 3) * 64;
  wait_bar(smem_address(smem + kRouterBar) + stage * 8, 0);
  uint8_t const *const page = smem + kRingOffset + stage * kSlotBytes;
  float const *const w0p =
      reinterpret_cast<float const *>(page) + e8 * kFusedRouterStride + kbase + kq;
  float const *const w1p = w0p + 8 * kFusedRouterStride;
  uint8_t const *const xlo = page + kFusedRouterXLo + (kbase + kq) * 16;
  uint8_t const *const xhi = page + kFusedRouterXHi + (kbase + kq) * 16;
  // acc[s * 4 + p]: expert e8 + 8s, token pair (2p, 2p+1).
  uint64_t acc[8] = {};
#pragma unroll 4
  for (int j = 0; j < 16; ++j) {
    uint32_t const b0 = __float_as_uint(w0p[4 * j]), b1 = __float_as_uint(w1p[4 * j]);
    uint64_t const wa = uint64_t(b0) | (uint64_t(b0) << 32);
    uint64_t const wb = uint64_t(b1) | (uint64_t(b1) << 32);
    uint4 const lo = *reinterpret_cast<uint4 const *>(xlo + j * 64);
    uint4 const hi = *reinterpret_cast<uint4 const *>(xhi + j * 64);
    uint64_t const x[4] = {
        uint64_t(lo.x) | (uint64_t(lo.y) << 32), uint64_t(lo.z) | (uint64_t(lo.w) << 32),
        uint64_t(hi.x) | (uint64_t(hi.y) << 32), uint64_t(hi.z) | (uint64_t(hi.w) << 32)};
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      asm("fma.rn.f32x2 %0,%1,%2,%0;" : "+l"(acc[p]) : "l"(wa), "l"(x[p]));
      asm("fma.rn.f32x2 %0,%1,%2,%0;" : "+l"(acc[4 + p]) : "l"(wb), "l"(x[p]));
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int d = 1; d <= 2; d <<= 1) {
      uint32_t const lo = __shfl_xor_sync(0xffffffffu, uint32_t(acc[i]), d);
      uint32_t const hi = __shfl_xor_sync(0xffffffffu, uint32_t(acc[i] >> 32), d);
      uint64_t const other = uint64_t(lo) | (uint64_t(hi) << 32);
      asm("add.rn.f32x2 %0,%0,%1;" : "+l"(acc[i]) : "l"(other));
    }
  }
  // Warp partials [warp][expert][token], then a fixed warp-order sum.
  float *const partials = reinterpret_cast<float *>(smem + kRingOffset + kFusedRouterPartials);
  if (kq == 0) {
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      uint4 *dst = reinterpret_cast<uint4 *>(partials + warp * 128 + (e8 + 8 * s) * 8);
      dst[0] = make_uint4(uint32_t(acc[4 * s]), uint32_t(acc[4 * s] >> 32),
                          uint32_t(acc[4 * s + 1]), uint32_t(acc[4 * s + 1] >> 32));
      dst[1] = make_uint4(uint32_t(acc[4 * s + 2]), uint32_t(acc[4 * s + 2] >> 32),
                          uint32_t(acc[4 * s + 3]), uint32_t(acc[4 * s + 3] >> 32));
    }
  }
  consumer_sync();
  if (tid < 128) {
    int const e = tid / 8, t = tid & 7;
    float logit = partials[e * 8 + t];
#pragma unroll
    for (int w = 1; w < 8; ++w)
      logit += partials[w * 128 + e * 8 + t];
    // Epoch-tagged records: each one carries its own readiness.
    router_record_store(router_records(a) + size_t(job % kRouterSplits) * T * E + t * E +
                            (job / kRouterSplits) * 16 + e,
                        __float_as_uint(logit), epoch);
  }
  // Every consumer has finished reading the entry-staged pages; release
  // them to this CTA's FC1 producer, which overlays them. Route selection
  // polls the epoch-tagged records themselves, so no arrival hint follows.
  consumer_sync();
  if (tid == 0)
    arrive_bar(smem_address(smem + kRouterRetired));
}

} // namespace mega_moe::token_parallel
