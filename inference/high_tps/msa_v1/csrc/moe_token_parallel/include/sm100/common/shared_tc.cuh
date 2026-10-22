#pragma once

#include "include/sm100/common/m3_numeric.cuh"
#include "include/sm100/common/params.hpp"
#include <cstdint>
#include <cuda_runtime.h>

// Native tcgen05 primitives of the MXFP8 shared expert. Scales remain
// independent raw UE8M0 operands of the block-scaled tensor instruction;
// multiplying scale factors in FP32 would lose valid extreme-scale inputs.

namespace mega_moe::shared_tc {

__device__ __forceinline__ uint32_t smem_addr(void const *p) {
  return uint32_t(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint64_t desc_encode(uint64_t x) { return (x & 0x3ffffull) >> 4; }

__device__ __forceinline__ uint64_t value_desc(void const *p) {
  // K-major E4M3 rows with 128-byte swizzle, identically for A and B.
  return desc_encode(smem_addr(p)) | (desc_encode(16) << 16) | (desc_encode(8 * 128) << 32) |
         (1ull << 46) | (2ull << 61);
}

__device__ __forceinline__ uint64_t scale_desc(void const *p) {
  return desc_encode(smem_addr(p)) | (desc_encode(16) << 16) | (desc_encode(8 * 16) << 32) |
         (1ull << 46);
}

// Allocate and deallocate with one complete warp, after the caller has joined
// all threads that may still read the allocation.  MMA/copy/commit are instead
// issued by one elected consumer thread.
__device__ __forceinline__ void tmem_alloc(uint32_t shared_destination, int columns) {
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(shared_destination),
      "r"(columns)
      : "memory");
}

__device__ __forceinline__ void tmem_free(uint32_t base, int columns) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(base), "r"(columns)
               : "memory");
}

__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}

// The full participating warp elects one deterministic issuer. Making that
// property explicit lets ptxas avoid serializing an already-single-thread
// native MMA/copy stream through an election loop for every instruction.
__device__ __forceinline__ bool elect_issuer() {
  uint32_t selected;
  asm volatile("{ .reg .pred p; elect.sync _|p,0xffffffff; "
               "selp.u32 %0,1,0,p; }"
               : "=r"(selected)::"memory");
  return selected != 0;
}

__device__ __forceinline__ void commit(uint32_t barrier) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(barrier)
      : "memory");
}

__device__ __forceinline__ void copy_scales(uint32_t destination, uint64_t descriptor) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(destination),
               "l"(descriptor)
               : "memory");
}

__device__ __forceinline__ void mma_mxf8(uint32_t accumulator, uint64_t a_desc, uint64_t b_desc,
                                         uint32_t sfa, uint32_t sfb, bool accumulate,
                                         uint32_t scale_byte) {
  // E4M3 x E4M3, independent UE8M0 scales, M128 x N8 x K32, FP32 D.
  constexpr uint32_t kDescriptor = (1u << 17) | (1u << 23) | (1u << 27);
  uint32_t const descriptor = kDescriptor | (scale_byte << 29) | (scale_byte << 4);
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %6, 0; "
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
               "[%0], %1, %2, %3, [%4], [%5], p; }" ::"r"(accumulator),
               "l"(a_desc), "l"(b_desc), "r"(descriptor), "r"(sfa), "r"(sfb), "r"(int(accumulate))
               : "memory");
}

__device__ __forceinline__ void load_acc8(uint32_t address, float (&v)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]), "=f"(v[6]),
                 "=f"(v[7])
               : "r"(address)
               : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ float mul_ftz(float x, float y) {
  float z;
  asm("mul.ftz.f32 %0,%1,%2;" : "=f"(z) : "f"(x), "f"(y));
  return z;
}
// Matches the msa_v1 Shared epilogue, which is compiled with fast math:
// BF16-rounded gate/up, clamped, approximate sigmoid reciprocal, flush-to-zero
// products, and a BF16-rounded result before MXFP8 quantization.
__device__ __forceinline__ float activation(float gate, float up) {
  gate = fminf(m3::bf16_round(gate), 7.f);
  up = fminf(fmaxf(m3::bf16_round(up), -7.f), 7.f);
  float const sigmoid = __fdividef(1.f, 1.f + __expf(-1.702f * gate));
  return m3::bf16_round(mul_ftz(mul_ftz(gate, sigmoid), up + 1.f));
}

__device__ __forceinline__ void quantize(float value, uint8_t *output, uint8_t *scales,
                                         int channel) {
  // Match the established Shared fmax semantics for exceptional values too.
  int const magnitude = __float_as_int(fabsf(value));
  int const ordered = magnitude > 0x7f800000 ? -1 : magnitude;
  float const amax = __int_as_float(__reduce_max_sync(0xffffffffu, ordered));
  uint8_t const scale = m3::ue8m0_up(amax * (1.f / 448.f));
  output[channel] = m3::to_e4m3(value * m3::input_inv_scale(scale));
  if (!(channel & 31))
    scales[channel / 32] = scale;
}

} // namespace mega_moe::shared_tc
