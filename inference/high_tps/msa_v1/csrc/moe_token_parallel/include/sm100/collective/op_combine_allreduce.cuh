#pragma once
#include "include/sm100/common/schedule.cuh"
#include "include/sm100/common/tma.cuh"
namespace mega_moe::token_parallel {
// One completion-arrival counter line in the once-initialized private arena.
constexpr size_t kCombineArrive = 2 * 1024 * 1024;
static_assert(kCombineArrive >= (size_t(1) << 20));
// Sender-pushed single-copy-atomic 16B records in the receiver's arena:
// [parity][segment][sender][thread] = {bf16 x6, epoch}. Parity double
// buffering (comm_epoch&1) makes call N+1's writes land in the slab last read
// at call N-1, never the one read at call N.
constexpr int kCombineCols = 6;
constexpr size_t kCombinePayload = 4 * 1024 * 1024;
constexpr size_t kCombineRecBytes = 16;
constexpr size_t kCombineSegBytes = size_t(kConsumers) * kCombineRecBytes;
constexpr size_t kCombineParityBytes = size_t(T * 4) * WORLD * kCombineSegBytes;
static_assert(kConsumers * kCombineCols == H / 4);
static_assert(kCombinePayload + 2 * kCombineParityBytes <= kRoutedRecords);
__device__ inline void *combine_record(Args const &a, int receiver, uint32_t parity, int segment,
                                       int sender, int slot) {
  return arena(a, receiver) + kCombinePayload + size_t(parity) * kCombineParityBytes +
         ((size_t(segment) * WORLD + sender) * size_t(kConsumers) + size_t(slot)) *
             kCombineRecBytes;
}
// 16-byte records: one scalar .b128 access is single-copy atomic, so the six
// BF16 payload values and the generation word are observed together
// (SASS STG/LDG.E.128.STRONG.SYS). No separate flag or fence is needed for
// payload/flag ordering; the generation IS the arrival signal.
__device__ inline void combine_st_rec(void *a, uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3) {
  const uint64_t lo = uint64_t(w0) | (uint64_t(w1) << 32), hi = uint64_t(w2) | (uint64_t(w3) << 32);
  asm volatile("{\n\t.reg .b128 r;\n\tmov.b128 r,{%1,%2};\n\t"
               "st.relaxed.sys.global.b128 [%0],r;\n\t}" ::"l"(a),
               "l"(lo), "l"(hi)
               : "memory");
}
__device__ inline void combine_ld_rec(const void *a, uint64_t &lo, uint64_t &hi) {
  asm volatile("{\n\t.reg .b128 r;\n\tld.relaxed.sys.global.b128 r,[%2];\n\t"
               "mov.b128 {%0,%1},r;\n\t}"
               : "=l"(lo), "=l"(hi)
               : "l"(a)
               : "memory");
}
// Round-to-nearest-even FP32 -> BF16 in one hardware conversion. For every
// non-NaN input (subnormals and overflow to infinity included) this is
// bit-identical to m3::to_bf16; a NaN stays a NaN.
__device__ __forceinline__ uint32_t bf16x2_rn(float hi, float lo) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0,%1,%2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float bf16_round_rn(float x) {
  uint16_t r;
  asm("cvt.rn.bf16.f32 %0,%1;" : "=h"(r) : "f"(x));
  return __uint_as_float(uint32_t(r) << 16);
}
__device__ __forceinline__ uint64_t combine_pair(float lo, float hi) {
  return uint64_t(__float_as_uint(lo)) | (uint64_t(__float_as_uint(hi)) << 32);
}
__device__ __forceinline__ uint64_t combine_expand_pair(uint32_t bits) {
  // The wire codes are already rounded BF16 values. Expand their bits into
  // two full FP32 operands without another conversion or rounding boundary.
  return uint64_t(bits << 16) | (uint64_t(bits & 0xffff0000u) << 32);
}
__device__ __forceinline__ void combine_poll_batch(Context const &c, Args const &a, int segment,
                                                   uint32_t parity, unsigned &pending,
                                                   uint3 &payload0, uint3 &payload1,
                                                   uint3 &payload2, uint3 &payload3) {
  // The pending set is a snapshot: all independent record loads issue before
  // any returned epoch is inspected. Only current-generation words are adopted.
  unsigned const requested = pending;
  uint64_t lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
  if (requested & 1u)
    combine_ld_rec(combine_record(a, a.rank, parity, segment, 0, c.tid), lo0, hi0);
  if (requested & 2u)
    combine_ld_rec(combine_record(a, a.rank, parity, segment, 1, c.tid), lo1, hi1);
  if (requested & 4u)
    combine_ld_rec(combine_record(a, a.rank, parity, segment, 2, c.tid), lo2, hi2);
  if (requested & 8u)
    combine_ld_rec(combine_record(a, a.rank, parity, segment, 3, c.tid), lo3, hi3);
  if ((requested & 1u) && uint32_t(hi0 >> 32) == c.comm_epoch) {
    payload0 = make_uint3(uint32_t(lo0), uint32_t(lo0 >> 32), uint32_t(hi0));
    pending &= ~1u;
  }
  if ((requested & 2u) && uint32_t(hi1 >> 32) == c.comm_epoch) {
    payload1 = make_uint3(uint32_t(lo1), uint32_t(lo1 >> 32), uint32_t(hi1));
    pending &= ~2u;
  }
  if ((requested & 4u) && uint32_t(hi2 >> 32) == c.comm_epoch) {
    payload2 = make_uint3(uint32_t(lo2), uint32_t(lo2 >> 32), uint32_t(hi2));
    pending &= ~4u;
  }
  if ((requested & 8u) && uint32_t(hi3 >> 32) == c.comm_epoch) {
    payload3 = make_uint3(uint32_t(lo3), uint32_t(lo3 >> 32), uint32_t(hi3));
    pending &= ~8u;
  }
}
// Record-fused push combine. Each of the 256 consumer threads owns the six
// columns {tid+256k} of its (token,part) segment: it computes the FP32 local
// combine in the fixed arithmetic order and pushes its BF16 result as one
// 16B record (six BF16 values + comm_epoch) into each peer's arena. Receivers
// poll their three remote records for generation equality and then reduce all
// four rank payloads in rank order, exactly as the pull form did. Scalar
// 2B/4B user-buffer accesses keep every externally supplied alignment legal.
//
// Why the generation-equality poll always terminates and reads call-N data:
// sender r writes records for call N with parity N&1 and generation N. The
// only other writes that ever land in that slab are r's calls N-2/N+2. r can
// only enter call N+2 after call N+1 completed on r, and r's call N+1 poll
// observed q's call-N+1 records, which q wrote after its call-N kernel fully
// exited (same-stream kernel ordering on q) - i.e. after q's call-N reads of
// parity N&1 finished. Kernel boundaries on both ranks are full fences, and
// the b128 record accesses are morally-strong relaxed atomics on the same
// location, so the fence/atomic/fence chain gives the cross-rank
// happens-before: q's call-N reads precede r's call-N+2 overwrites, and a
// call-N reader can only ever observe generation N-2 (stale, keeps polling)
// or N (correct) in its slab. A separate consumed-inbox exchange is therefore
// no longer a required dependency, and out is only ever written locally.
//
// Completion: thread 0 of every combine CTA adds one to a monotonic arena
// counter; the final arrival (previous == comm_epoch*T*4-1) publishes the
// ready epoch. The counter is never reset, so repeated calls and graph
// replays stay valid.
__device__ inline void op_combine_allreduce(Context const &c, Args const &a, int t, int part) {
  static_assert(WORLD == 4 && kCombineCols == 6);
  int segment = t * 4 + part, tid = c.tid;
  uint32_t parity = c.comm_epoch & 1u;
  // Inputs independent of the down flags issue before the dependent wait so
  // their latency hides under it: this CTA's own route selection and the
  // residual row. msa_v1 rounds each route's alpha-scaled FC2 output and its
  // route weight to BF16 before the weighted FP32 sum.
  float alpha[K], weight[K];
  wait_routes();
#pragma unroll
  for (int s = 0; s < K; ++s) {
    alpha[s] = a.fc2_alpha[route_expert(s)];
    weight[s] = bf16_round_rn(route_weight(s));
  }
  int row0 = part * (H / 4) + tid, base = t * H + row0;
  float xf[kCombineCols];
#pragma unroll
  for (int k = 0; k < kCombineCols; ++k)
    xf[k] = m3::bf16(a.attention_residual[base + k * kConsumers]);
  uint16_t mine[kCombineCols];
  // Warp 7 acquired the four shared down publications during FC2; the
  // one-shot barrier extends that acquire to every combine thread. The
  // three sibling routes' FC2 rows arrive as epoch-tagged records: every
  // round issues all pending loads and adopts only this invocation's records.
  wait_bar(smem_address(c.smem + kSharedSeenBar), 0);
  float shared[kCombineCols];
#pragma unroll
  for (int k = 0; k < kCombineCols; ++k)
    shared[k] = bf16_round_rn(c.shared_down[t * H + row0 + k * kConsumers]);
  // Every route's FC2 row is rounded to its BF16 alpha-scaled value as soon
  // as it is available (the records inside their poll loop), so only the
  // FP32 route-order sum and the record pushes follow the last arrival.
  float down[K][kCombineCols];
  // Route 0 is this CTA's own FC2 output, staged on chip by its epilogue
  // threads (slab rows tid+128*tile); the other routes arrive as records.
  // It is converted after the record poll, so the first record round trip
  // is issued without waiting on the fc2_alpha loads.
  consumer_sync();
  unsigned pending = ((1u << (2 * K)) - 1u) & ~3u;
  int spins = 0;
  while (true) {
    uint64_t lo[2 * K], hi[2 * K];
#pragma unroll
    for (int i = 0; i < 2 * K; ++i)
      if (pending & (1u << i))
        routed_record_load(routed_record(a, t * K + i / 2, part, i & 1, tid), lo[i], hi[i]);
#pragma unroll
    for (int i = 0; i < 2 * K; ++i)
      if ((pending & (1u << i)) && uint32_t(hi[i] >> 32) == c.local_epoch) {
        int const s = i / 2, k0 = 3 * (i & 1);
        down[s][k0] = bf16_round_rn(__uint_as_float(uint32_t(lo[i])) * alpha[s]);
        down[s][k0 + 1] = bf16_round_rn(__uint_as_float(uint32_t(lo[i] >> 32)) * alpha[s]);
        down[s][k0 + 2] = bf16_round_rn(__uint_as_float(uint32_t(hi[i])) * alpha[s]);
        pending &= ~(1u << i);
      }
    if (!pending)
      break;
    if (++spins > 64)
      __nanosleep(32);
  }
  {
    float const *const own = reinterpret_cast<float const *>(c.smem + kDownNativeScaleScratch);
    int const odd = tid >= 128 ? 1 : 0, lane = tid & 127;
#pragma unroll
    for (int k = 0; k < kCombineCols; ++k)
      down[0][k] = bf16_round_rn(own[(2 * k + odd) * 128 + lane] * alpha[0]);
  }
  // Route-order FP32 sum, then BF16(BF16(shared) + routed).
  float sum[kCombineCols];
#pragma unroll
  for (int k = 0; k < kCombineCols; ++k) {
    float routed = 0.f;
#pragma unroll
    for (int s = 0; s < K; ++s)
      routed = fmaf(down[s][k], weight[s], routed);
    sum[k] = shared[k] + routed;
  }
  uint32_t const pack[3] = {bf16x2_rn(sum[1], sum[0]), bf16x2_rn(sum[3], sum[2]),
                            bf16x2_rn(sum[5], sum[4])};
#pragma unroll
  for (int k = 0; k < kCombineCols; ++k)
    mine[k] = uint16_t(pack[k / 2] >> (16 * (k & 1)));
  uint32_t w0 = uint32_t(mine[0]) | (uint32_t(mine[1]) << 16);
  uint32_t w1 = uint32_t(mine[2]) | (uint32_t(mine[3]) << 16);
  uint32_t w2 = uint32_t(mine[4]) | (uint32_t(mine[5]) << 16);
#pragma unroll
  for (int q = 0; q < WORLD; ++q)
    if (q != a.rank)
      combine_st_rec(combine_record(a, q, parity, segment, a.rank, tid), w0, w1, w2, c.comm_epoch);
  // Fused execution: the completion counter only elects which segment
  // publishes the completion epoch, whose sole reader is this rank's next
  // invocation, ordered after this kernel by the stream. Its relaxed arrival
  // therefore issues here and its round trip hides under the records' NVLink
  // flight instead of following the final stores.
  uint32_t arrival = 0u;
  if (tid == 0)
    asm volatile("atom.relaxed.gpu.global.add.u32 %0,[%1],%2;"
                 : "=r"(arrival)
                 : "l"(arena(a, a.rank) + kCombineArrive), "r"(1u)
                 : "memory");
  // Preserve the already-rounded BF16 wire values as their exact bit codes
  // until their rank-ordered FP32 addition. This changes no arithmetic or
  // rounding boundary and halves the live payload representation.
  // Every remote entry is overwritten by its current generation before
  // pending becomes zero; only this rank retains the local initialization.
  uint3 payload0 = make_uint3(w0, w1, w2), payload1 = payload0;
  uint3 payload2 = payload0, payload3 = payload0;
  {
    unsigned pending = 0;
#pragma unroll
    for (int q = 0; q < WORLD; ++q)
      if (q != a.rank)
        pending |= 1u << q;
    int spins = 0;
    while (pending) {
      combine_poll_batch(c, a, segment, parity, pending, payload0, payload1, payload2, payload3);
      if (pending && ++spins > 256)
        __nanosleep(32);
    }
  }
#pragma unroll
  for (int k = 0; k < kCombineCols; k += 2) {
    uint32_t b0 = k < 2 ? payload0.x : k < 4 ? payload0.y : payload0.z;
    uint32_t b1 = k < 2 ? payload1.x : k < 4 ? payload1.y : payload1.z;
    uint32_t b2 = k < 2 ? payload2.x : k < 4 ? payload2.y : payload2.z;
    uint32_t b3 = k < 2 ? payload3.x : k < 4 ? payload3.y : payload3.z;
    uint64_t total = 0;
    uint64_t const r0 = combine_expand_pair(b0), r1 = combine_expand_pair(b1);
    uint64_t const r2 = combine_expand_pair(b2), r3 = combine_expand_pair(b3);
    // Each packed lane retains the rank0,1,2,3 FP32 addition sequence.
    asm("add.rn.f32x2 %0,%0,%1;" : "+l"(total) : "l"(r0));
    asm("add.rn.f32x2 %0,%0,%1;" : "+l"(total) : "l"(r1));
    asm("add.rn.f32x2 %0,%0,%1;" : "+l"(total) : "l"(r2));
    asm("add.rn.f32x2 %0,%0,%1;" : "+l"(total) : "l"(r3));
    uint32_t const reduced =
        bf16x2_rn(__uint_as_float(uint32_t(total >> 32)), __uint_as_float(uint32_t(total)));
    uint64_t output = combine_expand_pair(reduced);
    uint64_t const residual = combine_pair(xf[k], xf[k + 1]);
    asm("add.rn.f32x2 %0,%0,%1;" : "+l"(output) : "l"(residual));
    uint32_t const result =
        bf16x2_rn(__uint_as_float(uint32_t(output >> 32)), __uint_as_float(uint32_t(output)));
    a.out[base + k * kConsumers] = uint16_t(result);
    a.out[base + (k + 1) * kConsumers] = uint16_t(result >> 16);
  }
  // The final arrival publishes the epoch after its own segment's stores.
  // Kernel completion orders every segment's stores before the next
  // invocation, the epoch's only reader.
  if (tid == 0 && arrival + 1u == c.comm_epoch * uint32_t(T * 4))
    asm volatile("st.relaxed.gpu.global.u32 [%0],%1;" ::"l"(a.ready[a.rank]), "r"(c.comm_epoch)
                 : "memory");
}
} // namespace mega_moe::token_parallel
