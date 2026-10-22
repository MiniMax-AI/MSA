#pragma once
#include "include/sm100/common/layout.hpp"
#include "include/sm100/common/numeric.cuh"
#include <cstdint>
#include <cuda_runtime.h>

namespace mega_moe::token_parallel {
constexpr int kSlots = 4, kSlotBytes = 32768, kRingOffset = 8192;
constexpr int kBridgeOffset = 1024;
static_assert(T == 8 && K == 4 && I == 768 && H == 6144 && WORLD == 4);
static_assert(I == 16 * 48 && H == 16 * 384, "sixteen shared CTAs cover FC1 and FC2");
// Twelve K512 router jobs per 16-expert tile use otherwise route-blocked CTAs
// while retaining full FP32 products and fixed-order FP32 partials.
constexpr int kRouterSplits = 12, kRouterJobs = 8 * kRouterSplits;

__device__ inline uint8_t *arena(Args const &a, int rank) {
  return reinterpret_cast<uint8_t *>(a.ready[rank]);
}
__device__ inline uint32_t *word(Args const &a, size_t offset, int index) {
  return reinterpret_cast<uint32_t *>(arena(a, a.rank) + offset + size_t(index) * 16);
}
__device__ inline float *shared_partials(Args const &a) {
  return reinterpret_cast<float *>(arena(a, a.rank) + kSharedPartials);
}
__device__ inline int token(int role) { return role / 16; }
__device__ inline int selected(int role) { return (role % 16) / 4; }
__device__ inline int slab(int role) { return role % 4; }
__device__ inline int route(int role) { return token(role) * K + selected(role); }
__device__ inline uint32_t load_gpu(uint32_t const *p) {
  uint32_t value;
  asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(value) : "l"(p) : "memory");
  return value;
}
__device__ inline uint32_t load_relaxed_gpu(uint32_t const *p) {
  uint32_t value;
  asm volatile("ld.relaxed.gpu.global.u32 %0,[%1];" : "=r"(value) : "l"(p) : "memory");
  return value;
}
__device__ inline void store_gpu(uint32_t *p, uint32_t value) {
  asm volatile("st.release.gpu.global.u32 [%0],%1;" ::"l"(p), "r"(value) : "memory");
}
__device__ inline void wait_gpu(uint32_t const *p, uint32_t epoch) {
  while (int32_t(load_gpu(p) - epoch) < 0)
    __nanosleep(64);
}
__device__ inline void wait_local(Args const &a, size_t off, int index, uint32_t epoch) {
  wait_gpu(word(a, off, index), epoch);
}
__device__ __forceinline__ uint32_t smem_address(void const *p) {
  return uint32_t(__cvta_generic_to_shared(p));
}
__device__ inline void init_bar(uint32_t p, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" ::"r"(p), "r"(count) : "memory");
}
__device__ inline void wait_bar(uint32_t p, uint32_t phase) {
  asm volatile("{ .reg .pred done; again: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
               "done,[%0],%1; @!done bra again; }" ::"r"(p),
               "r"(phase)
               : "memory");
}
__device__ inline void arrive_bar(uint32_t p, int count = 1) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _,[%0],%1;" ::"r"(p), "r"(count)
               : "memory");
}
__device__ inline void consumer_sync() { asm volatile("bar.sync 1,256;" ::: "memory"); }
// Route selection leaves this CTA's token routes in shared memory: a one-shot
// readiness barrier, the four expert IDs and their normalized FP32 weights.
// Two early router-page barriers and the selection scratch are also fixed.
constexpr int kRouterBar = 784, kRouteBar = 976, kRouteIds = 984, kRouteWeights = 1000;
// One-shot barrier: a combine owner's idle warp has acquired its four shared
// down publications during routed FC2.
constexpr int kSharedSeenBar = 1016;
constexpr int kRouteScratch = 2560;
__device__ __forceinline__ uint8_t *dynamic_smem() {
  extern __shared__ __align__(1024) uint8_t dynamic_smem_base[];
  return dynamic_smem_base;
}
__device__ __forceinline__ void wait_routes() {
  wait_bar(smem_address(dynamic_smem() + kRouteBar), 0);
}
__device__ __forceinline__ int route_expert(int s) {
  return reinterpret_cast<int const *>(dynamic_smem() + kRouteIds)[s];
}
__device__ __forceinline__ float route_weight(int s) {
  return reinterpret_cast<float const *>(dynamic_smem() + kRouteWeights)[s];
}
__device__ inline void producer_sync() { asm volatile("bar.sync 2,128;" ::: "memory"); }

struct Ring {
  uint8_t *payload;
  uint32_t full, empty, done, loaded;
  int stride;
  __device__ uint8_t *slot(uint32_t item) const { return payload + (item & 3) * stride; }
};
__device__ inline Ring make_ring(uint8_t *smem) {
  auto bars = reinterpret_cast<uint64_t *>(smem + 16);
  int const tid = int(threadIdx.x);
  if (tid < kSlots) {
    init_bar(smem_address(bars + tid), 4);
    init_bar(smem_address(bars + kSlots + tid), kConsumers);
    init_bar(smem_address(bars + 2 * kSlots + tid), 1);
    init_bar(smem_address(bars + 3 * kSlots + tid), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  return {smem + kRingOffset,
          smem_address(bars),
          smem_address(bars + kSlots),
          smem_address(bars + 2 * kSlots),
          smem_address(bars + 3 * kSlots),
          kSlotBytes};
}
struct Producer {
  Ring ring;
  uint32_t item = 0;
  int ptid;
  __device__ uint8_t *acquire() const {
    wait_bar(ring.empty + (item & 3) * 8, ((item >> 2) & 1) ^ 1);
    return ring.slot(item);
  }
  __device__ uint32_t page_barrier() const { return ring.full + (item & 3) * 8; }
  __device__ void wait_loaded() const { wait_bar(ring.loaded + (item & 3) * 8, (item >> 2) & 1); }
  __device__ void publish() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncwarp();
    if ((ptid & 31) == 0)
      arrive_bar(ring.full + (item & 3) * 8);
    ++item;
  }
};
struct Consumer {
  Ring ring;
  uint32_t item = 0;
  int tid;
  __device__ uint8_t const *acquire() const {
    wait_bar(ring.full + (item & 3) * 8, (item >> 2) & 1);
    return ring.slot(item);
  }
  __device__ uint32_t completion() const { return ring.done + (item & 3) * 8; }
  __device__ void retire() {
    __syncwarp();
    if ((tid & 31) == 0)
      arrive_bar(ring.empty + (item & 3) * 8, 32);
    ++item;
  }
  __device__ void release_tc() {
    wait_bar(completion(), (item >> 2) & 1);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    retire();
  }
};
struct Context {
  int tid;
  uint32_t local_epoch, comm_epoch;
  uint32_t tmem;
  uint8_t *smem;
  float *bridge;
  Producer producer;
  Consumer consumer;
  float *shared_down;
  __device__ bool produces() const { return tid >= kConsumers; }
};
// Fused routed FC2 outputs travel as single-copy-atomic 16-byte records
// {three FP32 values, epoch} indexed [route][slab][half][combine thread]:
// record (r,q,h,j) holds slab-relative rows j+256*(3h+m), m=0..2, exactly the
// rows combine thread j consumes. The epoch tag is the readiness signal, so a
// combine owner reads data and readiness in one round trip.
constexpr size_t kRoutedRecords = size_t(6) << 20;
static_assert(kRoutedRecords + size_t(R) * 4 * 2 * kConsumers * 16 <= (size_t(7) << 20));
__device__ __forceinline__ uint4 *routed_record(Args const &a, int route, int part, int half,
                                                int j) {
  return reinterpret_cast<uint4 *>(arena(a, a.rank) + kRoutedRecords) +
         ((size_t(route) * 4 + part) * 2 + half) * kConsumers + j;
}
__device__ __forceinline__ void routed_record_store(uint4 *p, float x, float y, float z,
                                                    uint32_t epoch) {
  uint64_t const lo = uint64_t(__float_as_uint(x)) | (uint64_t(__float_as_uint(y)) << 32);
  uint64_t const hi = uint64_t(__float_as_uint(z)) | (uint64_t(epoch) << 32);
  asm volatile(
      "{\n\t.reg .b128 r;\n\tmov.b128 r,{%1,%2};\n\tst.relaxed.gpu.global.b128 [%0],r;\n\t}" ::"l"(
          p),
      "l"(lo), "l"(hi)
      : "memory");
}
__device__ __forceinline__ void routed_record_load(uint4 const *p, uint64_t &lo, uint64_t &hi) {
  asm volatile(
      "{\n\t.reg .b128 r;\n\tld.relaxed.gpu.global.b128 r,[%2];\n\tmov.b128 {%0,%1},r;\n\t}"
      : "=l"(lo), "=l"(hi)
      : "l"(p)
      : "memory");
}
// Fused routed FC1 outputs (the FC2 input row) travel as single-copy-atomic
// 16-byte records {12 code bytes, epoch} indexed [route][slab][tile][part]:
// part p holds bytes 12p..12p+11 of the tile's 36-byte block (32 NVFP4 value
// bytes, then 4 UE4M3 scale bytes). The epoch tag is the readiness signal.
constexpr size_t kRoutedMidRecords = size_t(10) << 20;
static_assert(kRoutedMidRecords + size_t(R) * 4 * 9 * 16 <= (size_t(11) << 20));
__device__ __forceinline__ uint4 *routed_mid_record(Args const &a, int route, int slab, int tile,
                                                    int part) {
  return reinterpret_cast<uint4 *>(arena(a, a.rank) + kRoutedMidRecords) +
         ((size_t(route) * 4 + slab) * 3 + tile) * 3 + part;
}
__device__ __forceinline__ void record_store_words(uint4 *p, uint32_t w0, uint32_t w1, uint32_t w2,
                                                   uint32_t epoch) {
  uint64_t const lo = uint64_t(w0) | (uint64_t(w1) << 32);
  uint64_t const hi = uint64_t(w2) | (uint64_t(epoch) << 32);
  asm volatile(
      "{\n\t.reg .b128 r;\n\tmov.b128 r,{%1,%2};\n\tst.relaxed.gpu.global.b128 [%0],r;\n\t}" ::"l"(
          p),
      "l"(lo), "l"(hi)
      : "memory");
}
// FC2 epilogue thread tid<128 holds slab rows tid+128*tile, tile 0..11.
__device__ __forceinline__ void store_routed_records(Args const &a, int role, int tid,
                                                     float const (&v)[12], uint32_t epoch) {
  int const r = route(role), q = slab(role);
  routed_record_store(routed_record(a, r, q, 0, tid), v[0], v[2], v[4], epoch);
  routed_record_store(routed_record(a, r, q, 1, tid), v[6], v[8], v[10], epoch);
  routed_record_store(routed_record(a, r, q, 0, tid + 128), v[1], v[3], v[5], epoch);
  routed_record_store(routed_record(a, r, q, 1, tid + 128), v[7], v[9], v[11], epoch);
}
// A combine owner's otherwise idle consumer warp 7 acquires the four shared
// down publications of its slab while routed FC2 still runs.
__device__ inline void acquire_shared_down(Context const &c, Args const &a, int role) {
  int const lane = c.tid & 31;
  if (lane < 4)
    wait_local(a, kDownReady, kRouted + slab(role) * 4 + lane, c.local_epoch);
  __syncwarp();
  if (lane == 0)
    arrive_bar(smem_address(c.smem + kSharedSeenBar));
}
__device__ inline void publish_local(Context const &c, Args const &a, size_t off, int index) {
  // bar.sync orders every participating consumer's prior global writes into
  // the leader's causality chain. The leader's GPU-scope release then
  // publishes that complete chain; a device fence in every lane is redundant.
  consumer_sync();
  if (c.tid == 0)
    store_gpu(word(a, off, index), c.local_epoch);
}
} // namespace mega_moe::token_parallel
