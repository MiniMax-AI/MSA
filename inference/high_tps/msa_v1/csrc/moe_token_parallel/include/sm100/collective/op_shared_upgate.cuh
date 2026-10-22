#pragma once
#include "include/sm100/collective/expert_body.cuh"
namespace mega_moe::token_parallel {
// 768 channels contain twenty-four complete groups of thirty-two.
// Adjacent 48-channel CTAs share only their middle group. Full groups stay
// local; sixteen-element maxima cross the existing private global arena.
constexpr size_t kShared16Counter = 14 * 1024 * 1024;
constexpr size_t kShared16Amax = kShared16Counter + 128;
constexpr size_t kShared16Flags = kShared16Amax + kShared * T * 16;
static_assert(kShared16Flags + kShared * 16 <= kArenaBytes);
static_assert(kActivationEnd <= kShared16Counter);
__device__ inline void quantize_shared48(Context const &c, Args const &a, int role, int row) {
  int id = role - kRouted, t = c.tid / 32, lane = c.tid & 31;
  bool odd = (id & 1) != 0;
  float const *activated = c.bridge + 512;
  int cross = (odd ? 0 : 32) + lane;
  float value = lane < 16 ? activated[t * 64 + cross] : 0.f;
  int magnitude = __float_as_int(fabsf(value));
  int ordered = magnitude > 0x7f800000 ? -1 : magnitude;
  int local_max = __reduce_max_sync(0xffffffffu, ordered);
  auto records = reinterpret_cast<uint64_t *>(arena(a, a.rank) + kShared16Amax);
  int const record_index = (id >> 1) * (2 * T) + 2 * t + (id & 1);
  // Each token warp publishes readiness and its current partial maximum as
  // one naturally aligned, single-copy atomic release/acquire record. The two
  // records consumed by a CTA pair share one adjacent 16-byte token slot.
  uint32_t epoch = 0;
  if (lane == 0) {
    epoch = *reinterpret_cast<uint32_t const *>(c.smem + 4);
    uint64_t record = (uint64_t(epoch) << 32) | uint32_t(local_max);
    asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(records + record_index), "l"(record)
                 : "memory");
  }
  int local = (odd ? 16 : 0) + lane;
  float full = activated[t * 64 + local];
  shared_tc::quantize(full, a.shared_mid + size_t(t) * I, a.shared_mid_sf + size_t(t) * (I / 32),
                      row + local);
  int peer = 0;
  if (lane == 0) {
    uint64_t record;
    do {
      asm volatile("ld.acquire.gpu.global.u64 %0, [%1];"
                   : "=l"(record)
                   : "l"(records + (record_index ^ 1))
                   : "memory");
    } while (uint32_t(record >> 32) != epoch);
    peer = int(uint32_t(record));
  }
  peer = __shfl_sync(0xffffffffu, peer, 0);
  float amax = __int_as_float(local_max > peer ? local_max : peer);
  uint8_t scale = m3::ue8m0_up(amax * (1.f / 448.f));
  if (lane < 16)
    a.shared_mid[size_t(t) * I + row + cross] = m3::to_e4m3(value * m3::input_inv_scale(scale));
  if (!odd && lane == 0)
    a.shared_mid_sf[size_t(t) * (I / 32) + (row + 32) / 32] = scale;
}

__device__ inline void op_shared_upgate(Context &c, Args const &a, int role) {
  // Producers acquire this invocation's input-pack flags before staging input and
  // scales. Each issuer acquires their completion-tracked FULL barrier below;
  // the epilogue joins both issuers before reading their accumulators.
  constexpr int pages = H / (128 * kSharedGuPagesPerSlot);
  static_assert(pages % (2 * kSharedGuRingSlots) == 0);
  static_assert(kSharedGuResumeItems % (2 * kSlots) == 0);
  int const row = (role - kRouted) * 48;
  if ((c.tid < 32 || (c.tid >= 64 && c.tid < 96)) && down_elect_one()) {
    // Two elected issuer threads split the four independent K32-phase chains;
    // each chain stays within one thread so its same-accumulator MMA pairing
    // preserves its accumulation order.
    int const half = c.tid < 32 ? 0 : 2;
    Consumer issued = c.consumer;
    for (int cycle = 0; cycle < pages / kSharedGuRingSlots; ++cycle) {
#pragma unroll
      for (int slot = 0; slot < kSharedGuRingSlots; ++slot) {
        wait_bar(issued.ring.full + slot * 8, cycle & 1);
        // FULL needs both scale producers' completion-tracked commits and
        // every TMA byte. The after-fence orders that acquire before MMA.
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        uint8_t const *page = issued.ring.payload + slot * kSharedGuSlotBytes;
        uint32_t const empty = issued.ring.empty + slot * 8;
        if (!half)
          arrive_bar(empty, kConsumers - 2);
#pragma unroll
        for (int sub = 0; sub < kSharedGuPagesPerSlot; ++sub)
          issue_upgate_ready_page(page, c.tmem, slot * kSharedGuPagesPerSlot + sub, sub,
                                  cycle != 0 || slot != 0 || sub != 0, half);
        // TC completion from both issuers supplies the final empty credits, so
        // producers may refill only after every native MMA read has finished.
        shared_tc::commit(empty);
        ++issued.item;
      }
    }
    if (!half) {
      // The last slot's EMPTY flip needs both issuers' final commits, each of
      // which tracks completion of every prior MMA from its thread.
      uint32_t const last = issued.item - 1;
      wait_bar(issued.ring.empty + (last % kSharedGuRingSlots) * 8,
               (last / kSharedGuRingSlots) & 1);
    }
  }
  c.consumer.item = kSharedGuResumeItems;
  consumer_sync();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  int const warp = c.tid >> 5, q = warp & 3, t0 = 4 * (warp >> 2);
  int const ch = 32 * (q & 1) + (c.tid & 31);
  bool const up = q >= 2;
  uint32_t const address = c.tmem + (uint32_t(q) << 21) + t0;
  // Sum the four K32-phase accumulator chains in FP32, in a fixed order.
  float p0[4], p1[4], p2[4], p3[4];
#pragma unroll
  for (int acc = 0; acc < 4; ++acc)
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(p0[acc]), "=f"(p1[acc]), "=f"(p2[acc]), "=f"(p3[acc])
                 : "r"(address + upgate_acc_column(acc))
                 : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  float const v0 = (p0[0] + p0[1]) + (p0[2] + p0[3]);
  float const v1 = (p1[0] + p1[1]) + (p1[2] + p1[3]);
  float const v2 = (p2[0] + p2[1]) + (p2[2] + p2[3]);
  float const v3 = (p3[0] + p3[1]) + (p3[2] + p3[3]);
  float const own0 = up ? v2 : v0, own1 = up ? v3 : v1;
  int const exchange_t = t0 + (up ? 0 : 2);
  c.bridge[exchange_t * 64 + ch] = up ? v0 : v2;
  c.bridge[(exchange_t + 1) * 64 + ch] = up ? v1 : v3;
  consumer_sync();
  float activated[2];
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    int const t = t0 + (up ? 2 : 0) + pass;
    float const own = pass ? own1 : own0, other = c.bridge[t * 64 + ch];
    float const h = up ? shared_tc::activation(other, own) : shared_tc::activation(own, other);
    activated[pass] = h;
  }
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    int const t = t0 + (up ? 2 : 0) + pass;
    // Keep gate/up exchange separate from activation output so outstanding
    // exchange reads never race this store.
    if (ch < 48)
      c.bridge[512 + t * 64 + ch] = activated[pass];
  }
  consumer_sync();
  quantize_shared48(c, a, role, row);
  asm volatile("fence.proxy.async.global;" ::: "memory");
  publish_local(c, a, kGuReady, role);
}
} // namespace mega_moe::token_parallel
