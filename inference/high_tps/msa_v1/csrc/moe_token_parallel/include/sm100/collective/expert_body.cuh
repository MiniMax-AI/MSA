#pragma once
#include "include/sm100/collective/shared_down_tile.cuh"
#include "include/sm100/collective/upgate_pipeline.cuh"
#include "include/sm100/common/routed_tc.cuh"
#include "include/sm100/common/tma.cuh"

namespace mega_moe::token_parallel {
__device__ inline void routed_upgate_mapped_issue(Context &c, Args const &a);
// Shared FC2 producer: acquire the sixteen shared FC1 publications once per
// CTA, then stream the CTA's weight tiles.
__device__ inline void produce_shared_down(Context &c, KernelArgs const &k, int role) {
  // One poller per shared FC1 flag, then the producer barrier broadcasts
  // the gpu-scope acquires instead of 128 threads spinning on the same
  // arena words while the shared upgate drains.
  if (c.producer.ptid < kShared)
    wait_local(k.a, kGuReady, kRouted + c.producer.ptid, c.local_epoch);
  producer_sync();
  produce_shared_down_tile(c, k, role);
}
// Shared FC1 producer. The shared down weights depend on no in-flight stage:
// warm L2 before blocking on input-pack readiness (the shared up/gate warm-up
// was already issued at kernel entry); later demand loads read the same
// public bytes, merely L2-resident.
__device__ inline void produce_shared_upgate(Context &c, KernelArgs const &k, int role) {
  if (c.producer.ptid == 0) {
    int const pf2 = (role - kRouted) * 384;
    for (int t2 = 0; t2 < 3; ++t2)
      prefetch_down_full(&k.maps.map[SharedFc2Value], pf2 + t2 * 128);
  }
  // Dedicated pollers acquire the stage inputs once per CTA and the producer
  // barrier extends the gpu-scope acquires to every producer.
  if (c.producer.ptid < T)
    wait_local(k.a, kInputReady, c.producer.ptid, c.local_epoch);
  producer_sync();
  produce_upgate(c, k, (role - kRouted) * 48);
}
// Routed FC1 producer. Producer warp 1 issues the MMAs; lane zero of warp 0
// owns every weight transfer. Routed producers already joined after their own
// route selection; routed weights depend on no input-pack publication.
__device__ inline void produce_routed_upgate(Context &c, KernelArgs const &k, int role) {
  if (c.producer.ptid >= 32 && c.producer.ptid < 64) {
    routed_upgate_mapped_issue(c, k.a);
    return;
  }
  if (c.producer.ptid != 0)
    return;
  constexpr int kpages = H / kUpgateK, slots = kUpgateMegaSlots;
  int const first_row = slab(role) * 192;
  uint32_t const full = smem_address(c.smem + kUpgateFull),
                 empty = smem_address(c.smem + kUpgateEmpty);
  // Sequential K per output tile keeps each weight row's 256B chunks
  // adjacent in time for the DRAM scheduler; the three tiles still
  // accumulate into independent TMEM banks for a single final drain.
  // FC2 weights and scales load only on demand into retiring FC1 slots:
  // evict_first lines warmed tens of microseconds ahead do not survive the
  // concurrent weight stream, so speculative copies only add HBM traffic.
  uint32_t item = 0;
  for (int tile = 0; tile < 3; ++tile) {
    for (int page_index = 0; page_index < kpages; ++page_index, ++item) {
      uint32_t slot = item % slots, phase = (item / slots) & 1;
      // Bound the requests in flight below the slot count: once HBM is
      // saturated, deeper per-CTA queues only add latency to every
      // dependent hand-off. The slot's own retirement is still awaited.
      // The bound counts arrivals (FULL): a late first MMA never throttles
      // the stream. Item prior's phase is the only one of its slot's FULL
      // issued so far.
      if (item >= uint32_t(kUpgateInFlight)) {
        uint32_t const prior = item - uint32_t(kUpgateInFlight);
        wait_bar(full + (prior % slots) * 8, (prior / slots) & 1);
      }
      wait_bar(empty + slot * 8, phase ^ 1);
      stage_routed_upgate(upgate_slot(c.smem, item, slots), k, role, first_row + tile * 64,
                          page_index * kUpgateK, full + slot * 8,
                          smem_address(c.smem + kUpgateScaleReady) + slot * 8);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      arrive_bar(full + slot * 8, 4);
    }
  }
}
} // namespace mega_moe::token_parallel
