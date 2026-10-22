#pragma once
#include "include/sm100/common/schedule.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

namespace mega_moe::token_parallel {
__device__ __forceinline__ void prefetch_map(CUtensorMap const *map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory");
}

__device__ inline void expect_bytes(uint32_t barrier, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;" ::"r"(barrier),
               "r"(bytes)
               : "memory");
}
__device__ inline void expect_page(uint32_t barrier, uint32_t bytes) {
  if (bytes)
    asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0],%1;" ::"r"(barrier),
                 "r"(bytes)
                 : "memory");
}
__device__ inline void bulk_load_ef(void *dst, void const *src, uint32_t bytes, uint32_t barrier) {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0;" : "=l"(policy));
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
               "[%0],[%1],%2,[%3],%4;" ::"r"(smem_address(dst)),
               "l"(src), "r"(bytes), "r"(barrier), "l"(policy)
               : "memory");
}
__device__ inline void bulk_load(void *dst, void const *src, uint32_t bytes, uint32_t barrier) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];" ::"r"(
          smem_address(dst)),
      "l"(src), "r"(bytes), "r"(barrier)
      : "memory");
}
// Single-use weight streams are evict_first: they must not displace the L2
// working set of the surrounding layer.
__device__ __forceinline__ uint64_t evict_first_policy() {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0;" : "=l"(policy));
  return policy;
}
// Three K256 routed FC2 value panels in one request.
__device__ inline void tensor_load3_values(void *dst, CUtensorMap const *map, int row,
                                           uint32_t barrier) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes."
               "L2::cache_hint "
               "[%0],[%1,{%2,%3,%2}],[%4],%5;" ::"r"(smem_address(dst)),
               "l"(map), "r"(0), "r"(row), "r"(barrier), "l"(evict_first_policy())
               : "memory");
}
__device__ inline void tensor_load_upgate_scales(void *dst, CUtensorMap const *map, int row,
                                                 int expert, int kpanel, uint32_t barrier) {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0;" : "=l"(policy));
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_"
      "hint [%0],[%1,{%2,%3,%4,%5}],[%6],%7;" ::"r"(smem_address(dst)),
      "l"(map), "r"(0), "r"(row), "r"(expert * 2), "r"(kpanel / 2), "r"(barrier), "l"(policy)
      : "memory");
}
// L2-only prefetch of one shared FC1 slice of this call. It moves no data to
// shared memory; the later demand loads read the same bytes, L2-resident.
__device__ inline void prefetch_shared16_pair(CUtensorMap const *map, int slice, int group) {
  asm volatile(
      "cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint [%0,{%1,%1,%1,%2,%3}],%4;" ::
          "l"(map),
      "r"(0), "r"(slice), "r"(group), "l"(evict_first_policy())
      : "memory");
}

constexpr int kUpgateInput = 4864, kUpgateInputScale = 256;
// Each K512 item holds two K256 panels in eight-row atoms. Six independently
// retired slots retain exactly 221184 payload bytes, keeping five items in
// flight while one slot is in its format/MMA turnaround.
constexpr int kUpgateK = 512, kUpgateScaleWidth = kUpgateK / 16;
constexpr int kUpgateSlotBytes = 128 * (kUpgateK / 2 + kUpgateK / 16);
static_assert(H % kUpgateK == 0 && kUpgateK % 512 == 0);
// The FC1 slot payloads overlay a router job's entry-staged pages, which the
// first FC1 transfer may overwrite only after kRouterRetired (see
// select_routes); FC2 later reuses the retired FC1 slots.
constexpr int kUpgateMegaSlots = 6;
// Routed up/gate weight requests in flight per CTA (at most the slot count).
constexpr int kUpgateInFlight = 4;
static_assert(kUpgateInFlight >= 2 && kUpgateInFlight <= kUpgateMegaSlots);
constexpr int kUpgateFull = 640, kUpgateEmpty = 688, kUpgateScaleReady = 736;
constexpr int kUpgateRawScale = 128 * (kUpgateK / 2);
constexpr int kUpgateTileReady = 896;
// One-shot barriers: the staged routed input has landed, and its scales are
// resident in TMEM for the issuing warp.
constexpr int kUpgateInputBar = 144, kUpgateScalesBar = 152;
static_assert(kUpgateInputBar + 8 == kUpgateScalesBar && kUpgateScalesBar + 8 <= kUpgateInputScale);
static_assert(kRoutedInput % 16 == 0 && kRoutedInputScale % 16 == 0 && (H / 2) % 16 == 0 &&
              (H / 16) % 16 == 0);
static_assert(kRingOffset + kUpgateMegaSlots * kUpgateSlotBytes <= kSmemBytes);
// Columns 0..23 hold three tile accumulators, 24..215 the resident input
// scales at their legal two-column spacing, and 216.. the per-slot
// weight-scale banks of kUpgateScaleWidth columns each.
static_assert(216 + kUpgateMegaSlots * kUpgateScaleWidth <= 512);
static_assert(kUpgateTileReady + 3 * 8 <= 1024);
static_assert(kUpgateFull + kUpgateMegaSlots * 8 <= kUpgateEmpty &&
              kUpgateEmpty + kUpgateMegaSlots * 8 <= kUpgateScaleReady &&
              kUpgateScaleReady + kUpgateMegaSlots * 8 <= kRouterBar &&
              kRouterBar + 2 * 8 <= kUpgateTileReady);
__device__ inline uint8_t *upgate_slot(uint8_t *smem, uint32_t item, int slots) {
  return smem + kRingOffset + (item % slots) * kUpgateSlotBytes;
}
static_assert(kUpgateInput + H / 2 + 128 <= kRingOffset);
static_assert(kUpgateInputScale + H / 16 <= kBridgeOffset);

// The input is reused by three output tiles of this invocation. One otherwise
// idle consumer warp stages this small operand while the producers
// immediately begin streaming the much larger weights.
__device__ inline void stage_upgate_input(Context const &c, Args const &a, int role) {
  int lane = c.tid & 31;
  // Routes no longer pass through the input-pack CTA, so acquire this
  // token's packed input explicitly before the bulk copies read it.
  if (lane == 0)
    wait_local(a, kInputReady, token(role), c.local_epoch);
  __syncwarp();
  // Arena rows are 16-byte aligned, as bulk copies require.
  auto x = a.routed_input + size_t(token(role)) * (H / 2);
  auto s = a.routed_input_sf + size_t(token(role)) * (H / 16);
  uint32_t barrier = smem_address(c.smem + kUpgateInputBar);
  // Pad the unused N columns of the direct, non-swizzled routed up/gate B operand.
  reinterpret_cast<uint32_t *>(c.smem + kUpgateInput + H / 2)[lane] = 0;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncwarp();
  if (lane == 0) {
    expect_bytes(barrier, H / 2 + H / 16);
    bulk_load(c.smem + kUpgateInput, x, H / 2, barrier);
    bulk_load(c.smem + kUpgateInputScale, s, H / 16, barrier);
  }
}

// Shared up/gate groups three K128 contractions into one ownership transfer.
// Distinct native pages and TMEM scale panels preserve the K32 MMA order.
constexpr int kSharedGuPagesPerSlot = 3;
constexpr int kSharedGuRingSlots = 4;
constexpr int kSharedGuResumeItems = 16;
constexpr int kSharedGuNativeBytes = 18432;
constexpr int kSharedGuSlotBytes = kSharedGuPagesPerSlot * kSharedGuNativeBytes;
constexpr int kSharedGuSmemBytes = kRingOffset + kSharedGuRingSlots * kSharedGuSlotBytes;
constexpr int kSharedGuTmemColumns = 128;
static_assert(kSharedGuSmemBytes < 227 * 1024);

// Request one routed FC1 item: the K-panel's scale slice on its own barrier,
// so formatters can fill TMEM while the values arrive, then the values.
__device__ inline void stage_routed_upgate(uint8_t *page, KernelArgs const &k, int role, int row,
                                           int k0, uint32_t barrier, uint32_t scale_ready) {
  int const expert = route_expert(selected(role));
  expect_page(barrier, 128 * (kUpgateK / 2));
  expect_page(scale_ready, 128 * kUpgateScaleWidth);
  tensor_load_upgate_scales(page + kUpgateRawScale, &k.maps.map[RoutedFc1Scale], row, expert,
                            k0 / 256, scale_ready);
  arrive_bar(scale_ready);
  // Rank-five traversal keeps adjacent K panels inside each eight-row atom.
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::"
      "cache_hint [%0],[%1,{%2,%2,%3,%4,%5}],[%6],%7;" ::"r"(smem_address(page)),
      "l"(&k.maps.map[RoutedFc1Value]), "r"(0), "r"(k0 / 256), "r"(row / 8), "r"(expert * 2),
      "r"(barrier), "l"(evict_first_policy())
      : "memory");
}
// Routed down stages three native K256 panels as one K768 transaction group.
// Each 54KiB buffer holds values and a raw scale panel; formatter warps
// transport the complete scales to TMEM.
constexpr int kDownFullStride = 55296, kDownScales = 49152;
constexpr int kDownSlots = 4, kDownSmemBytes = kRingOffset + kDownSlots * kDownFullStride;
// Routed FC2 value tiles in flight per CTA (at most the down-buffer count).
constexpr int kDownInFlight = 2;
static_assert(kDownInFlight >= 1 && kDownInFlight <= kDownSlots);
static_assert(kDownSmemBytes < 227 * 1024 && kDownFullStride % 1024 == 0);
// Routed CTAs stage the FC1->FC2 record blocks beyond every routed ring:
// 108 bytes of FC1 output blocks, then the twelve FC2 input-scale words.
constexpr int kRoutedMidStage = 229376;
static_assert(kRoutedMidStage >= kDownSmemBytes &&
              kRoutedMidStage >= kRingOffset + kUpgateMegaSlots * kUpgateSlotBytes &&
              kRoutedMidStage + 128 + 48 <= kSmemBytes);
// Each FC2 scale formatter warp owns one 2 KiB native-copy segment.
constexpr int kDownNativeScaleScratch = 1536;
static_assert(1024 + I / 2 + 128 <= kDownNativeScaleScratch);
static_assert(kDownNativeScaleScratch + 6144 <= kRingOffset);

// Fused mapped execution streams FC2 straight after FC1 without a ring reset.
// Its tiles reuse the four 54 KiB down-buffer positions of the same payload:
// before down buffer j<4 is written, every FC1 slot it overlaps must have
// retired its final MMA (FC1 slots retire in order). Later tiles
// acquire the matching down buffer's previous MMA retirement. Nothing is
// prefetched beyond the four buffers: unbounded early FC2 traffic from fast
// CTAs starves slower sibling slabs whose FC1 gates every FC2. Dedicated
// one-shot-per-call barriers replace the ring.
constexpr int kOvFull = 160, kOvEmpty = 192, kOvDone = 224, kOvLoaded = 920, kOvRetired = 952;
static_assert(kOvDone + 4 * 8 <= kUpgateInputScale && kOvRetired + 3 * 8 <= kRouteBar);
static_assert(kUpgateTileReady + 3 * 8 <= kOvLoaded);
static_assert(kRingOffset + kDownSlots * kDownFullStride ==
              kRingOffset + kUpgateMegaSlots * kUpgateSlotBytes);
__device__ inline void produce_down_overlap(Context &c, KernelArgs const &k, int role) {
  int const expert = route_expert(selected(role));
  int const first = slab(role) * 1536;
  uint32_t const full = smem_address(c.smem + kOvFull), empty = smem_address(c.smem + kOvEmpty);
  uint32_t const done = smem_address(c.smem + kOvDone);
  uint32_t const fc1_empty = smem_address(c.smem + kUpgateEmpty);
  constexpr int fc1_uses = 3 * (H / kUpgateK) / kUpgateMegaSlots;
  static_assert(3 * (H / kUpgateK) % kUpgateMegaSlots == 0);
  int fc1_retired = 0;
  for (int tile = 0; tile < 12; ++tile) {
    int const slot = tile % kDownSlots, row = first + tile * 128;
    if (tile < kDownSlots) {
      int const last = ((tile + 1) * kDownFullStride - 1) / kUpgateSlotBytes;
      for (; fc1_retired <= last; ++fc1_retired)
        wait_bar(fc1_empty + fc1_retired * 8, (fc1_uses - 1) & 1);
    } else
      wait_bar(empty + slot * 8, ((tile / kDownSlots) - 1) & 1);
    // One request budget across the FC1->FC2 seam: FC2 tile t<kDownInFlight
    // also waits for FC1 items N-4+2t and N-3+2t to land, so the seam never
    // stacks the FC1 tail and the FC2 head on top of each other.
    if (tile < kDownInFlight) {
      constexpr int fc1_items = 3 * (H / kUpgateK);
      uint32_t const fc1_full = smem_address(c.smem + kUpgateFull);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int const item = fc1_items - 4 + 2 * tile + j;
        wait_bar(fc1_full + (item % kUpgateMegaSlots) * 8, (item / kUpgateMegaSlots) & 1);
      }
    }
    // Bound the value requests in flight below the buffer count: tile t is
    // requested only once tile t-kDownInFlight has landed (arrival, not MMA
    // completion, so the FC1->FC2 activation hand-off never stalls the
    // stream). Tile t-kDownInFlight is the only phase of its buffer's FULL
    // issued so far.
    if (tile >= kDownInFlight) {
      int const prior = tile - kDownInFlight;
      wait_bar(full + (prior % kDownSlots) * 8, (prior / kDownSlots) & 1);
    }
    uint8_t *page = c.smem + kRingOffset + slot * kDownFullStride;
    expect_bytes(done + slot * 8, 6144);
    bulk_load_ef(page + kDownScales, k.a.routed_fc2_sf + (size_t(expert) * H + row) * (I / 16),
                 6144, done + slot * 8);
    expect_bytes(full + slot * 8, 49152);
    tensor_load3_values(page, &k.maps.routed_fc2_full, expert * H + row, full + slot * 8);
  }
}

} // namespace mega_moe::token_parallel
