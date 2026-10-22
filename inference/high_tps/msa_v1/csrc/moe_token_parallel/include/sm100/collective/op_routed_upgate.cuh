#pragma once
#include "include/sm100/collective/expert_body.cuh"
namespace mega_moe::token_parallel {
constexpr int kUpgateFormatBase = 64;
static_assert(kUpgateK % 512 == 0 && kUpgateFormatBase + (kUpgateK / 256) * 32 <= kConsumers);
// One formatter warp per K256 slice of an item.
constexpr int kUpgateFormatters = kUpgateK / 256;
// Refill all input scale words for this invocation. N8 addresses one scale
// live column per K64 contraction at the legal two-column spacing, duplicated
// across the four TMEM lane groups.
__device__ inline void upgate_input_scales(Context const &c) {
  wait_bar(smem_address(c.smem + kUpgateInputBar), 0);
  uint32_t lane_group = uint32_t(c.tid / 32) << 21;
  auto sf = reinterpret_cast<uint32_t const *>(c.smem + kUpgateInputScale);
  for (int g = 0; g < H / 64; g += 16) {
    uint32_t address = c.tmem + 24 + g * 2 + lane_group;
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
                 "{%1,%17,%2,%17,%3,%17,%4,%17,%5,%17,%6,%17,%7,%17,%8,%17,"
                 "%9,%17,%10,%17,%11,%17,%12,%17,%13,%17,%14,%17,%15,%17,%16,%17};" ::"r"(address),
                 "r"(sf[g]), "r"(sf[g + 1]), "r"(sf[g + 2]), "r"(sf[g + 3]), "r"(sf[g + 4]),
                 "r"(sf[g + 5]), "r"(sf[g + 6]), "r"(sf[g + 7]), "r"(sf[g + 8]), "r"(sf[g + 9]),
                 "r"(sf[g + 10]), "r"(sf[g + 11]), "r"(sf[g + 12]), "r"(sf[g + 13]),
                 "r"(sf[g + 14]), "r"(sf[g + 15]), "r"(0)
                 : "memory");
  }
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  asm volatile("bar.sync 3,128;" ::: "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ inline void upgate_issue(uint8_t const *page, uint32_t base, int kpage,
                                    uint8_t const *input, int slot, int tile) {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  // Only N0 is a public result for this token-owned CTA. A non-swizzled
  // descriptor reads its two K32 byte vectors directly from the input prefix.
  // The other seven N columns read initialized shifted/padded input bytes.
  uint64_t db = shared_tc::scale_desc(input + kpage * (kUpgateK / 2));
  constexpr uint32_t descriptor = (1u << 7) | (1u << 10) | (1u << 17) | (1u << 27);
  uint32_t acc = base + uint32_t(tile) * 8;
#pragma unroll
  for (int g = 0; g < kUpgateK / 64; ++g) {
    uint64_t da = shared_tc::value_desc(page + (g / 4) * 1024);
    da = (da & ~(0x3fffull << 32)) | (shared_tc::desc_encode((kUpgateK / 256) * 1024) << 32);
    da += (g & 3) * 2;
    uint32_t sfa = base + 216 + slot * kUpgateScaleWidth + g * 4,
             sfb = base + 24 + (kpage * (kUpgateK / 64) + g) * 2;
    asm volatile("{ .reg .pred acc; setp.ne.b32 acc,%6,0; "
                 "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
                 "[%0],%1,%2,%3,[%4],[%5],acc; }" ::"r"(acc),
                 "l"(da), "l"(db + g * 2), "r"(descriptor), "r"(sfa), "r"(sfb),
                 "r"(int(kpage != 0 || g != 0))
                 : "memory");
  }
}

// The mapped path uses a producer warp for issue, so all four TMEM reader
// warps are available for epilogues while later weight pages are in flight.
__device__ inline void routed_upgate_mapped_issue(Context &c, Args const &a) {
  constexpr int pages = H / kUpgateK, slots = kUpgateMegaSlots;
  if (down_elect_one()) {
    wait_bar(smem_address(c.smem + kUpgateScalesBar), 0);
    uint32_t const full = smem_address(c.smem + kUpgateFull);
    uint32_t const empty = smem_address(c.smem + kUpgateEmpty);
    uint32_t const ready = smem_address(c.smem + kUpgateTileReady);
    uint32_t item = 0;
    for (int tile = 0; tile < 3; ++tile)
      for (int page = 0; page < pages; ++page, ++item) {
        int const slot = item % slots;
        wait_bar(full + slot * 8, (item / slots) & 1);
        upgate_issue(upgate_slot(c.smem, item, slots), c.tmem, page, c.smem + kUpgateInput, slot,
                     tile);
        // EMPTY retires all value/scale readers. READY independently
        // publishes the complete tile to its four physical owner warps.
        shared_tc::commit(empty + slot * 8);
        if (page == pages - 1)
          shared_tc::commit(ready + tile * 8);
      }
  }
  c.producer.item += 3 * (H / 256);
}

__device__ inline void upgate_layout(Context const &c, uint8_t *page, int slot) {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  int tid = c.tid - kUpgateFormatBase;
  // Each warp reads one K256 scale slice exactly once.  Native TMEM copies
  // replicate its packed rows into the four lane groups required by MMA.
  int lane = tid & 31, piece = tid / 32;
  static_assert(kUpgateK % 256 == 0 && kUpgateK <= 1536);
  uint4 w[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    int m = lane + q * 32;
    int off = (piece / 2) * 4096 + m * 32 + (piece & 1) * 16;
    off ^= (off >> 3) & 0x10;
    w[q] = *reinterpret_cast<uint4 const *>(page + kUpgateRawScale + off);
  }
  // Two formatters share one raw 4 KiB panel. Join exactly that pair after
  // every raw read, before either warp writes its native 2 KiB half.
  asm volatile("bar.sync %0,64;" ::"r"(6 + piece / 2) : "memory");
  uint8_t *native = page + kUpgateRawScale + piece * 2048;
  *reinterpret_cast<uint4 *>(native + lane * 16) = make_uint4(w[0].x, w[1].x, w[2].x, w[3].x);
  *reinterpret_cast<uint4 *>(native + 512 + lane * 16) = make_uint4(w[0].y, w[1].y, w[2].y, w[3].y);
  *reinterpret_cast<uint4 *>(native + 1024 + lane * 16) =
      make_uint4(w[0].z, w[1].z, w[2].z, w[3].z);
  *reinterpret_cast<uint4 *>(native + 1536 + lane * 16) =
      make_uint4(w[0].w, w[1].w, w[2].w, w[3].w);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncwarp();
  if (down_elect_one()) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
    for (int g = 0; g < 4; ++g)
      shared_tc::copy_scales(c.tmem + 216 + slot * kUpgateScaleWidth + piece * 16 + g * 4,
                             shared_tc::scale_desc(native + g * 512));
    // FULL also counts every formatter's native scale-copy completion. Its
    // single acquire covers both the value TMA and every TMEM scale panel.
    shared_tc::commit(smem_address(c.smem + kUpgateFull) + slot * 8);
  }
}

__device__ inline void routed_upgate_mapped_format(Context const &c, Args const &a, int role) {
  constexpr int pages = H / kUpgateK, slots = kUpgateMegaSlots;
  static_assert(kUpgateK % 512 == 0 && kUpgateFormatters <= 4);
  Context formatter = c;
  formatter.tid = kUpgateFormatBase + c.tid - 128;
  // Four formatters cover the four K256 slices of a K1024 item. Each
  // snapshots its native scale panel and the two disjoint pairs keep
  // their own rendezvous. All four observe every phase of all three slots.
  for (uint32_t item = 0; item < 3 * pages; ++item) {
    wait_bar(smem_address(c.smem + kUpgateScaleReady) + (item % slots) * 8, (item / slots) & 1);
    upgate_layout(formatter, upgate_slot(c.smem, item, slots), item % slots);
  }
}

// NVFP4 quantization of one routed FC1 output, additionally staging each code
// byte for the FC1->FC2 records.
__device__ inline void quant_routed_staged(float v, uint8_t *dst, uint8_t *sf, int i,
                                           uint8_t *stage, int li) {
  float amax = group_max(fabsf(v), 16);
  uint8_t s = m3::to_e4m3(fminf(amax * (1.f / 6.f), 448.f));
  float scale = m3::e4m3(s), inv = fminf(reciprocal(scale), 3.4028234663852886e38f);
  if (!(scale > 0.f))
    inv = 0.f;
  float q = v * inv, next = __shfl_down_sync(0xffffffffu, q, 1, 16);
  if (!(i & 1)) {
    uint8_t const b = pack_fp4(q, next);
    dst[i / 2] = b;
    stage[li / 2] = b;
  }
  if (!(i & 15)) {
    sf[i / 16] = s;
    stage[32 + li / 16] = s;
  }
}
__device__ inline void routed_upgate_mapped_epilogue(Context &c, Args const &a, int role) {
  if (c.tid >= 64 && c.tid < 96)
    stage_upgate_input(c, a, role);
  upgate_input_scales(c);
  if (c.tid == 0) {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    arrive_bar(smem_address(c.smem + kUpgateScalesBar));
  }
  int const r = route(role), first_row = slab(role) * 192;
  for (int tile = 0; tile < 3; ++tile) {
    wait_bar(smem_address(c.smem + kUpgateTileReady) + tile * 8, 0);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    uint32_t const address = c.tmem + tile * 8 + (uint32_t(c.tid / 32) << 21);
    float value;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                 : "=f"(value)
                 : "r"(address)
                 : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    c.bridge[tile * 128 + c.tid] = value;
    asm volatile("bar.sync 3,128;" ::: "memory");
    if (c.tid < 64) {
      wait_routes();
      float const alpha = a.fc1_alpha[route_expert(selected(role))];
      float const h =
          gated(c.bridge[tile * 128 + c.tid] * alpha, c.bridge[tile * 128 + c.tid + 64] * alpha);
      quant_routed_staged(h, a.routed_mid + size_t(r) * (I / 2),
                          a.routed_mid_sf + size_t(r) * (I / 16), first_row + tile * 64 + c.tid,
                          c.smem + kRoutedMidStage + tile * 36, c.tid);
    }
  }
  // Publish the three 36-byte output blocks as epoch-tagged records; the four
  // sibling FC2 CTAs read data and readiness in one round trip.
  asm volatile("bar.sync 3,128;" ::: "memory");
  if (c.tid < 9) {
    int const tile = c.tid / 3, part = c.tid % 3;
    uint32_t const *w =
        reinterpret_cast<uint32_t const *>(c.smem + kRoutedMidStage + tile * 36 + part * 12);
    record_store_words(routed_mid_record(a, r, slab(role), tile, part), w[0], w[1], w[2],
                       c.local_epoch);
  }
}

__device__ inline void op_routed_upgate(Context &c, Args const &a, int role) {
  if (c.tid < 128)
    routed_upgate_mapped_epilogue(c, a, role);
  else if (c.tid < 128 + kUpgateFormatters * 32)
    routed_upgate_mapped_format(c, a, role);
  // FC2 inputs travel as records. This join retires FC1's TMEM banks and
  // shared input before the FC2 formatter warps enter.
  consumer_sync();
}
} // namespace mega_moe::token_parallel
