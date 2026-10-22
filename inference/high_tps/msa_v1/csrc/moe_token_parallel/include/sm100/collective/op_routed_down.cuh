#pragma once
#include "include/sm100/collective/expert_body.cuh"
namespace mega_moe::token_parallel {
// Fused mapped FC2 input: the four sibling slabs' FC1 output blocks arrive as
// epoch-tagged records. 36 pollers adopt only this invocation's records and
// place their bytes exactly where the generic staging puts routed_mid codes.
__device__ inline void prepare_routed_down_records(Context const &c, Args const &a, int role,
                                                   uint32_t sfb_base) {
  int const r = route(role), lane = c.tid & 31;
  uint8_t *const in = c.smem + 1024;
  uint32_t *const sfw = reinterpret_cast<uint32_t *>(c.smem + kRoutedMidStage + 128);
  if (c.tid < 36) {
    int const q = c.tid / 9, t = (c.tid % 9) / 3, p = c.tid % 3;
    uint4 const *rec = routed_mid_record(a, r, q, t, p);
    uint64_t lo, hi;
    int spins = 0;
    while (true) {
      routed_record_load(rec, lo, hi);
      if (uint32_t(hi >> 32) == c.local_epoch)
        break;
      if (++spins > 4)
        __nanosleep(32);
    }
    uint32_t *dst = reinterpret_cast<uint32_t *>(in + 96 * q + 32 * t + 12 * p);
    dst[0] = uint32_t(lo);
    dst[1] = uint32_t(lo >> 32);
    if (p < 2)
      dst[2] = uint32_t(hi);
    else
      sfw[3 * q + t] = uint32_t(hi);
  }
  if (c.tid >= 64 && c.tid < 96)
    *reinterpret_cast<uint32_t *>(in + I / 2 + lane * 4) = 0u;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("bar.sync 3,128;" ::: "memory");
  uint32_t sf[12];
#pragma unroll
  for (int g = 0; g < 12; ++g)
    sf[g] = sfw[g];
  uint32_t const target = c.tmem + sfb_base + (uint32_t(c.tid / 32) << 21);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0],"
               "{%1,%9,%2,%9,%3,%9,%4,%9,%5,%9,%6,%9,%7,%9,%8,%9};" ::"r"(target),
               "r"(sf[0]), "r"(sf[1]), "r"(sf[2]), "r"(sf[3]), "r"(sf[4]), "r"(sf[5]), "r"(sf[6]),
               "r"(sf[7]), "r"(0)
               : "memory");
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0],"
               "{%1,%5,%2,%5,%3,%5,%4,%5};" ::"r"(target + 16),
               "r"(sf[8]), "r"(sf[9]), "r"(sf[10]), "r"(sf[11]), "r"(0)
               : "memory");
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  asm volatile("bar.sync 3,128;" ::: "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// Each formatter owns a disjoint K256 slice and a private native-copy source.
// Raw TMA panels remain intact, so the formatters need no cross-warp join.
__device__ inline void copy_down_scale_panel(Context const &c, uint8_t const *raw, int bank,
                                             uint32_t ready, int turn) {
  int const lane = c.tid & 31, piece = (c.tid - 128) / 32;
  if (piece >= 3)
    return;
  uint4 s[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    int const off = (lane + q * 32) * 48 + piece * 16;
    s[q] = *reinterpret_cast<uint4 const *>(raw + off);
  }
  uint32_t const retired = smem_address(c.smem + kOvRetired + piece * 8);
  // Current raw loads can overlap the preceding copy. Reuse only this
  // warp's scratch after its own asynchronous readers have retired.
  wait_bar(retired, (turn & 1) ^ 1);
  uint8_t *native = c.smem + kDownNativeScaleScratch + piece * 2048;
  *reinterpret_cast<uint4 *>(native + lane * 16) = make_uint4(s[0].x, s[1].x, s[2].x, s[3].x);
  *reinterpret_cast<uint4 *>(native + 512 + lane * 16) = make_uint4(s[0].y, s[1].y, s[2].y, s[3].y);
  *reinterpret_cast<uint4 *>(native + 1024 + lane * 16) =
      make_uint4(s[0].z, s[1].z, s[2].z, s[3].z);
  *reinterpret_cast<uint4 *>(native + 1536 + lane * 16) =
      make_uint4(s[0].w, s[1].w, s[2].w, s[3].w);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncwarp();
  if (down_elect_one()) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
    for (int g = 0; g < 4; ++g)
      shared_tc::copy_scales(c.tmem + 96 + bank * 48 + piece * 16 + g * 4,
                             shared_tc::scale_desc(native + g * 512));
    // The common readiness carries all three formatter groups; this warp's
    // private completion separately controls its shared scratch lifetime.
    shared_tc::commit(ready);
    shared_tc::commit(retired);
  }
  if (turn == H / (4 * 128) - 1)
    wait_bar(retired, turn & 1);
}

// Fused mapped FC2 against produce_down_overlap's buffers. The formatter
// warps enter only after this CTA's FC1 publication join, so FC1's TMEM
// accumulators, scale banks and its shared input/bridge are all retired.
__device__ inline void op_routed_down_overlap(Context &c, Args const &a, int role) {
  uint32_t const sfb_base = 96u + uint32_t(kDownSlots) * 48u;
  if (c.tid < 128)
    prepare_routed_down_records(c, a, role, sfb_base);
  int const first_row = slab(role) * 1536;
  uint8_t *const payload = c.smem + kRingOffset;
  uint32_t const full = smem_address(c.smem + kOvFull), empty = smem_address(c.smem + kOvEmpty);
  uint32_t const done = smem_address(c.smem + kOvDone);
  uint32_t const loaded = smem_address(c.smem + kOvLoaded);
  // Two issuers alternate independent output tiles; each tile's scale bank
  // and buffer are released by the completion commit of its own issuer.
  if (c.tid < 64 && down_elect_one()) {
    for (int tile = c.tid / 32; tile < 12; tile += 2) {
      int const slot = tile % kDownSlots, use = tile / kDownSlots;
      wait_bar(loaded + slot * 8, use & 1);
      wait_bar(full + slot * 8, use & 1);
      routed_tc::issue_down_tile(payload + slot * kDownFullStride, c.smem + 1024, c.tmem, tile,
                                 slot, sfb_base);
      shared_tc::commit(empty + slot * 8);
    }
  } else if (c.tid >= 128 && c.tid < 224) {
    for (int tile = 0; tile < 12; ++tile) {
      int const slot = tile % kDownSlots, use = tile / kDownSlots;
      wait_bar(done + slot * 8, use & 1);
      copy_down_scale_panel(c, payload + slot * kDownFullStride + kDownScales, slot,
                            loaded + slot * 8, tile);
    }
  } else if (c.tid >= 224 && selected(role) == 0)
    acquire_shared_down(c, a, role);
  consumer_sync();
  if (c.tid < 128) {
    // Down buffers 3 and 2 last retire tiles 11 and 10: one commit from each
    // issuer, covering all of that issuer's ordered MMAs (third phase each).
    wait_bar(empty + 3 * 8, 0);
    wait_bar(empty + 2 * 8, 0);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    float values[12];
#pragma unroll
    for (int tile = 0; tile < 12; ++tile) {
      uint32_t address = c.tmem + tile * 8 + (uint32_t(c.tid / 32) << 21);
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0},[%1];"
                   : "=f"(values[tile])
                   : "r"(address)
                   : "memory");
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    if (selected(role) == 0) {
      // The combine owner consumes its own route on chip: every final MMA has
      // completed, so the native scale scratch is free. Row tid+128*tile.
      float *const own = reinterpret_cast<float *>(c.smem + kDownNativeScaleScratch);
#pragma unroll
      for (int tile = 0; tile < 12; ++tile)
        own[tile * 128 + c.tid] = values[tile];
    } else
      store_routed_records(a, role, c.tid, values, c.local_epoch);
  }
}
} // namespace mega_moe::token_parallel
