#pragma once
#include "include/sm100/common/shared_tc.cuh"
#include "include/sm100/common/tma.cuh"

namespace mega_moe::token_parallel {
// Shared down alternates two complete K768 weight tiles. The producer can
// fill the next 96 KiB tile while Tensor Cores consume the current one. Each
// output tile has one TMA/TC handoff and preserves the K-page MMA order.
constexpr int kSharedDownWBytes = 6 * 16384;
constexpr int kSharedDownW0 = 0;
constexpr int kSharedDownW1 = kSharedDownW0 + kSharedDownWBytes;
constexpr int kSharedDownB = kSharedDownW1 + kSharedDownWBytes;
constexpr int kSharedDownSfa0 = kSharedDownB + 6 * 1024;
constexpr int kSharedDownSfa1 = kSharedDownSfa0 + 6 * 512;
constexpr int kSharedDownSfb = kSharedDownSfa1 + 6 * 512;
constexpr int kSharedDownRaw = kSharedDownSfb + 6 * 512;
constexpr int kSharedDownPayloadBytes = kSharedDownRaw + 512 * 24;
static_assert(kSharedDownPayloadBytes == kSmemBytes - kRingOffset);
constexpr int kSharedDownInputScale = kBridgeOffset;
static_assert(kSharedDownInputScale + T * 24 <= kRingOffset);

// Each output tile owns one scale bank for this invocation. All four banks
// and the common input scales fit beside the accumulator in 128 columns.
constexpr int kSharedDownTmemColumns = 128;
constexpr int kSharedDownScaleBase = 8, kSharedDownInputScaleBase = 104;
static_assert(kSharedDownScaleBase + 4 * 24 == kSharedDownInputScaleBase);
static_assert(kSharedDownInputScaleBase + 24 == kSharedDownTmemColumns);

__device__ inline void store_down_scale8(uint32_t address, uint32_t x0, uint32_t y0, uint32_t z0,
                                         uint32_t w0, uint32_t x1, uint32_t y1, uint32_t z1,
                                         uint32_t w1) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 "
               "[%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"r"(address),
               "r"(x0), "r"(y0), "r"(z0), "r"(w0), "r"(x1), "r"(y1), "r"(z1), "r"(w1)
               : "memory");
}

__device__ __forceinline__ bool down_elect_one() {
  uint32_t one;
  asm volatile("{\n\t.reg .pred p;\n\telect.sync _|p, 0xffffffff;\n\t"
               "selp.u32 %0,1,0,p;\n\t}"
               : "=r"(one)::"memory");
  return one != 0;
}

__device__ inline void issue_down_tile(Context const &c, uint32_t a_base, uint32_t b_base, int ping,
                                       int tile) {
  // One fused asm block: 24 ordered block-scale MMAs with running 64-bit
  // descriptor adds, four immediate instruction descriptors, and two
  // predicates hoisted out of the chain (down from 24 mov+setp pairs).
  constexpr uint64_t upper = 0x4000404000000000ull;
  uint32_t const a_tile = a_base + uint32_t(ping) * (kSharedDownWBytes / 16);
  uint64_t const ad0 = upper | uint64_t(a_tile);
  uint64_t const bd0 = upper | uint64_t(b_base);
  uint32_t const sfa0 = c.tmem + kSharedDownScaleBase + tile * 24;
  uint32_t const sfb0 = c.tmem + kSharedDownInputScaleBase;
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  asm volatile("{\n\t"
               ".reg .pred pf,pt;\n\t"
               ".reg .b64 ad,bd;\n\t"
               ".reg .b32 sa,sb,d0,d1,d2,d3;\n\t"
               "setp.eq.b32 pf,1,0;\n\t"
               "setp.ne.b32 pt,1,0;\n\t"
               "mov.b32 d0,0x08820000;\n\t"
               "mov.b32 d1,0x28820010;\n\t"
               "mov.b32 d2,0x48820020;\n\t"
               "mov.b32 d3,0x68820030;\n\t"
               "mov.b64 ad,%1;\n\t"
               "mov.b64 bd,%2;\n\t"
               "mov.b32 sa,%3;\n\t"
               "mov.b32 sb,%4;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d0,[sa],[sb],pf;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d1,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d2,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d3,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,1018;\n\t"
               "add.u64 bd,bd,58;\n\t"
               "add.u32 sa,sa,4;\n\t"
               "add.u32 sb,sb,4;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d0,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d1,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d2,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d3,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,1018;\n\t"
               "add.u64 bd,bd,58;\n\t"
               "add.u32 sa,sa,4;\n\t"
               "add.u32 sb,sb,4;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d0,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d1,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d2,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d3,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,1018;\n\t"
               "add.u64 bd,bd,58;\n\t"
               "add.u32 sa,sa,4;\n\t"
               "add.u32 sb,sb,4;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d0,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d1,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d2,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d3,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,1018;\n\t"
               "add.u64 bd,bd,58;\n\t"
               "add.u32 sa,sa,4;\n\t"
               "add.u32 sb,sb,4;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d0,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d1,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d2,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d3,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,1018;\n\t"
               "add.u64 bd,bd,58;\n\t"
               "add.u32 sa,sa,4;\n\t"
               "add.u32 sb,sb,4;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d0,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d1,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d2,[sa],[sb],pt;\n\t"
               "add.u64 ad,ad,2;\n\t"
               "add.u64 bd,bd,2;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],ad,bd,d3,[sa],[sb],pt;\n\t"
               "}" ::"r"(c.tmem),
               "l"(ad0), "l"(bd0), "r"(sfa0), "r"(sfb0)
               : "memory");
}

__device__ inline void load_down_full(void *dst, CUtensorMap const *map, int row,
                                      uint32_t barrier) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes."
               "L2::cache_hint "
               "[%0],[%1,{%5,%2,%3}],[%4],%6;" ::"r"(smem_address(dst)),
               "l"(map), "r"(row), "r"(0), "r"(barrier), "r"(0), "l"(evict_first_policy())
               : "memory");
}

__device__ __forceinline__ void begin_shared_down(KernelArgs const &k, int role, uint8_t *smem,
                                                  Ring const &ring) {
  int const row = (role - kRouted) * 384;
  uint8_t const *ws = k.a.shared_fc2_sf + size_t(row) * 24;
  uint8_t const *xs = k.a.shared_mid_sf;
  expect_bytes(ring.loaded, 384 * 24 + T * 24);
  bulk_load_ef(ring.payload + kSharedDownRaw, ws, 384 * 24, ring.loaded);
  bulk_load(smem + kSharedDownInputScale, xs, T * 24, ring.loaded);
  expect_page(ring.full, kSharedDownWBytes + 6 * 1024);
  load_down_full(ring.payload + kSharedDownW0, &k.maps.map[SharedFc2Value], row, ring.full);
  load_down_full(ring.payload + kSharedDownB, &k.maps.map[SharedMidValue], 0, ring.full);
}

__device__ inline void prefetch_down_full(CUtensorMap const *map, int row) {
  int const zero = 0;
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0,{%1,%2,%3}];" ::"l"(map),
               "r"(zero), "r"(row), "r"(zero)
               : "memory");
}

__device__ inline void stage_down_scales(Context const &c, uint8_t const *raw,
                                         uint8_t const *input_scales, int tile) {
  int const lane = c.producer.ptid & 31;
  uint32_t const partition = uint32_t(c.producer.ptid / 32) << 21;
#pragma unroll
  for (int p = 0; p < 6; p += 2) {
    uint32_t x0 = *reinterpret_cast<uint32_t const *>(raw + (tile * 128 + lane) * 24 + p * 4);
    uint32_t y0 = *reinterpret_cast<uint32_t const *>(raw + (tile * 128 + lane + 32) * 24 + p * 4);
    uint32_t z0 = *reinterpret_cast<uint32_t const *>(raw + (tile * 128 + lane + 64) * 24 + p * 4);
    uint32_t w0 = *reinterpret_cast<uint32_t const *>(raw + (tile * 128 + lane + 96) * 24 + p * 4);
    uint32_t x1 = *reinterpret_cast<uint32_t const *>(raw + (tile * 128 + lane) * 24 + (p + 1) * 4);
    uint32_t y1 =
        *reinterpret_cast<uint32_t const *>(raw + (tile * 128 + lane + 32) * 24 + (p + 1) * 4);
    uint32_t z1 =
        *reinterpret_cast<uint32_t const *>(raw + (tile * 128 + lane + 64) * 24 + (p + 1) * 4);
    uint32_t w1 =
        *reinterpret_cast<uint32_t const *>(raw + (tile * 128 + lane + 96) * 24 + (p + 1) * 4);
    store_down_scale8(c.tmem + partition + kSharedDownScaleBase + tile * 24 + p * 4, x0, y0, z0, w0,
                      x1, y1, z1, w1);
    if (tile == 0) {
      uint32_t packed0 =
          lane < T ? *reinterpret_cast<uint32_t const *>(input_scales + lane * 24 + p * 4) : 0u;
      uint32_t packed1 =
          lane < T ? *reinterpret_cast<uint32_t const *>(input_scales + lane * 24 + (p + 1) * 4)
                   : 0u;
      store_down_scale8(c.tmem + partition + kSharedDownInputScaleBase + p * 4, packed0, 0, 0, 0,
                        packed1, 0, 0, 0);
    }
  }
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ inline void produce_shared_down_tile(Context &c, KernelArgs const &k, int role) {
  auto &prod = c.producer;
  uint8_t *buf = prod.ring.payload;
  // Shared up/gate and shared down reuse this allocation. Retire every prior ring
  // slot before overlaying the complete opt-in shared-memory payload.
#pragma unroll
  for (int i = 0; i < kSlots; ++i) {
    uint32_t next = prod.item + uint32_t(i);
    wait_bar(prod.ring.empty + (next & 3) * 8, ((next >> 2) & 1) ^ 1);
  }
  int const first_row = (role - kRouted) * 384;
  uint8_t *raw = buf + kSharedDownRaw;
  uint8_t *input_scales = c.smem + kSharedDownInputScale;
  if (prod.ptid == 0)
    begin_shared_down(k, role, c.smem, prod.ring);
  for (int tile = 0; tile < 3; ++tile) {
    uint32_t const item = prod.item;
    int const ping = int(item & 1u);
    int row = first_row + tile * 128;
    uint8_t *weights = buf + (ping ? kSharedDownW1 : kSharedDownW0);
    if (tile >= 2 && prod.ptid == 0)
      prefetch_down_full(&k.maps.map[SharedFc2Value], row);
    // These scale columns are disjoint from every in-flight tile.
    if (tile >= 2)
      stage_down_scales(c, raw, input_scales, tile);
    prod.acquire();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    if (prod.ptid == 0 && tile != 0) {
      expect_page(prod.page_barrier(), kSharedDownWBytes);
      load_down_full(weights, &k.maps.map[SharedFc2Value], row, prod.page_barrier());
    }
    // The full value/input transfers are already in flight. Consume the
    // independent current-call weight-scale slab only at its first use.
    if (tile == 0) {
      prod.wait_loaded();
      producer_sync();
    }
    if (tile < 2)
      stage_down_scales(c, raw, input_scales, tile);
    prod.publish();
    // Items 48,49,52,53 after up/gate alternate two full-K buffers.
    if (tile & 1)
      prod.item += 2;
  }
}

__device__ inline void consume_shared_down_tile(Context &c, Args const &a, int role) {
  auto &con = c.consumer;
  uint8_t const *buf = con.ring.payload;
  uint32_t const a_base = uint32_t(shared_tc::value_desc(buf + kSharedDownW0));
  uint32_t const b_base = uint32_t(shared_tc::value_desc(buf + kSharedDownB));
#pragma unroll 1
  for (int tile = 0; tile < 3; ++tile) {
    uint32_t const item = con.item;
    int const ping = int(item & 1u);
    con.acquire();
    if (c.tid < 32 && down_elect_one()) {
      issue_down_tile(c, a_base, b_base, ping, tile);
      shared_tc::commit(con.completion());
    }
    con.release_tc();
    if (c.tid < 128) {
      float v[8];
      shared_tc::load_acc8(c.tmem + (uint32_t(c.tid / 32) << 21), v);
      int row = (role - kRouted) * 384 + tile * 128 + c.tid;
#pragma unroll
      for (int t = 0; t < T; ++t)
        c.shared_down[size_t(t) * H + row] = v[t];
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    if (tile & 1)
      con.item += 2;
    if (tile + 1 < 3)
      consumer_sync();
  }
  // Publish readiness for the in-kernel combine.
  publish_local(c, a, kDownReady, role);
}
} // namespace mega_moe::token_parallel
