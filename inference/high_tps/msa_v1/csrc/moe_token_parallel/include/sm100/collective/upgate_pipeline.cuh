#pragma once
#include "include/sm100/common/shared_tc.cuh"
#include "include/sm100/common/tma.cuh"

namespace mega_moe::token_parallel {

constexpr int kSharedGuBStorage = 16384 * kSharedGuPagesPerSlot;
constexpr int kSharedGuSfaStorage = 17408 * kSharedGuPagesPerSlot;
constexpr int kSharedGuSfbStorage = 17920 * kSharedGuPagesPerSlot;

__device__ __forceinline__ uint4 upgate_scale16(uint8_t const *p) {
  return *reinterpret_cast<uint4 const *>(p);
}

__device__ inline void acquire_upgate(Producer const &p) {
  wait_bar(p.ring.empty + (p.item % kSharedGuRingSlots) * 8,
           ((p.item / kSharedGuRingSlots) & 1) ^ 1);
}

__device__ inline void upgate_pair_load(void *dst, CUtensorMap const *map, int k0, int row,
                                        uint32_t barrier) {
  asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes."
               "L2::cache_hint "
               "[%0],[%1,{%2,%2,%2,%3,%4}],[%5],%6;" ::"r"(smem_address(dst)),
               "l"(map), "r"(0), "r"(row / 48), "r"(k0), "r"(barrier), "l"(evict_first_policy())
               : "memory");
}
__device__ inline void upgate_input_load(void *dst, CUtensorMap const *map, int k0,
                                         uint32_t barrier) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
               "[%0],[%1,{%2,%3,%4}],[%5];" ::"r"(smem_address(dst)),
               "l"(map), "r"(0), "r"(0), "r"(k0), "r"(barrier)
               : "memory");
}

// One elected thread issues both TMA transfers per slot and supplies the two
// value-producer credits itself.
__device__ inline void produce_upgate_values(Context &c, KernelArgs const &k, int row) {
  if (c.producer.ptid != 96)
    return;
  for (int group = 0; group < H / (128 * kSharedGuPagesPerSlot); ++group) {
    acquire_upgate(c.producer);
    uint8_t *slot =
        c.producer.ring.payload + (c.producer.item % kSharedGuRingSlots) * kSharedGuSlotBytes;
    uint32_t const full = c.producer.ring.full + (c.producer.item % kSharedGuRingSlots) * 8;
    expect_page(full, kSharedGuPagesPerSlot * (16384 + 1024));
    upgate_pair_load(slot, &k.maps.map[SharedFc1Value], group * kSharedGuPagesPerSlot, row, full);
    upgate_input_load(slot + kSharedGuBStorage, &k.maps.map[SharedInputValue],
                      group * kSharedGuPagesPerSlot, full);
    arrive_bar(full, 2);
    ++c.producer.item;
  }
}

__device__ inline void produce_upgate_weight_scales(Context &c, KernelArgs const &k, int row) {
  int const lane = c.producer.ptid & 31;
  int const base_row = lane;
  constexpr int panel = kSharedGuSfaStorage;
  // One warp owns each operand's complete scale panels. Each call loads all
  // current raw codes; vectorization only changes their transport.
  for (int first = 0; first < H / 128; first += 12) {
    uint4 raw[4][3];
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      int const m = base_row + chunk * 32;
#pragma unroll
      for (int vec = 0; vec < 3; ++vec) {
        int const offset = first * 4 + vec * 16;
        int const logical = row + (m & 63) + (m >= 64 ? I : 0);
        raw[chunk][vec] =
            (m & 63) < 48 ? upgate_scale16(k.a.shared_fc1_sf + size_t(logical) * (H / 32) + offset)
                          : make_uint4(0, 0, 0, 0);
      }
    }
#pragma unroll
    for (int j = 0; j < 12; j += kSharedGuPagesPerSlot) {
      acquire_upgate(c.producer);
      uint8_t *slot =
          c.producer.ring.payload + (c.producer.item % kSharedGuRingSlots) * kSharedGuSlotBytes;
#pragma unroll
      for (int sub = 0; sub < kSharedGuPagesPerSlot; ++sub) {
        int const q = j + sub;
        uint8_t *scales = slot + panel + sub * 512;
        uint32_t code[4];
#pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
          uint4 const bits = raw[chunk][q / 4];
          code[chunk] = (q & 3) == 0   ? bits.x
                        : (q & 3) == 1 ? bits.y
                        : (q & 3) == 2 ? bits.z
                                       : bits.w;
        }
        // Lane r supplies native rows r,r+32,r+64,r+96 as one vector.
        // This preserves (row&31)*16+(row>>5)*4 byte placement exactly.
        *reinterpret_cast<uint4 *>(scales + lane * 16) =
            make_uint4(code[0], code[1], code[2], code[3]);
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncwarp();
      if (shared_tc::elect_issuer()) {
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
        for (int sub = 0; sub < kSharedGuPagesPerSlot; ++sub) {
          uint32_t const target =
              c.tmem + 8 +
              ((c.producer.item % kSharedGuRingSlots) * kSharedGuPagesPerSlot + sub) * 8;
          shared_tc::copy_scales(target, shared_tc::scale_desc(slot + panel + sub * 512));
        }
        // Each producer's FULL credit tracks completion of all of its scale
        // copies. Both consumer warps acquire FULL before issuing MMA, so
        // this handoff does not depend on cross-warp implicit pipelining.
        shared_tc::commit(c.producer.ring.full + (c.producer.item % kSharedGuRingSlots) * 8);
      }
      ++c.producer.item;
    }
  }
}

__device__ inline void produce_upgate_input_scales_impl(Context &c, KernelArgs const &k, int row) {
  int const lane = c.producer.ptid & 31;
  bool const valid = lane < T;
  for (int first = 0; first < H / 128; first += 12) {
    uint4 codes[3];
#pragma unroll
    for (int vec = 0; vec < 3; ++vec) {
      int const offset = first * 4 + vec * 16;
      codes[vec] = valid ? upgate_scale16(k.a.shared_input_sf + size_t(lane) * (H / 32) + offset)
                         : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int j = 0; j < 12; j += kSharedGuPagesPerSlot) {
      acquire_upgate(c.producer);
      uint8_t *slot =
          c.producer.ring.payload + (c.producer.item % kSharedGuRingSlots) * kSharedGuSlotBytes;
#pragma unroll
      for (int sub = 0; sub < kSharedGuPagesPerSlot; ++sub) {
        int const q = j + sub;
        uint4 const bits = codes[q / 4];
        uint32_t const code = (q & 3) == 0   ? bits.x
                              : (q & 3) == 1 ? bits.y
                              : (q & 3) == 2 ? bits.z
                                             : bits.w;
        uint8_t *scales = slot + kSharedGuSfbStorage + sub * 512;
        *reinterpret_cast<uint4 *>(scales + lane * 16) = make_uint4(code, 0u, 0u, 0u);
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncwarp();
      if (shared_tc::elect_issuer()) {
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
        for (int sub = 0; sub < kSharedGuPagesPerSlot; ++sub) {
          uint32_t const target =
              c.tmem + 12 +
              ((c.producer.item % kSharedGuRingSlots) * kSharedGuPagesPerSlot + sub) * 8;
          shared_tc::copy_scales(target,
                                 shared_tc::scale_desc(slot + kSharedGuSfbStorage + sub * 512));
        }
        shared_tc::commit(c.producer.ring.full + (c.producer.item % kSharedGuRingSlots) * 8);
      }
      ++c.producer.item;
    }
  }
}

// Four independent FP32 TMEM accumulators, one per K32 phase, placed in the
// otherwise-unused columns above the scale panels. Each K32 product chain
// preserves its accumulation order and stays FP32 end to end; the final
// cross-chain sums are FP32 register adds in a fixed epilogue order.
__host__ __device__ constexpr uint32_t upgate_acc_column(int k) {
  return k == 0 ? 0u : uint32_t(96 + 8 * k);
}

__device__ inline void issue_upgate_ready_page(uint8_t const *slot, uint32_t tmem, int native_slot,
                                               int sub, bool accumulate, int half) {
  uint32_t const sfa = tmem + 8 + native_slot * 8, sfb = sfa + 4;
  uint64_t const ad = shared_tc::value_desc(slot + sub * 16384);
  uint64_t const bd = shared_tc::value_desc(slot + kSharedGuBStorage + sub * 1024);
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    int const k = half + j;
    uint32_t const word = uint32_t(k) << 30;
    shared_tc::mma_mxf8(tmem + upgate_acc_column(k), ad + uint64_t(k * 2), bd + uint64_t(k * 2),
                        sfa | word, sfb | word, accumulate, uint32_t(k));
  }
}

__device__ inline void produce_upgate(Context &c, KernelArgs const &k, int row) {
  if (c.producer.ptid >= 32 && c.producer.ptid < 64)
    produce_upgate_weight_scales(c, k, row);
  else if (c.producer.ptid >= 64 && c.producer.ptid < 96)
    produce_upgate_input_scales_impl(c, k, row);
  else
    produce_upgate_values(c, k, row);
  // Four cycles leave all full/empty barriers at phase zero. Resume the
  // frozen FC2 ring with the same item count and generation phase.
  c.producer.item = kSharedGuResumeItems;
}
} // namespace mega_moe::token_parallel
