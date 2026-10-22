#pragma once

#include <cuda.h>

#include <cstdint>

#include "common/kernel/include/sm100_m3_decode_storage.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/arch/util.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

namespace cutlass::m3_decode::collective {

struct Sm100M3DecodeBarrier {
  CUTLASS_DEVICE static void expect_tx_cluster_lane0(uint64_t *barrier, uint32_t bytes,
                                                     uint32_t lane_idx) {
    if (lane_idx == 0) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(barrier, bytes);
    }
  }

  CUTLASS_DEVICE static void arrive(uint64_t *barrier) {
    cutlass::arch::ClusterBarrier::arrive(barrier);
  }

  CUTLASS_DEVICE static void arrive_cluster_zero(uint64_t *barrier) {
    cutlass::arch::ClusterBarrier::arrive(barrier);
  }

  CUTLASS_DEVICE static void umma_arrive(uint64_t *barrier) { cutlass::arch::umma_arrive(barrier); }

  CUTLASS_DEVICE static bool try_wait(uint64_t *barrier, uint32_t phase) {
    return cutlass::arch::ClusterBarrier::try_wait(barrier, phase);
  }

  CUTLASS_DEVICE static cutlass::BarrierStatus try_wait_token(uint64_t *barrier, uint32_t phase) {
    bool const barrier_status = cutlass::arch::ClusterBarrier::try_wait(barrier, phase);
    return barrier_status ? cutlass::BarrierStatus::WaitDone : cutlass::BarrierStatus::WaitAgain;
  }

  CUTLASS_DEVICE static void wait(uint64_t *barrier, uint32_t phase, cutlass::BarrierStatus token,
                                  uint32_t wait_id = 0) {
    (void)wait_id;
    if (token == cutlass::BarrierStatus::WaitAgain) {
#if M3_DECODE_WAIT_DEADLINE
      while (!cutlass::arch::ClusterBarrier::try_wait(barrier, phase)) {
        m3_decode_check_deadline();
      }
#else
      cutlass::arch::ClusterBarrier::wait(barrier, phase);
#endif
    }
  }

  CUTLASS_DEVICE static void wait(uint64_t *barrier, uint32_t phase, uint32_t wait_id = 0) {
    (void)wait_id;
    cutlass::BarrierStatus const token = try_wait_token(barrier, phase);
    wait(barrier, phase, token);
  }
};

struct Sm100M3DecodeNamedBarrier {
  CUTLASS_DEVICE static void sync(uint32_t num_threads, uint32_t barrier_id) {
    cutlass::arch::NamedBarrier::sync(
        num_threads, static_cast<cutlass::arch::ReservedNamedBarriers>(barrier_id));
  }
};

struct Sm100M3DecodeTma {
  CUTLASS_DEVICE static void prefetch_tensormap(CUtensorMap const *desc) {
    cute::prefetch_tma_descriptor(desc);
  }

  CUTLASS_DEVICE static void
  load_2d_predicated(CUtensorMap const *desc, void *smem_ptr, uint64_t *barrier, int c0, int c1,
                     uint32_t pred,
                     cute::TMA::CacheHintSm90 cache_hint = cute::TMA::CacheHintSm90::EVICT_NORMAL) {
    if (pred != 0) {
      cute::SM90_TMA_LOAD_2D::copy(desc, barrier, static_cast<uint64_t>(cache_hint), smem_ptr, c0,
                                   c1);
    }
  }

  CUTLASS_DEVICE static void load_2d_gather4_predicated(CUtensorMap const *desc, void *smem_ptr,
                                                        uint64_t *barrier, int c0, int c1_i0,
                                                        int c1_i1, int c1_i2, int c1_i3,
                                                        uint32_t pred) {
    if (pred != 0) {
      uint64_t const descriptor = reinterpret_cast<uint64_t>(desc);
      uint32_t const destination = cute::cast_smem_ptr_to_uint(smem_ptr);
      uint32_t const mbarrier = cute::cast_smem_ptr_to_uint(barrier);
      asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4."
                   "mbarrier::complete_tx::bytes.cta_group::1 "
                   "[%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
                   :
                   : "r"(destination), "l"(descriptor), "r"(mbarrier), "r"(c0), "r"(c1_i0),
                     "r"(c1_i1), "r"(c1_i2), "r"(c1_i3)
                   : "memory");
    }
  }

  CUTLASS_DEVICE static void
  load_3d_predicated(CUtensorMap const *desc, void *smem_ptr, uint64_t *barrier, int c0, int c1,
                     int c2, uint32_t pred,
                     cute::TMA::CacheHintSm90 cache_hint = cute::TMA::CacheHintSm90::EVICT_NORMAL) {
    if (pred != 0) {
      cute::SM90_TMA_LOAD_3D::copy(desc, barrier, static_cast<uint64_t>(cache_hint), smem_ptr, c0,
                                   c1, c2);
    }
  }

  CUTLASS_DEVICE static void load_3d_multicast_predicated(
      CUtensorMap const *desc, void *smem_ptr, uint64_t *barrier, uint16_t multicast_mask, int c0,
      int c1, int c2, uint32_t pred,
      cute::TMA::CacheHintSm90 cache_hint = cute::TMA::CacheHintSm90::EVICT_NORMAL) {
    if (pred != 0) {
      cute::SM90_TMA_LOAD_MULTICAST_3D::copy(
          desc, barrier, multicast_mask, static_cast<uint64_t>(cache_hint), smem_ptr, c0, c1, c2);
    }
  }

  CUTLASS_DEVICE static void
  load_4d_predicated(CUtensorMap const *desc, void *smem_ptr, uint64_t *barrier, int c0, int c1,
                     int c2, int c3, uint32_t pred,
                     cute::TMA::CacheHintSm90 cache_hint = cute::TMA::CacheHintSm90::EVICT_NORMAL) {
    if (pred != 0) {
      cute::SM90_TMA_LOAD_4D::copy(desc, barrier, static_cast<uint64_t>(cache_hint), smem_ptr, c0,
                                   c1, c2, c3);
    }
  }

  CUTLASS_DEVICE static void load_4d_multicast_predicated(
      CUtensorMap const *desc, void *smem_ptr, uint64_t *barrier, uint16_t multicast_mask, int c0,
      int c1, int c2, int c3, uint32_t pred,
      cute::TMA::CacheHintSm90 cache_hint = cute::TMA::CacheHintSm90::EVICT_NORMAL) {
    if (pred != 0) {
      cute::SM90_TMA_LOAD_MULTICAST_4D::copy(desc, barrier, multicast_mask,
                                             static_cast<uint64_t>(cache_hint), smem_ptr, c0, c1,
                                             c2, c3);
    }
  }

  CUTLASS_DEVICE static void load_5d_predicated(CUtensorMap const *desc, void *smem_ptr,
                                                uint64_t *barrier, int c0, int c1, int c2, int c3,
                                                int c4, uint32_t pred) {
    if (pred != 0) {
      cute::SM90_TMA_LOAD_5D::copy(desc, barrier,
                                   static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                   smem_ptr, c0, c1, c2, c3, c4);
    }
  }
};

} // namespace cutlass::m3_decode::collective
