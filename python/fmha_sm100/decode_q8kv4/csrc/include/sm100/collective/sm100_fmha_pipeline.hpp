// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include <cuda.h>

#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/arch/util.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

namespace cutlass::fmha::collective {

struct Sm100FmhaBarrier {
  CUTLASS_DEVICE static void expect_tx_cluster_lane0(uint64_t* barrier,
                                                     uint32_t bytes,
                                                     uint32_t lane_idx) {
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
        barrier, bytes, lane_idx, static_cast<uint32_t>(lane_idx == 0));
  }

  CUTLASS_DEVICE static void arrive(uint64_t* barrier) {
    cutlass::arch::ClusterBarrier::arrive(barrier);
  }

  CUTLASS_DEVICE static void arrive_cluster_zero(uint64_t* barrier) {
    cutlass::arch::ClusterBarrier::arrive(barrier, 0, true);
  }

  CUTLASS_DEVICE static void umma_arrive(uint64_t* barrier) {
    cutlass::arch::umma_arrive(barrier);
  }

  CUTLASS_DEVICE static bool try_wait(uint64_t* barrier, uint32_t phase) {
    return cutlass::arch::ClusterBarrier::try_wait(barrier, phase);
  }

  CUTLASS_DEVICE static cutlass::BarrierStatus try_wait_token(uint64_t* barrier,
                                                              uint32_t phase) {
    bool const barrier_status =
        cutlass::arch::ClusterBarrier::try_wait(barrier, phase);
    return barrier_status ? cutlass::BarrierStatus::WaitDone
                          : cutlass::BarrierStatus::WaitAgain;
  }

  CUTLASS_DEVICE static void wait(uint64_t* barrier,
                                  uint32_t phase,
                                  cutlass::BarrierStatus token,
                                  uint32_t wait_id = 0) {
    (void)wait_id;
    if (token == cutlass::BarrierStatus::WaitAgain) {
      cutlass::arch::ClusterBarrier::wait(barrier, phase);
    }
  }

  CUTLASS_DEVICE static void wait(uint64_t* barrier,
                                  uint32_t phase,
                                  uint32_t wait_id = 0) {
    (void)wait_id;
    cutlass::BarrierStatus const token = try_wait_token(barrier, phase);
    wait(barrier, phase, token);
  }
};

struct Sm100FmhaNamedBarrier {
  CUTLASS_DEVICE static void sync(uint32_t num_threads, uint32_t barrier_id) {
    cutlass::arch::NamedBarrier::sync(
        num_threads,
        static_cast<cutlass::arch::ReservedNamedBarriers>(barrier_id));
  }
};

struct Sm100FmhaTma {
  CUTLASS_DEVICE static void prefetch_tensormap(CUtensorMap const* desc) {
    cute::prefetch_tma_descriptor(desc);
  }

  CUTLASS_DEVICE static void load_4d_predicated(CUtensorMap const* desc,
                                                void* smem_ptr,
                                                uint64_t* barrier,
                                                int c0, int c1, int c2, int c3,
                                                uint32_t pred) {
    if (pred != 0) {
      cute::SM90_TMA_LOAD_4D::copy(
          desc, barrier,
          static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
          smem_ptr, c0, c1, c2, c3);
    }
  }

  CUTLASS_DEVICE static void load_5d_predicated(CUtensorMap const* desc,
                                                void* smem_ptr,
                                                uint64_t* barrier,
                                                int c0, int c1, int c2, int c3,
                                                int c4,
                                                uint32_t pred) {
    if (pred != 0) {
      cute::SM90_TMA_LOAD_5D::copy(
          desc, barrier,
          static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
          smem_ptr, c0, c1, c2, c3, c4);
    }
  }
};

}  // namespace cutlass::fmha::collective
