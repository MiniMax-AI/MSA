#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "cute/arch/util.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "sm100/common/indexer_gemm_params.hpp"
#include "sm100/common/indexer_gemm_traits.hpp"
#include "sm100/common/nvfp4_to_e4m3.cuh"

namespace minimax::msa_v1::indexer::decode::q8kv4 {

template <class Traits> struct IndexerGemmConfig {
  using Params = IndexerGemmParams;

  struct alignas(1024) SharedStorage {
    uint8_t pages[Traits::kMaxPagesPerCta][Traits::kPageBytes];
    uint64_t dequantized_ready[Traits::kDequantStages];
    uint64_t mma_done[Traits::kAccumulatorStages];
    uint32_t tmem_base;
    alignas(1024) uint8_t q_tile[Traits::kQueryColumns * Traits::kHeadDim];
    float query_maxima[Traits::kConsumerGroups][Traits::kConsumerWarpsPerGroup]
                      [Traits::kQueryColumns];
    uint64_t page_barriers[Traits::kMaxPagesPerCta];
    uint64_t page_consumed_barriers[Traits::kMaxPagesPerCta];
    int32_t next_page;
    int32_t worker_end;
    int32_t batch_idx;
    int32_t page_begin;
    int32_t page_count;
  };

  static constexpr int kNamedBarrierId = 1;

  CUTE_DEVICE static uint32_t smem_address(void const *pointer) {
    return cute::cast_smem_ptr_to_uint(pointer);
  }

  CUTE_DEVICE static void init_barrier(uint64_t *barrier, uint32_t count) {
    cutlass::arch::ClusterBarrier::init(barrier, count);
  }

  CUTE_DEVICE static void expect_transactions(uint64_t *barrier, uint32_t bytes) {
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(barrier, bytes);
  }

  CUTE_DEVICE static void arrive(uint64_t *barrier) {
    cutlass::arch::ClusterBarrier::arrive(barrier);
  }

  CUTE_DEVICE static void wait(uint64_t *barrier, uint32_t phase) {
    cutlass::arch::ClusterBarrier::wait(barrier, phase);
  }

#if MINIMAX_MSA_Q8KV4_INDEXER_HAS_QMUL4
  CUTE_DEVICE static uint32_t qmul4(uint16_t packed, uint32_t broadcast_scale) {
    return ::minimax::msa_v1::indexer::decode::q8kv4::sm100::common::nvfp4_to_e4m3x4(
        packed, broadcast_scale);
  }
#endif

#if !MINIMAX_MSA_Q8KV4_INDEXER_HAS_QMUL4
  CUTE_DEVICE static void dequantize_fp4x8(uint32_t &output0, uint32_t &output1, uint32_t &output2,
                                           uint32_t &output3, uint32_t packed,
                                           uint32_t broadcast_scale) {
    ::minimax::msa_v1::indexer::decode::q8kv4::sm100::common::nvfp4x8_to_f16x2x4(
        output0, output1, output2, output3, packed, static_cast<uint16_t>(broadcast_scale));
  }
#endif

  CUTE_DEVICE static int swizzled_page_offset(int token, int quarter) {
    return (token >> 1) * 128 + ((((token & 1) * 4 + quarter) ^ ((token >> 1) & 7)) * 16);
  }

  CUTE_DEVICE static uint32_t broadcast_scale_byte(uint2 value, int index) {
    uint32_t const word = index < 4 ? value.x : value.y;
    return __byte_perm(word, 0, (index & 3) * 0x1111);
  }

  CUTE_DEVICE static void issue_page_load(Params const &params, SharedStorage &storage, int slot,
                                          int physical_page) {
    uint64_t *barrier = storage.page_barriers + slot;
    expect_transactions(barrier, Traits::kPageBytes);
    uint64_t constexpr cache_hint_evict_first = 0x12f0000000000000ULL;
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::"
                 "complete_tx::bytes.L2::cache_hint "
                 "[%0], [%1, {%2, %3, %4}], [%5], %6;"
                 :
                 : "r"(smem_address(storage.pages[slot])),
                   "l"(reinterpret_cast<uint64_t>(&params.packed_k)), "r"(0), "r"(0),
                   "r"(physical_page), "r"(smem_address(barrier)), "l"(cache_hint_evict_first)
                 : "memory");
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::"
                 "complete_tx::bytes.L2::cache_hint "
                 "[%0], [%1, {%2, %3, %4}], [%5], %6;"
                 :
                 : "r"(smem_address(storage.pages[slot]) + Traits::kPackedKBytes),
                   "l"(reinterpret_cast<uint64_t>(&params.k_scale)), "r"(0), "r"(0),
                   "r"(physical_page), "r"(smem_address(barrier)), "l"(cache_hint_evict_first)
                 : "memory");
  }

  CUTE_DEVICE static void next_work(Params const &params, SharedStorage &storage) {
    int const begin = storage.next_page;
    storage.page_count = 0;
    if (begin >= storage.worker_end) {
      return;
    }
    while (params.scheduler_workspace_ptr[storage.batch_idx + 1] <= begin) {
      ++storage.batch_idx;
    }
    storage.page_begin = begin - params.scheduler_workspace_ptr[storage.batch_idx];
    int const end = min(storage.worker_end, params.scheduler_workspace_ptr[storage.batch_idx + 1]);
    // UMMA stores scores directly; a work span only ends at a request or worker boundary.
    storage.page_count = end - begin;
    storage.next_page = begin + storage.page_count;
  }

  CUTE_DEVICE static void initialize_work(Params const &params, SharedStorage &storage) {
    int const worker = int(blockIdx.x);
    int32_t const *boundaries = params.scheduler_workspace_ptr + params.batch + 1;
    storage.next_page = boundaries[worker];
    storage.worker_end = boundaries[worker + 1];
    int32_t const *start = params.scheduler_workspace_ptr + params.batch + params.sm_count + 2;
    storage.batch_idx = start[worker];
    next_work(params, storage);
  }
};

} // namespace minimax::msa_v1::indexer::decode::q8kv4
