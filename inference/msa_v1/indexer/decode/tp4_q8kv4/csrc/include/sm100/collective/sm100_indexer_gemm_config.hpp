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

namespace minimax::msa_v1::indexer::decode::tp4_q8kv4 {

template <class Traits, int SchedulerCounterOffset> struct IndexerGemmConfig {
  using Params = IndexerGemmParams;

  struct alignas(1024) SharedStorage {
    uint8_t pages[Traits::kMaxPagesPerCta][Traits::kPageBytes];
    alignas(16) float page_partials[Traits::kPartialPageCapacity][Traits::kPartialQueryCapacity]
                                   [Traits::kPartialWarpCapacity];
    uint8_t q_tile[Traits::kQueryLength * Traits::kHeadDim];
#if !MINIMAX_MSA_Q8KV4_INDEXER_HAS_QMUL4
    uint4 query_fragments[4][cutlass::NumThreadsPerWarp];
#endif
    uint64_t page_barriers[Traits::kMaxPagesPerCta];
    uint64_t page_consumed_barriers[Traits::kMaxPagesPerCta];
    int32_t work_tile_id;
    int32_t batch_idx;
    int32_t work_tile_idx;
  };

  static constexpr int kNamedBarrierId = 1;
  static constexpr int kScoreThreads = Traits::kScoreWarps * cutlass::NumThreadsPerWarp;

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

  CUTE_DEVICE static void score_barrier() {
    asm volatile("bar.sync %0, %1;" : : "r"(kNamedBarrierId), "r"(kScoreThreads) : "memory");
  }

#if MINIMAX_MSA_Q8KV4_INDEXER_HAS_QMUL4
  CUTE_DEVICE static uint32_t qmul4(uint16_t packed, uint32_t broadcast_scale) {
    return ::minimax::msa_v1::indexer::decode::tp4_q8kv4::sm100::common::nvfp4_to_e4m3x4(
        packed, broadcast_scale);
  }
#endif

  CUTE_DEVICE static void convert_e4m3x4_to_f16x2(uint32_t value, uint32_t &low, uint32_t &high) {
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(low) : "h"(static_cast<uint16_t>(value)));
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
                 : "=r"(high)
                 : "h"(static_cast<uint16_t>(value >> 16)));
  }

#if !MINIMAX_MSA_Q8KV4_INDEXER_HAS_QMUL4
  CUTE_DEVICE static void dequantize_fp4x8(uint32_t &output0, uint32_t &output1, uint32_t &output2,
                                           uint32_t &output3, uint32_t packed,
                                           uint32_t broadcast_scale) {
    ::minimax::msa_v1::indexer::decode::tp4_q8kv4::sm100::common::nvfp4x8_to_f16x2x4(
        output0, output1, output2, output3, packed, static_cast<uint16_t>(broadcast_scale));
  }
#endif

  CUTE_DEVICE static void hmma_f16(float (&accumulator)[4], uint32_t const (&matrix)[4],
                                   uint2 query) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]),
                   "+f"(accumulator[3])
                 : "r"(matrix[0]), "r"(matrix[1]), "r"(matrix[2]), "r"(matrix[3]), "r"(query.x),
                   "r"(query.y));
  }

  CUTE_DEVICE static void load_matrix_x4(uint32_t (&matrix)[4], uint32_t address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(matrix[0]), "=r"(matrix[1]), "=r"(matrix[2]), "=r"(matrix[3])
                 : "r"(address));
  }

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

  CUTE_DEVICE static void claim_work(Params const &params, SharedStorage &storage) {
    int const work_tile_id = atomicAdd(params.scheduler_workspace_ptr + SchedulerCounterOffset, 1);
    storage.work_tile_id = work_tile_id;
    int const work_count = params.scheduler_workspace_ptr[params.batch];
    if (work_tile_id >= work_count) {
      return;
    }
    int permuted_work_tile_id = work_tile_id;
    if (work_count > 1) {
      int const permutation_bits = 32 - __clz(work_count - 1);
      int const candidate =
          static_cast<int>(__brev(static_cast<uint32_t>(work_tile_id)) >> (32 - permutation_bits));
      if (candidate < work_count) {
        permuted_work_tile_id = candidate;
      }
    }
    int lower = 0;
    int upper = params.batch;
    while (lower + 1 < upper) {
      int const middle = (lower + upper) >> 1;
      if (params.scheduler_workspace_ptr[middle] <= permuted_work_tile_id) {
        lower = middle;
      } else {
        upper = middle;
      }
    }
    storage.batch_idx = lower;
    storage.work_tile_idx = permuted_work_tile_id - params.scheduler_workspace_ptr[lower];
  }

  CUTE_DEVICE static float fold_page_score(SharedStorage const &storage, int page, int row) {
    float4 const first = *reinterpret_cast<float4 const *>(&storage.page_partials[page][row][0]);
    float4 const second = *reinterpret_cast<float4 const *>(&storage.page_partials[page][row][4]);
    return fmaxf(fmaxf(fmaxf(first.x, first.y), fmaxf(first.z, first.w)),
                 fmaxf(fmaxf(second.x, second.y), fmaxf(second.z, second.w)));
  }
};

} // namespace minimax::msa_v1::indexer::decode::tp4_q8kv4
