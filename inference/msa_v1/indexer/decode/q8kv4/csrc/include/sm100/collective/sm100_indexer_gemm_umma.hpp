#pragma once

#include <cmath>

#include "cute/arch/copy_sm100.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/atom/mma_traits_sm100.hpp"
#include "cute/tensor.hpp"
#include "sm100/collective/sm100_indexer_gemm_config.hpp"

namespace minimax::msa_v1::indexer::decode::q8kv4 {

template <class Traits> struct IndexerGemmUmmaCollective : IndexerGemmConfig<Traits> {
  using Base = IndexerGemmConfig<Traits>;
  using typename Base::Params;
  using typename Base::SharedStorage;
  using Element = cutlass::float_e4m3_t;
  using Mma = cute::SM100_MMA_F8F6F4_TS<Element, Element, float, 128, Traits::kQueryColumns,
                                        cute::UMMA::Major::K, cute::UMMA::Major::K>;
  static constexpr int kDequantWarps = Traits::kDequantWarpsPerGroup;
  static constexpr int kDequantGroups = Traits::kDequantGroups;
  static constexpr int kConsumerWarps = Traits::kConsumerWarpsPerGroup;
  static constexpr int kConsumerGroups = Traits::kConsumerGroups;
  static constexpr int kDequantStages = Traits::kDequantStages;
  static_assert(kDequantStages % kDequantGroups == 0);
  // Each page worker observes every generation of its packed and TMEM slots.
  static_assert(Traits::kMaxPagesPerCta % Traits::kPageWorkerGroups == 0);
  static constexpr int kAccumulatorStages = Traits::kAccumulatorStages;
  static_assert(kDequantStages == kAccumulatorStages);
  static_assert(kDequantGroups == kConsumerGroups);
  static_assert(kDequantWarps == kConsumerWarps);
  static constexpr int kOperandColumns = Traits::kHeadDim / sizeof(uint32_t);
  static constexpr int kTmemColumnBudget = 512;
  static_assert(kAccumulatorStages == kConsumerGroups);
  static constexpr int kAccumulatorColumns = Traits::kQueryColumns * kAccumulatorStages;
  static constexpr int kRequiredTmemColumns =
      kAccumulatorColumns + kDequantStages * kOperandColumns;
  static_assert(kRequiredTmemColumns <= kTmemColumnBudget);
  static constexpr int kTmemColumns =
      kRequiredTmemColumns <= 128 ? 128 : (kRequiredTmemColumns <= 256 ? 256 : 512);

  CUTE_DEVICE static uint32_t operand_address(uint32_t tmem_base, int slot) {
    return tmem_base + kAccumulatorColumns + slot * kOperandColumns;
  }
  static constexpr int kMmaWarp = Traits::kPageWorkerGroups * Traits::kPageWorkerWarps;
  static constexpr int kTransportWarp = kMmaWarp + 1;

  CUTE_DEVICE static uint32_t accumulator_address(uint32_t tmem_base, int slot) {
    // Token-major pages occupy adjacent groups of query columns.
    return tmem_base + uint32_t(slot * Traits::kQueryColumns);
  }

  CUTE_DEVICE static int fp8_offset(int row, int column) {
    return row * Traits::kHeadDim + (column ^ ((row & 7) * 16));
  }

  CUTE_DEVICE static uint2 dequantize_word(uint32_t packed, uint32_t scale) {
#if MINIMAX_MSA_Q8KV4_INDEXER_HAS_QMUL4
    return {Base::qmul4(static_cast<uint16_t>(packed), scale),
            Base::qmul4(static_cast<uint16_t>(packed >> 16), scale)};
#else
    uint32_t half[4];
    Base::dequantize_fp4x8(half[0], half[1], half[2], half[3], packed, scale);
    uint16_t fp8[4];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(fp8[i]) : "r"(half[i]));
    }
    return {uint32_t(fp8[0]) | (uint32_t(fp8[1]) << 16),
            uint32_t(fp8[2]) | (uint32_t(fp8[3]) << 16)};
#endif
  }

  CUTE_DEVICE static void dequantize_page(SharedStorage &storage, int page_slot, int dequant_slot,
                                          int thread_idx, uint32_t tmem_base) {
    constexpr int kQuarters = Traits::kHeadDim / 32;
    int const token = thread_idx;
    uint8_t const *page = storage.pages[page_slot];
    uint4 packed_fragments[kQuarters];
    CUTLASS_PRAGMA_UNROLL
    for (int quarter = 0; quarter < kQuarters; ++quarter) {
      packed_fragments[quarter] =
          *reinterpret_cast<uint4 const *>(page + Base::swizzled_page_offset(token, quarter));
    }
    uint2 const scales = *reinterpret_cast<uint2 const *>(page + Traits::kPackedKBytes +
                                                          token * Traits::kScaleGroups);
    __syncwarp();
    if (cute::elect_one_sync()) {
      Base::arrive(storage.page_consumed_barriers + page_slot);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int quarter = 0; quarter < kQuarters; ++quarter) {
      uint4 const packed = packed_fragments[quarter];
      uint32_t const scale0 = Base::broadcast_scale_byte(scales, 2 * quarter);
      uint32_t const scale1 = Base::broadcast_scale_byte(scales, 2 * quarter + 1);
      uint2 const v0 = dequantize_word(packed.x, scale0);
      uint2 const v1 = dequantize_word(packed.y, scale0);
      uint2 const v2 = dequantize_word(packed.z, scale1);
      uint2 const v3 = dequantize_word(packed.w, scale1);
      // Every warp writes only its own 32 datapaths; four FP8 values share a column.
      uint32_t const address = operand_address(tmem_base, dequant_slot) + quarter * 8 +
                               (uint32_t((thread_idx / 32) * 32) << 16);
      cute::SM100_TMEM_STORE_32dp32b8x::copy(v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, v3.x, v3.y,
                                             address);
    }
  }

  CUTE_DEVICE static void issue_mma(SharedStorage &storage, int slot, int accumulator_slot,
                                    uint32_t tmem_base) {
    auto const q_layout =
        cute::tile_to_shape(cute::UMMA::Layout_K_SW128_Atom<Element>{},
                            cute::make_shape(cute::Int<Traits::kQueryColumns>{}, cute::_128{}));
    auto sQ = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element *>(storage.q_tile)),
                                q_layout);
    auto desc_q = cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(sQ);
    auto const descriptor =
        cute::UMMA::make_instr_desc<Element, Element, float, 128, Traits::kQueryColumns,
                                    cute::UMMA::Major::K, cute::UMMA::Major::K>();
    uint64_t const instruction = cute::UMMA::make_runtime_instr_desc<>(descriptor);
    // Both TMEM locations remain fixed while issuing the four K fragments.
    uint32_t const operand_base = operand_address(tmem_base, slot);
    uint32_t const accumulator_base = accumulator_address(tmem_base, accumulator_slot);
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Traits::kHeadDim / 32; ++k) {
      uint32_t const tmem_k = operand_base + k * 8;
      Mma::fma(tmem_k, desc_q, accumulator_base, k != 0, instruction);
      desc_q.start_address_ += 2;
    }
  }

  CUTE_DEVICE static void reduce_page(Params const &params, SharedStorage &storage, int batch_idx,
                                      int logical_page, int kv_length, int consumer_warp,
                                      int lane_idx, int accumulator_slot, int consumer_group,
                                      uint32_t tmem_base) {
    constexpr int kReductionColumns = 8;
    uint32_t const page_address =
        accumulator_address(tmem_base, accumulator_slot) + (uint32_t(consumer_warp * 32) << 16);
    CUTLASS_PRAGMA_UNROLL
    for (int group = 0; group < Traits::kQueryColumns / kReductionColumns; ++group) {
      if (group * kReductionColumns < params.query_length * Traits::kNumIndexHeads) {
        uint32_t const address = page_address + group * kReductionColumns;
        uint32_t values[kReductionColumns];
        cute::SM100_TMEM_LOAD_32dp32b8x::copy(address, values[0], values[1], values[2], values[3],
                                              values[4], values[5], values[6], values[7]);
        cutlass::arch::fence_view_async_tmem_load();
        CUTLASS_PRAGMA_UNROLL
        for (int column = 0; column < kReductionColumns; ++column) {
          float maximum = __uint_as_float(values[column]);
          // All lanes participate; only wholly invalid query groups are skipped.
          asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(maximum) : "f"(maximum));
          values[column] = __float_as_uint(maximum);
        }
        if (lane_idx == 0) {
          // Publish each contiguous eight-column partial with two vector stores.
          auto *partials = reinterpret_cast<uint4 *>(
              &storage.query_maxima[consumer_group][consumer_warp][group * kReductionColumns]);
          partials[0] = uint4{values[0], values[1], values[2], values[3]};
          partials[1] = uint4{values[4], values[5], values[6], values[7]};
        }
      }
    }
    // The next operand-ready notification also orders these accumulator reads.
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    // Each warp owns 32 TMEM rows; combine their page partials after publication.
    asm volatile("bar.sync %0, %1;"
                 :
                 : "r"(Base::kNamedBarrierId + consumer_group),
                   "r"(kConsumerWarps * cutlass::NumThreadsPerWarp)
                 : "memory");
    int const column = consumer_warp * cutlass::NumThreadsPerWarp + lane_idx;
    if (column < params.query_length * Traits::kNumIndexHeads) {
      int const query = column / Traits::kNumIndexHeads;
      int const head = column % Traits::kNumIndexHeads;
      int const page = logical_page;
      int const local_page = (kv_length - params.query_length + query) / Traits::kPageTokens;
      if (page < local_page) {
        size_t const row = (size_t(head) * params.batch + batch_idx) * params.query_length + query;
        float maximum = storage.query_maxima[consumer_group][0][column];
        CUTLASS_PRAGMA_UNROLL
        for (int partial = 1; partial < kConsumerWarps; ++partial) {
          maximum = fmaxf(maximum, storage.query_maxima[consumer_group][partial][column]);
        }
        params.output_ptr[row * params.max_pages + page] = maximum;
      }
    }
    // The next operand-ready waits for all four warps to finish these reads.
    // Its MMA completion therefore orders the next overwrite of this scratch.
  }

  CUTE_DEVICE static void run(Params const &params, SharedStorage &storage) {
    int const thread_idx = int(threadIdx.x);
    int const warp_idx = thread_idx / 32;
    int const lane_idx = thread_idx % 32;
    // Work metadata is independent of barrier setup and TMEM allocation.
    if (thread_idx == kTransportWarp * cutlass::NumThreadsPerWarp) {
      Base::initialize_work(params, storage);
      for (int slot = 0; slot < Traits::kMaxPagesPerCta; ++slot) {
        Base::init_barrier(storage.page_barriers + slot, 1);
      }
    }
    // Separate barrier setup from warp 0 TMEM allocation before the CTA fence.
    if (thread_idx == kMmaWarp * cutlass::NumThreadsPerWarp) {
      for (int slot = 0; slot < kAccumulatorStages; ++slot) {
        Base::init_barrier(storage.mma_done + slot, 1);
      }
      for (int slot = 0; slot < Traits::kMaxPagesPerCta; ++slot) {
        Base::init_barrier(storage.page_consumed_barriers + slot, kDequantWarps);
      }
      for (int slot = 0; slot < kDequantStages; ++slot) {
        Base::init_barrier(storage.dequantized_ready + slot, kDequantWarps);
      }
    }
    if (warp_idx == 0) {
      cute::TMEM::Allocator1Sm allocator;
      allocator.allocate(kTmemColumns, &storage.tmem_base);
      allocator.release_allocation_lock();
    }
    cutlass::arch::fence_barrier_init();
    if (warp_idx == kTransportWarp) {
      // Publish this warp's metadata and barrier initialization before first-page TMA.
      __syncwarp();
      if (lane_idx < min(storage.page_count, Traits::kMaxPagesPerCta)) {
        int32_t const *table = params.page_table_ptr + size_t(storage.batch_idx) * params.max_pages;
        Base::issue_page_load(params, storage, lane_idx,
                              __ldg(table + storage.page_begin + lane_idx));
      }
    }
    __syncthreads();
    // Allocation is published above and remains unchanged until the final deallocation.
    uint32_t const tmem_base = storage.tmem_base;
    int ticket_base = 0;
    bool first_span = true;
    while (storage.page_count > 0) {
      int const batch_idx = storage.batch_idx;
      int const page_begin = storage.page_begin;
      int const kv_length = __ldg(params.kv_lengths_ptr + batch_idx);
      int const page_count = storage.page_count;
      int32_t const *table = params.page_table_ptr + size_t(batch_idx) * params.max_pages;
      for (int vector = thread_idx; vector < Traits::kQueryColumns * Traits::kHeadDim / 16;
           vector += Traits::kThreads) {
        int const row = vector / 8;
        int const column = (vector % 8) * 16;
        size_t const base =
            size_t(batch_idx) * params.query_length * Traits::kNumIndexHeads * Traits::kHeadDim;
        *reinterpret_cast<uint4 *>(storage.q_tile + fp8_offset(row, column)) =
            row < params.query_length * Traits::kNumIndexHeads
                ? reinterpret_cast<uint4 const *>(params.q_ptr + base)[vector]
                : uint4{0, 0, 0, 0};
      }
      cutlass::arch::fence_view_async_shared();
      // Start page transport independently of the query loads on the first warps.
      if (!first_span && warp_idx == kTransportWarp &&
          lane_idx < min(page_count, Traits::kMaxPagesPerCta)) {
        int const ticket = ticket_base + lane_idx;
        int const slot = ticket % Traits::kMaxPagesPerCta;
        if (ticket >= Traits::kMaxPagesPerCta) {
          Base::wait(storage.page_consumed_barriers + slot,
                     ((ticket / Traits::kMaxPagesPerCta) - 1) & 1);
        }
        Base::issue_page_load(params, storage, slot, __ldg(table + page_begin + lane_idx));
      }
      __syncthreads();
      if (warp_idx == kTransportWarp) {
        for (int page = Traits::kMaxPagesPerCta; page < page_count; ++page) {
          int const ticket = ticket_base + page;
          int const slot = ticket % Traits::kMaxPagesPerCta;
          Base::wait(storage.page_consumed_barriers + slot,
                     ((ticket / Traits::kMaxPagesPerCta) - 1) & 1);
          if (cute::elect_one_sync()) {
            Base::issue_page_load(params, storage, slot, __ldg(table + page_begin + page));
          }
        }
        if (cute::elect_one_sync()) {
          Base::next_work(params, storage);
        }
      } else if (warp_idx < kDequantGroups * kDequantWarps) {
        int const dequant_group = warp_idx / kDequantWarps;
        int const dequant_thread = thread_idx % (kDequantWarps * cutlass::NumThreadsPerWarp);
        int const first_page =
            (dequant_group + kDequantGroups - ticket_base % kDequantGroups) % kDequantGroups;
        for (int page = first_page; page < page_count; page += kDequantGroups) {
          int const ticket = ticket_base + page;
          int const slot = ticket % Traits::kMaxPagesPerCta;
          int const dequant_slot = ticket % kDequantStages;
          Base::wait(storage.page_barriers + slot, (ticket / Traits::kMaxPagesPerCta) & 1);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          dequantize_page(storage, slot, dequant_slot, dequant_thread, tmem_base);
          cutlass::arch::fence_view_async_tmem_store();
          asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
          __syncwarp();
          if (cute::elect_one_sync()) {
            Base::arrive(storage.dequantized_ready + dequant_slot);
          }
          // This worker finishes both roles before reusing its operand/accumulator slot.
          int const accumulator_slot = ticket % kAccumulatorStages;
          Base::wait(storage.mma_done + accumulator_slot, (ticket / kAccumulatorStages) & 1);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          reduce_page(params, storage, batch_idx, page_begin + page, kv_length,
                      warp_idx % kConsumerWarps, lane_idx, accumulator_slot, dequant_group,
                      tmem_base);
        }
      } else if (warp_idx == kMmaWarp) {
        for (int page = 0; page < page_count; ++page) {
          int const ticket = ticket_base + page;
          int const accumulator_slot = ticket % kAccumulatorStages;
          int const slot = ticket % kDequantStages;
          Base::wait(storage.dequantized_ready + slot, (ticket / kDequantStages) & 1);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          issue_mma(storage, slot, accumulator_slot, tmem_base);
          // Publish only after every accumulator column is ready.
          cutlass::arch::umma_arrive(storage.mma_done + accumulator_slot);
        }
      }
      ticket_base += page_count;
      first_span = false;
      __syncthreads();
    }
    __syncthreads();
    if (warp_idx == 0) {
      cute::TMEM::Allocator1Sm allocator;
      allocator.free(tmem_base, kTmemColumns);
    }
  }
};

} // namespace minimax::msa_v1::indexer::decode::q8kv4
