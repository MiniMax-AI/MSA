#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "sm100/collective/sm100_indexer_gemm_config.hpp"

namespace minimax::msa_v1::indexer::decode::tp4_q8kv4 {

template <class Traits, int SchedulerCounterOffset>
struct IndexerGemmCollective : IndexerGemmConfig<Traits, SchedulerCounterOffset> {
  using Base = IndexerGemmConfig<Traits, SchedulerCounterOffset>;
  using Base::arrive;
  using Base::broadcast_scale_byte;
  using Base::claim_work;
  using Base::convert_e4m3x4_to_f16x2;
  using typename Base::Params;
  using typename Base::SharedStorage;
#if !MINIMAX_MSA_Q8KV4_INDEXER_HAS_QMUL4
  using Base::dequantize_fp4x8;
#endif
  using Base::fold_page_score;
  using Base::hmma_f16;
  using Base::init_barrier;
  using Base::issue_page_load;
  using Base::load_matrix_x4;
#if MINIMAX_MSA_Q8KV4_INDEXER_HAS_QMUL4
  using Base::qmul4;
#endif
  using Base::score_barrier;
  using Base::smem_address;
  using Base::swizzled_page_offset;
  using Base::wait;

  struct PageFragments {
    uint32_t matrix_a[4];
    uint32_t matrix_b[4];
    uint2 scale_a;
    uint2 scale_b;
  };

  CUTE_DEVICE static void run(Params const &params, SharedStorage &storage) {
    int const thread_idx = static_cast<int>(threadIdx.x);
    int const warp_idx = thread_idx / cutlass::NumThreadsPerWarp;
    int const lane_idx = thread_idx % cutlass::NumThreadsPerWarp;

    if (thread_idx == 0) {
      claim_work(params, storage);
    }
    if (thread_idx >= 1 && thread_idx <= Traits::kMaxPagesPerCta) {
      init_barrier(storage.page_barriers + thread_idx - 1, 1);
    }
    if (thread_idx > Traits::kMaxPagesPerCta && thread_idx <= 2 * Traits::kMaxPagesPerCta) {
      init_barrier(storage.page_consumed_barriers + thread_idx - Traits::kMaxPagesPerCta - 1,
                   Traits::kScoreWarps);
    }

    cutlass::arch::fence_barrier_init();
    __syncthreads();

    int page_ticket_base = 0;
    while (true) {
      if (storage.work_tile_id >= params.scheduler_workspace_ptr[params.batch]) {
        break;
      }
      int const batch_idx = storage.batch_idx;
      int const work_tile_idx = storage.work_tile_idx;
      int const kv_length = __ldg(params.kv_lengths_ptr + batch_idx);
      int const page_count =
          kv_length > 0 ? (kv_length + Traits::kPageTokens - 1) / Traits::kPageTokens : 0;
      int const scored_page_count = max(page_count - 1, 0);
      int const page_begin = work_tile_idx * Traits::kPagesPerWorkTile;
      int const page_end = min(page_begin + Traits::kPagesPerWorkTile, params.max_pages);
      int const local_pages = max(min(page_end, scored_page_count) - page_begin, 0);
      int32_t const *page_table =
          params.page_table_ptr + static_cast<size_t>(batch_idx) * params.max_pages;

      if (local_pages > 0 && thread_idx < (Traits::kQueryLength * Traits::kHeadDim) / 16) {
        size_t const query_offset =
            static_cast<size_t>(batch_idx) * Traits::kQueryLength * Traits::kHeadDim;
        reinterpret_cast<uint4 *>(storage.q_tile)[thread_idx] =
            reinterpret_cast<uint4 const *>(params.q_ptr + query_offset)[thread_idx];
      }

      int const initial_pages = min(local_pages, Traits::kMaxPagesPerCta);
      if (thread_idx < initial_pages) {
        int const page_ticket = page_ticket_base + thread_idx;
        int const slot = page_ticket & (Traits::kMaxPagesPerCta - 1);
        if (page_ticket >= Traits::kMaxPagesPerCta) {
          wait(storage.page_consumed_barriers + slot,
               static_cast<uint32_t>((page_ticket / Traits::kMaxPagesPerCta - 1) & 1));
        }
        int const physical_page = __ldg(page_table + page_begin + thread_idx);
        issue_page_load(params, storage, slot, physical_page);
      }
      __syncthreads();

      if (warp_idx == Traits::kScoreWarps) {
        for (int local_page = Traits::kMaxPagesPerCta; local_page < local_pages; ++local_page) {
          int const page_ticket = page_ticket_base + local_page;
          int const slot = page_ticket & (Traits::kMaxPagesPerCta - 1);
          wait(storage.page_consumed_barriers + slot,
               static_cast<uint32_t>((page_ticket / Traits::kMaxPagesPerCta - 1) & 1));
          if (cute::elect_one_sync()) {
            int const physical_page = __ldg(page_table + page_begin + local_page);
            issue_page_load(params, storage, slot, physical_page);
          }
        }
        if (cute::elect_one_sync()) {
          claim_work(params, storage);
        }
      } else if (local_pages > 0) {
        int const query_column = lane_idx >> 2;
        int const thread_group = lane_idx & 3;
        int const token_begin = warp_idx * 16;
        int const matrix_offset_a =
            swizzled_page_offset(token_begin + (lane_idx & 7), lane_idx >> 3);
        int const matrix_offset_b =
            swizzled_page_offset(token_begin + 8 + (lane_idx & 7), lane_idx >> 3);
        int const scale_offset_a =
            Traits::kPackedKBytes + (token_begin + (lane_idx >> 2)) * Traits::kScaleGroups;
        int const scale_offset_b = scale_offset_a + 8 * Traits::kScaleGroups;
#if MINIMAX_MSA_Q8KV4_INDEXER_HAS_QMUL4
        uint2 query_fragments_low[4];
        uint2 query_fragments_high[4];
        CUTLASS_PRAGMA_UNROLL
        for (int k_chunk = 0; k_chunk < 4; ++k_chunk) {
          int const query_offset =
              query_column * Traits::kHeadDim + 32 * k_chunk + 8 * thread_group;
          uint2 const query_fragment =
              *reinterpret_cast<uint2 const *>(storage.q_tile + query_offset);
          convert_e4m3x4_to_f16x2(query_fragment.x, query_fragments_low[k_chunk].x,
                                  query_fragments_high[k_chunk].x);
          convert_e4m3x4_to_f16x2(query_fragment.y, query_fragments_low[k_chunk].y,
                                  query_fragments_high[k_chunk].y);
        }
#else
        if (warp_idx == 0) {
          CUTLASS_PRAGMA_UNROLL
          for (int k_chunk = 0; k_chunk < 4; ++k_chunk) {
            int const query_offset =
                query_column * Traits::kHeadDim + 32 * k_chunk + 8 * thread_group;
            uint2 const query_fragment =
                *reinterpret_cast<uint2 const *>(storage.q_tile + query_offset);
            uint4 transformed;
            convert_e4m3x4_to_f16x2(query_fragment.x, transformed.x, transformed.z);
            convert_e4m3x4_to_f16x2(query_fragment.y, transformed.y, transformed.w);
            storage.query_fragments[k_chunk][lane_idx] = transformed;
          }
        }
        score_barrier();
#endif

        auto load_page = [&](int local_page, PageFragments &fragments) {
          int const page_ticket = page_ticket_base + local_page;
          int const slot = page_ticket & (Traits::kMaxPagesPerCta - 1);
          uint32_t const phase = static_cast<uint32_t>(page_ticket / Traits::kMaxPagesPerCta) & 1u;
          wait(storage.page_barriers + slot, phase);
          uint8_t const *packed = storage.pages[slot];
          load_matrix_x4(fragments.matrix_a, smem_address(packed + matrix_offset_a));
          load_matrix_x4(fragments.matrix_b, smem_address(packed + matrix_offset_b));
          fragments.scale_a = *reinterpret_cast<uint2 const *>(packed + scale_offset_a);
          fragments.scale_b = *reinterpret_cast<uint2 const *>(packed + scale_offset_b);
        };

        auto compute_page = [&](int local_page, PageFragments const &fragments) {
          float accumulator[4] = {0.0F, 0.0F, 0.0F, 0.0F};
          CUTLASS_PRAGMA_UNROLL
          for (int k_chunk = 0; k_chunk < 4; ++k_chunk) {
            uint32_t const scale_a =
                broadcast_scale_byte(fragments.scale_a, 2 * k_chunk + (thread_group >> 1));
            uint32_t const scale_b =
                broadcast_scale_byte(fragments.scale_b, 2 * k_chunk + (thread_group >> 1));
            uint32_t matrix_low[4];
            uint32_t matrix_high[4];
#if MINIMAX_MSA_Q8KV4_INDEXER_HAS_QMUL4
            uint32_t matrix[4];
            matrix[0] = qmul4(static_cast<uint16_t>(fragments.matrix_a[k_chunk]), scale_a);
            matrix[2] = qmul4(static_cast<uint16_t>(fragments.matrix_a[k_chunk] >> 16), scale_a);
            matrix[1] = qmul4(static_cast<uint16_t>(fragments.matrix_b[k_chunk]), scale_b);
            matrix[3] = qmul4(static_cast<uint16_t>(fragments.matrix_b[k_chunk] >> 16), scale_b);
            CUTLASS_PRAGMA_UNROLL
            for (int matrix_idx = 0; matrix_idx < 4; ++matrix_idx) {
              convert_e4m3x4_to_f16x2(matrix[matrix_idx], matrix_low[matrix_idx],
                                      matrix_high[matrix_idx]);
            }
#else
            dequantize_fp4x8(matrix_low[0], matrix_high[0], matrix_low[2], matrix_high[2],
                             fragments.matrix_a[k_chunk], scale_a);
            dequantize_fp4x8(matrix_low[1], matrix_high[1], matrix_low[3], matrix_high[3],
                             fragments.matrix_b[k_chunk], scale_b);
#endif
#if MINIMAX_MSA_Q8KV4_INDEXER_HAS_QMUL4
            hmma_f16(accumulator, matrix_low, query_fragments_low[k_chunk]);
            hmma_f16(accumulator, matrix_high, query_fragments_high[k_chunk]);
#else
            uint4 const query_fragment = storage.query_fragments[k_chunk][lane_idx];
            uint2 const query_low{query_fragment.x, query_fragment.y};
            uint2 const query_high{query_fragment.z, query_fragment.w};
            hmma_f16(accumulator, matrix_low, query_low);
            hmma_f16(accumulator, matrix_high, query_high);
#endif
          }

          float maximum0 = fmaxf(accumulator[0], accumulator[2]);
          float maximum1 = fmaxf(accumulator[1], accumulator[3]);
          CUTLASS_PRAGMA_UNROLL
          for (int offset = 4; offset <= 16; offset <<= 1) {
            maximum0 = fmaxf(maximum0, __shfl_xor_sync(0xffffffffu, maximum0, offset));
            maximum1 = fmaxf(maximum1, __shfl_xor_sync(0xffffffffu, maximum1, offset));
          }
          if (lane_idx < 4) {
            storage.page_partials[local_page][2 * thread_group][warp_idx] = maximum0;
            storage.page_partials[local_page][2 * thread_group + 1][warp_idx] = maximum1;
          }
        };

        if constexpr (Traits::kMaxPagesPerCta == 4) {
          PageFragments fragments;
          for (int local_page = 0; local_page < local_pages; ++local_page) {
            load_page(local_page, fragments);
            compute_page(local_page, fragments);
            if (cute::elect_one_sync()) {
              int const page_ticket = page_ticket_base + local_page;
              arrive(storage.page_consumed_barriers +
                     (page_ticket & (Traits::kMaxPagesPerCta - 1)));
            }
          }
        } else {
          PageFragments fragments_a;
          PageFragments fragments_b;
          int local_page = 0;
          for (; local_page + 1 < local_pages; local_page += 2) {
            load_page(local_page, fragments_a);
            load_page(local_page + 1, fragments_b);
            compute_page(local_page, fragments_a);
            compute_page(local_page + 1, fragments_b);
            if (cute::elect_one_sync()) {
              int const page_ticket_a = page_ticket_base + local_page;
              int const page_ticket_b = page_ticket_a + 1;
              arrive(storage.page_consumed_barriers +
                     (page_ticket_a & (Traits::kMaxPagesPerCta - 1)));
              arrive(storage.page_consumed_barriers +
                     (page_ticket_b & (Traits::kMaxPagesPerCta - 1)));
            }
          }
          if (local_page < local_pages) {
            load_page(local_page, fragments_a);
            compute_page(local_page, fragments_a);
            if (cute::elect_one_sync()) {
              int const page_ticket = page_ticket_base + local_page;
              arrive(storage.page_consumed_barriers +
                     (page_ticket & (Traits::kMaxPagesPerCta - 1)));
            }
          }
        }

        score_barrier();
        int const row = warp_idx;
        int const query_position = max(kv_length - Traits::kQueryLength + row, 0);
        int const query_local_page = query_position / Traits::kPageTokens;
        for (int tile_page = lane_idx; tile_page < local_pages;
             tile_page += cutlass::NumThreadsPerWarp) {
          int const logical_page = page_begin + tile_page;
          if (logical_page < query_local_page) {
            float const score = fold_page_score(storage, tile_page, row);
            size_t const output_offset =
                (static_cast<size_t>(batch_idx) * Traits::kQueryLength + row) * params.max_pages +
                logical_page;
            params.output_ptr[output_offset] = score;
          }
        }
      }

      page_ticket_base += local_pages;
      __syncthreads();
    }
  }
};

} // namespace minimax::msa_v1::indexer::decode::tp4_q8kv4
