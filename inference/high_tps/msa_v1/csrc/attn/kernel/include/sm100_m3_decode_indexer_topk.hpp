#pragma once

#include "attn/kernel/include/sm100_m3_decode_indexer_config.hpp"

namespace cutlass::m3_decode::collective {

template <class Traits> struct Sm100M3DecodeIndexerTopK : Sm100M3DecodeIndexerConfig<Traits> {
  using Base = Sm100M3DecodeIndexerConfig<Traits>;
  using Base::cta_candidate_scratch;
  using Base::exact_key;
  using Base::exact_key_page;
  using Base::kCtaCandidateCount;
  using Base::kCtaCandidateItemsPerThread;
  using Base::kFastTopKItemsPerThread;
  using Base::kFastTopKMaxPages;
  using Base::kFastTopKPageBits;
  using Base::kFastTopKPageMask;
  using Base::kFastTopKScoreMaximum;
  using Base::kMaximumTopKWarps;
  using Base::kMaxPages;
  using Base::kPageTokens;
  using Base::kRankedTopK;
  using Base::kRows;
  using Base::kTopK;
  using Base::kTopKBarrierId;
  using Base::kTopKCandidatesPerLane;
  using Base::kTopKLocalCandidates;
  using Base::kTopKThreads;
  using Base::kTopKWarps;
  using Base::ordered_float_bits;
  using Base::raw_score_key;
  using Base::raw_score_key_page;
  using Base::raw_score_key_score;
  using Base::score_row;
  using Base::sort_descending;
  using Base::sort_descending_by_raw_score;
  using Base::sort_descending_by_score;
  using Base::sort_descending_nine_raw_score;
  using Base::store_remote_int4;
  using Base::wait;
  using Base::warp_max;
  using Base::warp_reduce_max;
  using Base::warp_reduce_min;
  using typename Base::IndexerStorage;
  using typename Base::Params;
  using typename Base::Storage;
  CUTE_DEVICE static void select_topk_serial(Storage &storage, Params const &params, int row,
                                             int kv_length) {
    int const visible_pages = ((kv_length - kRows + row) / kPageTokens) + 1;
    if (visible_pages <= kTopK) {
      CUTLASS_PRAGMA_UNROLL
      for (int selected = 0; selected < kTopK; ++selected) {
        storage.indexer_topk[selected] = selected < visible_pages ? selected : -1;
      }
    } else {
      float selected_scores[kTopK - 1];
      int selected_pages[kTopK - 1];
      CUTLASS_PRAGMA_UNROLL
      for (int selected = 0; selected < kTopK - 1; ++selected) {
        selected_scores[selected] = -1.0e30f;
        selected_pages[selected] = 0x7fffffff;
      }
      float const *row_scores = score_row(params, row, kv_length);
      int const ranked_pages = visible_pages - 1;
      for (int page = 0; page < ranked_pages; ++page) {
        float candidate_score = __ldcg(row_scores + page);
        int candidate_page = page;
        CUTLASS_PRAGMA_UNROLL
        for (int selected = 0; selected < kTopK - 1; ++selected) {
          bool const swap = candidate_score > selected_scores[selected] ||
                            (candidate_score == selected_scores[selected] &&
                             candidate_page < selected_pages[selected]);
          if (swap) {
            float const score = selected_scores[selected];
            int const page_id = selected_pages[selected];
            selected_scores[selected] = candidate_score;
            selected_pages[selected] = candidate_page;
            candidate_score = score;
            candidate_page = page_id;
          }
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int selected = 0; selected < kTopK - 1; ++selected) {
        storage.indexer_topk[selected] = selected_pages[selected];
      }
      storage.indexer_topk[kTopK - 1] = visible_pages - 1;
    }
  }

  CUTE_DEVICE static void select_topk_fast(IndexerStorage &indexer, Storage &storage,
                                           Params const &params, int row, int kv_length,
                                           int thread_idx, int warp_idx, int lane_idx) {
    int const visible_pages = ((kv_length - kRows + row) / kPageTokens) + 1;
    int const ranked_pages = visible_pages - 1;
    uint32_t keys[kFastTopKItemsPerThread] = {};
    // Retain FP32 scores for the exact boundary fallback.
    float exact_scores[kFastTopKItemsPerThread] = {};
    float const *row_scores = score_row(params, row, kv_length);
    if (warp_idx < kTopKWarps) {
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < kFastTopKItemsPerThread; ++item) {
        int const page = thread_idx + item * kTopKThreads;
        if (page < ranked_pages) {
          exact_scores[item] = __ldcg(row_scores + page);
          uint32_t const score = ordered_float_bits(exact_scores[item]) >> kFastTopKPageBits;
          keys[item] =
              score << kFastTopKPageBits | (static_cast<uint32_t>(~page) & kFastTopKPageMask);
        }
      }
      sort_descending(keys);
      CUTLASS_PRAGMA_UNROLL
      for (int selected = 0; selected < kTopKLocalCandidates; ++selected) {
        uint32_t const best = warp_max(keys[0]);
        if (lane_idx == 0) {
          indexer.tail.quantized.candidates[warp_idx][selected] = best;
        }
        bool const winner = keys[0] == best;
        CUTLASS_PRAGMA_UNROLL
        for (int item = 0; item < kFastTopKItemsPerThread - 1; ++item) {
          keys[item] = winner ? keys[item + 1] : keys[item];
        }
        keys[kFastTopKItemsPerThread - 1] = winner ? 0u : keys[kFastTopKItemsPerThread - 1];
      }
      Sm100M3DecodeNamedBarrier::sync(kTopKThreads, kTopKBarrierId);
    }

    if (warp_idx == 0) {
      uint32_t candidates[kTopKCandidatesPerLane];
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < kTopKCandidatesPerLane; ++item) {
        int const index = lane_idx + item * cutlass::NumThreadsPerWarp;
        auto const *flat_candidates =
            reinterpret_cast<uint32_t const *>(indexer.tail.quantized.candidates);
        candidates[item] = flat_candidates[index];
      }
      sort_descending(candidates);
      uint32_t boundary_score = 0u;
      CUTLASS_PRAGMA_UNROLL
      for (int selected = 0; selected < kTopK - 1; ++selected) {
        uint32_t const best = warp_max(candidates[0]);
        boundary_score = best >> kFastTopKPageBits;
        if (lane_idx == 0) {
          storage.indexer_topk[selected] = static_cast<int>((~best) & kFastTopKPageMask);
        }
        bool const winner = candidates[0] == best;
        CUTLASS_PRAGMA_UNROLL
        for (int item = 0; item < kTopKCandidatesPerLane - 1; ++item) {
          candidates[item] = winner ? candidates[item + 1] : candidates[item];
        }
        candidates[kTopKCandidatesPerLane - 1] =
            winner ? 0u : candidates[kTopKCandidatesPerLane - 1];
      }
      if (lane_idx == 0) {
        storage.indexer_topk[kTopK - 1] = visible_pages - 1;
      }
      uint32_t const next_best = warp_max(candidates[0]);
      bool const collision = (next_best >> kFastTopKPageBits) == boundary_score;
      bool strict = false;
      if (collision && lane_idx < kRankedTopK) {
        int const page = storage.indexer_topk[lane_idx];
        strict =
            (ordered_float_bits(__ldcg(row_scores + page)) >> kFastTopKPageBits) > boundary_score;
      }
      uint32_t const strict_mask = __ballot_sync(0xffffffffu, strict);
      if (lane_idx == 0) {
        indexer.tail.quantized.exact_fallback = static_cast<uint32_t>(collision);
        if (collision) {
          indexer.tail.quantized.boundary_score = boundary_score;
          indexer.tail.quantized.strict_count = __popc(strict_mask);
        }
      }
      __syncwarp();
    }
    if (warp_idx < kTopKWarps) {
      Sm100M3DecodeNamedBarrier::sync(kTopKThreads, kTopKBarrierId);
      if (indexer.tail.quantized.exact_fallback != 0) {
        select_topk_boundary_exact(indexer, storage, params, row, kv_length, thread_idx, warp_idx,
                                   lane_idx, exact_scores);
      } else if (warp_idx == 0 && params.kv_block_indexes_ptr != nullptr && lane_idx < kTopK) {
        const_cast<int *>(params.kv_block_indexes_ptr)[row * kTopK + lane_idx] =
            storage.indexer_topk[lane_idx];
      }
    }
  }

  CUTE_DEVICE static void select_topk_chunk(IndexerStorage &indexer, Params const &params, int row,
                                            int kv_length, int chunk_index, int page_begin,
                                            int page_end, int thread_idx, int warp_idx,
                                            int lane_idx) {
    float values[kFastTopKItemsPerThread];
    float local_maximum = -1.0e30f;
    float local_minimum = 1.0e30f;
    float const *row_scores = score_row(params, row, kv_length);
    CUTLASS_PRAGMA_UNROLL
    for (int item = 0; item < kFastTopKItemsPerThread; ++item) {
      values[item] = -1.0e30f;
    }
    if (warp_idx < kTopKWarps) {
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < kFastTopKItemsPerThread; ++item) {
        int const local_page = thread_idx + item * kTopKThreads;
        int const page = page_begin + local_page;
        if (page < page_end) {
          float const value = __ldcg(row_scores + page);
          values[item] = value;
          local_maximum = fmaxf(local_maximum, value);
          local_minimum = fminf(local_minimum, value);
        }
      }
      local_maximum = warp_reduce_max(local_maximum);
      local_minimum = warp_reduce_min(local_minimum);
      if (lane_idx == 0) {
        indexer.tail.quantized.extrema[0][warp_idx] = local_maximum;
        indexer.tail.quantized.extrema[1][warp_idx] = local_minimum;
      }
      Sm100M3DecodeNamedBarrier::sync(kTopKThreads, kTopKBarrierId);
    }

    if (warp_idx < kTopKWarps) {
      float row_maximum = indexer.tail.quantized.extrema[0][0];
      float row_minimum = indexer.tail.quantized.extrema[1][0];
      CUTLASS_PRAGMA_UNROLL
      for (int warp = 1; warp < kTopKWarps; ++warp) {
        row_maximum = fmaxf(row_maximum, indexer.tail.quantized.extrema[0][warp]);
        row_minimum = fminf(row_minimum, indexer.tail.quantized.extrema[1][warp]);
      }
      float const scale =
          row_maximum > row_minimum
              ? static_cast<float>(kFastTopKScoreMaximum - 1) / (row_maximum - row_minimum)
              : 0.0f;
      uint32_t keys[kFastTopKItemsPerThread] = {};
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < kFastTopKItemsPerThread; ++item) {
        int const local_page = thread_idx + item * kTopKThreads;
        int const page = page_begin + local_page;
        if (page < page_end) {
          uint32_t quantized =
              1u + static_cast<uint32_t>((values[item] - row_minimum) * scale + 0.5f);
          quantized = quantized > kFastTopKScoreMaximum ? kFastTopKScoreMaximum : quantized;
          keys[item] = quantized << kFastTopKPageBits |
                       (static_cast<uint32_t>(~local_page) & kFastTopKPageMask);
        }
      }
      sort_descending(keys);
      CUTLASS_PRAGMA_UNROLL
      for (int selected = 0; selected < kTopKLocalCandidates; ++selected) {
        uint32_t const best = warp_max(keys[0]);
        if (lane_idx == 0) {
          indexer.tail.quantized.candidates[warp_idx][selected] = best;
        }
        bool const winner = keys[0] == best;
        CUTLASS_PRAGMA_UNROLL
        for (int item = 0; item < kFastTopKItemsPerThread - 1; ++item) {
          keys[item] = winner ? keys[item + 1] : keys[item];
        }
        keys[kFastTopKItemsPerThread - 1] = winner ? 0u : keys[kFastTopKItemsPerThread - 1];
      }
      Sm100M3DecodeNamedBarrier::sync(kTopKThreads, kTopKBarrierId);
    }

    if (warp_idx == 0) {
      uint32_t candidates[kTopKCandidatesPerLane];
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < kTopKCandidatesPerLane; ++item) {
        int const index = lane_idx + item * cutlass::NumThreadsPerWarp;
        auto const *flat_candidates =
            reinterpret_cast<uint32_t const *>(indexer.tail.quantized.candidates);
        candidates[item] = flat_candidates[index];
      }
      sort_descending(candidates);
      uint32_t previous_score = 0xffffffffu;
      bool collision = false;
      CUTLASS_PRAGMA_UNROLL
      for (int selected = 0; selected < kTopK; ++selected) {
        uint32_t const best = warp_max(candidates[0]);
        uint32_t const score = best >> kFastTopKPageBits;
        collision = collision || (selected != 0 && score == previous_score);
        previous_score = score;
        if (lane_idx == 0) {
          int const local_page = static_cast<int>((~best) & kFastTopKPageMask);
          int const page = page_begin + local_page;
          bool const valid = best != 0 && page < page_end;
          indexer.tail.chunks.candidate_scores[chunk_index][selected] =
              valid ? __ldcg(row_scores + page) : -1.0e30f;
          indexer.tail.chunks.candidate_pages[chunk_index][selected] = valid ? page : kMaxPages;
        }
        bool const winner = candidates[0] == best;
        CUTLASS_PRAGMA_UNROLL
        for (int item = 0; item < kTopKCandidatesPerLane - 1; ++item) {
          candidates[item] = winner ? candidates[item + 1] : candidates[item];
        }
        candidates[kTopKCandidatesPerLane - 1] =
            winner ? 0u : candidates[kTopKCandidatesPerLane - 1];
      }
      uint32_t const next_best = warp_max(candidates[0]);
      collision = collision || (next_best >> kFastTopKPageBits) == previous_score;
      if (lane_idx == 0 && collision) {
        float selected_scores[kTopK];
        int selected_pages[kTopK];
        CUTLASS_PRAGMA_UNROLL
        for (int selected = 0; selected < kTopK; ++selected) {
          selected_scores[selected] = -1.0e30f;
          selected_pages[selected] = kMaxPages;
        }
        for (int page = page_begin; page < page_end; ++page) {
          float candidate_score = __ldcg(row_scores + page);
          int candidate_page = page;
          CUTLASS_PRAGMA_UNROLL
          for (int selected = 0; selected < kTopK; ++selected) {
            bool const swap = candidate_score > selected_scores[selected] ||
                              (candidate_score == selected_scores[selected] &&
                               candidate_page < selected_pages[selected]);
            if (swap) {
              float const previous_value = selected_scores[selected];
              int const previous_page = selected_pages[selected];
              selected_scores[selected] = candidate_score;
              selected_pages[selected] = candidate_page;
              candidate_score = previous_value;
              candidate_page = previous_page;
            }
          }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int selected = 0; selected < kTopK; ++selected) {
          indexer.tail.chunks.candidate_scores[chunk_index][selected] = selected_scores[selected];
          indexer.tail.chunks.candidate_pages[chunk_index][selected] = selected_pages[selected];
        }
      }
      __syncwarp();
    }
    if (warp_idx < kTopKWarps) {
      Sm100M3DecodeNamedBarrier::sync(kTopKThreads, kTopKBarrierId);
    }
  }

  template <int ItemsPerThread, int SelectionWarps = kTopKWarps>
  CUTE_DEVICE static void select_topk_exact_two_level(IndexerStorage &indexer, Storage &storage,
                                                      Params const &params, int row, int kv_length,
                                                      int thread_idx, int warp_idx, int lane_idx) {
    static_assert(ItemsPerThread > 0);
    static_assert(SelectionWarps > 0 && SelectionWarps <= kMaximumTopKWarps);
    constexpr int kSelectionThreads = SelectionWarps * cutlass::NumThreadsPerWarp;
    static_assert(ItemsPerThread * kSelectionThreads <= kMaxPages);
    int const visible_pages = ((kv_length - kRows + row) / kPageTokens) + 1;
    int const ranked_pages = visible_pages - 1;
    float const *row_scores = score_row(params, row, kv_length);

    if (warp_idx < SelectionWarps) {
      uint64_t keys[ItemsPerThread];
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < ItemsPerThread; ++item) {
        int const page = thread_idx + item * kSelectionThreads;
        keys[item] = page < ranked_pages ? exact_key(__ldcg(row_scores + page), page) : 0u;
      }
      sort_descending_by_score(keys);
      int selected_count = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int round = 0; round < kRankedTopK; ++round) {
        uint32_t const score = static_cast<uint32_t>(keys[0] >> 32);
        uint32_t const best_score = warp_max(score);
        uint32_t const winners = __ballot_sync(0xffffffffu, score == best_score);
        uint32_t lane_mask_lt;
        asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
        int const winner_rank = __popc(winners & lane_mask_lt);
        bool const winner = score == best_score;
        int const output = selected_count + winner_rank;
        if (winner && output < kRankedTopK) {
          indexer.tail.chunks.candidates[warp_idx][output] = keys[0];
        }
        CUTLASS_PRAGMA_UNROLL
        for (int item = 0; item < ItemsPerThread - 1; ++item) {
          keys[item] = winner ? keys[item + 1] : keys[item];
        }
        keys[ItemsPerThread - 1] = winner ? 0u : keys[ItemsPerThread - 1];
        int const remaining = kRankedTopK - selected_count;
        int const winner_count = __popc(winners);
        selected_count += winner_count < remaining ? winner_count : remaining;
      }
      if constexpr (SelectionWarps == kMaximumTopKWarps) {
        __syncthreads();
      } else {
        Sm100M3DecodeNamedBarrier::sync(kSelectionThreads, kTopKBarrierId);
      }
    }

    if constexpr (SelectionWarps == kMaximumTopKWarps) {
      constexpr int kIntermediateWarps = 4;
      constexpr int kSourceWarpsPerIntermediate = SelectionWarps / kIntermediateWarps;
      constexpr int kIntermediateCandidatesPerLane =
          (kSourceWarpsPerIntermediate * kRankedTopK + cutlass::NumThreadsPerWarp - 1) /
          cutlass::NumThreadsPerWarp;
      static_assert(kIntermediateCandidatesPerLane == 2);
      if (warp_idx < kIntermediateWarps) {
        uint64_t candidates[kIntermediateCandidatesPerLane];
        CUTLASS_PRAGMA_UNROLL
        for (int item = 0; item < kIntermediateCandidatesPerLane; ++item) {
          int const index = lane_idx + item * cutlass::NumThreadsPerWarp;
          int const source = warp_idx * kSourceWarpsPerIntermediate * kRankedTopK + index;
          auto const *flat_candidates =
              reinterpret_cast<uint64_t const *>(indexer.tail.chunks.candidates);
          candidates[item] =
              index < kSourceWarpsPerIntermediate * kRankedTopK ? flat_candidates[source] : 0u;
        }
        sort_descending_by_score(candidates);
        __syncwarp();
        int selected_count = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int round = 0; round < kRankedTopK; ++round) {
          uint32_t const score = static_cast<uint32_t>(candidates[0] >> 32);
          uint32_t const best_score = warp_max(score);
          uint32_t const winners = __ballot_sync(0xffffffffu, score == best_score);
          uint32_t lane_mask_lt;
          asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
          int const winner_rank = __popc(winners & lane_mask_lt);
          bool const winner = score == best_score;
          int const output = selected_count + winner_rank;
          if (winner && output < kRankedTopK) {
            indexer.tail.chunks.candidates[warp_idx][output] = candidates[0];
          }
          candidates[0] = winner ? candidates[1] : candidates[0];
          candidates[1] = winner ? 0u : candidates[1];
          int const remaining = kRankedTopK - selected_count;
          int const winner_count = __popc(winners);
          selected_count += winner_count < remaining ? winner_count : remaining;
        }
        Sm100M3DecodeNamedBarrier::sync(kIntermediateWarps * cutlass::NumThreadsPerWarp,
                                        kTopKBarrierId);
      }
    }

    if (warp_idx == 0) {
      constexpr int kFinalCandidateWarps = SelectionWarps == kMaximumTopKWarps ? 4 : SelectionWarps;
      constexpr int kCandidatesPerLane =
          (kFinalCandidateWarps * kRankedTopK + cutlass::NumThreadsPerWarp - 1) /
          cutlass::NumThreadsPerWarp;
      uint64_t candidates[kCandidatesPerLane];
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < kCandidatesPerLane; ++item) {
        int const index = lane_idx + item * cutlass::NumThreadsPerWarp;
        auto const *flat_candidates =
            reinterpret_cast<uint64_t const *>(indexer.tail.chunks.candidates);
        candidates[item] = index < kFinalCandidateWarps * kRankedTopK ? flat_candidates[index] : 0u;
      }
      sort_descending_by_score(candidates);
      int selected_count = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int round = 0; round < kRankedTopK; ++round) {
        uint32_t const score = static_cast<uint32_t>(candidates[0] >> 32);
        uint32_t const best_score = warp_max(score);
        uint32_t const winners = __ballot_sync(0xffffffffu, score == best_score);
        uint32_t lane_mask_lt;
        asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
        int const winner_rank = __popc(winners & lane_mask_lt);
        bool const winner = score == best_score;
        int const output = selected_count + winner_rank;
        if (winner && output < kRankedTopK) {
          storage.indexer_topk[output] = exact_key_page(candidates[0]);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int item = 0; item < kCandidatesPerLane - 1; ++item) {
          candidates[item] = winner ? candidates[item + 1] : candidates[item];
        }
        candidates[kCandidatesPerLane - 1] = winner ? 0u : candidates[kCandidatesPerLane - 1];
        int const remaining = kRankedTopK - selected_count;
        int const winner_count = __popc(winners);
        selected_count += winner_count < remaining ? winner_count : remaining;
      }
      if (lane_idx == kRankedTopK) {
        storage.indexer_topk[kTopK - 1] = visible_pages - 1;
      }
      __syncwarp();
      if (params.kv_block_indexes_ptr != nullptr && lane_idx < kTopK) {
        const_cast<int *>(params.kv_block_indexes_ptr)[row * kTopK + lane_idx] =
            storage.indexer_topk[lane_idx];
      }
    }
  }

  template <int ItemsPerThread, int SelectionWarps>
  CUTE_DEVICE static void
  select_topk_exact_two_level_raw_score(IndexerStorage &indexer, Storage &storage,
                                        Params const &params, int row, int kv_length,
                                        int thread_idx, int warp_idx, int lane_idx) {
    static_assert(ItemsPerThread > 0);
    static_assert(SelectionWarps == kMaximumTopKWarps);
    constexpr int kSelectionThreads = SelectionWarps * cutlass::NumThreadsPerWarp;
    static_assert(ItemsPerThread * kSelectionThreads <= kMaxPages);
    int const visible_pages = ((kv_length - kRows + row) / kPageTokens) + 1;
    int const ranked_pages = visible_pages - 1;
    float const *row_scores = score_row(params, row, kv_length);

    if (warp_idx < SelectionWarps) {
      uint64_t keys[ItemsPerThread];
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < ItemsPerThread; ++item) {
        int const page = thread_idx + item * kSelectionThreads;
        keys[item] = page < ranked_pages ? raw_score_key(__ldcg(row_scores + page), page)
                                         : raw_score_key(-3.402823466e+38F, -1);
      }
      sort_descending_by_raw_score(keys);
      CUTLASS_PRAGMA_UNROLL
      for (int round = 0; round < kRankedTopK; ++round) {
        float const score = raw_score_key_score(keys[0]);
        float const best_score = warp_reduce_max(score);
        int const winner_lane = __reduce_min_sync(
            0xffffffffu, score == best_score ? lane_idx : cutlass::NumThreadsPerWarp);
        bool const winner = lane_idx == winner_lane;
        if (winner) {
          indexer.tail.chunks.candidates[warp_idx][round] = keys[0];
        }
        CUTLASS_PRAGMA_UNROLL
        for (int item = 0; item < ItemsPerThread - 1; ++item) {
          keys[item] = winner ? keys[item + 1] : keys[item];
        }
        keys[ItemsPerThread - 1] =
            winner ? raw_score_key(-3.402823466e+38F, -1) : keys[ItemsPerThread - 1];
      }
      __syncthreads();
    }

    constexpr int kIntermediateWarps = 4;
    constexpr int kSourceWarpsPerIntermediate = SelectionWarps / kIntermediateWarps;
    constexpr int kIntermediateCandidatesPerLane =
        (kSourceWarpsPerIntermediate * kRankedTopK + cutlass::NumThreadsPerWarp - 1) /
        cutlass::NumThreadsPerWarp;
    static_assert(kIntermediateCandidatesPerLane == 2);
    if (warp_idx < kIntermediateWarps) {
      uint64_t candidates[kIntermediateCandidatesPerLane];
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < kIntermediateCandidatesPerLane; ++item) {
        int const index = lane_idx + item * cutlass::NumThreadsPerWarp;
        int const source = warp_idx * kSourceWarpsPerIntermediate * kRankedTopK + index;
        auto const *flat_candidates =
            reinterpret_cast<uint64_t const *>(indexer.tail.chunks.candidates);
        candidates[item] = index < kSourceWarpsPerIntermediate * kRankedTopK
                               ? flat_candidates[source]
                               : raw_score_key(-3.402823466e+38F, -1);
      }
      sort_descending_by_raw_score(candidates);
      __syncwarp();
      CUTLASS_PRAGMA_UNROLL
      for (int round = 0; round < kRankedTopK; ++round) {
        float const score = raw_score_key_score(candidates[0]);
        float const best_score = warp_reduce_max(score);
        int const winner_lane = __reduce_min_sync(
            0xffffffffu, score == best_score ? lane_idx : cutlass::NumThreadsPerWarp);
        bool const winner = lane_idx == winner_lane;
        if (winner) {
          indexer.tail.chunks.candidates[warp_idx][round] = candidates[0];
        }
        candidates[0] = winner ? candidates[1] : candidates[0];
        candidates[1] = winner ? raw_score_key(-3.402823466e+38F, -1) : candidates[1];
      }
      Sm100M3DecodeNamedBarrier::sync(kIntermediateWarps * cutlass::NumThreadsPerWarp,
                                      kTopKBarrierId);
    }

    if (warp_idx == 0) {
      constexpr int kCandidatesPerLane =
          (kIntermediateWarps * kRankedTopK + cutlass::NumThreadsPerWarp - 1) /
          cutlass::NumThreadsPerWarp;
      uint64_t candidates[kCandidatesPerLane];
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < kCandidatesPerLane; ++item) {
        int const index = lane_idx + item * cutlass::NumThreadsPerWarp;
        auto const *flat_candidates =
            reinterpret_cast<uint64_t const *>(indexer.tail.chunks.candidates);
        candidates[item] = index < kIntermediateWarps * kRankedTopK
                               ? flat_candidates[index]
                               : raw_score_key(-3.402823466e+38F, -1);
      }
      sort_descending_by_raw_score(candidates);
      CUTLASS_PRAGMA_UNROLL
      for (int round = 0; round < kRankedTopK; ++round) {
        float const score = raw_score_key_score(candidates[0]);
        float const best_score = warp_reduce_max(score);
        int const winner_lane = __reduce_min_sync(
            0xffffffffu, score == best_score ? lane_idx : cutlass::NumThreadsPerWarp);
        bool const winner = lane_idx == winner_lane;
        if (winner) {
          storage.indexer_topk[round] = raw_score_key_page(candidates[0]);
        }
        candidates[0] = winner ? candidates[1] : candidates[0];
        candidates[1] = winner ? raw_score_key(-3.402823466e+38F, -1) : candidates[1];
      }
      if (lane_idx == kRankedTopK) {
        storage.indexer_topk[kTopK - 1] = visible_pages - 1;
      }
      __syncwarp();
      if (params.kv_block_indexes_ptr != nullptr && lane_idx < kTopK) {
        const_cast<int *>(params.kv_block_indexes_ptr)[row * kTopK + lane_idx] =
            storage.indexer_topk[lane_idx];
      }
    }
  }

  CUTE_DEVICE static void
  select_topk_boundary_exact(IndexerStorage &indexer, Storage &storage, Params const &params,
                             int row, int kv_length, int thread_idx, int warp_idx, int lane_idx,
                             float const (&exact_scores)[kFastTopKItemsPerThread]) {
    int const visible_pages = ((kv_length - kRows + row) / kPageTokens) + 1;
    int const ranked_pages = visible_pages - 1;
    uint32_t const boundary_score = indexer.tail.quantized.boundary_score;
    int const strict_count = indexer.tail.quantized.strict_count;
    int const needed = kRankedTopK - strict_count;

    if (warp_idx < kTopKWarps) {
      uint64_t keys[kFastTopKItemsPerThread];
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < kFastTopKItemsPerThread; ++item) {
        int const page = thread_idx + item * kTopKThreads;
        if (page < ranked_pages) {
          float const score = exact_scores[item];
          uint32_t const ordered_score = ordered_float_bits(score);
          keys[item] =
              (ordered_score >> kFastTopKPageBits) == boundary_score ? exact_key(score, page) : 0u;
        } else {
          keys[item] = 0u;
        }
      }
      sort_descending_by_score(keys);
      int selected_count = 0;
      while (selected_count < needed) {
        uint32_t const score = static_cast<uint32_t>(keys[0] >> 32);
        uint32_t const best_score = warp_max(score);
        uint32_t const winners = __ballot_sync(0xffffffffu, score == best_score);
        uint32_t lane_mask_lt;
        asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
        int const winner_rank = __popc(winners & lane_mask_lt);
        bool const winner = score == best_score;
        int const output = selected_count + winner_rank;
        if (winner && output < needed) {
          indexer.tail.chunks.candidates[warp_idx][output] = keys[0];
        }
        CUTLASS_PRAGMA_UNROLL
        for (int item = 0; item < kFastTopKItemsPerThread - 1; ++item) {
          keys[item] = winner ? keys[item + 1] : keys[item];
        }
        keys[kFastTopKItemsPerThread - 1] = winner ? 0u : keys[kFastTopKItemsPerThread - 1];
        int const remaining = needed - selected_count;
        int const winner_count = __popc(winners);
        selected_count += winner_count < remaining ? winner_count : remaining;
      }
      Sm100M3DecodeNamedBarrier::sync(kTopKThreads, kTopKBarrierId);
    }

    if (warp_idx == 0) {
      constexpr int kCandidatesPerLane =
          (kTopKWarps * kRankedTopK + cutlass::NumThreadsPerWarp - 1) / cutlass::NumThreadsPerWarp;
      uint64_t candidates[kCandidatesPerLane];
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < kCandidatesPerLane; ++item) {
        int const index = lane_idx + item * cutlass::NumThreadsPerWarp;
        int const local_index = index % kRankedTopK;
        auto const *flat_candidates =
            reinterpret_cast<uint64_t const *>(indexer.tail.chunks.candidates);
        candidates[item] =
            index < kTopKWarps * kRankedTopK && local_index < needed ? flat_candidates[index] : 0u;
      }
      sort_descending_by_score(candidates);
      int selected_count = 0;
      while (selected_count < needed) {
        uint32_t const score = static_cast<uint32_t>(candidates[0] >> 32);
        uint32_t const best_score = warp_max(score);
        uint32_t const winners = __ballot_sync(0xffffffffu, score == best_score);
        uint32_t lane_mask_lt;
        asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
        int const winner_rank = __popc(winners & lane_mask_lt);
        bool const winner = score == best_score;
        int const output = selected_count + winner_rank;
        if (winner && output < needed) {
          storage.indexer_topk[strict_count + output] = exact_key_page(candidates[0]);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int item = 0; item < kCandidatesPerLane - 1; ++item) {
          candidates[item] = winner ? candidates[item + 1] : candidates[item];
        }
        candidates[kCandidatesPerLane - 1] = winner ? 0u : candidates[kCandidatesPerLane - 1];
        int const remaining = needed - selected_count;
        int const winner_count = __popc(winners);
        selected_count += winner_count < remaining ? winner_count : remaining;
      }
      if (lane_idx == kRankedTopK) {
        storage.indexer_topk[kTopK - 1] = visible_pages - 1;
      }
      __syncwarp();
      if (params.kv_block_indexes_ptr != nullptr && lane_idx < kTopK) {
        const_cast<int *>(params.kv_block_indexes_ptr)[row * kTopK + lane_idx] =
            storage.indexer_topk[lane_idx];
      }
    }
  }

  template <int ItemsPerThread>
  CUTE_DEVICE static void
  select_topk_atomic_with_exact_fallback(IndexerStorage &indexer, Storage &storage,
                                         Params const &params, int row, int kv_length,
                                         int thread_idx, int warp_idx, int lane_idx) {
    static_assert(ItemsPerThread > 0);
    static_assert(ItemsPerThread * kTopKThreads <= kMaxPages);
    int const visible_pages = ((kv_length - kRows + row) / kPageTokens) + 1;
    int const ranked_pages = visible_pages - 1;
    float const *row_scores = score_row(params, row, kv_length);
    auto *slots = reinterpret_cast<uint32_t *>(indexer.tail.quantized.candidates);

    if (warp_idx < kTopKWarps) {
      uint32_t keys[ItemsPerThread];
      float local_maximum = -1.0e30f;
      float local_minimum = 1.0e30f;
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < ItemsPerThread; ++item) {
        int const page = thread_idx + item * kTopKThreads;
        if (page < ranked_pages) {
          float const score = __ldcg(row_scores + page);
          keys[item] = __float_as_uint(score);
          local_maximum = fmaxf(local_maximum, score);
          local_minimum = fminf(local_minimum, score);
        } else {
          keys[item] = 0u;
        }
      }
      local_maximum = warp_reduce_max(local_maximum);
      local_minimum = warp_reduce_min(local_minimum);
      if (lane_idx == 0) {
        indexer.tail.quantized.extrema[0][warp_idx] = local_maximum;
        indexer.tail.quantized.extrema[1][warp_idx] = local_minimum;
      }
      Sm100M3DecodeNamedBarrier::sync(kTopKThreads, kTopKBarrierId);

      float row_maximum = indexer.tail.quantized.extrema[0][0];
      float row_minimum = indexer.tail.quantized.extrema[1][0];
      CUTLASS_PRAGMA_UNROLL
      for (int source_warp = 1; source_warp < kTopKWarps; ++source_warp) {
        row_maximum = fmaxf(row_maximum, indexer.tail.quantized.extrema[0][source_warp]);
        row_minimum = fminf(row_minimum, indexer.tail.quantized.extrema[1][source_warp]);
      }
      float const scale = row_maximum > row_minimum ? 65534.0f / (row_maximum - row_minimum) : 0.0f;
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < ItemsPerThread; ++item) {
        int const page = thread_idx + item * kTopKThreads;
        if (page < ranked_pages) {
          float const score = __uint_as_float(keys[item]);
          uint32_t quantized = 1u + static_cast<uint32_t>((score - row_minimum) * scale + 0.5f);
          quantized = quantized > 65535u ? 65535u : quantized;
          keys[item] = (quantized << 16) | (static_cast<uint32_t>(~page) & 0xffffu);
        }
      }
      sort_descending(keys);

      if (thread_idx < kTopK) {
        slots[thread_idx] = 0u;
      }
      Sm100M3DecodeNamedBarrier::sync(kTopKThreads, kTopKBarrierId);
      CUTLASS_PRAGMA_UNROLL
      for (int selected = 0; selected < kTopK; ++selected) {
        atomicMax(slots + selected, keys[0]);
        Sm100M3DecodeNamedBarrier::sync(kTopKThreads, kTopKBarrierId);
        bool const winner = keys[0] == slots[selected];
        CUTLASS_PRAGMA_UNROLL
        for (int item = 0; item < ItemsPerThread - 1; ++item) {
          keys[item] = winner ? keys[item + 1] : keys[item];
        }
        keys[ItemsPerThread - 1] = winner ? 0u : keys[ItemsPerThread - 1];
      }
    }

    bool const boundary_collision = (slots[kRankedTopK - 1] >> 16) == (slots[kRankedTopK] >> 16);
    if (boundary_collision) {
      select_topk_exact_two_level<ItemsPerThread>(indexer, storage, params, row, kv_length,
                                                  thread_idx, warp_idx, lane_idx);
      return;
    }

    if (warp_idx == 0) {
      if (lane_idx < kRankedTopK) {
        storage.indexer_topk[lane_idx] = static_cast<int>((~slots[lane_idx]) & 0xffffu);
      } else if (lane_idx == kRankedTopK) {
        storage.indexer_topk[kTopK - 1] = visible_pages - 1;
      }
      __syncwarp();
      if (params.kv_block_indexes_ptr != nullptr && lane_idx < kTopK) {
        const_cast<int *>(params.kv_block_indexes_ptr)[row * kTopK + lane_idx] =
            storage.indexer_topk[lane_idx];
      }
    }
  }

  CUTE_DEVICE static void select_topk_cta_candidates(IndexerStorage &indexer, Storage &storage,
                                                     Params const &params, int row, int kv_length,
                                                     int thread_idx, int warp_idx, int lane_idx) {
    uint64_t const *row_candidates =
        cta_candidate_scratch(params) + static_cast<size_t>(row) * kCtaCandidateCount;
    if (warp_idx < kTopKWarps) {
      uint64_t keys[kCtaCandidateItemsPerThread];
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < kCtaCandidateItemsPerThread; ++item) {
        int const index = thread_idx + item * kTopKThreads;
        keys[item] = index < kCtaCandidateCount ? __ldcg(row_candidates + index)
                                                : raw_score_key(-3.402823466e+38F, -1);
      }
      sort_descending_nine_raw_score(keys);
      int selected_count = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int round = 0; round < kRankedTopK; ++round) {
        float const score = raw_score_key_score(keys[0]);
        float const best_score = warp_reduce_max(score);
        uint32_t const winners = __ballot_sync(0xffffffffu, score == best_score);
        uint32_t lane_mask_lt;
        asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
        int const winner_rank = __popc(winners & lane_mask_lt);
        bool const winner = score == best_score;
        int const output = selected_count + winner_rank;
        if (winner && output < kRankedTopK) {
          indexer.tail.chunks.candidates[warp_idx][output] = keys[0];
        }
        CUTLASS_PRAGMA_UNROLL
        for (int item = 0; item < kCtaCandidateItemsPerThread - 1; ++item) {
          keys[item] = winner ? keys[item + 1] : keys[item];
        }
        keys[kCtaCandidateItemsPerThread - 1] =
            winner ? raw_score_key(-3.402823466e+38F, -1) : keys[kCtaCandidateItemsPerThread - 1];
        int const remaining = kRankedTopK - selected_count;
        int const winner_count = __popc(winners);
        selected_count += winner_count < remaining ? winner_count : remaining;
      }
      Sm100M3DecodeNamedBarrier::sync(kTopKThreads, kTopKBarrierId);
    }

    if (warp_idx == 0) {
      constexpr int kCandidatesPerLane =
          (kTopKWarps * kRankedTopK + cutlass::NumThreadsPerWarp - 1) / cutlass::NumThreadsPerWarp;
      uint64_t candidates[kCandidatesPerLane];
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < kCandidatesPerLane; ++item) {
        int const index = lane_idx + item * cutlass::NumThreadsPerWarp;
        auto const *flat_candidates =
            reinterpret_cast<uint64_t const *>(indexer.tail.chunks.candidates);
        candidates[item] = index < kTopKWarps * kRankedTopK ? flat_candidates[index]
                                                            : raw_score_key(-3.402823466e+38F, -1);
      }
      sort_descending_by_raw_score(candidates);
      int selected_count = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int round = 0; round < kRankedTopK; ++round) {
        float const score = raw_score_key_score(candidates[0]);
        float const best_score = warp_reduce_max(score);
        uint32_t const winners = __ballot_sync(0xffffffffu, score == best_score);
        uint32_t lane_mask_lt;
        asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
        int const winner_rank = __popc(winners & lane_mask_lt);
        bool const winner = score == best_score;
        int const output = selected_count + winner_rank;
        if (winner && output < kRankedTopK) {
          storage.indexer_topk[output] = raw_score_key_page(candidates[0]);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int item = 0; item < kCandidatesPerLane - 1; ++item) {
          candidates[item] = winner ? candidates[item + 1] : candidates[item];
        }
        candidates[kCandidatesPerLane - 1] =
            winner ? raw_score_key(-3.402823466e+38F, -1) : candidates[kCandidatesPerLane - 1];
        int const remaining = kRankedTopK - selected_count;
        int const winner_count = __popc(winners);
        selected_count += winner_count < remaining ? winner_count : remaining;
      }
      if (lane_idx == kRankedTopK) {
        int const visible_pages = ((kv_length - kRows + row) / kPageTokens) + 1;
        storage.indexer_topk[kTopK - 1] = visible_pages - 1;
      }
      __syncwarp();
      if (params.kv_block_indexes_ptr != nullptr && lane_idx < kTopK) {
        const_cast<int *>(params.kv_block_indexes_ptr)[row * kTopK + lane_idx] =
            storage.indexer_topk[lane_idx];
      }
    }
  }

  CUTE_DEVICE static void select_topk_quantized(IndexerStorage &indexer, Storage &storage,
                                                Params const &params, int row, int kv_length,
                                                int thread_idx, int warp_idx, int lane_idx) {
    int const visible_pages = ((kv_length - kRows + row) / kPageTokens) + 1;
    if (visible_pages <= kTopK) {
      if (warp_idx == 0) {
        if (lane_idx == 0) {
          select_topk_serial(storage, params, row, kv_length);
          if (params.kv_block_indexes_ptr != nullptr) {
            CUTLASS_PRAGMA_UNROLL
            for (int selected = 0; selected < kTopK; ++selected) {
              const_cast<int *>(params.kv_block_indexes_ptr)[row * kTopK + selected] =
                  storage.indexer_topk[selected];
            }
          }
        }
        __syncwarp();
      }
      return;
    }
    if constexpr (Traits::kIndexerTopKItemsPerThread != 0) {
      static_assert(Traits::kIndexerTopKItemsPerThread == 8 ||
                    Traits::kIndexerTopKItemsPerThread == 16 ||
                    Traits::kIndexerTopKItemsPerThread == 32);
      if constexpr (Traits::kIndexerTopKItemsPerThread == 8) {
        select_topk_atomic_with_exact_fallback<Traits::kIndexerTopKItemsPerThread>(
            indexer, storage, params, row, kv_length, thread_idx, warp_idx, lane_idx);
      } else if constexpr (Traits::kIndexerTopKItemsPerThread == 16) {
        select_topk_exact_two_level_raw_score<8, kMaximumTopKWarps>(
            indexer, storage, params, row, kv_length, thread_idx, warp_idx, lane_idx);
      } else {
        select_topk_exact_two_level<Traits::kIndexerTopKItemsPerThread>(
            indexer, storage, params, row, kv_length, thread_idx, warp_idx, lane_idx);
      }
      return;
    }
    if (visible_pages <= kFastTopKMaxPages) {
      select_topk_fast(indexer, storage, params, row, kv_length, thread_idx, warp_idx, lane_idx);
      return;
    }

    int const ranked_pages = visible_pages - 1;
    int const chunk_count = (ranked_pages + kFastTopKMaxPages - 1) / kFastTopKMaxPages;
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
      int const page_begin = chunk * kFastTopKMaxPages;
      int const chunk_end = page_begin + kFastTopKMaxPages;
      int const page_end = chunk_end < ranked_pages ? chunk_end : ranked_pages;
      select_topk_chunk(indexer, params, row, kv_length, chunk, page_begin, page_end, thread_idx,
                        warp_idx, lane_idx);
    }

    if (warp_idx == 0) {
      float candidate_scores[kTopKCandidatesPerLane];
      int candidate_pages[kTopKCandidatesPerLane];
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < kTopKCandidatesPerLane; ++item) {
        int const index = lane_idx + item * cutlass::NumThreadsPerWarp;
        auto const *flat_scores =
            reinterpret_cast<float const *>(indexer.tail.chunks.candidate_scores);
        auto const *flat_pages = reinterpret_cast<int const *>(indexer.tail.chunks.candidate_pages);
        bool const valid = index < chunk_count * kTopK;
        candidate_scores[item] = valid ? flat_scores[index] : -1.0e30f;
        candidate_pages[item] = valid ? flat_pages[index] : kMaxPages;
      }
      CUTLASS_PRAGMA_UNROLL
      for (int selected = 0; selected < kTopK - 1; ++selected) {
        float lane_score = -1.0e30f;
        int lane_page = kMaxPages;
        CUTLASS_PRAGMA_UNROLL
        for (int item = 0; item < kTopKCandidatesPerLane; ++item) {
          bool const better =
              candidate_scores[item] > lane_score ||
              (candidate_scores[item] == lane_score && candidate_pages[item] < lane_page);
          if (better) {
            lane_score = candidate_scores[item];
            lane_page = candidate_pages[item];
          }
        }
        float const best_score = warp_reduce_max(lane_score);
        uint32_t const page_key = lane_score == best_score && lane_page < kMaxPages
                                      ? static_cast<uint32_t>(kMaxPages - lane_page)
                                      : 0u;
        uint32_t const best_page_key = warp_max(page_key);
        int const best_page = kMaxPages - static_cast<int>(best_page_key);
        if (lane_idx == 0) {
          storage.indexer_topk[selected] = best_page;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int item = 0; item < kTopKCandidatesPerLane; ++item) {
          bool const winner =
              candidate_scores[item] == best_score && candidate_pages[item] == best_page;
          candidate_scores[item] = winner ? -1.0e30f : candidate_scores[item];
          candidate_pages[item] = winner ? kMaxPages : candidate_pages[item];
        }
      }
      if (lane_idx == 0) {
        storage.indexer_topk[kTopK - 1] = visible_pages - 1;
        if (params.kv_block_indexes_ptr != nullptr) {
          CUTLASS_PRAGMA_UNROLL
          for (int selected = 0; selected < kTopK; ++selected) {
            const_cast<int *>(params.kv_block_indexes_ptr)[row * kTopK + selected] =
                storage.indexer_topk[selected];
          }
        }
      }
      __syncwarp();
    }
  }

  CUTE_DEVICE static void publish_topk(Storage &storage, int cluster_size, int cluster_rank,
                                       int thread_idx, int lane_idx) {
    if (cluster_rank == 0 && thread_idx < 4) {
      int4 const value = reinterpret_cast<int4 const *>(storage.indexer_topk)[lane_idx];
      CUTLASS_PRAGMA_UNROLL
      for (int peer = 0; peer < 8; ++peer) {
        if (peer < cluster_size) {
          store_remote_int4(value, reinterpret_cast<int4 *>(storage.indexer_topk) + lane_idx,
                            storage.indexer_topk_barrier_ptr(), peer);
        }
      }
    }
    if (thread_idx < cutlass::NumThreadsPerWarp) {
      __syncwarp();
      if (lane_idx == 0) {
        wait(storage.indexer_topk_barrier_ptr(), 0);
      }
    }
  }
};

} // namespace cutlass::m3_decode::collective
