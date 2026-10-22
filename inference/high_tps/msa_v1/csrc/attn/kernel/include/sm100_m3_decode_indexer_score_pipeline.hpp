#pragma once

#include "attn/kernel/include/sm100_m3_decode_indexer_topk.hpp"
#include "common/kernel/include/sm100_m3_decode_storage.hpp"

namespace cutlass::m3_decode::collective {

template <class Traits>
struct Sm100M3DecodeIndexerScorePipeline : Sm100M3DecodeIndexerTopK<Traits> {
  using Base = Sm100M3DecodeIndexerTopK<Traits>;
  using Base::arrive;
  using Base::broadcast_scale_byte;
  using Base::cta_candidate_scratch;
  using Base::exact_key;
  using Base::expect_transactions;
  using Base::global_control;
  using Base::hmma_e4m3;
  using Base::init_barrier;
  using Base::initialize_topk_mailbox;
  using Base::kCtaCandidateStride;
  using Base::kFallbackScoreCtaStart;
  using Base::kGridCtas;
  using Base::kHeadDim;
  using Base::kMaxPagesPerCta;
  using Base::kMaxScoreProducerCtas;
  using Base::kNeedsPageRefill;
  using Base::kPackedKBytes;
  using Base::kPageBytes;
  using Base::kPageTokens;
  using Base::kPreferredScoreCtas;
  using Base::kRankedTopK;
  using Base::kRows;
  using Base::kScaleGroups;
  using Base::kScoreGroups;
  using Base::kScorePublishBarrierId;
  using Base::kScoreWarps;
  using Base::kTargetPagesPerProducer;
  using Base::kTokensPerScoreWarp;
  using Base::kTopK;
  using Base::kTopKBarrierId;
  using Base::kTopKThreads;
  using Base::kTopKWarps;
  using Base::kUnifiedTopK;
  using Base::kUseCtaCandidateHandoff;
  using Base::load_matrix_x4;
  using Base::publish_generation;
  using Base::publish_scores;
  using Base::publish_topk;
  using Base::qmul4;
  using Base::raw_score_key;
  using Base::raw_score_key_score;
  using Base::release_add;
  using Base::score_row;
  using Base::select_topk_cta_candidates;
  using Base::select_topk_quantized;
  using Base::select_topk_serial;
  using Base::smem_address;
  using Base::sort_descending_by_raw_score;
  using Base::store_no_allocate;
  using Base::swizzled_page_offset;
  using Base::wait;
  using Base::wait_all_scores;
  using Base::wait_generation;
  using Base::warp_reduce_max;
  using Base::warp_sort_descending;
  using typename Base::IndexerStorage;
  using typename Base::Params;
  using typename Base::Scheduler;
  using typename Base::Storage;
  CUTE_DEVICE static int producer_count(int page_count) {
    int const required = (page_count + kTargetPagesPerProducer - 1) / kTargetPagesPerProducer;
    return required < kMaxScoreProducerCtas ? required : kMaxScoreProducerCtas;
  }

  CUTE_DEVICE static int producer_ordinal(int producers) {
    int const block = m3_decode_logical_block_idx();
    int const preferred = producers < kPreferredScoreCtas ? producers : kPreferredScoreCtas;
    if (block < preferred) {
      return block;
    }
    int const fallback = producers - preferred;
    if (block >= kFallbackScoreCtaStart && block < kFallbackScoreCtaStart + fallback) {
      return preferred + block - kFallbackScoreCtaStart;
    }
    return -1;
  }

  CUTE_DEVICE static void producer_page_range(int page_count, int producers, int producer,
                                              int &page_begin, int &page_end) {
    if (producers == kMaxScoreProducerCtas) {
      if constexpr (kUnifiedTopK || Traits::kIndexerTopKItemsPerThread >= 16) {
        constexpr int kAttentionProjectionCtaBegin = 6 * 8;
        constexpr int kAttentionProjectionCtaEnd = 14 * 8;
        constexpr int kAttentionProjectionStartupPages = 1;
        constexpr int kNonAttentionProjectionCtas =
            kMaxScoreProducerCtas - (kAttentionProjectionCtaEnd - kAttentionProjectionCtaBegin);
        int const base_pages =
            page_count - kNonAttentionProjectionCtas * kAttentionProjectionStartupPages;
        int const block = m3_decode_logical_block_idx();
        int const non_attention_projection_before =
            block <= kAttentionProjectionCtaBegin ? block
            : block < kAttentionProjectionCtaEnd
                ? kAttentionProjectionCtaBegin
                : block - (kAttentionProjectionCtaEnd - kAttentionProjectionCtaBegin);
        int const non_attention_projection_after =
            block + 1 <= kAttentionProjectionCtaBegin ? block + 1
            : block + 1 < kAttentionProjectionCtaEnd
                ? kAttentionProjectionCtaBegin
                : block + 1 - (kAttentionProjectionCtaEnd - kAttentionProjectionCtaBegin);
        if constexpr (!kUnifiedTopK && Traits::kWaveBalancedIndexerPages) {
          int const attention_projection_before = block - non_attention_projection_before;
          int const attention_projection_after = block + 1 - non_attention_projection_after;
          constexpr int kAttentionProjectionCtas =
              kAttentionProjectionCtaEnd - kAttentionProjectionCtaBegin;
          int const pages_per_cta = base_pages / kMaxScoreProducerCtas;
          int const remainder = base_pages - pages_per_cta * kMaxScoreProducerCtas;
          page_begin = block * pages_per_cta +
                       non_attention_projection_before * kAttentionProjectionStartupPages +
                       attention_projection_before * remainder / kAttentionProjectionCtas;
          page_end = (block + 1) * pages_per_cta +
                     non_attention_projection_after * kAttentionProjectionStartupPages +
                     attention_projection_after * remainder / kAttentionProjectionCtas;
          return;
        }
        page_begin = block * base_pages / kMaxScoreProducerCtas +
                     non_attention_projection_before * kAttentionProjectionStartupPages;
        page_end = (block + 1) * base_pages / kMaxScoreProducerCtas +
                   non_attention_projection_after * kAttentionProjectionStartupPages;
        return;
      }

      // Attention Projection producers start after the QI/KI handoff.
      // Earlier producers receive proportionally more score pages. Within
      // each producer cluster, move one page from each Q output owner to a
      // shorter special-head split.
      constexpr int kAttentionProjectionCtaBegin = 6 * 8;
      constexpr int kAttentionProjectionCtaEnd = 14 * 8;
      constexpr int kAttentionProjectionClusterWeight = 16;
      constexpr int kNonAttentionProjectionWeight = 8;
      constexpr int kTotalWeight =
          ((kAttentionProjectionCtaEnd - kAttentionProjectionCtaBegin) / 8) *
              kAttentionProjectionClusterWeight +
          (kMaxScoreProducerCtas - (kAttentionProjectionCtaEnd - kAttentionProjectionCtaBegin)) *
              kNonAttentionProjectionWeight;
      int const block = m3_decode_logical_block_idx();
      int weight_begin = block * kNonAttentionProjectionWeight;
      int weight = kNonAttentionProjectionWeight;
      if (block >= kAttentionProjectionCtaBegin && block < kAttentionProjectionCtaEnd) {
        int const local = block - kAttentionProjectionCtaBegin;
        int const attention_projection_cluster = local / 8;
        int const cluster_rank = local & 7;
        int const cluster_prefix = cluster_rank * 2 - int(cluster_rank >= 1) -
                                   int(cluster_rank >= 4) + int(cluster_rank == 7);
        weight_begin = kAttentionProjectionCtaBegin * kNonAttentionProjectionWeight +
                       attention_projection_cluster * kAttentionProjectionClusterWeight +
                       cluster_prefix;
        weight = 2 - int(cluster_rank == 0 || cluster_rank == 3) + int(cluster_rank >= 6);
      } else if (block >= kAttentionProjectionCtaEnd) {
        weight_begin -= (kAttentionProjectionCtaEnd - kAttentionProjectionCtaBegin) *
                            kNonAttentionProjectionWeight -
                        ((kAttentionProjectionCtaEnd - kAttentionProjectionCtaBegin) / 8) *
                            kAttentionProjectionClusterWeight;
      }
      page_begin = weight_begin * page_count / kTotalWeight;
      page_end = (weight_begin + weight) * page_count / kTotalWeight;
      return;
    }
    page_begin = producer >= 0 ? producer * page_count / producers : 0;
    page_end = producer >= 0 ? (producer + 1) * page_count / producers : 0;
  }

  CUTE_DEVICE static void issue_page_load(Params const &params, IndexerStorage &indexer,
                                          int page_begin, int local_page, int page_count) {
    int const logical_page = page_begin + local_page;
    int const physical_page = __ldg(params.indexer_page_table_ptr + logical_page);
    issue_physical_page_load(params, indexer, local_page, physical_page, logical_page, page_count);
  }

  CUTE_DEVICE static void issue_physical_page_load(Params const &params, IndexerStorage &indexer,
                                                   int local_page, int physical_page,
                                                   int logical_page, int page_count) {
    if (logical_page + 1 == page_count) {
      Scheduler::wait_tagged(Scheduler::attention_projection_ki_ready_address(params),
                             Scheduler::current_generation_prefix(params));
    }
    int const slot = local_page & (kMaxPagesPerCta - 1);
    if (local_page >= kMaxPagesPerCta) {
      wait(indexer.page_consumed_barriers + slot,
           static_cast<uint32_t>((local_page / kMaxPagesPerCta - 1) & 1));
    }
    uint64_t *barrier = indexer.page_barriers + slot;
    expect_transactions(barrier, kPageBytes);
    Sm100M3DecodeTma::load_3d_predicated(&params.tma.indexer_k, indexer.pages[slot], barrier, 0, 0,
                                         physical_page, 1, cute::TMA::CacheHintSm90::EVICT_FIRST);
    Sm100M3DecodeTma::load_3d_predicated(&params.tma.indexer_k_scale,
                                         indexer.pages[slot] + kPackedKBytes, barrier, 0, 0,
                                         physical_page, 1, cute::TMA::CacheHintSm90::EVICT_FIRST);
  }

  CUTE_DEVICE static float fold_page_score(IndexerStorage const &indexer, int page, int row) {
    float4 const partials0 =
        *reinterpret_cast<float4 const *>(&indexer.page_partials[page][row][0]);
    float4 const partials1 =
        *reinterpret_cast<float4 const *>(&indexer.page_partials[page][row][4]);
    return fmaxf(fmaxf(fmaxf(partials0.x, partials0.y), fmaxf(partials0.z, partials0.w)),
                 fmaxf(fmaxf(partials1.x, partials1.y), fmaxf(partials1.z, partials1.w)));
  }

  CUTE_DEVICE static void publish_cta_candidates(Params const &params, IndexerStorage &indexer,
                                                 int page_begin, int local_pages, int kv_length,
                                                 int warp_idx, int lane_idx) {
    if (warp_idx < kRows) {
      int const row = warp_idx;
      int const visible_pages = ((kv_length - kRows + row) / kPageTokens) + 1;
      int const ranked_pages = visible_pages - 1;
      if constexpr (Traits::kIndexerTopKItemsPerThread == 16) {
        uint64_t key = 0u;
        int const page = lane_idx;
        if (page < local_pages) {
          float const score = fold_page_score(indexer, page, row);
          int const logical_page = page_begin + page;
          if (logical_page < ranked_pages) {
            key = exact_key(score, logical_page);
          }
        }
        key = warp_sort_descending(key, lane_idx);
        if (lane_idx < kCtaCandidateStride) {
          uint64_t *destination =
              cta_candidate_scratch(params) +
              (static_cast<size_t>(row) * kGridCtas + m3_decode_logical_block_idx()) *
                  kCtaCandidateStride +
              lane_idx;
          store_no_allocate(destination, key);
        }
        return;
      }

      uint64_t keys[2] = {raw_score_key(-3.402823466e+38F, -1),
                          raw_score_key(-3.402823466e+38F, -1)};
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < 2; ++item) {
        int const page = lane_idx + item * cutlass::NumThreadsPerWarp;
        if (page < local_pages) {
          float const score = fold_page_score(indexer, page, row);
          int const logical_page = page_begin + page;
          if (logical_page < ranked_pages) {
            keys[item] = raw_score_key(score, logical_page);
          }
        }
      }
      sort_descending_by_raw_score(keys);
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
        keys[0] = winner ? keys[1] : keys[0];
        keys[1] = winner ? raw_score_key(-3.402823466e+38F, -1) : keys[1];
        int const remaining = kRankedTopK - selected_count;
        int const winner_count = __popc(winners);
        selected_count += winner_count < remaining ? winner_count : remaining;
      }
      __syncwarp();
      if (lane_idx < kCtaCandidateStride) {
        uint64_t *destination =
            cta_candidate_scratch(params) +
            (static_cast<size_t>(row) * kGridCtas + m3_decode_logical_block_idx()) *
                kCtaCandidateStride +
            lane_idx;
        uint64_t const candidate = lane_idx < kRankedTopK
                                       ? indexer.tail.chunks.candidates[warp_idx][lane_idx]
                                       : raw_score_key(-3.402823466e+38F, -1);
        store_no_allocate(destination, candidate);
      }
    }
  }

  template <bool Standalone = false>
  CUTE_DEVICE static void run_hmma(Params const &params, Storage &storage, uint8_t *smem,
                                   int thread_idx, int warp_idx, int lane_idx, int cluster_size,
                                   int cluster_rank, bool attention_cluster, int attention_row,
                                   bool retain_tmem) {
    IndexerStorage &indexer = *reinterpret_cast<IndexerStorage *>(smem);
    int const kv_length = params.kv_segment_lens_ptr != nullptr
                              ? __ldg(params.kv_segment_lens_ptr)
                              : params.kv_page_stride * kPageTokens;
    int const page_count = (kv_length + kPageTokens - 1) / kPageTokens;
    int const producers = producer_count(page_count);
    int const producer = producer_ordinal(producers);
    bool const is_producer = producer >= 0;
    int page_begin;
    int page_end;
    producer_page_range(page_count, producers, producer, page_begin, page_end);
    int const local_pages = page_end - page_begin;
    bool const use_page_refill = kNeedsPageRefill && local_pages > kMaxPagesPerCta;
    uint64_t const generation = publish_generation(params, thread_idx);

    if constexpr (!Standalone) {
      initialize_topk_mailbox(storage, attention_cluster, thread_idx);
    }

    if (page_count <= kTopK && !attention_cluster && !retain_tmem) {
      if (thread_idx == 0) {
        publish_scores(params);
      }
      return;
    }

    if (!is_producer && !attention_cluster && !retain_tmem) {
      if constexpr (kUseCtaCandidateHandoff) {
        // Legacy whole-layer TopK still consumes every CTA's candidate slots.
        if (thread_idx < kRows * kCtaCandidateStride) {
          int const row = thread_idx / kCtaCandidateStride;
          int const slot = thread_idx % kCtaCandidateStride;
          store_no_allocate(
              cta_candidate_scratch(params) +
                  (static_cast<size_t>(row) * kGridCtas + m3_decode_logical_block_idx()) *
                      kCtaCandidateStride +
                  slot,
              raw_score_key(-3.402823466e+38F, -1));
        }
        __syncthreads();
      }
      if (thread_idx == 0) {
        publish_scores(params);
      }
      return;
    }

    if (retain_tmem && thread_idx == 0) {
      init_barrier(&indexer.allocation_barrier, 1);
    }

    if (page_count > kTopK && is_producer) {
      if (thread_idx < kMaxPagesPerCta) {
        init_barrier(indexer.page_barriers + thread_idx, 1);
      } else if constexpr (kNeedsPageRefill) {
        if (use_page_refill && thread_idx < 2 * kMaxPagesPerCta) {
          init_barrier(indexer.page_consumed_barriers + thread_idx - kMaxPagesPerCta, kScoreWarps);
        }
      }
    }
    cutlass::arch::fence_barrier_init();

    // Start the initial page transfers after publishing the mbarrier
    // initialization. The final CTA barrier makes both the barriers and the
    // staged query tile visible to all score warps.
    if (page_count > kTopK && is_producer) {
      if (thread_idx < local_pages && thread_idx < kMaxPagesPerCta) {
        issue_page_load(params, indexer, page_begin, thread_idx, page_count);
      }
      if (thread_idx < (kRows * kHeadDim) / 16) {
        reinterpret_cast<uint4 *>(indexer.q_tile)[thread_idx] =
            reinterpret_cast<uint4 const *>(params.indexer_q_ptr)[thread_idx];
      }
    }
    int owned_refill_physical_page = -1;
    if constexpr (Traits::kIndexerTopKItemsPerThread == 16) {
      if (is_producer && warp_idx < kScoreGroups * kScoreWarps) {
        int const score_group = warp_idx / kScoreWarps;
        int const score_warp = warp_idx % kScoreWarps;
        int const owned_refill_page = kMaxPagesPerCta + 2 * score_warp + score_group;
        if (lane_idx == 0 && owned_refill_page < local_pages) {
          owned_refill_physical_page =
              __ldg(params.indexer_page_table_ptr + page_begin + owned_refill_page);
        }
        owned_refill_physical_page = __shfl_sync(0xffffffffu, owned_refill_physical_page, 0);
      }
    } else if constexpr (Traits::kIndexerTopKItemsPerThread == 32) {
      if (is_producer && warp_idx < kScoreGroups * kScoreWarps) {
        int const score_group = warp_idx / kScoreWarps;
        int const score_warp = warp_idx % kScoreWarps;
        int const owned_refill_page =
            kMaxPagesPerCta + 2 * score_warp + score_group + lane_idx * kMaxPagesPerCta;
        if (lane_idx < 3 && owned_refill_page < local_pages) {
          owned_refill_physical_page =
              __ldg(params.indexer_page_table_ptr + page_begin + owned_refill_page);
        }
      }
    }
    __syncthreads();

    if (retain_tmem && warp_idx == 0) {
      cute::TMEM::Allocator1Sm allocator;
      allocator.allocate(Traits::kNumTmemCols, &indexer.tmem_base);
      if (cute::elect_one_sync()) {
        arrive(&indexer.allocation_barrier);
      }
    }

    if (page_count <= kTopK) {
      if (attention_cluster && !Standalone) {
        storage.attention_pipelines().init_pipeline_barriers(warp_idx, lane_idx);
        if (thread_idx == 0) {
          cutlass::arch::ClusterTransactionBarrier::init(storage.cluster_reduction_barrier_ptr(),
                                                         1);
        }
        cutlass::arch::fence_barrier_init();
        if (thread_idx == 0) {
          cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
              storage.cluster_reduction_barrier_ptr(), Traits::kClusterReductionTransactionBytes);
        }
      }
      if (thread_idx == 0) {
        publish_scores(params);
      }
      if (attention_cluster && cluster_rank == 0 && thread_idx == 0) {
        uint64_t const epoch = wait_generation(params, generation);
        wait_all_scores(params, epoch);
        select_topk_serial(storage, params, attention_row, kv_length);
        if (params.kv_block_indexes_ptr != nullptr) {
          CUTLASS_PRAGMA_UNROLL
          for (int selected = 0; selected < kTopK; ++selected) {
            const_cast<int *>(params.kv_block_indexes_ptr)[attention_row * kTopK + selected] =
                storage.indexer_topk[selected];
          }
        }
      }
      if (attention_cluster && cluster_rank == 0 && warp_idx == 0) {
        __syncwarp();
      }
      if (retain_tmem && thread_idx == 0) {
        wait(&indexer.allocation_barrier, 0);
        storage.tmem_state_ptr()[0] = indexer.tmem_base;
        cutlass::arch::ClusterBarrier::invalidate(&indexer.allocation_barrier);
      }
      if (attention_cluster && !Standalone) {
        publish_topk(storage, cluster_size, cluster_rank, thread_idx, lane_idx);
      } else if (!Standalone) {
        __syncthreads();
      }
      return;
    }

    if constexpr (!Standalone) {
      if (attention_cluster && warp_idx == 15 && cute::elect_one_sync()) {
        Sm100M3DecodeTma::prefetch_tensormap(&params.tma.q);
        Sm100M3DecodeTma::prefetch_tensormap(&params.tma.k);
        Sm100M3DecodeTma::prefetch_tensormap(&params.tma.v);
      }
    }

    if (is_producer && warp_idx < kScoreGroups * kScoreWarps) {
      int const score_group = warp_idx / kScoreWarps;
      int const score_warp = warp_idx % kScoreWarps;
      int const query_column = lane_idx >> 2;
      int const thread_group = lane_idx & 3;

      int const mask_page = (kv_length - kRows) / kPageTokens;
      int const token_begin = score_warp * kTokensPerScoreWarp;
      int const matrix_offset_a = swizzled_page_offset(token_begin + (lane_idx & 7), lane_idx >> 3);
      int const matrix_offset_b =
          swizzled_page_offset(token_begin + 8 + (lane_idx & 7), lane_idx >> 3);
      int const scale_offset_a = kPackedKBytes + (token_begin + (lane_idx >> 2)) * kScaleGroups;
      int const scale_offset_b = scale_offset_a + 8 * kScaleGroups;

      struct PageFragments {
        uint32_t matrix_a[4];
        uint32_t matrix_b[4];
        uint2 scale_a;
        uint2 scale_b;
      };

      auto load_page = [&](int slot, uint32_t phase, PageFragments &fragments) {
        wait(indexer.page_barriers + slot, phase);
        uint8_t const *packed = indexer.pages[slot];
        load_matrix_x4(fragments.matrix_a, smem_address(packed + matrix_offset_a));
        load_matrix_x4(fragments.matrix_b, smem_address(packed + matrix_offset_b));
        fragments.scale_a = *reinterpret_cast<uint2 const *>(packed + scale_offset_a);
        fragments.scale_b = *reinterpret_cast<uint2 const *>(packed + scale_offset_b);
      };

      auto compute_page = [&](int page, PageFragments const &fragments) {
        float accumulator[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        CUTLASS_PRAGMA_UNROLL
        for (int k_chunk = 0; k_chunk < 4; ++k_chunk) {
          uint32_t matrix[4];
          int const query_offset = query_column * kHeadDim + 32 * k_chunk + 8 * thread_group;
          uint2 const query_fragment =
              *reinterpret_cast<uint2 const *>(indexer.q_tile + query_offset);
          uint32_t const scale_a =
              broadcast_scale_byte(fragments.scale_a, 2 * k_chunk + (thread_group >> 1));
          uint32_t const scale_b =
              broadcast_scale_byte(fragments.scale_b, 2 * k_chunk + (thread_group >> 1));
          matrix[0] = qmul4(static_cast<uint16_t>(fragments.matrix_a[k_chunk]), scale_a);
          matrix[2] = qmul4(static_cast<uint16_t>(fragments.matrix_a[k_chunk] >> 16), scale_a);
          matrix[1] = qmul4(static_cast<uint16_t>(fragments.matrix_b[k_chunk]), scale_b);
          matrix[3] = qmul4(static_cast<uint16_t>(fragments.matrix_b[k_chunk] >> 16), scale_b);
          hmma_e4m3(accumulator, matrix, query_fragment);
        }

        int const logical_page = page_begin + page;
        if (logical_page >= mask_page) {
          int const token = logical_page * kPageTokens + token_begin + (lane_idx >> 2);
          int const query = kv_length - kRows + 2 * thread_group;
          if (token > query) {
            accumulator[0] = -1.0e30F;
          }
          if (token > query + 1) {
            accumulator[1] = -1.0e30F;
          }
          if (token + 8 > query) {
            accumulator[2] = -1.0e30F;
          }
          if (token + 8 > query + 1) {
            accumulator[3] = -1.0e30F;
          }
        }

        float maximum0 = fmaxf(accumulator[0], accumulator[2]);
        float maximum1 = fmaxf(accumulator[1], accumulator[3]);
        CUTLASS_PRAGMA_UNROLL
        for (int offset = 4; offset <= 16; offset <<= 1) {
          maximum0 = fmaxf(maximum0, __shfl_xor_sync(0xffffffffu, maximum0, offset));
          maximum1 = fmaxf(maximum1, __shfl_xor_sync(0xffffffffu, maximum1, offset));
        }
        if (lane_idx < 4) {
          indexer.page_partials[page][2 * thread_group][score_warp] = maximum0;
          indexer.page_partials[page][2 * thread_group + 1][score_warp] = maximum1;
        }
      };

      auto compute_page_pair = [&](int page0, PageFragments const &fragments0, int page1,
                                   PageFragments const &fragments1) {
        float accumulator0[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        float accumulator1[4] = {0.0F, 0.0F, 0.0F, 0.0F};

        CUTLASS_PRAGMA_UNROLL
        for (int k_chunk = 0; k_chunk < 4; ++k_chunk) {
          int const query_offset = query_column * kHeadDim + 32 * k_chunk + 8 * thread_group;
          uint2 const query_fragment =
              *reinterpret_cast<uint2 const *>(indexer.q_tile + query_offset);
          uint32_t matrix0[4];
          uint32_t const scale0_a =
              broadcast_scale_byte(fragments0.scale_a, 2 * k_chunk + (thread_group >> 1));
          uint32_t const scale0_b =
              broadcast_scale_byte(fragments0.scale_b, 2 * k_chunk + (thread_group >> 1));
          matrix0[0] = qmul4(static_cast<uint16_t>(fragments0.matrix_a[k_chunk]), scale0_a);
          matrix0[2] = qmul4(static_cast<uint16_t>(fragments0.matrix_a[k_chunk] >> 16), scale0_a);
          matrix0[1] = qmul4(static_cast<uint16_t>(fragments0.matrix_b[k_chunk]), scale0_b);
          matrix0[3] = qmul4(static_cast<uint16_t>(fragments0.matrix_b[k_chunk] >> 16), scale0_b);
          hmma_e4m3(accumulator0, matrix0, query_fragment);

          uint32_t matrix1[4];
          uint32_t const scale1_a =
              broadcast_scale_byte(fragments1.scale_a, 2 * k_chunk + (thread_group >> 1));
          uint32_t const scale1_b =
              broadcast_scale_byte(fragments1.scale_b, 2 * k_chunk + (thread_group >> 1));
          matrix1[0] = qmul4(static_cast<uint16_t>(fragments1.matrix_a[k_chunk]), scale1_a);
          matrix1[2] = qmul4(static_cast<uint16_t>(fragments1.matrix_a[k_chunk] >> 16), scale1_a);
          matrix1[1] = qmul4(static_cast<uint16_t>(fragments1.matrix_b[k_chunk]), scale1_b);
          matrix1[3] = qmul4(static_cast<uint16_t>(fragments1.matrix_b[k_chunk] >> 16), scale1_b);
          hmma_e4m3(accumulator1, matrix1, query_fragment);
        }

        int const query = kv_length - kRows + 2 * thread_group;
        int const logical_page0 = page_begin + page0;
        if (logical_page0 >= mask_page) {
          int const token = logical_page0 * kPageTokens + token_begin + (lane_idx >> 2);
          if (token > query) {
            accumulator0[0] = -1.0e30F;
          }
          if (token > query + 1) {
            accumulator0[1] = -1.0e30F;
          }
          if (token + 8 > query) {
            accumulator0[2] = -1.0e30F;
          }
          if (token + 8 > query + 1) {
            accumulator0[3] = -1.0e30F;
          }
        }
        int const logical_page1 = page_begin + page1;
        if (logical_page1 >= mask_page) {
          int const token = logical_page1 * kPageTokens + token_begin + (lane_idx >> 2);
          if (token > query) {
            accumulator1[0] = -1.0e30F;
          }
          if (token > query + 1) {
            accumulator1[1] = -1.0e30F;
          }
          if (token + 8 > query) {
            accumulator1[2] = -1.0e30F;
          }
          if (token + 8 > query + 1) {
            accumulator1[3] = -1.0e30F;
          }
        }

        float maximum0_0 = fmaxf(accumulator0[0], accumulator0[2]);
        float maximum0_1 = fmaxf(accumulator0[1], accumulator0[3]);
        float maximum1_0 = fmaxf(accumulator1[0], accumulator1[2]);
        float maximum1_1 = fmaxf(accumulator1[1], accumulator1[3]);
        CUTLASS_PRAGMA_UNROLL
        for (int offset = 4; offset <= 16; offset <<= 1) {
          maximum0_0 = fmaxf(maximum0_0, __shfl_xor_sync(0xffffffffu, maximum0_0, offset));
          maximum0_1 = fmaxf(maximum0_1, __shfl_xor_sync(0xffffffffu, maximum0_1, offset));
          maximum1_0 = fmaxf(maximum1_0, __shfl_xor_sync(0xffffffffu, maximum1_0, offset));
          maximum1_1 = fmaxf(maximum1_1, __shfl_xor_sync(0xffffffffu, maximum1_1, offset));
        }
        if (lane_idx < 4) {
          indexer.page_partials[page0][2 * thread_group][score_warp] = maximum0_0;
          indexer.page_partials[page0][2 * thread_group + 1][score_warp] = maximum0_1;
          indexer.page_partials[page1][2 * thread_group][score_warp] = maximum1_0;
          indexer.page_partials[page1][2 * thread_group + 1][score_warp] = maximum1_1;
        }
      };

      auto arrive_consumed = [&](int slot) {
        if (cute::elect_one_sync()) {
          arrive(indexer.page_consumed_barriers + slot);
        }
      };

      auto refill_page = [&](int page) {
        int const owner = (page >> 1) & (kScoreWarps - 1);
        if (page < local_pages && score_warp == owner) {
          if constexpr (Traits::kIndexerTopKItemsPerThread == 16) {
            if (cute::elect_one_sync()) {
              issue_physical_page_load(params, indexer, page, owned_refill_physical_page,
                                       page_begin + page, page_count);
            }
          } else if constexpr (Traits::kIndexerTopKItemsPerThread == 32) {
            int const first_owned_page = kMaxPagesPerCta + 2 * score_warp + score_group;
            int const refill_ordinal = (page - first_owned_page) / kMaxPagesPerCta;
            int const physical_page =
                __shfl_sync(0xffffffffu, owned_refill_physical_page, refill_ordinal);
            if (cute::elect_one_sync()) {
              issue_physical_page_load(params, indexer, page, physical_page, page_begin + page,
                                       page_count);
            }
          } else {
            if (cute::elect_one_sync()) {
              issue_page_load(params, indexer, page_begin, page, page_count);
            }
          }
        }
      };

      for (int page = score_group; page < local_pages; page += 2 * kScoreGroups) {
        int const next_page = page + kScoreGroups;
        int const slot = page & (kMaxPagesPerCta - 1);
        PageFragments fragments0;
        load_page(slot, static_cast<uint32_t>(page / kMaxPagesPerCta) & 1u, fragments0);
        if (next_page < local_pages) {
          int const next_slot = next_page & (kMaxPagesPerCta - 1);
          PageFragments fragments1;
          load_page(next_slot, static_cast<uint32_t>(next_page / kMaxPagesPerCta) & 1u, fragments1);
          if constexpr (kNeedsPageRefill) {
            if (use_page_refill) {
              if constexpr (Traits::kIndexerTopKItemsPerThread != 16) {
                __syncwarp();
              }
              arrive_consumed(slot);
              arrive_consumed(next_slot);
              refill_page(page + kMaxPagesPerCta);
              refill_page(next_page + kMaxPagesPerCta);
            }
          }
          compute_page_pair(page, fragments0, next_page, fragments1);
        } else {
          if constexpr (kNeedsPageRefill) {
            if (use_page_refill) {
              if constexpr (Traits::kIndexerTopKItemsPerThread != 16) {
                __syncwarp();
              }
              arrive_consumed(slot);
              refill_page(page + kMaxPagesPerCta);
            }
          }
          compute_page(page, fragments0);
        }
      }
    }

    __syncthreads();
    if (page_count > kTopK && is_producer) {
      if (thread_idx < kMaxPagesPerCta) {
        cutlass::arch::ClusterBarrier::invalidate(indexer.page_barriers + thread_idx);
      } else if constexpr (kNeedsPageRefill) {
        if (use_page_refill && thread_idx < 2 * kMaxPagesPerCta) {
          cutlass::arch::ClusterBarrier::invalidate(indexer.page_consumed_barriers + thread_idx -
                                                    kMaxPagesPerCta);
        }
      }
    }
    __syncthreads();
    if constexpr (kUseCtaCandidateHandoff) {
      publish_cta_candidates(params, indexer, page_begin, local_pages, kv_length, warp_idx,
                             lane_idx);
    } else if (is_producer && thread_idx < 128) {
      int const row = thread_idx >> 4;
      int const page_group = thread_idx & 15;
      for (int page = page_group; page < local_pages; page += 16) {
        score_row(params, row, kv_length)[page_begin + page] = fold_page_score(indexer, page, row);
      }
    }

    if constexpr (kUseCtaCandidateHandoff) {
      if (warp_idx < kRows) {
        Sm100M3DecodeNamedBarrier::sync(kRows * cutlass::NumThreadsPerWarp, kTopKBarrierId);
      }
    } else if (is_producer && thread_idx < 128) {
      Sm100M3DecodeNamedBarrier::sync(128, kScorePublishBarrierId);
    }
    if (attention_cluster && !Standalone) {
      storage.attention_pipelines().init_pipeline_barriers(warp_idx, lane_idx);
      if (thread_idx == 0) {
        cutlass::arch::ClusterTransactionBarrier::init(storage.cluster_reduction_barrier_ptr(), 1);
      }
      cutlass::arch::fence_barrier_init();
      if (thread_idx == 0) {
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
            storage.cluster_reduction_barrier_ptr(), Traits::kClusterReductionTransactionBytes);
      }
    }
    if (thread_idx == 0) {
      publish_scores(params);
    }

    if (attention_cluster) {
      if (cluster_rank == 0 && thread_idx == 0) {
        uint64_t const epoch = wait_generation(params, generation);
        wait_all_scores(params, epoch);
      }
      if (cluster_rank == 0) {
        if constexpr (Traits::kIndexerTopKItemsPerThread == 16) {
          __syncthreads();
        } else if (warp_idx < kTopKWarps) {
          Sm100M3DecodeNamedBarrier::sync(kTopKThreads, kTopKBarrierId);
        }
        if constexpr (kUnifiedTopK) {
          int const visible_pages = ((kv_length - kRows + attention_row) / kPageTokens) + 1;
          runtime_topk::select(score_row(params, attention_row, kv_length), visible_pages,
                               indexer.runtime_topk, storage.indexer_topk);
          if (warp_idx == 0 && lane_idx < kTopK && params.kv_block_indexes_ptr != nullptr) {
            const_cast<int *>(params.kv_block_indexes_ptr)[attention_row * kTopK + lane_idx] =
                storage.indexer_topk[lane_idx];
          }
        } else if constexpr (kUseCtaCandidateHandoff) {
          select_topk_cta_candidates(indexer, storage, params, attention_row, kv_length, thread_idx,
                                     warp_idx, lane_idx);
        } else {
          select_topk_quantized(indexer, storage, params, attention_row, kv_length, thread_idx,
                                warp_idx, lane_idx);
        }
      }
      if (retain_tmem && thread_idx == 0) {
        wait(&indexer.allocation_barrier, 0);
        storage.tmem_state_ptr()[0] = indexer.tmem_base;
        cutlass::arch::ClusterBarrier::invalidate(&indexer.allocation_barrier);
      }
      if constexpr (!Standalone) {
        publish_topk(storage, cluster_size, cluster_rank, thread_idx, lane_idx);
      }
    } else if (retain_tmem) {
      if (thread_idx == 0) {
        wait(&indexer.allocation_barrier, 0);
        storage.tmem_state_ptr()[0] = indexer.tmem_base;
        cutlass::arch::ClusterBarrier::invalidate(&indexer.allocation_barrier);
      }
      __syncthreads();
    }
  }
};

} // namespace cutlass::m3_decode::collective
