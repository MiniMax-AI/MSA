#pragma once

#include "attn/kernel/include/sm100_m3_decode_indexer_score_pipeline.hpp"

namespace cutlass::m3_decode::collective {

template <class Traits> struct Sm100M3DecodeIndexer : Sm100M3DecodeIndexerScorePipeline<Traits> {
  using Base = Sm100M3DecodeIndexerScorePipeline<Traits>;
  using Base::run_hmma;
  using typename Base::Params;
  using typename Base::Storage;
  CUTE_DEVICE static void run(Params const &params, Storage &storage, uint8_t *smem, int thread_idx,
                              int warp_idx, int lane_idx, int cluster_size, int cluster_rank,
                              bool attention_cluster, int attention_row, bool retain_tmem) {
    run_hmma(params, storage, smem, thread_idx, warp_idx, lane_idx, cluster_size, cluster_rank,
             attention_cluster, attention_row, retain_tmem);
  }

  CUTE_DEVICE static void run_standalone(Params const &params, Storage &storage, uint8_t *smem,
                                         int thread_idx, int warp_idx, int lane_idx,
                                         int cluster_size, int cluster_rank, bool selector_cluster,
                                         int selector_row) {
    run_hmma<true>(params, storage, smem, thread_idx, warp_idx, lane_idx, cluster_size,
                   cluster_rank, selector_cluster, selector_row, false);
  }
};

} // namespace cutlass::m3_decode::collective
