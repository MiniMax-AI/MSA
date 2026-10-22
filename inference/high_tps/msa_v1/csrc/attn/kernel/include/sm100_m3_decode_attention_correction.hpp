#pragma once

#include "attn/kernel/include/sm100_m3_decode_attention_output_epilogue.hpp"

namespace cutlass::m3_decode::collective {

template <class Traits>
struct Sm100M3DecodeAttentionCorrection : Sm100M3DecodeAttentionOutputEpilogue<Traits> {
  using Base = Sm100M3DecodeAttentionOutputEpilogue<Traits>;
  using Base::kv_tile_count;
  using Base::run_tile;
  using typename Base::Params;
  using typename Base::State;
  using typename Base::Storage;
  CUTLASS_DEVICE void operator()(Storage &storage, Params const &params, int batch_idx,
                                 int kv_head_idx, int q_token_idx, int lane_idx,
                                 int warp_group_warp_idx) const {
    State state;
    int const full_tiles = kv_tile_count(params, batch_idx, kv_head_idx, q_token_idx);
    run_tile(storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx, warp_group_warp_idx,
             state, full_tiles);
  }
};

} // namespace cutlass::m3_decode::collective
