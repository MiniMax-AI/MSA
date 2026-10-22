#pragma once

#include "attn/kernel/include/sm100_m3_decode_attention_correction_reduction.hpp"

namespace cutlass::m3_decode::collective {

template <class Traits>
struct Sm100M3DecodeAttentionOutputEpilogue : Sm100M3DecodeAttentionCorrectionReduction<Traits> {
  using Base = Sm100M3DecodeAttentionCorrectionReduction<Traits>;
  using Base::complete_cluster_reduction_transactions;
  using Base::copy_o_smem_to_cluster;
  using Base::float_as_uint;
  using Base::full_phase;
  using Base::kv_tile_count;
  using Base::load_o_regs;
  using Base::load_stats;
  using Base::make_rescale;
  using Base::o_empty_barrier;
  using Base::o_full_barrier;
  using Base::pack_scaled_o_bf16;
  using Base::reduce_final_sum;
  using Base::scale_o_regs;
  using Base::softmax_local_empty_barrier;
  using Base::softmax_local_full_barrier;
  using Base::store_neutral_to_cluster;
  using Base::store_o_regs;
  using Base::store_o_sync;
  using Base::store_stats_to_cluster;
  using Base::uint_as_float;
  using typename Base::Bf16OFragment;
  using typename Base::OFragment;
  using typename Base::OutputAccumulatorFragment;
  using typename Base::Params;
  using typename Base::ScaleFragment;
  using typename Base::State;
  using typename Base::StatsFragment;
  using typename Base::Storage;
  CUTLASS_DEVICE static float2 bf16x2_to_float2(uint32_t raw) {
    uint32_t const lo = raw & 0xffffu;
    uint32_t const hi = raw >> 16;
    return make_f32x2(__uint_as_float(lo << 16), __uint_as_float(hi << 16));
  }

  CUTLASS_DEVICE static float mxfp8_trunc_abs_float_to_pow2(float value) {
    constexpr uint32_t kExponentMask = 0x7f800000u;
    return __uint_as_float(__float_as_uint(value) & kExponentMask);
  }

  CUTLASS_DEVICE static float mxfp8_scale_rcp_exp_only(float value) {
    uint32_t const bits = 0x7f000000u - __float_as_uint(value);
    return __uint_as_float(bits);
  }

  CUTLASS_DEVICE static void compute_mxe4m3_sf_and_output_scale(float &output_scale,
                                                                uint8_t &sf_out, float amax) {
    float const amax_pow2 = mxfp8_trunc_abs_float_to_pow2(amax);
    float const sf_val = amax_pow2 * (1.f / 256.f);
    cutlass::Array<float, 1> sf_array;
    sf_array[0] = sf_val;
    using Converter =
        cutlass::NumericArrayConverter<cutlass::float_ue8m0_t, float, 1,
                                       cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
    cutlass::float_ue8m0_t const sf = Converter::convert(sf_array)[0];
    sf_out = sf.storage;
    output_scale = sf_val != 0.f ? mxfp8_scale_rcp_exp_only(sf_val) : 0.f;
  }

  CUTLASS_DEVICE static int64_t mxfp8_sf_offset(int64_t row, int sf_col, int num_sf_per_row) {
    // Layout128x4: [mTile, kTile, outerM(32), innerM(4), innerK(4)].
    constexpr int kRowsPerTile = 128;
    constexpr int kSfColumnsPerTile = 4;
    constexpr int kBytesPerTile = 512;
    int const num_k_tiles = (num_sf_per_row + kSfColumnsPerTile - 1) / kSfColumnsPerTile;
    int64_t const tile_row = row / kRowsPerTile;
    int const row_in_tile = static_cast<int>(row % kRowsPerTile);
    int const outer_m = row_in_tile & 31;
    int const inner_m = row_in_tile >> 5;
    int const k_tile = sf_col >> 2;
    int const inner_k = sf_col & 3;
    return tile_row * num_k_tiles * kBytesPerTile + k_tile * kBytesPerTile + outer_m * 16 +
           inner_m * 4 + inner_k;
  }

  CUTLASS_DEVICE static float float_fragment_local_amax(OutputAccumulatorFragment const &values) {
    float local_amax = 0.f;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      local_amax = fmaxf(local_amax, fabsf(values[i]));
    }
    return local_amax;
  }

  CUTLASS_DEVICE static void
  store_mxfp8_vec_from_float_fragment(OutputAccumulatorFragment const &values, float local_amax,
                                      Params const &params, int64_t linear_row, int col,
                                      int warp_group_lane) {
    static_assert(Traits::kHeadDim == 128, "MXFP8 O store is specialized for D=128.");

    uint32_t const mask = __activemask();
    float group_amax = local_amax;
    group_amax = fmaxf(group_amax, __shfl_xor_sync(mask, group_amax, 1));
    group_amax = fmaxf(group_amax, __shfl_xor_sync(mask, group_amax, 2));

    uint8_t sf = 0;
    float output_scale = 0.f;
    compute_mxe4m3_sf_and_output_scale(output_scale, sf, group_amax);

    cutlass::Array<float, 8> scaled;
    float2 const output_scale2 = make_f32x2(output_scale, output_scale);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      float2 const vals = fmul2(make_f32x2(values[2 * i + 0], values[2 * i + 1]), output_scale2);
      scaled[2 * i + 0] = vals.x;
      scaled[2 * i + 1] = vals.y;
    }
    uint2 const out_vec = pack_float8_to_e4m3x8(scaled);

    uint8_t *o_base =
        reinterpret_cast<uint8_t *>(params.o_ptr) + linear_row * Traits::kHeadDim + col;
    *reinterpret_cast<uint2 *>(o_base) = out_vec;

    int const group_lane = (warp_group_lane & 3);
    if (group_lane == 0) {
      int const sf_col = col / 32;
      uint8_t *sf_base = reinterpret_cast<uint8_t *>(params.o_sf_ptr);
      constexpr int kOutputProjectionHeads = Traits::kHeadGroup;
      constexpr int kPaddedSfbStageBytes = 512;
      int const q_token = static_cast<int>(linear_row) / kOutputProjectionHeads;
      int const k_stage = static_cast<int>(linear_row) % kOutputProjectionHeads;
      sf_base[k_stage * kPaddedSfbStageBytes + q_token * 16 + sf_col] = sf;
    }
  }

  CUTLASS_DEVICE static void reduce_cluster_partials_to_global(Storage &storage,
                                                               Params const &params, int batch_idx,
                                                               int kv_head_idx, int q_token_idx,
                                                               int warp_group_lane) {
    constexpr int kElementsPerVec = 8;
    constexpr int kDstRowBytes = Traits::kHeadDim * static_cast<int>(sizeof(cutlass::bfloat16_t));
    static_assert(kElementsPerVec * static_cast<int>(sizeof(cutlass::bfloat16_t)) == 16,
                  "cluster reduction threads store one 16B vector.");

    int const rank = static_cast<int>(cute::block_rank_in_cluster());
    constexpr int rows_per_rank = Traits::kHeadGroup / Traits::kNumKvSplits;
    int const local_row = warp_group_lane / (Traits::kHeadDim / kElementsPerVec);
    int const col =
        (warp_group_lane & ((Traits::kHeadDim / kElementsPerVec) - 1)) * kElementsPerVec;
    int const row = rank * rows_per_rank + local_row;
    int const head_base = kv_head_idx * Traits::kHeadGroup;
    int const remaining_heads = params.num_qo_heads_orig - head_base;
    int const valid_rows =
        remaining_heads < Traits::kHeadGroup ? remaining_heads : Traits::kHeadGroup;
    if (rank >= Traits::kNumKvSplits || local_row >= rows_per_rank || row >= valid_rows) {
      return;
    }

    OutputAccumulatorFragment out_vals = {{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}};
    float running_sum = 0.f;
    float global_max = -FLT_MAX;

    CUTLASS_PRAGMA_UNROLL
    for (int split = 0; split < Traits::kNumKvSplits; ++split) {
      int const stats_offset =
          (split * rows_per_rank + local_row) * Traits::kClusterReductionStatsPerRow;
      float const local_max = storage.smem_cluster_reduction.stats[stats_offset + 0];
      global_max = fmaxf(global_max, local_max);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int split = 0; split < Traits::kNumKvSplits; ++split) {
      int const stats_offset =
          (split * rows_per_rank + local_row) * Traits::kClusterReductionStatsPerRow;
      float const local_max = storage.smem_cluster_reduction.stats[stats_offset + 0];
      float const local_sum = storage.smem_cluster_reduction.stats[stats_offset + 1];
      float const local_scale = exp2f(params.scale_softmax_log2 * (local_max - global_max));
      running_sum = fmaf(local_sum, local_scale, running_sum);

      int const src_offset = cluster_partial_o_offset(
          split, local_row, rows_per_rank, col * static_cast<int>(sizeof(cutlass::bfloat16_t)));
      uint4 const vec =
          *reinterpret_cast<uint4 const *>(storage.smem_cluster_reduction.partial_o + src_offset);
      float2 const local_scale2 = make_f32x2(local_scale, local_scale);
      float2 f2 = bf16x2_to_float2(vec.x);
      float2 accum = ffma2(f2, local_scale2, make_f32x2(out_vals[0], out_vals[1]));
      out_vals[0] = accum.x;
      out_vals[1] = accum.y;
      f2 = bf16x2_to_float2(vec.y);
      accum = ffma2(f2, local_scale2, make_f32x2(out_vals[2], out_vals[3]));
      out_vals[2] = accum.x;
      out_vals[3] = accum.y;
      f2 = bf16x2_to_float2(vec.z);
      accum = ffma2(f2, local_scale2, make_f32x2(out_vals[4], out_vals[5]));
      out_vals[4] = accum.x;
      out_vals[5] = accum.y;
      f2 = bf16x2_to_float2(vec.w);
      accum = ffma2(f2, local_scale2, make_f32x2(out_vals[6], out_vals[7]));
      out_vals[6] = accum.x;
      out_vals[7] = accum.y;
    }

    float const final_scale = running_sum > 0.f ? params.scale_o / running_sum : 0.f;
    OutputAccumulatorFragment final_vals;
    float2 const final_scale2 = make_f32x2(final_scale, final_scale);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerVec / 2; ++i) {
      float2 const vals = fmul2(make_f32x2(out_vals[2 * i + 0], out_vals[2 * i + 1]), final_scale2);
      final_vals[2 * i + 0] = vals.x;
      final_vals[2 * i + 1] = vals.y;
    }

    int const qo_offset = m3_decode_q_token_global_index<Traits>(params, batch_idx, q_token_idx);
    int64_t const linear_row =
        static_cast<int64_t>(qo_offset) * params.num_qo_heads_orig + head_base + row;
    float const local_amax = float_fragment_local_amax(final_vals);
    store_mxfp8_vec_from_float_fragment(final_vals, local_amax, params, linear_row, col,
                                        warp_group_lane);
  }

  CUTLASS_DEVICE static void store_o_to_global(Storage &storage, Params const &params,
                                               int kv_split_idx, OFragment const &regs_o,
                                               ScaleFragment const &scale, int warp_group_lane) {
    Bf16OFragment regs_bf16;
    pack_scaled_o_bf16(regs_o, scale, regs_bf16);
    store_transposed_smem_16b_128x16(storage.smem_o.data, regs_bf16.data, warp_group_lane);
    store_o_sync();
    copy_o_smem_to_cluster(storage, kv_split_idx, warp_group_lane);
  }

  CUTLASS_DEVICE void run_tile(Storage &storage, Params const &params, int batch_idx,
                               int kv_head_idx, int q_token_idx, int lane_idx,
                               int warp_group_warp_idx, State &state, int full_tiles,
                               int kv_tile_begin = 0, int kv_tile_end = INT_MAX,
                               int kv_split_idx = 0) const {
    Sm100M3DecodeKvTileRange const tile_range =
        make_kv_tile_range(full_tiles, kv_tile_begin, kv_tile_end);
    int const tiles = tile_range.count;
    int const warp_group_lane = warp_group_warp_idx * cutlass::NumThreadsPerWarp + lane_idx;
    if (tiles <= 0) {
      store_neutral_to_cluster(storage, kv_split_idx, warp_group_lane);
      complete_cluster_reduction_transactions(storage, warp_group_lane);
      Sm100M3DecodeBarrier::wait(storage.cluster_reduction_barrier_ptr(), 0, 350);
      reduce_cluster_partials_to_global(storage, params, batch_idx, kv_head_idx, q_token_idx,
                                        warp_group_lane);
      return;
    }

    uint32_t volatile *tmem_state = storage.tmem_state_ptr();
    uint32_t const tmem_base = tmem_state[0];
    float const scale_softmax_log2 = params.scale_softmax_log2;

    int const first_event = state.softmax_event;
    int const first_stage = first_event & 1;
    Sm100M3DecodeBarrier::wait(softmax_local_full_barrier(storage, first_stage),
                               full_phase(first_event, 2),
                               static_cast<uint32_t>(300 + first_stage));
    StatsFragment stats;
    Sm100M3DecodeBarrier::arrive_cluster_zero(softmax_local_empty_barrier(storage, first_stage));

    CUTLASS_PRAGMA_NO_UNROLL
    for (int tile = 0; tile + 1 < tiles; ++tile) {
      int const stats_event = state.softmax_event + tile + 1;
      int const stats_stage = stats_event & 1;

      Sm100M3DecodeBarrier::wait(softmax_local_full_barrier(storage, stats_stage),
                                 full_phase(stats_event, 2),
                                 static_cast<uint32_t>(310 + stats_stage));
      load_stats(tmem_base, stats_event, stats);
      Sm100M3DecodeBarrier::arrive_cluster_zero(softmax_local_empty_barrier(storage, stats_stage));

      Sm100M3DecodeBarrier::wait(o_full_barrier(storage), full_phase(state.o_event + tile, 1), 302);

      ScaleFragment scale;
      OFragment regs_o;
      make_rescale(stats, scale_softmax_log2, scale);
      load_o_regs(tmem_base, regs_o);
      scale_o_regs(regs_o, scale);
      store_o_regs(tmem_base, regs_o);

      Sm100M3DecodeBarrier::arrive(o_empty_barrier(storage));
    }

    int const final_event = state.softmax_event + tiles;
    int const final_stage = final_event & 1;
    Sm100M3DecodeBarrier::wait(softmax_local_full_barrier(storage, final_stage),
                               full_phase(final_event, 2),
                               static_cast<uint32_t>(330 + final_stage));
    load_stats(tmem_base, final_event, stats);
    Sm100M3DecodeBarrier::arrive_cluster_zero(softmax_local_empty_barrier(storage, final_stage));

    Sm100M3DecodeBarrier::wait(o_full_barrier(storage), full_phase(state.o_event + tiles - 1, 1),
                               304);

    ScaleFragment total_sum;
    ScaleFragment final_scale;
    OFragment regs_o;
    ScaleFragment partial_sum{{stats[0], stats[1], stats[2], stats[3]}};
    reduce_final_sum(storage, partial_sum, total_sum, warp_group_lane);
    load_o_regs(tmem_base, regs_o);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      final_scale[i] = params.scale_v;
    }
    store_o_to_global(storage, params, kv_split_idx, regs_o, final_scale, warp_group_lane);

    store_stats_to_cluster(storage, kv_split_idx, stats, total_sum, warp_group_lane);
    complete_cluster_reduction_transactions(storage, warp_group_lane);
    Sm100M3DecodeBarrier::wait(storage.cluster_reduction_barrier_ptr(), 0, 350);
    reduce_cluster_partials_to_global(storage, params, batch_idx, kv_head_idx, q_token_idx,
                                      warp_group_lane);

    Sm100M3DecodeBarrier::arrive(o_empty_barrier(storage));
    state.softmax_event += tiles + 1;
    state.o_event += tiles;
  }
};

} // namespace cutlass::m3_decode::collective
