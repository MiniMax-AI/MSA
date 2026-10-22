#pragma once

#include <climits>
#include <cstdint>

#include "attn/kernel/include/sm100_m3_decode_problem.hpp"
#include "common/kernel/include/sm100_m3_decode_fp4_transform.cuh"
#include "common/kernel/include/sm100_m3_decode_pipeline.hpp"
#include "common/kernel/include/sm100_m3_decode_storage.hpp"
#include "common/kernel/include/sm100_m3_decode_traits.hpp"
#include "cutlass/cutlass.h"

namespace cutlass::m3_decode::collective {

template <class Traits> struct Sm100M3DecodeAttentionKvTransform {
  using Storage = SharedStorage<Traits>;
  using Params = Sm100M3DecodeKernelParams<Traits>;
  using Tmem = TmemLayout<Traits>;
  using Barriers = BarrierLayout<Traits>;
  static constexpr int kRawKvReleaseBarrierId = 6;
  static constexpr int kScaleReorderBarrierId = 5;

  struct State {
    int raw_stage = 0;
    uint32_t raw_full_phase = 0;
    uint32_t raw_empty_phase = 0;
    int transformed_stage = 0;
    uint32_t transformed_full_phase = 0;
    uint32_t transformed_empty_phase = 1;
  };

  struct ScaleFragment {
    uint32_t data[4][2];

    CUTLASS_DEVICE uint32_t &operator()(int row, int word) { return data[row][word]; }

    CUTLASS_DEVICE uint32_t const &operator()(int row, int word) const { return data[row][word]; }

    CUTLASS_DEVICE void load_row(int row, uint64_t value) {
      reinterpret_cast<uint64_t &>(data[row]) = value;
    }
  };

  CUTLASS_DEVICE static void pack_scale_pairs(uint32_t &packed0, uint32_t &packed1, uint32_t h0,
                                              uint32_t h1, uint32_t h2, uint32_t h3) {
    uint32_t const pair01 = (h0 & 0xffffu) | ((h1 & 0xffffu) << 16);
    uint32_t const pair23 = (h2 & 0xffffu) | ((h3 & 0xffffu) << 16);
    asm volatile("prmt.b32 %0, %2, %3, 0x6420;\n"
                 "prmt.b32 %1, %2, %3, 0x7531;\n"
                 : "=&r"(packed0), "=&r"(packed1)
                 : "r"(pair01), "r"(pair23));
  }

  CUTLASS_DEVICE static uint32_t tmem_stage_col(int stage_idx, int loop_offset) {
    return Tmem::kPreparedKv +
           static_cast<uint32_t>(stage_idx * Tmem::kColsPerPreparedKvStage + loop_offset * 4);
  }

  CUTLASS_DEVICE static uint64_t *kv_full_barrier(Storage &storage, int stage) {
    return storage.attention_pipelines().ptr(Barriers::kKvFullArv1 + stage);
  }

  CUTLASS_DEVICE static uint64_t *kv_empty_barrier(Storage &storage, int stage) {
    return storage.attention_pipelines().ptr(Barriers::kKvEmptyArv1 + stage);
  }

  CUTLASS_DEVICE static uint64_t *transformed_full_barrier(Storage &storage, int stage) {
    return storage.attention_pipelines().ptr(Barriers::kTransformedKvFullArv128 + stage);
  }

  CUTLASS_DEVICE static uint64_t *transformed_empty_barrier(Storage &storage, int stage) {
    return storage.attention_pipelines().ptr(Barriers::kTransformedKvEmptyArv1 + stage);
  }

  CUTLASS_DEVICE static int kv_length_for_batch(Params const &params, int batch_idx,
                                                int kv_head_idx, int q_token_idx) {
    return m3_decode_kv_length_for_batch<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
  }

  CUTLASS_DEVICE static int kv_tile_count(Params const &params, int batch_idx, int kv_head_idx,
                                          int q_token_idx) {
    return m3_decode_kv_tile_count_for_batch<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
  }

  CUTLASS_DEVICE static void sync_transform_warpgroup_before_raw_release() {
    Sm100M3DecodeNamedBarrier::sync(128, kRawKvReleaseBarrierId);
  }

  template <int Stages>
  CUTLASS_DEVICE static void advance_pipeline_state(int &stage, uint32_t &full_phase,
                                                    uint32_t &empty_phase) {
    ++stage;
    if (stage == Stages) {
      stage = 0;
      full_phase ^= 1u;
      empty_phase ^= 1u;
    }
  }

  CUTLASS_DEVICE void transform_k_stage(uint32_t tmem_base, int stage_idx, int lane_idx,
                                        int warp_group_warp_idx,
                                        uint8_t const *smem_stage_base) const {
    uint8_t const *raw_stage_base = smem_stage_base;
    uint8_t const *scale_stage_base = smem_stage_base + Traits::kRawKvDataBytesPerStage;

    int const warp_group_lane = warp_group_warp_idx * cutlass::NumThreadsPerWarp + lane_idx;
    int const swizzle_mask = (warp_group_lane & 7) * 16;
    uint8_t const *raw_lane_base = raw_stage_base + warp_group_lane * Traits::kHeadDim;
    uint8_t const *scale_lane_base =
        scale_stage_base + ((lane_idx / 4) + warp_group_warp_idx * cutlass::NumThreadsPerWarp) * 8;

    ScaleFragment scale;
    CUTLASS_PRAGMA_UNROLL
    for (int row = 0; row < 4; ++row) {
      scale.load_row(row, reinterpret_cast<uint64_t const *>(scale_lane_base)[row * 8]);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int loop_offset = 0; loop_offset < 8; ++loop_offset) {
      uint32_t src0, src1, src2, src3;
      uint32_t dst0, dst1, dst2, dst3;

      int const raw_offset0 = (loop_offset * 16) ^ swizzle_mask;
      int const raw_offset1 = (2048 + loop_offset * 16) ^ swizzle_mask;
      ldsm_unpack_fp4_8x16_x2(src0, src1, raw_lane_base + raw_offset0);
      ldsm_unpack_fp4_8x16_x2(src2, src3, raw_lane_base + raw_offset1);

      int const scale_word = loop_offset / 4;
      int const scale_byte = loop_offset & 3;
      convert_e2m1x4_pair_to_e4m3x4_select_byte(dst0, dst1, src0, src1, scale(0, scale_word),
                                                scale(1, scale_word), scale_byte);
      convert_e2m1x4_pair_to_e4m3x4_select_byte(dst2, dst3, src2, src3, scale(2, scale_word),
                                                scale(3, scale_word), scale_byte);

      uint32_t const tmem_col = tmem_base + tmem_stage_col(stage_idx, loop_offset);
      tmem_store_16x128b(tmem_col, dst0, dst1);
      tmem_store_16x128b(tmem_col + 0x100000u, dst2, dst3);
    }
    fence_tmem_store();
  }

  CUTLASS_DEVICE void prepare_v_scale_stage(int lane_idx, int warp_group_warp_idx,
                                            uint8_t *smem_stage_base) const {
    uint8_t const *scale_stage_base = smem_stage_base + Traits::kRawKvDataBytesPerStage;
    uint8_t *scale_scratch_base =
        smem_stage_base + Traits::kRawKvDataBytesPerStage + Traits::kRawKvScaleBytesPerStage;
    int const warp_group_lane = warp_group_warp_idx * cutlass::NumThreadsPerWarp + lane_idx;
    int const scale_pair = warp_group_lane & 3;
    int const token_quad = (warp_group_lane >> 2) & 3;
    int const loop_offset = (warp_group_lane >> 4) & 7;
    uint8_t const *scale_src =
        scale_stage_base + loop_offset * 128 + token_quad * 32 + scale_pair * 2;
    uint32_t const h0 = *reinterpret_cast<uint16_t const *>(scale_src + 0);
    uint32_t const h1 = *reinterpret_cast<uint16_t const *>(scale_src + 8);
    uint32_t const h2 = *reinterpret_cast<uint16_t const *>(scale_src + 16);
    uint32_t const h3 = *reinterpret_cast<uint16_t const *>(scale_src + 24);
    uint32_t packed0;
    uint32_t packed1;
    pack_scale_pairs(packed0, packed1, h0, h1, h2, h3);
    int const scale_dst_offset = loop_offset * 128 + token_quad * 32 + scale_pair * 8;
    *reinterpret_cast<uint32_t *>(scale_scratch_base + scale_dst_offset) = packed0;
    *reinterpret_cast<uint32_t *>(scale_scratch_base + scale_dst_offset + 4) = packed1;
    Sm100M3DecodeNamedBarrier::sync(128, kScaleReorderBarrierId);
  }

  CUTLASS_DEVICE void transform_v_stage(uint32_t tmem_base, int stage_idx, int lane_idx,
                                        int warp_group_warp_idx,
                                        uint8_t const *smem_stage_base) const {
    uint8_t const *raw_stage_base = smem_stage_base;
    uint8_t const *scale_scratch_base =
        smem_stage_base + Traits::kRawKvDataBytesPerStage + Traits::kRawKvScaleBytesPerStage;

    int const swizzle_mask = (lane_idx & 7) * 16;
    int const warp_col = warp_group_warp_idx * cutlass::NumThreadsPerWarp;
    int const swizzled_col0 = warp_col ^ swizzle_mask;
    int const swizzled_col1 = (warp_col + 16) ^ swizzle_mask;
    uint8_t const *raw_lane_base = raw_stage_base + lane_idx * Traits::kHeadDim;
    uint8_t const *scale_reordered_lane_base =
        scale_scratch_base + (lane_idx & 3) * 32 + warp_group_warp_idx * 8;

    CUTLASS_PRAGMA_UNROLL
    for (int loop_offset = 0; loop_offset < 8; ++loop_offset) {
      uint32_t src0, src1, src2, src3;
      uint32_t dst0, dst1, dst2, dst3;
      uint32_t scale_word0;
      uint32_t scale_word1;
      uint32_t const *scale_words =
          reinterpret_cast<uint32_t const *>(scale_reordered_lane_base + loop_offset * 128);
      scale_word0 = scale_words[0];
      scale_word1 = scale_words[1];

      int const row_offset = (loop_offset * 16) * Traits::kHeadDim;
      int const raw_offset0 = row_offset + swizzled_col0;
      int const raw_offset1 = row_offset + swizzled_col1;
      ldsm_unpack_fp4_transpose_16x16_x1(src0, src1, raw_lane_base + raw_offset0);
      ldsm_unpack_fp4_transpose_16x16_x1(src2, src3, raw_lane_base + raw_offset1);

      convert_e2m1x4_pair_to_e4m3x4(dst0, dst1, src0, src1, scale_word0);
      convert_e2m1x4_pair_to_e4m3x4(dst2, dst3, src2, src3, scale_word1);

      uint32_t const tmem_col = tmem_base + tmem_stage_col(stage_idx, loop_offset);
      tmem_store_16x128b(tmem_col, dst0, dst1);
      tmem_store_16x128b(tmem_col + 0x100000u, dst2, dst3);
    }
    fence_tmem_store();
  }

  CUTLASS_DEVICE void transform_k(Storage &storage, uint32_t tmem_base, int raw_stage,
                                  int transformed_stage, uint32_t raw_full_phase,
                                  uint32_t raw_empty_phase, uint32_t transformed_full_phase,
                                  uint32_t transformed_empty_phase, int lane_idx,
                                  int warp_group_warp_idx) const {
    Sm100M3DecodeBarrier::wait(kv_full_barrier(storage, raw_stage), raw_full_phase,
                               static_cast<uint32_t>(400 + raw_stage));
    Sm100M3DecodeBarrier::wait(transformed_empty_barrier(storage, transformed_stage),
                               transformed_empty_phase,
                               static_cast<uint32_t>(420 + transformed_stage));

    transform_k_stage(tmem_base, transformed_stage, lane_idx, warp_group_warp_idx,
                      storage.smem_kv.stage_ptr(raw_stage));

    Sm100M3DecodeBarrier::arrive(transformed_full_barrier(storage, transformed_stage));
    sync_transform_warpgroup_before_raw_release();
    if (warp_group_warp_idx == 0 && lane_idx == 0) {
      Sm100M3DecodeBarrier::arrive(kv_empty_barrier(storage, raw_stage));
    }
  }

  CUTLASS_DEVICE void transform_v(Storage &storage, uint32_t tmem_base, int raw_stage,
                                  int transformed_stage, uint32_t raw_full_phase,
                                  uint32_t raw_empty_phase, uint32_t transformed_full_phase,
                                  uint32_t transformed_empty_phase, int lane_idx,
                                  int warp_group_warp_idx) const {
    Sm100M3DecodeBarrier::wait(kv_full_barrier(storage, raw_stage), raw_full_phase,
                               static_cast<uint32_t>(440 + raw_stage));
    prepare_v_scale_stage(lane_idx, warp_group_warp_idx, storage.smem_kv.stage_ptr(raw_stage));
    Sm100M3DecodeBarrier::wait(transformed_empty_barrier(storage, transformed_stage),
                               transformed_empty_phase,
                               static_cast<uint32_t>(460 + transformed_stage));

    transform_v_stage(tmem_base, transformed_stage, lane_idx, warp_group_warp_idx,
                      storage.smem_kv.stage_ptr(raw_stage));

    Sm100M3DecodeBarrier::arrive(transformed_full_barrier(storage, transformed_stage));
    sync_transform_warpgroup_before_raw_release();
    if (warp_group_warp_idx == 0 && lane_idx == 0) {
      Sm100M3DecodeBarrier::arrive(kv_empty_barrier(storage, raw_stage));
    }
  }

  CUTLASS_DEVICE void run_tile(Storage &storage, Params const &params, int batch_idx,
                               int kv_head_idx, int q_token_idx, int lane_idx,
                               int warp_group_warp_idx, State &state, int full_tiles,
                               int kv_tile_begin = 0, int kv_tile_end = INT_MAX) const {
    Sm100M3DecodeKvTileRange const tile_range =
        make_kv_tile_range(full_tiles, kv_tile_begin, kv_tile_end);
    int const tiles = tile_range.count;
    if (tiles <= 0) {
      return;
    }

    uint32_t const tmem_base = storage.tmem_state_ptr()[0];

    // Transform schedule: prologue K0, steady K(i+1)+V(i), epilogue Vlast.
    transform_k(storage, tmem_base, state.raw_stage, state.transformed_stage, state.raw_full_phase,
                state.raw_empty_phase, state.transformed_full_phase, state.transformed_empty_phase,
                lane_idx, warp_group_warp_idx);
    advance_pipeline_state<Traits::kNumStagesRawKv>(state.raw_stage, state.raw_full_phase,
                                                    state.raw_empty_phase);
    advance_pipeline_state<Traits::kNumStagesTransform>(
        state.transformed_stage, state.transformed_full_phase, state.transformed_empty_phase);

    CUTLASS_PRAGMA_NO_UNROLL
    for (int tile = 0; tile + 1 < tiles; ++tile) {
      transform_k(storage, tmem_base, state.raw_stage, state.transformed_stage,
                  state.raw_full_phase, state.raw_empty_phase, state.transformed_full_phase,
                  state.transformed_empty_phase, lane_idx, warp_group_warp_idx);
      advance_pipeline_state<Traits::kNumStagesRawKv>(state.raw_stage, state.raw_full_phase,
                                                      state.raw_empty_phase);
      advance_pipeline_state<Traits::kNumStagesTransform>(
          state.transformed_stage, state.transformed_full_phase, state.transformed_empty_phase);

      transform_v(storage, tmem_base, state.raw_stage, state.transformed_stage,
                  state.raw_full_phase, state.raw_empty_phase, state.transformed_full_phase,
                  state.transformed_empty_phase, lane_idx, warp_group_warp_idx);
      advance_pipeline_state<Traits::kNumStagesRawKv>(state.raw_stage, state.raw_full_phase,
                                                      state.raw_empty_phase);
      advance_pipeline_state<Traits::kNumStagesTransform>(
          state.transformed_stage, state.transformed_full_phase, state.transformed_empty_phase);
    }

    transform_v(storage, tmem_base, state.raw_stage, state.transformed_stage, state.raw_full_phase,
                state.raw_empty_phase, state.transformed_full_phase, state.transformed_empty_phase,
                lane_idx, warp_group_warp_idx);
    advance_pipeline_state<Traits::kNumStagesRawKv>(state.raw_stage, state.raw_full_phase,
                                                    state.raw_empty_phase);
    advance_pipeline_state<Traits::kNumStagesTransform>(
        state.transformed_stage, state.transformed_full_phase, state.transformed_empty_phase);
  }

  CUTLASS_DEVICE void run_schedule(Storage &storage, Params const &params, int batch_idx,
                                   int kv_head_idx, int q_token_idx, int lane_idx,
                                   int warp_group_warp_idx, int kv_tile_begin = 0,
                                   int kv_tile_end = INT_MAX) const {
    State state;
    int const full_tiles = kv_tile_count(params, batch_idx, kv_head_idx, q_token_idx);
    run_tile(storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx, warp_group_warp_idx,
             state, full_tiles, kv_tile_begin, kv_tile_end);
  }
};

} // namespace cutlass::m3_decode::collective
