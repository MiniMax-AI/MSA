// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <climits>
#include <cstdint>

#include "cutlass/cutlass.h"
#include "fmha_common.hpp"
#include "sm100_fmha_fp4_transform.cuh"
#include "sm100_fmha_pipeline.hpp"
#include "sm100_fmha_q8kv4_traits.hpp"
#include "sm100_fmha_selection_ring.hpp"
#include "sm100_fmha_storage.hpp"

namespace cutlass::fmha::collective {

template <class Traits> struct Sm100FmhaKvTransformTmaWarpspecialized {
  using Storage = SharedStorage<Traits>;
  using Params = Sm100FmhaFwdKernelParams<Traits>;
  using Tmem = TmemLayout<Traits>;
  using Barriers = BarrierLayout<Traits>;
  static constexpr int kRawKvReleaseBarrierId = 6;
  static constexpr int kRawKvReleaseVBarrierId = 13;
  static constexpr int kScaleReorderBarrierId = 5;
  static constexpr int kAssistRawKvReleaseVBarrierId = 8;
  static constexpr int kAssistScaleReorderBarrierId = 7;
  // Fraction of V tiles converted by the transform warpgroup: 1 in kTransformVTilePeriod.
  static constexpr int kTransformVTilePeriod = 4;
  // Consecutive uses of a raw V stage are one ring cycle (kNumStagesRawKv / 2 tiles) apart. When
  // the cycle is not a multiple of the period, those uses alternate between the transform and the
  // correction warpgroups. A parity wait on the stage's full barrier cannot tell "this phase is
  // complete" from "the previous phase is still pending" (phases p and p - 2 share a parity), so a
  // consumer that reaches its wait a full ring cycle before the other group has consumed the stage
  // passes early, reads stale smem and releases the stage while its TMA is in flight. With the
  // softmax-owned O rescale the correction warpgroup's loop has no PV coupling, so transform_v
  // waits for the TMEM stage release before the raw-full parity: the PV that freed the stage
  // consumed the previous V tile of the same raw stage, so that fill completed and the wait is at
  // most one phase ahead, for both consumer groups. The transform warpgroup's assist-V wait
  // depends only on K data and QK order, so run_sparse_k_tile additionally guards it by first
  // waiting for the previous use of the stage to be released (the loader's own condition for
  // issuing the event). K stages have a single consumer and are safe by in-order consumption.
  //
  // Ring depths. The TMEM-empty-before-raw-full order stands in for "the raw stage's previous fill
  // landed" only when the TMEM stage's previous user is at or after the raw stage's previous user,
  // i.e. kNumStagesTransform <= kNumStagesRawKv. When the TMEM ring is deeper (SM100/SM103: 8 raw,
  // 12 TMEM) transform_sparse_v_event pins on the raw stage's release instead
  // (kRawVStagesPinOnEmpty). The TMEM ring must never be shallower than the raw ring: a group's
  // TMEM-empty parity wait for event e is valid only once the previous user of that TMEM stage
  // (event e - kNumStagesTransform) has passed its own wait, and the raw ring bounds the other
  // group's lag to kNumStagesRawKv / 2 tiles, which covers kNumStagesTransform / 2 tiles only when
  // the TMEM ring is at least as deep.
  static_assert(Traits::kNumStagesRawKv % 2 == 0, "raw KV ring alternates K and V stages");
  static_assert(Traits::kNumStagesTransform >= Traits::kNumStagesRawKv,
                "the TMEM (transformed KV) ring must be at least as deep as the raw KV ring: a "
                "shallower TMEM ring lets one converter group lap the other and alias the "
                "transformed-empty parity");
  static constexpr bool kRawVStagesAlternateGroups =
      ((Traits::kNumStagesRawKv / 2) % kTransformVTilePeriod) != 0;
  static constexpr bool kRawVStagesPinOnEmpty =
      kRawVStagesAlternateGroups && Traits::kNumStagesTransform > Traits::kNumStagesRawKv;

  struct State {
    int raw_stage = 0;
    uint32_t raw_full_phase = 0;
    uint32_t raw_empty_phase = 0;
    int transformed_stage = 0;
    uint32_t transformed_full_phase = 0;
    uint32_t transformed_empty_phase = 1;
    int sparse_k_event = 0;
    int selection_event = 0;
  };

  struct VState {
    int event = 1;
  };

  struct ScaleFragment {
    uint32_t data[4][2];

    CUTLASS_DEVICE uint32_t &operator()(int row, int word) { return data[row][word]; }

    CUTLASS_DEVICE uint32_t const &operator()(int row, int word) const { return data[row][word]; }

    CUTLASS_DEVICE void load_row(int row, uint64_t value) {
      reinterpret_cast<uint64_t &>(data[row]) = value;
    }
  };

  CUTLASS_DEVICE static uint32_t tmem_stage_col(int stage_idx, int loop_offset) {
    return Tmem::kPreparedKv +
           static_cast<uint32_t>(stage_idx * Tmem::kColsPerPreparedKvStage + loop_offset * 4);
  }

  CUTLASS_DEVICE static uint64_t *kv_full_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kKvFullArv1 + stage);
  }

  CUTLASS_DEVICE static uint64_t *kv_empty_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kKvEmptyArv1 + stage);
  }

  CUTLASS_DEVICE static uint64_t *transformed_full_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kTransformedKvFullArv128 + stage);
  }

  CUTLASS_DEVICE static uint64_t *transformed_empty_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kTransformedKvEmptyArv1 + stage);
  }

  CUTLASS_DEVICE static int kv_tile_count(Params const &params, int batch_idx, int kv_head_idx,
                                          int q_token_idx) {
    return fmha_fwd_kv_tile_count_for_batch<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
  }

  CUTLASS_DEVICE static void sync_transform_warpgroup_before_raw_release(int barrier_id) {
    Sm100FmhaNamedBarrier::sync(128, barrier_id);
  }

  CUTLASS_DEVICE static uint32_t full_phase_for_event(int event, int stages) {
    return static_cast<uint32_t>((event / stages) & 1);
  }

  CUTLASS_DEVICE static uint32_t empty_phase_for_event(int event, int stages) {
    return static_cast<uint32_t>(1 ^ ((event / stages) & 1));
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

    uint8_t const *scale_lane_base =
        scale_stage_base + ((lane_idx / 4) + warp_group_warp_idx * cutlass::NumThreadsPerWarp) * 8;

    ScaleFragment scale;
    CUTLASS_PRAGMA_UNROLL
    for (int row = 0; row < 4; ++row) {
      scale.load_row(row, reinterpret_cast<uint64_t const *>(scale_lane_base)[row * 8]);
    }

    // 16x256b TMEM stores: thread (quad = lane & 3, row = lane / 4) owns 64 bits, i.e. eight
    // consecutive head dims, of TMEM lanes row and row + 8. Each 32-bit raw load therefore yields
    // the eight E2M1 values of one token, and a 32-dim block spans head-dim groups 2 * block
    // (quads 0, 1) and 2 * block + 1 (quads 2, 3), so the scale byte index is per thread.
    int const token0 = warp_group_warp_idx * cutlass::NumThreadsPerWarp + lane_idx / 4;
    int const quad = lane_idx & 3;
    int const group_parity = quad >> 1;
    uint32_t const selector_even_block = scale_pair_selector(group_parity);
    uint32_t const selector_odd_block = scale_pair_selector(2 + group_parity);

    CUTLASS_PRAGMA_UNROLL
    for (int block = 0; block < 4; ++block) {
      int const packed_byte = block * 16 + quad * 4;
      uint32_t const src0 = *reinterpret_cast<uint32_t const *>(
          raw_stage_base + packed_fp4_swizzled_offset(token0, packed_byte));
      uint32_t const src1 = *reinterpret_cast<uint32_t const *>(
          raw_stage_base + packed_fp4_swizzled_offset(token0 + 8, packed_byte));
      uint32_t const src2 = *reinterpret_cast<uint32_t const *>(
          raw_stage_base + packed_fp4_swizzled_offset(token0 + 16, packed_byte));
      uint32_t const src3 = *reinterpret_cast<uint32_t const *>(
          raw_stage_base + packed_fp4_swizzled_offset(token0 + 24, packed_byte));
      int const scale_word = block / 2;
      uint32_t const selector = (block & 1) ? selector_odd_block : selector_even_block;
      uint16_t const scale_pair01 =
          static_cast<uint16_t>(byte_permute(scale(0, scale_word), scale(1, scale_word), selector));
      uint16_t const scale_pair23 =
          static_cast<uint16_t>(byte_permute(scale(2, scale_word), scale(3, scale_word), selector));

      uint32_t dst0_lo, dst0_hi, dst1_lo, dst1_hi, dst2_lo, dst2_hi, dst3_lo, dst3_hi;
      convert_e2m1x8_token_pair_to_e4m3x8(dst0_lo, dst0_hi, dst1_lo, dst1_hi, src0, src1,
                                          scale_pair01);
      convert_e2m1x8_token_pair_to_e4m3x8(dst2_lo, dst2_hi, dst3_lo, dst3_hi, src2, src3,
                                          scale_pair23);

      uint32_t const tmem_col = tmem_base + tmem_stage_col(stage_idx, block * 2);
      tmem_store_16x256b(tmem_col, dst0_lo, dst0_hi, dst1_lo, dst1_hi);
      tmem_store_16x256b(tmem_col + 0x100000u, dst2_lo, dst2_hi, dst3_lo, dst3_hi);
    }
    fence_tmem_store();
  }

#if MINIMAX_MSA_Q8KV4_HAS_QMUL4 && MINIMAX_MSA_Q8KV4_BLOCK_SCALE_SHIFT != 0
  // Block-scale staging on the QMUL4 path: the converters multiply the raw E4M3 scale words, so
  // the stage's 1 KB scale block is shifted in place once, 128 threads x 8 consecutive bytes
  // (conflict-free), before the converters read it.
  CUTLASS_DEVICE static void stage_scale_words(uint8_t *smem_stage_base, int lane_idx,
                                               int warp_group_warp_idx) {
    static_assert(Traits::kRawKvScaleBytesPerStage == 128 * sizeof(uint2),
                  "one 8-byte word pair per transform thread.");
    uint2 *words = reinterpret_cast<uint2 *>(smem_stage_base + Traits::kRawKvDataBytesPerStage);
    int const thread = warp_group_warp_idx * cutlass::NumThreadsPerWarp + lane_idx;
    uint2 pair = words[thread];
    pair.x = stage_e4m3x4_block_scales(pair.x);
    pair.y = stage_e4m3x4_block_scales(pair.y);
    words[thread] = pair;
  }

  // K: thread t shifts token row t, and transform_k_stage's warp w reads only rows w * 32 ..
  // w * 32 + 31, so a warp-level sync orders the stores before its own loads.
  CUTLASS_DEVICE static void stage_k_scales(uint8_t *smem_stage_base, int lane_idx,
                                            int warp_group_warp_idx) {
    stage_scale_words(smem_stage_base, lane_idx, warp_group_warp_idx);
    __syncwarp();
  }

  // V: transform_v_stage's warp w reads group words 2w, 2w + 1 of every token quad, a 32 B column
  // across the eight 128 B rows that the other warps' threads shifted, so the warpgroup syncs.
  CUTLASS_DEVICE static void stage_v_scales(uint8_t *smem_stage_base, int lane_idx,
                                            int warp_group_warp_idx, int barrier_id) {
    stage_scale_words(smem_stage_base, lane_idx, warp_group_warp_idx);
    Sm100FmhaNamedBarrier::sync(128, barrier_id);
  }
#endif

#if !MINIMAX_MSA_Q8KV4_HAS_QMUL4
  // FP16 fallback: convert one raw stage's V scales to F16 pairs once, instead of in every V
  // iteration (Traits::kF16VScaleScratch sizes the scratch). The cache's token-quad order already
  // holds, per head-dim group, the four tokens' E4M3 scales in one word, so a lane fetches its
  // pair of group words with one 8-byte load. The QMUL4 path consumes those words in place.
  CUTLASS_DEVICE void prepare_v_scale_stage(int lane_idx, int warp_group_warp_idx,
                                            uint8_t *smem_stage_base, int barrier_id) const {
    uint8_t const *scale_stage_base = smem_stage_base + Traits::kRawKvDataBytesPerStage;
    uint8_t *scale_scratch_base =
        smem_stage_base + Traits::kRawKvDataBytesPerStage + Traits::kRawKvScaleBytesPerStage;
    int const warp_group_lane = warp_group_warp_idx * cutlass::NumThreadsPerWarp + lane_idx;
    // F16 scratch: quad fastest so the eight lanes of a 128-bit store phase cover one 128 B row.
    int const scale_pair = (warp_group_lane >> 2) & 3;
    int const token_quad = warp_group_lane & 3;
    int const loop_offset = (warp_group_lane >> 4) & 7;
    uint2 const words = *reinterpret_cast<uint2 const *>(scale_stage_base + loop_offset * 128 +
                                                         token_quad * 32 + scale_pair * 8);
    // 16 B chunk per (loop offset, token quad, pair): {g0: f16x2(t0,t1), f16x2(t2,t3)}
    // {g1: f16x2(t0,t1), f16x2(t2,t3)}. Quads are 16 B apart so the four chunks a warp reads per
    // V iteration sit in distinct banks (a 64 B quad stride aliases quads 0/2 and 1/3 across the
    // 128 B bank wrap and makes the 128-bit loads 2-way conflicted).
    uint4 f16_chunk;
    convert_e4m3x4_scales_to_f16x2_pair(f16_chunk.x, f16_chunk.y, words.x);
    convert_e4m3x4_scales_to_f16x2_pair(f16_chunk.z, f16_chunk.w, words.y);
#if MINIMAX_MSA_Q8KV4_BLOCK_SCALE_SHIFT != 0
    f16_chunk.x = mul_f16x2(f16_chunk.x, kBlockScaleMultF16x2);
    f16_chunk.y = mul_f16x2(f16_chunk.y, kBlockScaleMultF16x2);
    f16_chunk.z = mul_f16x2(f16_chunk.z, kBlockScaleMultF16x2);
    f16_chunk.w = mul_f16x2(f16_chunk.w, kBlockScaleMultF16x2);
#endif
    int const scale_dst_offset = loop_offset * 256 + token_quad * 16 + scale_pair * 64;
    *reinterpret_cast<uint4 *>(scale_scratch_base + scale_dst_offset) = f16_chunk;
    Sm100FmhaNamedBarrier::sync(128, barrier_id);
  }
#endif

  CUTLASS_DEVICE void transform_v_stage(uint32_t tmem_base, int stage_idx, int lane_idx,
                                        int warp_group_warp_idx,
                                        uint8_t const *smem_stage_base) const {
    uint8_t const *raw_stage_base = smem_stage_base;
    uint8_t const *scale_stage_base = smem_stage_base + Traits::kRawKvDataBytesPerStage;

    int const swizzle_mask = (lane_idx & 7) * 16;
    int const warp_col = warp_group_warp_idx * cutlass::NumThreadsPerWarp;
    int const swizzled_col0 = warp_col ^ swizzle_mask;
    int const swizzled_col1 = (warp_col + 16) ^ swizzle_mask;
    uint8_t const *raw_lane_base = raw_stage_base + lane_idx * Traits::kHeadDim;
#if MINIMAX_MSA_Q8KV4_HAS_QMUL4
    // The cache's token-quad scale order is the converters' order: this lane's quad block, group
    // words 2 * warp and 2 * warp + 1, read in place from the TMA-landed scales.
    uint8_t const *scale_reordered_lane_base =
        scale_stage_base + (lane_idx & 3) * 32 + warp_group_warp_idx * 8;
#else
    uint8_t const *scale_reordered_lane_base = scale_stage_base + Traits::kRawKvScaleBytesPerStage +
                                               (lane_idx & 3) * 16 + warp_group_warp_idx * 64;
#endif

    CUTLASS_PRAGMA_UNROLL
    for (int loop_offset = 0; loop_offset < 8; ++loop_offset) {
      uint32_t src0, src1, src2, src3;
      uint32_t dst0, dst1, dst2, dst3;
      int const row_offset = (loop_offset * 16) * Traits::kHeadDim;
      int const raw_offset0 = row_offset + swizzled_col0;
      int const raw_offset1 = row_offset + swizzled_col1;
      ldsm_unpack_fp4_transpose_16x16_x1(src0, src1, raw_lane_base + raw_offset0);
      ldsm_unpack_fp4_transpose_16x16_x1(src2, src3, raw_lane_base + raw_offset1);

#if MINIMAX_MSA_Q8KV4_HAS_QMUL4
      uint32_t const *scale_words =
          reinterpret_cast<uint32_t const *>(scale_reordered_lane_base + loop_offset * 128);
      convert_e2m1x4_pair_to_e4m3x4(dst0, dst1, src0, src1, scale_words[0]);
      convert_e2m1x4_pair_to_e4m3x4(dst2, dst3, src2, src3, scale_words[1]);
#else
      uint4 const scale_f16 =
          *reinterpret_cast<uint4 const *>(scale_reordered_lane_base + loop_offset * 256);
      convert_unpacked_e2m1x4_pair_to_e4m3x4_f16scales(dst0, dst1, src0, src1, scale_f16.x,
                                                       scale_f16.y);
      convert_unpacked_e2m1x4_pair_to_e4m3x4_f16scales(dst2, dst3, src2, src3, scale_f16.z,
                                                       scale_f16.w);
#endif

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
    Sm100FmhaBarrier::wait(kv_full_barrier(storage, raw_stage), raw_full_phase,
                           static_cast<uint32_t>(400 + raw_stage));
    Sm100FmhaBarrier::wait(transformed_empty_barrier(storage, transformed_stage),
                           transformed_empty_phase, static_cast<uint32_t>(420 + transformed_stage));
#if MINIMAX_MSA_Q8KV4_HAS_QMUL4 && MINIMAX_MSA_Q8KV4_BLOCK_SCALE_SHIFT != 0
    stage_k_scales(storage.smem_kv.stage_ptr(raw_stage), lane_idx, warp_group_warp_idx);
#endif

    transform_k_stage(tmem_base, transformed_stage, lane_idx, warp_group_warp_idx,
                      storage.smem_kv.stage_ptr(raw_stage));

    Sm100FmhaBarrier::arrive(transformed_full_barrier(storage, transformed_stage));
    sync_transform_warpgroup_before_raw_release(kRawKvReleaseBarrierId);
    if (warp_group_warp_idx == 0 && lane_idx == 0) {
      Sm100FmhaBarrier::arrive(kv_empty_barrier(storage, raw_stage));
    }
  }

  CUTLASS_DEVICE void transform_v(Storage &storage, uint32_t tmem_base, int raw_stage,
                                  int transformed_stage, uint32_t raw_full_phase,
                                  uint32_t raw_empty_phase, uint32_t transformed_full_phase,
                                  uint32_t transformed_empty_phase, int lane_idx,
                                  int warp_group_warp_idx,
                                  int scale_barrier_id = kScaleReorderBarrierId,
                                  int release_barrier_id = kRawKvReleaseVBarrierId) const {
    // Observe the TMEM stage release (PV of this stage's previous V tile) before the raw-full
    // parity wait: that previous V tile's fill then completed, so the wait is one phase ahead at
    // most (see kRawVStagesAlternateGroups).
    Sm100FmhaBarrier::wait(transformed_empty_barrier(storage, transformed_stage),
                           transformed_empty_phase, static_cast<uint32_t>(460 + transformed_stage));
    Sm100FmhaBarrier::wait(kv_full_barrier(storage, raw_stage), raw_full_phase,
                           static_cast<uint32_t>(440 + raw_stage));
#if MINIMAX_MSA_Q8KV4_HAS_QMUL4
#if MINIMAX_MSA_Q8KV4_BLOCK_SCALE_SHIFT != 0
    stage_v_scales(storage.smem_kv.stage_ptr(raw_stage), lane_idx, warp_group_warp_idx,
                   scale_barrier_id);
#endif
#else
    prepare_v_scale_stage(lane_idx, warp_group_warp_idx, storage.smem_kv.stage_ptr(raw_stage),
                          scale_barrier_id);
#endif

    transform_v_stage(tmem_base, transformed_stage, lane_idx, warp_group_warp_idx,
                      storage.smem_kv.stage_ptr(raw_stage));

    Sm100FmhaBarrier::arrive(transformed_full_barrier(storage, transformed_stage));
    sync_transform_warpgroup_before_raw_release(release_barrier_id);
    if (warp_group_warp_idx == 0 && lane_idx == 0) {
      Sm100FmhaBarrier::arrive(kv_empty_barrier(storage, raw_stage));
    }
  }

  CUTLASS_DEVICE void run_sequential_tile(Storage &storage, Params const &params, int batch_idx,
                                          int kv_head_idx, int q_token_idx, int lane_idx,
                                          int warp_group_warp_idx, State &state,
                                          int kv_tile_begin = 0, int kv_tile_end = INT_MAX) const {
    int const full_tiles = kv_tile_count(params, batch_idx, kv_head_idx, q_token_idx);
    Sm100FmhaKvTileRange const tile_range =
        make_kv_tile_range(full_tiles, kv_tile_begin, kv_tile_end);
    int const tiles = tile_range.count;
    if (tiles <= 0) {
      return;
    }

    uint32_t const tmem_base = storage.tmem_state_ptr()[0];

    // Transform schedule: prologue K0, steady V(i)+K(i+1), epilogue Vlast.
    transform_k(storage, tmem_base, state.raw_stage, state.transformed_stage, state.raw_full_phase,
                state.raw_empty_phase, state.transformed_full_phase, state.transformed_empty_phase,
                lane_idx, warp_group_warp_idx);
    advance_pipeline_state<Traits::kNumStagesRawKv>(state.raw_stage, state.raw_full_phase,
                                                    state.raw_empty_phase);
    advance_pipeline_state<Traits::kNumStagesTransform>(
        state.transformed_stage, state.transformed_full_phase, state.transformed_empty_phase);

    CUTLASS_PRAGMA_NO_UNROLL
    for (int tile = 0; tile + 1 < tiles; ++tile) {
      transform_v(storage, tmem_base, state.raw_stage, state.transformed_stage,
                  state.raw_full_phase, state.raw_empty_phase, state.transformed_full_phase,
                  state.transformed_empty_phase, lane_idx, warp_group_warp_idx);
      advance_pipeline_state<Traits::kNumStagesRawKv>(state.raw_stage, state.raw_full_phase,
                                                      state.raw_empty_phase);
      advance_pipeline_state<Traits::kNumStagesTransform>(
          state.transformed_stage, state.transformed_full_phase, state.transformed_empty_phase);

      transform_k(storage, tmem_base, state.raw_stage, state.transformed_stage,
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

  CUTLASS_DEVICE void transform_sparse_k_event(Storage &storage, uint32_t tmem_base, int event,
                                               int lane_idx, int warp_group_warp_idx) const {
    int const raw_stage = event % Traits::kNumStagesRawKv;
    int const transformed_stage = event % Traits::kNumStagesTransform;
    transform_k(storage, tmem_base, raw_stage, transformed_stage,
                full_phase_for_event(event, Traits::kNumStagesRawKv),
                empty_phase_for_event(event, Traits::kNumStagesRawKv),
                full_phase_for_event(event, Traits::kNumStagesTransform),
                empty_phase_for_event(event, Traits::kNumStagesTransform), lane_idx,
                warp_group_warp_idx);
  }

  CUTLASS_DEVICE void transform_sparse_v_event(Storage &storage, int lane_idx,
                                               int warp_group_warp_idx, VState &state) const {
    int const event = state.event;
    state.event += 2;
    int const tile = event / 2;
    // V split: the transform warpgroup converts V on every kTransformVTilePeriod-th tile,
    // the correction warpgroup converts the rest.
    if ((tile % kTransformVTilePeriod) == 0) {
      return;
    }
    uint32_t const tmem_base = storage.tmem_state_ptr()[0];
    int const raw_stage = event % Traits::kNumStagesRawKv;
    int const transformed_stage = event % Traits::kNumStagesTransform;
    if constexpr (kRawVStagesPinOnEmpty) {
      // Deeper TMEM ring: transform_v's TMEM-empty wait says nothing about this raw stage's
      // previous fill, so wait for that use to be released first (the loader's own condition for
      // issuing this event), as run_sparse_k_tile does for the assist tile.
      Sm100FmhaBarrier::wait(kv_empty_barrier(storage, raw_stage),
                             empty_phase_for_event(event, Traits::kNumStagesRawKv),
                             static_cast<uint32_t>(480 + raw_stage));
    }
    transform_v(storage, tmem_base, raw_stage, transformed_stage,
                full_phase_for_event(event, Traits::kNumStagesRawKv),
                empty_phase_for_event(event, Traits::kNumStagesRawKv),
                full_phase_for_event(event, Traits::kNumStagesTransform),
                empty_phase_for_event(event, Traits::kNumStagesTransform), lane_idx,
                warp_group_warp_idx);
  }

  CUTLASS_DEVICE void run_sparse_k_tile(Storage &storage, Params const &params, int batch_idx,
                                        int kv_head_idx, int q_token_idx, int lane_idx,
                                        int warp_group_warp_idx, State &state,
                                        int kv_tile_begin = 0, int kv_tile_end = INT_MAX) const {
    int const full_tiles =
        Sm100FmhaSelectionRing<Traits>::selected_pages(
            Sm100FmhaSelectionRing<Traits>::consume(storage, lane_idx, state.selection_event));
    Sm100FmhaKvTileRange const tile_range =
        make_kv_tile_range(full_tiles, kv_tile_begin, kv_tile_end);
    uint32_t const tmem_base = storage.tmem_state_ptr()[0];
    CUTLASS_PRAGMA_NO_UNROLL
    for (int tile = 0; tile < tile_range.count; ++tile) {
      int const k_event = state.sparse_k_event;
      transform_sparse_k_event(storage, tmem_base, k_event, lane_idx, warp_group_warp_idx);
      state.sparse_k_event += 2;
      if (((k_event / 2) % kTransformVTilePeriod) == 0) {
        int const v_event = k_event + 1;
        int const raw_stage = v_event % Traits::kNumStagesRawKv;
        int const transformed_stage = v_event % Traits::kNumStagesTransform;
        if constexpr (kRawVStagesAlternateGroups) {
          // See kRawVStagesAlternateGroups: pin the full barrier to this event's phase before the
          // parity wait inside transform_v.
          Sm100FmhaBarrier::wait(kv_empty_barrier(storage, raw_stage),
                                 empty_phase_for_event(v_event, Traits::kNumStagesRawKv),
                                 static_cast<uint32_t>(480 + raw_stage));
        }
        transform_v(storage, tmem_base, raw_stage, transformed_stage,
                    full_phase_for_event(v_event, Traits::kNumStagesRawKv),
                    empty_phase_for_event(v_event, Traits::kNumStagesRawKv),
                    full_phase_for_event(v_event, Traits::kNumStagesTransform),
                    empty_phase_for_event(v_event, Traits::kNumStagesTransform), lane_idx,
                    warp_group_warp_idx, kAssistScaleReorderBarrierId,
                    kAssistRawKvReleaseVBarrierId);
      }
    }
  }

  CUTLASS_DEVICE void run_tile(Storage &storage, Params const &params, int batch_idx,
                               int kv_head_idx, int q_token_idx, int lane_idx,
                               int warp_group_warp_idx, State &state, int kv_tile_begin = 0,
                               int kv_tile_end = INT_MAX) const {
    {
      run_sparse_k_tile(storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx,
                        warp_group_warp_idx, state, kv_tile_begin, kv_tile_end);
    }
  }

  CUTLASS_DEVICE void run_schedule(Storage &storage, Params const &params, int batch_idx,
                                   int kv_head_idx, int q_token_idx, int lane_idx,
                                   int warp_group_warp_idx, int kv_tile_begin = 0,
                                   int kv_tile_end = INT_MAX) const {
    State state;
    run_tile(storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx, warp_group_warp_idx,
             state, kv_tile_begin, kv_tile_end);
  }
};

} // namespace cutlass::fmha::collective
