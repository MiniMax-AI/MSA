// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cfloat>
#include <climits>
#include <cstdint>

#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "fmha_common.hpp"
#include "sm100_fmha_correction_tma_warpspecialized.hpp"
#include "sm100_fmha_fp4_transform.cuh"
#include "sm100_fmha_pipeline.hpp"
#include "sm100_fmha_q8kv4_traits.hpp"
#include "sm100_fmha_selection_ring.hpp"
#include "sm100_fmha_storage.hpp"

// The softmax warps rescale the O accumulator themselves with the per-head factor they already
// compute for the running sum, instead of publishing (old_max, new_max) through TMEM for the
// correction warps to redo it. Stats are exchanged once per item (final sum and max for the
// epilogue) and the correction loop is V conversion only.

namespace cutlass::fmha::collective {

template <class Traits> struct Sm100FmhaSoftmaxTmaWarpspecialized {
  using Storage = SharedStorage<Traits>;
  using Params = Sm100FmhaFwdKernelParams<Traits>;
  using Barriers = BarrierLayout<Traits>;
  using Tmem = TmemLayout<Traits>;

  static constexpr int kReductionBarrierId = 1;
  static constexpr uint32_t kTmemRow16Offset = 0x100000u;
  static constexpr float kLog2E4m3Scale = 8.807354922057604f;
  static constexpr int kColumnsPerThread = Traits::kHeadGroup / 4;
  static constexpr int kWordsPerHalf = Traits::kHeadGroup / 2;

  struct State {
    int s_event = 0;
    int local_event = 0;
    int o_event = 0;
    int selection_event = 0;
    bool grid_dependency_synchronized = false;
    float scale_softmax_log2 = 0.f; // host scale x K global scale, read once per CTA
  };

  struct ScoreFragment {
    float data[Traits::kHeadGroup];

    CUTLASS_DEVICE float &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE float const &operator[](int idx) const { return data[idx]; }
  };

  struct ColumnFragment {
    float data[kColumnsPerThread];

    CUTLASS_DEVICE float &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE float const &operator[](int idx) const { return data[idx]; }
  };

  struct TmemWordFragment {
    uint32_t data[kWordsPerHalf];

    CUTLASS_DEVICE uint32_t &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE uint32_t const &operator[](int idx) const { return data[idx]; }
  };

  struct PackedPFragment {
    uint32_t data[kColumnsPerThread];

    CUTLASS_DEVICE uint32_t &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE uint32_t const &operator[](int idx) const { return data[idx]; }
  };

  CUTLASS_DEVICE static uint64_t *s_full_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kS0FullArv1 + stage);
  }

  CUTLASS_DEVICE static uint64_t *s_empty_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kS0EmptyArv128 + stage);
  }

  CUTLASS_DEVICE static uint64_t *softmax_local_full_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kSoftmaxLocalFullArv128 + stage);
  }

  CUTLASS_DEVICE static uint64_t *softmax_local_empty_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kSoftmaxLocalEmptyArv128 + stage);
  }

  CUTLASS_DEVICE static uint64_t *p_full_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kPFullArv128 + stage);
  }

  CUTLASS_DEVICE static uint64_t *p_empty_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kPEmptyArv1 + stage);
  }

  CUTLASS_DEVICE static uint32_t full_phase(int event, int stages) {
    return static_cast<uint32_t>((event / stages) & 1);
  }

  CUTLASS_DEVICE static uint32_t empty_phase(int event, int stages) {
    return static_cast<uint32_t>(1 ^ ((event / stages) & 1));
  }

  CUTLASS_DEVICE static uint32_t float_as_uint(float value) {
    return *reinterpret_cast<uint32_t *>(&value);
  }

  CUTLASS_DEVICE static float uint_as_float(uint32_t value) {
    return *reinterpret_cast<float *>(&value);
  }

  CUTLASS_DEVICE static uint32_t float_to_ordered_uint(float value) {
    uint32_t bits = float_as_uint(value);
    uint32_t mask = -static_cast<int32_t>(bits >> 31) | 0x80000000u;
    return bits ^ mask;
  }

  CUTLASS_DEVICE static float ordered_uint_to_float(uint32_t value) {
    uint32_t mask = ((value >> 31) - 1u) | 0x80000000u;
    return uint_as_float(value ^ mask);
  }

  CUTLASS_DEVICE static void warpgroup_sync() {
    Sm100FmhaNamedBarrier::sync(128, kReductionBarrierId);
  }

  CUTLASS_DEVICE static void init_reduction_smem(Storage &storage, int warp_group_lane) {
    uint32_t *smem = reinterpret_cast<uint32_t *>(storage.smem_softmax_red0.data);
    if (warp_group_lane < Traits::kHeadGroup) {
      smem[warp_group_lane] = float_to_ordered_uint(-FLT_MAX);
    }
    warpgroup_sync();
  }

  CUTLASS_DEVICE static uint32_t s_stage_tmem_col(int stage) {
    return (stage & 1) == 0 ? Tmem::kS0 : Tmem::kS1;
  }

  CUTLASS_DEVICE static uint32_t stats_stage_tmem_col(int event) {
    return (event & 1) == 0 ? Tmem::kStats0 : Tmem::kStats1;
  }

  CUTLASS_DEVICE static void load_s_regs(ScoreFragment &qk, uint32_t tmem_base, int stage) {
    TmemWordFragment raw0;
    TmemWordFragment raw1;
    uint32_t const tmem_col = tmem_base + s_stage_tmem_col(stage);
    if constexpr (Traits::kHeadGroup == 8) {
      tmem_load_16x256b_x1(raw0.data, tmem_col);
      tmem_load_16x256b_x1(raw1.data, tmem_col + kTmemRow16Offset);
    } else {
      tmem_load_16x256b_x2(raw0.data, tmem_col);
      tmem_load_16x256b_x2(raw1.data, tmem_col + kTmemRow16Offset);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kWordsPerHalf; ++i) {
      qk[i] = uint_as_float(raw0[i]);
      qk[i + kWordsPerHalf] = uint_as_float(raw1[i]);
    }
    fence_tmem_load();
  }

  CUTLASS_DEVICE static void apply_dense_tail_mask(ScoreFragment &qk, int tile, int kv_len,
                                                   int warp_group_lane) {
    int const row_base =
        (warp_group_lane / cutlass::NumThreadsPerWarp) * 32 + ((warp_group_lane & 31) >> 2);
    int const tile_offset = tile * Traits::kTileKv;
    if constexpr (Traits::kHeadGroup == 8) {
      CUTLASS_PRAGMA_UNROLL
      for (int row = 0; row < 4; ++row) {
        if (tile_offset + row_base + row * 8 >= kv_len) {
          qk[row * 2] = qk[row * 2 + 1] = -FLT_MAX;
        }
      }
    } else {
      if (tile_offset + row_base >= kv_len) {
        qk[0] = qk[1] = qk[4] = qk[5] = -FLT_MAX;
      }
      if (tile_offset + row_base + 8 >= kv_len) {
        qk[2] = qk[3] = qk[6] = qk[7] = -FLT_MAX;
      }
      if (tile_offset + row_base + 16 >= kv_len) {
        qk[8] = qk[9] = qk[12] = qk[13] = -FLT_MAX;
      }
      if (tile_offset + row_base + 24 >= kv_len) {
        qk[10] = qk[11] = qk[14] = qk[15] = -FLT_MAX;
      }
    }
  }

  CUTLASS_DEVICE static void local_col_max(ColumnFragment &dst, ScoreFragment const &qk) {
    if constexpr (Traits::kHeadGroup == 8) {
      CUTLASS_PRAGMA_UNROLL
      for (int col = 0; col < kColumnsPerThread; ++col) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < 4; ++row) {
          dst[col] = fmaxf(dst[col], qk[row * 2 + col]);
        }
      }
    } else {
      dst[0] = fmaxf(dst[0], qk[0]);
      dst[0] = fmaxf(dst[0], qk[2]);
      dst[0] = fmaxf(dst[0], qk[8]);
      dst[0] = fmaxf(dst[0], qk[10]);
      dst[1] = fmaxf(dst[1], qk[1]);
      dst[1] = fmaxf(dst[1], qk[3]);
      dst[1] = fmaxf(dst[1], qk[9]);
      dst[1] = fmaxf(dst[1], qk[11]);
      dst[2] = fmaxf(dst[2], qk[4]);
      dst[2] = fmaxf(dst[2], qk[6]);
      dst[2] = fmaxf(dst[2], qk[12]);
      dst[2] = fmaxf(dst[2], qk[14]);
      dst[3] = fmaxf(dst[3], qk[5]);
      dst[3] = fmaxf(dst[3], qk[7]);
      dst[3] = fmaxf(dst[3], qk[13]);
      dst[3] = fmaxf(dst[3], qk[15]);
    }
  }

  CUTLASS_DEVICE static void reduce_col_max(Storage &storage, ColumnFragment &values,
                                            int warp_group_lane) {
    uint32_t *smem = reinterpret_cast<uint32_t *>(storage.smem_softmax_red0.data);
    if constexpr (Traits::kHeadGroup == 8) {
      int const col_base = (warp_group_lane & 3) * kColumnsPerThread;
      int const row_idx = (warp_group_lane & 31) / 4;
      // Exchange columns while combining adjacent rows so each lane reduces one column.
      float left = values[0];
      float right = values[1];
      if ((row_idx & 1) == 0) {
        float const temporary = left;
        left = right;
        right = temporary;
      }
      float value = fmaxf(__shfl_xor_sync(0xffffffffu, left, 4), right);
      value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 8));
      value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 16));
      if (row_idx < 2) {
        atomicMax(smem + col_base + (row_idx & 1), float_to_ordered_uint(value));
      }
      warpgroup_sync();
      uint2 const reduced = *reinterpret_cast<uint2 const *>(smem + col_base);
      values[0] = ordered_uint_to_float(reduced.x);
      values[1] = ordered_uint_to_float(reduced.y);
    } else {
      int const col_base = (warp_group_lane & 3) * 4;
      int const lane_idx = warp_group_lane & 31;
      int const row_idx = lane_idx / 4;
      int const local_row_idx = row_idx & 3;

      // Specialized WSPRO max reduction for NumColsPerThread=4,
      // ReduceGroupSize=4, group_stride=4.
      ColumnFragment wspro{{values[0], values[1], values[2], values[3]}};
      float left = wspro[0];
      float right = wspro[1];
      if ((local_row_idx & 1) == 0) {
        float tmp = left;
        left = right;
        right = tmp;
      }
      left = __shfl_xor_sync(0xffffffffu, left, 4);
      wspro[0] = fmaxf(left, right);

      left = wspro[2];
      right = wspro[3];
      if ((local_row_idx & 1) == 0) {
        float tmp = left;
        left = right;
        right = tmp;
      }
      left = __shfl_xor_sync(0xffffffffu, left, 4);
      wspro[2] = fmaxf(left, right);

      left = wspro[0];
      right = wspro[2];
      if (local_row_idx < 2) {
        float tmp = left;
        left = right;
        right = tmp;
      }
      left = __shfl_xor_sync(0xffffffffu, left, 8);
      wspro[0] = fmaxf(left, right);

      atomicMax(smem + col_base + local_row_idx, float_to_ordered_uint(wspro[0]));
      warpgroup_sync();
      uint4 vals = *reinterpret_cast<uint4 const *>(smem + col_base);
      values[0] = ordered_uint_to_float(vals.x);
      values[1] = ordered_uint_to_float(vals.y);
      values[2] = ordered_uint_to_float(vals.z);
      values[3] = ordered_uint_to_float(vals.w);
    }
  }

  CUTLASS_DEVICE static void store_stats(uint32_t tmem_base, int event, ColumnFragment const &lo,
                                         ColumnFragment const &hi) {
    TmemWordFragment stats;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kColumnsPerThread; ++i) {
      stats[i] = float_as_uint(lo[i]);
      stats[i + kColumnsPerThread] = float_as_uint(hi[i]);
    }
    if constexpr (Traits::kHeadGroup == 8) {
      tmem_store_32x32b_x4(tmem_base + stats_stage_tmem_col(event), stats.data);
    } else {
      tmem_store_32x32b_x8(tmem_base + stats_stage_tmem_col(event), stats.data);
    }
    fence_tmem_store();
  }

  CUTLASS_DEVICE static void make_p_and_store(Storage &storage, int stage, ScoreFragment &qk,
                                              int warp_group_lane, ColumnFragment const &new_max,
                                              float scale_softmax_log2) {
    float2 const scale2 = make_f32x2(scale_softmax_log2, scale_softmax_log2);
    float2 const neg_scale2 = make_f32x2(-scale_softmax_log2, -scale_softmax_log2);
    float2 const e4m3_bias2 = make_f32x2(kLog2E4m3Scale, kLog2E4m3Scale);
    float2 const neg_scaled_max01 =
        ffma2(make_f32x2(new_max[0], new_max[1]), neg_scale2, e4m3_bias2);
    if constexpr (Traits::kHeadGroup == 8) {
      CUTLASS_PRAGMA_UNROLL
      for (int row = 0; row < 4; ++row) {
        float2 const p = ffma2(make_f32x2(qk[row * 2], qk[row * 2 + 1]), scale2, neg_scaled_max01);
        qk[row * 2] = exp2f(p.x);
        qk[row * 2 + 1] = exp2f(p.y);
      }
      PackedPFragment regs_p;
      CUTLASS_PRAGMA_UNROLL
      for (int half = 0; half < 2; ++half) {
        int const offset = half * kWordsPerHalf;
        regs_p[half] =
            pack_float4_to_e4m3(qk[offset], qk[offset + 1], qk[offset + 2], qk[offset + 3]);
      }
      store_transposed_smem_8b_8x128(storage.smem_p.stage_ptr(stage), regs_p.data, warp_group_lane);
    } else {
      float2 const neg_scaled_max23 =
          ffma2(make_f32x2(new_max[2], new_max[3]), neg_scale2, e4m3_bias2);

      float2 p01 = ffma2(make_f32x2(qk[0], qk[1]), scale2, neg_scaled_max01);
      float2 p23 = ffma2(make_f32x2(qk[2], qk[3]), scale2, neg_scaled_max01);
      PackedPFragment regs_p;

      qk[0] = exp2f(p01.x);
      float2 p45 = ffma2(make_f32x2(qk[4], qk[5]), scale2, neg_scaled_max23);
      qk[1] = exp2f(p01.y);

      qk[2] = exp2f(p23.x);
      float2 p67 = ffma2(make_f32x2(qk[6], qk[7]), scale2, neg_scaled_max23);
      qk[3] = exp2f(p23.y);

      qk[4] = exp2f(p45.x);
      float2 p89 = ffma2(make_f32x2(qk[8], qk[9]), scale2, neg_scaled_max01);
      qk[5] = exp2f(p45.y);

      qk[6] = exp2f(p67.x);
      float2 p1011 = ffma2(make_f32x2(qk[10], qk[11]), scale2, neg_scaled_max01);
      qk[7] = exp2f(p67.y);

      qk[8] = exp2f(p89.x);
      float2 p1213 = ffma2(make_f32x2(qk[12], qk[13]), scale2, neg_scaled_max23);
      qk[9] = exp2f(p89.y);

      qk[10] = exp2f(p1011.x);
      float2 p1415 = ffma2(make_f32x2(qk[14], qk[15]), scale2, neg_scaled_max23);
      qk[11] = exp2f(p1011.y);
      regs_p[0] = pack_float4_to_e4m3(qk[0], qk[1], qk[2], qk[3]);

      qk[12] = exp2f(p1213.x);
      qk[13] = exp2f(p1213.y);

      qk[14] = exp2f(p1415.x);
      qk[15] = exp2f(p1415.y);
      regs_p[1] = pack_float4_to_e4m3(qk[4], qk[5], qk[6], qk[7]);
      regs_p[2] = pack_float4_to_e4m3(qk[8], qk[9], qk[10], qk[11]);
      regs_p[3] = pack_float4_to_e4m3(qk[12], qk[13], qk[14], qk[15]);

      store_transposed_smem_8b_16x128(storage.smem_p.stage_ptr(stage), regs_p.data,
                                      warp_group_lane);
    }
    cutlass::arch::fence_view_async_shared();
  }

  using Correction = Sm100FmhaCorrectionTmaWarpspecialized<Traits>;

  CUTLASS_DEVICE static uint64_t *o_full_barrier(Storage &storage) {
    return storage.pipelines.ptr(Barriers::kOFullArv1);
  }

  CUTLASS_DEVICE static uint64_t *o_empty_barrier(Storage &storage) {
    return storage.pipelines.ptr(Barriers::kOEmptyArv128);
  }

  // exp2(scale * (old_max - new_max)) per head. The O accumulator and the running sum are
  // rescaled by this same value, computed once per tile.
  CUTLASS_DEVICE static void make_corr(ColumnFragment const &old_max, ColumnFragment const &new_max,
                                       float scale_softmax_log2, ColumnFragment &corr) {
    float2 const scale2 = make_f32x2(scale_softmax_log2, scale_softmax_log2);
    float2 const corr01 = fmul2(
        fadd2(make_f32x2(old_max[0], old_max[1]), make_f32x2(-new_max[0], -new_max[1])), scale2);
    corr[0] = exp2f(corr01.x);
    corr[1] = exp2f(corr01.y);
    if constexpr (Traits::kHeadGroup == 16) {
      float2 const corr23 = fmul2(
          fadd2(make_f32x2(old_max[2], old_max[3]), make_f32x2(-new_max[2], -new_max[3])), scale2);
      corr[2] = exp2f(corr23.x);
      corr[3] = exp2f(corr23.y);
    }
  }

  CUTLASS_DEVICE static void update_sum_from_p(ColumnFragment &sum, ScoreFragment const &p,
                                               ColumnFragment const &corr) {
    if constexpr (Traits::kHeadGroup == 8) {
      float2 value =
          ffma2(make_f32x2(corr[0], corr[1]), make_f32x2(sum[0], sum[1]), make_f32x2(p[0], p[1]));
      CUTLASS_PRAGMA_UNROLL
      for (int row = 1; row < 4; ++row) {
        value = fadd2(value, make_f32x2(p[row * 2], p[row * 2 + 1]));
      }
      sum[0] = value.x;
      sum[1] = value.y;
    } else {
      float2 sum01 =
          ffma2(make_f32x2(corr[0], corr[1]), make_f32x2(sum[0], sum[1]), make_f32x2(p[0], p[1]));
      sum01 = fadd2(sum01, make_f32x2(p[2], p[3]));
      sum01 = fadd2(sum01, make_f32x2(p[8], p[9]));
      sum01 = fadd2(sum01, make_f32x2(p[10], p[11]));
      sum[0] = sum01.x;
      sum[1] = sum01.y;
      float2 sum23 =
          ffma2(make_f32x2(corr[2], corr[3]), make_f32x2(sum[2], sum[3]), make_f32x2(p[4], p[5]));
      sum23 = fadd2(sum23, make_f32x2(p[6], p[7]));
      sum23 = fadd2(sum23, make_f32x2(p[12], p[13]));
      sum23 = fadd2(sum23, make_f32x2(p[14], p[15]));
      sum[2] = sum23.x;
      sum[3] = sum23.y;
    }
  }

  // One tile with the O rescale on this warpgroup. `o_full_phase` is the phase of PV(tile - 1)
  // (for tile 0: the previous item's last PV, waited but not rescaled) so that the o_full wait is
  // never more than one phase ahead of the barrier.
  template <bool ApplyTailMask>
  CUTLASS_DEVICE void process_tile(Storage &storage, uint32_t tmem_base, int tile, int s_stage,
                                   uint32_t s_full_phase, bool rescale_o, uint32_t o_full_phase,
                                   int tail_bound, int warp_group_lane, float scale_softmax_log2,
                                   ColumnFragment &old_max, ColumnFragment &new_max,
                                   ColumnFragment &sum) const {
    Sm100FmhaBarrier::wait(s_full_barrier(storage, s_stage), s_full_phase,
                           static_cast<uint32_t>(200 + s_stage));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kColumnsPerThread; ++i) {
      old_max[i] = new_max[i];
    }

    ScoreFragment qk;
    load_s_regs(qk, tmem_base, s_stage);
    if constexpr (ApplyTailMask) {
      apply_dense_tail_mask(qk, tile, tail_bound, warp_group_lane);
    }
    local_col_max(new_max, qk);
    reduce_col_max(storage, new_max, warp_group_lane);

    ColumnFragment corr;
    make_corr(old_max, new_max, scale_softmax_log2, corr);
    Sm100FmhaBarrier::wait(o_full_barrier(storage), o_full_phase, 230);
    if (rescale_o) {
      typename Correction::OFragment regs_o;
      typename Correction::ScaleFragment scale;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kColumnsPerThread; ++i) {
        scale[i] = corr[i];
      }
      Correction::load_o_regs(tmem_base, regs_o);
      Correction::scale_o_regs(regs_o, scale);
      Correction::store_o_regs(tmem_base, regs_o);
      Sm100FmhaBarrier::arrive(o_empty_barrier(storage));
    }

    make_p_and_store(storage, s_stage, qk, warp_group_lane, new_max, scale_softmax_log2);
    Sm100FmhaBarrier::arrive(s_empty_barrier(storage, s_stage));
    update_sum_from_p(sum, qk, corr);
  }

  CUTLASS_DEVICE void run_tile(Storage &storage, Params const &params, int batch_idx,
                               int kv_head_idx, int q_token_idx, int lane_idx,
                               int warp_group_warp_idx, State &state, int kv_tile_begin = 0,
                               int kv_tile_end = INT_MAX) const {
    using Ring = Sm100FmhaSelectionRing<Traits>;
    uint32_t const record = Ring::consume(storage, lane_idx, state.selection_event);
    Sm100FmhaKvTileRange const tile_range =
        make_kv_tile_range(Ring::selected_pages(record), kv_tile_begin, kv_tile_end);
    int const tiles = tile_range.count;
    if (tiles <= 0) {
      return;
    }

    int const warp_group_lane = warp_group_warp_idx * cutlass::NumThreadsPerWarp + lane_idx;
    uint32_t volatile *tmem_state = storage.tmem_state_ptr();
    uint32_t const tmem_base = tmem_state[0];

    ColumnFragment old_max;
    ColumnFragment new_max;
    ColumnFragment sum;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kColumnsPerThread; ++i) {
      old_max[i] = new_max[i] = -FLT_MAX;
      sum[i] = 0.f;
    }

    init_reduction_smem(storage, warp_group_lane);
    if (!state.grid_dependency_synchronized) {
      cudaGridDependencySynchronize();
      state.grid_dependency_synchronized = true;
      // Once per CTA, after the grid dependency: the global scale may be produced by the
      // preceding kernel in the stream.
      state.scale_softmax_log2 = params.scale_softmax_log2 * __ldg(params.k_global_scale_ptr);
    }
    float const scale_softmax_log2 = state.scale_softmax_log2;

    // At most one tile holds the query's own page and carries a causal tail (the producer
    // resolved which); every other tile is fully visible.
    int const tail_tile = Ring::tail_tile(record);
    int const tail_bound = tail_tile * Traits::kTileKv + Ring::tail_limit(record);
    auto advance_stage = [&](int local_tile, int global_tile, auto apply_tail_mask) {
      int const s_event = state.s_event + local_tile;
      process_tile<decltype(apply_tail_mask)::value>(
          storage, tmem_base, global_tile, s_event & 1, full_phase(s_event, 2), local_tile > 0,
          full_phase(state.o_event + local_tile - 1, 1), tail_bound, warp_group_lane,
          scale_softmax_log2, old_max, new_max, sum);
    };

    CUTLASS_PRAGMA_NO_UNROLL
    for (int tile = 0; tile < tiles; ++tile) {
      int const global_tile = tile_range.begin + tile;
      if (global_tile == tail_tile) {
        advance_stage(tile, global_tile, cute::true_type{});
      } else {
        advance_stage(tile, global_tile, cute::false_type{});
      }
    }

    // One stats exchange per item: (sum, max) for the correction warps' epilogue.
    int const final_local_event = state.local_event;
    int const final_local_stage = final_local_event & 1;
    Sm100FmhaBarrier::wait(softmax_local_empty_barrier(storage, final_local_stage),
                           empty_phase(final_local_event, 2),
                           static_cast<uint32_t>(220 + final_local_stage));
    store_stats(tmem_base, final_local_event, sum, new_max);
    Sm100FmhaBarrier::arrive(softmax_local_full_barrier(storage, final_local_stage));
    state.s_event += tiles;
    state.local_event += 1;
    state.o_event += tiles;
  }

  CUTLASS_DEVICE void operator()(Storage &storage, Params const &params, int batch_idx,
                                 int kv_head_idx, int q_token_idx, int lane_idx,
                                 int warp_group_warp_idx) const {
    State state;
    run_tile(storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx, warp_group_warp_idx,
             state);
  }
};

} // namespace cutlass::fmha::collective
