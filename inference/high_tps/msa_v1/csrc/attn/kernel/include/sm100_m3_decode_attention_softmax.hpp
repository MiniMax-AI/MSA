#pragma once

#include <cfloat>
#include <climits>
#include <cstdint>

#include "attn/kernel/include/sm100_m3_decode_problem.hpp"
#include "common/kernel/include/sm100_m3_decode_fp4_transform.cuh"
#include "common/kernel/include/sm100_m3_decode_pipeline.hpp"
#include "common/kernel/include/sm100_m3_decode_storage.hpp"
#include "common/kernel/include/sm100_m3_decode_traits.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"

namespace cutlass::m3_decode::collective {

template <class Traits> struct Sm100M3DecodeAttentionSoftmax {
  using Storage = SharedStorage<Traits>;
  using Params = Sm100M3DecodeKernelParams<Traits>;
  using Barriers = BarrierLayout<Traits>;
  using Tmem = TmemLayout<Traits>;

  static constexpr int kReductionBarrierId = 1;
  static constexpr uint32_t kTmemRow16Offset = 0x100000u;
  static constexpr float kLog2E4m3Scale = 8.807354922057604f;

  struct State {
    int s_event = 0;
    int local_event = 0;
  };

  struct ScoreFragment {
    float data[16];

    CUTLASS_DEVICE float &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE float const &operator[](int idx) const { return data[idx]; }
  };

  struct ColumnFragment {
    float data[4];

    CUTLASS_DEVICE float &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE float const &operator[](int idx) const { return data[idx]; }
  };

  struct TmemWordFragment {
    uint32_t data[8];

    CUTLASS_DEVICE uint32_t &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE uint32_t const &operator[](int idx) const { return data[idx]; }
  };

  struct PackedPFragment {
    uint32_t data[4];

    CUTLASS_DEVICE uint32_t &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE uint32_t const &operator[](int idx) const { return data[idx]; }
  };

  CUTLASS_DEVICE static uint64_t *s_full_barrier(Storage &storage, int stage) {
    return storage.attention_pipelines().ptr(Barriers::kS0FullArv1 + stage);
  }

  CUTLASS_DEVICE static uint64_t *s_empty_barrier(Storage &storage, int stage) {
    return storage.attention_pipelines().ptr(Barriers::kS0EmptyArv128 + stage);
  }

  CUTLASS_DEVICE static uint64_t *softmax_local_full_barrier(Storage &storage, int stage) {
    return storage.attention_pipelines().ptr(Barriers::kSoftmaxLocalFullArv128 + stage);
  }

  CUTLASS_DEVICE static uint64_t *softmax_local_empty_barrier(Storage &storage, int stage) {
    return storage.attention_pipelines().ptr(Barriers::kSoftmaxLocalEmptyArv128 + stage);
  }

  CUTLASS_DEVICE static uint64_t *p_full_barrier(Storage &storage, int stage) {
    return storage.attention_pipelines().ptr(Barriers::kPFullArv128 + stage);
  }

  CUTLASS_DEVICE static uint64_t *p_empty_barrier(Storage &storage, int stage) {
    return storage.attention_pipelines().ptr(Barriers::kPEmptyArv1 + stage);
  }

  CUTLASS_DEVICE static int kv_length_for_batch(Params const &params, int batch_idx,
                                                int kv_head_idx, int q_token_idx) {
    return m3_decode_kv_length_for_batch<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
  }

  CUTLASS_DEVICE static int kv_tile_count(Params const &params, int batch_idx, int kv_head_idx,
                                          int q_token_idx) {
    int const kv_len = kv_length_for_batch(params, batch_idx, kv_head_idx, q_token_idx);
    return (kv_len + Traits::kTileKv - 1) / Traits::kTileKv;
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
    Sm100M3DecodeNamedBarrier::sync(128, kReductionBarrierId);
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
    tmem_load_16x256b_x2(raw0.data, tmem_col);
    tmem_load_16x256b_x2(raw1.data, tmem_col + kTmemRow16Offset);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      qk[i] = uint_as_float(raw0[i]);
      qk[i + 8] = uint_as_float(raw1[i]);
    }
    fence_tmem_load();
  }

  CUTLASS_DEVICE static void apply_dense_tail_mask(ScoreFragment &qk, int tile, int kv_len,
                                                   int warp_group_lane) {
    int const row_base =
        (warp_group_lane / cutlass::NumThreadsPerWarp) * 32 + ((warp_group_lane & 31) >> 2);
    int const tile_offset = tile * Traits::kTileKv;
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

  CUTLASS_DEVICE static void local_col_max(ColumnFragment &dst, ScoreFragment const &qk) {
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

  CUTLASS_DEVICE static void reduce_col_max(Storage &storage, ColumnFragment &values,
                                            int warp_group_lane) {
    uint32_t *smem = reinterpret_cast<uint32_t *>(storage.smem_softmax_red0.data);
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

  CUTLASS_DEVICE static void store_stats(uint32_t tmem_base, int event, ColumnFragment const &lo,
                                         ColumnFragment const &hi) {
    TmemWordFragment stats;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      stats[i] = float_as_uint(lo[i]);
      stats[i + 4] = float_as_uint(hi[i]);
    }
    tmem_store_32x32b_x8(tmem_base + stats_stage_tmem_col(event), stats.data);
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

    store_transposed_smem_8b_16x128(storage.smem_p.stage_ptr(stage), regs_p.data, warp_group_lane);
    cutlass::arch::fence_view_async_shared();
  }

  CUTLASS_DEVICE static void update_sum_from_p(ColumnFragment &sum, ScoreFragment const &p,
                                               ColumnFragment const &old_max,
                                               ColumnFragment const &new_max,
                                               float scale_softmax_log2) {
    float2 const scale2 = make_f32x2(scale_softmax_log2, scale_softmax_log2);
    float2 corr01 = fmul2(
        fadd2(make_f32x2(old_max[0], old_max[1]), make_f32x2(-new_max[0], -new_max[1])), scale2);
    corr01.x = exp2f(corr01.x);
    corr01.y = exp2f(corr01.y);
    float2 sum01 = ffma2(corr01, make_f32x2(sum[0], sum[1]), make_f32x2(p[0], p[1]));
    sum01 = fadd2(sum01, make_f32x2(p[2], p[3]));
    sum01 = fadd2(sum01, make_f32x2(p[8], p[9]));
    sum01 = fadd2(sum01, make_f32x2(p[10], p[11]));
    sum[0] = sum01.x;
    sum[1] = sum01.y;

    float2 corr23 = fmul2(
        fadd2(make_f32x2(old_max[2], old_max[3]), make_f32x2(-new_max[2], -new_max[3])), scale2);
    corr23.x = exp2f(corr23.x);
    corr23.y = exp2f(corr23.y);
    float2 sum23 = ffma2(corr23, make_f32x2(sum[2], sum[3]), make_f32x2(p[4], p[5]));
    sum23 = fadd2(sum23, make_f32x2(p[6], p[7]));
    sum23 = fadd2(sum23, make_f32x2(p[12], p[13]));
    sum23 = fadd2(sum23, make_f32x2(p[14], p[15]));
    sum[2] = sum23.x;
    sum[3] = sum23.y;
  }

  template <bool ApplyTailMask>
  CUTLASS_DEVICE void process_tile(Storage &storage, uint32_t tmem_base, int tile, int local_event,
                                   int s_stage, int local_stage, uint32_t s_full_phase,
                                   int next_local_stage, uint32_t next_softmax_empty_phase,
                                   int kv_len, int warp_group_lane, float scale_softmax_log2,
                                   ColumnFragment &old_max, ColumnFragment &new_max,
                                   ColumnFragment &sum) const {
    Sm100M3DecodeBarrier::wait(s_full_barrier(storage, s_stage), s_full_phase,
                               static_cast<uint32_t>(200 + s_stage));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      old_max[i] = new_max[i];
    }

    ScoreFragment qk;
    load_s_regs(qk, tmem_base, s_stage);
    if constexpr (ApplyTailMask) {
      apply_dense_tail_mask(qk, tile, kv_len, warp_group_lane);
    }
    local_col_max(new_max, qk);
    reduce_col_max(storage, new_max, warp_group_lane);
    store_stats(tmem_base, local_event, old_max, new_max);
    Sm100M3DecodeBarrier::arrive(softmax_local_full_barrier(storage, local_stage));

    make_p_and_store(storage, s_stage, qk, warp_group_lane, new_max, scale_softmax_log2);

    cutlass::BarrierStatus const next_empty_token = Sm100M3DecodeBarrier::try_wait_token(
        softmax_local_empty_barrier(storage, next_local_stage), next_softmax_empty_phase);
    Sm100M3DecodeBarrier::wait(softmax_local_empty_barrier(storage, next_local_stage),
                               next_softmax_empty_phase, next_empty_token,
                               static_cast<uint32_t>(210 + next_local_stage));
    Sm100M3DecodeBarrier::arrive(s_empty_barrier(storage, s_stage));
    update_sum_from_p(sum, qk, old_max, new_max, scale_softmax_log2);
  }

  CUTLASS_DEVICE void run_tile(Storage &storage, Params const &params, int batch_idx,
                               int kv_head_idx, int q_token_idx, int lane_idx,
                               int warp_group_warp_idx, State &state, int kv_len, int full_tiles,
                               int kv_tile_begin = 0, int kv_tile_end = INT_MAX) const {
    Sm100M3DecodeKvTileRange const tile_range =
        make_kv_tile_range(full_tiles, kv_tile_begin, kv_tile_end);
    int const tiles = tile_range.count;
    if (tiles <= 0) {
      return;
    }

    int const warp_group_lane = warp_group_warp_idx * cutlass::NumThreadsPerWarp + lane_idx;
    uint32_t volatile *tmem_state = storage.tmem_state_ptr();
    uint32_t const tmem_base = tmem_state[0];

    ColumnFragment old_max{{-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX}};
    ColumnFragment new_max{{-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX}};
    ColumnFragment sum{{0.f, 0.f, 0.f, 0.f}};

    init_reduction_smem(storage, warp_group_lane);
    float const scale_softmax_log2 = params.scale_softmax_log2;

    cutlass::BarrierStatus empty_token = Sm100M3DecodeBarrier::try_wait_token(
        softmax_local_empty_barrier(storage, state.local_event & 1),
        empty_phase(state.local_event, 2));
    Sm100M3DecodeBarrier::wait(softmax_local_empty_barrier(storage, state.local_event & 1),
                               empty_phase(state.local_event, 2), empty_token,
                               static_cast<uint32_t>(220 + (state.local_event & 1)));
    auto advance_stage = [&](int local_tile, int global_tile, auto apply_tail_mask) {
      int const s_event = state.s_event + local_tile;
      int const local_event = state.local_event + local_tile;
      int const s_stage = s_event & 1;
      int const local_stage = local_event & 1;
      int const next_local_event = local_event + 1;
      int const next_local_stage = next_local_event & 1;
      uint32_t const next_softmax_empty_phase = empty_phase(next_local_event, 2);
      process_tile<decltype(apply_tail_mask)::value>(
          storage, tmem_base, global_tile, local_event, s_stage, local_stage,
          full_phase(s_event, 2), next_local_stage, next_softmax_empty_phase, kv_len,
          warp_group_lane, scale_softmax_log2, old_max, new_max, sum);
    };

    int const tail_tokens = kv_len % Traits::kTileKv;
    CUTLASS_PRAGMA_NO_UNROLL
    for (int tile = 0; tile < tiles; ++tile) {
      int const global_tile = tile_range.begin + tile;
      if (tail_tokens != 0 && global_tile == full_tiles - 1) {
        advance_stage(tile, global_tile, cute::true_type{});
      } else {
        advance_stage(tile, global_tile, cute::false_type{});
      }
    }

    int const final_local_event = state.local_event + tiles;
    int const final_local_stage = final_local_event & 1;
    store_stats(tmem_base, final_local_event, sum, new_max);
    Sm100M3DecodeBarrier::arrive(softmax_local_full_barrier(storage, final_local_stage));
    state.s_event += tiles;
    state.local_event += tiles + 1;
  }

  CUTLASS_DEVICE void operator()(Storage &storage, Params const &params, int batch_idx,
                                 int kv_head_idx, int q_token_idx, int lane_idx,
                                 int warp_group_warp_idx) const {
    State state;
    int const kv_len = kv_length_for_batch(params, batch_idx, kv_head_idx, q_token_idx);
    int const full_tiles = (kv_len + Traits::kTileKv - 1) / Traits::kTileKv;
    run_tile(storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx, warp_group_warp_idx,
             state, kv_len, full_tiles);
  }
};

} // namespace cutlass::m3_decode::collective
