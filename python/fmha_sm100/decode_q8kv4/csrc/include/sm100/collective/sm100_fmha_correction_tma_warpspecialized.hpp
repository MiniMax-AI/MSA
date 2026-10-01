// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "fmha_common.hpp"
#include "sm100_fmha_fp4_transform.cuh"
#include "sm100_fmha_kv_transform_tma_warpspecialized.hpp"
#include "sm100_fmha_pipeline.hpp"
#include "sm100_fmha_q8kv4_traits.hpp"
#include "sm100_fmha_selection_ring.hpp"
#include "sm100_fmha_storage.hpp"

namespace cutlass::fmha::collective {

template <class Traits> struct Sm100FmhaCorrectionTmaWarpspecialized {
  using Storage = SharedStorage<Traits>;
  using Params = Sm100FmhaFwdKernelParams<Traits>;
  using Barriers = BarrierLayout<Traits>;
  using Tmem = TmemLayout<Traits>;
  using KvTransform = Sm100FmhaKvTransformTmaWarpspecialized<Traits>;

  static constexpr int kStoreOBarrierId = 3;
  static constexpr int kFinalReductionBarrierId = 4;
  static constexpr int kMergeBarrierId = 9;
  static constexpr uint32_t kTmemRow16Offset = 0x100000u;
  static constexpr int kColumnsPerThread = Traits::kHeadGroup / 4;
  static constexpr int kWordsPerHalf = Traits::kHeadGroup / 2;

  struct State {
    int softmax_event = 0;
    int o_event = 0;
    int selection_event = 0;
    uint32_t merge_stage_phase = 0;
    bool grid_dependency_synchronized = false;
    typename KvTransform::VState sparse_v_state;
    // Host scales x global scales, read once per CTA after the grid dependency.
    float scale_log2 = 0.f;
    float output_scale = 0.f;
  };

  struct StatsFragment {
    float data[kWordsPerHalf];

    CUTLASS_DEVICE float &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE float const &operator[](int idx) const { return data[idx]; }
  };

  struct ScaleFragment {
    float data[kColumnsPerThread];

    CUTLASS_DEVICE float &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE float const &operator[](int idx) const { return data[idx]; }
  };

  struct OFragment {
    float data[Traits::kHeadGroup];

    CUTLASS_DEVICE float &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE float const &operator[](int idx) const { return data[idx]; }
  };

  struct Bf16OFragment {
    uint32_t data[kWordsPerHalf];

    CUTLASS_DEVICE uint32_t &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE uint32_t const &operator[](int idx) const { return data[idx]; }
  };

  struct TmemWordFragment {
    uint32_t data[kWordsPerHalf];

    CUTLASS_DEVICE uint32_t &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE uint32_t const &operator[](int idx) const { return data[idx]; }
  };

  CUTLASS_DEVICE static uint64_t *softmax_local_full_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kSoftmaxLocalFullArv128 + stage);
  }

  CUTLASS_DEVICE static uint64_t *softmax_local_empty_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kSoftmaxLocalEmptyArv128 + stage);
  }

  CUTLASS_DEVICE static uint64_t *o_full_barrier(Storage &storage) {
    return storage.pipelines.ptr(Barriers::kOFullArv1);
  }

  CUTLASS_DEVICE static uint64_t *o_empty_barrier(Storage &storage) {
    return storage.pipelines.ptr(Barriers::kOEmptyArv128);
  }

  CUTLASS_DEVICE static bool use_workspace_split(Params const &params) {
    return params.num_kv_splits > 1 && params.workspace_o_ptr != nullptr;
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

  CUTLASS_DEVICE static void store_o_sync() { Sm100FmhaNamedBarrier::sync(128, kStoreOBarrierId); }

  CUTLASS_DEVICE static uint32_t stats_stage_tmem_col(int event) {
    return (event & 1) == 0 ? Tmem::kStats0 : Tmem::kStats1;
  }

  CUTLASS_DEVICE static void load_stats(uint32_t tmem_base, int event, StatsFragment &stats) {
    TmemWordFragment raw;
    if constexpr (Traits::kHeadGroup == 8) {
      tmem_load_32x32b_x4(raw.data, tmem_base + stats_stage_tmem_col(event));
    } else {
      tmem_load_32x32b_x8(raw.data, tmem_base + stats_stage_tmem_col(event));
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kWordsPerHalf; ++i) {
      stats[i] = uint_as_float(raw[i]);
    }
    fence_tmem_load();
  }

  CUTLASS_DEVICE static void load_o_regs(uint32_t tmem_base, OFragment &regs_o) {
    TmemWordFragment raw0;
    TmemWordFragment raw1;
    uint32_t const tmem_col = tmem_base + Tmem::kO0;
    if constexpr (Traits::kHeadGroup == 8) {
      tmem_load_16x256b_x1(raw0.data, tmem_col);
      tmem_load_16x256b_x1(raw1.data, tmem_col + kTmemRow16Offset);
    } else {
      tmem_load_16x256b_x2(raw0.data, tmem_col);
      tmem_load_16x256b_x2(raw1.data, tmem_col + kTmemRow16Offset);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kWordsPerHalf; ++i) {
      regs_o[i] = uint_as_float(raw0[i]);
      regs_o[i + kWordsPerHalf] = uint_as_float(raw1[i]);
    }
    fence_tmem_load();
  }

  CUTLASS_DEVICE static void store_o_regs(uint32_t tmem_base, OFragment const &regs_o) {
    TmemWordFragment raw0;
    TmemWordFragment raw1;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kWordsPerHalf; ++i) {
      raw0[i] = float_as_uint(regs_o[i]);
      raw1[i] = float_as_uint(regs_o[i + kWordsPerHalf]);
    }
    uint32_t const tmem_col = tmem_base + Tmem::kO0;
    if constexpr (Traits::kHeadGroup == 8) {
      tmem_store_16x256b_x1(tmem_col, raw0.data);
      tmem_store_16x256b_x1(tmem_col + kTmemRow16Offset, raw1.data);
    } else {
      tmem_store_16x256b_x2(tmem_col, raw0.data);
      tmem_store_16x256b_x2(tmem_col + kTmemRow16Offset, raw1.data);
    }
    fence_tmem_store();
  }

  CUTLASS_DEVICE static void scale_o_regs(OFragment &regs_o, ScaleFragment const &scale) {
    float2 const scale01 = make_f32x2(scale[0], scale[1]);
    if constexpr (Traits::kHeadGroup == 8) {
      CUTLASS_PRAGMA_UNROLL
      for (int pair = 0; pair < kWordsPerHalf; ++pair) {
        float2 const vals = fmul2(make_f32x2(regs_o[pair * 2], regs_o[pair * 2 + 1]), scale01);
        regs_o[pair * 2] = vals.x;
        regs_o[pair * 2 + 1] = vals.y;
      }
    } else {
      float2 const scale23 = make_f32x2(scale[2], scale[3]);

      float2 vals = fmul2(make_f32x2(regs_o[0], regs_o[1]), scale01);
      regs_o[0] = vals.x;
      regs_o[1] = vals.y;
      vals = fmul2(make_f32x2(regs_o[2], regs_o[3]), scale01);
      regs_o[2] = vals.x;
      regs_o[3] = vals.y;
      vals = fmul2(make_f32x2(regs_o[4], regs_o[5]), scale23);
      regs_o[4] = vals.x;
      regs_o[5] = vals.y;
      vals = fmul2(make_f32x2(regs_o[6], regs_o[7]), scale23);
      regs_o[6] = vals.x;
      regs_o[7] = vals.y;
      vals = fmul2(make_f32x2(regs_o[8], regs_o[9]), scale01);
      regs_o[8] = vals.x;
      regs_o[9] = vals.y;
      vals = fmul2(make_f32x2(regs_o[10], regs_o[11]), scale01);
      regs_o[10] = vals.x;
      regs_o[11] = vals.y;
      vals = fmul2(make_f32x2(regs_o[12], regs_o[13]), scale23);
      regs_o[12] = vals.x;
      regs_o[13] = vals.y;
      vals = fmul2(make_f32x2(regs_o[14], regs_o[15]), scale23);
      regs_o[14] = vals.x;
      regs_o[15] = vals.y;
    }
  }

  CUTLASS_DEVICE static void reduce_final_sum(Storage &storage, ScaleFragment const &partial_sum,
                                              ScaleFragment &total_sum, int warp_group_lane) {
    float *smem = storage.smem_corr_red1.data;
    if constexpr (Traits::kHeadGroup == 8) {
      float2 sum = make_f32x2(partial_sum[0], partial_sum[1]);
      CUTLASS_PRAGMA_UNROLL
      for (int mask = 16; mask >= 4; mask /= 2) {
        sum = fadd2(sum, make_f32x2(__shfl_xor_sync(0xffffffffu, sum.x, mask),
                                    __shfl_xor_sync(0xffffffffu, sum.y, mask)));
      }
      int const col = (warp_group_lane & 3) * kColumnsPerThread;
      int const warp = warp_group_lane / cutlass::NumThreadsPerWarp;
      if ((warp_group_lane & 31) < 4) {
        *reinterpret_cast<float2 *>(smem + warp * Traits::kHeadGroup + col) = sum;
      }
      Sm100FmhaNamedBarrier::sync(128, kFinalReductionBarrierId);
      sum = *reinterpret_cast<float2 const *>(smem + col);
      CUTLASS_PRAGMA_UNROLL
      for (int w = 1; w < Traits::kNumCorrectionWarps; ++w) {
        sum = fadd2(sum, *reinterpret_cast<float2 const *>(smem + w * Traits::kHeadGroup + col));
      }
      total_sum[0] = sum.x;
      total_sum[1] = sum.y;
    } else {
      float2 sums[2] = {make_f32x2(partial_sum[0], partial_sum[1]),
                        make_f32x2(partial_sum[2], partial_sum[3])};

      CUTLASS_PRAGMA_UNROLL
      for (int lane_mask = 16; lane_mask >= 4; lane_mask /= 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i) {
          float2 other = make_f32x2(__shfl_xor_sync(0xffffffffu, sums[i].x, lane_mask),
                                    __shfl_xor_sync(0xffffffffu, sums[i].y, lane_mask));
          sums[i] = fadd2(sums[i], other);
        }
      }

      int const warp_idx = warp_group_lane / cutlass::NumThreadsPerWarp;
      int const lane_idx = warp_group_lane % cutlass::NumThreadsPerWarp;
      int const col_idx = warp_group_lane & 3;
      int const smem_dst = warp_idx * Traits::kHeadGroup + col_idx * 4;

      if (lane_idx < 4) {
        float4 tmp;
        tmp.x = sums[0].x;
        tmp.y = sums[0].y;
        tmp.z = sums[1].x;
        tmp.w = sums[1].y;
        *reinterpret_cast<float4 *>(smem + smem_dst) = tmp;
      }
      Sm100FmhaNamedBarrier::sync(128, kFinalReductionBarrierId);

      int const smem_src = col_idx * 4;
      float4 tmp = *reinterpret_cast<float4 const *>(smem + smem_src);
      sums[0] = make_f32x2(tmp.x, tmp.y);
      sums[1] = make_f32x2(tmp.z, tmp.w);

      CUTLASS_PRAGMA_UNROLL
      for (int warp = 1; warp < Traits::kNumCorrectionWarps; ++warp) {
        float4 other_tmp =
            *reinterpret_cast<float4 const *>(smem + smem_src + warp * Traits::kHeadGroup);
        float2 other[2] = {make_f32x2(other_tmp.x, other_tmp.y),
                           make_f32x2(other_tmp.z, other_tmp.w)};
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i) {
          sums[i] = fadd2(sums[i], other[i]);
        }
      }

      total_sum[0] = sums[0].x;
      total_sum[1] = sums[0].y;
      total_sum[2] = sums[1].x;
      total_sum[3] = sums[1].y;
    }
  }

  CUTLASS_DEVICE static void make_final_scale(ScaleFragment const &total_sum, float output_scale,
                                              ScaleFragment &scale) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kColumnsPerThread; ++i) {
      scale[i] = output_scale / total_sum[i];
    }
  }

  CUTLASS_DEVICE static int packed_row_base(Params const &params, int batch_idx, int q_token_idx) {
    return fmha_fwd_packed_row_base<Traits>(params, batch_idx, q_token_idx);
  }

  CUTLASS_DEVICE static int stats_local_row(int warp_group_lane, int stats_idx) {
    // Stats fragments follow the 32x32 TMEM row layout: each of the first
    // four lanes owns two rows in the lower half and two rows in the upper half.
    return ((stats_idx >> 1) << 3) + (warp_group_lane << 1) + (stats_idx & 1);
  }

  CUTLASS_DEVICE static void pack_o_bf16(OFragment const &regs_o, Bf16OFragment &regs_bf16) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kWordsPerHalf; ++i) {
      regs_bf16[i] = pack_float2_to_bfloat16(regs_o[i * 2], regs_o[i * 2 + 1]);
    }
  }

  CUTLASS_DEVICE static void pack_scaled_o_bf16(OFragment const &regs_o, ScaleFragment const &scale,
                                                Bf16OFragment &regs_bf16) {
    float2 const scale01 = make_f32x2(scale[0], scale[1]);
    if constexpr (Traits::kHeadGroup == 8) {
      CUTLASS_PRAGMA_UNROLL
      for (int pair = 0; pair < kWordsPerHalf; ++pair) {
        float2 const vals = fmul2(make_f32x2(regs_o[pair * 2], regs_o[pair * 2 + 1]), scale01);
        regs_bf16[pair] = pack_float2_to_bfloat16(vals.x, vals.y);
      }
    } else {
      float2 const scale23 = make_f32x2(scale[2], scale[3]);

      float2 vals = fmul2(make_f32x2(regs_o[0], regs_o[1]), scale01);
      regs_bf16[0] = pack_float2_to_bfloat16(vals.x, vals.y);
      vals = fmul2(make_f32x2(regs_o[2], regs_o[3]), scale01);
      regs_bf16[1] = pack_float2_to_bfloat16(vals.x, vals.y);
      vals = fmul2(make_f32x2(regs_o[4], regs_o[5]), scale23);
      regs_bf16[2] = pack_float2_to_bfloat16(vals.x, vals.y);
      vals = fmul2(make_f32x2(regs_o[6], regs_o[7]), scale23);
      regs_bf16[3] = pack_float2_to_bfloat16(vals.x, vals.y);
      vals = fmul2(make_f32x2(regs_o[8], regs_o[9]), scale01);
      regs_bf16[4] = pack_float2_to_bfloat16(vals.x, vals.y);
      vals = fmul2(make_f32x2(regs_o[10], regs_o[11]), scale01);
      regs_bf16[5] = pack_float2_to_bfloat16(vals.x, vals.y);
      vals = fmul2(make_f32x2(regs_o[12], regs_o[13]), scale23);
      regs_bf16[6] = pack_float2_to_bfloat16(vals.x, vals.y);
      vals = fmul2(make_f32x2(regs_o[14], regs_o[15]), scale23);
      regs_bf16[7] = pack_float2_to_bfloat16(vals.x, vals.y);
    }
  }

  CUTLASS_DEVICE static void copy_o_smem_to_global(Storage &storage, Params const &params,
                                                   int batch_idx, int kv_head_idx, int q_token_idx,
                                                   int warp_group_lane) {
    static_assert(Traits::kHeadDim == 128,
                  "FMHA forward decode output copy is specialized for D=128.");
    static_assert(Traits::kHeadGroup == 8 || Traits::kHeadGroup == 16,
                  "FMHA forward decode output copy is specialized for 16 Q heads per KV head.");

    constexpr int kVecBytes = 16;
    constexpr int kSmemRowBytes = 128;
    constexpr int kDstRowBytes = Traits::kHeadDim * static_cast<int>(sizeof(cutlass::bfloat16_t));
    constexpr int kCopyCalls = (Traits::kHeadGroup * kDstRowBytes) /
                               (Traits::kNumSoftmaxWarps * cutlass::NumThreadsPerWarp * kVecBytes);
    static_assert(kDstRowBytes == 256, "BF16 D=128 output rows are 256B.");
    static_assert(kCopyCalls == Traits::kHeadGroup / 8,
                  "128-thread warpgroup should copy the 4096B O tile in two passes.");

    int const head_base = kv_head_idx * Traits::kHeadGroup;
    int const remaining_heads = params.num_qo_heads_orig - head_base;
    int const valid_rows =
        remaining_heads < Traits::kHeadGroup ? remaining_heads : Traits::kHeadGroup;
    int const qo_offset = fmha_fwd_q_token_global_index<Traits>(params, batch_idx, q_token_idx);

    uint8_t *smem_base = storage.smem_o.data;
    uint8_t *gmem_base = reinterpret_cast<uint8_t *>(params.o_ptr) +
                         (static_cast<int64_t>(qo_offset) * params.num_qo_heads_orig + head_base) *
                             Traits::kHeadDim * static_cast<int64_t>(sizeof(cutlass::bfloat16_t));

    CUTLASS_PRAGMA_UNROLL
    for (int copy = 0; copy < kCopyCalls; ++copy) {
      int const base_offset = warp_group_lane * kVecBytes + copy * Traits::kNumSoftmaxWarps *
                                                                cutlass::NumThreadsPerWarp *
                                                                kVecBytes;
      int const smem_row = base_offset / kSmemRowBytes;
      int const smem_col = base_offset & (kSmemRowBytes - 1);
      int const load_offset = base_offset ^ ((smem_row & 7) * kVecBytes);
      int const dst_row = smem_row & (Traits::kHeadGroup - 1);
      int const dst_col = (smem_row / Traits::kHeadGroup) * kSmemRowBytes + smem_col;

      if (dst_row < valid_rows && dst_col < kDstRowBytes) {
        auto src = cute::make_tensor(
            cute::make_smem_ptr(reinterpret_cast<uint32_t const *>(smem_base + load_offset)),
            cute::make_shape(cute::Int<4>{}));
        auto dst = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<uint32_t *>(
                                         gmem_base + dst_row * kDstRowBytes + dst_col)),
                                     cute::make_shape(cute::Int<4>{}));
        cute::copy(cute::AutoVectorizingCopyWithAssumedAlignment<128>{}, src, dst);
      }
    }
  }

  // Balanced schedule: the workspace holds the slots of an item contiguously ([item][slot]), each
  // slot one 16 x 256 B partial O block followed by its 16 row LSEs, so the merge fetches a slot
  // with one bulk copy. The legacy split path keeps its [slot][packed row][head] layout and the
  // separate LSE array for the reduction kernel.
  CUTLASS_DEVICE static uint8_t *merge_slot_ptr(Params const &params, int q_token_global,
                                                int kv_head_idx, int slot) {
    int64_t const item = static_cast<int64_t>(q_token_global) * params.num_kv_heads + kv_head_idx -
                         params.merge_item_base;
    return static_cast<uint8_t *>(params.workspace_o_ptr) +
           (item * params.num_kv_splits + slot) * Traits::kMergeSlotBytes;
  }

  CUTLASS_DEVICE static float *merge_slot_lse_ptr(Params const &params, int q_token_global,
                                                  int kv_head_idx, int slot) {
    return reinterpret_cast<float *>(merge_slot_ptr(params, q_token_global, kv_head_idx, slot) +
                                     Traits::kMergeSlotOBytes);
  }

  CUTLASS_DEVICE static void copy_o_smem_to_workspace(Storage &storage, Params const &params,
                                                      int batch_idx, int kv_head_idx,
                                                      int q_token_idx, int kv_split_idx,
                                                      int warp_group_lane) {
    static_assert(Traits::kHeadDim == 128,
                  "FMHA forward decode output copy is specialized for D=128.");
    static_assert(Traits::kHeadGroup == 8 || Traits::kHeadGroup == 16,
                  "FMHA forward decode output copy is specialized for 16 packed rows.");

    constexpr int kVecBytes = 16;
    constexpr int kSmemRowBytes = 128;
    constexpr int kDstRowBytes = Traits::kHeadDim * static_cast<int>(sizeof(cutlass::bfloat16_t));
    constexpr int kCopyCalls = (Traits::kHeadGroup * kDstRowBytes) /
                               (Traits::kNumSoftmaxWarps * cutlass::NumThreadsPerWarp * kVecBytes);
    static_assert(kDstRowBytes == 256, "BF16 D=128 output rows are 256B.");
    static_assert(kCopyCalls == Traits::kHeadGroup / 8,
                  "128-thread warpgroup should copy the 4096B O tile in two passes.");

    int const row_base = packed_row_base(params, batch_idx, q_token_idx);
    int const split_idx = kv_split_idx < 0 ? 0 : kv_split_idx;
    uint8_t *smem_base = storage.smem_o.data;
    uint8_t *gmem_base;
    int packed_row_stride;
    if (params.merge_counter_ptr != nullptr) {
      gmem_base = merge_slot_ptr(
          params, fmha_fwd_q_token_global_index<Traits>(params, batch_idx, q_token_idx),
          kv_head_idx, split_idx);
      packed_row_stride = kDstRowBytes;
    } else {
      gmem_base = reinterpret_cast<uint8_t *>(params.workspace_o_ptr) +
                  ((static_cast<int64_t>(split_idx) * params.total_qo_len + row_base) *
                       params.num_qo_heads +
                   kv_head_idx) *
                      Traits::kHeadDim * static_cast<int64_t>(sizeof(cutlass::bfloat16_t));
      packed_row_stride =
          params.num_qo_heads * Traits::kHeadDim * static_cast<int>(sizeof(cutlass::bfloat16_t));
    }

    CUTLASS_PRAGMA_UNROLL
    for (int copy = 0; copy < kCopyCalls; ++copy) {
      int const base_offset = warp_group_lane * kVecBytes + copy * Traits::kNumSoftmaxWarps *
                                                                cutlass::NumThreadsPerWarp *
                                                                kVecBytes;
      int const smem_row = base_offset / kSmemRowBytes;
      int const smem_col = base_offset & (kSmemRowBytes - 1);
      int const load_offset = base_offset ^ ((smem_row & 7) * kVecBytes);
      int const dst_row = smem_row & (Traits::kHeadGroup - 1);
      int const dst_col = (smem_row / Traits::kHeadGroup) * kSmemRowBytes + smem_col;

      if (row_base + dst_row < params.total_qo_len && kv_head_idx < params.num_qo_heads &&
          dst_col < kDstRowBytes) {
        auto src = cute::make_tensor(
            cute::make_smem_ptr(reinterpret_cast<uint32_t const *>(smem_base + load_offset)),
            cute::make_shape(cute::Int<4>{}));
        auto dst = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<uint32_t *>(
                                         gmem_base + dst_row * packed_row_stride + dst_col)),
                                     cute::make_shape(cute::Int<4>{}));
        cute::copy(cute::AutoVectorizingCopyWithAssumedAlignment<128>{}, src, dst);
      }
    }
  }

  CUTLASS_DEVICE static void store_lse_to_workspace(Params const &params, int batch_idx,
                                                    int kv_head_idx, int q_token_idx,
                                                    int kv_split_idx, StatsFragment const &stats,
                                                    ScaleFragment const &total_sum,
                                                    int warp_group_lane, float scale_log2) {
    if ((params.workspace_lse_ptr == nullptr && params.merge_counter_ptr == nullptr) ||
        warp_group_lane >= 4) {
      return;
    }
    int const packed_base = packed_row_base(params, batch_idx, q_token_idx);
    int const split_idx = kv_split_idx < 0 ? 0 : kv_split_idx;
    float *slot_lse =
        params.merge_counter_ptr != nullptr
            ? merge_slot_lse_ptr(
                  params, fmha_fwd_q_token_global_index<Traits>(params, batch_idx, q_token_idx),
                  kv_head_idx, split_idx)
            : nullptr;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kColumnsPerThread; ++i) {
      int const local_row = stats_local_row(warp_group_lane, i);
      int const row = packed_base + local_row;
      if (row < params.total_qo_len && kv_head_idx < params.num_qo_heads) {
        float const row_sum = total_sum[i];
        float const lse = row_sum == 0.f
                              ? -INFINITY
                              : stats[i + kColumnsPerThread] + __log2f(row_sum) / scale_log2;
        if (slot_lse != nullptr) {
          slot_lse[local_row] = lse;
        } else {
          int const stats_offset = split_idx * params.total_qo_len * params.num_qo_heads +
                                   row * params.num_qo_heads + kv_head_idx;
          params.workspace_lse_ptr[stats_offset] = lse;
        }
      }
    }
  }

  CUTLASS_DEVICE static void store_o_to_global(Storage &storage, Params const &params,
                                               int batch_idx, int kv_head_idx, int q_token_idx,
                                               int kv_split_idx, OFragment const &regs_o,
                                               ScaleFragment const &scale, int warp_group_lane,
                                               bool direct_store = false) {
    Bf16OFragment regs_bf16;
    pack_scaled_o_bf16(regs_o, scale, regs_bf16);
    if constexpr (Traits::kHeadGroup == 8) {
      store_transposed_smem_16b_128x8(storage.smem_o.data, regs_bf16.data, warp_group_lane);
    } else {
      store_transposed_smem_16b_128x16(storage.smem_o.data, regs_bf16.data, warp_group_lane);
    }
    store_o_sync();
    if (use_workspace_split(params) && !direct_store) {
      copy_o_smem_to_workspace(storage, params, batch_idx, kv_head_idx, q_token_idx, kv_split_idx,
                               warp_group_lane);
    } else {
      copy_o_smem_to_global(storage, params, batch_idx, kv_head_idx, q_token_idx, warp_group_lane);
    }
  }

  // In-kernel merge of a split item on the balanced (stream-K) schedule. Every segment CTA has
  // stored its normalized partial O and LSE to the workspace slot of its segment; each then bumps
  // the item's arrival counter, and the last arriver folds the slots in slot order and stores the
  // output. The fixed fold order keeps the result deterministic whichever CTA arrives last, and
  // no CTA ever waits for another. atomicInc wraps the counter back to zero on the last arrival,
  // so the counters need no reset between launches. Slots of segments that ran empty hold zeros
  // with -inf LSE and contribute nothing.
  //
  // The last arriver fetches the slots with bulk copies into the staging buffer (one copy per
  // slot) and folds from shared memory. Bulk copies run in the async proxy and bypass the tagged
  // L1, whose capacity for in-flight lines depends on the shared-memory carveout; the fold's
  // loads would otherwise serialize on that capacity.
  struct MergeItem {
    int q_token_global;
    int kv_head_idx;
    int kv_split_count;
  };

  CUTLASS_DEVICE static int *merge_item_counter(Params const &params, MergeItem const &item) {
    return params.merge_counter_ptr +
           static_cast<int64_t>(item.q_token_global) * params.num_kv_heads + item.kv_head_idx -
           params.merge_item_base;
  }

  // Arrival on the item counter by the calling thread; true when this segment arrived last.
  CUTLASS_DEVICE static bool merge_arrive(Params const &params, MergeItem const &item) {
    unsigned int *counter = reinterpret_cast<unsigned int *>(merge_item_counter(params, item));
    unsigned int const limit = static_cast<unsigned int>(item.kv_split_count - 1);
    unsigned int previous;
    asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                 : "=r"(previous)
                 : "l"(counter), "r"(limit)
                 : "memory");
    return previous == limit;
  }

  // Fetch every slot of the item into the staging buffer, fold, and store the output on the
  // correction warpgroup: each thread owns one row's stretch of 16 columns.
  //
  // Fold in slot order with the standalone reduction kernel's arithmetic: a slot whose LSE
  // exceeds the running one rescales the accumulator by exp2(scale (run - lse)) and is added;
  // otherwise the slot is rescaled by exp2(scale (lse - run)) and added. Both cases are one FMA
  // with the operands swapped, so the fold predicates instead of branching (the decision
  // differs per row, and a warp holds several rows). exp2 sees min - max in both cases, the new
  // running LSE is the max, and w + r equals fmaf(1, r, w), so the scalar chain is branch-free
  // too. Slots holding zeros with -inf LSE (empty segments, and the staging slots beyond the
  // segment count, cleared below) get weight exp2(-inf) = 0 and contribute nothing, so the
  // fold runs over whole groups of slots without per-slot guards and the scheduler can overlap
  // one slot's exp2 chain with the next slot's vector work.
  CUTLASS_DEVICE static void merge_fetch_fold_store(Storage &storage, Params const &params,
                                                    MergeItem const &item, int thread_idx,
                                                    uint64_t *stage_full, uint32_t &stage_phase,
                                                    float scale_log2, float output_scale) {
    static_assert(Traits::kHeadDim == 128 && (Traits::kHeadGroup == 8 || Traits::kHeadGroup == 16),
                  "in-kernel split merge is specialized for 16 rows of 128 elements.");
    constexpr int kThreads = cutlass::NumThreadsPerWarp * Traits::kNumCorrectionWarps;
    constexpr int kWarps = Traits::kNumCorrectionWarps;
    constexpr int kThreadsPerRow = kThreads / Traits::kHeadGroup;
    constexpr int kElementsPerThread = Traits::kHeadDim / kThreadsPerRow;
    constexpr int kPassElements = kElementsPerThread;
    constexpr int kPasses = kElementsPerThread / kPassElements;
    // Slots folded per group: the whole staging in two groups.
    constexpr int kFoldGroup = 4;
    static_assert(Traits::kMergeMaxSlots % kFoldGroup == 0, "fold groups must tile the slots.");
    int const kv_split_count = item.kv_split_count;
    int const row = thread_idx / kThreadsPerRow;
    int const col_base = (thread_idx % kThreadsPerRow) * kElementsPerThread;
    auto &stage = storage.smem_merge_stage;

    // One bulk copy per slot, issued by lane 0 of each participating warp (the copy is a uniform
    // instruction; per-lane issue from one warp would serialize). The transaction count is
    // registered once; copies completing before it only make the count transiently negative.
    if ((thread_idx & (cutlass::NumThreadsPerWarp - 1)) == 0) {
      int const warp = thread_idx / cutlass::NumThreadsPerWarp;
      // Order this thread's generic-proxy view (acquired slots, previous reads of the staging
      // buffer) before its async-proxy copies.
      asm volatile("fence.proxy.async;" ::: "memory");
      if (warp == 0) {
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
            stage_full, static_cast<uint32_t>(kv_split_count * Traits::kMergeSlotBytes));
      }
      uint32_t const bar = cute::cast_smem_ptr_to_uint(stage_full);
      CUTLASS_PRAGMA_NO_UNROLL
      for (int slot = warp; slot < kv_split_count; slot += kWarps) {
        void const *src = merge_slot_ptr(params, item.q_token_global, item.kv_head_idx, slot);
        uint32_t const dst = cute::cast_smem_ptr_to_uint(stage.slot_o(slot));
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], "
                     "[%1], %2, [%3];"
                     :
                     : "r"(dst), "l"(src), "r"(static_cast<uint32_t>(Traits::kMergeSlotBytes)),
                       "r"(bar)
                     : "memory");
      }
    }
    int const slot_groups = (kv_split_count + kFoldGroup - 1) / kFoldGroup;
    // Clear the staging slots of the last group that the copies do not fill (this thread's own
    // stretch of O and, on the first 16 threads, its row's LSE). Generic stores to slots the
    // bulk copies do not touch.
    for (int slot = kv_split_count; slot < slot_groups * kFoldGroup; ++slot) {
      uint4 *o_dst = reinterpret_cast<uint4 *>(stage.slot_o(slot) + row * (Traits::kHeadDim * 2) +
                                               col_base * 2);
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < kElementsPerThread / 8; ++v) {
        o_dst[v] = make_uint4(0u, 0u, 0u, 0u);
      }
      if (thread_idx < Traits::kHeadGroup) {
        stage.slot_lse(slot)[thread_idx] = -INFINITY;
      }
    }
    Sm100FmhaBarrier::wait(stage_full, stage_phase);
    stage_phase ^= 1u;
    if (slot_groups * kFoldGroup > kv_split_count) {
      // The clearing stores and the copies target different bytes; only the clears have to be
      // ordered before the fold's loads across the warpgroup.
      Sm100FmhaNamedBarrier::sync(128, kMergeBarrierId);
    }
    int const head_base = item.kv_head_idx * Traits::kHeadGroup;
    uint16_t *out_row =
        static_cast<uint16_t *>(params.o_ptr) +
        (static_cast<int64_t>(item.q_token_global) * params.num_qo_heads_orig + head_base + row) *
            Traits::kHeadDim;
    CUTLASS_PRAGMA_UNROLL
    for (int pass = 0; pass < kPasses; ++pass) {
      int const col = col_base + pass * kPassElements;
      float running_lse = -FLT_MAX;
      float running_w = 0.f;
      float running_o[kPassElements];
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kPassElements; ++i) {
        running_o[i] = 0.f;
      }
      auto fold_group = [&](int group) {
        uint4 v0[kFoldGroup];
        uint4 v1[kFoldGroup];
        float lse[kFoldGroup];
        CUTLASS_PRAGMA_UNROLL
        for (int g = 0; g < kFoldGroup; ++g) {
          int const slot = group * kFoldGroup + g;
          uint4 const *src = reinterpret_cast<uint4 const *>(
              stage.slot_o(slot) + row * (Traits::kHeadDim * 2) + col * 2);
          v0[g] = src[0];
          if constexpr (kPassElements == 16) {
            v1[g] = src[1];
          }
          lse[g] = stage.slot_lse(slot)[row];
        }
        CUTLASS_PRAGMA_UNROLL
        for (int g = 0; g < kFoldGroup; ++g) {
          uint32_t words[kPassElements / 2];
          words[0] = v0[g].x;
          words[1] = v0[g].y;
          words[2] = v0[g].z;
          words[3] = v0[g].w;
          if constexpr (kPassElements == 16) {
            words[4] = v1[g].x;
            words[5] = v1[g].y;
            words[6] = v1[g].z;
            words[7] = v1[g].w;
          }
          bool const raise = lse[g] > running_lse;
          float const rescale =
              exp2f(scale_log2 * (fminf(lse[g], running_lse) - fmaxf(lse[g], running_lse)));
          // Both FMA forms are issued under complementary predicates: two full-rate FFMAs cost
          // less than one FFMA plus two operand selects on the half-rate ALU pipe.
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < kPassElements / 2; ++i) {
            float const p0 = __uint_as_float(words[i] << 16);
            float const p1 = __uint_as_float(words[i] & 0xffff0000u);
            float const o0 = running_o[2 * i];
            float const o1 = running_o[2 * i + 1];
            running_o[2 * i] = raise ? fmaf(o0, rescale, p0) : fmaf(p0, rescale, o0);
            running_o[2 * i + 1] = raise ? fmaf(o1, rescale, p1) : fmaf(p1, rescale, o1);
          }
          running_w = fmaf(raise ? running_w : 1.f, rescale, raise ? 1.f : running_w);
          running_lse = fmaxf(lse[g], running_lse);
        }
      };
      CUTLASS_PRAGMA_NO_UNROLL
      for (int group = 0; group < slot_groups; ++group) {
        fold_group(group);
      }
      uint32_t packed[kPassElements / 2];
      if (running_w == 0.f) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPassElements / 2; ++i) {
          packed[i] = 0u;
        }
      } else {
        float const inv_w = output_scale / running_w;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPassElements / 2; ++i) {
          cutlass::bfloat16_t const lo(running_o[2 * i] * inv_w);
          cutlass::bfloat16_t const hi(running_o[2 * i + 1] * inv_w);
          packed[i] = static_cast<uint32_t>(lo.raw()) | (static_cast<uint32_t>(hi.raw()) << 16);
        }
      }
      uint4 *out_vec = reinterpret_cast<uint4 *>(out_row + col);
      out_vec[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
      if constexpr (kPassElements == 16) {
        out_vec[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
      }
    }
  }

  // Arrival and, as the last arriver, the fold.
  CUTLASS_DEVICE static void merge_split_partials(Storage &storage, Params const &params,
                                                  MergeItem const &item, int warp_group_lane,
                                                  State &state) {
    // Arrival: the warpgroup barrier orders every thread's workspace stores before one thread's
    // acq_rel atomic on the item counter (fence cumulativity, as CUTLASS's split-K semaphore and
    // trtllm-gen's GmemReduction do), so no thread needs a full fence of its own.
    Sm100FmhaNamedBarrier::sync(128, kMergeBarrierId);
    int32_t *flag = storage.merge_flag_ptr();
    if (warp_group_lane == 0) {
      *flag = merge_arrive(params, item) ? 1 : 0;
    }
    Sm100FmhaNamedBarrier::sync(128, kMergeBarrierId);
    if (*flag == 0) {
      return;
    }
    merge_fetch_fold_store(storage, params, item, warp_group_lane,
                           storage.pipelines.ptr(Barriers::kMergeStageFullArv1),
                           state.merge_stage_phase, state.scale_log2, state.output_scale);
  }

  CUTLASS_DEVICE void run_tile(Storage &storage, Params const &params, int batch_idx,
                               int kv_head_idx, int q_token_idx, int lane_idx,
                               int warp_group_warp_idx, State &state, int kv_tile_begin = 0,
                               int kv_tile_end = INT_MAX, int kv_split_idx = 0,
                               int kv_split_count = 0) const {
    uint32_t const record =
        Sm100FmhaSelectionRing<Traits>::consume(storage, lane_idx, state.selection_event);
    int const full_tiles = Sm100FmhaSelectionRing<Traits>::selected_pages(record);
    Sm100FmhaKvTileRange const tile_range =
        make_kv_tile_range(full_tiles, kv_tile_begin, kv_tile_end);
    int const tiles = tile_range.count;
    int const warp_group_lane = warp_group_warp_idx * cutlass::NumThreadsPerWarp + lane_idx;
    if (!state.grid_dependency_synchronized) {
      cudaGridDependencySynchronize();
      state.grid_dependency_synchronized = true;
      // Once per CTA, after the grid dependency: the global scales may be produced by the
      // preceding kernel in the stream. Empty segments merge and store too, so this comes first.
      state.scale_log2 = params.scale_softmax_log2 * __ldg(params.k_global_scale_ptr);
      state.output_scale = params.scale_output * __ldg(params.v_global_scale_ptr);
    }
    // Balanced schedule (arrival counters present): items cut into several segments merge in
    // the kernel; items that stayed whole store directly and never touch the workspace.
    bool const balanced_schedule =
        use_workspace_split(params) && params.merge_counter_ptr != nullptr;
    bool const merge_in_kernel = balanced_schedule && kv_split_count > 1;
    bool const direct_store = balanced_schedule && kv_split_count <= 1;
    if (tiles <= 0) {
      if (merge_in_kernel) {
        // A segment past the request's pages still has to arrive; it publishes a zero partial
        // with -inf LSEs, which the fold weights by exp2(-inf) = 0 and adds as zeros, so the
        // fold needs no validity test (the separate reduction kernel substitutes zeros for
        // such slots, which yields the same values).
        uint8_t *slot = merge_slot_ptr(
            params, fmha_fwd_q_token_global_index<Traits>(params, batch_idx, q_token_idx),
            kv_head_idx, kv_split_idx < 0 ? 0 : kv_split_idx);
        constexpr int kVectorsPerThread = Traits::kHeadGroup / 8;
        uint4 *slot_o = reinterpret_cast<uint4 *>(slot) + kVectorsPerThread * warp_group_lane;
        CUTLASS_PRAGMA_UNROLL
        for (int vector = 0; vector < kVectorsPerThread; ++vector) {
          slot_o[vector] = make_uint4(0u, 0u, 0u, 0u);
        }
        if (warp_group_lane < Traits::kHeadGroup) {
          reinterpret_cast<float *>(slot + Traits::kMergeSlotOBytes)[warp_group_lane] = -INFINITY;
        }
        merge_split_partials(
            storage, params,
            MergeItem{fmha_fwd_q_token_global_index<Traits>(params, batch_idx, q_token_idx),
                      kv_head_idx, kv_split_count},
            warp_group_lane, state);
      } else if (use_workspace_split(params) && !direct_store) {
        // A reused plan must overwrite the LSE of an empty split from the previous request.
        StatsFragment stats{};
        ScaleFragment total_sum{};
        store_lse_to_workspace(params, batch_idx, kv_head_idx, q_token_idx, kv_split_idx, stats,
                               total_sum, warp_group_lane, state.scale_log2);
      } else {
        OFragment zero{};
        ScaleFragment scale{};
        store_o_to_global(storage, params, batch_idx, kv_head_idx, q_token_idx, kv_split_idx, zero,
                          scale, warp_group_lane, direct_store);
      }
      return;
    }

    uint32_t volatile *tmem_state = storage.tmem_state_ptr();
    uint32_t const tmem_base = tmem_state[0];
    // Partials for the separate reduction stay unscaled; whole items apply the output scale and
    // the V global scale here.
    float const output_scale = use_workspace_split(params) && !direct_store
                                   ? params.scale_output_split
                                   : state.output_scale;

    {
      KvTransform{}.transform_sparse_v_event(
          storage, lane_idx, warp_group_warp_idx, state.sparse_v_state,
          KvTransform::visible_v_tokens(record, tile_range.begin));
    }

    StatsFragment stats;
    // The softmax warps rescale O with the factors they compute anyway; this group only
    // converts its share of V (V(tile + 1) in iteration `tile`; transform_v's TMEM-empty wait
    // before the raw-full wait keeps the raw-stage parity one phase ahead) and runs the epilogue.
    CUTLASS_PRAGMA_NO_UNROLL
    for (int tile = 0; tile + 1 < tiles; ++tile) {
      {
        KvTransform{}.transform_sparse_v_event(
            storage, lane_idx, warp_group_warp_idx, state.sparse_v_state,
            KvTransform::visible_v_tokens(record, tile_range.begin + tile + 1));
      }
    }
    int const final_event = state.softmax_event;
    int const final_stage = final_event & 1;
    Sm100FmhaBarrier::wait(softmax_local_full_barrier(storage, final_stage),
                           full_phase(final_event, 2), static_cast<uint32_t>(330 + final_stage));
    load_stats(tmem_base, final_event, stats);
    Sm100FmhaBarrier::arrive_cluster_zero(softmax_local_empty_barrier(storage, final_stage));

    Sm100FmhaBarrier::wait(o_full_barrier(storage), full_phase(state.o_event + tiles - 1, 1), 304);

    ScaleFragment total_sum;
    ScaleFragment final_scale;
    OFragment regs_o;
    ScaleFragment partial_sum;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kColumnsPerThread; ++i) {
      partial_sum[i] = stats[i];
    }
    reduce_final_sum(storage, partial_sum, total_sum, warp_group_lane);
    load_o_regs(tmem_base, regs_o);
    {
      make_final_scale(total_sum, output_scale, final_scale);
    }
    if (use_workspace_split(params) && !direct_store) {
      store_lse_to_workspace(params, batch_idx, kv_head_idx, q_token_idx, kv_split_idx, stats,
                             total_sum, warp_group_lane, state.scale_log2);
    }
    store_o_to_global(storage, params, batch_idx, kv_head_idx, q_token_idx, kv_split_idx, regs_o,
                      final_scale, warp_group_lane, direct_store);

    // O has left TMEM; release it before the merge so the next item's MMA can start.
    Sm100FmhaBarrier::arrive(o_empty_barrier(storage));
    if (merge_in_kernel) {
      merge_split_partials(
          storage, params,
          MergeItem{fmha_fwd_q_token_global_index<Traits>(params, batch_idx, q_token_idx),
                    kv_head_idx, kv_split_count},
          warp_group_lane, state);
    }
    state.softmax_event += 1;
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
