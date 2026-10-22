#pragma once

#include <cfloat>
#include <climits>
#include <cstdint>

#include "attn/kernel/include/sm100_m3_decode_problem.hpp"
#include "common/kernel/include/sm100_m3_decode_fp4_transform.cuh"
#include "common/kernel/include/sm100_m3_decode_pipeline.hpp"
#include "common/kernel/include/sm100_m3_decode_storage.hpp"
#include "common/kernel/include/sm100_m3_decode_traits.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"

namespace cutlass::m3_decode::collective {

template <class Traits> struct Sm100M3DecodeAttentionCorrectionReduction {
  using Storage = SharedStorage<Traits>;
  using Params = Sm100M3DecodeKernelParams<Traits>;
  using Barriers = BarrierLayout<Traits>;
  using Tmem = TmemLayout<Traits>;

  static constexpr int kStoreOBarrierId = 3;
  static constexpr int kFinalReductionBarrierId = 4;
  static constexpr uint32_t kTmemRow16Offset = 0x100000u;

  struct State {
    int softmax_event = 0;
    int o_event = 0;
  };

  struct StatsFragment {
    float data[8];

    CUTLASS_DEVICE float &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE float const &operator[](int idx) const { return data[idx]; }
  };

  struct ScaleFragment {
    float data[4];

    CUTLASS_DEVICE float &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE float const &operator[](int idx) const { return data[idx]; }
  };

  struct OFragment {
    float data[16];

    CUTLASS_DEVICE float &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE float const &operator[](int idx) const { return data[idx]; }
  };

  struct Bf16OFragment {
    uint32_t data[8];

    CUTLASS_DEVICE uint32_t &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE uint32_t const &operator[](int idx) const { return data[idx]; }
  };

  struct TmemWordFragment {
    uint32_t data[8];

    CUTLASS_DEVICE uint32_t &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE uint32_t const &operator[](int idx) const { return data[idx]; }
  };

  struct OutputAccumulatorFragment {
    float data[8];

    CUTLASS_DEVICE float &operator[](int idx) { return data[idx]; }

    CUTLASS_DEVICE float const &operator[](int idx) const { return data[idx]; }
  };

  CUTLASS_DEVICE static uint64_t *softmax_local_full_barrier(Storage &storage, int stage) {
    return storage.attention_pipelines().ptr(Barriers::kSoftmaxLocalFullArv128 + stage);
  }

  CUTLASS_DEVICE static uint64_t *softmax_local_empty_barrier(Storage &storage, int stage) {
    return storage.attention_pipelines().ptr(Barriers::kSoftmaxLocalEmptyArv128 + stage);
  }

  CUTLASS_DEVICE static uint64_t *o_full_barrier(Storage &storage) {
    return storage.attention_pipelines().ptr(Barriers::kOFullArv1);
  }

  CUTLASS_DEVICE static uint64_t *o_empty_barrier(Storage &storage) {
    return storage.attention_pipelines().ptr(Barriers::kOEmptyArv128);
  }

  CUTLASS_DEVICE static int kv_length_for_batch(Params const &params, int batch_idx,
                                                int kv_head_idx, int q_token_idx) {
    return m3_decode_kv_length_for_batch<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
  }

  CUTLASS_DEVICE static int kv_tile_count(Params const &params, int batch_idx, int kv_head_idx,
                                          int q_token_idx) {
    return m3_decode_kv_tile_count_for_batch<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
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

  CUTLASS_DEVICE static void store_o_sync() {
    Sm100M3DecodeNamedBarrier::sync(128, kStoreOBarrierId);
  }

  CUTLASS_DEVICE static uint32_t stats_stage_tmem_col(int event) {
    return (event & 1) == 0 ? Tmem::kStats0 : Tmem::kStats1;
  }

  CUTLASS_DEVICE static void load_stats(uint32_t tmem_base, int event, StatsFragment &stats) {
    TmemWordFragment raw;
    tmem_load_32x32b_x8(raw.data, tmem_base + stats_stage_tmem_col(event));
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      stats[i] = uint_as_float(raw[i]);
    }
    fence_tmem_load();
  }

  CUTLASS_DEVICE static void load_o_regs(uint32_t tmem_base, OFragment &regs_o) {
    TmemWordFragment raw0;
    TmemWordFragment raw1;
    uint32_t const tmem_col = tmem_base + Tmem::kO0;
    tmem_load_16x256b_x2(raw0.data, tmem_col);
    tmem_load_16x256b_x2(raw1.data, tmem_col + kTmemRow16Offset);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      regs_o[i] = uint_as_float(raw0[i]);
      regs_o[i + 8] = uint_as_float(raw1[i]);
    }
    fence_tmem_load();
  }

  CUTLASS_DEVICE static void store_o_regs(uint32_t tmem_base, OFragment const &regs_o) {
    TmemWordFragment raw0;
    TmemWordFragment raw1;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      raw0[i] = float_as_uint(regs_o[i]);
      raw1[i] = float_as_uint(regs_o[i + 8]);
    }
    uint32_t const tmem_col = tmem_base + Tmem::kO0;
    tmem_store_16x256b_x2(tmem_col, raw0.data);
    tmem_store_16x256b_x2(tmem_col + kTmemRow16Offset, raw1.data);
    fence_tmem_store();
  }

  CUTLASS_DEVICE static void scale_o_regs(OFragment &regs_o, ScaleFragment const &scale) {
    float2 const scale01 = make_f32x2(scale[0], scale[1]);
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

  CUTLASS_DEVICE static void make_rescale(StatsFragment const &stats, float scale_softmax_log2,
                                          ScaleFragment &scale) {
    float2 const scale_softmax2 = make_f32x2(scale_softmax_log2, scale_softmax_log2);
    float2 scale01 = fmul2(fadd2(make_f32x2(stats[0], stats[1]), make_f32x2(-stats[4], -stats[5])),
                           scale_softmax2);
    scale01.x = exp2f(scale01.x);
    scale01.y = exp2f(scale01.y);
    scale[0] = scale01.x;
    scale[1] = scale01.y;

    float2 scale23 = fmul2(fadd2(make_f32x2(stats[2], stats[3]), make_f32x2(-stats[6], -stats[7])),
                           scale_softmax2);
    scale23.x = exp2f(scale23.x);
    scale23.y = exp2f(scale23.y);
    scale[2] = scale23.x;
    scale[3] = scale23.y;
  }

  CUTLASS_DEVICE static void reduce_final_sum(Storage &storage, ScaleFragment const &partial_sum,
                                              ScaleFragment &total_sum, int warp_group_lane) {
    float *smem = storage.smem_corr_red1.data;
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
    Sm100M3DecodeNamedBarrier::sync(128, kFinalReductionBarrierId);

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

  CUTLASS_DEVICE static int packed_row_base(Params const &params, int batch_idx, int q_token_idx) {
    return m3_decode_packed_row_base<Traits>(params, batch_idx, q_token_idx);
  }

  CUTLASS_DEVICE static int stats_local_row(int warp_group_lane, int stats_idx) {
    // Stats fragments follow the 32x32 TMEM row layout: each of the first
    // four lanes owns two rows in the lower half and two rows in the upper half.
    return ((stats_idx >> 1) << 3) + (warp_group_lane << 1) + (stats_idx & 1);
  }

  CUTLASS_DEVICE static void pack_o_bf16(OFragment const &regs_o, Bf16OFragment &regs_bf16) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      regs_bf16[i] = pack_float2_to_bfloat16(regs_o[i * 2], regs_o[i * 2 + 1]);
    }
  }

  CUTLASS_DEVICE static void pack_scaled_o_bf16(OFragment const &regs_o, ScaleFragment const &scale,
                                                Bf16OFragment &regs_bf16) {
    float2 const scale01 = make_f32x2(scale[0], scale[1]);
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

  CUTLASS_DEVICE static int cluster_partial_o_offset(int split_idx, int rank_local_row,
                                                     int rows_per_rank, int col_bytes) {
    constexpr int kDstRowBytes = Traits::kHeadDim * static_cast<int>(sizeof(cutlass::bfloat16_t));
    return (split_idx * rows_per_rank + rank_local_row) * kDstRowBytes + col_bytes;
  }

  CUTLASS_DEVICE static void copy_o_smem_to_cluster(Storage &storage, int kv_split_idx,
                                                    int warp_group_lane) {
    static_assert(Traits::kHeadDim == 128, "cluster output copy is specialized for D=128.");
    static_assert(Traits::kHeadGroup == 16,
                  "cluster output copy is specialized for 16 Q heads per KV head.");

    constexpr int kWordBytes = static_cast<int>(sizeof(uint32_t));
    constexpr int kSmemRowBytes = 128;
    constexpr int kDstRowBytes = Traits::kHeadDim * static_cast<int>(sizeof(cutlass::bfloat16_t));
    constexpr int kSmemRows = (Traits::kHeadGroup * kDstRowBytes) / kSmemRowBytes;
    constexpr int kRowsPerWarp = kSmemRows / Traits::kNumSoftmaxWarps;
    static_assert(kDstRowBytes == 256, "BF16 D=128 output rows are 256B.");
    static_assert(kSmemRows == 32, "BF16 D=128 output tile has 32 128B rows.");
    static_assert(kSmemRows % Traits::kNumSoftmaxWarps == 0,
                  "softmax warps must evenly cover the output half-rows.");

    uint8_t *smem_base = storage.smem_o.data;
    uint32_t const barrier_addr =
        cute::cast_smem_ptr_to_uint(storage.cluster_reduction_barrier_ptr());
    int const warp_idx = warp_group_lane / cutlass::NumThreadsPerWarp;
    int const lane_idx = warp_group_lane & (cutlass::NumThreadsPerWarp - 1);

    // One warp copies one 128B half-row per iteration.  All lanes issue the
    // same 32-bit remote store to contiguous addresses, giving the cluster
    // fabric a full-row store pattern while keeping the row-major mailbox.
    CUTLASS_PRAGMA_UNROLL
    for (int row_iter = 0; row_iter < kRowsPerWarp; ++row_iter) {
      int const smem_row = row_iter * Traits::kNumSoftmaxWarps + warp_idx;
      int const smem_col = lane_idx * kWordBytes;
      int const base_offset = smem_row * kSmemRowBytes + smem_col;
      int const load_offset = base_offset ^ ((smem_row & 7) * 16);
      int const dst_row = smem_row & (Traits::kHeadGroup - 1);
      int const dst_col = (smem_row / Traits::kHeadGroup) * kSmemRowBytes + smem_col;
      constexpr int rows_per_rank = Traits::kHeadGroup / Traits::kNumKvSplits;
      int const dst_rank = dst_row / rows_per_rank;
      int const remote_row = dst_row - dst_rank * rows_per_rank;
      int const dst_offset =
          cluster_partial_o_offset(kv_split_idx, remote_row, rows_per_rank, dst_col);

      uint32_t const word = *reinterpret_cast<uint32_t const *>(smem_base + load_offset);
      uint32_t const dst_addr =
          cute::cast_smem_ptr_to_uint(storage.smem_cluster_reduction.partial_o + dst_offset);
      cute::store_shared_remote(word, dst_addr, barrier_addr, dst_rank);
    }
  }

  CUTLASS_DEVICE static void store_stats_to_cluster(Storage &storage, int kv_split_idx,
                                                    StatsFragment const &stats,
                                                    ScaleFragment const &total_sum,
                                                    int warp_group_lane) {
    if (warp_group_lane >= 4) {
      return;
    }

    uint32_t const barrier_addr =
        cute::cast_smem_ptr_to_uint(storage.cluster_reduction_barrier_ptr());
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      int const row = stats_local_row(warp_group_lane, i);
      constexpr int rows_per_rank = Traits::kHeadGroup / Traits::kNumKvSplits;
      int const dst_rank = row / rows_per_rank;
      int const remote_row = row - dst_rank * rows_per_rank;
      int const stats_offset =
          (kv_split_idx * rows_per_rank + remote_row) * Traits::kClusterReductionStatsPerRow;
      uint32_t const dst_addr =
          cute::cast_smem_ptr_to_uint(storage.smem_cluster_reduction.stats + stats_offset);
      cute::store_shared_remote(float_as_uint(stats[i + 4]), dst_addr, barrier_addr, dst_rank);
      cute::store_shared_remote(float_as_uint(total_sum[i]), dst_addr + sizeof(float), barrier_addr,
                                dst_rank);
    }
  }

  CUTLASS_DEVICE static void store_neutral_to_cluster(Storage &storage, int kv_split_idx,
                                                      int warp_group_lane) {
    static_assert(Traits::kHeadDim == 128, "cluster neutral output copy is specialized for D=128.");
    static_assert(Traits::kHeadGroup == 16,
                  "cluster neutral output copy is specialized for 16 Q heads.");

    constexpr int kWordBytes = static_cast<int>(sizeof(uint32_t));
    constexpr int kSmemRowBytes = 128;
    constexpr int kDstRowBytes = Traits::kHeadDim * static_cast<int>(sizeof(cutlass::bfloat16_t));
    constexpr int kSmemRows = (Traits::kHeadGroup * kDstRowBytes) / kSmemRowBytes;
    constexpr int kRowsPerWarp = kSmemRows / Traits::kNumSoftmaxWarps;
    static_assert(kDstRowBytes == 256, "BF16 D=128 output rows are 256B.");
    static_assert(kSmemRows == 32, "BF16 D=128 output tile has 32 128B rows.");
    static_assert(kSmemRows % Traits::kNumSoftmaxWarps == 0,
                  "softmax warps must evenly cover the output half-rows.");
    uint32_t const barrier_addr =
        cute::cast_smem_ptr_to_uint(storage.cluster_reduction_barrier_ptr());
    constexpr int rows_per_rank = Traits::kHeadGroup / Traits::kNumKvSplits;
    int const warp_idx = warp_group_lane / cutlass::NumThreadsPerWarp;
    int const lane_idx = warp_group_lane & (cutlass::NumThreadsPerWarp - 1);

    // Match the live-output copy pattern so neutral rows use the same
    // contiguous remote-store address sequence.
    CUTLASS_PRAGMA_UNROLL
    for (int row_iter = 0; row_iter < kRowsPerWarp; ++row_iter) {
      int const smem_row = row_iter * Traits::kNumSoftmaxWarps + warp_idx;
      int const dst_row = smem_row & (Traits::kHeadGroup - 1);
      int const dst_col = (smem_row / Traits::kHeadGroup) * kSmemRowBytes + lane_idx * kWordBytes;
      int const dst_rank = dst_row / rows_per_rank;
      int const remote_row = dst_row - dst_rank * rows_per_rank;
      int const dst_offset =
          cluster_partial_o_offset(kv_split_idx, remote_row, rows_per_rank, dst_col);
      uint32_t const dst_addr =
          cute::cast_smem_ptr_to_uint(storage.smem_cluster_reduction.partial_o + dst_offset);
      cute::store_shared_remote(0u, dst_addr, barrier_addr, dst_rank);
    }

    if (warp_group_lane >= 4) {
      return;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      int const row = stats_local_row(warp_group_lane, i);
      int const dst_rank = row / rows_per_rank;
      int const remote_row = row - dst_rank * rows_per_rank;
      int const stats_offset =
          (kv_split_idx * rows_per_rank + remote_row) * Traits::kClusterReductionStatsPerRow;
      uint32_t const dst_addr =
          cute::cast_smem_ptr_to_uint(storage.smem_cluster_reduction.stats + stats_offset);
      cute::store_shared_remote(float_as_uint(-FLT_MAX), dst_addr, barrier_addr, dst_rank);
      cute::store_shared_remote(0u, dst_addr + sizeof(float), barrier_addr, dst_rank);
    }
  }

  CUTLASS_DEVICE static void complete_cluster_reduction_transactions(Storage &storage,
                                                                     int warp_group_lane) {
    if (warp_group_lane != 0) {
      return;
    }
    constexpr uint32_t kCompletionBytes = static_cast<uint32_t>(
        Traits::kClusterReductionTransactionBytes - Traits::kClusterReductionPayloadBytes);
    cutlass::arch::ClusterTransactionBarrier::complete_transaction(
        storage.cluster_reduction_barrier_ptr(),
        static_cast<uint32_t>(cute::block_rank_in_cluster()), kCompletionBytes);
  }
};

} // namespace cutlass::m3_decode::collective
