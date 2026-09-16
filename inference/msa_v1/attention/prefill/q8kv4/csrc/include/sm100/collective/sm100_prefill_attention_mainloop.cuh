#pragma once

#include "sm100/collective/sm100_prefill_attention_fp4_transform.cuh"
#include "sm100/common/prefill_attention_config.cuh"
#include "sm100/common/prefill_attention_math.cuh"
#include "sm100/device/prefill_attention.hpp"
#include "sm100/kernel/sm100_prefill_attention_shared_storage.cuh"

#include <cstdint>

#include <cuda_runtime.h>

#include "cute/arch/copy_sm100.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"

namespace minimax::msa_v1::attention::prefill::q8kv4 {
namespace detail {

using namespace cute;

template <class Storage>
CUTLASS_DEVICE void initialize_work_tile(Storage &storage, PrefillArguments const &arguments) {
  int const *metadata = arguments.scheduler_metadata_ptr + static_cast<int>(blockIdx.x) * 6;
  int const head_kv = metadata[0];
  int const row_linear = metadata[1];
  int const q_begin = metadata[2];
  int const q_count = metadata[3];
  int const batch = metadata[4];
  int const logical_page = metadata[5];
  int const row_start =
      arguments.k2q_row_ptr[head_kv * (arguments.total_rows + 1) + row_linear] + q_begin;
  int const q_batch_offset = arguments.cu_seqlens_q_ptr[batch];
  int const q_length = arguments.cu_seqlens_q_ptr[batch + 1] - q_batch_offset;
  int const kv_batch_offset = arguments.cu_seqlens_k_ptr[batch];
  int const kv_length = arguments.cu_seqlens_k_ptr[batch + 1] - kv_batch_offset;
  int const valid_cols = max(0, min(kPageSize, kv_length - logical_page * kPageSize));
  int physical_page = -1;
  if (logical_page >= 0 && logical_page < arguments.max_pages) {
    physical_page = arguments.page_table_ptr[batch * arguments.max_pages + logical_page];
  }

  storage.work_tile.head_kv = head_kv;
  storage.work_tile.row_start = row_start;
  storage.work_tile.q_count = q_count;
  storage.work_tile.batch = batch;
  storage.work_tile.logical_page = logical_page;
  storage.work_tile.q_batch_offset = q_batch_offset;
  storage.work_tile.q_length = q_length;
  storage.work_tile.kv_length = kv_length;
  storage.work_tile.valid_cols =
      physical_page >= 0 && physical_page < arguments.physical_pages ? valid_cols : 0;
  storage.work_tile.physical_page = physical_page;
}

template <class Storage> CUTLASS_DEVICE void initialize_pipeline_barriers(Storage &storage) {
  cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterTransactionBarrier,
                                                          kQStages>(storage.q_full,
                                                                    1 + kQueriesPerGroup);
  cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterBarrier, kQStages>(
      storage.q_empty, 1);
  cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterTransactionBarrier,
                                                          1>(&storage.k_tma_full, 1);
  cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterTransactionBarrier,
                                                          1>(&storage.v_tma_full, 1);
  cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterBarrier, 1>(
      &storage.k_dequant_full, 1);
  cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterBarrier, 1>(
      &storage.v_dequant_full, 1);
  cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterBarrier,
                                                          kScoreStages>(storage.score_full, 1);
  cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterBarrier,
                                                          kScoreStages>(storage.score_empty, 128);
  cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterBarrier,
                                                          kScoreStages>(
      storage.probability_early_full, 128);
  cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterBarrier,
                                                          kScoreStages>(storage.probability_full,
                                                                        128);
  cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterBarrier,
                                                          kScoreStages>(storage.probability_empty,
                                                                        1);
  cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterBarrier,
                                                          kScoreStages>(
      storage.probability_last_empty, 1);
  cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterBarrier,
                                                          kScoreStages>(storage.output_full, 1);
  cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterBarrier,
                                                          kScoreStages>(storage.output_empty, 128);
  cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterBarrier,
                                                          kScoreStages>(storage.stats_full, 128);
  cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterBarrier,
                                                          kScoreStages>(storage.stats_empty, 128);
}

template <class QSmemLayout, class Storage, class Params>
CUTLASS_DEVICE void run_load_warps(Storage &storage, Params const &params, int warp_idx, int lane) {
  PrefillArguments const &arguments = params.arguments;
  auto const &tma_q = params.tma_q;
  auto const &q_gmem_shape = params.q_gmem_shape;

  cutlass::arch::warpgroup_reg_dealloc<kOtherRegisters>();
  int const head_kv = storage.work_tile.head_kv;
  int const row_start = storage.work_tile.row_start;
  int const q_count = storage.work_tile.q_count;
  int const q_batch_offset = storage.work_tile.q_batch_offset;
  int const physical_page = storage.work_tile.physical_page;
  int const num_groups = (q_count + kQueriesPerGroup - 1) / kQueriesPerGroup;

  if (physical_page >= 0 && physical_page < arguments.physical_pages) {
    int const page_head = physical_page * arguments.num_kv_heads + head_kv;
    if (warp_idx == WarpSpecialization::kLoadKWarp && cute::elect_one_sync()) {
      cute::prefetch_tma_descriptor(&params.tma_packed_k);
      cute::prefetch_tma_descriptor(&params.tma_k_scale);
      cute::set_barrier_transaction_bytes(storage.k_tma_full, kPackedKvBytes + kScaleBytes);
      uint32_t const smem_data_address =
          static_cast<uint32_t>(__cvta_generic_to_shared(&storage.k_raw[0][0]));
      uint32_t const smem_scale_address =
          static_cast<uint32_t>(__cvta_generic_to_shared(&storage.k_scale[0][0]));
      uint32_t const barrier_address =
          static_cast<uint32_t>(__cvta_generic_to_shared(&storage.k_tma_full));
      uint64_t const data_descriptor = reinterpret_cast<uint64_t>(&params.tma_packed_k);
      uint64_t const scale_descriptor = reinterpret_cast<uint64_t>(&params.tma_k_scale);
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                   ".mbarrier::complete_tx::bytes"
                   " [%0], [%1, {%2, %3}], [%4];"
                   :
                   : "r"(smem_data_address), "l"(data_descriptor), "r"(0), "r"(page_head * 64),
                     "r"(barrier_address)
                   : "memory");
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                   ".mbarrier::complete_tx::bytes"
                   " [%0], [%1, {%2, %3}], [%4];"
                   :
                   : "r"(smem_scale_address), "l"(scale_descriptor), "r"(0), "r"(page_head * 8),
                     "r"(barrier_address)
                   : "memory");
    } else if (warp_idx == WarpSpecialization::kLoadVWarp && cute::elect_one_sync()) {
      cute::prefetch_tma_descriptor(&params.tma_packed_v);
      cute::prefetch_tma_descriptor(&params.tma_v_scale);
      cute::set_barrier_transaction_bytes(storage.v_tma_full, kPackedKvBytes + kScaleBytes);
      uint32_t const smem_data_address =
          static_cast<uint32_t>(__cvta_generic_to_shared(&storage.v_raw[0][0]));
      uint32_t const smem_scale_address =
          static_cast<uint32_t>(__cvta_generic_to_shared(&storage.v_scale[0][0]));
      uint32_t const barrier_address =
          static_cast<uint32_t>(__cvta_generic_to_shared(&storage.v_tma_full));
      uint64_t const data_descriptor = reinterpret_cast<uint64_t>(&params.tma_packed_v);
      uint64_t const scale_descriptor = reinterpret_cast<uint64_t>(&params.tma_v_scale);
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                   ".mbarrier::complete_tx::bytes"
                   " [%0], [%1, {%2, %3}], [%4];"
                   :
                   : "r"(smem_data_address), "l"(data_descriptor), "r"(0), "r"(page_head * 64),
                     "r"(barrier_address)
                   : "memory");
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                   ".mbarrier::complete_tx::bytes"
                   " [%0], [%1, {%2, %3}], [%4];"
                   :
                   : "r"(smem_scale_address), "l"(scale_descriptor), "r"(0), "r"(page_head * 8),
                     "r"(barrier_address)
                   : "memory");
    }
  }

  int const warp_in_group = warp_idx - WarpSpecialization::kLoadWarp;
  constexpr int kTokensPerWarp =
      (kQueriesPerGroup + WarpSpecialization::kQLoadWarps - 1) / WarpSpecialization::kQLoadWarps;
  auto thr_tma_q = tma_q.get_slice(_0{});
  auto gQ = tma_q.get_tma_tensor(q_gmem_shape);
  cutlass::arch::NamedBarrier load_barrier(WarpSpecialization::kQLoadThreads, kQLoadBarrierId);
  if (warp_idx == WarpSpecialization::kLoadWarp && cute::elect_one_sync()) {
    cute::prefetch_tma_descriptor(tma_q.get_tma_descriptor());
  }

  for (int group = 0; group < num_groups; ++group) {
    int const q_stage = group % kQStages;
    int const q_phase = (group / kQStages) & 1;
    int const producer_phase = q_phase ^ 1;
    int const metadata_stage = group & (kQMetadataStages - 1);

    if (warp_idx == WarpSpecialization::kLoadWarp) {
      cute::wait_barrier(storage.q_empty[q_stage], producer_phase);
      if (lane == 0) {
        cute::set_barrier_transaction_bytes(storage.q_full[q_stage],
                                            kQueriesPerGroup * kQHeadsPerKv * kHeadDim);
      }
      if (lane < kQueriesPerGroup) {
        int const qi = group * kQueriesPerGroup + lane;
        int packed_qsplit = 0;
        if (qi < q_count) {
          packed_qsplit =
              arguments.qsplit_indices_ptr[head_kv * arguments.qsplit_stride + row_start + qi];
        }
        storage.qsplit_indices[metadata_stage][lane] = packed_qsplit;
        cutlass::arch::ClusterBarrier::arrive(&storage.q_full[q_stage]);
      }
    }
    load_barrier.arrive_and_wait();

    if (lane == 0) {
      CUTLASS_PRAGMA_UNROLL
      for (int local_token = 0; local_token < kTokensPerWarp; ++local_token) {
        int const token = warp_in_group * kTokensPerWarp + local_token;
        if (token < kQueriesPerGroup) {
          int const qi = group * kQueriesPerGroup + token;
          int q_tile = arguments.total_q * arguments.num_kv_heads;
          if (qi < q_count) {
            int const packed_qsplit = storage.qsplit_indices[metadata_stage][token];
            int const q_abs = q_batch_offset + decode_q_idx(packed_qsplit);
            q_tile = q_abs * arguments.num_kv_heads + head_kv;
          }
          auto gQTile = local_tile(gQ, make_shape(Int<kQHeadsPerKv>{}, Int<kHeadDim>{}),
                                   make_coord(q_tile, _0{}));
          auto sQTile = make_tensor(make_smem_ptr(storage.q.begin() + q_stage * 128 * kHeadDim +
                                                  token * cosize_v<QTokenSmemLayout>),
                                    QTokenSmemLayout{});
          cute::copy(tma_q.with(reinterpret_cast<uint64_t &>(storage.q_full[q_stage])),
                     thr_tma_q.partition_S(gQTile), thr_tma_q.partition_D(sQTile));
        }
      }
    }
  }

  if (warp_idx == WarpSpecialization::kLoadWarp) {
    int const next_stage = num_groups % kQStages;
    int const next_phase = ((num_groups / kQStages) & 1) ^ 1;
    cute::wait_barrier(storage.q_empty[next_stage], next_phase);
  }
}

template <class Storage, class TiledMmaQK, class ScoreTensor, class QTensor, class KTensor>
CUTLASS_DEVICE void issue_qk_group(Storage &storage, TiledMmaQK tiled_mma_qk, ScoreTensor tS,
                                   QTensor const &tQ, KTensor const &tK, int group) {
  int const q_stage = group % kQStages;
  int const q_phase = (group / kQStages) & 1;
  int const score_stage = group % kScoreStages;
  int const score_phase = (group / kScoreStages) & 1;
  int const score_producer_phase = score_phase ^ 1;
  cute::wait_barrier(storage.q_full[q_stage], q_phase);
  cute::wait_barrier(storage.score_empty[score_stage], score_producer_phase);
  tS.data() = storage.tmem_base + kTmemScoreOffset + score_stage * kTmemScoreStride;
  issue_qk(tiled_mma_qk, tS, tQ(_, _, _, q_stage), tK(_, _, _, 0),
           &storage.score_full[score_stage]);
  cutlass::arch::umma_arrive(&storage.q_empty[q_stage]);
}

template <class Storage, class TiledMmaPV, class OutputTensor, class ProbabilityTensor,
          class VTensor>
CUTLASS_DEVICE void issue_pv_group(Storage &storage, TiledMmaPV tiled_mma_pv, OutputTensor tO,
                                   ProbabilityTensor tP, VTensor const &tV, int group) {
  int const score_stage = group % kScoreStages;
  int const score_phase = (group / kScoreStages) & 1;
  int const output_producer_phase = score_phase ^ 1;
  cute::wait_barrier(storage.probability_early_full[score_stage], score_phase);
  cute::wait_barrier(storage.output_empty[score_stage], output_producer_phase);
  tO.data() = storage.tmem_base + kTmemOutputOffset + score_stage * kTmemOutputStride;
  tP.data() = storage.tmem_base + score_stage * kTmemScoreStride + kTmemProbabilityOffset;

  tiled_mma_pv.accumulate_ = UMMA::ScaleOut::Zero;
  CUTLASS_PRAGMA_UNROLL
  for (int k_block = 0; k_block < 2; ++k_block) {
    gemm(tiled_mma_pv, tP(_, _, k_block), tV(_, _, k_block, 0), tO);
    tiled_mma_pv.accumulate_ = UMMA::ScaleOut::One;
  }
  cute::wait_barrier(storage.probability_full[score_stage], score_phase);
  CUTLASS_PRAGMA_UNROLL
  for (int k_block = 2; k_block < size<2>(tP); ++k_block) {
    gemm(tiled_mma_pv, tP(_, _, k_block), tV(_, _, k_block, 0), tO);
  }
  cutlass::arch::umma_arrive(&storage.output_full[score_stage]);
  cutlass::arch::umma_arrive(&storage.probability_empty[score_stage]);
  cutlass::arch::umma_arrive(&storage.probability_last_empty[score_stage]);
}

template <class QSmemLayout, class KSmemLayout, class VSmemLayout, class Storage, class TiledMmaQK,
          class TiledMmaPV>
CUTLASS_DEVICE void run_mma_warp(Storage &storage, TiledMmaQK tiled_mma_qk,
                                 TiledMmaPV tiled_mma_pv) {
  cutlass::arch::warpgroup_reg_dealloc<kOtherRegisters>();
  int const q_count = storage.work_tile.q_count;
  int const num_groups = (q_count + kQueriesPerGroup - 1) / kQueriesPerGroup;
  auto sQ = make_tensor(make_smem_ptr(storage.q.begin()), QSmemLayout{});
  auto sK = make_tensor(make_smem_ptr(storage.k.begin()), KSmemLayout{});
  auto sV = make_tensor(make_smem_ptr(storage.v.begin()), VSmemLayout{});
  auto thr_mma_qk = tiled_mma_qk.get_slice(0);
  auto thr_mma_pv = tiled_mma_pv.get_slice(0);
  Tensor tQ = thr_mma_qk.make_fragment_A(sQ);
  Tensor tK = thr_mma_qk.make_fragment_B(sK);
  Tensor tV = thr_mma_pv.make_fragment_B(sV);
  Tensor tS = partition_fragment_C(tiled_mma_qk, make_shape(Int<128>{}, Int<128>{}));
  Tensor tO = partition_fragment_C(tiled_mma_pv, make_shape(Int<128>{}, Int<128>{}));
  auto tP = thr_mma_pv.make_fragment_A(
      partition_shape_A(tiled_mma_pv, make_shape(Int<128>{}, Int<128>{})));

  cute::wait_barrier(storage.k_dequant_full, 0);
  issue_qk_group(storage, tiled_mma_qk, tS, tQ, tK, 0);
  if (num_groups > 1) {
    issue_qk_group(storage, tiled_mma_qk, tS, tQ, tK, 1);
  }

  cute::wait_barrier(storage.v_dequant_full, 0);
  for (int group = 2; group < num_groups; ++group) {
    issue_pv_group(storage, tiled_mma_pv, tO, tP, tV, group - 2);
    issue_qk_group(storage, tiled_mma_qk, tS, tQ, tK, group);
  }
  int const drain_begin = num_groups > 1 ? num_groups - 2 : 0;
  for (int group = drain_begin; group < num_groups; ++group) {
    issue_pv_group(storage, tiled_mma_pv, tO, tP, tV, group);
  }

  cutlass::arch::NamedBarrier tmem_barrier(WarpSpecialization::kTmemParticipantWarps * 32,
                                           kTmemBarrierId);
  tmem_barrier.arrive_and_wait();
  cute::TMEM::Allocator1Sm allocator;
  allocator.free(storage.tmem_base, kTmemColumns);
  cutlass::arch::launch_dependent_grids();
}

template <class Storage, class TiledMmaQK>
CUTLASS_DEVICE void run_softmax_warpgroup(Storage &storage, PrefillArguments const &arguments,
                                          TiledMmaQK tiled_mma_qk, int warp_idx, int tid) {
  int const score_stage = WarpSpecialization::softmax_stage(warp_idx);
  if (score_stage == 0) {
    cutlass::arch::warpgroup_reg_alloc<kSoftmax0Registers>();
  } else {
    cutlass::arch::warpgroup_reg_alloc<kSoftmax1Registers>();
  }
  int const warp_base = WarpSpecialization::softmax_warp_base(score_stage);
  int const group_thread = tid - warp_base * 32;
  int const physical_page = storage.work_tile.physical_page;
  bool const valid_page = physical_page >= 0 && physical_page < arguments.physical_pages;

  // The two softmax warpgroups perform KV dequant before consuming their
  // respective score stages: stage 0 owns K and stage 1 owns V.
  if (score_stage == 0) {
    if (valid_page) {
      cute::wait_barrier(storage.k_tma_full, 0);
      dequant_fp4_tile_to_fp8_smem<kPageSize, kHeadDim, 16>(
          &storage.k_raw[0][0], &storage.k_scale[0][0],
          reinterpret_cast<uint8_t *>(storage.k.begin()), group_thread);
    } else {
      clear_fp8_tile_smem<kPageSize, kHeadDim>(reinterpret_cast<uint8_t *>(storage.k.begin()),
                                               group_thread);
    }
    cutlass::arch::NamedBarrier dequant_barrier(WarpSpecialization::kSoftmaxWarps * 32,
                                                kKvDequantKBarrierId);
    dequant_barrier.arrive_and_wait();
    if (group_thread == 0) {
      cutlass::arch::ClusterBarrier::arrive(&storage.k_dequant_full);
    }
  } else {
    if (valid_page) {
      cute::wait_barrier(storage.v_tma_full, 0);
      dequant_fp4_tile_to_fp8_smem<kPageSize, kHeadDim, 16>(
          &storage.v_raw[0][0], &storage.v_scale[0][0],
          reinterpret_cast<uint8_t *>(storage.v.begin()), group_thread);
    } else {
      clear_fp8_tile_smem<kPageSize, kHeadDim>(reinterpret_cast<uint8_t *>(storage.v.begin()),
                                               group_thread);
    }
    cutlass::arch::NamedBarrier dequant_barrier(WarpSpecialization::kSoftmaxWarps * 32,
                                                kKvDequantVBarrierId);
    dequant_barrier.arrive_and_wait();
    if (group_thread == 0) {
      cutlass::arch::ClusterBarrier::arrive(&storage.v_dequant_full);
    }
  }

  int const q_count = storage.work_tile.q_count;
  int const logical_page = storage.work_tile.logical_page;
  int const q_length = storage.work_tile.q_length;
  int const kv_length = storage.work_tile.kv_length;
  int const valid_cols = storage.work_tile.valid_cols;
  int const num_groups = (q_count + kQueriesPerGroup - 1) / kQueriesPerGroup;
  int const stage_groups = (num_groups + 1 - score_stage) / kScoreStages;

  for (int iteration = 0; iteration < stage_groups; ++iteration) {
    int const group = iteration * kScoreStages + score_stage;
    int const score_phase = iteration & 1;
    int const producer_phase = score_phase ^ 1;
    int const metadata_stage = group & (kQMetadataStages - 1);
    int const valid_queries = min(kQueriesPerGroup, q_count - group * kQueriesPerGroup);
    cute::wait_barrier(storage.score_full[score_stage], score_phase);

    int const token = group_thread / kQHeadsPerKv;
    int visible = 0;
    if (token < valid_queries) {
      int const packed_qsplit = storage.qsplit_indices[metadata_stage][token];
      int const q_idx = decode_q_idx(packed_qsplit);
      int const causal_position = kv_length - q_length + q_idx;
      int const causal_cols = causal_position - logical_page * kPageSize + 1;
      visible = max(0, min(valid_cols, causal_cols));
    }

    cute::wait_barrier(storage.probability_empty[score_stage], producer_phase);
    cute::wait_barrier(storage.probability_last_empty[score_stage], producer_phase);
    float row_sum;
    float row_max;
    softmax_to_tmem(
        tiled_mma_qk, storage.tmem_base + kTmemScoreOffset + score_stage * kTmemScoreStride,
        storage.tmem_base + kTmemProbabilityOffset + score_stage * kTmemScoreStride,
        arguments.softmax_scale_log2, visible, &storage.probability_early_full[score_stage],
        &storage.probability_full[score_stage], row_sum, row_max);

    cute::wait_barrier(storage.stats_empty[score_stage], producer_phase);
    storage.row_sum[score_stage][group_thread] = row_sum;
    storage.row_max[score_stage][group_thread] = row_max;
    cutlass::arch::fence_view_async_shared();
    cutlass::arch::ClusterBarrier::arrive(&storage.stats_full[score_stage]);
    cutlass::arch::ClusterBarrier::arrive(&storage.score_empty[score_stage]);
  }

  cutlass::arch::NamedBarrier tmem_barrier(WarpSpecialization::kTmemParticipantWarps * 32,
                                           kTmemBarrierId);
  tmem_barrier.arrive();
}

template <class Storage, class TiledMmaPV>
CUTLASS_DEVICE void run_epilogue_warpgroup(Storage &storage, PrefillArguments const &arguments,
                                           TiledMmaPV tiled_mma_pv, int tid) {
  cutlass::arch::warpgroup_reg_dealloc<kEpilogueRegisters>();
  int const head_kv = storage.work_tile.head_kv;
  int const q_count = storage.work_tile.q_count;
  int const q_batch_offset = storage.work_tile.q_batch_offset;
  int const num_groups = (q_count + kQueriesPerGroup - 1) / kQueriesPerGroup;
  Tensor tO = partition_fragment_C(tiled_mma_pv, make_shape(Int<128>{}, Int<128>{}));
  Tensor cO = make_identity_tensor(make_shape(Int<128>{}, Int<128>{}));
  int const group_thread = tid - WarpSpecialization::kEpilogueWarp * 32;
  cutlass::arch::NamedBarrier epilogue_barrier(WarpSpecialization::kEpilogueWarps * 32,
                                               kEpilogueBarrierId);

  for (int group = 0; group < num_groups; ++group) {
    int const score_stage = group % kScoreStages;
    int const score_phase = (group / kScoreStages) & 1;
    int const metadata_stage = group & (kQMetadataStages - 1);
    cute::wait_barrier(storage.output_full[score_stage], score_phase);
    cute::wait_barrier(storage.stats_full[score_stage], score_phase);
    Tensor tOCurrent = tO;
    tOCurrent.data() = storage.tmem_base + kTmemOutputOffset + score_stage * kTmemOutputStride;
    store_partial_output(tiled_mma_pv, tOCurrent, cO, arguments.o_partial_ptr,
                         arguments.lse_partial_ptr, arguments.total_q, arguments.num_q_heads,
                         arguments.softmax_scale_log2, head_kv, q_count, q_batch_offset, group,
                         group_thread, storage.qsplit_indices[metadata_stage],
                         storage.row_sum[score_stage], storage.row_max[score_stage]);
    epilogue_barrier.arrive_and_wait();
    cutlass::arch::ClusterBarrier::arrive(&storage.output_empty[score_stage]);
    cutlass::arch::ClusterBarrier::arrive(&storage.stats_empty[score_stage]);
  }

  cutlass::arch::NamedBarrier tmem_barrier(WarpSpecialization::kTmemParticipantWarps * 32,
                                           kTmemBarrierId);
  tmem_barrier.arrive();
}

} // namespace detail
} // namespace minimax::msa_v1::attention::prefill::q8kv4
