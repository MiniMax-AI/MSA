// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "decode_attention_params.hpp"
#include "fmha.hpp"
#include "fmha_fusion.hpp"
#include "fmha_options.hpp"
#include "sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
#include "sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"

namespace flashinfer {

using namespace cute;
using namespace cutlass::fmha::collective;

template <typename DTypeIn, typename DTypeOut, typename IdType, class TileShapeQK,
          class TileShapePV, class ActiveMask, class ThreadShape = Shape<_2, _1, _1>,
          bool IsSplitKV = false, bool SingleSoftmaxWarpGroup = false, int KVPageSize = -1,
          cutlass::fmha::collective::SparseAttnMode kSparseAttnMode =
              cutlass::fmha::collective::SparseAttnMode::Off,
          bool IsQ8KV4 = false, int SparseTopK = 16, int FixedQTokensPerBatch = 0>
struct FwdRunner {
  using Traits = typename cutlass::fmha::collective::Sm100FmhaQ8Kv4TraitSelector<
      IsSplitKV, kSparseAttnMode, IsQ8KV4, SparseTopK, FixedQTokensPerBatch,
      pack_factor_of<ActiveMask>::value>::type;
  using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
      DTypeIn, float, float, TileShapeQK, TileShapePV, void, void, void, ActiveMask, ThreadShape,
      IsSplitKV, KVPageSize, kSparseAttnMode, IsQ8KV4, SparseTopK, FixedQTokensPerBatch>;
  using Epilogue = cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<Traits>;
  using ProblemShape = cute::tuple<VariableLength, VariableLength, int,
                                   cute::tuple<cute::tuple<int, int>, int>, PerBatchOffset>;
  using TileScheduler = cutlass::fmha::kernel::HostPrecomputedTileScheduler;
  using KernelSchedule = void;
  using Operation =
      cutlass::fmha::device::FMHA<cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
          ProblemShape, Mainloop, Epilogue, TileScheduler, KernelSchedule>>;

  static constexpr int kPackFactor = cutlass::fmha::collective::pack_factor_of<ActiveMask>::value;

  static typename Operation::Arguments make_arguments(const FMHACutlassSM100Params &params) {
    int const h_r =
        params.h_r_original > 0 ? params.h_r_original : params.num_qo_heads / params.num_kv_heads;
    ProblemShape problem_shape = cute::make_tuple(
        VariableLength{params.qo_segment_lens_ptr, params.qo_segment_offsets_ptr},
        VariableLength{params.kv_segment_lens_ptr, params.kv_segment_offsets_ptr},
        params.head_dim_qk,
        cute::make_tuple(cute::make_tuple(h_r, params.num_kv_heads), params.batch_size),
        PerBatchOffset{params.qo_offsets_ptr});
    typename Operation::Arguments arguments{};
    arguments.problem_shape = problem_shape;
    arguments.mainloop.load.fmha = params;
    arguments.epilogue.o_ptr = params.o_ptr;
    arguments.epilogue.o_direct_ptr = params.o_direct_ptr;
    arguments.epilogue.total_qo_len_orig = params.total_qo_len_orig;
    arguments.epilogue.num_qo_heads_orig = params.num_qo_heads_orig;
    arguments.epilogue.tma_direct_o_enabled = params.tma_direct_o_enabled;
    arguments.tile_scheduler.packed_work_range = params.packed_work_range_ptr;
    arguments.tile_scheduler.packed_work_info = params.packed_work_info_ptr;
    arguments.tile_scheduler.kv_tile_begin_indices = params.kv_tile_begin_ptr;
    arguments.tile_scheduler.kv_tile_end_indices = params.kv_tile_end_ptr;
    arguments.tile_scheduler.kv_split_indices = params.kv_split_ptr;
    arguments.hw_info.sm_count = params.num_ctas;
    return arguments;
  }

  static cudaError_t run(const FMHACutlassSM100Params &params) {
    (void)kPackFactor;
    (void)sizeof(Mainloop);
    (void)sizeof(Epilogue);
    typename Operation::Arguments arguments = make_arguments(params);
    Operation op;
    cutlass::Status status = op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      return cudaErrorNotSupported;
    }
    status = op.initialize(arguments, params.workspace_buffer_ptr, params.stream);
    if (status != cutlass::Status::kSuccess) {
      return cudaErrorLaunchFailure;
    }
    status = op.run(params.stream);
    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorLaunchFailure;
  }

  static cudaError_t can_implement(const FMHACutlassSM100Params &params) {
    (void)kPackFactor;
    (void)sizeof(Mainloop);
    (void)sizeof(Epilogue);
    return Operation::can_implement(make_arguments(params)) == cutlass::Status::kSuccess
               ? cudaSuccess
               : cudaErrorNotSupported;
  }
};

template <typename DTypeIn, typename DTypeOut, typename IdType, class TileShapeQK,
          class TileShapePV, class ActiveMask, class ThreadShape = Shape<_2, _1, _1>,
          bool IsSplitKV = false, bool SingleSoftmaxWarpGroup = false, int KVPageSize = -1,
          cutlass::fmha::collective::SparseAttnMode kSparseAttnMode =
              cutlass::fmha::collective::SparseAttnMode::Off,
          bool IsQ8KV4 = false, int SparseTopK = 16, int FixedQTokensPerBatch = 0>
cudaError_t run_fmha_fwd(const FMHACutlassSM100Params &params) {
  return FwdRunner<DTypeIn, DTypeOut, IdType, TileShapeQK, TileShapePV, ActiveMask, ThreadShape,
                   IsSplitKV, SingleSoftmaxWarpGroup, KVPageSize, kSparseAttnMode, IsQ8KV4,
                   SparseTopK, FixedQTokensPerBatch>::run(params);
}

} // namespace flashinfer
