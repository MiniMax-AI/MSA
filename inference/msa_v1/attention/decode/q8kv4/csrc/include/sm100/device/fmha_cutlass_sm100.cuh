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
  static constexpr bool kOutputMxfp8 = cute::is_same_v<DTypeOut, cutlass::float_e4m3_t>;
  using Traits = typename cutlass::fmha::collective::Sm100FmhaQ8Kv4TraitSelector<
      IsSplitKV, kSparseAttnMode, IsQ8KV4, SparseTopK, FixedQTokensPerBatch,
      pack_factor_of<ActiveMask>::value, kOutputMxfp8>::type;
  using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
      DTypeIn, float, float, TileShapeQK, TileShapePV, void, void, void, ActiveMask, ThreadShape,
      IsSplitKV, KVPageSize, kSparseAttnMode, IsQ8KV4, SparseTopK, FixedQTokensPerBatch,
      kOutputMxfp8>;
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
    arguments.epilogue.o_sf_ptr = params.o_sf_ptr;
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

template <typename DTypeIn, typename DTypeOut, typename IdType, class TileShapeQK,
          class TileShapePV, class ActiveMask, class ThreadShape = Shape<_2, _1, _1>,
          bool IsSplitKV = false, bool SingleSoftmaxWarpGroup = false, int KVPageSize = -1,
          cutlass::fmha::collective::SparseAttnMode kSparseAttnMode =
              cutlass::fmha::collective::SparseAttnMode::Off,
          bool IsQ8KV4 = false, int SparseTopK = 16, int FixedQTokensPerBatch = 0>
cudaError_t
run_fmha_fwd(void *workspace_buffer, DTypeIn *q, DTypeIn *k, DTypeIn *v, IdType *qo_segment_lens,
             IdType *kv_segment_lens, IdType *qo_segment_offsets, IdType *kv_segment_offsets,
             uint64_t *packed_work_range, uint64_t *packed_work_info, DTypeOut *o,
             int mask_mode_code, double sm_scale, int num_qo_heads, int num_kv_heads,
             int head_dim_qk, int head_dim_vo, int q_stride_n, int q_stride_h, int k_stride_n,
             int k_stride_h, int v_stride_n, int v_stride_h, int batch_size, int total_qo_len,
             int total_kv_len, int max_qo_len, IdType *qo_offsets, const void *k_scale_ptr,
             const void *v_scale_ptr, cudaStream_t stream, int num_kv_splits = 1,
             IdType *kv_tile_begin_indices = nullptr, IdType *kv_tile_end_indices = nullptr,
             IdType *kv_split_indices = nullptr, float *ptr_lse_accum = nullptr,
             IdType *kv_indices = nullptr, int kv_page_stride = 0, int total_page_num = 0,
             IdType *kv_block_indexes = nullptr, int kv_block_num = 0, int pack_factor = 1,
             int q_tokens_per_batch = 0, int q_stride_n_original = 0, int q_stride_h_original = 0,
             int h_r_original = 0, int total_qo_len_orig = 0, void *o_direct = nullptr,
             int num_qo_heads_orig = 0, bool tma_direct_o_enabled = false, int num_ctas = 0,
             IdType *kv_split_count_indices = nullptr, int *merge_counters = nullptr,
             int64_t merge_item_base = 0, void *o_sf = nullptr) {
  FMHACutlassSM100Params params{};
  params.workspace_buffer_ptr = workspace_buffer;
  params.q_ptr = q;
  params.k_ptr = k;
  params.v_ptr = v;
  params.qo_segment_lens_ptr = reinterpret_cast<int *>(qo_segment_lens);
  params.kv_segment_lens_ptr = reinterpret_cast<int *>(kv_segment_lens);
  params.qo_segment_offsets_ptr = reinterpret_cast<int *>(qo_segment_offsets);
  params.kv_segment_offsets_ptr = reinterpret_cast<int *>(kv_segment_offsets);
  params.packed_work_range_ptr = packed_work_range;
  params.packed_work_info_ptr = packed_work_info;
  params.o_ptr = IsSplitKV ? nullptr : o;
  params.mask_mode_code = mask_mode_code;
  params.sm_scale = static_cast<float>(sm_scale);
  params.num_qo_heads = num_qo_heads;
  params.num_kv_heads = num_kv_heads;
  params.head_dim_qk = head_dim_qk;
  params.head_dim_vo = head_dim_vo;
  params.q_stride_n = q_stride_n;
  params.q_stride_h = q_stride_h;
  params.k_stride_n = k_stride_n;
  params.k_stride_h = k_stride_h;
  params.v_stride_n = v_stride_n;
  params.v_stride_h = v_stride_h;
  params.batch_size = batch_size;
  params.total_qo_len = total_qo_len;
  params.total_kv_len = total_kv_len;
  params.max_qo_len = max_qo_len;
  params.qo_offsets_ptr = reinterpret_cast<int *>(qo_offsets);
  params.stream = stream;
  params.num_kv_splits = num_kv_splits;
  params.kv_tile_begin_ptr = reinterpret_cast<int *>(kv_tile_begin_indices);
  params.kv_tile_end_ptr = reinterpret_cast<int *>(kv_tile_end_indices);
  params.kv_split_ptr = reinterpret_cast<int *>(kv_split_indices);
  params.kv_split_count_ptr = reinterpret_cast<int *>(kv_split_count_indices);
  params.merge_counter_ptr = merge_counters;
  params.merge_item_base = merge_item_base;
  params.workspace_o_ptr = IsSplitKV ? o : nullptr;
  params.workspace_lse_ptr = ptr_lse_accum;
  params.kv_indices_ptr = reinterpret_cast<int *>(kv_indices);
  params.kv_page_stride = kv_page_stride;
  params.total_page_num = total_page_num;
  params.kv_block_indexes_ptr = reinterpret_cast<int *>(kv_block_indexes);
  params.kv_block_num = kv_block_num;
  params.pack_factor = pack_factor;
  params.q_tokens_per_batch = q_tokens_per_batch;
  params.h_r_original = h_r_original;
  params.q_stride_n_original = q_stride_n_original;
  params.q_stride_h_original = q_stride_h_original;
  params.total_qo_len_orig = total_qo_len_orig;
  params.o_direct_ptr = o_direct;
  params.o_sf_ptr = o_sf;
  params.num_qo_heads_orig = num_qo_heads_orig;
  params.num_ctas = num_ctas;
  params.tma_direct_o_enabled = tma_direct_o_enabled;
  params.k_scale_ptr = const_cast<void *>(k_scale_ptr);
  params.v_scale_ptr = const_cast<void *>(v_scale_ptr);
  return run_fmha_fwd<DTypeIn, DTypeOut, IdType, TileShapeQK, TileShapePV, ActiveMask, ThreadShape,
                      IsSplitKV, SingleSoftmaxWarpGroup, KVPageSize, kSparseAttnMode, IsQ8KV4,
                      SparseTopK, FixedQTokensPerBatch>(params);
}

} // namespace flashinfer
