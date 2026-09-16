#pragma once

#include <climits>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "cutlass/gemm_coord.h"
#include "cutlass/kernel_hardware_info.h"
#include "sm100_fmha_q8kv4_traits.hpp"

namespace cutlass::fmha::kernel {

struct HostPrecomputedTileScheduler {
  struct Arguments {
    uint64_t *packed_work_range = nullptr;
    uint64_t *packed_work_info = nullptr;
    int *kv_tile_begin_indices = nullptr;
    int *kv_tile_end_indices = nullptr;
    int *kv_split_indices = nullptr;
  };

  struct Params {
    uint64_t *packed_work_range = nullptr;
    uint64_t *packed_work_info = nullptr;
    int num_sm = 0;
  };

  Params params;
  int work_ptr = 0;
  int work_ptr_end = 0;
  int qo_tile_idx = 0;
  int batch_idx = 0;
  int qo_head_idx = 0;
  bool is_valid_ = false;

  CUTLASS_DEVICE void load_work_item() {
    uint64_t packed = params.packed_work_info[work_ptr];
    qo_tile_idx = static_cast<int>(packed >> 32);
    qo_head_idx = static_cast<int>((packed >> 16) & 0xffffu);
    batch_idx = static_cast<int>(packed & 0xffffu);
  }

  CUTLASS_DEVICE HostPrecomputedTileScheduler(Params const &params) {
    this->params = params;
    uint64_t range = params.packed_work_range[blockIdx.x];
    work_ptr = static_cast<int>(range & 0xffffffffu);
    work_ptr_end = static_cast<int>(range >> 32);
    if (work_ptr < work_ptr_end) {
      is_valid_ = true;
      load_work_item();
    }
  }

  static Params to_underlying_arguments(Arguments const &args,
                                        cutlass::KernelHardwareInfo hw_info) {
    Params p{};
    p.packed_work_range = args.packed_work_range;
    p.packed_work_info = args.packed_work_info;
    p.num_sm = hw_info.sm_count;
    return p;
  }

  static dim3 get_grid_shape(Params const &params) { return dim3(params.num_sm); }

  CUTLASS_DEVICE bool is_valid() const { return is_valid_; }

  CUTLASS_DEVICE auto get_block_coord() {
    return cute::make_coord(qo_tile_idx, cute::_0{}, cute::make_coord(qo_head_idx, batch_idx));
  }

  CUTLASS_DEVICE int get_work_ptr() const { return work_ptr; }

  CUTLASS_DEVICE HostPrecomputedTileScheduler &operator++() {
    ++work_ptr;
    is_valid_ = work_ptr < work_ptr_end;
    if (is_valid_) {
      load_work_item();
    }
    return *this;
  }
};

} // namespace cutlass::fmha::kernel

namespace cutlass::fmha::collective {

template <class Traits> struct Sm100FmhaScheduler {
  static constexpr uint32_t kStages = 2;

  using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
  using Scheduler =
      cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<ClusterShape, kStages>;
  using Params = typename Scheduler::Params;
  using Arguments = typename Scheduler::Arguments;
  using SharedStorage = typename Scheduler::SharedStorage;
  using WorkTileInfo = typename Scheduler::WorkTileInfo;
  using Pipeline = typename Scheduler::Pipeline;
  using PipelineState = typename Pipeline::PipelineState;
  using ThrottlePipeline = typename Scheduler::ThrottlePipeline;
  using ThrottlePipelineState = typename ThrottlePipeline::PipelineState;

  struct WorkTile {
    int q_token_idx = 0;
    int kv_head_idx = 0;
    int batch_idx = 0;
    bool valid = true;
  };

  struct SplitWorkTile {
    int q_token_idx = 0;
    int kv_head_idx = 0;
    int batch_idx = 0;
    int kv_tile_begin = 0;
    int kv_tile_end = 0;
    int kv_split_idx = 0;
    int kv_split_count = 0; // segments of this item; 0 when the plan did not record it
    bool valid = false;
  };

  struct PrecomputedWorkRange {
    int begin = 0;
    int end = 0;
  };

  static Params to_underlying_arguments(int q_tiles, int kv_heads, int batches,
                                        cutlass::KernelHardwareInfo hw_info) {
    if (hw_info.sm_count <= 0) {
      hw_info.sm_count =
          cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    }

    Arguments args;
    args.max_swizzle_size = 0;
    args.raster_order = Scheduler::RasterOrderOptions::AlongM;

    Params params;
    params.initialize(dim3(1, kv_heads, batches * q_tiles), cutlass::gemm::GemmCoord(1, 1, 1),
                      hw_info, args.max_swizzle_size, args.raster_order);
    return params;
  }

  CUTLASS_DEVICE static WorkTile work_tile_from_info(WorkTileInfo const &info,
                                                     int q_tokens_per_batch) {
    int const expanded_batch_idx = info.L_idx;
    int const q_token_idx = expanded_batch_idx & (q_tokens_per_batch - 1);
    int const q_token_shift = 31 - __clz(q_tokens_per_batch);
    int const batch_idx = expanded_batch_idx >> q_token_shift;
    return WorkTile{q_token_idx, info.N_idx, batch_idx, info.is_valid_tile};
  }

  template <class Params>
  CUTLASS_DEVICE static PrecomputedWorkRange precomputed_work_range(Params const &params,
                                                                    int cta_idx) {
    uint64_t const packed = params.packed_work_range_ptr[cta_idx];
    return PrecomputedWorkRange{static_cast<int>(packed & 0xffffffffu),
                                static_cast<int>(packed >> 32)};
  }

  template <class Params>
  CUTLASS_DEVICE static SplitWorkTile split_work_tile(Params const &params, int work_idx) {
    uint64_t const packed = params.packed_work_info_ptr[work_idx];
    int const qo_tile_idx = static_cast<int>(packed >> 32);
    int const qo_head_idx = static_cast<int>((packed >> 16) & 0xffffu);
    int const batch_idx = static_cast<int>(packed & 0xffffu);
    // The device plan kernel records KV ranges in plan tiles; the balanced (stream-K) host
    // schedule, recognizable by its segment counts, records them in kernel tiles.
    int const kPlanToKernelTile =
        params.kv_split_count_ptr != nullptr ? 1 : Traits::kSplitPlanTileKv / Traits::kTileKv;
    bool const split_kv = params.num_kv_splits > 1;
    return SplitWorkTile{
        qo_tile_idx,
        qo_head_idx,
        batch_idx,
        split_kv ? params.kv_tile_begin_ptr[work_idx] * kPlanToKernelTile : 0,
        split_kv ? params.kv_tile_end_ptr[work_idx] * kPlanToKernelTile : INT_MAX,
        split_kv ? params.kv_split_ptr[work_idx] : 0,
        split_kv && params.kv_split_count_ptr != nullptr ? params.kv_split_count_ptr[work_idx] : 0,
        true};
  }
};

} // namespace cutlass::fmha::collective
