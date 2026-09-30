#pragma once

#include "sm100/collective/sm100_prefill_attention_mainloop.cuh"

namespace minimax::msa_v1::attention::prefill::q8kv4::detail {

// Warp roles:
//   0-3: softmax stage 0 and K dequant
//   4-7: softmax stage 1 and V dequant
//   8-11: partial-output epilogue
//   12: UMMA issuer and TMEM owner
//   13-15: Q producers; warps 13 and 14 also issue K and V TMA
template <class Storage, class TiledMmaQK, class TiledMmaPV,
          class QSmemLayout, class KSmemLayout, class VSmemLayout,
          class TmaQ, class QGmemShape>
__global__ __launch_bounds__(WarpSpecialization::kThreads, 1)
void prefill_attention_kernel(
    __grid_constant__ const KernelParams<TmaQ, QGmemShape> params) {
  PrefillArguments const& arguments = params.arguments;
  // PDL may start this kernel before the schedule producer has published its metadata.
  cutlass::arch::wait_on_dependent_grids();
  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  extern __shared__ char shared_memory[];
  Storage& storage = *reinterpret_cast<Storage*>(shared_memory);
  int const tid = static_cast<int>(threadIdx.x);
  int const warp_idx = tid / 32;
  int const lane = tid % 32;

  bool const active =
      static_cast<int>(blockIdx.x) < arguments.work_count_ptr[0];
  if (!active) {
    if (WarpSpecialization::is_mma_warp(warp_idx)) {
      cute::TMEM::Allocator1Sm allocator;
      allocator.release_allocation_lock();
    }
    if (tid == 0) {
      cutlass::arch::launch_dependent_grids();
    }
    return;
  }

  if (tid == 0) {
    initialize_work_tile(storage, arguments);
  }
  if (warp_idx == 0) {
    initialize_pipeline_barriers(storage);
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  cutlass::arch::NamedBarrier tmem_barrier(
      WarpSpecialization::kTmemParticipantWarps * 32, kTmemBarrierId);
  if (WarpSpecialization::is_mma_warp(warp_idx)) {
    cute::TMEM::Allocator1Sm allocator;
    allocator.allocate(kTmemColumns, &storage.tmem_base);
    allocator.release_allocation_lock();
  }
  if (warp_idx < WarpSpecialization::kLoadWarp) {
    tmem_barrier.arrive_and_wait();
  }

  if (WarpSpecialization::is_load_warp(warp_idx)) {
    run_load_warps<QSmemLayout>(storage, params, warp_idx, lane);
    return;
  }
  if (WarpSpecialization::is_mma_warp(warp_idx)) {
    run_mma_warp<QSmemLayout, KSmemLayout, VSmemLayout>(
        storage, tiled_mma_qk, tiled_mma_pv);
    return;
  }
  if (WarpSpecialization::is_softmax_warp(warp_idx)) {
    run_softmax_warpgroup(
        storage, arguments, tiled_mma_qk, warp_idx, tid);
    return;
  }
  if (WarpSpecialization::is_epilogue_warp(warp_idx)) {
    run_epilogue_warpgroup(storage, arguments, tiled_mma_pv, tid);
  }
}

}  // namespace minimax::msa_v1::attention::prefill::q8kv4::detail
