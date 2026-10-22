#pragma once
#include "common/kernel/include/sm100_m3_decode_scheduler.hpp"
#include "common/kernel/include/sm100_m3_decode_storage.hpp"
#include "cute/arch/cluster_sm90.hpp"
namespace cutlass::m3_decode::collective {

// The launch requests 2-CTA clusters with a preferred size of 8. The grid splits into 18
// eight-block regions, each formed as one 8-CTA cluster or four 2-CTA clusters, and the
// hardware chooses which regions get which shape: when other kernels hold SMs at launch, the
// 8-CTA clusters may land on blocks [120, 144) and 2-CTA clusters on [0, 120). Roles are
// assigned per cluster from a per-launch ticket when placement is enabled. The first
// 15 eight-CTA clusters cover preferred ordinals 0..14; MegaAttention assigns projection
// producers first. Two-CTA clusters take pair ordinals (canonical blocks [120 + 2k, 122 + 2k)).
// Disabled placement instead requires the canonical shape and uses blockIdx.x directly.
//
// The schedule needs 15 preferred clusters plus 24 CTAs of pair roles. An idle B300 forms
// 15 eight-CTA + 12 two-CTA clusters. SMs that free up while the launch is still forming
// clusters (another stream's kernel finishing) can change that mix:
// - more 8-CTA clusters: the attention-only kernel runs each surplus 8-CTA cluster as four
//   virtual pairs;
// - 14 8-CTA clusters + 16 pairs (a concurrent kernel holding an eight-SM cluster slot at
//   launch): the attention-only kernel runs the four extra pairs in the slots of preferred
//   ordinal 14, which has no role that uses the hardware cluster.
// The whole-layer kernel's pair roles and preferred ordinal 14 use the hardware cluster, so it
// traps on any non-canonical shape, as does the attention-only kernel on 13 or fewer 8-CTA
// clusters: the launch faults instead of deadlocking or running wrong roles.
template <class Traits>
CUTLASS_DEVICE void resolve_cluster_placement(Sm100M3DecodeKernelParams<Traits> const &params,
                                              int hardware_cluster_size, int hardware_rank,
                                              int thread_idx) {
  using Scheduler = Sm100M3DecodeScheduler<Traits>;
  constexpr bool kFlexibleShape = Traits::kKernelPath == Sm100M3DecodeKernelPath::kMegaAtten;
  bool const preferred = hardware_cluster_size == Scheduler::kPreferredClusterSize;
  uint8_t *launch_state = m3_decode_dynamic_smem();
  int *placement = reinterpret_cast<int *>(launch_state + kM3DecodePlacementOffset);
  if (thread_idx == 0) {
    if (hardware_rank == 0) {
      placement[1] = Scheduler::template claim_cluster_ticket<kFlexibleShape>(params, preferred);
    }
  }
  // barrier.cluster.arrive/wait default to release/acquire: rank 0's first block is visible to
  // every block of the cluster after the wait. The second barrier keeps rank 0's shared
  // memory alive until every peer has read it.
  cute::cluster_arrive();
  cute::cluster_wait();
  if (thread_idx == 0) {
    int first_block;
    uint32_t const remote = cute::set_block_rank(cute::cast_smem_ptr_to_uint(placement + 1), 0);
    asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(first_block) : "r"(remote) : "memory");
    placement[0] = first_block + hardware_rank;
  }
  cute::cluster_arrive();
  cute::cluster_wait();
}

template <class Traits, class InitialTask, class TailTask>
CUTLASS_DEVICE void run_decode_schedule(Sm100M3DecodeKernelParams<Traits> const &params,
                                        uint8_t *smem) {
  using Scheduler = Sm100M3DecodeScheduler<Traits>;
  using Storage = SharedStorage<Traits>;
  Storage &storage = *reinterpret_cast<Storage *>(smem);
  int const thread_idx = static_cast<int>(threadIdx.x);
#if M3_DECODE_WAIT_DEADLINE
  if (thread_idx == 0) {
    *reinterpret_cast<uint64_t *>(smem + kM3DecodeDeadlineOffset) =
        params.wait_timeout_ns == 0 ? 0 : m3_decode_global_timer() + params.wait_timeout_ns;
  }
#endif
  // MegaAttention publishes the deadline with its invocation barrier. Whole-layer execution
  // publishes it with the placement barrier below.
  Scheduler::begin_invocation(params, thread_idx);
  int const warp_idx = __shfl_sync(0xffffffffu, thread_idx / cutlass::NumThreadsPerWarp, 0,
                                   cutlass::NumThreadsPerWarp);
  int const lane_idx = thread_idx % cutlass::NumThreadsPerWarp;
  uint32_t cluster_size_raw;
  // The launch ABI fixes cluster y/z to one; avoid materializing unused
  // special-register values in the large fused kernel.
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(cluster_size_raw));
  int const hardware_cluster_size = static_cast<int>(cluster_size_raw);
  int const hardware_rank = static_cast<int>(cute::block_rank_in_cluster());
  if constexpr (M3DecodePlacementPolicy::kEnabled) {
    if (hardware_cluster_size != Scheduler::kPreferredClusterSize &&
        hardware_cluster_size != Scheduler::kFallbackClusterSize) {
      asm volatile("trap;");
    }
    resolve_cluster_placement(params, hardware_cluster_size, hardware_rank, thread_idx);
  } else {
    int const block = static_cast<int>(blockIdx.x);
    int const expected_size = block < Scheduler::kPreferredCtas ? Scheduler::kPreferredClusterSize
                                                                : Scheduler::kFallbackClusterSize;
    if (hardware_cluster_size != expected_size || hardware_rank != block % expected_size) {
      m3_decode_canonical_placement_trap(hardware_cluster_size, hardware_rank);
    }
  }
  // With placement enabled, roles see the logical cluster: a surplus 8-CTA cluster runs as four
  // virtual pairs (cluster size 2, rank = hardware rank % 2), and four extra pairs placed on
  // preferred ordinal 14 run as one virtual 8-CTA cluster (rank = logical block % 8).
  int const logical_block = m3_decode_logical_block_idx();
  bool const pair_role = logical_block >= Scheduler::kPreferredCtas;
  int const cluster_size =
      pair_role ? Scheduler::kFallbackClusterSize : Scheduler::kPreferredClusterSize;
  int const cluster_rank = pair_role ? hardware_rank % Scheduler::kFallbackClusterSize
                                     : logical_block % Scheduler::kPreferredClusterSize;
  int const cluster_start = logical_block - cluster_rank;
  int const cluster_ordinal =
      pair_role ? (cluster_start - Scheduler::kPreferredCtas) / Scheduler::kFallbackClusterSize
                : cluster_start / Scheduler::kPreferredClusterSize;

  uint32_t const shared_epoch =
      InitialTask::run(params, storage, smem, thread_idx, warp_idx, lane_idx, cluster_size,
                       cluster_rank, cluster_ordinal);
  TailTask::run(params, storage, smem, thread_idx, warp_idx, lane_idx, cluster_size, cluster_rank,
                cluster_ordinal, shared_epoch);
  if constexpr (M3DecodePlacementPolicy::kEnabled) {
    if (thread_idx == 0) {
      Scheduler::wait_placement_final(params);
    }
  }
}

} // namespace cutlass::m3_decode::collective
