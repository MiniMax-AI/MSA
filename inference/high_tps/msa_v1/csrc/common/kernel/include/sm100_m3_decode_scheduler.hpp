#pragma once

#include <cstddef>
#include <cstdint>

#include "attn/kernel/include/sm100_m3_decode_problem.hpp"
#include "common/kernel/include/sm100_m3_decode_shared_control.hpp"
#include "common/kernel/include/sm100_m3_decode_storage.hpp"
#include "common/kernel/include/tp4_nvls_layout.hpp"
#include "cutlass/cutlass.h"

namespace cutlass::m3_decode::collective {

template <class Traits> struct Sm100M3DecodeScheduler {
  using Params = Sm100M3DecodeKernelParams<Traits>;
  using Storage = SharedStorage<Traits>;
  using FusionControl = Sm100M3DecodeSharedControl<Traits>;

  static constexpr int kGridCtas = 144;
  static constexpr int kPreferredClusterSize = 8;
  static constexpr int kFallbackClusterSize = 2;
  static constexpr int kPreferredCtas = 120;
  static constexpr int kFallbackCtas = kGridCtas - kPreferredCtas;
  static constexpr int kPreferredClusters = kPreferredCtas / kPreferredClusterSize;

  static constexpr int kAttentionTasks = 8;
  static constexpr int kOutputProjectionTasks = 48;
  static constexpr int kOutputProjectionPreferredClusterStart = kAttentionTasks;
  static constexpr int kOutputProjectionPreferredClusters =
      kOutputProjectionTasks / kPreferredClusterSize;
  static constexpr int kOutputProjectionPreferredClusterEnd =
      kOutputProjectionPreferredClusterStart + kOutputProjectionPreferredClusters;

  static constexpr int kSharedCtas = 12;
  static constexpr int kSharedFallbackClusterStart = 0;
  static constexpr int kSharedFallbackClusters = kSharedCtas / kFallbackClusterSize;
  static constexpr int kRoutedFallbackClusterStart =
      kSharedFallbackClusterStart + kSharedFallbackClusters;
  static constexpr int kRoutedFallbackClusters =
      kFallbackCtas / kFallbackClusterSize - kRoutedFallbackClusterStart;

  static constexpr int kRoutedWorkPerExpert = 4;
  static constexpr int kActiveExpertBound = 32;
  static constexpr int kE28RoutedExpertFc1RemapCtas = 24;
  static constexpr int kFallbackStateExperts = 6;
  static constexpr int kRoutedExpertFc2Tasks = 48;
  // Zero is reserved for not-ready; active expert count is encoded as count + 1.
  static constexpr uint64_t kTaggedPayloadMask = 0xffu;

  struct alignas(128) InvocationState {
    uint64_t routed_ready;
    uint64_t attention_projection_ready;
    uint64_t attention_projection_qi_ready;
    uint64_t attention_projection_ki_ready;
    uint32_t router_expert_arrivals;
    uint32_t router_expert_arrivals_padding[kAttentionTasks - 1];
    // Cluster placement tickets claimed by this launch: preferred clusters in bits [0, 16),
    // fallback clusters in bits [16, 32). The last cluster to claim resets the word to zero.
    uint32_t cluster_tickets;
    uint32_t cluster_tickets_padding[3];
    // Generation of the last launch whose clusters have all claimed tickets.
    uint64_t placement_final;
    // Diagnostics, never reset: clusters that ran in another shape's slots (surplus 8-CTA
    // clusters as virtual pairs, extra pairs in preferred ordinal 14) so far.
    uint32_t reshaped_clusters;
    uint32_t reshaped_padding;
    uint64_t padding[4];
  };

  struct alignas(128) Counter {
    uint32_t value;
    uint32_t padding[31];
  };

  struct alignas(128) SchedulerStorage {
    InvocationState invocation;
    Counter router_arrivals;
    alignas(128) uint32_t routed_expert_fc1_ready_epoch[kActiveExpertBound];
    alignas(128) uint32_t routed_expert_fc1_fallback_arrivals[kFallbackStateExperts];
    uint32_t routed_expert_fc1_fallback_arrivals_padding[26];
    alignas(128) uint32_t shared_fc2_completion[kRoutedExpertFc2Tasks];
    uint32_t shared_fc2_completion_padding[16];
    alignas(128) uint32_t shared_fc2_consumed[kRoutedExpertFc2Tasks];
    uint32_t shared_fc2_consumed_padding[16];
  };

  static constexpr size_t kSchedulerOffset =
      FusionControl::kControlOffset + sizeof(typename FusionControl::Storage);
  static constexpr int kRouterExperts = kActiveExpertBound * kRoutedWorkPerExpert;
  static constexpr int kRoutedWorkStateIndex = 3;

  static_assert(kPreferredCtas % kPreferredClusterSize == 0);
  static_assert(kFallbackCtas % kFallbackClusterSize == 0);
  static_assert(kOutputProjectionTasks % kPreferredClusterSize == 0);
  static_assert(kOutputProjectionPreferredClusterEnd == kPreferredClusters - 1);
  static_assert((kAttentionTasks & (kAttentionTasks - 1)) == 0);
  static_assert(kSharedCtas % kFallbackClusterSize == 0);
  static_assert(kE28RoutedExpertFc1RemapCtas % kPreferredClusterSize == 0);
  static_assert(kE28RoutedExpertFc1RemapCtas <= kPreferredClusterSize + kFallbackCtas);
  static_assert(kRoutedFallbackClusterStart * kFallbackClusterSize == kSharedCtas);
  static_assert(Traits::kTmemSwStateBytes >= 4 * sizeof(uint32_t));
  static_assert(sizeof(InvocationState) == 128);
  static_assert(offsetof(InvocationState, reshaped_clusters) == 88);
  static_assert(sizeof(Counter) == 128);
  static_assert(sizeof(SchedulerStorage) == 8 * 128);
  static_assert(kSchedulerOffset + sizeof(SchedulerStorage) <=
                FusionControl::kEntryOffset + 688 * sizeof(uint32_t));

  CUTE_DEVICE static SchedulerStorage *scheduler_storage(Params const &params) {
    auto *base = static_cast<uint8_t *>(params.tp_unicast_ptr);
    return reinterpret_cast<SchedulerStorage *>(base + kSchedulerOffset);
  }

  CUTE_DEVICE static uint64_t current_grid_id() {
    uint64_t grid_id;
    asm volatile("mov.u64 %0, %%gridid;" : "=l"(grid_id));
    return grid_id;
  }

  CUTE_DEVICE static uint64_t current_generation_prefix() { return current_grid_id() << 8; }

  CUTE_DEVICE static auto *runtime_indexer_control(Params const &params) {
    namespace layout = mega_atten::m3_decode::tp4_nvls;
    return reinterpret_cast<layout::RuntimeIndexerControl *>(
        static_cast<uint8_t *>(params.tp_unicast_ptr) + layout::kRuntimeIndexerControlOffset);
  }

  CUTE_DEVICE static uint64_t &invocation_epoch_cache() {
    __shared__ uint64_t epoch;
    return epoch;
  }

  CUTE_DEVICE static void begin_invocation(Params const &params, int thread_idx) {
    if constexpr (Traits::kKernelPath == Sm100M3DecodeKernelPath::kMegaAtten) {
      // This CTA has not published its score completion yet. At most the
      // other kGridCtas - 1 CTAs can have published in this invocation, so
      // integer division recovers the same epoch without a global write.
      if (thread_idx == 0) {
        uint64_t const completed = load_acquire(&runtime_indexer_control(params)->arrivals);
        invocation_epoch_cache() = completed / kGridCtas + 1;
      }
      __syncthreads();
    }
  }

  CUTE_DEVICE static uint64_t invocation_epoch(Params const &params) {
    return invocation_epoch_cache();
  }

  CUTE_DEVICE static uint64_t current_generation_prefix(Params const &params) {
    if constexpr (Traits::kKernelPath == Sm100M3DecodeKernelPath::kMegaAtten) {
      return invocation_epoch(params) << 8;
    } else {
      return current_generation_prefix();
    }
  }

  CUTE_DEVICE static uint64_t load_relaxed(uint64_t const *address) {
    uint64_t value;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.u64 %0, [%1];"
                 : "=l"(value)
                 : "l"(address)
                 : "memory");
    return value;
  }

  CUTE_DEVICE static uint64_t load_acquire(uint64_t const *address) {
    uint64_t value;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
    return value;
  }

  CUTE_DEVICE static void store_release(uint64_t *address, uint64_t value) {
    asm volatile("st.release.gpu.global.u64 [%0], %1;" : : "l"(address), "l"(value) : "memory");
  }

  CUTE_DEVICE static uint32_t fetch_add_acq_rel(uint32_t *address, uint32_t value) {
    uint32_t previous;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                 : "=r"(previous)
                 : "l"(address), "r"(value)
                 : "memory");
    return previous;
  }

  CUTE_DEVICE static void reduce_add_release(uint32_t *address, uint32_t value) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;"
                 :
                 : "l"(address), "r"(value)
                 : "memory");
  }

  CUTE_DEVICE static uint32_t load_acquire(uint32_t const *address) {
    uint32_t value;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
    return value;
  }

  CUTE_DEVICE static uint32_t load_relaxed(uint32_t const *address) {
    uint32_t value;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.u32 %0, [%1];"
                 : "=r"(value)
                 : "l"(address)
                 : "memory");
    return value;
  }

  CUTE_DEVICE static void store_release(uint32_t *address, uint32_t value) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(address), "r"(value) : "memory");
  }

  CUTE_DEVICE static uint64_t compare_exchange_acq_rel(uint64_t *address, uint64_t expected,
                                                       uint64_t desired) {
    uint64_t previous;
    asm volatile("atom.acq_rel.gpu.global.cas.b64 %0, [%1], %2, %3;"
                 : "=l"(previous)
                 : "l"(address), "l"(expected), "l"(desired)
                 : "memory");
    return previous;
  }

  CUTE_DEVICE static void wait_equal(uint32_t const *address, uint32_t expected) {
    uint32_t observed;
    do {
      observed = load_acquire(address);
      if (observed != expected) {
        m3_decode_check_deadline();
        __nanosleep(64);
      }
    } while (observed != expected);
  }

  CUTE_DEVICE static void wait_equal_relaxed_acquire(uint32_t const *address, uint32_t expected) {
    uint32_t observed;
    do {
      do {
        observed = load_relaxed(address);
        if (observed != expected) {
          m3_decode_check_deadline();
          __nanosleep(64);
        }
      } while (observed != expected);
      observed = load_acquire(address);
    } while (observed != expected);
  }

  CUTE_DEVICE static void wait_at_least(uint32_t const *address, uint32_t expected) {
    uint32_t observed;
    do {
      observed = load_acquire(address);
      if (observed < expected) {
        m3_decode_check_deadline();
        __nanosleep(64);
      }
    } while (observed < expected);
  }

  CUTE_DEVICE static bool tagged_ready(uint64_t value, uint64_t generation_prefix) {
    return (value & ~kTaggedPayloadMask) == generation_prefix && (value & kTaggedPayloadMask) != 0;
  }

  CUTE_DEVICE static uint64_t wait_tagged(uint64_t const *address, uint64_t generation_prefix) {
    uint64_t value;
    do {
      do {
        value = load_relaxed(address);
        if (!tagged_ready(value, generation_prefix)) {
          m3_decode_check_deadline();
          __nanosleep(64);
        }
      } while (!tagged_ready(value, generation_prefix));
      value = load_acquire(address);
    } while (!tagged_ready(value, generation_prefix));
    return value;
  }

  CUTE_DEVICE static uint64_t *routed_ready_address(Params const &params) {
    return &scheduler_storage(params)->invocation.routed_ready;
  }

  CUTE_DEVICE static uint64_t *attention_projection_ready_address(Params const &params) {
    return &scheduler_storage(params)->invocation.attention_projection_ready;
  }

  CUTE_DEVICE static uint64_t *attention_projection_qi_ready_address(Params const &params) {
    return &scheduler_storage(params)->invocation.attention_projection_qi_ready;
  }

  CUTE_DEVICE static uint64_t *attention_projection_ki_ready_address(Params const &params) {
    return &scheduler_storage(params)->invocation.attention_projection_ki_ready;
  }

  static constexpr uint32_t kFallbackTicketShift = 16;
  static constexpr int kFallbackClusters = kFallbackCtas / kFallbackClusterSize;
  static constexpr int kPairsPerPreferredCluster = kPreferredClusterSize / kFallbackClusterSize;

  // Claims this launch's cluster ticket with one atomic: bits [0, 16) count 8-CTA clusters and
  // bits [16, 32) 2-CTA clusters claimed so far. Launches on one workspace are serialized, and
  // the claim that completes all 144 CTAs resets the word, so the next launch starts from zero
  // without a host reset.
  //
  // The first 15 8-CTA clusters cover preferred ordinals 0..14, with projection producers first
  // in the attention-only kernel. The first 12 2-CTA clusters take pair ordinals 0..11. Only the
  // attention-only kernel (FlexibleShape) runs the other shapes the hardware forms, because its
  // pair roles and last preferred cluster (ordinal 14, with no attention, projection or
  // output-projection role) never use the hardware cluster:
  // - a surplus 8-CTA cluster takes four pair ordinals from the top (8..11, then 4..7, ...) and
  //   runs them as four virtual pairs;
  // - pairs 13..16 (14 eight-CTA clusters) take the four pair slots of preferred ordinal 14.
  // Total CTAs are fixed, so the two cases never mix and ordinals never collide. Any other
  // shape cannot run the schedule: the launch reports it and traps. Returns the cluster's
  // first logical block.
  template <bool FlexibleShape>
  CUTE_DEVICE static int claim_cluster_ticket(Params const &params, bool preferred) {
    uint32_t *tickets = &scheduler_storage(params)->invocation.cluster_tickets;
    uint32_t const increment = preferred ? 1u : 1u << kFallbackTicketShift;
    uint32_t const previous = fetch_add_acq_rel(tickets, increment);
    int const preferred_before = static_cast<int>(previous & 0xffffu);
    int const pairs_before = static_cast<int>(previous >> kFallbackTicketShift);
    int first_block;
    bool valid;
    if (!preferred && pairs_before < kFallbackClusters) {
      first_block = kPreferredCtas + pairs_before * kFallbackClusterSize;
      valid = true;
    } else if (!preferred) {
      int const extra = pairs_before - kFallbackClusters;
      first_block = kPreferredCtas - kPreferredClusterSize + extra * kFallbackClusterSize;
      valid = FlexibleShape && extra < kPairsPerPreferredCluster;
    } else if (preferred_before < kPreferredClusters) {
      int ordinal = preferred_before;
      if constexpr (FlexibleShape) {
        // Start projection producers before consumers that wait for QI/KI.
        // Keep the last preferred slot available for the four extra pairs.
        constexpr int kProjectionClusters = attention_projection::kClusterCount;
        constexpr int kProjectionEnd = kPreferredClusters - 1;
        ordinal = preferred_before < kProjectionClusters
                      ? kProjectionEnd - 1 - preferred_before
                      : (preferred_before < kProjectionEnd ? preferred_before - kProjectionClusters
                                                           : preferred_before);
      }
      first_block = ordinal * kPreferredClusterSize;
      valid = true;
    } else {
      int const surplus = preferred_before - kPreferredClusters;
      int const first_pair = kFallbackClusters - (surplus + 1) * kPairsPerPreferredCluster;
      first_block = kPreferredCtas + first_pair * kFallbackClusterSize;
      valid = FlexibleShape && first_pair >= 0;
    }
    bool const reshaped =
        preferred ? preferred_before >= kPreferredClusters : pairs_before >= kFallbackClusters;
    if (valid && reshaped) {
      reduce_add_release(&scheduler_storage(params)->invocation.reshaped_clusters, 1u);
    }
    if (!valid) {
      int const clusters = preferred ? kPreferredClusters : kFallbackClusters;
      m3_decode_cluster_shape_trap(clusters, preferred, previous);
    }
    uint32_t const now = previous + increment;
    if ((now & 0xffffu) * kPreferredClusterSize +
            (now >> kFallbackTicketShift) * kFallbackClusterSize ==
        static_cast<uint32_t>(kGridCtas)) {
      store_release(tickets, 0u);
      store_release(&scheduler_storage(params)->invocation.placement_final,
                    current_generation_prefix(params));
    }
    return first_block;
  }

  // The hardware forms clusters while the launch's blocks start; an SM freed during that
  // window (here: a block that already finished its role) can take a cluster of either shape,
  // so a launch whose short roles exit early forms 16+ eight-CTA clusters. Each block's thread
  // 0 therefore waits, before the block exits, until the last cluster of this launch has
  // claimed its ticket: no SM is released before the launch's cluster shape is final.
  CUTE_DEVICE static void wait_placement_final(Params const &params) {
    uint64_t const generation = current_generation_prefix(params);
    uint64_t const *final_word = &scheduler_storage(params)->invocation.placement_final;
    while (load_acquire(final_word) != generation) {
      m3_decode_check_deadline();
    }
  }

  CUTE_DEVICE static bool arrive_router_expert(Params const &params) {
    static_assert((kRouterExperts & (kRouterExperts - 1)) == 0);
    uint32_t *arrivals = &scheduler_storage(params)->invocation.router_expert_arrivals;
    uint32_t const previous = fetch_add_acq_rel(arrivals, 1);
    return ((previous + 1) & (kRouterExperts - 1)) == 0;
  }

  CUTE_DEVICE static void publish_attention_projection_ready(Params const &params, int thread_idx,
                                                             bool is_leader) {
    if (is_leader && thread_idx == 0) {
      store_release(attention_projection_ready_address(params),
                    current_generation_prefix(params) | 1u);
    }
  }

  CUTE_DEVICE static void wait_attention_projection_ready(Params const &params, int thread_idx) {
    if (thread_idx == 0) {
      wait_tagged(attention_projection_ready_address(params), current_generation_prefix(params));
    }
    __syncthreads();
  }

  CUTE_DEVICE static void wait_attention_projection_indexer_ready(Params const &params,
                                                                  int thread_idx) {
    if (thread_idx == 0) {
      uint64_t const generation = current_generation_prefix(params);
      wait_tagged(attention_projection_qi_ready_address(params), generation);
    }
    __syncthreads();
  }

  CUTE_DEVICE static int wait_routed_work(Params const &params, Storage &storage, int thread_idx) {
    if (thread_idx == 0) {
      uint64_t const value =
          wait_tagged(routed_ready_address(params), current_generation_prefix(params));
      uint32_t const active_experts = static_cast<uint32_t>(value & kTaggedPayloadMask) - 1;
      storage.tmem_state_ptr()[kRoutedWorkStateIndex] = active_experts * kRoutedWorkPerExpert;
    }
    __syncthreads();
    return static_cast<int>(storage.tmem_state_ptr()[kRoutedWorkStateIndex]);
  }

  CUTE_DEVICE static bool arrive_router_token(Params const &params, int thread_idx,
                                              int cluster_rank) {
    if (cluster_rank != 0 || thread_idx >= cutlass::NumThreadsPerWarp) {
      return false;
    }
    uint32_t previous = 0;
    if (thread_idx == 0) {
      SchedulerStorage *state = scheduler_storage(params);
      // The zero-initialized counter advances by exactly one full group per
      // serialized workspace invocation, so its modulo phase needs no reset.
      previous = fetch_add_acq_rel(&state->router_arrivals.value, 1);
    }
    previous = __shfl_sync(0xffffffffu, previous, 0);
    __syncwarp();
    return ((previous + 1) & (kAttentionTasks - 1)) == 0;
  }

  CUTE_DEVICE static void publish_shared_fc2(Params const &params, int output_tile,
                                             int thread_idx) {
    if (thread_idx == 0) {
      reduce_add_release(scheduler_storage(params)->shared_fc2_completion + output_tile, 1);
    }
  }

  CUTE_DEVICE static void wait_shared_fc2(Params const &params, int output_tile) {
    SchedulerStorage *state = scheduler_storage(params);
    uint32_t *consumed = state->shared_fc2_consumed + output_tile;
    uint32_t const target = *consumed + 1;
    wait_equal(state->shared_fc2_completion + output_tile, target);
    store_release(consumed, target);
  }

  CUTE_DEVICE static void publish_routed_expert_fc1_cluster(Params const &params, int cluster_size,
                                                            int cluster_start, int routed_work,
                                                            uint32_t epoch, int thread_idx,
                                                            int cluster_rank) {
    if (cluster_rank != 0 || thread_idx != 0 || cluster_start >= routed_work) {
      return;
    }
    SchedulerStorage *state = scheduler_storage(params);
    if (cluster_size == kPreferredClusterSize) {
      int const first_expert = cluster_start / kRoutedWorkPerExpert;
      int const remaining_tasks = routed_work - cluster_start;
      int const valid_tasks = remaining_tasks < cluster_size ? remaining_tasks : cluster_size;
      int const first_expert_tasks =
          valid_tasks < kRoutedWorkPerExpert ? valid_tasks : kRoutedWorkPerExpert;
      if (first_expert_tasks == kRoutedWorkPerExpert) {
        store_release(state->routed_expert_fc1_ready_epoch + first_expert, epoch);
      }
      if (valid_tasks > kRoutedWorkPerExpert) {
        int const second_expert_tasks = valid_tasks - kRoutedWorkPerExpert;
        if (second_expert_tasks == kRoutedWorkPerExpert) {
          store_release(state->routed_expert_fc1_ready_epoch + first_expert + 1, epoch);
        }
      }
    } else {
      int const expert = cluster_start / kRoutedWorkPerExpert;
      // E28 remaps experts 2--5 to cluster2; E32 uses cluster2 for
      // experts 30--31. Keep the six monotonic counters compact.
      int const fallback_slot = expert >= 30 ? expert - 26 : expert - 2;
      uint32_t const previous =
          fetch_add_acq_rel(state->routed_expert_fc1_fallback_arrivals + fallback_slot, 1);
      if ((previous & 1u) != 0) {
        store_release(state->routed_expert_fc1_ready_epoch + expert, epoch);
      }
    }
  }

  CUTE_DEVICE static void wait_routed_expert_fc1_expert(Params const &params, int expert_slot,
                                                        uint32_t epoch) {
    if (cute::elect_one_sync()) {
      wait_equal_relaxed_acquire(
          scheduler_storage(params)->routed_expert_fc1_ready_epoch + expert_slot, epoch);
    }
    __syncwarp();
  }

  CUTE_DEVICE static uint32_t routed_expert_fc1_ready_mask(Params const &params, int active_experts,
                                                           uint32_t epoch) {
    int const lane = static_cast<int>(threadIdx.x) & 31;
    uint32_t observed = 0;
    if (lane < active_experts) {
      observed = load_acquire(scheduler_storage(params)->routed_expert_fc1_ready_epoch + lane);
    }
    uint32_t const ready = __ballot_sync(0xffffffffu, lane < active_experts && observed == epoch);
    __syncwarp();
    return ready;
  }
};

} // namespace cutlass::m3_decode::collective
