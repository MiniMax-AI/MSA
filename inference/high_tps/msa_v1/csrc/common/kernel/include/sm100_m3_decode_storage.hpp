#pragma once

#include <cstdint>

#include "common/kernel/include/sm100_m3_decode_traits.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"

namespace cutlass::m3_decode::collective {

#ifndef M3_DECODE_ENABLE_PLACEMENT
#define M3_DECODE_ENABLE_PLACEMENT 0
#endif

// Every role's dynamic SMEM ends at kM3DecodeRoleSmemBytes. The 16 bytes after it hold the
// block's launch state:
// - int 0: the block's index in the canonical placement (preferred clusters on blocks [0, 120),
//   fallback clusters on [120, 144)). When placement is enabled, run_decode_schedule resolves
//   it from the actual cluster. All roles use m3_decode_logical_block_idx(), which compiles
//   to blockIdx.x when placement is disabled.
// - int 1: the cluster ticket that cluster rank 0 claimed.
// - u64 at byte 8: the %globaltimer value (ns) after which every cross-CTA or cross-rank wait
//   traps, or zero for no deadline.
inline constexpr int kM3DecodeRoleSmemBytes = 199424;
inline constexpr int kM3DecodePlacementOffset = kM3DecodeRoleSmemBytes;
inline constexpr int kM3DecodeDeadlineOffset = kM3DecodePlacementOffset + 8;
inline constexpr int kM3DecodePlacementBytes = 16;

CUTLASS_DEVICE uint8_t *m3_decode_dynamic_smem() {
  extern __shared__ __align__(1024) uint8_t smem[];
  return smem;
}

template <bool EnablePlacement = false> struct M3DecodePlacement {
  static constexpr bool kEnabled = EnablePlacement;

  CUTLASS_DEVICE static int block_idx() {
    if constexpr (EnablePlacement) {
      return *reinterpret_cast<int const *>(m3_decode_dynamic_smem() + kM3DecodePlacementOffset);
    } else {
      return static_cast<int>(blockIdx.x);
    }
  }
};

// Each JIT translation unit binds one policy; all task helpers use the same CTA numbering.
using M3DecodePlacementPolicy = M3DecodePlacement<M3_DECODE_ENABLE_PLACEMENT != 0>;

CUTLASS_DEVICE int m3_decode_logical_block_idx() { return M3DecodePlacementPolicy::block_idx(); }

CUTLASS_DEVICE uint64_t m3_decode_global_timer() {
  uint64_t now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  return now;
}

// Cold paths: kept out of line so their printf does not inflate the spin loops' code, stack, or
// registers.
__device__ __noinline__ void m3_decode_wait_deadline_trap() {
  printf("M3_DECODE_WAIT_DEADLINE: block %d (logical %d) thread %d waited past the deadline\n",
         static_cast<int>(blockIdx.x), m3_decode_logical_block_idx(),
         static_cast<int>(threadIdx.x));
  asm volatile("trap;");
}

__device__ __noinline__ void m3_decode_cluster_shape_trap(int clusters, bool preferred,
                                                          uint32_t tickets) {
  printf("M3_DECODE_CLUSTER_SHAPE_TRAP: launch formed more than %d %s clusters "
         "(tickets before this claim 0x%08x)\n",
         clusters, preferred ? "8-CTA" : "2-CTA", tickets);
  asm volatile("trap;");
}

__device__ __noinline__ void m3_decode_canonical_placement_trap(int size, int rank) {
  printf("M3_DECODE_CANONICAL_PLACEMENT_REQUIRED: block %d cluster size %d rank %d; "
         "enable placement for non-canonical launches\n",
         static_cast<int>(blockIdx.x), size, rank);
  asm volatile("trap;");
}

// Variants launched with wait_timeout_ns > 0 are compiled with M3_DECODE_WAIT_DEADLINE=1; the
// default variants compile the check away to preserve the production wait path.
#ifndef M3_DECODE_WAIT_DEADLINE
#define M3_DECODE_WAIT_DEADLINE 0
#endif

// Called once per poll by every spin-wait on another CTA or rank. A wait still pending at the
// launch deadline means a peer will never arrive (a hung or failed peer, or a bug); trapping
// turns that hang into a launch failure the host sees, instead of a GPU spinning forever.
CUTLASS_DEVICE void m3_decode_check_deadline() {
#if M3_DECODE_WAIT_DEADLINE
  uint64_t const deadline =
      *reinterpret_cast<uint64_t const *>(m3_decode_dynamic_smem() + kM3DecodeDeadlineOffset);
  if (deadline != 0 && m3_decode_global_timer() > deadline) {
    m3_decode_wait_deadline_trap();
  }
#endif
}

template <class Traits> struct TmemLayout {
  enum : uint32_t {
    kS0 = 0,
    kS1 = 16,
    kStats0 = 32,
    kStats1 = 64,
    kO0 = 96,
    kPreparedKv = 112,
    kEnd = Traits::kNumTmemCols,
  };

  static constexpr uint32_t kColsPerPreparedKvStage =
      (Traits::kHeadDim * Traits::kTileKv) / (4 * 128);
  static constexpr uint32_t kPreparedKvCols = kEnd - kPreparedKv;
  static constexpr uint32_t kPreparedKvStages = kPreparedKvCols / kColsPerPreparedKvStage;

  static_assert(kEnd == 512, "SM100 TMEM has 512 columns per SM.");
  static_assert(kColsPerPreparedKvStage == 32,
                "Q8KV4 attention expects 32 TMEM columns per KV stage.");
  static_assert(kPreparedKvStages == Traits::kNumStagesTransform,
                "prepared KV TMEM staging must match the TransformKv pipeline.");
};

template <class Traits> struct BarrierLayout {
  static constexpr int kArv1Offset = 0;
  static constexpr int kArv32Offset = 41;
  static constexpr int kArv128Offset = 57;
  static constexpr int kArv512Offset = 78;

  static constexpr int kQFullArv1 = kArv1Offset + 0;
  static constexpr int kQEmptyArv1 = kArv1Offset + 2;
  static constexpr int kKvFullArv1 = kArv1Offset + 4;
  static constexpr int kKvEmptyArv1 = kArv1Offset + 13;
  static constexpr int kTransformedKvFullArv128 = kArv128Offset + 0;
  static constexpr int kTransformedKvEmptyArv1 = kArv1Offset + 22;
  static constexpr int kPageOffsetsFullArv32 = kArv32Offset + 0;
  static constexpr int kPageOffsetsEmptyArv32 = kArv32Offset + 6;
  static constexpr int kS0FullArv1 = kArv1Offset + 36;
  static constexpr int kS0EmptyArv128 = kArv128Offset + 12;
  static constexpr int kSoftmaxLocalFullArv128 = kArv128Offset + 14;
  static constexpr int kSoftmaxLocalEmptyArv128 = kArv128Offset + 16;
  static constexpr int kPFullArv128 = kArv128Offset + 18;
  static constexpr int kPEmptyArv1 = kArv1Offset + 38;
  static constexpr int kOFullArv1 = kArv1Offset + 40;
  static constexpr int kOEmptyArv128 = kArv128Offset + 20;

  static_assert(kOEmptyArv128 < Traits::kNumBarrierSlots,
                "barrier layout must fit the 96-slot pipeline storage.");
};

template <class Traits> struct SharedStorage {
  using Barriers = BarrierLayout<Traits>;
  static constexpr int kRoutedExpertFc2SmemBytes = 198656;
  static constexpr int kAttentionPipelineBarrierOffset = kRoutedExpertFc2SmemBytes;
  static constexpr int kAttentionPipelineBarrierBytes = Traits::kBarrierStorageBytes;
  static constexpr int kAttentionMiscBarrierBase = 78;
  static constexpr int kClusterReductionBarrierSlot = kAttentionMiscBarrierBase;
  static constexpr int kIndexerTopKBarrierSlot = kClusterReductionBarrierSlot + 1;
  static constexpr int kRouterEntryBarrierSlot = kIndexerTopKBarrierSlot + 1;
  static constexpr int kRouterResidualBarrierSlot = kRouterEntryBarrierSlot + 1;
  static constexpr int kRouterScaleBarrierSlot = kRouterResidualBarrierSlot + 1;
  static constexpr int kRouterPartialBarrierSlot = kRouterScaleBarrierSlot + 1;
  static constexpr int kRouterCandidateBarrierSlot = kRouterPartialBarrierSlot + 1;
  static constexpr int kDeviceOptInSmemBytes = 232448;
  static constexpr int kActiveSmemBytes = kM3DecodePlacementOffset + kM3DecodePlacementBytes;

  struct alignas(1024) SmemQ {
    uint8_t data[Traits::kNumStagesQ][Traits::kSmemQBytesPerStage];

    CUTLASS_DEVICE uint8_t *stage_ptr(int stage) { return data[stage]; }
  };

  struct alignas(1024) SmemKv {
    uint8_t data[Traits::kNumStagesRawKv][Traits::kRawKvStageBytes];

    CUTLASS_DEVICE uint8_t *stage_ptr(int stage) { return data[stage]; }

    CUTLASS_DEVICE uint8_t *raw_ptr(int stage) { return data[stage]; }

    CUTLASS_DEVICE uint8_t *scale_ptr(int stage) {
      return data[stage] + Traits::kRawKvDataBytesPerStage;
    }

    CUTLASS_DEVICE uint8_t *scale_scratch_ptr(int stage) {
      return data[stage] + Traits::kRawKvDataBytesPerStage + Traits::kRawKvScaleBytesPerStage;
    }
  };

  struct alignas(128) SmemPageOffsetsKv {
    int32_t data[Traits::kNumPageOffsetStages][Traits::kPageOffsetsPerStage];
  };

  struct alignas(1024) SmemP {
    uint8_t data[Traits::kSmemPBytes];

    CUTLASS_DEVICE uint8_t *stage_ptr(int stage) {
      return data + stage * Traits::kSmemQBytesPerStage;
    }
  };

  struct alignas(128) SmemO {
    uint8_t data[Traits::kSmemOBytes];
  };

  struct alignas(16) SmemWarpGroupReduction {
    float data[Traits::kWarpGroupReductionFloats];
  };

  struct alignas(16) SmemKvMetadata {
    int kv_length;
    int tile_count;
  };

  // Keep the recycled combine mailbox compatible with blockscaled SMEM descriptors.
  struct alignas(1024) SmemClusterReduction {
    static constexpr int kPartialOBytes = Traits::kClusterReductionOBytes;
    static constexpr int kStatsFloats = Traits::kClusterReductionStatsFloats;
    static constexpr int kPaddingBytes =
        Traits::kClusterReductionTransactionBytes - Traits::kClusterReductionPayloadBytes;

    uint8_t partial_o[kPartialOBytes];
    float stats[kStatsFloats];
    uint8_t padding[kPaddingBytes];
  };

  struct PipelineStorage {
    // Preserve the established data offsets while moving live Attention
    // barriers beyond the Routed K2 overlay.
    uint64_t barriers[Traits::kNumBarrierSlots];

    CUTLASS_DEVICE uint64_t *ptr(int slot) { return barriers + slot; }

    CUTLASS_DEVICE static void init_mbarrier(uint64_t *barrier, uint32_t arrive_count) {
      cutlass::arch::ClusterBarrier::init(barrier, arrive_count);
    }

    CUTLASS_DEVICE void init_pipeline_barriers(int warp_idx, int lane_idx) {
      if (warp_idx != 0) {
        return;
      }

      init_mbarrier(ptr(lane_idx), 1);
      uint32_t const mid_arrive_count = lane_idx < 9 ? 1u : (lane_idx < 25 ? 32u : 128u);
      init_mbarrier(ptr(32 + lane_idx), mid_arrive_count);
      if (lane_idx < kAttentionMiscBarrierBase - 64) {
        init_mbarrier(ptr(64 + lane_idx), 128);
      }
    }
  };

  struct alignas(16) ReusablePipelineStorage {
    static constexpr int kNumBarriers = 40;
    uint64_t barriers[kNumBarriers];

    CUTLASS_DEVICE uint64_t *ptr(int slot) { return barriers + slot; }
  };

  SmemQ smem_q;
  SmemKv smem_kv;
  SmemP smem_p;
  SmemPageOffsetsKv smem_page_offsets_kv;
  SmemO smem_o;
  SmemWarpGroupReduction smem_softmax_red0;
  SmemWarpGroupReduction smem_softmax_red1;
  SmemWarpGroupReduction smem_corr_red1;
  SmemKvMetadata smem_kv_metadata;
  uint32_t tmem_sw_state[Traits::kTmemSwStateBytes / sizeof(uint32_t)];
  PipelineStorage pipelines;
  ReusablePipelineStorage reusable_pipelines;
  alignas(8) uint64_t cluster_reduction_barrier;
  SmemClusterReduction smem_cluster_reduction;
  // The fused Indexer rank-0 CTA publishes TopK16 to this cluster-local
  // mailbox. It remains live until the page-offset warp has consumed it.
  alignas(16) int32_t indexer_topk[Traits::kSparseTopK > 0 ? Traits::kSparseTopK : 1];
  alignas(8) uint64_t indexer_topk_barrier;

  CUTLASS_DEVICE uint32_t *tmem_state_ptr() { return tmem_sw_state; }

  CUTLASS_DEVICE PipelineStorage &attention_pipelines() {
    auto *base = reinterpret_cast<uint8_t *>(this);
    return *reinterpret_cast<PipelineStorage *>(base + kAttentionPipelineBarrierOffset);
  }

  CUTLASS_DEVICE uint64_t *cluster_reduction_barrier_ptr() {
    return attention_pipelines().ptr(kClusterReductionBarrierSlot);
  }

  CUTLASS_DEVICE uint64_t *indexer_topk_barrier_ptr() {
    return attention_pipelines().ptr(kIndexerTopKBarrierSlot);
  }

  CUTLASS_DEVICE uint64_t *phase_barrier_ptr(int slot) {
    auto *base = reinterpret_cast<uint8_t *>(this);
    return reinterpret_cast<uint64_t *>(base + kPhaseBarrierOffset) + slot;
  }

  static constexpr int kMainloopSmemBytes =
      sizeof(SmemQ) + sizeof(SmemKv) + sizeof(SmemP) + sizeof(SmemPageOffsetsKv) + sizeof(SmemO) +
      3 * sizeof(SmemWarpGroupReduction) + sizeof(SmemKvMetadata) + Traits::kTmemSwStateBytes +
      Traits::kBarrierStorageBytes + sizeof(ReusablePipelineStorage);
  static constexpr int kExpectedSmemBytes = kMainloopSmemBytes + sizeof(SmemClusterReduction) +
                                            sizeof(uint64_t) + sizeof(indexer_topk) +
                                            sizeof(uint64_t);
  // Routed Expert FC2 reuses the complete attention/Shared storage as a contiguous
  // three-stage pipeline after Shared FC2 releases it.  Keep the K2 control
  // barriers and pair-local reduction tile beyond that pipeline so neither is
  // overwritten by stage 2 TMA traffic.
  // Phase-specific barriers live after the Routed FC2 overlay and before the
  // Shared Expert barrier tail, so no later phase overwrites a live object.
  static constexpr int kAttentionProjectionBarrierBase = 0;
  static constexpr int kAttentionProjectionBarrierCount = 23;
  static constexpr int kOutputProjectionBarrierBase =
      kAttentionProjectionBarrierBase + kAttentionProjectionBarrierCount;
  static constexpr int kOutputProjectionBarrierCount = 30;
  static constexpr int kRoutedExpertFc1BarrierBase =
      kOutputProjectionBarrierBase + kOutputProjectionBarrierCount;
  static constexpr int kRoutedExpertFc1BarrierCount = 16;
  static constexpr int kPhaseBarrierCount =
      kRoutedExpertFc1BarrierBase + kRoutedExpertFc1BarrierCount;
  static constexpr int kPhaseBarrierBytes = kPhaseBarrierCount * sizeof(uint64_t);
  static constexpr int kSharedExpertTailBarrierCount = 26;
  static constexpr int kSharedExpertTailBarrierBytes =
      kSharedExpertTailBarrierCount * sizeof(uint64_t);
  static constexpr int kSharedExpertTailBarrierOffset =
      kRoutedExpertFc2SmemBytes - kSharedExpertTailBarrierBytes;
  static constexpr int kPhaseBarrierOffset = kSharedExpertTailBarrierOffset - kPhaseBarrierBytes;

  static_assert(sizeof(SmemQ) == 4096, "SmemQ must match the Q8KV4 attention layout.");
  static_assert(sizeof(SmemKv) == Traits::kNumStagesRawKv * Traits::kRawKvStageBytes,
                "SmemKv must match the Q8KV4 attention layout.");
  static_assert(sizeof(SmemP) == 4096, "SmemP must match the Q8KV4 attention layout.");
  static_assert(sizeof(SmemPageOffsetsKv) == 768,
                "SmemPageOffsetsKv must match the Q8KV4 attention layout.");
  static_assert(sizeof(SmemO) == 4096, "SmemO must match the Q8KV4 attention layout.");
  static_assert(sizeof(SmemWarpGroupReduction) == 256,
                "warpgroup reduction buffer must match the Q8KV4 attention layout.");
  static_assert(sizeof(SmemKvMetadata) == 16,
                "KV metadata cache must remain one aligned SMEM record.");
  static_assert(sizeof(SmemClusterReduction) == Traits::kClusterReductionSmemBytes,
                "cluster reduction storage size should match the active path.");
  static_assert(sizeof(PipelineStorage) == Traits::kBarrierStorageBytes,
                "barrier storage must match the Q8KV4 attention layout.");
  static_assert(kActiveSmemBytes <= kDeviceOptInSmemBytes,
                "dynamic SMEM must not exceed the B300 opt-in device limit.");
  static_assert(kAttentionProjectionBarrierBase == 0);
  static_assert(kOutputProjectionBarrierBase == 23);
  static_assert(kRoutedExpertFc1BarrierBase == 53);
  static_assert(kPhaseBarrierCount == 69);
  static_assert(kPhaseBarrierOffset == 197896);
  static_assert(kSharedExpertTailBarrierOffset == 198448);
  static_assert(kPhaseBarrierOffset % alignof(uint64_t) == 0);
  static_assert(kSharedExpertTailBarrierOffset % alignof(uint64_t) == 0);
  static_assert(kExpectedSmemBytes <= kPhaseBarrierOffset,
                "phase barriers must not overlap the typed attention storage.");
  static_assert(kPhaseBarrierOffset + kPhaseBarrierBytes == kSharedExpertTailBarrierOffset);
  static_assert(kSharedExpertTailBarrierOffset + kSharedExpertTailBarrierBytes ==
                kRoutedExpertFc2SmemBytes);
  static_assert(kAttentionPipelineBarrierOffset == kRoutedExpertFc2SmemBytes);
  static_assert(kAttentionPipelineBarrierOffset + kAttentionPipelineBarrierBytes ==
                    kM3DecodeRoleSmemBytes,
                "the placement words must follow every role's storage.");
  static_assert(Barriers::kOEmptyArv128 < kAttentionMiscBarrierBase);
  static_assert(kRouterCandidateBarrierSlot < Traits::kNumBarrierSlots);
};

} // namespace cutlass::m3_decode::collective
