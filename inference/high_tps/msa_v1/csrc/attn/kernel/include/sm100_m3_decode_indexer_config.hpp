#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "attn/kernel/include/sm100_m3_decode_problem.hpp"
#include "attn/kernel/include/sm100_m3_decode_topk_runtime.hpp"
#include "common/kernel/include/sm100_m3_decode_pipeline.hpp"
#include "common/kernel/include/sm100_m3_decode_scheduler.hpp"
#include "common/kernel/include/sm100_m3_decode_storage.hpp"
#include "common/kernel/include/tp4_nvls_layout.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/arch/util.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"

namespace cutlass::m3_decode::collective {

template <class Traits> struct Sm100M3DecodeIndexerConfig {
  using Params = Sm100M3DecodeKernelParams<Traits>;
  using Storage = SharedStorage<Traits>;
  using Scheduler = Sm100M3DecodeScheduler<Traits>;

  static constexpr int kPageTokens = 128;
  static constexpr int kHeadDim = 128;
  static constexpr int kScaleGroups = 8;
  static constexpr int kPackedKBytes = kPageTokens * kHeadDim / 2;
  static constexpr int kScaleBytes = kPageTokens * kScaleGroups;
  static constexpr int kPageBytes = kPackedKBytes + kScaleBytes;
  static constexpr int kRows = 8;
  static constexpr int kTopK = 16;
  static constexpr int kRankedTopK = kTopK - 1;
  static constexpr int kGridCtas = 144;
  static constexpr int kPreferredScoreCtas = 120;
  static constexpr int kFallbackScoreCtaStart = 120;
  static constexpr int kMaxScoreProducerCtas = 144;
  static constexpr int kTargetPagesPerProducer = 1;
  static constexpr int kMaxPages = mega_atten::m3_decode::tp4_nvls::kIndexerMaxPages;
  static constexpr int kMaxPagesPerCta = 16;
  static constexpr bool kUnifiedTopK = Traits::kKernelPath == Sm100M3DecodeKernelPath::kMegaAtten;
  static constexpr bool kNeedsPageRefill = kUnifiedTopK || Traits::kIndexerTopKItemsPerThread >= 8;
  static constexpr bool kUseCtaCandidateHandoff =
      !kUnifiedTopK && Traits::kIndexerTopKItemsPerThread >= 32;
  static_assert(!Traits::kWaveBalancedIndexerPages || Traits::kIndexerTopKItemsPerThread >= 16);
  static constexpr int kScoreWarps = 8;
  static constexpr int kScoreGroups = 2;
  static constexpr int kTokensPerScoreWarp = kPageTokens / kScoreWarps;
  static constexpr int kTokenTilesPerScoreWarp = kTokensPerScoreWarp / 16;
  static constexpr int kPartialPageCapacity = 64;
  static constexpr int kTopKWarps = 8;
  static constexpr int kMaximumTopKWarps = 16;
  static constexpr int kTopKThreads = kTopKWarps * cutlass::NumThreadsPerWarp;
  static constexpr int kTopKLocalCandidates = kTopK;
  static constexpr int kTopKCandidatesPerLane =
      (kTopKWarps * kTopKLocalCandidates + cutlass::NumThreadsPerWarp - 1) /
      cutlass::NumThreadsPerWarp;
  static constexpr int kFastTopKMaxPages = 1024;
  static constexpr int kFastTopKItemsPerThread = kFastTopKMaxPages / kTopKThreads;
  static constexpr int kFastTopKPageBits = 10;
  static constexpr uint32_t kFastTopKPageMask = (1u << kFastTopKPageBits) - 1;
  static constexpr uint32_t kFastTopKScoreMaximum = (1u << (32 - kFastTopKPageBits)) - 1;
  static constexpr int kTopKBarrierId = 2;
  static constexpr int kScorePublishBarrierId = 8;
  static constexpr int kArrivalsPerCta = 1;
  static constexpr int kArrivalsPerInvocation = kGridCtas * kArrivalsPerCta;
  static constexpr int kTopKTransactionBytes = kTopK * sizeof(int32_t);
  static constexpr int kCtaCandidateStride = 16;
  static constexpr int kCtaCandidateCount = kGridCtas * kCtaCandidateStride;
  static constexpr int kCtaCandidateItemsPerThread =
      (kCtaCandidateCount + kTopKThreads - 1) / kTopKThreads;

  struct alignas(128) InvocationTag {
    uint64_t value;
    uint64_t padding[15];
  };

  struct alignas(128) Counter {
    uint32_t value;
    uint32_t padding[31];
  };

  struct alignas(128) GlobalControl {
    InvocationTag invocation;
    Counter arrivals;
  };

  struct QuantizedTailStorage {
    uint32_t candidates[kTopKWarps][kTopKLocalCandidates];
    float extrema[2][kTopKWarps];
    uint32_t exact_fallback;
    uint32_t boundary_score;
    int32_t strict_count;
  };

  struct ExactTailStorage {
    float candidate_scores[kTopKWarps][kTopKLocalCandidates];
    int candidate_pages[kTopKWarps][kTopKLocalCandidates];
    uint64_t candidates[kMaximumTopKWarps][kTopK - 1];
  };

  struct TailStorage {
    QuantizedTailStorage quantized;
    ExactTailStorage chunks;
  };

  struct alignas(1024) IndexerStorage {
    uint8_t pages[kMaxPagesPerCta][kPageBytes];
    union {
      alignas(16) float page_partials[kPartialPageCapacity][kRows][kScoreWarps];
      TailStorage tail;
      runtime_topk::Scratch runtime_topk;
    };
    uint8_t q_tile[kRows * kHeadDim];
    uint64_t page_barriers[kMaxPagesPerCta];
    uint64_t page_consumed_barriers[kMaxPagesPerCta];
    uint64_t allocation_barrier;
    uint32_t tmem_base;
  };

  static_assert(Traits::kNumThreads == 512);
  static_assert(Traits::kPageSize == kPageTokens);
  static_assert(Traits::kHeadDim == kHeadDim);
  static_assert(Traits::kSparseTopK == kTopK);
  static_assert(kPageTokens % kScoreWarps == 0);
  static_assert(kTokensPerScoreWarp % 16 == 0);
  static_assert(kTokenTilesPerScoreWarp == 1);
  static_assert((kMaxPages + kGridCtas - 1) / kGridCtas <= kPartialPageCapacity);
  static_assert(sizeof(IndexerStorage) <= offsetof(Storage, indexer_topk));
  static_assert(sizeof(GlobalControl) == mega_atten::m3_decode::tp4_nvls::kIndexerControlBytes);
  static_assert(kCtaCandidateItemsPerThread == 9);

  CUTE_DEVICE static uint32_t smem_address(void const *pointer) {
    return cute::cast_smem_ptr_to_uint(pointer);
  }

  CUTE_DEVICE static void init_barrier(uint64_t *barrier, uint32_t count) {
    cutlass::arch::ClusterBarrier::init(barrier, count);
  }

  CUTE_DEVICE static void expect_transactions(uint64_t *barrier, uint32_t bytes) {
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(barrier, bytes);
  }

  CUTE_DEVICE static void arrive(uint64_t *barrier) {
    cutlass::arch::ClusterBarrier::arrive(barrier);
  }

  CUTE_DEVICE static void wait(uint64_t *barrier, uint32_t phase) {
    Sm100M3DecodeBarrier::wait(barrier, phase);
  }

  CUTE_DEVICE static uint32_t qmul4(uint16_t packed, uint32_t broadcast_scale) {
    uint32_t output;
    asm volatile("{ .reg .b16 value;\n"
                 "  mov.b16 value, %1;\n"
                 "  mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, value, %2; }\n"
                 : "=r"(output)
                 : "h"(packed), "r"(broadcast_scale));
    return output;
  }

  CUTE_DEVICE static uint32_t warp_max(uint32_t value) {
    uint32_t output;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(output) : "r"(value));
    return output;
  }

  CUTE_DEVICE static uint64_t warp_max(uint64_t value) {
    uint32_t const high = static_cast<uint32_t>(value >> 32);
    uint32_t const low = static_cast<uint32_t>(value);
    uint32_t const max_high = warp_max(high);
    uint32_t const max_low = warp_max(high == max_high ? low : 0u);
    return (static_cast<uint64_t>(max_high) << 32) | max_low;
  }

  CUTE_DEVICE static uint64_t warp_sort_descending(uint64_t value, int lane_idx) {
    CUTLASS_PRAGMA_UNROLL
    for (int size = 2; size <= cutlass::NumThreadsPerWarp; size <<= 1) {
      CUTLASS_PRAGMA_UNROLL
      for (int stride = size >> 1; stride > 0; stride >>= 1) {
        uint64_t const peer = __shfl_xor_sync(0xffffffffu, value, stride);
        bool const descending = (lane_idx & size) == 0;
        bool const lower_lane = (lane_idx & stride) == 0;
        bool const take_maximum = descending == lower_lane;
        uint32_t const score = static_cast<uint32_t>(value >> 32);
        uint32_t const peer_score = static_cast<uint32_t>(peer >> 32);
        value = take_maximum ? (score >= peer_score ? value : peer)
                             : (score <= peer_score ? value : peer);
      }
    }
    return value;
  }

  template <int Count> CUTE_DEVICE static void sort_descending(uint32_t (&values)[Count]) {
    CUTLASS_PRAGMA_UNROLL
    for (int size = 2; size <= Count; size <<= 1) {
      CUTLASS_PRAGMA_UNROLL
      for (int stride = size >> 1; stride > 0; stride >>= 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int index = 0; index < Count; ++index) {
          int const peer = index ^ stride;
          if (peer > index) {
            bool const descending = (index & size) == 0;
            bool const swap =
                descending ? values[index] < values[peer] : values[index] > values[peer];
            if (swap) {
              uint32_t const temporary = values[index];
              values[index] = values[peer];
              values[peer] = temporary;
            }
          }
        }
      }
    }
  }

  template <int Count> CUTE_DEVICE static void sort_descending(uint64_t (&values)[Count]) {
    CUTLASS_PRAGMA_UNROLL
    for (int size = 2; size <= Count; size <<= 1) {
      CUTLASS_PRAGMA_UNROLL
      for (int stride = size >> 1; stride > 0; stride >>= 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int index = 0; index < Count; ++index) {
          int const peer = index ^ stride;
          if (peer > index) {
            bool const descending = (index & size) == 0;
            bool const swap =
                descending ? values[index] < values[peer] : values[index] > values[peer];
            if (swap) {
              uint64_t const temporary = values[index];
              values[index] = values[peer];
              values[peer] = temporary;
            }
          }
        }
      }
    }
  }

  template <int Count> CUTE_DEVICE static void sort_descending_by_score(uint64_t (&values)[Count]) {
    CUTLASS_PRAGMA_UNROLL
    for (int size = 2; size <= Count; size <<= 1) {
      CUTLASS_PRAGMA_UNROLL
      for (int stride = size >> 1; stride > 0; stride >>= 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int index = 0; index < Count; ++index) {
          int const peer = index ^ stride;
          if (peer > index) {
            bool const descending = (index & size) == 0;
            uint32_t const score = static_cast<uint32_t>(values[index] >> 32);
            uint32_t const peer_score = static_cast<uint32_t>(values[peer] >> 32);
            bool const swap = descending ? score < peer_score : score > peer_score;
            if (swap) {
              uint64_t const temporary = values[index];
              values[index] = values[peer];
              values[peer] = temporary;
            }
          }
        }
      }
    }
  }

  template <int Count>
  CUTE_DEVICE static void sort_descending_by_raw_score(uint64_t (&values)[Count]) {
    CUTLASS_PRAGMA_UNROLL
    for (int size = 2; size <= Count; size <<= 1) {
      CUTLASS_PRAGMA_UNROLL
      for (int stride = size >> 1; stride > 0; stride >>= 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int index = 0; index < Count; ++index) {
          int const peer = index ^ stride;
          if (peer > index) {
            bool const descending = (index & size) == 0;
            float const score = __uint_as_float(static_cast<uint32_t>(values[index] >> 32));
            float const peer_score = __uint_as_float(static_cast<uint32_t>(values[peer] >> 32));
            bool const swap = descending ? score < peer_score : score > peer_score;
            if (swap) {
              uint64_t const temporary = values[index];
              values[index] = values[peer];
              values[peer] = temporary;
            }
          }
        }
      }
    }
  }

  template <int Left, int Right, int Count>
  CUTE_DEVICE static void compare_exchange_descending_raw_score(uint64_t (&values)[Count]) {
    static_assert(Left < Right && Right < Count);
    float const left_score = __uint_as_float(static_cast<uint32_t>(values[Left] >> 32));
    float const right_score = __uint_as_float(static_cast<uint32_t>(values[Right] >> 32));
    if (left_score < right_score) {
      uint64_t const temporary = values[Left];
      values[Left] = values[Right];
      values[Right] = temporary;
    }
  }

  CUTE_DEVICE static void sort_descending_nine_raw_score(uint64_t (&values)[9]) {
    compare_exchange_descending_raw_score<0, 1>(values);
    compare_exchange_descending_raw_score<3, 4>(values);
    compare_exchange_descending_raw_score<6, 7>(values);
    compare_exchange_descending_raw_score<1, 2>(values);
    compare_exchange_descending_raw_score<4, 5>(values);
    compare_exchange_descending_raw_score<7, 8>(values);
    compare_exchange_descending_raw_score<0, 1>(values);
    compare_exchange_descending_raw_score<3, 4>(values);
    compare_exchange_descending_raw_score<6, 7>(values);
    compare_exchange_descending_raw_score<0, 3>(values);
    compare_exchange_descending_raw_score<3, 6>(values);
    compare_exchange_descending_raw_score<0, 3>(values);
    compare_exchange_descending_raw_score<1, 4>(values);
    compare_exchange_descending_raw_score<4, 7>(values);
    compare_exchange_descending_raw_score<1, 4>(values);
    compare_exchange_descending_raw_score<2, 5>(values);
    compare_exchange_descending_raw_score<5, 8>(values);
    compare_exchange_descending_raw_score<2, 5>(values);
    compare_exchange_descending_raw_score<1, 3>(values);
    compare_exchange_descending_raw_score<5, 7>(values);
    compare_exchange_descending_raw_score<2, 6>(values);
    compare_exchange_descending_raw_score<2, 4>(values);
    compare_exchange_descending_raw_score<4, 6>(values);
    compare_exchange_descending_raw_score<2, 3>(values);
    compare_exchange_descending_raw_score<5, 6>(values);
  }

  CUTE_DEVICE static uint32_t ordered_float_bits(float value) {
    uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffu) == 0) {
      bits = 0;
    }
    uint32_t const negative_mask = static_cast<uint32_t>(-static_cast<int32_t>(bits >> 31));
    return bits ^ (negative_mask | 0x80000000u);
  }

  CUTE_DEVICE static uint64_t exact_key(float score, int page) {
    return (static_cast<uint64_t>(ordered_float_bits(score)) << 32) | static_cast<uint32_t>(~page);
  }

  CUTE_DEVICE static int exact_key_page(uint64_t key) {
    return static_cast<int>(~static_cast<uint32_t>(key));
  }

  CUTE_DEVICE static uint64_t raw_score_key(float score, int page) {
    return (static_cast<uint64_t>(__float_as_uint(score)) << 32) | static_cast<uint32_t>(page);
  }

  CUTE_DEVICE static float raw_score_key_score(uint64_t key) {
    return __uint_as_float(static_cast<uint32_t>(key >> 32));
  }

  CUTE_DEVICE static int raw_score_key_page(uint64_t key) {
    return static_cast<int>(static_cast<uint32_t>(key));
  }

  CUTE_DEVICE static float warp_reduce_max(float value) {
    float output;
    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(output) : "f"(value));
    return output;
  }

  CUTE_DEVICE static float warp_reduce_min(float value) {
    float output;
    asm volatile("redux.sync.min.f32 %0, %1, 0xffffffff;" : "=f"(output) : "f"(value));
    return output;
  }

  CUTE_DEVICE static void hmma_e4m3(float (&accumulator)[4], uint32_t const (&a)[4],
                                    uint2 const &b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]),
                   "+f"(accumulator[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y));
  }

  CUTE_DEVICE static void load_matrix_x4(uint32_t (&values)[4], uint32_t address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3])
                 : "r"(address));
  }

  CUTE_DEVICE static int swizzled_page_offset(int token, int quarter) {
    return (token >> 1) * 128 + ((((token & 1) * 4 + quarter) ^ ((token >> 1) & 7)) * 16);
  }

  CUTE_DEVICE static uint32_t broadcast_scale_byte(uint2 scales, int index) {
    uint32_t const word = index < 4 ? scales.x : scales.y;
    return __byte_perm(word, 0, (index & 3) * 0x1111);
  }

  CUTE_DEVICE static GlobalControl *global_control(Params const &params) {
    auto *base = static_cast<uint8_t *>(params.tp_unicast_ptr);
    return reinterpret_cast<GlobalControl *>(
        base + mega_atten::m3_decode::tp4_nvls::kIndexerControlOffset);
  }

  CUTE_DEVICE static float *score_scratch(Params const &params) {
    auto *base = static_cast<uint8_t *>(params.tp_unicast_ptr);
    return reinterpret_cast<float *>(base + mega_atten::m3_decode::tp4_nvls::kIndexerScoresOffset);
  }

  CUTE_DEVICE static float *score_row(Params const &params, int row, int kv_length) {
    int const page_count = (kv_length + kPageTokens - 1) / kPageTokens;
    int const row_stride = Traits::kIndexerTopKItemsPerThread == 0 ? kMaxPages : page_count;
    return score_scratch(params) + static_cast<size_t>(row) * row_stride;
  }

  CUTE_DEVICE static uint64_t *cta_candidate_scratch(Params const &params) {
    return reinterpret_cast<uint64_t *>(score_scratch(params));
  }

  CUTE_DEVICE static uint64_t current_grid_id() {
    uint64_t grid_id;
    asm volatile("mov.u64 %0, %%gridid;" : "=l"(grid_id));
    return grid_id;
  }

  CUTE_DEVICE static uint32_t load_acquire(uint32_t const *address) {
    uint32_t value;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
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

  CUTE_DEVICE static void store_no_allocate(uint64_t *address, uint64_t value) {
    asm volatile("st.global.L1::no_allocate.u64 [%0], %1;" : : "l"(address), "l"(value) : "memory");
  }

  CUTE_DEVICE static void release_add(uint32_t *address, uint32_t value) {
    asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                 :
                 : "l"(address), "r"(value)
                 : "memory");
  }

  CUTE_DEVICE static void publish_scores(Params const &params) {
    if constexpr (kUnifiedTopK) {
      uint64_t *arrivals = &Scheduler::runtime_indexer_control(params)->arrivals;
      asm volatile("red.async.release.gpu.global.add.u64 [%0], %1;" ::"l"(arrivals),
                   "l"(uint64_t(1))
                   : "memory");
    } else {
      release_add(&global_control(params)->arrivals.value, 1);
    }
  }

  CUTE_DEVICE static uint64_t publish_generation(Params const &params, int thread_idx) {
    if constexpr (kUnifiedTopK) {
      return Scheduler::invocation_epoch(params);
    }
    GlobalControl *control = global_control(params);
    uint32_t const generation = static_cast<uint32_t>(current_grid_id());
    if (m3_decode_logical_block_idx() == 0 && thread_idx == 16) {
      uint32_t const next_epoch =
          load_acquire(&control->arrivals.value) / kArrivalsPerInvocation + 1;
      store_release(&control->invocation.value,
                    static_cast<uint64_t>(generation) << 32 | next_epoch);
    }
    return generation;
  }

  CUTE_DEVICE static uint64_t wait_generation(Params const &params, uint64_t generation) {
    if constexpr (kUnifiedTopK) {
      return generation;
    }
    uint64_t const *tagged = &global_control(params)->invocation.value;
    uint64_t observed;
    do {
      observed = load_acquire(tagged);
      if (static_cast<uint32_t>(observed >> 32) != generation) {
        m3_decode_check_deadline();
        __nanosleep(64);
      }
    } while (static_cast<uint32_t>(observed >> 32) != generation);
    return static_cast<uint32_t>(observed);
  }

  CUTE_DEVICE static void wait_all_scores(Params const &params, uint64_t epoch) {
    if constexpr (kUnifiedTopK) {
      uint64_t const target = epoch * kArrivalsPerInvocation;
      uint64_t const *arrivals = &Scheduler::runtime_indexer_control(params)->arrivals;
      while (load_acquire(arrivals) < target) {
        m3_decode_check_deadline();
        __nanosleep(32);
      }
    } else {
      uint32_t const target = static_cast<uint32_t>(epoch) * kArrivalsPerInvocation;
      uint32_t const *arrivals = &global_control(params)->arrivals.value;
      while (load_acquire(arrivals) < target) {
        m3_decode_check_deadline();
        __nanosleep(32);
      }
    }
  }

  CUTE_DEVICE static void store_remote_int4(int4 value, int4 *address, uint64_t *barrier,
                                            int peer_rank) {
    uint32_t const remote_address =
        cute::set_block_rank(cute::cast_smem_ptr_to_uint(address), peer_rank);
    uint32_t const remote_barrier =
        cute::set_block_rank(cute::cast_smem_ptr_to_uint(barrier), peer_rank);
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
                 "[%0], {%1,%2,%3,%4}, [%5];"
                 :
                 : "r"(remote_address), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w),
                   "r"(remote_barrier)
                 : "memory");
  }

  CUTE_DEVICE static void initialize_topk_mailbox(Storage &storage, bool attention_cluster,
                                                  int thread_idx) {
    if (!attention_cluster) {
      return;
    }
    if (thread_idx == 0) {
      cutlass::arch::ClusterTransactionBarrier::init(storage.indexer_topk_barrier_ptr(), 1);
    }
    cutlass::arch::fence_barrier_init();
    if (thread_idx == 0) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
          storage.indexer_topk_barrier_ptr(), kTopKTransactionBytes);
    }
  }
};

} // namespace cutlass::m3_decode::collective
