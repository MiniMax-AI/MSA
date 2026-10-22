#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "attn/kernel/include/sm100_m3_decode_post_attention_norm.hpp"
#include "attn/kernel/include/sm100_m3_decode_problem.hpp"
#include "common/kernel/include/sm100_m3_decode_nvfp4_quant.hpp"
#include "common/kernel/include/sm100_m3_decode_pipeline.hpp"
#include "common/kernel/include/sm100_m3_decode_scheduler.hpp"
#include "common/kernel/include/sm100_m3_decode_shared_control.hpp"
#include "common/kernel/include/sm100_m3_decode_storage.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/simd_sm100.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "moe_mega/kernel/include/sm100_m3_decode_router_traits.hpp"

namespace cutlass::m3_decode::collective {

struct alignas(16) Sm100RouterCandidate {
  uint32_t selection_key;
  float raw_score;
  int32_t expert_id;
  int32_t padding;
};

struct alignas(8) Sm100RouterResultItem {
  int32_t expert_id;
  float weight;
};

template <class Traits> struct Sm100M3DecodeRouter {
  using Config = Sm100M3DecodeRouterTraits;
  using Params = Sm100M3DecodeKernelParams<Traits>;
  using FusionControl = Sm100M3DecodeSharedControl<Traits>;
  using PostAttentionNorm = Sm100M3DecodePostAttentionNorm<Traits>;
  using Scheduler = Sm100M3DecodeScheduler<Traits>;
  using KernelStorage = cutlass::m3_decode::collective::SharedStorage<Traits>;

  struct alignas(16) SharedStorage {
    float partials[Config::kExperts];
    float peer_partials[Config::kPeerCount][Config::kExpertsPerOwner];
    Sm100RouterCandidate owner_candidates[Config::kExpertsPerOwner];
    Sm100RouterCandidate local_candidates[Config::kTopK];
    Sm100RouterCandidate peer_candidates[Config::kPeerCount][Config::kTopK];
    Sm100RouterResultItem result[Config::kTopK];
    float scale_partials[Config::kScalePartials];
    float norm_scale;
    uint32_t entry_phase;
    alignas(8) uint64_t entry_full;
    alignas(16) uint64_t residual_full;
    uint64_t scale_full;
    alignas(16) uint64_t partial_full;
    uint64_t candidate_full;
    alignas(16) float runtime_gamma_residual[Config::kHiddenPerCta];
  };

  static_assert(sizeof(Sm100RouterCandidate) == 16, "Router candidate must be 16 bytes.");
  static_assert(sizeof(Sm100RouterResultItem) == 8, "RouterResult item must be 8 bytes.");
  static_assert(sizeof(SharedStorage) <= Traits::kClusterReductionSmemBytes,
                "Router overlay must fit recycled attention shared memory.");
  static_assert(sizeof(SharedStorage) <= Config::kResidualHandoffOffset,
                "Router overlay must not overlap the BF16 residual handoff.");
  static_assert(offsetof(SharedStorage, partial_full) >=
                    offsetof(SharedStorage, result) +
                        Config::kTopK * static_cast<int>(sizeof(Sm100RouterResultItem)),
                "Router barriers must not overlap data storage.");

  CUTE_DEVICE static uint64_t *entry_full_barrier(SharedStorage &storage) {
    auto &kernel_storage = *reinterpret_cast<KernelStorage *>(&storage);
    return kernel_storage.attention_pipelines().ptr(KernelStorage::kRouterEntryBarrierSlot);
  }

  CUTE_DEVICE static uint64_t *residual_full_barrier(SharedStorage &storage) {
    auto &kernel_storage = *reinterpret_cast<KernelStorage *>(&storage);
    return kernel_storage.attention_pipelines().ptr(KernelStorage::kRouterResidualBarrierSlot);
  }

  CUTE_DEVICE static uint64_t *scale_full_barrier(SharedStorage &storage) {
    auto &kernel_storage = *reinterpret_cast<KernelStorage *>(&storage);
    return kernel_storage.attention_pipelines().ptr(KernelStorage::kRouterScaleBarrierSlot);
  }

  CUTE_DEVICE static uint64_t *partial_full_barrier(SharedStorage &storage) {
    auto &kernel_storage = *reinterpret_cast<KernelStorage *>(&storage);
    return kernel_storage.attention_pipelines().ptr(KernelStorage::kRouterPartialBarrierSlot);
  }

  CUTE_DEVICE static uint64_t *candidate_full_barrier(SharedStorage &storage) {
    auto &kernel_storage = *reinterpret_cast<KernelStorage *>(&storage);
    return kernel_storage.attention_pipelines().ptr(KernelStorage::kRouterCandidateBarrierSlot);
  }

  CUTE_DEVICE static float warp_sum(float value) {
    value += __shfl_xor_sync(0xffffffffu, value, 16);
    value += __shfl_xor_sync(0xffffffffu, value, 8);
    value += __shfl_xor_sync(0xffffffffu, value, 4);
    value += __shfl_xor_sync(0xffffffffu, value, 2);
    value += __shfl_xor_sync(0xffffffffu, value, 1);
    return value;
  }

  CUTE_DEVICE static void accumulate_prepacked_pair(SharedStorage const &storage,
                                                    float const *router_weight, int expert_0,
                                                    int global_hidden, int local_hidden,
                                                    float &partial_00, float &partial_01,
                                                    float &partial_10, float &partial_11) {
    auto const *residual_handoff = reinterpret_cast<uint2 const *>(
        reinterpret_cast<uint8_t const *>(&storage) + Config::kResidualHandoffOffset);
    uint32_t const residual_address =
        cute::cast_smem_ptr_to_uint(residual_handoff + local_hidden / 4);
    asm volatile("{\n"
                 "  .reg .b32 packed_01, packed_23;\n"
                 "  .reg .b32 residual_0_bits, residual_1_bits;\n"
                 "  .reg .b32 residual_2_bits, residual_3_bits;\n"
                 "  .reg .f32 residual_0, residual_1, residual_2, residual_3;\n"
                 "  .reg .f32 weight_0, weight_1, weight_2, weight_3;\n"
                 "  .reg .s64 global_offset;\n"
                 "  .reg .u64 weight_address;\n"
                 "  cvt.s64.s32 global_offset, %7;\n"
                 "  mad.wide.s32 weight_address, %6, %8, global_offset;\n"
                 "  shl.b64 weight_address, weight_address, 2;\n"
                 "  add.u64 weight_address, weight_address, %5;\n"
                 "  ld.shared.v2.b32 {packed_01, packed_23}, [%4];\n"
                 "  shl.b32 residual_0_bits, packed_01, 16;\n"
                 "  and.b32 residual_1_bits, packed_01, 0xffff0000;\n"
                 "  shl.b32 residual_2_bits, packed_23, 16;\n"
                 "  and.b32 residual_3_bits, packed_23, 0xffff0000;\n"
                 "  mov.b32 residual_0, residual_0_bits;\n"
                 "  mov.b32 residual_1, residual_1_bits;\n"
                 "  mov.b32 residual_2, residual_2_bits;\n"
                 "  mov.b32 residual_3, residual_3_bits;\n"
                 "  ld.global.v4.f32 "
                 "{weight_0, weight_1, weight_2, weight_3}, [weight_address];\n"
                 "  fma.rn.ftz.f32 %0, residual_0, weight_0, %0;\n"
                 "  fma.rn.ftz.f32 %0, residual_1, weight_1, %0;\n"
                 "  fma.rn.ftz.f32 %1, residual_2, weight_2, %1;\n"
                 "  fma.rn.ftz.f32 %1, residual_3, weight_3, %1;\n"
                 "  add.u64 weight_address, weight_address, %9;\n"
                 "  ld.global.v4.f32 "
                 "{weight_0, weight_1, weight_2, weight_3}, [weight_address];\n"
                 "  fma.rn.ftz.f32 %2, residual_0, weight_0, %2;\n"
                 "  fma.rn.ftz.f32 %2, residual_1, weight_1, %2;\n"
                 "  fma.rn.ftz.f32 %3, residual_2, weight_2, %3;\n"
                 "  fma.rn.ftz.f32 %3, residual_3, weight_3, %3;\n"
                 "}\n"
                 : "+f"(partial_00), "+f"(partial_01), "+f"(partial_10), "+f"(partial_11)
                 : "r"(residual_address), "l"(router_weight), "r"(expert_0), "r"(global_hidden),
                   "n"(Config::kHiddenSize),
                   "n"(Config::kDotWarps * Config::kHiddenSize * static_cast<int>(sizeof(float)))
                 : "memory");
  }

  CUTE_DEVICE static void prepare_runtime_gamma_residual(Params const &params,
                                                         SharedStorage &storage, int thread_idx,
                                                         int local_rank) {
    constexpr int kValuesPerThread = 2;
    constexpr int kThreads = Config::kHiddenPerCta / kValuesPerThread;
    if (thread_idx >= kThreads) {
      return;
    }
    int const local_hidden = thread_idx * kValuesPerThread;
    int const global_hidden = local_rank * Config::kHiddenPerCta + local_hidden;
    auto const *residual_handoff = reinterpret_cast<__nv_bfloat162 const *>(
        reinterpret_cast<uint8_t const *>(&storage) + Config::kResidualHandoffOffset);
    auto const *rms_weight = static_cast<__nv_bfloat162 const *>(params.rms_weight_ptr);
    float2 const residual = __bfloat1622float2(residual_handoff[thread_idx]);
    float2 const gamma = __bfloat1622float2(rms_weight[global_hidden / 2]);
    float2 const normalized{
        residual.x * (1.0f + gamma.x),
        residual.y * (1.0f + gamma.y),
    };
    *reinterpret_cast<float2 *>(storage.runtime_gamma_residual + local_hidden) = normalized;
  }

  CUTE_DEVICE static void accumulate_runtime_gamma_pair(SharedStorage const &storage,
                                                        float const *router_weight, int expert_0,
                                                        int global_hidden, int local_hidden,
                                                        float &partial_00, float &partial_01,
                                                        float &partial_10, float &partial_11) {
    auto const *normalized_residual =
        reinterpret_cast<float4 const *>(storage.runtime_gamma_residual);
    uint32_t const residual_address =
        cute::cast_smem_ptr_to_uint(normalized_residual + local_hidden / 4);
    asm volatile("{\n"
                 "  .reg .f32 residual_0, residual_1, residual_2, residual_3;\n"
                 "  .reg .f32 weight_0, weight_1, weight_2, weight_3;\n"
                 "  .reg .s64 global_offset;\n"
                 "  .reg .u64 weight_address;\n"
                 "  cvt.s64.s32 global_offset, %7;\n"
                 "  mad.wide.s32 weight_address, %6, %8, global_offset;\n"
                 "  shl.b64 weight_address, weight_address, 2;\n"
                 "  add.u64 weight_address, weight_address, %5;\n"
                 "  ld.shared.v4.f32 "
                 "{residual_0, residual_1, residual_2, residual_3}, [%4];\n"
                 "  ld.global.v4.f32 "
                 "{weight_0, weight_1, weight_2, weight_3}, [weight_address];\n"
                 "  fma.rn.ftz.f32 %0, residual_0, weight_0, %0;\n"
                 "  fma.rn.ftz.f32 %0, residual_1, weight_1, %0;\n"
                 "  fma.rn.ftz.f32 %1, residual_2, weight_2, %1;\n"
                 "  fma.rn.ftz.f32 %1, residual_3, weight_3, %1;\n"
                 "  add.u64 weight_address, weight_address, %9;\n"
                 "  ld.global.v4.f32 "
                 "{weight_0, weight_1, weight_2, weight_3}, [weight_address];\n"
                 "  fma.rn.ftz.f32 %2, residual_0, weight_0, %2;\n"
                 "  fma.rn.ftz.f32 %2, residual_1, weight_1, %2;\n"
                 "  fma.rn.ftz.f32 %3, residual_2, weight_2, %3;\n"
                 "  fma.rn.ftz.f32 %3, residual_3, weight_3, %3;\n"
                 "}\n"
                 : "+f"(partial_00), "+f"(partial_01), "+f"(partial_10), "+f"(partial_11)
                 : "r"(residual_address), "l"(router_weight), "r"(expert_0), "r"(global_hidden),
                   "n"(Config::kHiddenSize),
                   "n"(Config::kDotWarps * Config::kHiddenSize * static_cast<int>(sizeof(float)))
                 : "memory");
  }

  CUTE_DEVICE static float subgroup_max_4(float value) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1, 4));
    return fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2, 4));
  }

  template <bool InputAlreadyNormalized = false>
  CUTE_DEVICE static void
  quantize_shared_activation_slice(Params const &params, SharedStorage const &storage,
                                   float inverse_rms, int token, int local_rank, int thread_idx,
                                   int input_token = 0) {
    constexpr int kValuesPerThread = 8;
    constexpr int kThreadsPerSlice = Config::kHiddenPerCta / kValuesPerThread;
    constexpr int kThreadsPerScale = PostAttentionNorm::kScaleVectorSize / kValuesPerThread;
    static_assert(kThreadsPerSlice == 96);
    static_assert(kThreadsPerScale == 4);
    if (thread_idx >= kThreadsPerSlice) {
      return;
    }
    int const local_hidden = thread_idx * kValuesPerThread;
    int const global_hidden = local_rank * Config::kHiddenPerCta + local_hidden;
    auto const *residual_handoff = reinterpret_cast<int4 const *>(
        reinterpret_cast<uint8_t const *>(&storage) + Config::kResidualHandoffOffset);
    int4 const residual = residual_handoff[input_token * kThreadsPerSlice + thread_idx];
    int4 gamma_packed{};
    if constexpr (!InputAlreadyNormalized) {
      auto const *gamma = static_cast<__nv_bfloat16 const *>(params.rms_weight_ptr);
      gamma_packed = *reinterpret_cast<int4 const *>(gamma + global_hidden);
    }
    auto const *residual_pair = reinterpret_cast<__nv_bfloat162 const *>(&residual);
    auto const *gamma_pair = reinterpret_cast<__nv_bfloat162 const *>(&gamma_packed);
    float values[kValuesPerThread];
    float absmax = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int pair = 0; pair < kValuesPerThread / 2; ++pair) {
      float2 const residual_f32 = __bfloat1622float2(residual_pair[pair]);
      if constexpr (InputAlreadyNormalized) {
        values[pair * 2] = residual_f32.x;
        values[pair * 2 + 1] = residual_f32.y;
      } else {
        float2 const gamma_f32 = __bfloat1622float2(gamma_pair[pair]);
        values[pair * 2] = residual_f32.x * inverse_rms * (1.0f + gamma_f32.x);
        values[pair * 2 + 1] = residual_f32.y * inverse_rms * (1.0f + gamma_f32.y);
      }
      absmax = fmaxf(absmax, fabsf(values[pair * 2]));
      absmax = fmaxf(absmax, fabsf(values[pair * 2 + 1]));
    }
    float const routed_absmax = fmaxf(absmax, __shfl_xor_sync(0xffffffffu, absmax, 1, 2));
    absmax = subgroup_max_4(absmax);
    uint8_t const scale = PostAttentionNorm::float_to_ue8m0_rp(absmax * (1.0f / 448.0f));
    float const inverse_scale = PostAttentionNorm::ue8m0_to_inverse_scale(scale);
    uint2 const packed{
        PostAttentionNorm::pack_e4m3(values[0] * inverse_scale, values[1] * inverse_scale,
                                     values[2] * inverse_scale, values[3] * inverse_scale),
        PostAttentionNorm::pack_e4m3(values[4] * inverse_scale, values[5] * inverse_scale,
                                     values[6] * inverse_scale, values[7] * inverse_scale)};
    constexpr int kVectorsPerToken = Config::kHiddenSize / kValuesPerThread;
    int const vector = global_hidden / kValuesPerThread;
    static_cast<uint2 *>(params.shared_activation_ptr)[token * kVectorsPerToken + vector] = packed;
    if (thread_idx % kThreadsPerScale == 0) {
      auto *scales = static_cast<uint8_t *>(params.shared_activation_sf_ptr);
      scales[PostAttentionNorm::scale_offset(
          token, global_hidden / PostAttentionNorm::kScaleVectorSize)] = scale;
    }
    float const routed_scale_f32 = fminf(routed_absmax * (1.0f / 6.0f), 448.0f);
    uint8_t const routed_scale = Sm100M3DecodeNvfp4Quant::float_to_e4m3(routed_scale_f32);
    float const routed_scale_rt = Sm100M3DecodeNvfp4Quant::e4m3_to_float(routed_scale);
    float routed_inverse_scale = Sm100M3DecodeNvfp4Quant::reciprocal_approx(routed_scale_rt);
    routed_inverse_scale = fminf(routed_inverse_scale, 3.4028234663852886e38f);
    if (!(routed_scale_rt > 0.0f)) {
      routed_inverse_scale = 0.0f;
    }

    uint32_t routed_packed;
    auto *routed_bytes = reinterpret_cast<uint8_t *>(&routed_packed);
    CUTLASS_PRAGMA_UNROLL
    for (int index = 0; index < kValuesPerThread / 2; ++index) {
      routed_bytes[index] = Sm100M3DecodeNvfp4Quant::pack_e2m1x2_canonical(
          values[index * 2] * routed_inverse_scale, values[index * 2 + 1] * routed_inverse_scale);
    }

    constexpr int kRoutedBytesPerToken = Config::kHiddenSize / 2;
    constexpr int kRoutedBytesPerCta = Config::kHiddenPerCta / 2;
    int const routed_offset = token * kRoutedBytesPerToken + local_rank * kRoutedBytesPerCta +
                              thread_idx * sizeof(routed_packed);
    *reinterpret_cast<uint32_t *>(static_cast<uint8_t *>(params.routed_activation_ptr) +
                                  routed_offset) = routed_packed;

    if ((thread_idx & 1) == 0) {
      constexpr int kRoutedScaleColumns = Config::kHiddenSize / 16;
      constexpr int kRoutedScaleColumnsPerCta = Config::kHiddenPerCta / 16;
      int const routed_scale_column = local_rank * kRoutedScaleColumnsPerCta + thread_idx / 2;
      static_cast<uint8_t *>(
          params.routed_activation_sf_ptr)[token * kRoutedScaleColumns + routed_scale_column] =
          routed_scale;
    }
  }

  CUTE_DEVICE static uint32_t ordered_float_key(float value) {
    value = value == 0.0f ? 0.0f : value;
    uint32_t const bits = __float_as_uint(value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u;
  }

  CUTE_DEVICE static Sm100RouterCandidate invalid_candidate() {
    return Sm100RouterCandidate{0, 0.0f, std::numeric_limits<int32_t>::max(), 0};
  }

  CUTE_DEVICE static void store_local_top4(SharedStorage &storage, Sm100RouterCandidate candidate,
                                           int lane, int expert_base) {
    if (lane >= Config::kExpertsPerOwner) {
      return;
    }
    constexpr uint32_t kActiveMask = (1u << Config::kExpertsPerOwner) - 1;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Config::kTopK; ++k) {
      uint32_t const best_key = __reduce_max_sync(kActiveMask, candidate.selection_key);
      uint32_t const winner_mask = __ballot_sync(kActiveMask, candidate.selection_key == best_key);
      int const winner = __ffs(winner_mask) - 1;
      float const raw_score = __shfl_sync(kActiveMask, candidate.raw_score, winner);
      if (lane == 0) {
        storage.local_candidates[k] =
            Sm100RouterCandidate{best_key, raw_score, expert_base + winner, 0};
      }
      if (lane == winner) {
        candidate = invalid_candidate();
      }
    }
  }

  CUTE_DEVICE static void store_remote_word(uint32_t value, uint32_t *address, uint64_t *barrier,
                                            int destination_rank) {
    cute::store_shared_remote(value, cute::cast_smem_ptr_to_uint(address),
                              cute::cast_smem_ptr_to_uint(barrier), destination_rank);
  }

  CUTE_DEVICE static void store_remote_int4(int4 value, int4 *address, uint64_t *barrier,
                                            int destination_rank) {
    uint32_t const remote_address =
        cute::set_block_rank(cute::cast_smem_ptr_to_uint(address), destination_rank);
    uint32_t const remote_barrier =
        cute::set_block_rank(cute::cast_smem_ptr_to_uint(barrier), destination_rank);
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
                 "[%0], {%1,%2,%3,%4}, [%5];"
                 :
                 : "r"(remote_address), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w),
                   "r"(remote_barrier)
                 : "memory");
  }

  CUTE_DEVICE static void initialize_barriers(SharedStorage &storage, int local_rank,
                                              int thread_idx, int leader_thread) {
    if (thread_idx == leader_thread) {
      cutlass::arch::ClusterTransactionBarrier::init(residual_full_barrier(storage), 1);
      cutlass::arch::ClusterTransactionBarrier::init(scale_full_barrier(storage), 1);
      cutlass::arch::ClusterTransactionBarrier::init(partial_full_barrier(storage), 1);
      if (local_rank > 0 && local_rank < Config::kScalePartials) {
        cutlass::arch::ClusterTransactionBarrier::init(entry_full_barrier(storage), 1);
      }
      if (local_rank == 0) {
        cutlass::arch::ClusterTransactionBarrier::init(candidate_full_barrier(storage), 1);
      }
    }
    cutlass::arch::fence_barrier_init();
  }

  CUTE_DEVICE static void arm_barriers(SharedStorage &storage, int local_rank, int thread_idx,
                                       int leader_thread) {
    if (thread_idx != leader_thread) {
      return;
    }
    if (local_rank > 0 && local_rank < Config::kScalePartials) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(entry_full_barrier(storage),
                                                                     Config::kEntryReadyBytes);
    }
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(residual_full_barrier(storage),
                                                                   Config::kResidualRemoteBytes);
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(scale_full_barrier(storage),
                                                                   Config::kScaleRemoteBytes);
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(partial_full_barrier(storage),
                                                                   Config::kPartialRemoteBytes);
    if (local_rank == 0) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
          candidate_full_barrier(storage), Config::kCandidateRemoteBytes);
    }
  }

  CUTE_DEVICE static void prepare_barriers(SharedStorage &storage, int local_rank, int thread_idx,
                                           int leader_thread) {
    initialize_barriers(storage, local_rank, thread_idx, leader_thread);
    arm_barriers(storage, local_rank, thread_idx, leader_thread);
  }

  CUTE_DEVICE static void prepare_post_attention_norm_barriers(SharedStorage &storage,
                                                               int local_rank, int thread_idx,
                                                               int leader_thread) {
    if (thread_idx == leader_thread) {
      if (local_rank > 0 && local_rank < Config::kScalePartials) {
        cutlass::arch::ClusterTransactionBarrier::init(entry_full_barrier(storage), 1);
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(entry_full_barrier(storage),
                                                                       Config::kEntryReadyBytes);
      }
      cutlass::arch::ClusterTransactionBarrier::init(residual_full_barrier(storage), 1);
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(residual_full_barrier(storage),
                                                                     Config::kResidualRemoteBytes);
      cutlass::arch::ClusterTransactionBarrier::init(scale_full_barrier(storage), 1);
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(scale_full_barrier(storage),
                                                                     Config::kScaleRemoteBytes);
      cutlass::arch::fence_barrier_init();
    }
  }

  CUTE_DEVICE static void load_post_attention_norm_slice(Params const &params,
                                                         SharedStorage &storage, int token,
                                                         int local_rank, int thread_idx) {
    constexpr int kValuesPerThread = 8;
    constexpr int kThreadsPerSlice = Config::kHiddenPerCta / kValuesPerThread;
    if (thread_idx >= kThreadsPerSlice) {
      return;
    }
    constexpr int kVectorsPerToken = Config::kHiddenSize / kValuesPerThread;
    int const vector = local_rank * kThreadsPerSlice + thread_idx;
    auto const *input = static_cast<int4 const *>(params.post_attention_norm_ptr);
    auto *residual_handoff = reinterpret_cast<int4 *>(reinterpret_cast<uint8_t *>(&storage) +
                                                      Config::kResidualHandoffOffset);
    residual_handoff[thread_idx] = input[token * kVectorsPerToken + vector];
  }

  CUTE_DEVICE static void scatter_partials(SharedStorage &storage, int local_rank, int thread_idx) {
    constexpr int kFloatsPerVector = sizeof(int4) / sizeof(float);
    constexpr int kVectorsPerOwner = Config::kExpertsPerOwner / kFloatsPerVector;
    constexpr int kRemoteVectors = Config::kPeerCount * kVectorsPerOwner;
    static_assert(Config::kExpertsPerOwner % kFloatsPerVector == 0);
    if (thread_idx >= kRemoteVectors) {
      return;
    }
    int const peer_slot = thread_idx / kVectorsPerOwner;
    int const owner = peer_slot + (peer_slot >= local_rank ? 1 : 0);
    int const vector = thread_idx - peer_slot * kVectorsPerOwner;
    int const expert_local = vector * kFloatsPerVector;
    int const expert = owner * Config::kExpertsPerOwner + expert_local;
    int const sender_slot = local_rank - (local_rank > owner ? 1 : 0);
    int4 const value = *reinterpret_cast<int4 const *>(&storage.partials[expert]);
    auto *destination = reinterpret_cast<int4 *>(&storage.peer_partials[sender_slot][expert_local]);
    store_remote_int4(value, destination, partial_full_barrier(storage), owner);
  }

  CUTE_DEVICE static float *router_expert_scores(Params const &params) {
    // Routed FC1 overwrites this temporary score workspace only after the
    // Router publishes its schedule.
    return static_cast<float *>(params.routed_mid_ptr);
  }

  CUTE_DEVICE static void run_expert_dot(Params const &params, SharedStorage &storage, int expert,
                                         int warp, int lane) {
    constexpr int kValuesPerVector = 4;
    constexpr int kVectorsPerThread = Config::kHiddenSize / (kValuesPerVector * Config::kThreads);
    static_assert(kVectorsPerThread == 3);

    auto const *router_weight = static_cast<float const *>(params.router_weight_ptr);
    auto const *input = static_cast<__nv_bfloat16 const *>(params.post_attention_norm_ptr);
    float2 partial[Traits::kMoeQueryTokens]{};

    CUTLASS_PRAGMA_UNROLL
    for (int vector = 0; vector < kVectorsPerThread; ++vector) {
      int const hidden = (vector * Config::kThreads + threadIdx.x) * kValuesPerVector;
      float4 const weight =
          *reinterpret_cast<float4 const *>(router_weight + expert * Config::kHiddenSize + hidden);
      float2 const weight_01{weight.x, weight.y};
      float2 const weight_23{weight.z, weight.w};
      CUTLASS_PRAGMA_UNROLL
      for (int token = 0; token < Traits::kMoeQueryTokens; ++token) {
        auto const *input_pair =
            reinterpret_cast<__nv_bfloat162 const *>(input + token * Config::kHiddenSize + hidden);
        float2 const input_01 = __bfloat1622float2(input_pair[0]);
        float2 const input_23 = __bfloat1622float2(input_pair[1]);
        cute::fma(partial[token], input_01, weight_01, partial[token]);
        cute::fma(partial[token], input_23, weight_23, partial[token]);
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int token = 0; token < Traits::kMoeQueryTokens; ++token) {
      float const reduced = warp_sum(partial[token].x + partial[token].y);
      if (lane == 0) {
        storage.partials[token * Config::kDotWarps + warp] = reduced;
      }
    }
    __syncthreads();

    if (warp < Traits::kMoeQueryTokens) {
      float partial_sum =
          lane < Config::kDotWarps ? storage.partials[warp * Config::kDotWarps + lane] : 0.0f;
      partial_sum = warp_sum(partial_sum);
      if (lane == 0) {
        float raw_score = 1.0f / (1.0f + __expf(-partial_sum));
        if (raw_score != raw_score) {
          raw_score = 0.0f;
        }
        router_expert_scores(params)[warp * Config::kExperts + expert] = raw_score;
      }
    }
  }

  CUTE_DEVICE static void finalize_global_token_top4_from_scores(Params const &params, int token,
                                                                 int lane) {
    float const *scores = router_expert_scores(params);
    auto const *correction_bias = static_cast<float const *>(params.router_correction_bias_ptr);
    int const score_base = token * Config::kExperts + lane;
    float const raw_score_0 = scores[score_base];
    float const raw_score_1 = scores[score_base + cutlass::NumThreadsPerWarp];
    float const raw_score_2 = scores[score_base + 2 * cutlass::NumThreadsPerWarp];
    float const raw_score_3 = scores[score_base + 3 * cutlass::NumThreadsPerWarp];
    uint32_t key_0 = ordered_float_key(raw_score_0 + correction_bias[lane]);
    uint32_t key_1 =
        ordered_float_key(raw_score_1 + correction_bias[lane + cutlass::NumThreadsPerWarp]);
    uint32_t key_2 =
        ordered_float_key(raw_score_2 + correction_bias[lane + 2 * cutlass::NumThreadsPerWarp]);
    uint32_t key_3 =
        ordered_float_key(raw_score_3 + correction_bias[lane + 3 * cutlass::NumThreadsPerWarp]);

    float denominator = 0.0f;
    float result_raw_score = 0.0f;
    int result_expert_id = std::numeric_limits<int32_t>::max();
    CUTLASS_PRAGMA_UNROLL
    for (int result = 0; result < Config::kTopK; ++result) {
      uint32_t local_key = key_0;
      float local_raw_score = raw_score_0;
      int local_expert = lane;
      if (key_1 > local_key) {
        local_key = key_1;
        local_raw_score = raw_score_1;
        local_expert = lane + cutlass::NumThreadsPerWarp;
      }
      if (key_2 > local_key) {
        local_key = key_2;
        local_raw_score = raw_score_2;
        local_expert = lane + 2 * cutlass::NumThreadsPerWarp;
      }
      if (key_3 > local_key) {
        local_key = key_3;
        local_raw_score = raw_score_3;
        local_expert = lane + 3 * cutlass::NumThreadsPerWarp;
      }
      uint32_t const best_key = __reduce_max_sync(0xffffffffu, local_key);
      uint32_t const winner_mask = __ballot_sync(0xffffffffu, local_key == best_key);
      int const winner_lane = __ffs(winner_mask) - 1;
      float const raw_score = __shfl_sync(0xffffffffu, local_raw_score, winner_lane);
      int const expert_id = __shfl_sync(0xffffffffu, local_expert, winner_lane);
      denominator += raw_score;
      if (lane == result) {
        result_raw_score = raw_score;
        result_expert_id = expert_id;
      }
      if (lane == winner_lane) {
        if (local_expert == lane) {
          key_0 = 0;
        } else if (local_expert == lane + cutlass::NumThreadsPerWarp) {
          key_1 = 0;
        } else if (local_expert == lane + 2 * cutlass::NumThreadsPerWarp) {
          key_2 = 0;
        } else {
          key_3 = 0;
        }
      }
    }

    denominator = denominator > 0.0f ? denominator : 1.0f;
    if (lane < Config::kTopK) {
      int const output = token * Config::kTopK + lane;
      static_cast<int32_t *>(params.router_topk_idx_ptr)[output] = result_expert_id;
      static_cast<float *>(params.router_topk_weight_ptr)[output] = result_raw_score / denominator;
    }
  }

  CUTE_DEVICE static bool run_expert_normalized_input_task(Params const &params, uint8_t *smem,
                                                           int thread_idx, int expert,
                                                           int quant_token, int quant_local_rank) {
    auto &storage = *reinterpret_cast<SharedStorage *>(smem);
    int const warp = thread_idx / cutlass::NumThreadsPerWarp;
    int const lane = thread_idx % cutlass::NumThreadsPerWarp;

    if (quant_token >= 0 && (Traits::kMoeQueryTokens == Config::kClusterCtas ||
                             quant_token < Traits::kMoeQueryTokens)) {
      load_post_attention_norm_slice(params, storage, quant_token, quant_local_rank, thread_idx);
      __syncthreads();
      quantize_shared_activation_slice<true>(params, storage, 1.0f, quant_token, quant_local_rank,
                                             thread_idx);
      __syncthreads();
      if (thread_idx == 0) {
        FusionControl::publish_activation_partition(params, quant_local_rank / 2);
      }
    }

    run_expert_dot(params, storage, expert, warp, lane);
    __syncthreads();
    if (thread_idx == 0) {
      storage.entry_phase = static_cast<uint32_t>(Scheduler::arrive_router_expert(params));
    }
    __syncthreads();
    bool const is_last_expert = storage.entry_phase != 0;
    if (is_last_expert && warp < Traits::kMoeQueryTokens) {
      finalize_global_token_top4_from_scores(params, warp, lane);
    }
    __syncthreads();
    return is_last_expert && warp == 0;
  }

  CUTE_DEVICE static void send_owner_candidates(SharedStorage &storage, int local_rank, int lane) {
    if (local_rank == 0 || lane >= Config::kCandidateBytesPerOwner / sizeof(uint32_t)) {
      return;
    }
    auto *source = reinterpret_cast<uint32_t *>(storage.local_candidates);
    auto *destination = reinterpret_cast<uint32_t *>(storage.peer_candidates[local_rank - 1]);
    store_remote_word(source[lane], destination + lane, candidate_full_barrier(storage), 0);
  }

  CUTE_DEVICE static Sm100RouterCandidate load_owner_candidate(SharedStorage const &storage,
                                                               int owner, int index) {
    if (owner == 0) {
      return storage.local_candidates[index];
    }
    return storage.peer_candidates[owner - 1][index];
  }

  CUTE_DEVICE static void finalize_top4(SharedStorage &storage, int lane) {
    if (lane >= Config::kClusterCtas) {
      return;
    }
    constexpr uint32_t kActiveMask = (1u << Config::kClusterCtas) - 1;
    int cursor = 0;
    float denominator = 0.0f;
    float result_raw_score = 0.0f;
    int result_expert_id = std::numeric_limits<int32_t>::max();
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Config::kTopK; ++k) {
      Sm100RouterCandidate candidate = load_owner_candidate(storage, lane, cursor);
      uint32_t const best_key = __reduce_max_sync(kActiveMask, candidate.selection_key);
      uint32_t const winner_mask = __ballot_sync(kActiveMask, candidate.selection_key == best_key);
      int const winner = __ffs(winner_mask) - 1;
      float const selected_raw_score = __shfl_sync(kActiveMask, candidate.raw_score, winner);
      int const selected_expert_id = __shfl_sync(kActiveMask, candidate.expert_id, winner);
      denominator += selected_raw_score;
      if (lane == k) {
        result_raw_score = selected_raw_score;
        result_expert_id = selected_expert_id;
      }
      cursor += lane == winner;
    }
    denominator = denominator > 0.0f ? denominator : 1.0f;
    if (lane < Config::kTopK) {
      storage.result[lane] =
          Sm100RouterResultItem{result_expert_id, result_raw_score / denominator};
    }
  }

  CUTE_DEVICE static void store_result(Params const &params, SharedStorage const &storage,
                                       int token, int thread_idx) {
    if (thread_idx >= Config::kTopK) {
      return;
    }
    int const offset = token * Config::kTopK + thread_idx;
    static_cast<int32_t *>(params.router_topk_idx_ptr)[offset] =
        storage.result[thread_idx].expert_id;
    static_cast<float *>(params.router_topk_weight_ptr)[offset] = storage.result[thread_idx].weight;
  }

  CUTE_DEVICE static void run_paired_dot(Params const &params, SharedStorage &storage, int warp,
                                         int lane, int local_rank) {
    auto const *router_weight = static_cast<float const *>(params.router_weight_ptr);
    if constexpr (!Traits::kRouterGammaPrepacked) {
      prepare_runtime_gamma_residual(params, storage, warp * cutlass::NumThreadsPerWarp + lane,
                                     local_rank);
      __syncthreads();
    }
    CUTLASS_PRAGMA_NO_UNROLL
    for (int pair_wave = 0; pair_wave < Config::kExpertWaves / 2; ++pair_wave) {
      int const expert_0 = pair_wave * 2 * Config::kDotWarps + warp;
      int const expert_1 = expert_0 + Config::kDotWarps;
      float partial_00 = 0.0f;
      float partial_01 = 0.0f;
      float partial_10 = 0.0f;
      float partial_11 = 0.0f;
      CUTLASS_PRAGMA_UNROLL
      for (int tile = 0; tile < Config::kHiddenPerCta / (4 * cutlass::NumThreadsPerWarp); ++tile) {
        int const local_hidden = tile * 4 * cutlass::NumThreadsPerWarp + lane * 4;
        int const global_hidden = local_rank * Config::kHiddenPerCta + local_hidden;
        if constexpr (Traits::kRouterGammaPrepacked) {
          accumulate_prepacked_pair(storage, router_weight, expert_0, global_hidden, local_hidden,
                                    partial_00, partial_01, partial_10, partial_11);
        } else {
          accumulate_runtime_gamma_pair(storage, router_weight, expert_0, global_hidden,
                                        local_hidden, partial_00, partial_01, partial_10,
                                        partial_11);
        }
      }
      float const partial_0 = warp_sum(partial_00 + partial_01);
      float const partial_1 = warp_sum(partial_10 + partial_11);
      if (lane == 0) {
        storage.partials[expert_0] = partial_0;
        storage.partials[expert_1] = partial_1;
      }
    }
  }

  CUTE_DEVICE static void run_from_residual(Params const &params, SharedStorage &storage,
                                            int thread_idx, int local_rank, int token) {
    int const warp = thread_idx / cutlass::NumThreadsPerWarp;
    int const lane = thread_idx % cutlass::NumThreadsPerWarp;
    run_paired_dot(params, storage, warp, lane, local_rank);

    __syncthreads();
    scatter_partials(storage, local_rank, thread_idx);

    if (thread_idx == 0) {
      Sm100M3DecodeBarrier::wait(partial_full_barrier(storage), 0);
    }
    __syncthreads();

    float source_partial = 0.0f;
    if (lane < Config::kClusterCtas) {
      int const expert = local_rank * Config::kExpertsPerOwner + warp;
      if (lane == local_rank) {
        source_partial = storage.partials[expert];
      } else {
        int const sender_slot = lane - (lane > local_rank ? 1 : 0);
        source_partial = storage.peer_partials[sender_slot][warp];
      }
    }
    source_partial += __shfl_xor_sync(0xffffffffu, source_partial, 4);
    source_partial += __shfl_xor_sync(0xffffffffu, source_partial, 2);
    source_partial += __shfl_xor_sync(0xffffffffu, source_partial, 1);
    if (lane == 0) {
      source_partial *= storage.norm_scale;
      int const expert = local_rank * Config::kExpertsPerOwner + warp;
      float raw_score = 1.0f / (1.0f + __expf(-source_partial));
      if (raw_score != raw_score) {
        raw_score = 0.0f;
      }
      auto const *correction_bias = static_cast<float const *>(params.router_correction_bias_ptr);
      storage.owner_candidates[warp] = Sm100RouterCandidate{
          ordered_float_key(raw_score + correction_bias[expert]), raw_score, expert, 0};
    }
    __syncthreads();

    if (warp == 0) {
      Sm100RouterCandidate candidate = invalid_candidate();
      if (lane < Config::kExpertsPerOwner) {
        candidate = storage.owner_candidates[lane];
      }
      store_local_top4(storage, candidate, lane, local_rank * Config::kExpertsPerOwner);
      __syncwarp();
      send_owner_candidates(storage, local_rank, lane);

      if (local_rank == 0) {
        if (lane == 0) {
          Sm100M3DecodeBarrier::wait(candidate_full_barrier(storage), 0);
        }
        __syncwarp();
        finalize_top4(storage, lane);
      }
    }

    __syncthreads();
    if (local_rank == 0) {
      store_result(params, storage, token, thread_idx);
    }
  }

  CUTE_DEVICE static void run_task(Params const &params, uint8_t *smem, int thread_idx, int token) {
    auto &storage = *reinterpret_cast<SharedStorage *>(smem);
    int const local_rank = static_cast<int>(cute::block_rank_in_cluster());

    if (thread_idx == 0) {
      Sm100M3DecodeBarrier::wait(scale_full_barrier(storage), 0);
      float square_sum = 0.0f;
      CUTLASS_PRAGMA_UNROLL
      for (int partial = 0; partial < Config::kScalePartials; ++partial) {
        square_sum += storage.scale_partials[partial];
      }
      storage.norm_scale =
          rsqrtf(square_sum / static_cast<float>(Config::kHiddenSize) + params.rms_eps);
    }
    if (thread_idx == 0) {
      Sm100M3DecodeBarrier::wait(residual_full_barrier(storage), 0);
    }
    __syncthreads();
    quantize_shared_activation_slice(params, storage, storage.norm_scale, token, local_rank,
                                     thread_idx);
    __syncthreads();
    if (thread_idx == 0) {
      FusionControl::publish_activation_partition(params, local_rank / 2);
    }
    run_from_residual(params, storage, thread_idx, local_rank, token);
  }

  CUTE_DEVICE static void run(Params const &params, uint8_t *smem, int thread_idx) {
    run_task(params, smem, thread_idx, m3_decode_logical_block_idx() / Config::kClusterCtas);
  }
};

} // namespace cutlass::m3_decode::collective
