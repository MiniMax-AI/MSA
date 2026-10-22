#pragma once

#include <cuda_bf16.h>

#include <cstdint>

#include "attn/kernel/include/m3_decode_output_projection_handoff.hpp"
#include "common/kernel/include/sm100_m3_decode_storage.hpp"
#include "common/kernel/include/tp4_nvls_layout.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/config.hpp"
#include "cutlass/cutlass.h"
#include "moe_mega/kernel/include/sm100_m3_decode_router.hpp"

namespace cutlass::m3_decode::collective {

template <class Traits> struct Sm100M3DecodeTp4AllReduce {
  using Params = Sm100M3DecodeKernelParams<Traits>;

  static constexpr int kTokenCount = mega_atten::m3_decode::output_projection::kProblemN;
  static constexpr int kHiddenSize = mega_atten::m3_decode::output_projection::kProblemM;
  static constexpr int kClusterSize = Traits::kNumKvSplits;
  static constexpr int kReduceCtasPerToken = Sm100M3DecodeRouterTraits::kScalePartials;
  static constexpr int kEntryCoordinatorRank = 6;
  static constexpr int kElementsPerVector = sizeof(int4) / sizeof(__nv_bfloat16);
  static constexpr int kSystemEntryWord =
      mega_atten::m3_decode::tp4_nvls::kOutputProjectionSystemEntryWord;
  static constexpr int kGpuEntryWord =
      mega_atten::m3_decode::tp4_nvls::kOutputProjectionGpuEntryWord;
  static constexpr int kOutputProjectionEntryWord =
      mega_atten::m3_decode::tp4_nvls::kOutputProjectionCtaEntryWord;
  static constexpr int kTpWorldSize = mega_atten::m3_decode::tp4_nvls::kWorldSize;
  static constexpr int kOutputProjectionCtas = 48;
  static constexpr int kActivePayloadBufferScratch = 16;
  static constexpr int kSliceHidden = kHiddenSize / kClusterSize;
  static constexpr int kResidualHandoffOffset = 8192;
  static constexpr int kResidualVectorsPerCta = kSliceHidden / kElementsPerVector;
  static constexpr int kResidualHandoffBytes =
      kResidualVectorsPerCta * static_cast<int>(sizeof(int4));
  static constexpr int kRouterBarrierInitThread = cutlass::NumThreadsPerWarp;
  static constexpr int kPayloadBytes = mega_atten::m3_decode::tp4_nvls::kPayloadBytes;
  // Ping-pong payloads make the next entry barrier protect buffer reuse.
  static constexpr int kPayloadBufferCount = mega_atten::m3_decode::tp4_nvls::kPayloadBufferCount;
  static constexpr int kEntryOffset = mega_atten::m3_decode::tp4_nvls::kControlOffset;

  static_assert(kResidualHandoffOffset + kResidualHandoffBytes <=
                    Traits::kClusterReductionSmemBytes,
                "TP4 residual handoff must fit the recycled shared storage.");
  static_assert(kResidualHandoffOffset == Sm100M3DecodeRouterTraits::kResidualHandoffOffset,
                "TP4 tail and Router must use the same residual handoff offset.");
  static_assert(kHiddenSize % kClusterSize == 0,
                "TP4 hidden size must divide evenly across the CTA cluster.");
  static_assert(Traits::kNumThreads == 512, "TP4 tail requires 512 threads per CTA.");
  static_assert(kClusterSize == 8, "TP4 tail requires an eight-CTA cluster.");
  static_assert(kEntryCoordinatorRank >= kReduceCtasPerToken &&
                    kEntryCoordinatorRank < kClusterSize,
                "entry coordination must use a non-reducer CTA.");
  static_assert(kReduceCtasPerToken == 2, "TP4 reduction requires two CTA slices.");
  static_assert(kSliceHidden == 768, "TP4 rank-local hidden size must be 768.");
  static_assert(kResidualVectorsPerCta == 96,
                "each residual owner must receive 96 int4 vectors for hidden size 6144.");
  static_assert(kPayloadBytes == 98304, "one TP4 payload buffer must be 98,304 bytes.");
  static_assert(kEntryOffset == 196608,
                "the TP4 control region must follow two 98,304-byte payload buffers.");
  static_assert(kOutputProjectionCtas == 48, "dedicated Output Projection requires 48 CTAs.");
  CUTE_DEVICE static uint32_t load_acquire_sys(uint32_t const *address) {
    uint32_t value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
    return value;
  }

  CUTE_DEVICE static uint32_t load_acquire_gpu(uint32_t const *address) {
    uint32_t value;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
    return value;
  }

  CUTE_DEVICE static uint32_t fetch_add_relaxed_gpu(uint32_t *address, uint32_t value) {
    uint32_t previous;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;"
                 : "=r"(previous)
                 : "l"(address), "r"(value)
                 : "memory");
    return previous;
  }

  CUTE_DEVICE static uint32_t fetch_add_relaxed_sys(uint32_t *address, uint32_t value) {
    uint32_t previous;
    asm volatile("atom.relaxed.sys.global.add.u32 %0, [%1], %2;"
                 : "=r"(previous)
                 : "l"(address), "r"(value)
                 : "memory");
    return previous;
  }

  CUTE_DEVICE static void release_add_gpu(uint32_t *address, uint32_t value) {
    asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                 :
                 : "l"(address), "r"(value)
                 : "memory");
  }

  CUTE_DEVICE static void release_add_sys(uint32_t *address, uint32_t value) {
    asm volatile("red.async.release.sys.global.add.u32 [%0], %1;"
                 :
                 : "l"(address), "r"(value)
                 : "memory");
  }

  CUTE_DEVICE static int4 multicast_load_reduce(int4 const *address) {
    int4 value;
    asm volatile("multimem.ld_reduce.global.add.v4.bf16x2.acc::f32 "
                 "{%0,%1,%2,%3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(address)
                 : "memory");
    return value;
  }

  CUTE_DEVICE static float warp_sum(float value) {
    value += __shfl_xor_sync(0xffffffffu, value, 16);
    value += __shfl_xor_sync(0xffffffffu, value, 8);
    value += __shfl_xor_sync(0xffffffffu, value, 4);
    value += __shfl_xor_sync(0xffffffffu, value, 2);
    value += __shfl_xor_sync(0xffffffffu, value, 1);
    return value;
  }

  CUTE_DEVICE static float warp_sum_16(float value) {
    value += __shfl_xor_sync(0xffffffffu, value, 8);
    value += __shfl_xor_sync(0xffffffffu, value, 4);
    value += __shfl_xor_sync(0xffffffffu, value, 2);
    value += __shfl_xor_sync(0xffffffffu, value, 1);
    return value;
  }

  CUTE_DEVICE static void store_cluster_int4_transaction(int4 value, uint8_t *address,
                                                         uint64_t *barrier, int peer_rank) {
    uint32_t const dsmem_address =
        cute::set_block_rank(cute::cast_smem_ptr_to_uint(address), peer_rank);
    uint32_t const remote_barrier =
        cute::set_block_rank(cute::cast_smem_ptr_to_uint(barrier), peer_rank);
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
                 "[%0], {%1,%2,%3,%4}, [%5];"
                 :
                 : "r"(dsmem_address), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w),
                   "r"(remote_barrier)
                 : "memory");
  }

  CUTE_DEVICE static uint8_t *unicast_base(Params const &params) {
    return reinterpret_cast<uint8_t *>(params.tp_unicast_ptr);
  }

  CUTE_DEVICE static uint8_t *multicast_base(Params const &params) {
    return reinterpret_cast<uint8_t *>(params.tp_multicast_ptr);
  }

  CUTE_DEVICE static int entry_phase(Params const &params, int thread_idx) {
    auto *entry_base = reinterpret_cast<uint32_t const *>(unicast_base(params) + kEntryOffset);
    uint32_t value = 0;
    if ((thread_idx & 31) == 0) {
      value = load_acquire_sys(entry_base + kSystemEntryWord);
    }
    value = __shfl_sync(0xffffffffu, value, 0);
    return (value >> 31) & 1;
  }

  CUTE_DEVICE static uint8_t *output_projection_unicast_payload(Params const &params,
                                                                int thread_idx) {
    // Output Projection writes the selected payload before the entry barrier.
    int const buffer = entry_phase(params, thread_idx);
    return unicast_base(params) + buffer * kPayloadBytes;
  }

  CUTE_DEVICE static uint8_t *active_payload(Params const &params, uint8_t *smem, int thread_idx) {
    uint32_t buffer = 0;
    if ((thread_idx & 31) == 0) {
      auto *scratch = reinterpret_cast<uint32_t const *>(smem);
      buffer = scratch[kActivePayloadBufferScratch];
    }
    buffer = __shfl_sync(0xffffffffu, buffer, 0);
    return multicast_base(params) + buffer * kPayloadBytes;
  }

  CUTE_DEVICE static uint32_t *output_projection_entry(Params const &params) {
    auto *entry = reinterpret_cast<uint32_t *>(unicast_base(params) + kEntryOffset);
    return entry + kOutputProjectionEntryWord;
  }

  CUTE_DEVICE static void publish_output_projection_completion(Params const &params) {
    release_add_gpu(output_projection_entry(params), 1);
  }

  CUTE_DEVICE static void wait_phase_flip_gpu(uint32_t *address, uint32_t arrival_count,
                                              uint32_t total_arrivals) {
    uint32_t const phase_flip = arrival_count + 0x80000000u - total_arrivals;
    uint32_t const old_phase = fetch_add_relaxed_gpu(address, phase_flip) & 0x80000000u;
    while ((load_acquire_gpu(address) & 0x80000000u) == old_phase) {
      m3_decode_check_deadline();
    }
  }

  CUTE_DEVICE static uint32_t wait_phase_flip_sys(uint32_t *address, uint32_t total_arrivals) {
    uint32_t const phase_flip = 0x80000000u - total_arrivals;
    uint32_t const old_phase = fetch_add_relaxed_sys(address, phase_flip) & 0x80000000u;
    while ((load_acquire_sys(address) & 0x80000000u) == old_phase) {
      m3_decode_check_deadline();
    }
    return old_phase;
  }

  CUTE_DEVICE static void wait_phase_change_sys(uint32_t const *address, uint32_t old_phase) {
    while ((load_acquire_sys(address) & 0x80000000u) == old_phase) {
      m3_decode_check_deadline();
    }
  }

  CUTE_DEVICE static uint32_t entry_phase_word(Params const &params) {
    auto *invocation_entry_uc =
        reinterpret_cast<uint32_t const *>(unicast_base(params) + kEntryOffset);
    return load_acquire_sys(invocation_entry_uc + kSystemEntryWord) & 0x80000000u;
  }

  CUTE_DEVICE static int
  arrive_and_wait_entry(Params const &params,
                        typename Sm100M3DecodeRouter<Traits>::SharedStorage &router_storage,
                        int token, int local_rank, uint32_t old_phase) {
    auto *invocation_entry_uc = reinterpret_cast<uint32_t *>(unicast_base(params) + kEntryOffset);
    if (local_rank == kEntryCoordinatorRank) {
      if (token != 0) {
        release_add_gpu(invocation_entry_uc + kGpuEntryWord, 1);
      } else {
        wait_phase_flip_gpu(invocation_entry_uc + kGpuEntryWord, 1, kTokenCount);
        wait_phase_flip_gpu(output_projection_entry(params), 0, kOutputProjectionCtas);
        auto *invocation_entry_mc =
            reinterpret_cast<uint32_t *>(multicast_base(params) + kEntryOffset);
        release_add_sys(invocation_entry_mc + kSystemEntryWord, 1);
        static_cast<void>(
            wait_phase_flip_sys(invocation_entry_uc + kSystemEntryWord, kTpWorldSize));
      }
    }

    if (local_rank >= kReduceCtasPerToken) {
      return 0;
    }
    if (local_rank == 0) {
      wait_phase_change_sys(invocation_entry_uc + kSystemEntryWord, old_phase);
      uint32_t const active_buffer = old_phase >> 31;
      CUTLASS_PRAGMA_UNROLL
      for (int destination = 1; destination < kReduceCtasPerToken; ++destination) {
        Sm100M3DecodeRouter<Traits>::store_remote_word(
            active_buffer, &router_storage.entry_phase,
            Sm100M3DecodeRouter<Traits>::entry_full_barrier(router_storage), destination);
      }
      return static_cast<int>(active_buffer);
    }
    Sm100M3DecodeBarrier::wait(Sm100M3DecodeRouter<Traits>::entry_full_barrier(router_storage), 0);
    static_cast<void>(load_acquire_sys(invocation_entry_uc + kSystemEntryWord));
    return static_cast<int>(router_storage.entry_phase);
  }

  CUTE_DEVICE static void reduce_token(Params const &params, uint8_t *smem, int token,
                                       int local_rank, int thread_idx) {
    auto const *payload = reinterpret_cast<int4 const *>(active_payload(params, smem, thread_idx));
    auto const *residual_in = reinterpret_cast<int4 const *>(params.residual_in_ptr);
    auto *residual_out = reinterpret_cast<int4 *>(params.attention_residual_ptr);
    constexpr int kVectorsPerToken = kHiddenSize / kElementsPerVector;
    int const token_vector = token * kVectorsPerToken;
    constexpr int kVectorsPerCta = kVectorsPerToken / kReduceCtasPerToken;
    constexpr int kActiveReduceWarps =
        (kVectorsPerCta + cutlass::NumThreadsPerWarp - 1) / cutlass::NumThreadsPerWarp;
    static_assert(kActiveReduceWarps <= cutlass::NumThreadsPerWarp / 2,
                  "CTA square-sum partials must fit a 16-lane warp reduction.");
    int const vector_begin = local_rank * kVectorsPerCta;
    float2 local_square_sum{0.0f, 0.0f};
    if (thread_idx < kVectorsPerCta) {
      int const vector = vector_begin + thread_idx;
      int const offset = token_vector + vector;
      int4 value = multicast_load_reduce(payload + offset);
      int4 const residual_value = residual_in[offset];
      auto *value_pair = reinterpret_cast<__nv_bfloat162 *>(&value);
      auto const *residual_pair = reinterpret_cast<__nv_bfloat162 const *>(&residual_value);
      CUTLASS_PRAGMA_UNROLL
      for (int pair = 0; pair < 4; ++pair) {
        float2 const value_f32 = __bfloat1622float2(value_pair[pair]);
        float2 const residual_f32 = __bfloat1622float2(residual_pair[pair]);
        float2 residual_sum;
        cute::add(residual_sum, value_f32, residual_f32);
        value_pair[pair] = __floats2bfloat162_rn(residual_sum.x, residual_sum.y);
        float2 const rounded_f32 = __bfloat1622float2(value_pair[pair]);
        cute::fma(local_square_sum, rounded_f32, rounded_f32, local_square_sum);
      }
      residual_out[offset] = value;
      int const residual_owner = vector / kResidualVectorsPerCta;
      int const residual_vector = vector - residual_owner * kResidualVectorsPerCta;
      auto &router_storage =
          *reinterpret_cast<typename Sm100M3DecodeRouter<Traits>::SharedStorage *>(smem);
      uint64_t *residual_barrier =
          Sm100M3DecodeRouter<Traits>::residual_full_barrier(router_storage);
      store_cluster_int4_transaction(value,
                                     smem + kResidualHandoffOffset + residual_vector * sizeof(int4),
                                     residual_barrier, residual_owner);
    }

    float *scratch = reinterpret_cast<float *>(smem);
    int const lane = thread_idx & 31;
    int const warp = thread_idx >> 5;
    if (warp < kActiveReduceWarps) {
      float square_sum = local_square_sum.x + local_square_sum.y;
      square_sum = warp_sum(square_sum);
      if (lane == 0) {
        scratch[warp] = square_sum;
      }
    }
    // Router activation publication provides the GPU-scope release for
    // residual_out after this CTA-wide writer join.
    __syncthreads();
    if (warp == 0) {
      float cta_sum = lane < kActiveReduceWarps ? scratch[lane] : 0.0f;
      cta_sum = warp_sum_16(cta_sum);
      if (lane == 0) {
        scratch[0] = cta_sum;
      }
      if (lane < kClusterSize) {
        auto &router_storage =
            *reinterpret_cast<typename Sm100M3DecodeRouter<Traits>::SharedStorage *>(smem);
        Sm100M3DecodeRouter<Traits>::store_remote_word(
            __float_as_uint(cta_sum),
            reinterpret_cast<uint32_t *>(&router_storage.scale_partials[local_rank]),
            Sm100M3DecodeRouter<Traits>::scale_full_barrier(router_storage), lane);
      }
    }
  }

  CUTE_DEVICE static void run_task(Params const &params, uint8_t *smem, int thread_idx, int token) {
    int const local_rank = static_cast<int>(cute::block_rank_in_cluster());
    uint32_t old_entry_phase = 0;
    if (thread_idx == 0 && local_rank == 0) {
      old_entry_phase = entry_phase_word(params);
    }
    auto &router_storage =
        *reinterpret_cast<typename Sm100M3DecodeRouter<Traits>::SharedStorage *>(smem);
    Sm100M3DecodeRouter<Traits>::prepare_barriers(router_storage, local_rank, thread_idx,
                                                  kRouterBarrierInitThread);
    __syncthreads();
    cutlass::arch::fence_barrier_init();
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
    if (thread_idx == 0) {
      int const active_buffer =
          arrive_and_wait_entry(params, router_storage, token, local_rank, old_entry_phase);
      if (local_rank < kReduceCtasPerToken) {
        auto *scratch = reinterpret_cast<uint32_t *>(smem);
        scratch[kActivePayloadBufferScratch] = active_buffer;
      }
    }
    __syncthreads();
    if (local_rank < kReduceCtasPerToken) {
      reduce_token(params, smem, token, local_rank, thread_idx);
    }
    Sm100M3DecodeRouter<Traits>::run_task(params, smem, thread_idx, token);
  }

  CUTE_DEVICE static void run(Params const &params, uint8_t *smem, int thread_idx) {
    run_task(params, smem, thread_idx, m3_decode_logical_block_idx() / kClusterSize);
  }
};

} // namespace cutlass::m3_decode::collective
