#pragma once

#include "attn/kernel/include/sm100_m3_decode_output_projection_config.hpp"

namespace cutlass::m3_decode::collective {

template <class Traits>
struct Sm100M3DecodeOutputProjectionHandoff : Sm100M3DecodeOutputProjectionConfig<Traits> {
  using Base = Sm100M3DecodeOutputProjectionConfig<Traits>;
  using Base::kHandoffGroupCount;
  using Base::kHandoffGroupProducerCount;
  using Base::kProducerSplitsPerGroup;
  using typename Base::Params;
  struct alignas(16) StageHandoffState {
    uint32_t arrivals;
    uint32_t reserved[3];
  };

  static_assert(sizeof(StageHandoffState) ==
                    mega_atten::m3_decode::output_projection::kStageRecordWords32 *
                        sizeof(uint32_t),
                "projection handoff state layout must match its workspace.");

  CUTLASS_DEVICE static uint32_t load_relaxed_gpu(uint32_t const *ptr) {
    uint32_t value;
    uint64_t const address = __cvta_generic_to_global(ptr);
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.u32 %0, [%1];"
                 : "=r"(value)
                 : "l"(address)
                 : "memory");
    return value;
  }

  CUTLASS_DEVICE static void reduce_async_release_gpu(uint32_t *ptr, uint32_t value) {
    uint64_t const address = __cvta_generic_to_global(ptr);
    asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                 :
                 : "l"(address), "r"(value)
                 : "memory");
  }

  CUTLASS_DEVICE static void store_relaxed_gpu(uint32_t *ptr, uint32_t value) {
    uint64_t const address = __cvta_generic_to_global(ptr);
    asm volatile("st.relaxed.gpu.u32 [%0], %1;" : : "l"(address), "r"(value) : "memory");
  }

  CUTLASS_DEVICE static StageHandoffState *stage_handoff_states(Params const &params) {
    return reinterpret_cast<StageHandoffState *>(params.output_projection_barrier_ptr);
  }

  CUTLASS_DEVICE static uint32_t *handoff_consumer_epochs(Params const &params) {
    return reinterpret_cast<uint32_t *>(stage_handoff_states(params) + kHandoffGroupCount);
  }

  CUTLASS_DEVICE static void wait_activation_group(Params const &params, int group,
                                                   uint32_t epoch) {
    uint32_t const expected = (epoch + 1) * kHandoffGroupProducerCount;
    uint32_t const *arrivals = &stage_handoff_states(params)[group].arrivals;
    while (load_relaxed_gpu(arrivals) != expected) {
      m3_decode_check_deadline();
    }
  }

  CUTLASS_DEVICE static void publish_activation_stages(Params const &params, int split_idx,
                                                       int correction_warp_idx) {
    if (correction_warp_idx != 0) {
      return;
    }

    // The first correction warp stores both rows owned by this split CTA.
    // The warp sync makes those value/SF stores visible to the elected lane
    // before it publishes the GPU-scope release arrivals.
    __syncwarp();
    if (cute::elect_one_sync()) {
      int const group = split_idx / kProducerSplitsPerGroup;
      reduce_async_release_gpu(&stage_handoff_states(params)[group].arrivals, 1);
    }
  }
};

} // namespace cutlass::m3_decode::collective
