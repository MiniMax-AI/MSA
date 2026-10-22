#pragma once

#include <cstddef>
#include <cstdint>

#include "attn/kernel/include/m3_decode_output_projection_handoff.hpp"
#include "attn/kernel/include/sm100_m3_decode_problem.hpp"
#include "common/kernel/include/sm100_m3_decode_storage.hpp"
#include "cutlass/cutlass.h"

namespace cutlass::m3_decode::collective {

template <class Traits> struct Sm100M3DecodeSharedControl {
  using Params = Sm100M3DecodeKernelParams<Traits>;

  static constexpr int kTokenCount = mega_atten::m3_decode::output_projection::kProblemN;
  static constexpr int kHiddenSize = mega_atten::m3_decode::output_projection::kProblemM;
  static constexpr int kFc1Owners = 12;
  static constexpr int kActivationPartitions = 4;
  static constexpr int kActivationProducersPerPartition = 2 * Traits::kMoeQueryTokens;
  static constexpr size_t kPayloadBytes = kTokenCount * kHiddenSize * sizeof(uint16_t);
  static constexpr size_t kPayloadBufferCount = 2;
  static constexpr size_t kEntryOffset = kPayloadBufferCount * kPayloadBytes;
  static constexpr size_t kControlOffset = kEntryOffset + 256;

  struct alignas(128) Counter {
    uint32_t value;
    uint32_t padding[31];
  };

  struct alignas(128) TaggedEpoch {
    uint64_t value;
    uint64_t padding[15];
  };

  struct alignas(128) Storage {
    TaggedEpoch invocation_epoch;
    Counter fc1_arrivals;
    Counter activation_partition_arrivals[kActivationPartitions];
  };

  static_assert(kPayloadBytes == 98304);
  static_assert(kEntryOffset == 196608);
  static_assert(sizeof(Counter) == 128);
  static_assert(sizeof(TaggedEpoch) == 128);
  static_assert(sizeof(Storage) == 768);
  static_assert(kControlOffset + sizeof(Storage) <= kEntryOffset + 4096);

  CUTE_DEVICE static Storage *storage(Params const &params) {
    auto *base = static_cast<uint8_t *>(params.tp_unicast_ptr);
    return reinterpret_cast<Storage *>(base + kControlOffset);
  }

  CUTE_DEVICE static uint32_t load_acquire(uint32_t const *address) {
    uint32_t value;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
    return value;
  }

  CUTE_DEVICE static uint32_t load_relaxed(uint32_t const *address) {
    uint32_t value;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
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

  CUTE_DEVICE static uint64_t current_grid_id() {
    uint64_t grid_id;
    asm volatile("mov.u64 %0, %%gridid;" : "=l"(grid_id));
    return grid_id;
  }

  CUTE_DEVICE static void release_add(uint32_t *address, uint32_t value) {
    asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                 :
                 : "l"(address), "r"(value)
                 : "memory");
  }

  CUTE_DEVICE static uint32_t epoch(Params const &params, int publisher_cta = 0) {
    Storage *state = storage(params);
    uint32_t const generation = static_cast<uint32_t>(current_grid_id());
    uint64_t *tagged_epoch = &state->invocation_epoch.value;
    if (m3_decode_logical_block_idx() == publisher_cta && static_cast<int>(blockIdx.y) == 0 &&
        static_cast<int>(blockIdx.z) == 0 && static_cast<int>(threadIdx.x) == 0) {
      uint32_t const next_epoch = load_acquire(&state->fc1_arrivals.value) / kFc1Owners + 1;
      uint64_t const desired = static_cast<uint64_t>(generation) << 32 | next_epoch;
      store_release(tagged_epoch, desired);
    }
    uint64_t observed;
    do {
      m3_decode_check_deadline();
      observed = load_acquire(tagged_epoch);
    } while ((observed >> 32) != generation || static_cast<uint32_t>(observed) == 0);
    return static_cast<uint32_t>(observed);
  }

  CUTE_DEVICE static uint32_t wait_epoch(Params const &params) {
    uint32_t const generation = static_cast<uint32_t>(current_grid_id());
    uint64_t observed;
    do {
      m3_decode_check_deadline();
      observed = load_acquire(&storage(params)->invocation_epoch.value);
    } while ((observed >> 32) != generation || static_cast<uint32_t>(observed) == 0);
    return static_cast<uint32_t>(observed);
  }

  CUTE_DEVICE static void publish_activation_partition(Params const &params, int partition) {
    release_add(&storage(params)->activation_partition_arrivals[partition].value, 1);
  }

  CUTE_DEVICE static void wait_activation_partition(Params const &params, uint32_t epoch,
                                                    int partition) {
    uint32_t const expected = epoch * kActivationProducersPerPartition;
    uint32_t const *arrivals = &storage(params)->activation_partition_arrivals[partition].value;
    while (load_relaxed(arrivals) < expected) {
      m3_decode_check_deadline();
    }
    static_cast<void>(load_acquire(arrivals));
  }

  CUTE_DEVICE static void publish_fc1(Params const &params) {
    release_add(&storage(params)->fc1_arrivals.value, 1);
  }

  CUTE_DEVICE static void wait_fc1(Params const &params, uint32_t epoch) {
    uint32_t const expected = epoch * kFc1Owners;
    uint32_t const *arrivals = &storage(params)->fc1_arrivals.value;
    while (load_relaxed(arrivals) < expected) {
      m3_decode_check_deadline();
    }
    static_cast<void>(load_acquire(arrivals));
  }
};

} // namespace cutlass::m3_decode::collective
