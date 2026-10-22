#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>

#include "common/kernel/include/sm100_m3_decode_storage.hpp"
#include "common/kernel/include/tp4_mailbox_layout.hpp"
#include "cutlass/cutlass.h"

namespace cutlass::m3_decode::collective {

namespace mailbox = mega_atten::m3_decode::tp4_mailbox;

struct Sm100M3DecodeTp4Mailbox {
  static constexpr int kHidden = mailbox::kHiddenSize;
  static constexpr int kTokens = mailbox::kTokenCount;
  static constexpr int kRanks = mailbox::kWorldSize;
  static constexpr int kPayloadElements = kTokens * kHidden;
  static constexpr int kVectorElements = 8;
  static constexpr int kConsumerThreads = 128;
  static constexpr int kWarpCount = kConsumerThreads / cutlass::NumThreadsPerWarp;
  static constexpr uint32_t kEmptyPair = uint32_t(mailbox::kEmpty) << 16 | mailbox::kEmpty;

  CUTE_DEVICE static uint32_t sanitize(uint32_t word) {
    if ((word & 0xffffu) == mailbox::kEmpty) {
      word &= 0xffff0000u;
    }
    if ((word >> 16) == mailbox::kEmpty) {
      word &= 0x0000ffffu;
    }
    return word;
  }

  CUTE_DEVICE static bool incomplete(uint32_t word) {
    return (word & 0xffffu) == mailbox::kEmpty || (word >> 16) == mailbox::kEmpty;
  }

  CUTE_DEVICE static int4 load(uint64_t address) {
    int4 value;
    asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(address)
                 : "memory");
    return value;
  }

  CUTE_DEVICE static void publish(uint64_t address, int4 value) {
    // Negative zero is reserved for empty payload lanes; publish positive zero instead.
    value.x = sanitize(value.x);
    value.y = sanitize(value.y);
    value.z = sanitize(value.z);
    value.w = sanitize(value.w);
    asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};"
                 :
                 : "l"(address), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
                 : "memory");
  }

  CUTE_DEVICE static uint64_t offset(int stage, int rank, int token, int hidden) {
    return mailbox::kPayloadOffset +
           uint64_t((stage * kRanks + rank) * kPayloadElements + token * kHidden + hidden) *
               sizeof(__nv_bfloat16);
  }

  CUTE_DEVICE static float warp_sum(float x) {
    CUTLASS_PRAGMA_UNROLL
    for (int mask = 16; mask > 0; mask >>= 1) {
      x += __shfl_xor_sync(0xffffffffu, x, mask);
    }
    return x;
  }

  template <class Params>
  CUTE_DEVICE static uint32_t *epochs(Params const &params, size_t byte_offset) {
    return reinterpret_cast<uint32_t *>(static_cast<uint8_t *>(params.tp_unicast_ptr) +
                                        byte_offset);
  }

  template <class Params>
  CUTE_DEVICE static void publish_tile(Params const &params, int tile, int tid, int4 value) {
    constexpr int kTileChannels = mega_atten::m3_decode::tp4_nvls::kOutputTileChannels;
    constexpr int kVectorsPerTileRow = kTileChannels / kVectorElements;
    int const token = tid / kVectorsPerTileRow;
    int const hidden = tile * kTileChannels + (tid % kVectorsPerTileRow) * kVectorElements;
    int const stage = epochs(params, mailbox::kPublishEpochOffset)[tile];
    publish(reinterpret_cast<uint64_t>(params.tp_multicast_ptr) +
                offset(stage, params.tp_rank, token, hidden),
            value);
  }

  template <class Params> CUTE_DEVICE static void finish_publish(Params const &params, int tile) {
    auto *publish_epochs = epochs(params, mailbox::kPublishEpochOffset);
    publish_epochs[tile] = (publish_epochs[tile] + 1) % mailbox::kGenerationCount;
  }

  template <class Params>
  CUTE_DEVICE static void consume(Params const &params, uint8_t *smem, int token, int tid,
                                  int cluster_rank) {
    // All peers must retire Attention's previous use of the shared scratch.
    auto cluster = cooperative_groups::this_cluster();
    cluster.sync();
    uint64_t const unicast = reinterpret_cast<uint64_t>(params.tp_unicast_ptr);
    int const fragment = cluster_rank * kConsumerThreads + tid;
    bool const valid = tid < kConsumerThreads && fragment < kHidden / kVectorElements;
    int const hidden = fragment * kVectorElements;
    int4 gamma{}, residual{}, prenorm{};
    if (valid) {
      gamma = reinterpret_cast<int4 const *>(params.rms_weight_ptr)[fragment];
      residual = reinterpret_cast<int4 const *>(
          params.residual_in_ptr)[token * (kHidden / kVectorElements) + fragment];
    }
    auto *consume_epochs = epochs(params, mailbox::kConsumeEpochOffset);
    int const owner = token * mailbox::kClusterCtas + cluster_rank;
    int const stage = consume_epochs[owner];
    float sumsq = 0.0f;
    if (valid) {
      int4 peers[kRanks];
      bool retry;
      do {
        retry = false;
        CUTLASS_PRAGMA_UNROLL
        for (int rank = 0; rank < kRanks; ++rank) {
          peers[rank] = load(unicast + offset(stage, rank, token, hidden));
          retry |= incomplete(peers[rank].x) || incomplete(peers[rank].y) ||
                   incomplete(peers[rank].z) || incomplete(peers[rank].w);
        }
        if (retry) {
          m3_decode_check_deadline();
        }
      } while (retry);
      auto *values = reinterpret_cast<__nv_bfloat16 *>(&prenorm);
      auto const *add = reinterpret_cast<__nv_bfloat16 const *>(&residual);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kVectorElements; ++i) {
        float sum = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int rank = 0; rank < kRanks; ++rank) {
          sum += __bfloat162float(reinterpret_cast<__nv_bfloat16 const *>(&peers[rank])[i]);
        }
        values[i] = __float2bfloat16_rn(sum + __bfloat162float(add[i]));
        float const x = __bfloat162float(values[i]);
        sumsq += x * x;
      }
      reinterpret_cast<int4 *>(
          params.attention_residual_ptr)[token * (kHidden / kVectorElements) + fragment] = prenorm;
      // Each consumer clears only its local copy. Three generations separate reuse
      // from a peer's outstanding publication; full operations are stream ordered.
      CUTLASS_PRAGMA_UNROLL
      for (int rank = 0; rank < kRanks; ++rank) {
        auto *address =
            reinterpret_cast<volatile int4 *>(unicast + offset(stage, rank, token, hidden));
        address->x = kEmptyPair;
        address->y = kEmptyPair;
        address->z = kEmptyPair;
        address->w = kEmptyPair;
      }
    }
    auto *scratch = reinterpret_cast<float *>(smem);
    if (tid < kConsumerThreads) {
      sumsq = warp_sum(sumsq);
      if (tid % cutlass::NumThreadsPerWarp == 0) {
        scratch[tid / cutlass::NumThreadsPerWarp] = sumsq;
      }
    }
    __syncthreads();
    if (tid < cutlass::NumThreadsPerWarp) {
      float x = tid < kWarpCount ? scratch[tid] : 0.0f;
      x = warp_sum(x);
      if (tid < mailbox::kClusterCtas) {
        *cluster.map_shared_rank(scratch + kWarpCount + cluster_rank, tid) = x;
      }
    }
    cluster.sync();
    if (valid) {
      float total = 0.0f;
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < mailbox::kClusterCtas; ++c) {
        total += scratch[kWarpCount + c];
      }
      float const inverse = rsqrtf(total / float(kHidden) + params.rms_eps);
      int4 norm{};
      auto const *x = reinterpret_cast<__nv_bfloat16 const *>(&prenorm);
      auto const *g = reinterpret_cast<__nv_bfloat16 const *>(&gamma);
      auto *out = reinterpret_cast<__nv_bfloat16 *>(&norm);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kVectorElements; ++i) {
        out[i] =
            __float2bfloat16_rn(__bfloat162float(x[i]) * inverse * (1.0f + __bfloat162float(g[i])));
      }
      reinterpret_cast<int4 *>(
          params.post_attention_norm_ptr)[token * (kHidden / kVectorElements) + fragment] = norm;
    }
    __syncthreads();
    if (tid == 0) {
      consume_epochs[owner] = (stage + 1) % mailbox::kGenerationCount;
    }
    __syncthreads();
  }
};

} // namespace cutlass::m3_decode::collective
