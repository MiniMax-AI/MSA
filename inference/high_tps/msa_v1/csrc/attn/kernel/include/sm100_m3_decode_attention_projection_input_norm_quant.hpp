#pragma once

#include "attn/kernel/include/sm100_m3_decode_attention_projection_config.hpp"

namespace cutlass::m3_decode::collective::attention_projection {

CUTE_DEVICE float warp_xor_sum(float value) {
  for (int offset = 16; offset >= 1; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

CUTE_DEVICE float subgroup4_xor_max(float value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2));
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
  return value;
}

CUTE_DEVICE void store_ready_release(uint64_t *address, uint64_t value) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;" : : "l"(address), "l"(value) : "memory");
}

template <class Policy>
CUTE_DEVICE void input_workspace_phase(KernelParams const &params, SharedStorage &shared,
                                       BarrierStorage &barriers, int cluster_id, int rank,
                                       int tid) {
  int const lane = tid & 31;
  int const warp = tid >> 5;
  float values[3];

#pragma unroll
  for (int elem = 0; elem < 3; ++elem) {
    int const local_dim = tid + elem * kThreads;
    int const global_dim = rank * kInputKPerCta + local_dim;
    int const norm_index = local_dim + (local_dim / kSfVec) * 4;
    shared.input_norm_multiplier[norm_index] =
        1.0f + __bfloat162float(params.input_norm_weight[global_dim]);
    float value = 0.0f;
    if (Policy::valid_token(cluster_id, params)) {
      __nv_bfloat16 const input = params.layer_input[cluster_id * kHidden + global_dim];
      value = __bfloat162float(input);
      shared.input_raw[norm_index] = value;
    }
    values[elem] = value;
  }

  // This association is part of the frozen input-workspace contract.
  float thread_sum = (values[0] * values[0] + values[1] * values[1]) + values[2] * values[2];
  thread_sum = warp_xor_sum(thread_sum);
  if (lane == 0) {
    shared.input_warp_sums[warp] = thread_sum;
  }
  sm100::fence_proxy_async_shared();
  cutlass::arch::NamedBarrier(kThreads, kAttentionProjectionBarrierId).arrive_and_wait();

  if (tid == 0) {
    float const p01 = shared.input_warp_sums[0] + shared.input_warp_sums[1];
    float const p23 = shared.input_warp_sums[2] + shared.input_warp_sums[3];
    float const p45 = shared.input_warp_sums[4] + shared.input_warp_sums[5];
    float const p67 = shared.input_warp_sums[6] + shared.input_warp_sums[7];
    shared.input_partial[0] = (p01 + p23) + (p45 + p67);
    for (int peer = 0; peer < kClusterSize; ++peer) {
      if (peer != rank) {
        int const sender_slot = rank - (rank > peer ? 1 : 0);
        uint32_t const remote_destination =
            sm100::map_shared_rank(shared.input_mailbox + sender_slot * 4, uint32_t(peer));
        uint32_t const remote_barrier =
            sm100::map_shared_rank(&barriers.input_mailbox, uint32_t(peer));
        uint32_t const bits = __float_as_uint(shared.input_partial[0]);
        // Only the first FP32 in each padded mailbox slot is consumed.  Complete exactly those
        // four transaction bytes instead of sending a synthetic FP32x4 payload.
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 "
                     "[%0], %2, [%1];"
                     :
                     : "r"(remote_destination), "r"(remote_barrier), "r"(bits)
                     : "memory");
      }
    }
    cute::wait_barrier(barriers.input_mailbox, 0);
    float source[8];
#pragma unroll
    for (int source_rank = 0; source_rank < kClusterSize; ++source_rank) {
      if (source_rank == rank) {
        source[source_rank] = shared.input_partial[0];
      } else {
        int const sender_slot = source_rank - (source_rank > rank ? 1 : 0);
        source[source_rank] = shared.input_mailbox[sender_slot * 4];
      }
    }
    float const c01 = source[0] + source[1];
    float const c23 = source[2] + source[3];
    float const c45 = source[4] + source[5];
    float const c67 = source[6] + source[7];
    float const total = (c01 + c23) + (c45 + c67);
    shared.input_rms_inv[0] = 1.0f / sqrtf(total / float(kHidden) + params.input_norm_epsilon);
  }

  if (warp == 0 || warp == 1 || warp == 3) {
    cutlass::arch::NamedBarrier(96, 1).arrive_and_wait();
  }

  float const rms_inv = shared.input_rms_inv[0];
  // Production schedule: W1 owns local groups 0..11 and W3 owns 12..23; each K32 group is a
  // four-lane subgroup with eight values per lane.
  if (warp == 1 || warp == 3) {
    int const lane_in_group = lane & 3;
    int const group_in_warp = lane >> 2;
    int const group_base = warp == 1 ? 0 : 12;
#pragma unroll
    for (int round = 0; round < 2; ++round) {
      int const local_in_half = round * 8 + group_in_warp;
      bool const valid_group = local_in_half < 12;
      int const local_group = group_base + local_in_half;
      int const global_group = rank * (kInputKPerCta / kSfVec) + local_group;
      float normalized[8];
      float local_amax = 0.0f;
#pragma unroll
      for (int elem = 0; elem < 8; ++elem) {
        float value = 0.0f;
        int const norm_index = local_group * kNormGroupStride + elem * 4 + lane_in_group;
        if (valid_group && Policy::valid_token(cluster_id, params)) {
          value = __bfloat162float(__float2bfloat16_rn((shared.input_raw[norm_index] * rms_inv) *
                                                       shared.input_norm_multiplier[norm_index]));
        }
        normalized[elem] = value;
        local_amax = fmaxf(local_amax, fabsf(value));
      }
      float const amax = subgroup4_xor_max(local_amax);
      uint32_t const scale_code = sm100::float_to_ue8m0_rp_satfinite(amax / 448.0f);
      float const inverse = sm100::ue8m0_to_inv_scale(scale_code);
      if (valid_group) {
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          float scaled0 = 0.0f;
          float scaled1 = 0.0f;
          if (scale_code != 0) {
            scaled0 = normalized[pair * 2] * inverse;
            scaled1 = normalized[pair * 2 + 1] * inverse;
          }
          uint16_t const packed = sm100::fp32x2_to_fp8e4m3x2(scaled0, scaled1);
          int const global_base = global_group * kSfVec;
          int const k0 = pair * 8 + lane_in_group;
          params.workspace_data[cluster_id * kHidden + global_base + k0] = uint8_t(packed);
          params.workspace_data[cluster_id * kHidden + global_base + k0 + 4] = uint8_t(packed >> 8);
        }
        if (lane_in_group == 0) {
          params.workspace_scale[cluster_id * (kHidden / kSfVec) + global_group] =
              uint8_t(scale_code);
        }
      }
    }
  }
  if (warp == 1 || warp == 3) {
    sm100::fence_proxy_async_global();
  }
}

} // namespace cutlass::m3_decode::collective::attention_projection
