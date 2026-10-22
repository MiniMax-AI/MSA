#pragma once

#include "attn/kernel/include/sm100_m3_decode_attention_projection_epilogue.hpp"

namespace cutlass::m3_decode::collective::attention_projection {

// Run the 64-CTA QKV/QIKI projection prologue inside the 144-CTA decode
// MegaKernel. The first eight preferred cluster8 groups execute projection;
// the lower 256-thread role layout owns the projection pipeline, while the
// upper eight warps consume Q mailbox tokens as soon as their three splits
// arrive.
CUTE_DEVICE bool run(KernelParams const &params, uint8_t *smem, BarrierStorage &barriers,
                     int thread_idx, int cluster_id, int rank, uint64_t *qi_ready,
                     uint64_t *ki_ready, uint64_t generation) {
  using Policy = KernelPolicy;
  auto &shared = *reinterpret_cast<SharedStorage *>(smem);
  HeadSplit const hs = head_split_for_cta(cluster_id, rank);

  if (thread_idx < kThreads) {
    initialize_ab6_and_tma_descriptors(params, shared, barriers, hs, rank, thread_idx);
  }
  __syncthreads();
  cute::cluster_arrive_relaxed();
  cute::cluster_wait();
  if (thread_idx < kThreads) {
    issue_early_weight_tma(params, shared, barriers, hs, thread_idx);
  }

  if (thread_idx < kThreads) {
    input_workspace_phase<Policy>(params, shared, barriers, cluster_id, rank, thread_idx);
    if ((thread_idx >> 5) == 4) {
      prefetch_weight_tail(params, shared, hs);
    }
    sm100::reusable_grid_sync_active<kClusterCount * kClusterSize, kThreads,
                                     kAttentionProjectionBarrierId>(
        params.workspace_grid_epoch, uint32_t(cluster_id * kClusterSize + rank),
        uint32_t(thread_idx));
  }

  bool const is_output_owner =
      hs.head < kNumQHeads ? rank == ((hs.head & 1) ? 3 : 0) : hs.split == 0;
  if (thread_idx >= kThreads && is_output_owner && hs.head < kNumQHeads) {
    int const lane = thread_idx & 31;
    int const token = (thread_idx - kThreads) >> 5;
    process_q_owner_token<Policy>(params, shared, barriers, hs, token, lane);
  }

  if (thread_idx < kThreads) {
    int const lane = thread_idx & 31;
    int const token = thread_idx >> 5;
    int const dim_base = lane * 4;

    project_ab6_m128n8_mxfp8<Policy>(params, shared, barriers, hs, thread_idx);

    // Cross-CTA Q readiness is carried by q_mailbox_barrier and special
    // readiness by the generation protocol.
    cutlass::arch::NamedBarrier(kThreads, kAttentionProjectionBarrierId).arrive_and_wait();

    if (is_output_owner && hs.head >= kNumQHeads && Policy::valid_token(token, params)) {
      F32x4 values;
      int const special_index = hs.head - kNumQHeads;
      float const *base = params.workspace_special_partial +
                          (special_index * kSpecialSplitK * kTokensPerTile + token) * kHeadDim +
                          dim_base;
      values = sm100::load_global_fp32x4(base);
#pragma unroll
      for (int split = 1; split < kSpecialSplitK; ++split) {
        F32x4 const next = sm100::load_global_fp32x4(base + split * kTokensPerTile * kHeadDim);
        values.x += next.x;
        values.y += next.y;
        values.z += next.z;
        values.w += next.w;
      }
      apply_norm_and_rope(params, shared, hs.head, token, lane, values);
      store_final_row(params, hs.head, token, lane, values);
    }

    // DSM senders may not retire before both Q owners consume their mailbox.
    cutlass::arch::NamedBarrier(kThreads, kAttentionProjectionBarrierId).arrive_and_wait();
  }
  __syncthreads();
  if (thread_idx == 0 && is_output_owner) {
    if (hs.head == 18) {
      store_ready_release(qi_ready, generation | 1u);
    } else if (hs.head == 19) {
      store_ready_release(ki_ready, generation | 1u);
    }
  }
  if (is_output_owner) {
    sm100::reusable_grid_sync<kOutputOwnerCtas>(params.workspace_grid_epoch + 1, uint32_t(hs.head),
                                                uint32_t(thread_idx));
  }
  return is_output_owner;
}

} // namespace cutlass::m3_decode::collective::attention_projection
