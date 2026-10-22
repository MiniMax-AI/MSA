#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "attn/kernel/include/m3_decode_output_projection_handoff.hpp"
#include "common/kernel/include/m3_decode_sm100_params.hpp"
#include "common/kernel/include/sm100_m3_decode_params.hpp"

namespace cutlass::m3_decode::collective {

enum class Sm100M3DecodeTmaDtype {
  U8,
  E4M3,
  E2M1,
  BF16,
};

template <class Traits> struct M3DecodeMegaKernelParamsBuilder {
  using TmaDescriptors = typename Sm100M3DecodeKernelParams<Traits>::TmaDescriptors;

  static int q_tokens_per_batch(const M3DecodeMegaKernelParams &src);

  static int q_tokens_total(const M3DecodeMegaKernelParams &src);

  static int o_tokens_total(const M3DecodeMegaKernelParams &src);

  static cudaError_t encode_tma(CUtensorMap &desc, Sm100M3DecodeTmaDtype dtype, int rank,
                                void *gmem_ptr, uint64_t const *global_dim,
                                uint64_t const *global_stride_bytes, uint32_t const *box_dim,
                                CUtensorMapSwizzle swizzle, bool unpack4b = false);

  static cudaError_t build_q_desc(const M3DecodeMegaKernelParams &src, TmaDescriptors &dst);

  static cudaError_t build_kv_desc(CUtensorMap &desc, void *ptr, int num_kv_heads,
                                   int stride_head_bytes, int64_t stride_page_bytes,
                                   int total_pages);

  static cudaError_t build_indexer_k_desc(const M3DecodeMegaKernelParams &src, TmaDescriptors &dst);

  static cudaError_t build_indexer_k_scale_desc(const M3DecodeMegaKernelParams &src,
                                                TmaDescriptors &dst);

  static cudaError_t build_scale_desc(CUtensorMap &desc, void *ptr, int num_kv_heads,
                                      int64_t stride_page_bytes, int total_pages);

  static cudaError_t build_o_desc(const M3DecodeMegaKernelParams &src, TmaDescriptors &dst);

  static cudaError_t build_output_projection_descs(const M3DecodeMegaKernelParams &src,
                                                   TmaDescriptors &dst);

  static cudaError_t build_mxfp8_weight_descs(CUtensorMap &weight_desc, CUtensorMap &scale_desc,
                                              void *weight_ptr, void *scale_ptr, uint64_t problem_m,
                                              uint64_t problem_k);

  static cudaError_t build_mxfp8_activation_descs(CUtensorMap &activation_desc,
                                                  CUtensorMap &scale_desc, void *activation_ptr,
                                                  void *scale_ptr, uint64_t problem_k);

  static cudaError_t build_shared_expert_descs(const M3DecodeMegaKernelParams &src,
                                               TmaDescriptors &dst);

  static cudaError_t build_routed_expert_fc1_descs(const M3DecodeMegaKernelParams &src,
                                                   TmaDescriptors &dst);

  static cudaError_t build_routed_expert_fc2_descs(const M3DecodeMegaKernelParams &src,
                                                   TmaDescriptors &dst);

  static void copy_common_params(const M3DecodeMegaKernelParams &src,
                                 Sm100M3DecodeKernelParams<Traits> &dst);

  static void build_attention_projection_params(const M3DecodeMegaKernelParams &src,
                                                Sm100M3DecodeKernelParams<Traits> &dst);

  static cudaError_t build_attention_descs(const M3DecodeMegaKernelParams &src,
                                           Sm100M3DecodeKernelParams<Traits> &dst);

  static cudaError_t build_moe_descs(const M3DecodeMegaKernelParams &src,
                                     Sm100M3DecodeKernelParams<Traits> &dst);

  static cudaError_t build_mega_atten(const M3DecodeMegaKernelParams &src,
                                      Sm100M3DecodeKernelParams<Traits> &dst);

  static cudaError_t build_mega_moe(const M3DecodeMegaKernelParams &src,
                                    Sm100M3DecodeKernelParams<Traits> &dst);

  static cudaError_t build(const M3DecodeMegaKernelParams &src,
                           Sm100M3DecodeKernelParams<Traits> &dst);
};

} // namespace cutlass::m3_decode::collective
