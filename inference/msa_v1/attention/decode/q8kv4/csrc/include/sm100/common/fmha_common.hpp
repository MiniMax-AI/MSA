#pragma once

#include <climits>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

#include "decode_attention_params.hpp"
#include "fmha_tile_scheduler.hpp"
#include "sm100_fmha_q8kv4_traits.hpp"

namespace cutlass::fmha::collective {

struct Sm100FmhaTmaDescriptors {
  CUtensorMap q{};
  CUtensorMap k{};
  CUtensorMap v{};
  CUtensorMap k_scale{};
  CUtensorMap v_scale{};
  CUtensorMap o{};
};

struct Sm100FmhaKvTileRange {
  int begin = 0;
  int end = 0;
  int count = 0;
};

struct Sm100FmhaSparseSelection {
  int visible_kv_len = 0;
  int visible_pages = 0;
  int selected_pages = 0;
};

CUTLASS_DEVICE
Sm100FmhaKvTileRange make_kv_tile_range(int full_tile_count, int kv_tile_begin, int kv_tile_end) {
  int begin = kv_tile_begin < 0 ? 0 : kv_tile_begin;
  begin = begin < full_tile_count ? begin : full_tile_count;
  int end = kv_tile_end < full_tile_count ? kv_tile_end : full_tile_count;
  end = end < 0 ? 0 : end;
  end = end < begin ? begin : end;
  return Sm100FmhaKvTileRange{begin, end, end - begin};
}

template <class Traits> struct Sm100FmhaFwdKernelParams {
  using Scheduler = Sm100FmhaScheduler<Traits>;

  Sm100FmhaTmaDescriptors tma;
  typename Scheduler::Params scheduler;

  void const *q_ptr = nullptr;
  void const *k_ptr = nullptr;
  void const *v_ptr = nullptr;
  void const *k_scale_ptr = nullptr;
  void const *v_scale_ptr = nullptr;
  void *o_ptr = nullptr;
  void *workspace_o_ptr = nullptr;
  float *workspace_lse_ptr = nullptr;

  uint64_t const *packed_work_range_ptr = nullptr;
  uint64_t const *packed_work_info_ptr = nullptr;
  int const *kv_tile_begin_ptr = nullptr;
  int const *kv_tile_end_ptr = nullptr;
  int const *kv_split_ptr = nullptr;
  int const *kv_split_count_ptr = nullptr;
  int *merge_counter_ptr = nullptr;
  int64_t merge_item_base = 0;
  int const *kv_indices_ptr = nullptr;
  int const *kv_block_indexes_ptr = nullptr;
  int const *qo_segment_lens_ptr = nullptr;
  int const *kv_segment_lens_ptr = nullptr;
  int const *qo_segment_offsets_ptr = nullptr;
  int const *kv_segment_offsets_ptr = nullptr;

  int batch_size = 0;
  int total_qo_len = 0;
  int total_pages = 0;
  int num_qo_heads = 0;
  int num_kv_heads = 0;
  int num_qo_heads_orig = 0;
  int max_qo_len = 0;
  int q_tokens_per_batch = 1;
  int scheduler_total_logical_ctas = 0;
  int scheduler_max_active_ctas = 0;
  int max_kv_len = 0;
  int kv_page_stride = 0;
  int kv_block_num = 0;
  int num_ctas = 0;
  int num_kv_splits = 1;
  float scale_softmax = 1.0f;
  float scale_q = 1.0f;
  float scale_k = 1.0f;
  float scale_v = 1.0f;
  float scale_o = 1.0f;
  float scale_softmax_log2 = 1.0f;
  float scale_output = 1.0f;
  float scale_output_split = 1.0f;
  bool use_persistent_scheduler = false;
  bool use_precomputed_scheduler = false;
};

template <class Traits>
CUTLASS_DEVICE int fmha_fwd_visible_kv_length(Sm100FmhaFwdKernelParams<Traits> const &params,
                                              int batch_idx, int q_token_idx) {

  int const q_tokens =
      params.qo_segment_lens_ptr != nullptr
          ? (__ldg(params.qo_segment_lens_ptr + batch_idx) + Traits::kHeadGroup - 1) /
                Traits::kHeadGroup
          : params.q_tokens_per_batch;
  if (q_token_idx < 0 || q_token_idx >= q_tokens) {
    return 0;
  }
  int const full_kv_len = params.kv_segment_lens_ptr != nullptr
                              ? __ldg(params.kv_segment_lens_ptr + batch_idx)
                              : params.kv_page_stride * Traits::kPageSize;
  int const causal_trim = (q_tokens - 1) - q_token_idx;
  int const kv_len = full_kv_len - causal_trim;
  return kv_len > 0 ? kv_len : 0;
}

template <class Traits>
CUTLASS_DEVICE int fmha_fwd_q_token_count(Sm100FmhaFwdKernelParams<Traits> const &params,
                                          int batch_idx) {

  if (params.qo_segment_lens_ptr != nullptr) {
    int const packed_q_len = __ldg(params.qo_segment_lens_ptr + batch_idx);
    return (packed_q_len + Traits::kHeadGroup - 1) / Traits::kHeadGroup;
  }
  return params.q_tokens_per_batch;
}

template <class Traits>
CUTLASS_DEVICE int fmha_fwd_packed_q_offset(Sm100FmhaFwdKernelParams<Traits> const &params,
                                            int batch_idx) {

  return params.qo_segment_offsets_ptr != nullptr
             ? __ldg(params.qo_segment_offsets_ptr + batch_idx)
             : batch_idx * params.q_tokens_per_batch * Traits::kHeadGroup;
}

template <class Traits>
CUTLASS_DEVICE int fmha_fwd_q_token_offset(Sm100FmhaFwdKernelParams<Traits> const &params,
                                           int batch_idx) {
  return fmha_fwd_packed_q_offset<Traits>(params, batch_idx) / Traits::kHeadGroup;
}

template <class Traits>
CUTLASS_DEVICE int fmha_fwd_q_token_global_index(Sm100FmhaFwdKernelParams<Traits> const &params,
                                                 int batch_idx, int q_token_idx) {

  return fmha_fwd_q_token_offset<Traits>(params, batch_idx) + q_token_idx;
}

template <class Traits>
CUTLASS_DEVICE int fmha_fwd_packed_row_base(Sm100FmhaFwdKernelParams<Traits> const &params,
                                            int batch_idx, int q_token_idx) {
  return fmha_fwd_packed_q_offset<Traits>(params, batch_idx) + q_token_idx * Traits::kHeadGroup;
}

template <class Traits>
CUTLASS_DEVICE Sm100FmhaSparseSelection
fmha_fwd_sparse_selection(Sm100FmhaFwdKernelParams<Traits> const &params, int batch_idx,
                          int kv_head_idx, int q_token_idx) {
  int const visible_kv_len = fmha_fwd_visible_kv_length<Traits>(params, batch_idx, q_token_idx);
  int const visible_pages = (visible_kv_len + Traits::kPageSize - 1) / Traits::kPageSize;
  if (visible_pages <= 0 || params.kv_block_indexes_ptr == nullptr) {
    return Sm100FmhaSparseSelection{visible_kv_len, visible_pages, 0};
  }
  (void)kv_head_idx;
  int const selected_pages =
      visible_pages < Traits::kSparseTopK ? visible_pages : Traits::kSparseTopK;
  return Sm100FmhaSparseSelection{visible_kv_len, visible_pages, selected_pages};
}

template <class Traits>
CUTLASS_DEVICE int fmha_fwd_sparse_kv_length(Sm100FmhaFwdKernelParams<Traits> const &params,
                                             int batch_idx, int kv_head_idx, int q_token_idx) {
  Sm100FmhaSparseSelection const selection =
      fmha_fwd_sparse_selection<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
  if (selection.selected_pages <= 0) {
    return 0;
  }
  int const full_selected_kv_len = selection.selected_pages * Traits::kPageSize;
  int const page_mask = Traits::kPageSize - 1;
  int const tail_correction =
      (Traits::kPageSize - (selection.visible_kv_len & page_mask)) & page_mask;
  return full_selected_kv_len - tail_correction;
}

template <class Traits>
CUTLASS_DEVICE int fmha_fwd_sparse_kv_tile_count(Sm100FmhaFwdKernelParams<Traits> const &params,
                                                 int batch_idx, int kv_head_idx, int q_token_idx) {
  Sm100FmhaSparseSelection const selection =
      fmha_fwd_sparse_selection<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
  return selection.selected_pages;
}

template <class Traits>
CUTLASS_DEVICE int fmha_fwd_kv_length_for_batch(Sm100FmhaFwdKernelParams<Traits> const &params,
                                                int batch_idx, int kv_head_idx, int q_token_idx) {
  {
    return fmha_fwd_sparse_kv_length<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
  }
}

template <class Traits>
CUTLASS_DEVICE int fmha_fwd_kv_tile_count_for_batch(Sm100FmhaFwdKernelParams<Traits> const &params,
                                                    int batch_idx, int kv_head_idx,
                                                    int q_token_idx) {
  {
    return fmha_fwd_sparse_kv_tile_count<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
  }
}

enum class Sm100FmhaTmaDtype {
  E4M3,
  E2M1,
  BF16,
};

template <class Traits> struct FMHACutlassSM100ParamsBuilder {
  struct SchedulerPolicy {
    int total_logical_ctas = 0;
    int max_active_ctas = 0;
    bool multi_ctas_kv_disabled = true;
    bool persistent_scheduler_fits_smem = true;
    bool use_persistent_scheduler = false;
  };

  static int q_tokens_per_batch(const FMHACutlassSM100Params &src) {
    {
      if (src.q_tokens_per_batch > 0) {
        return src.q_tokens_per_batch;
      }
      int const rows_per_batch = src.batch_size * Traits::kHeadGroup;
      if (rows_per_batch <= 0 || src.total_qo_len <= 0 || src.total_qo_len % rows_per_batch != 0) {
        return 0;
      }
      return src.total_qo_len / rows_per_batch;
    }
    if (src.pack_factor != Traits::kHeadGroup || src.max_qo_len <= 0 ||
        src.max_qo_len % Traits::kHeadGroup != 0) {
      return 0;
    }
    return src.max_qo_len / Traits::kHeadGroup;
  }

  static int q_tokens_total(const FMHACutlassSM100Params &src) {
    if (src.pack_factor != Traits::kHeadGroup || src.total_qo_len <= 0 ||
        src.total_qo_len % Traits::kHeadGroup != 0) {
      return 0;
    }
    return src.total_qo_len / Traits::kHeadGroup;
  }

  static int o_tokens_total(const FMHACutlassSM100Params &src) {
    if (src.total_qo_len_orig > 0) {
      return src.total_qo_len_orig;
    }
    return q_tokens_total(src);
  }

  static int ceil_div(int a, int b) { return (a + b - 1) / b; }

  static int logical_cta_count(const FMHACutlassSM100Params &src, int q_tokens) {
    if (q_tokens <= 0 || src.batch_size <= 0 || src.num_kv_heads <= 0) {
      return 0;
    }

    int const num_qo_heads =
        src.num_qo_heads_orig > 0 ? src.num_qo_heads_orig : src.num_kv_heads * Traits::kHeadGroup;
    int const qhead_per_kv =
        src.h_r_original > 0 ? src.h_r_original : num_qo_heads / src.num_kv_heads;
    int const num_heads_per_cta = qhead_per_kv < Traits::kTileQ ? qhead_per_kv : Traits::kTileQ;
    if (num_heads_per_cta <= 0) {
      return 0;
    }
    int const head_dim_v = Traits::kHeadDim;
    int const head_dim_per_cta_v = Traits::kHeadDim;
    int const num_ctas_for_all_heads = ceil_div(num_qo_heads, num_heads_per_cta);
    int const num_ctas_per_head_dim = ceil_div(head_dim_v, head_dim_per_cta_v);
    int const num_ctas_y = num_ctas_for_all_heads * num_ctas_per_head_dim;
    int const num_ctas_z = src.batch_size;
    int const num_ctas_per_seq_q = q_tokens;
    int64_t const total_logical_ctas = static_cast<int64_t>(num_ctas_per_seq_q) *
                                       static_cast<int64_t>(num_ctas_y) *
                                       static_cast<int64_t>(num_ctas_z);
    return total_logical_ctas > static_cast<int64_t>(INT32_MAX)
               ? INT32_MAX
               : static_cast<int>(total_logical_ctas);
  }

  static SchedulerPolicy scheduler_policy(const FMHACutlassSM100Params &src, int max_active_ctas) {
    SchedulerPolicy policy;
    int const q_tokens = q_tokens_per_batch(src);
    policy.total_logical_ctas = logical_cta_count(src, q_tokens);
    policy.max_active_ctas = max_active_ctas > 0 ? max_active_ctas : 0;
    policy.multi_ctas_kv_disabled = src.num_kv_splits == 1;
    policy.persistent_scheduler_fits_smem = policy.max_active_ctas > 0;
    // CLC persistent scheduling requires a rectangular, uniform sparse grid.
    // Heterogeneous chunks and split-KV keep their existing direct or
    // workspace-scheduled paths. The folded batch raster decodes q_token_idx
    // with bit operations, so non-power-of-two query chunks use the direct
    // grid instead.
    bool const has_power_of_two_q_tokens = q_tokens > 0 && (q_tokens & (q_tokens - 1)) == 0;
    bool const persistent_scheduler_supported =
        has_power_of_two_q_tokens && src.qo_segment_lens_ptr == nullptr &&
        src.kv_block_num == Traits::kSparseTopK && src.num_kv_splits == 1;
    policy.use_persistent_scheduler =
        persistent_scheduler_supported && policy.total_logical_ctas > policy.max_active_ctas &&
        policy.multi_ctas_kv_disabled && policy.persistent_scheduler_fits_smem;
    return policy;
  }

  static void apply_scheduler_policy(const FMHACutlassSM100Params &src, int max_active_ctas,
                                     Sm100FmhaFwdKernelParams<Traits> &dst) {
    SchedulerPolicy const policy = scheduler_policy(src, max_active_ctas);
    dst.scheduler_total_logical_ctas = policy.total_logical_ctas;
    dst.scheduler_max_active_ctas = policy.max_active_ctas;
    dst.use_persistent_scheduler = policy.use_persistent_scheduler;
    dst.use_precomputed_scheduler = false;
  }

  static cudaError_t encode_tma(CUtensorMap &desc, Sm100FmhaTmaDtype dtype, int rank,
                                void *gmem_ptr, uint64_t const *global_dim,
                                uint64_t const *global_stride_bytes, uint32_t const *box_dim,
                                CUtensorMapSwizzle swizzle, bool unpack4b = false) {
    if (gmem_ptr == nullptr || rank < 2 || rank > 5) {
      return cudaErrorInvalidValue;
    }

    CUtensorMapDataType data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    if (dtype == Sm100FmhaTmaDtype::E2M1) {
      data_type = unpack4b ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B : CU_TENSOR_MAP_DATA_TYPE_UINT8;
    } else if (dtype == Sm100FmhaTmaDtype::BF16) {
      data_type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    }

    uint32_t element_stride[5] = {1, 1, 1, 1, 1};
    CUresult result = cuTensorMapEncodeTiled(
        &desc, data_type, rank, gmem_ptr, global_dim, global_stride_bytes, box_dim, element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
  }

  static cudaError_t build_q_desc(const FMHACutlassSM100Params &src, Sm100FmhaTmaDescriptors &dst) {
    int const q_stride_h = src.q_stride_h_original > 0 ? src.q_stride_h_original : src.q_stride_h;
    int const q_stride_n = src.q_stride_n_original > 0 ? src.q_stride_n_original : src.q_stride_n;
    int const q_tokens = q_tokens_total(src);
    if (q_tokens <= 0) {
      return cudaErrorInvalidValue;
    }

    uint64_t global_dim[5] = {
        static_cast<uint64_t>(Traits::kHeadDim), static_cast<uint64_t>(Traits::kHeadGroup),
        static_cast<uint64_t>(src.num_kv_heads), static_cast<uint64_t>(q_tokens), 1u};
    uint64_t global_stride_bytes[4] = {
        static_cast<uint64_t>(q_stride_h), static_cast<uint64_t>(q_stride_h * Traits::kHeadGroup),
        static_cast<uint64_t>(q_stride_n), static_cast<uint64_t>(q_stride_n * q_tokens)};
    uint32_t box_dim[5] = {static_cast<uint32_t>(Traits::kHeadDim),
                           static_cast<uint32_t>(Traits::kHeadGroup), 1u, 1u, 1u};

    return encode_tma(dst.q, Sm100FmhaTmaDtype::E4M3, 5, src.q_ptr, global_dim, global_stride_bytes,
                      box_dim, CU_TENSOR_MAP_SWIZZLE_128B);
  }

  static cudaError_t build_kv_desc(CUtensorMap &desc, void *ptr, int num_kv_heads,
                                   int stride_head_bytes, int stride_page_bytes, int total_pages,
                                   bool unpack4b) {
    if (!unpack4b) {
      uint64_t global_dim[4] = {
          static_cast<uint64_t>(Traits::kHeadDim), static_cast<uint64_t>(Traits::kPageSize / 2),
          static_cast<uint64_t>(num_kv_heads), static_cast<uint64_t>(total_pages)};
      uint64_t global_stride_bytes[3] = {static_cast<uint64_t>(Traits::kHeadDim),
                                         static_cast<uint64_t>(stride_head_bytes),
                                         static_cast<uint64_t>(stride_page_bytes)};
      uint32_t box_dim[4] = {static_cast<uint32_t>(Traits::kHeadDim),
                             static_cast<uint32_t>(Traits::kTileKv / 2), 1u, 1u};
      return encode_tma(desc, Sm100FmhaTmaDtype::E2M1, 4, ptr, global_dim, global_stride_bytes,
                        box_dim, CU_TENSOR_MAP_SWIZZLE_128B, false);
    }

    uint64_t global_dim[4] = {
        static_cast<uint64_t>(Traits::kHeadDim), static_cast<uint64_t>(Traits::kPageSize),
        static_cast<uint64_t>(num_kv_heads), static_cast<uint64_t>(total_pages)};
    uint64_t global_stride_bytes[3] = {static_cast<uint64_t>(Traits::kHeadDim / 2),
                                       static_cast<uint64_t>(stride_head_bytes),
                                       static_cast<uint64_t>(stride_page_bytes)};
    uint32_t box_dim[4] = {static_cast<uint32_t>(Traits::kHeadDim),
                           static_cast<uint32_t>(Traits::kTileKv), 1u, 1u};

    return encode_tma(desc, Sm100FmhaTmaDtype::E2M1, 4, ptr, global_dim, global_stride_bytes,
                      box_dim, CU_TENSOR_MAP_SWIZZLE_128B, true);
  }

  static cudaError_t build_scale_desc(CUtensorMap &desc, void *ptr, int num_kv_heads,
                                      int total_pages) {
    constexpr int kScaleGroups = Traits::kHeadDim / Traits::kScaleGroupSize;
    constexpr int kScaleReshape = 16;
    static_assert(kScaleGroups == 8, "q8kv4 FMHA forward expects 8 scale bytes per token.");
    static_assert(Traits::kTileKv % kScaleReshape == 0,
                  "scale TMA reshape must divide the KV tile.");

    uint64_t global_dim[4] = {static_cast<uint64_t>(kScaleGroups * kScaleReshape),
                              static_cast<uint64_t>(Traits::kPageSize / kScaleReshape),
                              static_cast<uint64_t>(num_kv_heads),
                              static_cast<uint64_t>(total_pages)};
    uint64_t global_stride_bytes[3] = {
        static_cast<uint64_t>(kScaleGroups * kScaleReshape),
        static_cast<uint64_t>(Traits::kPageSize * kScaleGroups),
        static_cast<uint64_t>(num_kv_heads * Traits::kPageSize * kScaleGroups)};
    uint32_t box_dim[4] = {static_cast<uint32_t>(kScaleGroups * kScaleReshape),
                           static_cast<uint32_t>(Traits::kTileKv / kScaleReshape), 1u, 1u};

    return encode_tma(desc, Sm100FmhaTmaDtype::E4M3, 4, ptr, global_dim, global_stride_bytes,
                      box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
  }

  static cudaError_t build_o_desc(const FMHACutlassSM100Params &src, Sm100FmhaTmaDescriptors &dst) {
    void *o_ptr = src.o_direct_ptr != nullptr ? src.o_direct_ptr : src.o_ptr;
    if (o_ptr == nullptr) {
      return cudaErrorInvalidValue;
    }

    constexpr int kOChunkDim = 64;
    constexpr int kOChunks = Traits::kHeadDim / kOChunkDim;
    int const q_tokens = o_tokens_total(src);
    if (q_tokens <= 0) {
      return cudaErrorInvalidValue;
    }
    static_assert(Traits::kHeadDim % kOChunkDim == 0,
                  "O TMA chunks must evenly cover the head dimension.");
    static_assert(kOChunkDim * 2 == 128, "BF16 O TMA innermost dimension must fit 128B swizzle.");

    uint64_t global_dim[5] = {static_cast<uint64_t>(kOChunkDim), static_cast<uint64_t>(kOChunks),
                              static_cast<uint64_t>(src.num_qo_heads_orig),
                              static_cast<uint64_t>(q_tokens), 1u};
    uint64_t global_stride_bytes[4] = {
        static_cast<uint64_t>(kOChunkDim * 2),
        static_cast<uint64_t>(Traits::kHeadDim * 2),
        static_cast<uint64_t>(src.num_qo_heads_orig * Traits::kHeadDim * 2),
        static_cast<uint64_t>(q_tokens * src.num_qo_heads_orig * Traits::kHeadDim * 2),
    };
    uint32_t box_dim[5] = {static_cast<uint32_t>(kOChunkDim), static_cast<uint32_t>(kOChunks),
                           static_cast<uint32_t>(Traits::kHeadGroup), 1u, 1u};

    return encode_tma(dst.o, Sm100FmhaTmaDtype::BF16, 5, o_ptr, global_dim, global_stride_bytes,
                      box_dim, CU_TENSOR_MAP_SWIZZLE_128B);
  }

  static cudaError_t build(const FMHACutlassSM100Params &src,
                           Sm100FmhaFwdKernelParams<Traits> &dst) {
    dst.q_ptr = src.q_ptr;
    dst.k_ptr = src.k_ptr;
    dst.v_ptr = src.v_ptr;
    dst.k_scale_ptr = src.k_scale_ptr;
    dst.v_scale_ptr = src.v_scale_ptr;
    dst.o_ptr = src.o_direct_ptr != nullptr ? src.o_direct_ptr : src.o_ptr;
    dst.workspace_o_ptr = src.workspace_o_ptr;
    dst.workspace_lse_ptr = src.workspace_lse_ptr;
    dst.packed_work_range_ptr = src.packed_work_range_ptr;
    dst.packed_work_info_ptr = src.packed_work_info_ptr;
    dst.kv_tile_begin_ptr = src.kv_tile_begin_ptr;
    dst.kv_tile_end_ptr = src.kv_tile_end_ptr;
    dst.kv_split_ptr = src.kv_split_ptr;
    dst.kv_split_count_ptr = src.kv_split_count_ptr;
    dst.merge_counter_ptr = src.merge_counter_ptr;
    dst.merge_item_base = src.merge_item_base;
    dst.kv_indices_ptr = src.kv_indices_ptr;
    dst.kv_block_indexes_ptr = src.kv_block_indexes_ptr;
    dst.qo_segment_lens_ptr = src.qo_segment_lens_ptr;
    dst.kv_segment_lens_ptr = src.kv_segment_lens_ptr;
    dst.qo_segment_offsets_ptr = src.qo_segment_offsets_ptr;
    dst.kv_segment_offsets_ptr = src.kv_segment_offsets_ptr;
    dst.batch_size = src.batch_size;
    dst.total_qo_len = src.total_qo_len;
    dst.total_pages = src.total_page_num;
    dst.num_qo_heads = src.num_qo_heads;
    dst.num_kv_heads = src.num_kv_heads;
    dst.num_qo_heads_orig = src.num_qo_heads_orig;
    dst.max_qo_len = src.max_qo_len;
    dst.q_tokens_per_batch = q_tokens_per_batch(src);
    dst.scheduler_total_logical_ctas = 0;
    dst.scheduler_max_active_ctas = 0;
    dst.use_persistent_scheduler = false;
    dst.use_precomputed_scheduler = false;
    dst.kv_page_stride = src.kv_page_stride;
    dst.kv_block_num = src.kv_block_num;
    dst.num_ctas = src.num_ctas;
    dst.num_kv_splits = src.num_kv_splits;
    dst.scale_softmax = src.sm_scale;
    dst.scale_q = 1.0f;
    dst.scale_k = 1.0f;
    dst.scale_v = 1.0f;
    dst.scale_o = 1.0f;
    dst.scale_softmax_log2 = src.sm_scale * 1.4426950408889634f;
    dst.scale_output = 1.0f;
    dst.scale_output_split = 1.0f;
    dst.scheduler = Sm100FmhaScheduler<Traits>::to_underlying_arguments(
        dst.q_tokens_per_batch, dst.num_kv_heads, dst.batch_size, cutlass::KernelHardwareInfo{});

    cudaError_t status = build_q_desc(src, dst.tma);
    if (status != cudaSuccess) {
      return status;
    }
    status = build_kv_desc(dst.tma.k, src.k_ptr, src.num_kv_heads, src.k_stride_h, src.k_stride_n,
                           src.total_page_num, false);
    if (status != cudaSuccess) {
      return status;
    }
    status = build_kv_desc(dst.tma.v, src.v_ptr, src.num_kv_heads, src.v_stride_h, src.v_stride_n,
                           src.total_page_num, true);
    if (status != cudaSuccess) {
      return status;
    }
    status =
        build_scale_desc(dst.tma.k_scale, src.k_scale_ptr, src.num_kv_heads, src.total_page_num);
    if (status != cudaSuccess) {
      return status;
    }
    status =
        build_scale_desc(dst.tma.v_scale, src.v_scale_ptr, src.num_kv_heads, src.total_page_num);
    if (status != cudaSuccess) {
      return status;
    }
    status = build_o_desc(src, dst.tma);
    return status;
  }
};

} // namespace cutlass::fmha::collective
