// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

struct FMHACutlassSM100Params {
  void *workspace_buffer_ptr;
  void *q_ptr;
  void *k_ptr;
  void *v_ptr;
  int *qo_segment_lens_ptr;
  int *kv_segment_lens_ptr;
  int *qo_segment_offsets_ptr;
  int *kv_segment_offsets_ptr;
  uint64_t *packed_work_range_ptr;
  uint64_t *packed_work_info_ptr;
  void *o_ptr;
  int mask_mode_code;
  float sm_scale;
  int num_qo_heads;
  int num_kv_heads;
  int head_dim_qk;
  int head_dim_vo;
  int q_stride_n;
  int q_stride_h;
  int k_stride_n;
  int k_stride_h;
  int v_stride_n;
  int v_stride_h;
  int batch_size;
  int total_qo_len;
  int total_kv_len;
  int max_qo_len;
  int *qo_offsets_ptr;
  cudaStream_t stream;
  int num_kv_splits;
  int *kv_tile_begin_ptr;
  int *kv_tile_end_ptr;
  int *kv_split_ptr;
  void *workspace_o_ptr;
  float *workspace_lse_ptr;
  int *num_kv_splits_per_row_ptr;
  int *kv_indices_ptr;
  // Per-request base into kv_indices_ptr: request b's logical page p is physical page
  // kv_indices_ptr[kv_indptr_ptr[b] + p]; batch_size + 1 entries.
  int *kv_indptr_ptr;
  int total_page_num;
  int *kv_block_indexes_ptr;
  int kv_block_num;
  int pack_factor = 1;
  int q_tokens_per_batch = 0;
  int h_r_original = 0;
  int q_stride_n_original = 0;
  int q_stride_h_original = 0;
  int total_qo_len_orig = 0;
  void *o_direct_ptr = nullptr;
  int num_qo_heads_orig = 0;
  int num_ctas = 0;
  // TMA fused direct-O unpack is enabled for uniform-Q direct output.
  // When false, epilogue falls back to vec16 software scatter.
  bool tma_direct_o_enabled = false;
  // NVFP4 KV: per-group E4M3 block scales, nominal shape (total_pages, num_kv_heads, page_size,
  // head_dim/16) with token rows of 8 contiguous bytes. Page and head strides are in bytes, like
  // the data strides, so a packed page may hold the data blocks of all heads followed by their
  // scale blocks. K scale blocks are linear (token * 8 + group); V scale blocks use the token-quad
  // order the V converters consume ((token / 4) * 32 + group * 4 + token % 4).
  void *k_scale_ptr = nullptr;
  void *v_scale_ptr = nullptr;
  int k_scale_stride_n = 0;
  int k_scale_stride_h = 0;
  int v_scale_stride_n = 0;
  int v_scale_stride_h = 0;
  // Per-tensor fp32 global scales, one element each: value = code x block_scale x global_scale.
  // The kernel folds the K scale into the softmax scale and the V scale into the output scale and
  // reads both on the device, so captured graphs follow updated scale tensors.
  const float *k_global_scale_ptr = nullptr;
  const float *v_global_scale_ptr = nullptr;
  // Balanced (stream-K) schedule: per work entry, the number of KV segments of its item; and one
  // self-resetting arrival counter per (q token, KV head) item for the in-kernel merge.
  int *kv_split_count_ptr = nullptr;
  int *merge_counter_ptr = nullptr;
  int64_t merge_item_base = 0;
};

using FMHAVariantFn = cudaError_t (*)(const FMHACutlassSM100Params &);

cudaError_t fmha_fwd_reduction_bf16(const void *ptr_O_partial, void *ptr_O, const float *ptr_lse,
                                    const int *num_kv_splits_per_row, float scale_softmax_log2,
                                    float inv_scale_o, int num_kv_splits, int total_qo_len,
                                    int num_qo_heads, int head_dim_vo, int stride_o_n,
                                    int stride_o_h, int stride_partial_n, int stride_partial_h,
                                    cudaStream_t stream);
