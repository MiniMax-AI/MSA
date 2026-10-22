#pragma once

#include <cutlass/cutlass.h>

#include "common/kernel/include/sm100_m3_decode_params.hpp"

namespace cutlass::m3_decode::collective {

struct Sm100M3DecodeKvTileRange {
  int begin = 0;
  int end = 0;
  int count = 0;
};

struct Sm100M3DecodeKvMetadata {
  int kv_length = 0;
  int tile_count = 0;
};

struct Sm100M3DecodeSparseSelection {
  int visible_kv_len = 0;
  int visible_pages = 0;
  int selected_pages = 0;
  int last_selected_page = -1;
};

CUTLASS_DEVICE
Sm100M3DecodeKvTileRange make_kv_tile_range(int full_tile_count, int kv_tile_begin,
                                            int kv_tile_end) {
  int begin = kv_tile_begin < 0 ? 0 : kv_tile_begin;
  begin = begin < full_tile_count ? begin : full_tile_count;
  int end = kv_tile_end < full_tile_count ? kv_tile_end : full_tile_count;
  end = end < 0 ? 0 : end;
  end = end < begin ? begin : end;
  return Sm100M3DecodeKvTileRange{begin, end, end - begin};
}

template <class Traits>
CUTLASS_DEVICE int m3_decode_visible_kv_length(Sm100M3DecodeKernelParams<Traits> const &params,
                                               int batch_idx, int q_token_idx) {
  int const q_tokens = params.q_tokens_per_batch;
  int const full_kv_len = params.kv_segment_lens_ptr != nullptr
                              ? __ldg(params.kv_segment_lens_ptr + batch_idx)
                              : params.kv_page_stride * Traits::kPageSize;
  int const causal_trim = (q_tokens - 1) - q_token_idx;
  return full_kv_len - causal_trim;
}

template <class Traits>
CUTLASS_DEVICE int m3_decode_packed_q_offset(Sm100M3DecodeKernelParams<Traits> const &params,
                                             int batch_idx) {
  return batch_idx * params.q_tokens_per_batch * Traits::kHeadGroup;
}

template <class Traits>
CUTLASS_DEVICE int m3_decode_q_token_offset(Sm100M3DecodeKernelParams<Traits> const &params,
                                            int batch_idx) {
  return m3_decode_packed_q_offset<Traits>(params, batch_idx) / Traits::kHeadGroup;
}

template <class Traits>
CUTLASS_DEVICE int m3_decode_q_token_global_index(Sm100M3DecodeKernelParams<Traits> const &params,
                                                  int batch_idx, int q_token_idx) {
  return m3_decode_q_token_offset<Traits>(params, batch_idx) + q_token_idx;
}

template <class Traits>
CUTLASS_DEVICE int m3_decode_packed_row_base(Sm100M3DecodeKernelParams<Traits> const &params,
                                             int batch_idx, int q_token_idx) {
  return m3_decode_packed_q_offset<Traits>(params, batch_idx) + q_token_idx * Traits::kHeadGroup;
}

template <class Traits>
CUTLASS_DEVICE Sm100M3DecodeSparseSelection
m3_decode_sparse_selection(Sm100M3DecodeKernelParams<Traits> const &params, int batch_idx,
                           int kv_head_idx, int q_token_idx) {
  int const visible_kv_len = m3_decode_visible_kv_length<Traits>(params, batch_idx, q_token_idx);
  int const visible_pages = (visible_kv_len + Traits::kPageSize - 1) / Traits::kPageSize;
  (void)kv_head_idx;
  return Sm100M3DecodeSparseSelection{visible_kv_len, visible_pages, Traits::kSparseTopK,
                                      visible_pages - 1};
}

template <class Traits>
CUTLASS_DEVICE Sm100M3DecodeKvMetadata
m3_decode_sparse_kv_metadata(Sm100M3DecodeKernelParams<Traits> const &params, int batch_idx,
                             int kv_head_idx, int q_token_idx) {
  Sm100M3DecodeSparseSelection const selection =
      m3_decode_sparse_selection<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
  int const current_valid_tokens = ((selection.visible_kv_len - 1) % Traits::kPageSize) + 1;
  return Sm100M3DecodeKvMetadata{
      (Traits::kSparseTopK - 1) * Traits::kPageSize + current_valid_tokens, Traits::kSparseTopK};
}

template <class Traits>
CUTLASS_DEVICE int m3_decode_sparse_kv_length(Sm100M3DecodeKernelParams<Traits> const &params,
                                              int batch_idx, int kv_head_idx, int q_token_idx) {
  return m3_decode_sparse_kv_metadata<Traits>(params, batch_idx, kv_head_idx, q_token_idx)
      .kv_length;
}

template <class Traits>
CUTLASS_DEVICE int m3_decode_sparse_kv_tile_count(Sm100M3DecodeKernelParams<Traits> const &params,
                                                  int batch_idx, int kv_head_idx, int q_token_idx) {
  return m3_decode_sparse_kv_metadata<Traits>(params, batch_idx, kv_head_idx, q_token_idx)
      .tile_count;
}

template <class Traits>
CUTLASS_DEVICE Sm100M3DecodeKvMetadata
m3_decode_kv_metadata_for_batch(Sm100M3DecodeKernelParams<Traits> const &params, int batch_idx,
                                int kv_head_idx, int q_token_idx) {
  return m3_decode_sparse_kv_metadata<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
}

template <class Traits>
CUTLASS_DEVICE int m3_decode_kv_length_for_batch(Sm100M3DecodeKernelParams<Traits> const &params,
                                                 int batch_idx, int kv_head_idx, int q_token_idx) {
  return m3_decode_sparse_kv_length<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
}

template <class Traits>
CUTLASS_DEVICE int
m3_decode_kv_tile_count_for_batch(Sm100M3DecodeKernelParams<Traits> const &params, int batch_idx,
                                  int kv_head_idx, int q_token_idx) {
  return m3_decode_sparse_kv_tile_count<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
}

} // namespace cutlass::m3_decode::collective
