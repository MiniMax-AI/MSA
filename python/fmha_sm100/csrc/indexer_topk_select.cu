// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

// Forced-tail TopK used by the Q8KV4/Q8KV8 paged indexers.

#include <limits>

#include "indexer_topk/m3_topk.cuh"
#include "tvm_ffi_utils.h"

namespace {

constexpr int64_t kMaximumColumns = m3::kMaxNvm;

template <typename T> T *tensor_data(TensorView tensor) {
  return reinterpret_cast<T *>(static_cast<char *>(tensor.data_ptr()) + tensor.byte_offset());
}

} // namespace

// ``lengths[row]`` counts the row's candidate pages including the forced local
// page, which is written to the final slot without being ranked.
void indexer_topk_select(TensorView scores, TensorView lengths, TensorView output,
                         int64_t stream_ptr) {
  CHECK_CUDA(scores);
  CHECK_DIM(2, scores);
  TVM_FFI_ICHECK(encode_dlpack_dtype(scores.dtype()) == float32_code) << "scores must be float32";
  TVM_FFI_ICHECK(scores.size(0) > 0 && scores.size(0) <= std::numeric_limits<int>::max())
      << "scores rows must be in [1, INT32_MAX]";
  TVM_FFI_ICHECK(scores.size(1) > 0 && scores.size(1) <= kMaximumColumns)
      << "scores columns must be in [1, " << kMaximumColumns << "]";
  TVM_FFI_ICHECK(scores.stride(1) == 1) << "scores must have stride(1) == 1";
  TVM_FFI_ICHECK(scores.stride(0) >= scores.size(1) &&
                 scores.stride(0) <= std::numeric_limits<int>::max())
      << "scores rows must not overlap and the row stride must fit in int32";

  CHECK_INPUT_AND_TYPE(lengths, dl_int32);
  CHECK_DIM(1, lengths);
  TVM_FFI_ICHECK(lengths.size(0) == scores.size(0)) << "lengths must have shape [rows]";
  CHECK_DEVICE(lengths, scores);

  CHECK_INPUT_AND_TYPE(output, dl_int32);
  CHECK_DIM(2, output);
  TVM_FFI_ICHECK(output.size(0) == scores.size(0) && output.size(1) == m3::kTopK)
      << "output must have shape [rows, " << m3::kTopK << "]";
  CHECK_DEVICE(output, scores);

  ffi::CUDADeviceGuard device_guard(scores.device().device_id);
  m3::m3_launch(tensor_data<float const>(scores), tensor_data<int const>(lengths),
                tensor_data<int>(output), static_cast<int>(scores.size(1)),
                static_cast<int>(scores.stride(0)), static_cast<int>(scores.size(0)),
                reinterpret_cast<cudaStream_t>(stream_ptr));
  cudaError_t const status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "indexer_topk_select launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(indexer_topk_select, indexer_topk_select);
