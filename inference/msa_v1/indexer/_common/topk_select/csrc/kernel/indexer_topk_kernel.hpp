#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace minimax::msa_v1::indexer::topk {

cudaError_t launch_indexer_topk(float const* scores, int32_t const* lengths,
                                int32_t* output, int max_cols,
                                int row_stride, int num_rows,
                                cudaStream_t stream);

}  // namespace minimax::msa_v1::indexer::topk
