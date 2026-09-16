#include "indexer_topk_kernel.hpp"

// The M3 headers below are copied verbatim from PR #23 commit
// 0dd596a32d458ed068464ea03ef2a992f5d197fb.
#include "m3/m3_topk.cuh"

namespace minimax::msa_v1::indexer::topk {

cudaError_t launch_indexer_topk(float const* scores, int32_t const* lengths,
                                int32_t* output, int max_cols,
                                int row_stride, int num_rows,
                                cudaStream_t stream) {
  m3::m3_launch(scores, lengths, output, max_cols, row_stride, num_rows,
                stream);
  return cudaGetLastError();
}

}  // namespace minimax::msa_v1::indexer::topk
