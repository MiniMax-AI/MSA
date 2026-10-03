#include "indexer_topk_kernel.hpp"

// The M3 headers originate from PR #23 commit
// 0dd596a32d458ed068464ea03ef2a992f5d197fb; dependency waits are added for PDL.
#include "m3/m3_topk.cuh"

namespace minimax::msa_v1::indexer::topk {

cudaError_t launch_indexer_topk(float const *scores, int32_t const *lengths, int32_t *output,
                                int max_cols, int row_stride, int num_rows, cudaStream_t stream,
                                bool compact_grid, bool enable_pdl) {
  if (compact_grid) {
    // Match the existing kernel's row mapping; only surplus CTAs are removed.
    int const rows_per_block = m3::m3_rows_per_block(max_cols);
    int const blocks = num_rows / rows_per_block + (num_rows % rows_per_block != 0);
    constexpr int kWarpThreads = 32;
    // The warp family owns one row per warp; the block family uses the full CTA.
    int const threads = rows_per_block > 1 ? rows_per_block * kWarpThreads : m3::kThreads;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = enable_pdl;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(blocks);
    config.blockDim = dim3(threads);
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    cudaError_t status = cudaLaunchKernelEx(&config, m3::m3_topk_kernel, scores, lengths, output,
                                            max_cols, row_stride, num_rows, enable_pdl);
    if (status != cudaSuccess)
      return status;
  } else {
    m3::m3_launch(scores, lengths, output, max_cols, row_stride, num_rows, stream);
  }
  return cudaGetLastError();
}

} // namespace minimax::msa_v1::indexer::topk
