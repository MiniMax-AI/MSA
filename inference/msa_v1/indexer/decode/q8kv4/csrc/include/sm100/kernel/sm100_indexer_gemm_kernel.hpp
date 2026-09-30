#pragma once

#include "sm100/collective/sm100_indexer_gemm_umma.hpp"

namespace minimax::msa_v1::indexer::decode::q8kv4 {

template <class Traits>
__global__ void __launch_bounds__(Traits::kThreads, 1)
    indexer_gemm_kernel(const __grid_constant__ IndexerGemmParams params) {
  using Collective = IndexerGemmUmmaCollective<Traits>;
  using SharedStorage = typename Collective::SharedStorage;
  extern __shared__ __align__(1024) uint8_t shared_memory[];
  SharedStorage &storage = *reinterpret_cast<SharedStorage *>(shared_memory);
  Collective::run(params, storage);
}

template <class Traits> struct IndexerGemmKernel {
  using Collective = IndexerGemmUmmaCollective<Traits>;
  using SharedStorage = typename Collective::SharedStorage;
  static constexpr int kThreadCount = Traits::kThreads;
  static constexpr int kSharedStorageBytes = sizeof(SharedStorage);
};

} // namespace minimax::msa_v1::indexer::decode::q8kv4
