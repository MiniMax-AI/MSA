#include <cub/device/device_scan.cuh>

#include "sm100/common/indexer_gemm_arguments.hpp"
#include "sm100/common/indexer_gemm_traits.hpp"

namespace minimax::msa_v1::indexer::decode::tp4_q8kv4 {

cudaError_t get_indexer_gemm_scheduler_temp_storage_bytes(
    int batch, size_t& temp_storage_bytes) {
  temp_storage_bytes = 0;
  if (batch <= IndexerGemmTraits::kPrepareThreads) {
    return cudaSuccess;
  }
  return cub::DeviceScan::InclusiveSum(
      nullptr, temp_storage_bytes, static_cast<int32_t*>(nullptr),
      static_cast<int32_t*>(nullptr), batch);
}

cudaError_t run_indexer_gemm_scheduler_scan(
    int32_t* prefix, int batch, void* temp_storage,
    size_t temp_storage_bytes, cudaStream_t stream) {
  return cub::DeviceScan::InclusiveSum(
      temp_storage, temp_storage_bytes, prefix, prefix, batch, stream);
}

}  // namespace minimax::msa_v1::indexer::decode::tp4_q8kv4
