#pragma once
#include <cuda_runtime.h>

#include "include/sm100/common/layout.hpp"

namespace mega_moe::token_parallel {
// Encode every tensor map of one plan; fails if any descriptor is invalid.
cudaError_t encode_maps(Args const &args, Maps *maps);
// Set the kernel's dynamic shared memory and report how many of its CTAs can be
// co-resident on the device (zero without cooperative launch support).
cudaError_t configure_kernel(int device, int *resident_ctas);
// Enqueue one cooperative launch of the complete MoE on the stream.
cudaError_t launch(KernelArgs const &arguments, cudaStream_t stream);
} // namespace mega_moe::token_parallel
