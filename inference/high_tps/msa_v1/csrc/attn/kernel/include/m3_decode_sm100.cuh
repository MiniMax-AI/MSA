#pragma once

#include "attn/kernel/include/sm100_m3_decode_mainloop.hpp"
#include "attn/kernel/include/sm100_m3_decode_tma_descriptors.hpp"
#include "common/kernel/include/m3_decode_sm100_params.hpp"
#include "common/kernel/include/sm100_m3_decode_megakernel.hpp"
#include "common/kernel/include/sm100_m3_decode_traits.hpp"
#include "cutlass/cutlass.h"
namespace mega_atten::m3_decode {
using namespace cutlass::m3_decode::collective;
struct M3DecodeMegaAttenRunner {
  using Traits = Sm100M3DecodeTraits<true, 0, false, Sm100M3DecodeKernelPath::kMegaAtten>;
  using Kernel = cutlass::m3_decode::kernel::Sm100M3DecodeMegaAttenKernel<Traits>;

  static cudaError_t run(M3DecodeMegaKernelParams const &params) {
    if (!Kernel::can_implement(params)) {
      return cudaErrorNotSupported;
    }
    typename Kernel::Params kernel_params;
    cudaError_t const status = Kernel::initialize_params(params, kernel_params);
    if (status != cudaSuccess) {
      return status;
    }
    return Kernel::run(kernel_params, params.stream);
  }
};

inline cudaError_t run_m3_decode_mega_atten(M3DecodeMegaKernelParams const &params) {
  return M3DecodeMegaAttenRunner::run(params);
}

} // namespace mega_atten::m3_decode
