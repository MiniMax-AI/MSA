#pragma once

#include "attn/kernel/include/sm100_m3_decode_output_projection_epilogue.hpp"

namespace cutlass::m3_decode::collective {

template <class Traits>
struct Sm100M3DecodeOutputProjection : Sm100M3DecodeOutputProjectionEpilogue<Traits> {
  using Base = Sm100M3DecodeOutputProjectionEpilogue<Traits>;
  using Base::is_output_projection_cta;
  using Base::output_projection_m_tile;
  using Base::run_task;
  using typename Base::Params;
  using typename Base::Storage;
  CUTLASS_DEVICE static void run(Storage &storage, Params const &params, int thread_idx,
                                 int warp_idx) {
    if (!is_output_projection_cta()) {
      __syncthreads();
      return;
    }
    run_task(storage, params, thread_idx, warp_idx, output_projection_m_tile());
  }
};

} // namespace cutlass::m3_decode::collective
