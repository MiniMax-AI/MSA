#pragma once

#include "sm100_fmha_correction_tma_warpspecialized.hpp"

namespace cutlass::fmha::collective {

template <class Traits>
struct Sm100FmhaFwdEpilogueTmaWarpspecialized : Sm100FmhaCorrectionTmaWarpspecialized<Traits> {
  using FmhaTraits = Traits;

  struct TensorStorage {};

  struct Arguments {
    void *o_ptr = nullptr;
    void *o_direct_ptr = nullptr;
    int total_qo_len_orig = 0;
    int num_qo_heads_orig = 0;
    bool tma_direct_o_enabled = false;
  };

  struct Params {
    Arguments output;
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const &, Arguments const &args, void *) {
    return Params{args};
  }

  static void prefetch_tma_descriptors(Params const &) {}

  template <class... StoreArgs> CUTLASS_DEVICE void store(StoreArgs &&...) const {
    // Output is committed by the q8kv4 correction path in this specialization.
  }
};

} // namespace cutlass::fmha::collective
