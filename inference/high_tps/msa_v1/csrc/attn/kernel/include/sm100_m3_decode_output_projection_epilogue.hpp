#pragma once

#include "attn/kernel/include/sm100_m3_decode_output_projection_mainloop.hpp"
#include "attn/kernel/include/sm100_m3_decode_tp4_mailbox.hpp"

namespace cutlass::m3_decode::collective {

template <class Traits>
struct Sm100M3DecodeOutputProjectionEpilogue : Sm100M3DecodeOutputProjectionMainloop<Traits> {
  using Base = Sm100M3DecodeOutputProjectionMainloop<Traits>;
  using Base::bind_tmem_base;
  using Base::epilogue_full_barrier;
  using Base::kActivationLoadWarp;
  using Base::kEpilogueWarps;
  using Base::kEpilogueWarpStart;
  using Base::kHandoffPollWarp;
  using Base::kProblemM;
  using Base::kScaleLoadWarp;
  using Base::kTcWarps;
  using Base::kTcWarpStart;
  using Base::kTileM;
  using Base::kTileN;
  using Base::kWeightLoadWarp;
  using Base::load_activation_pipeline;
  using Base::load_weight_pipeline;
  using Base::make_scale_tensors;
  using Base::poll_activation_pipeline;
  using Base::run_scale_pipeline;
  using Base::run_tc_pipeline;
  using Base::smem_overlay;
  using typename Base::ElementAccumulator;
  using typename Base::ElementOutput;
  using typename Base::Params;
  using typename Base::ScaleTensorSet;
  using typename Base::Storage;
  using typename Base::TiledMma;
  CUTLASS_DEVICE static void store_accumulator(ScaleTensorSet &tensors, Params const &params,
                                               int m_tile, int thread_idx) {
    ElementOutput *out = static_cast<ElementOutput *>(params.tp_unicast_ptr);
    if constexpr (Traits::kKernelPath != Sm100M3DecodeKernelPath::kMegaAtten) {
      out = reinterpret_cast<ElementOutput *>(
          Sm100M3DecodeTp4AllReduce<Traits>::output_projection_unicast_payload(params, thread_idx));
    }
    auto gC = cute::make_tensor(
        cute::make_gmem_ptr(out + m_tile * kTileM),
        cute::make_layout(cute::make_shape(cute::Int<kTileM>{}, cute::Int<kTileN>{}),
                          cute::make_stride(cute::Int<1>{}, cute::Int<kProblemM>{})));
    auto tiled_mma = TiledMma{};
    auto thr_mma = tiled_mma.get_slice(0);
    auto tCgC = thr_mma.partition_C(gC);

    constexpr int kAccColBits = kTileN * cutlass::sizeof_bits<ElementAccumulator>::value;
    auto tmem_load = cute::TMEM::op_repeater<cute::SM100_TMEM_LOAD_16dp256b1x, kAccColBits>();
    auto tiled_copy = cute::make_tmem_copy(tmem_load, tensors.acc);
    auto thr_copy = tiled_copy.get_slice(thread_idx);
    auto tmem_src = thr_copy.partition_S(tensors.acc);
    auto gmem_dst = thr_copy.partition_D(tCgC);
    auto rAcc = cute::make_tensor<ElementAccumulator>(cute::shape(gmem_dst));
    auto rOut = cute::make_tensor<ElementOutput>(cute::shape(gmem_dst));

    cute::copy(tiled_copy, tmem_src, rAcc);
    cutlass::arch::fence_view_async_tmem_load();

    cutlass::NumericConverter<ElementOutput, ElementAccumulator> converter;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < cute::size(rOut); ++i) {
      rOut[i] = converter(rAcc[i]);
    }
    if constexpr (Traits::kKernelPath == Sm100M3DecodeKernelPath::kMegaAtten) {
      // Transpose the TMEM fragment into contiguous mailbox publication vectors.
      __shared__ __align__(16) ElementOutput mailbox_tile[kTileM * kTileN];
      auto coordinates =
          cute::make_identity_tensor(cute::make_shape(cute::Int<kTileM>{}, cute::Int<kTileN>{}));
      auto partition = thr_mma.partition_C(coordinates);
      auto owned = thr_copy.partition_D(partition);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < cute::size(rOut); ++i) {
        mailbox_tile[cute::get<1>(owned(i)) * kTileM + cute::get<0>(owned(i))] = rOut(i);
      }
      constexpr int kPublicationBarrier = 13;
      Sm100M3DecodeNamedBarrier::sync(128, kPublicationBarrier);
      int4 const value = reinterpret_cast<int4 const *>(mailbox_tile)[thread_idx];
      Sm100M3DecodeTp4Mailbox::publish_tile(params, m_tile, thread_idx, value);
    } else {
      cute::copy(rOut, gmem_dst);
    }
  }

  CUTLASS_DEVICE static void run_task(Storage &storage, Params const &params, int thread_idx,
                                      int warp_idx, int m_tile) {
    __syncthreads();
    auto overlay = smem_overlay(storage);

    if (warp_idx == kWeightLoadWarp) {
      load_weight_pipeline(storage, params, overlay, m_tile);
    } else if (warp_idx == kActivationLoadWarp) {
      load_activation_pipeline(storage, params, overlay);
    } else if (warp_idx == kScaleLoadWarp) {
      run_scale_pipeline(storage, overlay);
    } else if (warp_idx == kHandoffPollWarp) {
      poll_activation_pipeline(storage, params, m_tile);
    } else if (warp_idx >= kEpilogueWarpStart && warp_idx < kEpilogueWarpStart + kEpilogueWarps) {
      Sm100M3DecodeBarrier::wait(epilogue_full_barrier(storage), 0);
      auto tensors = make_scale_tensors(overlay, 0, 0);
      bind_tmem_base(tensors, storage.tmem_state_ptr()[0], 0);
      int const epilogue_thread = thread_idx - kEpilogueWarpStart * cutlass::NumThreadsPerWarp;
      store_accumulator(tensors, params, m_tile, epilogue_thread);
    } else if (warp_idx >= kTcWarpStart && warp_idx < kTcWarpStart + kTcWarps) {
      run_tc_pipeline(storage, overlay, warp_idx - kTcWarpStart);
    }
  }
};

} // namespace cutlass::m3_decode::collective
