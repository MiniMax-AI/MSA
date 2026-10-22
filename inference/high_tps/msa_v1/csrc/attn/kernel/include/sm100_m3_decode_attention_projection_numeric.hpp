/***************************************************************************************************
 * Copyright (c) 2026, MegaGEMM contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Device helpers for the SM100 MiniMax M3 TP4 QKV/QIKI M128N8 kernel.
 *
 * This file deliberately keeps the PTX spellings used by the CuTe DSL implementation.  In
 * particular, the conversion routines below must not be compiled under --use_fast_math or replaced
 * with algebraically equivalent arithmetic: their bit-level results are part of the kernel's
 * workspace and output contract.
 **************************************************************************************************/

#pragma once

#include <cstdint>

#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/util.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/config.hpp>
#include <cute/tensor.hpp>

#include "common/kernel/include/sm100_m3_decode_storage.hpp"

namespace cutlass::m3_decode::collective::attention_projection::sm100 {

struct F32x2 {
  float x;
  float y;
};

struct F32x4 {
  float x;
  float y;
  float z;
  float w;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// M128N8, CTA-K128, MXFP8 scale-factor layouts
////////////////////////////////////////////////////////////////////////////////////////////////////

/// One shared-memory stage of the narrow-N SFB layout consumed by
/// SM100_UTCCP_4x32dp128bit_1cta.
///
/// The copy instruction addresses 128 N datapaths even though the MMA has N=8.
/// The physical tile is therefore padded to N=128, and the basic 128-datapath x
/// 4-scale chunks are stacked in N before K.
/// This expression is the CuTe C++ equivalent of make_small_n_sfb_smem_layout in the DSL helper.
template <int SfVecSize = 32> CUTE_HOST_DEVICE constexpr auto make_small_n_sfb_smem_stage_layout() {
  static_assert(SfVecSize == 32, "the M128N8 MXFP8 path requires a K32 scale vector");
  using SfAtom = typename cutlass::detail::Sm1xxBlockScaledBasicChunk<SfVecSize>::SfAtom;
  auto padded_nk = cute::tile_to_shape(SfAtom{}, cute::make_shape(cute::_128{}, cute::_128{}),
                                       cute::Step<cute::_1, cute::_2>{});

  // One N instruction and four K32 instructions in the CTA-K128 tile.
  auto per_instruction =
      cute::tiled_divide(padded_nk, cute::make_shape(cute::_128{}, cute::Int<SfVecSize>{}));
  return cute::logical_divide(
      per_instruction, cute::make_shape(cute::make_shape(cute::_128{}, cute::Int<SfVecSize>{})));
}

/// Staged narrow-N SFB shared-memory layout.  Each stage occupies exactly 512 bytes for UE8M0.
template <int NumStages, int SfVecSize = 32>
CUTE_HOST_DEVICE constexpr auto make_small_n_sfb_smem_layout() {
  static_assert(NumStages > 0, "SFB pipeline must have at least one stage");
  auto stage = make_small_n_sfb_smem_stage_layout<SfVecSize>();
  auto stage_mode =
      cute::make_layout(cute::Int<NumStages>{}, cute::cosize(cute::filter_zeros(stage)));
  return cute::append(stage, stage_mode);
}

/// Stock SFA stage for the same M128N8K128 tiled MMA.  Exposed next to the custom SFB layout so
/// callers can create matching UTCCP/TMEM tensors without invoking the high-level GEMM builder.
template <class TiledMma, int SfVecSize = 32>
CUTE_HOST_DEVICE constexpr auto make_m128n8_sfa_smem_stage_layout() {
  static_assert(SfVecSize == 32, "the M128N8 MXFP8 path requires a K32 scale vector");
  return cutlass::detail::Sm1xxBlockScaledConfig<SfVecSize>::deduce_smem_layoutSFA(
      TiledMma{}, cute::make_shape(cute::_128{}, cute::_8{}, cute::_128{}));
}

/// Construct the physical SFA TMEM tensor at tmem_address.
template <class TiledMma, int SfVecSize = 32>
CUTE_DEVICE auto make_m128n8_sfa_tmem_tensor(uint32_t tmem_address) {
  auto smem_stage = make_m128n8_sfa_smem_stage_layout<TiledMma, SfVecSize>();
  auto tensor = cute::make_tensor<typename TiledMma::FrgTypeSFA>(cute::shape(smem_stage));
  tensor.data() = tmem_address;
  return tensor;
}

/// Construct the physical narrow-N SFB TMEM tensor at tmem_address.
template <class TiledMma, int SfVecSize = 32>
CUTE_DEVICE auto make_m128n8_sfb_tmem_tensor(uint32_t tmem_address) {
  auto smem_stage = make_small_n_sfb_smem_stage_layout<SfVecSize>();
  auto tensor = cute::make_tensor<typename TiledMma::FrgTypeSFB>(cute::shape(smem_stage));
  tensor.data() = tmem_address;
  return tensor;
}

/// Rebuild a physical S2T scale tensor as one scale view per K32 tcgen05 MMA.
///
/// M128N8/CTA-K128/UE8M0 occupies one TMEM column per K128 scale stage.  The outer (4,1) mode
/// therefore enumerates the four K32 MMA instructions with strides (1,16), exactly matching the DSL
/// make_sf_tmem_mma_view specialization.
template <class TmemSfTensor>
CUTE_HOST_DEVICE constexpr auto make_sf_tmem_mma_view(TmemSfTensor const &physical_tensor) {
  auto const physical_layout = physical_tensor.layout();
  auto mma_layout =
      cute::make_layout(cute::make_shape(cute::get<0>(cute::shape(physical_layout)), cute::_1{},
                                         cute::make_shape(cute::_4{}, cute::_1{})),
                        cute::make_stride(cute::get<0>(cute::stride(physical_layout)), cute::_0{},
                                          cute::make_stride(cute::_1{}, cute::_16{})));
  return cute::make_tensor(physical_tensor.data(), mma_layout);
}

/// Make the one-CTA 4x32-datapath x 128-bit SMEM-to-TMEM copy for a compact SF TMEM tensor.
template <class CompactTmemSfTensor>
CUTE_HOST_DEVICE constexpr auto make_sf_s2t_copy(CompactTmemSfTensor const &compact_tmem_tensor) {
  return cute::make_utccp_copy(cute::SM100_UTCCP_4x32dp128bit_1cta{}, compact_tmem_tensor);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Exact MXFP8 / FP8 / FP4 conversions
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Convert a non-negative FP32 dynamic-MXFP8 scale to UE8M0 using round-up and SATFINITE.
///
/// This is bit-identical to the existing CuTe DSL helper for all FP32 bit patterns.  Code 255 is
/// never emitted.  Zero and negative values map to code 0.  The explicit subnormal half-way rule is
/// intentional and follows the frozen FlashInfer-derived implementation.
CUTE_DEVICE uint32_t float_to_ue8m0_rp_satfinite(float value) {
  uint32_t result;
  asm volatile("{\n\t"
               ".reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\n\t"
               ".reg .u32 bits, exp_biased, mantissa, bump, cvt_result;\n\t"
               "setp.le.f32 p_zero, %1, 0f00000000;\n\t"
               "mov.b32 bits, %1;\n\t"
               "shr.b32 exp_biased, bits, 23;\n\t"
               "and.b32 exp_biased, exp_biased, 255;\n\t"
               "and.b32 mantissa, bits, 0x7FFFFF;\n\t"
               "setp.ne.u32 p_has_mant, mantissa, 0;\n\t"
               "selp.u32 bump, 1, 0, p_has_mant;\n\t"
               "setp.eq.u32 p_exp_zero, exp_biased, 0;\n\t"
               "setp.le.u32 p_tiny_sub, mantissa, 0x400000;\n\t"
               "and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\n\t"
               "@p_tiny_sub mov.u32 bump, 0;\n\t"
               "add.u32 cvt_result, exp_biased, bump;\n\t"
               "setp.gt.u32 p_ovf, cvt_result, 254;\n\t"
               "selp.u32 cvt_result, 254, cvt_result, p_ovf;\n\t"
               "selp.u32 %0, 0, cvt_result, p_zero;\n\t"
               "}"
               : "=r"(result)
               : "f"(value));
  return result;
}

/// Construct the inverse MXFP8 scale 2**(127-code) directly from FP32 exponent bits.
/// Code 0 maps to +0.0f, matching the all-zero K32-block behavior in the DSL kernel.
CUTE_DEVICE float ue8m0_to_inv_scale(uint32_t ue8m0_code) {
  float result;
  asm volatile("{\n\t"
               ".reg .s32 new_exp;\n\t"
               ".reg .b32 float_bits;\n\t"
               ".reg .pred p_zero;\n\t"
               "setp.eq.u32 p_zero, %1, 0;\n\t"
               "sub.s32 new_exp, 254, %1;\n\t"
               "max.s32 new_exp, new_exp, 0;\n\t"
               "shl.b32 float_bits, new_exp, 23;\n\t"
               "mov.b32 %0, float_bits;\n\t"
               "@p_zero mov.b32 %0, 0;\n\t"
               "}"
               : "=f"(result)
               : "r"(ue8m0_code));
  return result;
}

/// Pack two FP32 values as E4M3, with a in the low byte and b in the high byte.
CUTE_DEVICE uint16_t fp32x2_to_fp8e4m3x2(float a, float b) {
  uint16_t result;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(result) : "f"(a), "f"(b));
  return result;
}

/// Unpack two E4M3 bytes through the hardware FP16-pair conversion path.
CUTE_DEVICE F32x2 fp8e4m3x2_to_fp32x2(uint16_t value) {
  F32x2 result;
  asm volatile("{\n\t"
               ".reg .b32 packed_f16;\n\t"
               ".reg .b16 lo, hi;\n\t"
               "cvt.rn.f16x2.e4m3x2 packed_f16, %2;\n\t"
               "mov.b32 {lo, hi}, packed_f16;\n\t"
               "cvt.f32.f16 %0, lo;\n\t"
               "cvt.f32.f16 %1, hi;\n\t"
               "}"
               : "=f"(result.x), "=f"(result.y)
               : "h"(value));
  return result;
}

/// Pack four FP32 values as four E4M3 bytes in x/y/z/w order from low to high byte.
CUTE_DEVICE uint32_t fp32x4_to_fp8x4(float a, float b, float c, float d) {
  uint32_t result;
  asm volatile("{\n\t"
               ".reg .b16 lo, hi;\n\t"
               "cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n\t"
               "cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n\t"
               "mov.b32 %0, {lo, hi};\n\t"
               "}"
               : "=r"(result)
               : "f"(a), "f"(b), "f"(c), "f"(d));
  return result;
}

/// Pack four FP32 values as four E2M1 nibbles in x/y/z/w order from low to high nibble.
CUTE_DEVICE uint16_t fp32x2_to_fp4e2m1x2(float a, float b) {
  uint16_t result;
  asm volatile("{\n\t"
               ".reg .b8 packed;\n\t"
               "cvt.rn.satfinite.e2m1x2.f32 packed, %2, %1;\n\t"
               "mov.b16 %0, {packed, 0};\n\t"
               "}"
               : "=h"(result)
               : "f"(a), "f"(b));
  return result;
}

CUTE_DEVICE uint16_t fp32x4_to_fp4x4(float a, float b, float c, float d) {
  uint16_t result;
  asm volatile("{\n\t"
               ".reg .b8 lo, hi;\n\t"
               "cvt.rn.satfinite.e2m1x2.f32 lo, %2, %1;\n\t"
               "cvt.rn.satfinite.e2m1x2.f32 hi, %4, %3;\n\t"
               "mov.b16 %0, {lo, hi};\n\t"
               "}"
               : "=h"(result)
               : "f"(a), "f"(b), "f"(c), "f"(d));
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Distributed shared memory helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Map a local shared-memory address into peer_rank's DSM address space.
CUTE_DEVICE uint32_t map_shared_rank(void const *shared_ptr, uint32_t peer_rank) {
  uint32_t const shared_addr = cute::cast_smem_ptr_to_uint(shared_ptr);
  uint32_t remote_addr;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(remote_addr)
               : "r"(shared_addr), "r"(peer_rank));
  return remote_addr;
}

/// Store 16 FP32 bytes into peer DSM and complete 16 transaction bytes on the peer mbarrier.
CUTE_DEVICE void store_shared_remote_fp32x4(float a, float b, float c, float d, float *destination,
                                            uint64_t *transaction_barrier, uint32_t peer_rank) {
  uint32_t const remote_destination = map_shared_rank(destination, peer_rank);
  uint32_t const remote_barrier = map_shared_rank(transaction_barrier, peer_rank);
  asm volatile("{\n\t"
               ".reg .v4 .f32 values;\n\t"
               "mov.f32 values.x, %2;\n\t"
               "mov.f32 values.y, %3;\n\t"
               "mov.f32 values.z, %4;\n\t"
               "mov.f32 values.w, %5;\n\t"
               "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 "
               "[%0], values, [%1];\n\t"
               "}"
               :
               : "r"(remote_destination), "r"(remote_barrier), "f"(a), "f"(b), "f"(c), "f"(d)
               : "memory");
}

CUTE_DEVICE void store_shared_remote_fp32x4(F32x4 values, float *destination,
                                            uint64_t *transaction_barrier, uint32_t peer_rank) {
  store_shared_remote_fp32x4(values.x, values.y, values.z, values.w, destination,
                             transaction_barrier, peer_rank);
}

/// Load one FP32 value from peer DSM using the cluster state space explicitly.
CUTE_DEVICE float load_shared_remote_f32(float const *source, uint32_t peer_rank) {
  uint32_t const remote_source = map_shared_rank(source, peer_rank);
  float result;
  asm volatile("ld.relaxed.cta.shared::cluster.f32 %0, [%1];"
               : "=f"(result)
               : "r"(remote_source)
               : "memory");
  return result;
}

/// Load one aligned FP32x4 vector from peer DSM using the cluster state space explicitly.
CUTE_DEVICE F32x4 load_shared_remote_fp32x4(float const *source, uint32_t peer_rank) {
  uint32_t const remote_source = map_shared_rank(source, peer_rank);
  F32x4 result;
  asm volatile("ld.relaxed.cta.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w)
               : "r"(remote_source)
               : "memory");
  return result;
}

/// Load one naturally aligned FP32x4 vector from local shared memory.
CUTE_DEVICE F32x4 load_shared_fp32x4(float const *source) {
  uint32_t const shared_source = cute::cast_smem_ptr_to_uint(source);
  F32x4 result;
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w)
               : "r"(shared_source)
               : "memory");
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Naturally aligned global/local vector access helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

CUTE_DEVICE void store_global_fp32x4(float *destination, float a, float b, float c, float d) {
  asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(destination), "f"(a), "f"(b), "f"(c), "f"(d)
               : "memory");
}

CUTE_DEVICE void store_global_fp32x4(float *destination, F32x4 values) {
  store_global_fp32x4(destination, values.x, values.y, values.z, values.w);
}

CUTE_DEVICE F32x4 load_global_fp32x4(float const *source) {
  F32x4 result;
  asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w)
               : "l"(source)
               : "memory");
  return result;
}

/// Copy one naturally aligned raw BF16x4 vector from global memory to shared memory.
/// uint16_t is used intentionally so the helper does not perform a numeric conversion.
CUTE_DEVICE void copy_global_to_shared_bf16x4(uint16_t const *source, uint16_t *destination) {
  uint32_t const shared_destination = cute::cast_smem_ptr_to_uint(destination);
  asm volatile("{\n\t"
               ".reg .b16 b0, b1, b2, b3;\n\t"
               "ld.global.v4.b16 {b0, b1, b2, b3}, [%0];\n\t"
               "st.shared.v4.b16 [%1], {b0, b1, b2, b3};\n\t"
               "}"
               :
               : "l"(source), "r"(shared_destination)
               : "memory");
}

/// Load one aligned raw BF16x4 vector from shared memory and convert each lane to FP32.
CUTE_DEVICE F32x4 load_shared_bf16x4_as_fp32(uint16_t const *source) {
  uint32_t const shared_source = cute::cast_smem_ptr_to_uint(source);
  F32x4 result;
  asm volatile("{\n\t"
               ".reg .b16 b0, b1, b2, b3;\n\t"
               "ld.shared.v4.b16 {b0, b1, b2, b3}, [%4];\n\t"
               "cvt.f32.bf16 %0, b0;\n\t"
               "cvt.f32.bf16 %1, b1;\n\t"
               "cvt.f32.bf16 %2, b2;\n\t"
               "cvt.f32.bf16 %3, b3;\n\t"
               "}"
               : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w)
               : "r"(shared_source)
               : "memory");
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Graph-stable global epochs
////////////////////////////////////////////////////////////////////////////////////////////////////

inline constexpr uint32_t kFinishSumTag = 0x80000000u;

/// Spin until bit 31 of `counter` differs from bit 31 of `old`, checking the launch's wait
/// deadline (cutlass::m3_decode::collective::m3_decode_check_deadline) once per poll.
CUTE_DEVICE void wait_phase_flip_from(uint32_t const *counter, uint32_t old) {
  while (true) {
    uint32_t cur;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(cur) : "l"(counter) : "memory");
    if (((cur ^ old) & kFinishSumTag) != 0) {
      return;
    }
    m3_decode_check_deadline();
  }
}

CUTE_DEVICE uint32_t atomic_add_release(uint32_t *counter, uint32_t delta) {
  uint32_t old;
  asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
               : "=r"(old)
               : "l"(counter), "r"(delta)
               : "memory");
  return old;
}

/// Reusable phase-flip barrier for a cooperative GridBlockCount-CTA launch.
///
/// CTA zero contributes TAG-(GridBlockCount-1), while every other CTA contributes one.  The phase
/// expected by the leader is derived from the old value returned by its atomic,
/// so the same isolated cache line can be reused across CUDA Graph replays
/// without a memset.
template <int GridBlockCount = 64>
CUTE_DEVICE void reusable_grid_sync(uint32_t *counter, uint32_t block_linear, uint32_t tid) {
  static_assert(GridBlockCount > 1, "grid epoch requires at least two CTAs");
  __syncthreads();

  if (tid == 0) {
    constexpr uint32_t leader_delta = kFinishSumTag - uint32_t(GridBlockCount - 1);
    uint32_t const delta = block_linear == 0 ? leader_delta : 1u;
    wait_phase_flip_from(counter, atomic_add_release(counter, delta));
  }

  __syncthreads();
}

/// Reusable phase-flip barrier for an active subset of each participating CTA.
template <int GridBlockCount, int ActiveThreads, int BarrierId>
CUTE_DEVICE void reusable_grid_sync_active(uint32_t *counter, uint32_t block_linear, uint32_t tid) {
  static_assert(GridBlockCount > 1, "grid epoch requires at least two CTAs");
  cutlass::arch::NamedBarrier(ActiveThreads, BarrierId).arrive_and_wait();

  if (tid == 0) {
    constexpr uint32_t leader_delta = kFinishSumTag - uint32_t(GridBlockCount - 1);
    uint32_t const delta = block_linear == 0 ? leader_delta : 1u;
    wait_phase_flip_from(counter, atomic_add_release(counter, delta));
  }

  cutlass::arch::NamedBarrier(ActiveThreads, BarrierId).arrive_and_wait();
}

/// Publish WriterCount global partials, then acquire them in split zero.
///
/// Every generation adds exactly bit 31 to the cache-line-isolated counter.  Only split zero spins;
/// the other writers return after the release atomic.  As with
/// reusable_grid_sync, the expected phase is derived from the atomic's returned
/// old value and is therefore stable across graph replay.
template <int WriterCount = 4>
CUTE_DEVICE void special_epoch_release_acquire(uint32_t *epoch, uint32_t split_idx, uint32_t tid) {
  static_assert(WriterCount > 1, "special epoch requires at least two writers");
  if (tid == 0) {
    constexpr uint32_t owner_delta = kFinishSumTag - uint32_t(WriterCount - 1);
    uint32_t const delta = split_idx == 0 ? owner_delta : 1u;
    uint32_t const old = atomic_add_release(epoch, delta);
    if (split_idx == 0) {
      wait_phase_flip_from(epoch, old);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Generic/async-proxy and TMEM visibility fences
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Make generic-proxy global writes visible to subsequent async-proxy operations (for example TMA).
CUTE_DEVICE void fence_proxy_async_global() {
  asm volatile("fence.proxy.async.global;" : : : "memory");
}

/// Make generic-proxy shared writes visible to subsequent CTA-scoped async-proxy operations.
CUTE_DEVICE void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
}

/// Publish mbarrier initialization before the matching CTA/cluster synchronization.
CUTE_DEVICE void fence_mbarrier_init() { cutlass::arch::fence_barrier_init(); }

/// Wait until prior tcgen05 loads from TMEM are complete and visible to the generic proxy.
CUTE_DEVICE void fence_view_async_tmem_load() { cutlass::arch::fence_view_async_tmem_load(); }

/// Wait until prior tcgen05 stores to TMEM are complete.
CUTE_DEVICE void fence_view_async_tmem_store() { cutlass::arch::fence_view_async_tmem_store(); }

} // namespace cutlass::m3_decode::collective::attention_projection::sm100
