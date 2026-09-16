#pragma once

#include "sm100/common/prefill_attention_config.cuh"

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "cute/arch/copy_sm100.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

namespace minimax::msa_v1::attention::prefill::q8kv4 {
namespace detail {

using namespace cute;

constexpr float kLog2Fp8ProbabilityScale = 8.807354922057604f;

CUTLASS_DEVICE int decode_q_idx(int packed_qsplit) { return packed_qsplit & 0x00ffffff; }

CUTLASS_DEVICE int decode_split_idx(int packed_qsplit) {
  return static_cast<unsigned>(packed_qsplit) >> 24;
}

CUTLASS_DEVICE int real_col_to_stg128_half_fake_col(int col) {
  int const tile = col / 32;
  int const col32 = col % 32;
  int const lane = (col32 % 8) / 2;
  int const group = col32 / 8;
  int const element = col32 % 2;
  return tile * 32 + lane * 8 + group * 2 + element;
}

CUTLASS_DEVICE float rcp_approx_ftz(float value) {
  float result;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

CUTLASS_DEVICE float exp2_approx_ftz(float value) {
  float result;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

CUTLASS_DEVICE float log2_approx_ftz(float value) {
  float result;
  asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

CUTLASS_DEVICE float2 add_packed_f32x2(float2 lhs, float2 rhs) {
  float2 result;
  asm volatile("add.rn.ftz.f32x2 %0, %1, %2;"
               : "=l"(reinterpret_cast<uint64_t &>(result))
               : "l"(reinterpret_cast<uint64_t const &>(lhs)),
                 "l"(reinterpret_cast<uint64_t const &>(rhs)));
  return result;
}

CUTLASS_DEVICE float2 mul_packed_f32x2(float2 lhs, float2 rhs) {
  float2 result;
  asm volatile("mul.rn.ftz.f32x2 %0, %1, %2;"
               : "=l"(reinterpret_cast<uint64_t &>(result))
               : "l"(reinterpret_cast<uint64_t const &>(lhs)),
                 "l"(reinterpret_cast<uint64_t const &>(rhs)));
  return result;
}

CUTLASS_DEVICE float2 fma_packed_f32x2(float2 lhs, float2 rhs, float2 addend) {
  float2 result;
  asm volatile("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
               : "=l"(reinterpret_cast<uint64_t &>(result))
               : "l"(reinterpret_cast<uint64_t const &>(lhs)),
                 "l"(reinterpret_cast<uint64_t const &>(rhs)),
                 "l"(reinterpret_cast<uint64_t const &>(addend)));
  return result;
}

template <class Tensor> CUTLASS_DEVICE auto make_16x256b_tensor_mn_view(Tensor tensor) {
  auto acc_layout = layout(tensor);
  auto acc_layout_col_major = make_layout(shape(acc_layout));
  auto acc_layout_sm90 = make_layout(
      make_shape(get<0, 0>(shape(acc_layout_col_major)), get<0, 1>(shape(acc_layout_col_major)),
                 get<1>(shape(acc_layout_col_major)), get<2>(shape(acc_layout_col_major))),
      make_stride(get<0, 0>(stride(acc_layout_col_major)), get<0, 1>(stride(acc_layout_col_major)),
                  get<1>(stride(acc_layout_col_major)), get<2>(stride(acc_layout_col_major))));
  auto converted_sm90 = composition(acc_layout, acc_layout_sm90);

  auto converted_col_major = make_layout(shape(converted_sm90));
  auto converted_mn = make_layout(
      make_shape(
          make_shape(get<0, 1>(shape(converted_col_major)), get<1>(shape(converted_col_major))),
          make_shape(get<0, 0>(shape(converted_col_major)), get<0, 2>(shape(converted_col_major)),
                     get<2>(shape(converted_col_major))),
          get<3>(shape(converted_col_major))),
      make_stride(
          make_stride(get<0, 1>(stride(converted_col_major)), get<1>(stride(converted_col_major))),
          make_stride(get<0, 0>(stride(converted_col_major)),
                      get<0, 2>(stride(converted_col_major)), get<2>(stride(converted_col_major))),
          get<3>(stride(converted_col_major))));
  return make_tensor(tensor.data(), composition(converted_sm90, converted_mn));
}

CUTLASS_DEVICE void store_bf16x8_cs(cutlass::bfloat16_t *destination, float value0, float value1,
                                    float value2, float value3, float value4, float value5,
                                    float value6, float value7) {
  asm volatile("{\n"
               ".reg .b16 h0, h1, h2, h3, h4, h5, h6, h7;\n"
               ".reg .b32 p0, p1, p2, p3;\n"
               "cvt.rn.bf16.f32 h0, %1;\n"
               "cvt.rn.bf16.f32 h1, %2;\n"
               "cvt.rn.bf16.f32 h2, %3;\n"
               "cvt.rn.bf16.f32 h3, %4;\n"
               "cvt.rn.bf16.f32 h4, %5;\n"
               "cvt.rn.bf16.f32 h5, %6;\n"
               "cvt.rn.bf16.f32 h6, %7;\n"
               "cvt.rn.bf16.f32 h7, %8;\n"
               "mov.b32 p0, {h0, h1};\n"
               "mov.b32 p1, {h2, h3};\n"
               "mov.b32 p2, {h4, h5};\n"
               "mov.b32 p3, {h6, h7};\n"
               "st.global.cs.v4.b32 [%0], {p0, p1, p2, p3};\n"
               "}\n"
               :
               : "l"(destination), "f"(value0), "f"(value1), "f"(value2), "f"(value3), "f"(value4),
                 "f"(value5), "f"(value6), "f"(value7)
               : "memory");
}

template <int Offset, class Tensor, size_t... Indices>
CUTLASS_DEVICE void tmem_store_16_impl(Tensor const &source, uint32_t destination,
                                       cute::index_sequence<Indices...>) {
  SM100_TMEM_STORE_32dp32b16x::copy(source(Offset + Indices)..., destination);
}

template <int Offset, class Tensor>
CUTLASS_DEVICE void tmem_store_16(Tensor const &source, uint32_t destination) {
  tmem_store_16_impl<Offset>(source, destination, cute::make_index_sequence<16>{});
}

template <class TiledMma, class ScoreTensor>
CUTLASS_DEVICE void issue_qk(TiledMma tiled_mma, ScoreTensor tS, auto const &tQ, auto const &tK,
                             uint64_t *done_barrier) {
  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
  CUTLASS_PRAGMA_UNROLL
  for (int k_block = 0; k_block < size<2>(tQ); ++k_block) {
    gemm(tiled_mma, tQ(_, _, k_block), tK(_, _, k_block), tS);
    tiled_mma.accumulate_ = UMMA::ScaleOut::One;
  }
  cutlass::arch::umma_arrive(done_barrier);
}

template <class TiledMmaQK>
CUTLASS_DEVICE void softmax_to_tmem(TiledMmaQK tiled_mma_qk, uint32_t score_address,
                                    uint32_t probability_address, float softmax_scale_log2,
                                    int visible_cols, uint64_t *p_early_full, uint64_t *p_full,
                                    float &row_sum, float &row_max) {
  Tensor tStS = partition_fragment_C(tiled_mma_qk, make_shape(Int<128>{}, Int<128>{}));
  tStS.data() = score_address;
  Tensor cS = make_identity_tensor(make_shape(Int<128>{}, Int<128>{}));
  Tensor tScS = tiled_mma_qk.get_slice(0).partition_C(cS);
  auto tiled_tmem_load = make_tmem_copy(SM100_TMEM_LOAD_32dp32b32x{}, tStS);
  int const group_thread = static_cast<int>(threadIdx.x) & 127;
  auto thr_tmem_load = tiled_tmem_load.get_slice(group_thread);
  Tensor tStS_t2r = thr_tmem_load.partition_S(tStS);
  Tensor tScS_t2r = thr_tmem_load.partition_D(tScS);
  Tensor rS = make_tensor<Accumulator>(shape(tScS_t2r));
  static_assert(size(rS) == 128);
  copy(tiled_tmem_load, tStS_t2r, rS);
  cutlass::arch::fence_view_async_tmem_load();

  float maximum0 = -INFINITY;
  float maximum1 = -INFINITY;
  float maximum2 = -INFINITY;
  float maximum3 = -INFINITY;
  CUTLASS_PRAGMA_UNROLL
  for (int col = 0; col < 128; col += 8) {
    CUTLASS_PRAGMA_UNROLL
    for (int value = 0; value < 8; ++value) {
      rS(col + value) = col + value < visible_cols ? rS(col + value) : -INFINITY;
    }
    maximum0 = fmaxf(maximum0, fmaxf(rS(col), rS(col + 1)));
    maximum1 = fmaxf(maximum1, fmaxf(rS(col + 2), rS(col + 3)));
    maximum2 = fmaxf(maximum2, fmaxf(rS(col + 4), rS(col + 5)));
    maximum3 = fmaxf(maximum3, fmaxf(rS(col + 6), rS(col + 7)));
  }
  float const maximum = fmaxf(fmaxf(maximum0, maximum1), fmaxf(maximum2, maximum3));

  float const maximum_safe = maximum == -INFINITY ? 0.0f : maximum;
  float2 const packed_scale = make_float2(softmax_scale_log2, softmax_scale_log2);
  float2 const packed_bias =
      fma_packed_f32x2(make_float2(-maximum_safe, -maximum_safe), packed_scale,
                       make_float2(kLog2Fp8ProbabilityScale, kLog2Fp8ProbabilityScale));
  float2 sum0 = make_float2(0.0f, 0.0f);
  float2 sum1 = make_float2(0.0f, 0.0f);
  float2 sum2 = make_float2(0.0f, 0.0f);
  float2 sum3 = make_float2(0.0f, 0.0f);
  Tensor rP = make_tensor<uint32_t>(make_shape(Int<32>{}));
  cutlass::NumericArrayConverter<Element, float, 4> convert;
  CUTLASS_PRAGMA_UNROLL
  for (int chunk = 0; chunk < 16; ++chunk) {
    int const col = chunk * 8;
    float2 probability =
        fma_packed_f32x2(make_float2(rS(col), rS(col + 1)), packed_scale, packed_bias);
    probability.x = exp2_approx_ftz(probability.x);
    probability.y = exp2_approx_ftz(probability.y);
    rS(col) = probability.x;
    rS(col + 1) = probability.y;
    sum0 = add_packed_f32x2(sum0, probability);

    probability =
        fma_packed_f32x2(make_float2(rS(col + 2), rS(col + 3)), packed_scale, packed_bias);
    probability.x = exp2_approx_ftz(probability.x);
    probability.y = exp2_approx_ftz(probability.y);
    rS(col + 2) = probability.x;
    rS(col + 3) = probability.y;
    sum1 = add_packed_f32x2(sum1, probability);

    probability =
        fma_packed_f32x2(make_float2(rS(col + 4), rS(col + 5)), packed_scale, packed_bias);
    probability.x = exp2_approx_ftz(probability.x);
    probability.y = exp2_approx_ftz(probability.y);
    rS(col + 4) = probability.x;
    rS(col + 5) = probability.y;
    sum2 = add_packed_f32x2(sum2, probability);

    probability =
        fma_packed_f32x2(make_float2(rS(col + 6), rS(col + 7)), packed_scale, packed_bias);
    probability.x = exp2_approx_ftz(probability.x);
    probability.y = exp2_approx_ftz(probability.y);
    rS(col + 6) = probability.x;
    rS(col + 7) = probability.y;
    sum3 = add_packed_f32x2(sum3, probability);

    cutlass::Array<float, 4> input0;
    input0[0] = rS(col);
    input0[1] = rS(col + 1);
    input0[2] = rS(col + 2);
    input0[3] = rS(col + 3);
    cutlass::Array<float, 4> input1;
    input1[0] = rS(col + 4);
    input1[1] = rS(col + 5);
    input1[2] = rS(col + 6);
    input1[3] = rS(col + 7);
    cutlass::Array<Element, 4> const packed0 = convert(input0);
    cutlass::Array<Element, 4> const packed1 = convert(input1);
    rP(chunk * 2) = reinterpret_cast<uint32_t const &>(packed0);
    rP(chunk * 2 + 1) = reinterpret_cast<uint32_t const &>(packed1);
  }

  sum0 = add_packed_f32x2(sum0, sum1);
  sum2 = add_packed_f32x2(sum2, sum3);
  sum0 = add_packed_f32x2(sum0, sum2);
  float const sum = sum0.x + sum0.y;

  tmem_store_16<0>(rP, probability_address);
  cutlass::arch::fence_view_async_tmem_store();
  cutlass::arch::ClusterBarrier::arrive(p_early_full);

  tmem_store_16<16>(rP, probability_address + 16);
  cutlass::arch::fence_view_async_tmem_store();
  cutlass::arch::ClusterBarrier::arrive(p_full);

  row_sum = sum;
  row_max = maximum;
}

template <int ColumnPass, class TiledMmaPV, class OutputTensor, class CoordinateTensor>
CUTLASS_DEVICE void
store_partial_output_pass(TiledMmaPV tiled_mma_pv, OutputTensor tO, CoordinateTensor cO,
                          cutlass::bfloat16_t *o_partial_ptr, int total_q, int num_q_heads,
                          int head_kv, int q_count, int q_batch_offset, int group, int group_thread,
                          int output_token_base, int output_qsplit0, int output_qsplit1,
                          float const *row_sum) {
  auto thr_mma = tiled_mma_pv.get_slice(0);
  Tensor tOcO = thr_mma.partition_C(cO);
  constexpr int kColumnPassSize = 64;
  Tensor tOcOPass =
      logical_divide(tOcO, make_layout(make_shape(Int<128>{}, Int<kColumnPassSize>{})));

  Tensor tOPass = tO;
  tOPass.data() = tO.data() + ColumnPass * kColumnPassSize;
  Tensor tOPassDivided =
      logical_divide(tOPass, make_layout(make_shape(Int<128>{}, Int<kColumnPassSize>{})));
  auto tiled_tmem_load =
      make_tmem_copy(SM100_TMEM_LOAD_16dp256b8x{}, tOPassDivided(make_coord(_, _), _0{}));
  auto thread_tmem_load = tiled_tmem_load.get_slice(group_thread);
  Tensor tTMEMtOAll = thread_tmem_load.partition_S(tOPassDivided(make_coord(_, _), _));
  Tensor tTMEMcOAll = thread_tmem_load.partition_D(tOcOPass(make_coord(_, _), _));
  Tensor tTMEMtOPass = tTMEMtOAll(_, _, _, _0{});
  Tensor tTMEMcOPass = tTMEMcOAll(_, _, _, _0{});
  Tensor rOPass = make_tensor<Accumulator>(shape(tTMEMcOPass));
  copy(tiled_tmem_load, tTMEMtOPass, rOPass);
  cutlass::arch::fence_view_async_tmem_load();

  Tensor rOMnFull = make_16x256b_tensor_mn_view(rOPass);
  Tensor cOMnFull = make_16x256b_tensor_mn_view(tTMEMcOPass);
  Tensor rOMn = rOMnFull(_, _, _0{});
  Tensor cOMn = cOMnFull(_, _, _0{});
  static_assert(rank(rOMn) == 2);
  static_assert(size<0>(rOMn) == 2);
  static_assert(size<1>(rOMn) == 32);
  static_assert(rank(cOMn) == 2);
  static_assert(size<0>(cOMn) == 2);
  static_assert(size<1>(cOMn) == 32);

  CUTLASS_PRAGMA_UNROLL
  for (int row_index = 0; row_index < size<0>(rOMn); ++row_index) {
    CUTLASS_PRAGMA_UNROLL
    for (int column_group = 0; column_group < size<1>(rOMn) / 8; ++column_group) {
      int const column_base = column_group * 8;
      int const row = get<0>(cOMn(row_index, column_base));
      int const token = row / kQHeadsPerKv;
      int const qi = group * kQueriesPerGroup + token;
      if (qi < q_count) {
        int const packed_qsplit = token == output_token_base ? output_qsplit0 : output_qsplit1;
        int const q_idx = decode_q_idx(packed_qsplit);
        int const split = decode_split_idx(packed_qsplit);
        int const q_abs = q_batch_offset + q_idx;
        int const head = head_kv * kQHeadsPerKv + row % kQHeadsPerKv;
        float const sum = row_sum[row];
        float const scale = sum > 0.0f ? rcp_approx_ftz(sum) : 0.0f;
        float2 const packed_scale = make_float2(scale, scale);
        float2 const output0 = mul_packed_f32x2(
            make_float2(rOMn(row_index, column_base), rOMn(row_index, column_base + 1)),
            packed_scale);
        float2 const output1 = mul_packed_f32x2(
            make_float2(rOMn(row_index, column_base + 2), rOMn(row_index, column_base + 3)),
            packed_scale);
        float2 const output2 = mul_packed_f32x2(
            make_float2(rOMn(row_index, column_base + 4), rOMn(row_index, column_base + 5)),
            packed_scale);
        float2 const output3 = mul_packed_f32x2(
            make_float2(rOMn(row_index, column_base + 6), rOMn(row_index, column_base + 7)),
            packed_scale);
        int const col = get<1>(cOMn(row_index, column_base)) + ColumnPass * kColumnPassSize;
        int const fake_col = real_col_to_stg128_half_fake_col(col);
        int64_t const output_offset =
            ((static_cast<int64_t>(split) * total_q + q_abs) * num_q_heads + head) * kHeadDim +
            fake_col;
        store_bf16x8_cs(o_partial_ptr + output_offset, output0.x, output0.y, output1.x, output1.y,
                        output2.x, output2.y, output3.x, output3.y);
      }
    }
  }

  cutlass::arch::fence_view_async_tmem_load();
}

template <class TiledMmaPV, class OutputTensor, class CoordinateTensor>
CUTLASS_DEVICE void
store_partial_output(TiledMmaPV tiled_mma_pv, OutputTensor tO, CoordinateTensor cO,
                     cutlass::bfloat16_t *o_partial_ptr, float *lse_partial_ptr, int total_q,
                     int num_q_heads, float softmax_scale_log2, int head_kv, int q_count,
                     int q_batch_offset, int group, int group_thread, int const *qsplit_indices,
                     float const *row_sum, float const *row_max) {
  int const lane = group_thread & 31;
  int const output_token_base = (group_thread / 32) * 2;
  int output_qsplit0 = 0;
  int output_qsplit1 = 0;
  if (lane == 0) {
    output_qsplit0 = qsplit_indices[output_token_base];
    output_qsplit1 = qsplit_indices[output_token_base + 1];
  }
  output_qsplit0 = __shfl_sync(0xffffffffu, output_qsplit0, 0);
  output_qsplit1 = __shfl_sync(0xffffffffu, output_qsplit1, 0);

  store_partial_output_pass<0>(tiled_mma_pv, tO, cO, o_partial_ptr, total_q, num_q_heads, head_kv,
                               q_count, q_batch_offset, group, group_thread, output_token_base,
                               output_qsplit0, output_qsplit1, row_sum);
  store_partial_output_pass<1>(tiled_mma_pv, tO, cO, o_partial_ptr, total_q, num_q_heads, head_kv,
                               q_count, q_batch_offset, group, group_thread, output_token_base,
                               output_qsplit0, output_qsplit1, row_sum);
  cutlass::arch::fence_view_async_tmem_load();

  int const row = group_thread;
  int const token = row / kQHeadsPerKv;
  int const qi = group * kQueriesPerGroup + token;
  if (qi < q_count) {
    int const packed_qsplit = token == output_token_base ? output_qsplit0 : output_qsplit1;
    int const q_idx = decode_q_idx(packed_qsplit);
    int const split = decode_split_idx(packed_qsplit);
    int const q_abs = q_batch_offset + q_idx;
    int const head = head_kv * kQHeadsPerKv + row % kQHeadsPerKv;
    float const sum = row_sum[row];
    float lse = -INFINITY;
    if (sum > 0.0f && isfinite(row_max[row])) {
      constexpr float kLn2 = 0.6931471805599453f;
      lse = (row_max[row] * softmax_scale_log2 + log2_approx_ftz(sum) - kLog2Fp8ProbabilityScale) *
            kLn2;
    }
    int64_t const lse_offset = (static_cast<int64_t>(split) * total_q + q_abs) * num_q_heads + head;
    lse_partial_ptr[lse_offset] = lse;
  }
}

} // namespace detail
} // namespace minimax::msa_v1::attention::prefill::q8kv4
