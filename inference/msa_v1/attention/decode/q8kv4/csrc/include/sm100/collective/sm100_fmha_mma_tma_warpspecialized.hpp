#pragma once

#include <climits>
#include <cstdint>
#include <type_traits>

#include "cute/arch/mma_sm100_desc.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/arch/util.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "fmha_common.hpp"
#include "sm100_fmha_pipeline.hpp"
#include "sm100_fmha_q8kv4_traits.hpp"
#include "sm100_fmha_storage.hpp"
#include "sm100_mma_n8.hpp"

namespace cutlass::fmha::collective {

template <class Traits> struct Sm100FmhaMmaTmaWarpspecialized {
  using Storage = SharedStorage<Traits>;
  using Params = Sm100FmhaFwdKernelParams<Traits>;
  using Tmem = TmemLayout<Traits>;
  using Barriers = BarrierLayout<Traits>;

  struct State {
    int q_event = 0;
    int transformed_event = 0;
    int s_event = 0;
    int s_acquire_event = 0;
    uint32_t o_empty_phase = 1;
  };

  using QkAtom = std::conditional_t<
      Traits::kHeadGroup == 8, Sm100MmaF8TsM128N8,
      cute::SM100_MMA_F8F6F4_TS<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                /*M=*/128,
                                /*N=*/16, cute::UMMA::Major::K, cute::UMMA::Major::K>>;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1070)
  // SM107: dense K=64 f8f6f4 MMA (instruction-descriptor k_size bit), two steps per tile.
  static constexpr int kMmaKSteps = 2;
  static constexpr uint32_t kTmemColsPerKStep = 16;
  static constexpr uint32_t kSmemDescUnitsPerKStep = 4; // 64 B in 16 B units
  CUTLASS_DEVICE static uint64_t qk_instr_desc() {
    return cute::UMMA::make_runtime_instr_desc<>(
        cute::UMMA::make_sm107_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                          /*M=*/128,
                                          /*N=*/Traits::kHeadGroup, cute::UMMA::Major::K,
                                          cute::UMMA::Major::K>());
  }
#else
  static constexpr int kMmaKSteps = 4;
  static constexpr uint32_t kTmemColsPerKStep = 8;
  static constexpr uint32_t kSmemDescUnitsPerKStep = 2; // 32 B in 16 B units
  CUTLASS_DEVICE static constexpr uint64_t qk_instr_desc() {
    return cute::UMMA::make_runtime_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                               /*M=*/128,
                                               /*N=*/Traits::kHeadGroup, cute::UMMA::Major::K,
                                               cute::UMMA::Major::K>();
  }
#endif

  CUTLASS_DEVICE static uint32_t s_stage_tmem_col(int stage) {
    return (stage & 1) == 0 ? Tmem::kS0 : Tmem::kS1;
  }

  CUTLASS_DEVICE static uint32_t o_stage_tmem_col(int) { return Tmem::kO0; }

  CUTLASS_DEVICE static uint32_t prepared_kv_tmem_col(uint32_t tmem_base, int transformed_stage) {
    return tmem_base + Tmem::kPreparedKv +
           static_cast<uint32_t>(transformed_stage * Tmem::kColsPerPreparedKvStage);
  }

  CUTLASS_DEVICE static uint64_t *q_full_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kQFullArv1 + stage);
  }

  CUTLASS_DEVICE static uint64_t *q_empty_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kQEmptyArv1 + stage);
  }

  CUTLASS_DEVICE static uint64_t *transformed_full_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kTransformedKvFullArv128 + stage);
  }

  CUTLASS_DEVICE static uint64_t *transformed_empty_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kTransformedKvEmptyArv1 + stage);
  }

  CUTLASS_DEVICE static uint64_t *s_full_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kS0FullArv1 + stage);
  }

  CUTLASS_DEVICE static uint64_t *s_empty_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kS0EmptyArv128 + stage);
  }

  CUTLASS_DEVICE static uint64_t *p_full_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kPFullArv128 + stage);
  }

  CUTLASS_DEVICE static uint64_t *p_empty_barrier(Storage &storage, int stage) {
    return storage.pipelines.ptr(Barriers::kPEmptyArv1 + stage);
  }

  CUTLASS_DEVICE static uint64_t *o_full_barrier(Storage &storage) {
    return storage.pipelines.ptr(Barriers::kOFullArv1);
  }

  CUTLASS_DEVICE static uint64_t *o_empty_barrier(Storage &storage) {
    return storage.pipelines.ptr(Barriers::kOEmptyArv128);
  }

  CUTLASS_DEVICE static int kv_length_for_batch(Params const &params, int batch_idx,
                                                int kv_head_idx, int q_token_idx) {
    return fmha_fwd_kv_length_for_batch<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
  }

  CUTLASS_DEVICE static int kv_tile_count(Params const &params, int batch_idx, int kv_head_idx,
                                          int q_token_idx) {
    return fmha_fwd_kv_tile_count_for_batch<Traits>(params, batch_idx, kv_head_idx, q_token_idx);
  }

  CUTLASS_DEVICE static uint32_t full_phase(int event, int stages) {
    return static_cast<uint32_t>((event / stages) & 1);
  }

  CUTLASS_DEVICE static uint32_t empty_phase(int event, int stages) {
    return static_cast<uint32_t>(1 ^ ((event / stages) & 1));
  }

  CUTLASS_DEVICE static void advance_two_stage_empty_state(int &stage, uint32_t &empty_phase) {
    uint32_t const phase_toggle = static_cast<uint32_t>(stage);
    stage ^= 1;
    empty_phase ^= phase_toggle;
  }

  CUTLASS_DEVICE static int s_stage_for_event(int event) { return event & 1; }

  CUTLASS_DEVICE static uint32_t s_empty_phase_for_event(int event) {
    return empty_phase(event, 2);
  }

  CUTLASS_DEVICE static void acquire_s_event(Storage &storage, State &state, int event) {
    CUTLASS_PRAGMA_NO_UNROLL
    while (state.s_acquire_event <= event) {
      acquire_s_stage(storage, s_stage_for_event(state.s_acquire_event),
                      s_empty_phase_for_event(state.s_acquire_event));
      ++state.s_acquire_event;
    }
  }

  CUTLASS_DEVICE static void commit_acquired_s_tail(Storage &storage, State &state) {
    CUTLASS_PRAGMA_NO_UNROLL
    while (state.s_event < state.s_acquire_event) {
      commit_s_stage(storage, s_stage_for_event(state.s_event));
      ++state.s_event;
    }
  }

  CUTLASS_DEVICE static uint64_t make_sw128_major_k_smem_desc(void const *smem_ptr,
                                                              int leading_bytes, int stride_bytes) {
    cute::UMMA::SmemDescriptor desc{};
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    smem_addr = __shfl_sync(0xffffffffu, smem_addr, 0);
    desc.version_ = 1;
    desc.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
#if defined(CUTLASS_ARCH_MMA_SM107A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM107F_ENABLED)
    // SM107: 15-bit start address (4 LSBs omitted) covers the 327 KB oversized smem window.
    desc.start_address_ = static_cast<uint16_t>((smem_addr & 0x7ffffu) >> 4);
#else
    desc.start_address_ = static_cast<uint16_t>((smem_addr & 0x3ffffu) >> 4);
#endif
    desc.leading_byte_offset_ = static_cast<uint16_t>(leading_bytes >> 4);
    desc.stride_byte_offset_ = static_cast<uint16_t>(stride_bytes >> 4);
    desc.base_offset_ = 0;
    desc.lbo_mode_ = 0;
    return desc;
  }

  CUTLASS_DEVICE static uint64_t make_q_smem_desc(uint8_t const *smem_q_stage) {
    return make_sw128_major_k_smem_desc(smem_q_stage,
                                        /*leading_bytes=*/Traits::kSmemQBytesPerStage,
                                        /*stride_bytes=*/8 * Traits::kHeadDim);
  }

  CUTLASS_DEVICE static uint64_t make_p_smem_desc(uint8_t const *smem_p_stage) {
    return make_sw128_major_k_smem_desc(smem_p_stage,
                                        /*leading_bytes=*/Traits::kSmemQBytesPerStage,
                                        /*stride_bytes=*/8 * Traits::kHeadDim);
  }

  CUTLASS_DEVICE static uint64_t advance_smem_desc_k_step(uint64_t smem_desc) {
    cute::UMMA::SmemDescriptor desc{};
    desc.desc_ = smem_desc;
    desc.lo += kSmemDescUnitsPerKStep;
    return desc;
  }

  CUTLASS_DEVICE static void qk_mma_tmem_a_smem_b(uint32_t tmem_s, uint32_t tmem_k,
                                                  uint64_t smem_q_desc, bool accumulate) {
    QkAtom::fma(tmem_k, smem_q_desc, tmem_s, static_cast<uint32_t>(accumulate), qk_instr_desc());
  }

  CUTLASS_DEVICE static void issue_qk_stage(uint32_t tmem_s, uint32_t tmem_k, uint64_t smem_q_desc,
                                            bool accumulate_s) {
    CUTLASS_PRAGMA_UNROLL
    for (int ki = 0; ki < kMmaKSteps; ++ki) {
      qk_mma_tmem_a_smem_b(tmem_s, tmem_k + static_cast<uint32_t>(ki) * kTmemColsPerKStep,
                           smem_q_desc, accumulate_s || ki != 0);
      smem_q_desc = advance_smem_desc_k_step(smem_q_desc);
    }
  }

  CUTLASS_DEVICE static void issue_pv_stage(uint32_t tmem_o, uint32_t tmem_v, uint64_t smem_p_desc,
                                            bool read_o) {
    CUTLASS_PRAGMA_UNROLL
    for (int ki = 0; ki < kMmaKSteps; ++ki) {
      qk_mma_tmem_a_smem_b(tmem_o, tmem_v + static_cast<uint32_t>(ki) * kTmemColsPerKStep,
                           smem_p_desc, read_o || ki != 0);
      smem_p_desc = advance_smem_desc_k_step(smem_p_desc);
    }
  }

  CUTLASS_DEVICE static void acquire_s_stage(Storage &storage, int s_stage,
                                             uint32_t s_empty_phase) {
    Sm100FmhaBarrier::wait(s_empty_barrier(storage, s_stage), s_empty_phase,
                           static_cast<uint32_t>(110 + s_stage));
  }

  CUTLASS_DEVICE static void commit_s_stage(Storage &storage, int s_stage) {
    Sm100FmhaBarrier::umma_arrive(s_full_barrier(storage, s_stage));
  }

  CUTLASS_DEVICE void issue_qk_tile(Storage &storage, uint32_t tmem_base, uint64_t q_desc,
                                    int s_stage, int transformed_stage,
                                    uint32_t transformed_full_phase) const {
    Sm100FmhaBarrier::wait(transformed_full_barrier(storage, transformed_stage),
                           transformed_full_phase, static_cast<uint32_t>(120 + transformed_stage));

    issue_qk_stage(tmem_base + s_stage_tmem_col(s_stage),
                   prepared_kv_tmem_col(tmem_base, transformed_stage), q_desc, false);

    Sm100FmhaBarrier::umma_arrive(transformed_empty_barrier(storage, transformed_stage));
    commit_s_stage(storage, s_stage);
  }

  CUTLASS_DEVICE void issue_pv_tile(Storage &storage, uint32_t tmem_base, int p_stage,
                                    uint32_t o_empty_phase, int transformed_stage,
                                    uint32_t transformed_full_phase, bool read_o,
                                    int lane_idx) const {
    uint64_t p_desc = make_p_smem_desc(storage.smem_p.stage_ptr(p_stage));

    Sm100FmhaBarrier::wait(o_empty_barrier(storage), o_empty_phase, 103);
    Sm100FmhaBarrier::wait(transformed_full_barrier(storage, transformed_stage),
                           transformed_full_phase, static_cast<uint32_t>(140 + transformed_stage));

    issue_pv_stage(tmem_base + o_stage_tmem_col(0),
                   prepared_kv_tmem_col(tmem_base, transformed_stage), p_desc, read_o);

    Sm100FmhaBarrier::umma_arrive(transformed_empty_barrier(storage, transformed_stage));
    Sm100FmhaBarrier::umma_arrive(o_full_barrier(storage));
  }

  CUTLASS_DEVICE void run_tile(Storage &storage, Params const &params, int batch_idx,
                               int kv_head_idx, int q_token_idx, int lane_idx, State &state,
                               int kv_tile_begin = 0, int kv_tile_end = INT_MAX) const {
    int const full_tiles = kv_tile_count(params, batch_idx, kv_head_idx, q_token_idx);
    Sm100FmhaKvTileRange const tile_range =
        make_kv_tile_range(full_tiles, kv_tile_begin, kv_tile_end);
    int const tiles = tile_range.count;
    if (tiles <= 0) {
      return;
    }

    int const q_stage = state.q_event % Traits::kNumStagesQ;
    uint64_t q_desc = make_q_smem_desc(storage.smem_q.stage_ptr(q_stage));

    uint32_t volatile *tmem_state = storage.tmem_state_ptr();
    uint32_t const tmem_base = tmem_state[0];

    Sm100FmhaBarrier::wait(q_full_barrier(storage, q_stage),
                           full_phase(state.q_event, Traits::kNumStagesQ),
                           static_cast<uint32_t>(100 + q_stage));
    int const s_event_base = state.s_event;
    int const first_qk_transform_event = state.transformed_event++;
    acquire_s_event(storage, state, s_event_base);
    issue_qk_tile(storage, tmem_base, q_desc, s_stage_for_event(s_event_base),
                  first_qk_transform_event % Traits::kNumStagesTransform,
                  full_phase(first_qk_transform_event, Traits::kNumStagesTransform));

    bool read_o = false;
    CUTLASS_PRAGMA_NO_UNROLL
    for (int tile = 0; tile + 1 < tiles; ++tile) {
      int const qk_event = s_event_base + tile + 1;
      int const pv_transform_event = state.transformed_event;
      int const qk_transform_event = pv_transform_event + 1;
      acquire_s_event(storage, state, qk_event);
      issue_qk_tile(storage, tmem_base, q_desc, s_stage_for_event(qk_event),
                    qk_transform_event % Traits::kNumStagesTransform,
                    full_phase(qk_transform_event, Traits::kNumStagesTransform));

      int const pv_event = s_event_base + tile;
      acquire_s_event(storage, state, pv_event + 2);
      issue_pv_tile(storage, tmem_base, s_stage_for_event(pv_event), state.o_empty_phase,
                    pv_transform_event % Traits::kNumStagesTransform,
                    full_phase(pv_transform_event, Traits::kNumStagesTransform), read_o, lane_idx);
      read_o = true;
      state.o_empty_phase ^= 1u;
      state.transformed_event += 2;
    }

    if (cute::elect_one_sync()) {
      Sm100FmhaBarrier::arrive(q_empty_barrier(storage, q_stage));
    }
    ++state.q_event;
    int const final_pv_event = s_event_base + tiles - 1;
    int const final_pv_transform_event = state.transformed_event++;
    acquire_s_event(storage, state, final_pv_event + 2);
    issue_pv_tile(storage, tmem_base, s_stage_for_event(final_pv_event), state.o_empty_phase,
                  final_pv_transform_event % Traits::kNumStagesTransform,
                  full_phase(final_pv_transform_event, Traits::kNumStagesTransform), read_o,
                  lane_idx);
    state.o_empty_phase ^= 1u;
    state.s_event += tiles;
  }

  CUTLASS_DEVICE void operator()(Storage &storage, Params const &params, int batch_idx,
                                 int kv_head_idx, int q_token_idx, int lane_idx) const {
    State state;
    run_tile(storage, params, batch_idx, kv_head_idx, q_token_idx, lane_idx, state);
  }
};

} // namespace cutlass::fmha::collective
