#pragma once
#include "include/sm100/common/tma.cuh"

namespace mega_moe::token_parallel {
constexpr int kRouterK = 256;
// Fused router partials travel as single-copy-atomic 64-bit records
// {FP32 bits, epoch} indexed [split][token][expert]. The epoch tag is the
// readiness proof, so router jobs publish without a release fence; a relaxed
// arrival counter is only a hint telling each routed CTA's single poller when
// to load its token's records.
constexpr size_t kRouterRecords = size_t(8) << 20;
static_assert(kRouterRecords >= (size_t(7) << 20) &&
              kRouterRecords + size_t(kRouterSplits) * T * E * 8 <= (size_t(9) << 20));
// One-shot CTA barrier: this CTA's router job has finished reading its
// entry-staged pages, which the first FC1 slots overlay.
constexpr int kRouterRetired = 800;
static_assert(kRouterBar + 2 * 8 <= kRouterRetired && kRouterRetired + 8 <= 832);
constexpr int kRouterPages = (H / kRouterSplits) / kRouterK;
// Fused router page: sixteen FP32 weight rows at a 260-float stride (16B
// aligned; 260 = 4 mod 32 words), then the token vectors as exact FP32
// values split into a low (tokens 0..3) and a high (tokens 4..7) 16-byte
// vector per K position. A consumer lane reads eight experts x four adjacent
// K positions per weight load, so the stride spreads one load over all 32
// banks, and the four K positions' token vectors are broadcasts.
constexpr int kFusedRouterStride = 260;
constexpr int kFusedRouterXLo = 16 * kFusedRouterStride * 4;
constexpr int kFusedRouterXHi = kFusedRouterXLo + kRouterK * 16;
constexpr int kFusedRouterPartials = 2 * kSlotBytes;
static_assert(kFusedRouterStride % 32 == 4 && (kFusedRouterStride * 4) % 16 == 0);
static_assert(kFusedRouterXHi + kRouterK * 16 <= kSlotBytes && kFusedRouterXLo % 16 == 0);
static_assert(kRouterPages <= kSlots);
static_assert((H / kRouterSplits) % kRouterK == 0 && kRouterK % 64 == 0);
// Order-equivalent total-order key: higher score -> larger key, -0 == +0,
// NaN -> 0 so a NaN candidate is never adopted across lanes (the specified selection behavior).
__device__ inline uint32_t score_key(float s) {
  s = (s == 0.f) ? 0.f : s;
  uint32_t b = __float_as_uint(s);
  b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  return s == s ? b : 0u;
}
__device__ __forceinline__ void order_candidate(uint32_t &ka, int &ia, float &ra, uint32_t &kb,
                                                int &ib, float &rb) {
  if (kb > ka || (kb == ka && ib < ia)) {
    uint32_t kt = ka;
    ka = kb;
    kb = kt;
    int it = ia;
    ia = ib;
    ib = it;
    float rt = ra;
    ra = rb;
    rb = rt;
  }
}
// Fused router jobs stage both weight pages and their BF16 token vectors at
// kernel entry, before the CTA's startup join: nothing here depends on this
// invocation's epoch. Two dedicated one-shot barriers join the four packing
// warps with the weight transfer; the ring stays untouched for later stages.
__device__ inline void produce_router_early(uint8_t *smem, KernelArgs const &k, int job, int ptid) {
  static_assert(kRouterPages == 2 && kRouterPages <= kSlots && kRouterK == 2 * kProducers);
  uint32_t const bars = smem_address(smem + kRouterBar);
  int const tile = job / kRouterSplits, split = job % kRouterSplits;
  if (ptid < 32) {
    // Warp 0 initializes both one-shot barriers and requests all 32 weight
    // rows, one per lane, without waiting for the other producer warps;
    // their arrivals follow the producer join below.
    if (ptid == 0) {
      init_bar(bars, 4);
      init_bar(bars + 8, 4);
      init_bar(smem_address(smem + kRouterRetired), 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      expect_page(bars, 16 * kRouterK * 4);
      expect_page(bars + 8, 16 * kRouterK * 4);
    }
    __syncwarp();
    // Publish the initialized page barriers to the consumer warps, which run
    // this job ahead of the CTA startup join (op_router_prejoin).
    asm volatile("bar.arrive 10, 288;" ::: "memory");
    // One 1 KiB weight row per lane; the rows need no tensor-map descriptor.
    int const stage = ptid >> 4, row = ptid & 15;
    int const k0 = split * (H / kRouterSplits) + stage * kRouterK;
    bulk_load_ef(smem + kRingOffset + stage * kSlotBytes + row * kFusedRouterStride * 4,
                 k.a.router_weight + size_t(tile * 16 + row) * H + k0, kRouterK * 4,
                 bars + stage * 8);
  }
  // Request this job's BF16 token vectors before the join so their latency
  // overlaps it. Adjacent producer lanes read adjacent K positions.
  uint32_t words[kRouterPages][2][4];
#pragma unroll
  for (int stage = 0; stage < kRouterPages; ++stage) {
    auto raw = k.a.post_attention_norm + split * (H / kRouterSplits) + stage * kRouterK;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int const kk = ptid + j * kProducers;
#pragma unroll
      for (int pair = 0; pair < 4; ++pair)
        words[stage][j][pair] =
            uint32_t(raw[(2 * pair) * H + kk]) | (uint32_t(raw[(2 * pair + 1) * H + kk]) << 16);
    }
  }
  producer_sync();
  // Each BF16 code becomes its exact FP32 value (the code in the upper
  // half), stored as the FFMA2 operand pairs (t0,t1),(t2,t3) | (t4,t5),(t6,t7).
#pragma unroll
  for (int stage = 0; stage < kRouterPages; ++stage) {
    uint8_t *const page = smem + kRingOffset + stage * kSlotBytes;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int const kk = ptid + j * kProducers;
      uint32_t const *w = words[stage][j];
      *reinterpret_cast<uint4 *>(page + kFusedRouterXLo + kk * 16) =
          make_uint4(w[0] << 16, w[0] & 0xffff0000u, w[1] << 16, w[1] & 0xffff0000u);
      *reinterpret_cast<uint4 *>(page + kFusedRouterXHi + kk * 16) =
          make_uint4(w[2] << 16, w[2] & 0xffff0000u, w[3] << 16, w[3] & 0xffff0000u);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncwarp();
    if ((ptid & 31) == 0)
      arrive_bar(bars + stage * 8);
  }
}
__device__ __forceinline__ uint64_t *router_records(Args const &a) {
  return reinterpret_cast<uint64_t *>(arena(a, a.rank) + kRouterRecords);
}
__device__ __forceinline__ void router_record_store(uint64_t *p, uint32_t bits, uint32_t epoch) {
  uint64_t const record = uint64_t(bits) | (uint64_t(epoch) << 32);
  asm volatile("st.relaxed.gpu.global.b64 [%0],%1;" ::"l"(p), "l"(record) : "memory");
}
__device__ __forceinline__ uint64_t router_record_load(uint64_t const *p) {
  uint64_t record;
  asm volatile("ld.relaxed.gpu.global.b64 %0,[%1];" : "=l"(record) : "l"(p) : "memory");
  return record;
}
// Every routed CTA selects its own token's routes from the published router
// partials with the owner's fixed split order, sigmoid and selection rule, so
// all sixteen CTAs of a token hold identical routes without a second
// cross-CTA hand-off. Each producer thread reduces one expert.
__device__ inline void select_routes(Context const &c, Args const &a, int role, float bias) {
  static_assert(kProducers == E);
  int const t = token(role), ptid = c.producer.ptid, lane = ptid & 31;
  float *raw_s = reinterpret_cast<float *>(c.smem + kRouteScratch);
  float *score_s = raw_s + E;
  {
    int const e = ptid;
    // Each thread polls its expert's twelve epoch-tagged split records
    // directly: the tags are the readiness proof, so no arrival-counter hint
    // has to be observed first, and the last job's records are adopted one
    // L2 round trip after they land. All pending loads issue before any tag
    // is inspected, and only this invocation's records are adopted (any
    // record not yet visible is re-read). The FP32 sum keeps the split order.
    uint64_t const *records = router_records(a) + size_t(t) * E + e;
    float part[kRouterSplits];
    uint32_t pending = (1u << kRouterSplits) - 1u;
    for (int spins = 0; pending; ++spins) {
      uint64_t value[kRouterSplits];
#pragma unroll
      for (int split = 0; split < kRouterSplits; ++split)
        if (pending & (1u << split))
          value[split] = router_record_load(records + size_t(split) * T * E);
#pragma unroll
      for (int split = 0; split < kRouterSplits; ++split)
        if ((pending & (1u << split)) && uint32_t(value[split] >> 32) == c.local_epoch) {
          part[split] = __uint_as_float(uint32_t(value[split]));
          pending &= ~(1u << split);
        }
      if (pending && spins > 2)
        __nanosleep(64);
    }
    float logit = 0.f;
#pragma unroll
    for (int split = 0; split < kRouterSplits; ++split)
      logit += part[split];
    float p = 1.f / (1.f + __expf(-logit));
    p = p == p ? p : 0.f;
    raw_s[e] = p;
    // This thread's expert bias was requested at kernel entry, off the
    // router hand-off's critical path.
    score_s[e] = p + bias;
  }
  producer_sync();
  if (ptid < 32) {
    int ids[K];
    float raw[4], score[4], weights[K];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int e = lane + i * 32;
      raw[i] = raw_s[e];
      score[i] = score_s[e];
    }
    float denominator = 0.f;
    bool any_nan = false;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      any_nan |= score[i] != score[i];
    any_nan = __any_sync(0xffffffffu, any_nan);
    if (!any_nan) {
      uint32_t key[4];
      int candidate[4];
      float candidate_raw[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        key[i] = score_key(score[i]);
        candidate[i] = lane + i * 32;
        candidate_raw[i] = raw[i];
      }
      order_candidate(key[0], candidate[0], candidate_raw[0], key[1], candidate[1],
                      candidate_raw[1]);
      order_candidate(key[2], candidate[2], candidate_raw[2], key[3], candidate[3],
                      candidate_raw[3]);
      order_candidate(key[0], candidate[0], candidate_raw[0], key[2], candidate[2],
                      candidate_raw[2]);
      order_candidate(key[1], candidate[1], candidate_raw[1], key[3], candidate[3],
                      candidate_raw[3]);
      order_candidate(key[1], candidate[1], candidate_raw[1], key[2], candidate[2],
                      candidate_raw[2]);
#pragma unroll
      for (int selected = 0; selected < K; ++selected) {
        uint32_t local_key = key[0];
        int winner = candidate[0];
        uint32_t kmax = __reduce_max_sync(0xffffffffu, local_key);
        winner = __reduce_min_sync(0xffffffffu, local_key == kmax ? winner : E);
        float weight = winner == candidate[0] ? candidate_raw[0] : 0.f;
        weight = __shfl_sync(0xffffffffu, weight, winner & 31);
        if (winner == candidate[0]) {
          key[0] = key[1];
          key[1] = key[2];
          key[2] = key[3];
          candidate[0] = candidate[1];
          candidate[1] = candidate[2];
          candidate[2] = candidate[3];
          candidate_raw[0] = candidate_raw[1];
          candidate_raw[1] = candidate_raw[2];
          candidate_raw[2] = candidate_raw[3];
        }
        ids[selected] = winner;
        weights[selected] = weight;
        denominator += weight;
      }
    } else {
      // Preserve the per-round shuffle behavior for every NaN case.
      bool used[4] = {};
#pragma unroll
      for (int selected = 0; selected < K; ++selected) {
        int winner = -1;
        float best = -__int_as_float(0x7f800000);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int e = lane + i * 32;
          if (!used[i] && (winner < 0 || score[i] > best || (score[i] == best && e < winner))) {
            best = score[i];
            winner = e;
          }
        }
        if (!__any_sync(0xffffffffu, best != best)) {
          uint32_t key = score_key(best);
          uint32_t kmax = __reduce_max_sync(0xffffffffu, key);
          winner = __reduce_min_sync(0xffffffffu, key == kmax ? winner : E);
        } else {
#pragma unroll
          for (int distance = 16; distance; distance >>= 1) {
            float other = __shfl_xor_sync(0xffffffffu, best, distance);
            int e = __shfl_xor_sync(0xffffffffu, winner, distance);
            if (e >= 0 && (winner < 0 || other > best || (other == best && e < winner))) {
              best = other;
              winner = e;
            }
          }
        }
        float weight = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i)
          if (lane + i * 32 == winner) {
            used[i] = true;
            weight = raw[i];
          }
        weight = __shfl_sync(0xffffffffu, weight, winner & 31);
        ids[selected] = winner;
        weights[selected] = weight;
        denominator += weight;
      }
    }
    if (!(denominator > 0.f))
      denominator = 1.f;
    // Lanes 0..3 each normalize one route with the same IEEE division, in
    // parallel instead of four sequential divisions on lane zero.
    if (lane < K) {
      auto ids_s = reinterpret_cast<int *>(c.smem + kRouteIds);
      auto weights_s = reinterpret_cast<float *>(c.smem + kRouteWeights);
      int id = ids[0];
      float weight = weights[0];
#pragma unroll
      for (int i = 1; i < K; ++i)
        if (lane == i) {
          id = ids[i];
          weight = weights[i];
        }
      float const w = weight / denominator;
      ids_s[lane] = id;
      weights_s[lane] = w;
    }
    // The warp barrier orders lanes 1..3's stores before lane zero's
    // CTA-scope release, which publishes the routes to every consumer warp.
    __syncwarp();
    if (!lane)
      arrive_bar(smem_address(c.smem + kRouteBar));
  }
  producer_sync();
  // A router-job CTA's first FC1 slots overlay its entry-staged router pages.
  if (ptid == 0 && role >= 8 && role < 8 + kRouterJobs)
    wait_bar(smem_address(c.smem + kRouterRetired), 0);
}
__device__ inline void quantize_input8(Args const &a, int t, int base, float const (&value)[8]) {
  int lane = int(threadIdx.x) & 31;
  float local_amax = fabsf(value[0]);
#pragma unroll
  for (int j = 1; j < 8; ++j)
    local_amax = fmaxf(local_amax, fabsf(value[j]));

  float shared_amax = group_max(local_amax, 4);
  uint32_t shared_code = 0;
  float shared_inv = 0.f;
  if (!(lane & 3)) {
    shared_code = m3::ue8m0_up(shared_amax * (1.f / 448.f));
    shared_inv = m3::input_inv_scale(uint8_t(shared_code));
  }
  int shared_leader = lane & ~3;
  shared_code = __shfl_sync(0xffffffffu, shared_code, shared_leader);
  shared_inv = __shfl_sync(0xffffffffu, shared_inv, shared_leader);
  uint64_t shared_bits = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint16_t pair;
    float lo = value[2 * j] * shared_inv, hi = value[2 * j + 1] * shared_inv;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0,%2,%1;" : "=h"(pair) : "f"(lo), "f"(hi));
    shared_bits |= uint64_t(pair) << (16 * j);
  }
  *reinterpret_cast<uint64_t *>(a.shared_input + size_t(t) * H + base) = shared_bits;
  if (!(lane & 3))
    a.shared_input_sf[size_t(t) * (H / 32) + base / 32] = uint8_t(shared_code);

  float routed_amax = group_max(local_amax, 2);
  uint32_t routed_code = 0;
  float routed_inv = 0.f;
  if (!(lane & 1)) {
    routed_code = m3::to_e4m3(fminf(routed_amax * (1.f / 6.f), 448.f));
    float scale = m3::e4m3(uint8_t(routed_code));
    routed_inv = fminf(reciprocal(scale), 3.4028234663852886e38f);
    if (!(scale > 0.f))
      routed_inv = 0.f;
  }
  int routed_leader = lane & ~1;
  routed_code = __shfl_sync(0xffffffffu, routed_code, routed_leader);
  routed_inv = __shfl_sync(0xffffffffu, routed_inv, routed_leader);
  uint32_t routed_bits = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j)
    routed_bits |= uint32_t(pack_fp4(value[2 * j] * routed_inv, value[2 * j + 1] * routed_inv))
                   << (8 * j);
  *reinterpret_cast<uint32_t *>(a.routed_input + size_t(t) * (H / 2) + base / 2) = routed_bits;
  if (!(lane & 1))
    a.routed_input_sf[size_t(t) * (H / 16) + base / 16] = uint8_t(routed_code);
}
__device__ inline void unpack_bf16x8(uint4 v, float (&f)[8]) {
  uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    f[2 * q] = m3::from_bits((w[q] & 0xffffu) << 16);
    f[2 * q + 1] = m3::from_bits(w[q] & 0xffff0000u);
  }
}
constexpr int kInputBlocks = H / 8 / kConsumers;
struct InputLoad {
  float value[kInputBlocks][8];
};
// Load the stored BF16 boundary exactly; no RMS reduction, gamma or FP32 norm buffer.
__device__ inline void op_input_load(Args const &a, int t, int tid, InputLoad &r) {
  uint16_t const *row = a.post_attention_norm + size_t(t) * H;
#pragma unroll
  for (int b = 0; b < kInputBlocks; ++b)
    unpack_bf16x8(*reinterpret_cast<uint4 const *>(row + (tid + b * kConsumers) * 8), r.value[b]);
}
__device__ inline void op_input_pack(Context const &c, Args const &a, int t, InputLoad const &r) {
#pragma unroll
  for (int b = 0; b < kInputBlocks; ++b)
    quantize_input8(a, t, (c.tid + b * kConsumers) * 8, r.value[b]);
  asm volatile("fence.proxy.async.global;" ::: "memory");
  publish_local(c, a, kInputReady, t);
}
} // namespace mega_moe::token_parallel
