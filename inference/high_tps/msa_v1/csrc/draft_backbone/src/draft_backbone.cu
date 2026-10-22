// MiniMax-M3.1 DSpark draft backbone, TP4, one request's seven-row block: the
// five dense decoder layers and the final norm in one cooperative launch.
//
// Every layer runs SGLang's operators in order, with their rounding points:
//   Gemma RMSNorm -> MXFP8 QKV -> per-head Gemma q/k norm + NeoX RoPE (K/V to
//   the cache) -> SWA attention over the last 4096 keys (bidirectional inside
//   the block) -> MXFP8 o_proj -> TP4 sum + residual + Gemma RMSNorm -> MXFP8
//   gate_up -> SwiGLU-OAI -> MXFP8 down -> TP4 sum + residual + the next norm.
//
// One CTA per SM. The last warp of every CTA streams the CTA's weight tiles for
// the whole call (every layer and GEMM, in consumption order) through an mbarrier
// ring, one tensor-map TMA op (16 rows x 2048 K bytes, 128-byte swizzle) plus one
// bulk copy of the tile's scales per stage; it never waits for
// activations, so the weight stream overlaps the dependent phases. Warps 0-7
// run the phases. Every GEMM is cut into 16-row full-K units (gate_up: 8 gate
// rows + the matching 8 up rows); the eight warps split K and add their sums in
// warp order. qkv units store BF16 rows, gate_up units apply SwiGLU, o_proj and
// down units push their BF16 rows (this rank's TP4 partial) straight to every
// rank. The attention CTAs apply the q/k norm + RoPE themselves and use the
// block's fresh K/V rows directly; the down CTAs quantize the SwiGLU output.
// The GEMM inner loop converts e4m3 pairs to f16 for two exact m16n8k16 HMMAs per
// 32-wide group and applies the two UE8M0 scales to the FP32 group result with
// packed f32x2 multiplies/FMAs (activation scales decoded once per phase).
//
// Hand-overs are self-validating 8-byte words that carry the call generation, so
// every reader polls the data itself (no release fence, no counter round trip):
// the normed MXFP8 activations, the qkv rows, the SwiGLU rows, the o_proj input
// and the TP4 partials; only the FP32 attention partials use a release counter.
// Every rank adds the four TP4 partials in rank order, so all ranks hold
// identical bits.
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "include/draft_args.hpp"

namespace m31_draft {
namespace {
constexpr int kConsumerWarps = 8;
constexpr int kConsumerThreads = kConsumerWarps * 32;
constexpr int kThreads = kConsumerThreads + 32; // + the weight-streaming warp
constexpr int kGemms = 4;                       // qkv, o, gate_up, down
constexpr int kTallRows = 16;                   // weight rows of a unit
// gate_up unit: 8 gate rows + the matching 8 up rows
constexpr int kPairRows = kTallRows / 2;
// K bytes of a weight row per stage (down's second stage: 1024)
constexpr int kStageRow = 2048;
constexpr int kKBlock = 128; // K bytes of one swizzled box row
// [K block][16 rows][128 B] (gate_up: two 8-row boxes)
constexpr int kSlotW = kTallRows * kStageRow;
constexpr int kMaxSfRow = kHidden / kScaleBlock; // scale bytes of a weight row (largest K)
// + the tile's scale rows
constexpr int kSlotBytes = (kSlotW + kTallRows * kMaxSfRow + 1023) / 1024 * 1024;
constexpr int kRing = 4;   // stages in shared memory
constexpr int kChunk = 64; // keys of one attention work item
constexpr int kMaxChunks = kWindow / kChunk;
constexpr int kAttnRows = kRows * kQHeads; // 112: row = token * 16 + head
// TP4 sums of one call: embedding, then o_proj, down per layer
constexpr int kSums = 2 * kLayers + 1;
constexpr int kLLWords = kHidden / 2; // TP4 words of one token row
constexpr float kSoftmaxScaleLog2 = 0.08838834764831845f * 1.4426950408889634f;
constexpr float kNegInf = -__builtin_huge_valf();
constexpr unsigned long long kWaitTimeoutNs = 10ull * 1000 * 1000 * 1000;

enum : int { kGemmQkv = 0, kGemmO = 1, kGemmGateUp = 2, kGemmDown = 3 };
// Per-call hand-over counters (only the attention partials use one); each receives a
// fixed number of arrivals per call.
enum : int { kSigAttn = 0, kSigKinds };

// Arena layout (bytes). The arena is zeroed once; counters and TP4 words carry the call generation.
constexpr size_t kCallWord = 0;    // u32: completed calls
constexpr size_t kStartWord = 128; // u32: CTAs that read the generation, every call
constexpr size_t kSignals = 4096;  // u32 per 128-byte line: [kind][16][kSignalSubs]
constexpr int kSignalSlots = 16;
// arrivals spread over 8 counters (arrival id mod 8), waited on in parallel
constexpr int kSignalSubs = 8;
// u64 [2][kTp sender][kRows][kLLWords]
constexpr size_t kLL = size_t(1) << 20;
constexpr size_t kResidual = size_t(3) << 20; // bf16 [kRows][kHidden]
// Self-validating LL6 words (low 48 bits payload, high 16 bits the call's tag), one buffer
// per use, each written once per call and polled directly by its readers:
//  - the normed MXFP8 activation of norm stage s < 2 * kLayers (stage 2l: qkv input of layer l,
//    stage 2l + 1: gate_up input): [kRows][1024 words of 6 e4m3 bytes | 32 words of 6 scale bytes];
//  - the raw qkv output of a layer: [kRows][kQkvRows / 2 words of a bf16 pair];
//  - the SwiGLU output of a layer: [kRows][gate_up unit][3 words: bf16 values 0-2 | 3-5 | 6-7].
constexpr int kActLLRowWords = kHidden / 6 + kHidden / kScaleBlock / 6; // 1056
constexpr int kActLLWords = kRows * kActLLRowWords;
constexpr int kQkvLLRowWords = kQkvRows / 2;
constexpr int kQkvLLWords = kRows * kQkvLLRowWords;
constexpr int kSwiLLRowWords = kInter / kPairRows * 3;
constexpr int kSwiLLWords = kRows * kSwiLLRowWords;
constexpr size_t kActLL = kResidual + (size_t(128) << 10);
constexpr size_t kQkvLL = kActLL + size_t(2 * kLayers) * kActLLWords * 8;
constexpr size_t kSwiLL = kQkvLL + size_t(kLayers) * kQkvLLWords * 8;
// f32 [kMaxChunks][kAttnRows][kHeadDim]
constexpr size_t kAttnPartial = size_t(8) << 20;
constexpr size_t kAttnStats = size_t(12) << 20; // f32 [kMaxChunks][kAttnRows][2]
// The o_proj activation as LL words (4 e4m3 bytes or one scale code, plus the call
// generation), one buffer per layer: u64 [kRows][kQSize / 4] then [kRows][kQSize / 32].
constexpr int kAttnActWords = kRows * (kQSize / 4);
constexpr int kAttnLLWords = kAttnActWords + kRows * (kQSize / kScaleBlock);
constexpr size_t kAttnLL = size_t(13) << 20;
static_assert(kSignals + size_t(kSigKinds) * kSignalSlots * kSignalSubs * 128 <= kLL);
static_assert(kLL + size_t(2) * kTp * kRows * kLLWords * 8 <= kResidual);
static_assert(kResidual + size_t(kRows) * kHidden * 2 <= kActLL);
static_assert(kHidden % 6 == 0 && kHidden / kScaleBlock % 6 == 0 && kQkvRows % 2 == 0);
static_assert(kSwiLL + size_t(kLayers) * kSwiLLWords * 8 <= kAttnPartial);
static_assert(kAttnPartial + size_t(kMaxChunks) * kAttnRows * kHeadDim * 4 <= kAttnStats);
static_assert(kAttnStats + size_t(kMaxChunks) * kAttnRows * 2 * 4 <= kAttnLL);
static_assert(kAttnLL + size_t(kLayers) * kAttnLLWords * 8 <= kArenaBytes);

// Shared memory. The ring is private to the weight stream; the phases share one union region.
constexpr int kSmemRing = 0; // [kRing][kSlotBytes], 1024-byte aligned (128-byte swizzle)
constexpr int kSmemUnion = kSmemRing + kRing * kSlotBytes;
constexpr int kARow = kHeadDim + 8; // attention row (bf16 elements)
constexpr int kSmemQ = kSmemUnion;
constexpr int kSmemK = kSmemQ + kAttnRows * kARow * 2;
constexpr int kSmemV = kSmemK + kChunk * kARow * 2;
// bf16 [2][kRows][kHeadDim]: block K | V
constexpr int kSmemFresh = kSmemV + kChunk * kARow * 2;
// f32 [2][kHeadDim]: 1 + q_norm | 1 + k_norm
constexpr int kSmemQkW = kSmemFresh + 2 * kRows * kHeadDim * 2;
// f32 [kRows][kRotaryDim]: cos | sin of each row
constexpr int kSmemCs = kSmemQkW + 2 * kHeadDim * 4;
constexpr int kSmemAttnEnd = kSmemCs + kRows * kRotaryDim * 4;
// norm CTAs: e4m3 [kHidden] then ue8m0 [kHidden / 32] of the row
constexpr int kSmemNormRow = kSmemUnion;
// GEMM activation row (bytes); 16 mod 128
constexpr int kActRow = kHidden + 16;
// activation scales row (bytes, 16-aligned)
constexpr int kActSfRow = kHidden / kScaleBlock + 16;
constexpr int kSmemAct = kSmemUnion;
constexpr int kSmemActSf = kSmemAct + kRows * kActRow;
constexpr int kSmemRed = (kSmemActSf + 8 * kActSfRow + 15) / 16 * 16; // f32 [8 warps][32][4]
// float2 per group; +16 bytes keeps the four tig rows on distinct banks
constexpr int kXfStride = kMaxSfRow + 2;
constexpr int kSmemXf = kSmemRed + 2 * kConsumerWarps * 32 * 4 * 4; // float2 [4 tig][kXfStride]
constexpr int kSmemActEnd = kSmemXf + 4 * kXfStride * 8;
constexpr int kSmemCombAcc = kSmemUnion; // f32 [8][kHeadDim]
constexpr int kSmemCombEnd = kSmemCombAcc + 8 * kHeadDim * 4;
constexpr int kSmemUnionEnd = kSmemAttnEnd > kSmemActEnd
                                  ? (kSmemAttnEnd > kSmemCombEnd ? kSmemAttnEnd : kSmemCombEnd)
                                  : (kSmemActEnd > kSmemCombEnd ? kSmemActEnd : kSmemCombEnd);
constexpr int kSmemMisc = (kSmemUnionEnd + 127) / 128 * 128;
static_assert(kSlotBytes % 1024 == 0 && kSlotW % 1024 == 0);
static_assert(kSmemK % 16 == 0 && kSmemV % 16 == 0 && kSmemFresh % 16 == 0 &&
              kSmemActSf % 16 == 0 && kActSfRow % 16 == 0);

struct SmemMisc {
  unsigned long long full[kRing];  // stage landed (TMA transaction bytes)
  unsigned long long empty[kRing]; // stage released by the eight consumer warps
  // the latency-critical inputs are on their way: start the weight stream
  unsigned long long go;
  int64_t slot_mapping[kRows];
  int64_t positions[kRows];
  int64_t input_ids[kRows];
  uint32_t epoch;
  int kv_len;
  float red[kConsumerWarps];
  float comb_l;
  float comb_stat[kMaxChunks][2];
  float comb_w[kMaxChunks];
  int fix[kChunk]; // block row whose fresh K/V replaces the window row, or -1
};
// + alignment of the dynamic base
constexpr int kSmemBytes = kSmemMisc + int(sizeof(SmemMisc)) + 1024;
static_assert(kSmemBytes <= 232448, "shared memory");

// Weight tensor maps: dims (128 bytes, rows, K / 128 blocks), byte strides (K, 128), box
// (128, 16 rows (gate_up: 8), 16 blocks (down's second stage: 8)), 128-byte swizzle.
enum : int { kMapQkv = 0, kMapO, kMapGateUp, kMapDown, kMapDownTail, kMapKinds };
struct WeightMaps {
  CUtensorMap map[kLayers][kMapKinds];
};
struct PlanState {
  int grid;                                   // one CTA per SM
  uint8_t maps_raw[sizeof(WeightMaps) + 128]; // the maps, 128-byte aligned inside
};
static_assert(sizeof(PlanState) <= kPlanStateBytes, "plan state");

__device__ __forceinline__ float bf2f(uint16_t v) { return __uint_as_float(uint32_t(v) << 16); }
__device__ __forceinline__ uint16_t f2bf(float x) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}
__device__ __forceinline__ float bf16r(float x) { return bf2f(f2bf(x)); }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  return uint32_t(f2bf(lo)) | (uint32_t(f2bf(hi)) << 16);
}
// 2^(code - 127) of a UE8M0 scale code 0..254 (code 0 is the subnormal 2^-127). The product of two
// such powers of two is exactly 2^(a + b - 254), rounded like ldexpf where it under- or overflows.
__device__ __forceinline__ float ue8m0(uint32_t code) {
  return __int_as_float(code ? code << 23 : 0x00400000u);
}
// The same value from byte `byte` (0-3) of a packed word: max(code << 23, 2^-127) is code << 23 for
// codes 1..254 and the subnormal 2^-127 (0x00400000) for code 0.
__device__ __forceinline__ float ue8m0_byte(uint32_t word, int byte) {
  int const shifted = int(byte == 3 ? word >> 1 : word << (23 - 8 * byte)) & 0x7f800000;
  return __int_as_float(max(shifted, 0x00400000));
}
__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// A wait longer than kWaitTimeoutNs (e.g. a rank that never issues the call) traps, so
// the launch fails with an error instead of hanging the device.
__device__ __forceinline__ void check_deadline(unsigned long long start) {
  if (globaltimer() - start > kWaitTimeoutNs)
    __trap();
}

// MXFP8 scale of a 32-value group: the smallest power of two that brings
// amax within 448, as SGLang's MXFP8 quantizers choose it.
__device__ __forceinline__ int e8m0_from_amax(float amax, float *inv) {
  int const bits = __float_as_int(amax);
  int const e0 = ((bits >> 23) & 0xff) - 135;
  float const threshold = __int_as_float(((e0 + 135) << 23) | 0x600000);
  int e = e0 + (amax > threshold ? 1 : 0);
  e = e < -127 ? -127 : e;
  *inv = __int_as_float((127 - e) << 23);
  return e + 127;
}
__device__ __forceinline__ uint8_t to_e4m3(float x) {
  return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t v) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(uint16_t(v)));
  return r;
}
__device__ __forceinline__ void mma_f16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                        uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
               "{%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// d = A (16 x 32 e4m3) x B (32 x 8 e4m3), the m16n8k32 fragments: every e4m3 pair converts
// exactly to f16 and the low / high pairs of each register form two m16n8k16 products
// with the same k assignment in A and B, so every product enters the FP32 sum once.
__device__ __forceinline__ void mma_e4m3_f16(float (&d)[4], uint32_t const (&a)[4],
                                             uint32_t const (&b)[2]) {
  d[0] = d[1] = d[2] = d[3] = 0.0f;
  mma_f16(d, e4m3x2_to_f16x2(a[0]), e4m3x2_to_f16x2(a[1]), e4m3x2_to_f16x2(a[2]),
          e4m3x2_to_f16x2(a[3]), e4m3x2_to_f16x2(b[0]), e4m3x2_to_f16x2(b[1]));
  mma_f16(d, e4m3x2_to_f16x2(a[0] >> 16), e4m3x2_to_f16x2(a[1] >> 16), e4m3x2_to_f16x2(a[2] >> 16),
          e4m3x2_to_f16x2(a[3] >> 16), e4m3x2_to_f16x2(b[0] >> 16), e4m3x2_to_f16x2(b[1] >> 16));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t const (&a)[4], uint32_t b0,
                                         uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
               "{%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t smem_u32(void const *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// 16-byte global -> shared copies that bypass L1 (data written by other CTAs), all in flight
// together.
__device__ __forceinline__ void cp_async_16(uint32_t dst, void const *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;" ::: "memory");
}
__device__ __forceinline__ void lds_u64(uint32_t addr, uint32_t (&v)[2]) {
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v[0]), "=r"(v[1]) : "r"(addr) : "memory");
}
// the low 48 bits, 2-byte aligned
__device__ __forceinline__ void sts_u16x3(uint32_t addr, uint64_t v) {
  asm volatile(
      "st.shared.u16 [%0], %1;\n\tst.shared.u16 [%0+2], %2;\n\tst.shared.u16 [%0+4], %3;" ::"r"(
          addr),
      "h"(uint16_t(v)), "h"(uint16_t(v >> 16)), "h"(uint16_t(v >> 32))
      : "memory");
}
__device__ __forceinline__ uint32_t lds_u32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr)
               : "memory");
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr)
               : "memory");
}

// mbarrier ring and TMA.
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n "
               "selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok)
               : "r"(bar), "r"(parity)
               : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  if (mbar_try_wait(bar, parity))
    return;
  unsigned long long const start = globaltimer();
  while (!mbar_try_wait(bar, parity))
    check_deadline(start);
}
// Weights are streamed once per call: they go first when L2 needs room, so the
// latency-critical inputs prefetched at entry and the activations stay resident.
__device__ __forceinline__ uint64_t evict_first_policy() {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}
__device__ __forceinline__ void bulk_load(uint32_t dst, void const *src, uint32_t bytes,
                                          uint32_t bar, uint64_t policy) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
               "[%0], [%1], %2, [%3], %4;" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar), "l"(policy)
               : "memory");
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, CUtensorMap const *map, int c0, int c1,
                                            int c2, uint32_t bar, uint64_t policy) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes."
               "L2::cache_hint "
               "[%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
               "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(bar),
               "l"(policy)
               : "memory");
}
__device__ __forceinline__ void bulk_prefetch_l2(void const *src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}

// Cross-CTA hand-over (GPU scope) and the cross-rank TP4 words (system scope).
__device__ __forceinline__ uint64_t ld_relaxed_gpu_u64(uint64_t const *p) {
  uint64_t v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_gpu_u64(uint64_t *p, uint64_t v) {
  asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_relaxed_gpu(uint32_t const *p) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void fence_acq_rel_gpu() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
}
__device__ __forceinline__ uint32_t ld_acquire_gpu(uint32_t const *p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_add_release_gpu(uint32_t *p, uint32_t v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_ll2(uint64_t *p, uint64_t a, uint64_t b) {
  asm volatile("st.relaxed.sys.global.v2.u64 [%0], {%1, %2};" ::"l"(p), "l"(a), "l"(b) : "memory");
}
__device__ __forceinline__ void st_ll(uint64_t *p, uint64_t a) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(a) : "memory");
}
__device__ __forceinline__ void ld_ll2(uint64_t const *p, uint64_t &a, uint64_t &b) {
  asm volatile("ld.relaxed.sys.global.v2.u64 {%0, %1}, [%2];"
               : "=l"(a), "=l"(b)
               : "l"(p)
               : "memory");
}
__device__ __forceinline__ void ld_relaxed_gpu_v2(uint64_t const *p, uint64_t &a, uint64_t &b) {
  asm volatile("ld.relaxed.gpu.global.v2.u64 {%0, %1}, [%2];"
               : "=l"(a), "=l"(b)
               : "l"(p)
               : "memory");
}
__device__ __forceinline__ void consumer_sync() {
  asm volatile("bar.sync 1, %0;" ::"n"(kConsumerThreads) : "memory");
}

// LL6 words: 48 payload bits and a 16-bit tag of the call generation (never 0, so the
// zeroed arena never validates; consecutive calls always differ).
__device__ __forceinline__ uint32_t ll_tag(uint32_t epoch) { return epoch % 65535u + 1u; }
__device__ __forceinline__ bool ll_ok(uint64_t word, uint32_t tag) {
  return uint32_t(word >> 48) == tag;
}
__device__ __forceinline__ uint64_t ll_word(uint64_t payload, uint32_t tag) {
  return payload | (uint64_t(tag) << 48);
}

// One thread waits until a monotonic counter reaches this call's target.
__device__ void wait_count(uint32_t const *counter, uint32_t target) {
  if (int32_t(ld_acquire_gpu(counter) - target) >= 0)
    return;
  unsigned long long const start = globaltimer();
  while (int32_t(ld_acquire_gpu(counter) - target) < 0)
    check_deadline(start);
}

struct Ctx;
template <class T> __device__ __forceinline__ T *at(uint8_t *base, size_t offset) {
  return reinterpret_cast<T *>(base + offset);
}
__device__ __forceinline__ uint32_t *signal(uint8_t *arena, int kind, int index, int id) {
  return at<uint32_t>(
      arena,
      kSignals + ((size_t(kind) * kSignalSlots + index) * kSignalSubs + id % kSignalSubs) * 128);
}
// TP4 words of one token row from one sender, in a receiver's arena.
__device__ __forceinline__ uint64_t *ll_row(uint8_t *arena, int buf, int sender, int t) {
  return at<uint64_t>(arena, kLL) + (size_t(buf * kTp + sender) * kRows + t) * kLLWords;
}
__device__ __forceinline__ uint64_t *act_ll(uint8_t *arena, int stage) {
  return at<uint64_t>(arena, kActLL) + size_t(stage) * kActLLWords;
}
__device__ __forceinline__ uint64_t *qkv_ll(uint8_t *arena, int layer) {
  return at<uint64_t>(arena, kQkvLL) + size_t(layer) * kQkvLLWords;
}
__device__ __forceinline__ uint64_t *swi_ll(uint8_t *arena, int layer) {
  return at<uint64_t>(arena, kSwiLL) + size_t(layer) * kSwiLLWords;
}

// ---------------------------------------------------------------------------
// GEMM geometry: units of 16 weight rows x the full K (gate_up: 8 gate rows and
// the matching 8 up rows), stages of 16 rows x 2048 K bytes (down's last stage:
// 1024), stored as 128-byte swizzled K blocks.
__host__ __device__ constexpr int gemm_k(int g) {
  return g == kGemmO ? kQSize : g == kGemmDown ? kInter : kHidden;
}
__host__ __device__ constexpr int gemm_n(int g) {
  return g == kGemmQkv ? kQkvRows : g == kGemmGateUp ? kGateUpRows : kHidden;
}
__host__ __device__ constexpr int gemm_units(int g) { return gemm_n(g) / kTallRows; }
struct Unit {
  int nst;    // stages
  int y0, y1; // first weight row of each box (one 16-row box, or gate | up)
};
__device__ __forceinline__ Unit make_unit(int g, int u) {
  Unit t;
  t.nst = (gemm_k(g) + kStageRow - 1) / kStageRow;
  if (g == kGemmGateUp) {
    t.y0 = u * kPairRows;
    t.y1 = kInter + u * kPairRows;
  } else {
    t.y0 = t.y1 = u * kTallRows;
  }
  return t;
}
__device__ __forceinline__ uint8_t const *gemm_scale(LayerArgs const &layer, int g) {
  return g == kGemmQkv      ? layer.qkv_sf
         : g == kGemmO      ? layer.o_sf
         : g == kGemmGateUp ? layer.gate_up_sf
                            : layer.down_sf;
}

// The sequence of weight stages one CTA consumes in a call: layer, GEMM, unit, stage.
struct StageIter {
  int layer, gemm, u, st, units;
  Unit unit;
  bool valid;
  __device__ void seek(int cta, int grid) {
    while (layer < kLayers) {
      units = gemm_units(gemm);
      if (grid - 1 - cta < units) {
        u = grid - 1 - cta;
        st = 0;
        unit = make_unit(gemm, u);
        valid = true;
        return;
      }
      if (++gemm == kGemms) {
        gemm = 0;
        ++layer;
      }
    }
    valid = false;
  }
  __device__ void start(int cta, int grid) {
    layer = 0;
    gemm = 0;
    seek(cta, grid);
  }
  __device__ void next(int cta, int grid) {
    if (++st < unit.nst)
      return;
    st = 0;
    u += grid;
    if (u < units) {
      unit = make_unit(gemm, u);
      return;
    }
    if (++gemm == kGemms) {
      gemm = 0;
      ++layer;
    }
    seek(cta, grid);
  }
};

// The weight stream: the last warp issues every stage of the call in consumption
// order. Per stage lane 0 issues one 3D TMA of the 16 weight rows (gate_up: two
// 8-row boxes) and one bulk copy of their scale rows into a ring slot.
__device__ void producer(Args const &a, WeightMaps const &maps, uint8_t *smem, SmemMisc *misc,
                         int cta, int grid, int lane) {
  uint64_t const policy = evict_first_policy();

  StageIter ld;
  ld.start(cta, grid);
  mbar_wait(smem_u32(&misc->go), 0u);
  uint32_t const ring = smem_u32(smem + kSmemRing);
  for (uint32_t iter = 0; ld.valid; ++iter) {
    uint32_t const slot = iter % kRing;
    if (iter >= kRing)
      mbar_wait(smem_u32(&misc->empty[slot]), ((iter / kRing) & 1u) ^ 1u);
    if (lane == 0) {
      LayerArgs const &layer = a.layer[ld.layer];
      int const k = gemm_k(ld.gemm), ksf = k / kScaleBlock;
      int const kc = k - ld.st * kStageRow < kStageRow ? k - ld.st * kStageRow : kStageRow;
      uint32_t const full = smem_u32(&misc->full[slot]);
      uint32_t const dst = ring + slot * kSlotBytes;
      mbar_expect_tx(full, uint32_t(kTallRows * kc + kTallRows * ksf));
      int const kb = ld.st * (kStageRow / kKBlock);
      uint8_t const *sf = gemm_scale(layer, ld.gemm);
      if (ld.gemm == kGemmGateUp) { // rows 0-7: gate, rows 8-15: the matching up rows
        tma_load_3d(dst, &maps.map[ld.layer][kMapGateUp], 0, ld.unit.y0, kb, full, policy);
        tma_load_3d(dst + kSlotW / 2, &maps.map[ld.layer][kMapGateUp], 0, ld.unit.y1, kb, full,
                    policy);
        bulk_load(dst + kSlotW, sf + size_t(ld.unit.y0) * ksf, uint32_t(kPairRows * ksf), full,
                  policy);
        bulk_load(dst + kSlotW + kPairRows * ksf, sf + size_t(ld.unit.y1) * ksf,
                  uint32_t(kPairRows * ksf), full, policy);
      } else {
        int const kind = ld.gemm == kGemmQkv ? kMapQkv
                         : ld.gemm == kGemmO ? kMapO
                         : kc == kStageRow   ? kMapDown
                                             : kMapDownTail;
        tma_load_3d(dst, &maps.map[ld.layer][kind], 0, ld.unit.y0, kb, full, policy);
        bulk_load(dst + kSlotW, sf + size_t(ld.unit.y0) * ksf, uint32_t(kTallRows * ksf), full,
                  policy);
      }
    }
    __syncwarp();
    ld.next(cta, grid);
  }
}

// ---------------------------------------------------------------------------
struct Ctx {
  Args const &a;
  uint8_t *arena; // this rank's
  uint8_t *smem;
  SmemMisc *misc;
  uint32_t epoch; // this call's generation, 1-based
  int cta, grid, tid, warp, lane;
  uint32_t ring; // weight stages consumed so far in this call
};

// Waits until all `arrivals` (ids 0 .. arrivals-1, one red per id and call) of this call
// reached a split counter: thread s polls sub-counter s, then acquires.
__device__ void wait_signal(Ctx const &c, int kind, int index, uint32_t arrivals) {
  if (c.tid < kSignalSubs) {
    uint32_t const per = (arrivals + kSignalSubs - 1 - uint32_t(c.tid)) / kSignalSubs;
    if (per != 0u) {
      uint32_t const *counter = signal(c.arena, kind, index, c.tid);
      uint32_t const target = c.epoch * per;
      if (int32_t(ld_relaxed_gpu(counter) - target) < 0) {
        unsigned long long const start = globaltimer();
        while (int32_t(ld_relaxed_gpu(counter) - target) < 0)
          check_deadline(start);
      }
      fence_acq_rel_gpu();
    }
  }
  consumer_sync();
}

__device__ __forceinline__ uint32_t sum_generation(Ctx const &c, int stage) {
  return (c.epoch - 1u) * uint32_t(kSums) + uint32_t(stage);
}

// Vocab-parallel embedding: each rank pushes its shard's row of every token it
// owns and zeros for the rest, so TP4 sum 0 is the embedding.
__device__ void embedding_phase(Ctx &c) {
  uint32_t const gen = sum_generation(c, 0);
  uint64_t const flag = uint64_t(gen + 1u) << 32;
  int const buf = int(gen & 1u);
  constexpr int kChunks = kHidden / 8;
  for (int i = c.cta * kConsumerThreads + c.tid; i < kRows * kChunks;
       i += c.grid * kConsumerThreads) {
    int const t = i / kChunks, ch = i - t * kChunks;
    int64_t const local = c.misc->input_ids[t] - c.a.vocab_start;
    int4 v = make_int4(0, 0, 0, 0);
    if (local >= 0 && local < c.a.vocab_rows)
      v = __ldg(reinterpret_cast<int4 const *>(c.a.embed + size_t(local) * kHidden) + ch);
    uint64_t const w0 = uint64_t(uint32_t(v.x)) | flag, w1 = uint64_t(uint32_t(v.y)) | flag;
    uint64_t const w2 = uint64_t(uint32_t(v.z)) | flag, w3 = uint64_t(uint32_t(v.w)) | flag;
    for (int receiver = 0; receiver < kTp; ++receiver) {
      uint64_t *dst = ll_row(c.a.arena[receiver], buf, c.a.rank, t) + ch * 4;
      st_ll2(dst, w0, w1);
      st_ll2(dst + 2, w2, w3);
    }
  }
}

// ---------------------------------------------------------------------------
// Row phase: TP4 sum + residual + Gemma RMSNorm, then MXFP8 quantization of
// the normed row (or the final output). The norm reads the FP32 sum; the
// residual stream keeps its BF16 rounding. CTA t < kRows owns token row t;
// thread i owns the 8-column chunks i, i + 256, i + 512.
__device__ void norm_phase(Ctx &c, int stage, uint16_t const *weight, bool output) {
  int const t = c.cta;
  if (t >= kRows)
    return;
  uint32_t const gen = sum_generation(c, stage);
  uint32_t const flag = gen + 1u;
  int const buf = int(gen & 1u);
  uint16_t *residual = at<uint16_t>(c.arena, kResidual) + size_t(t) * kHidden;
  int4 wv[3], rv[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    int const ch = c.tid + kConsumerThreads * j;
    wv[j] = __ldg(reinterpret_cast<int4 const *>(weight) + ch);
    rv[j] = stage > 0 ? *reinterpret_cast<int4 const *>(residual + ch * 8) : make_int4(0, 0, 0, 0);
  }
  uint64_t raw[3][kTp][4];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
#pragma unroll
    for (int sender = 0; sender < kTp; ++sender) {
      uint64_t const *src = ll_row(c.arena, buf, sender, t) + (c.tid + kConsumerThreads * j) * 4;
      ld_ll2(src, raw[j][sender][0], raw[j][sender][1]);
      ld_ll2(src + 2, raw[j][sender][2], raw[j][sender][3]);
    }
  }
  unsigned long long start = 0;
  for (;;) {
    bool ready = true;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
      for (int sender = 0; sender < kTp; ++sender) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          if (uint32_t(raw[j][sender][2 * h] >> 32) != flag ||
              uint32_t(raw[j][sender][2 * h + 1] >> 32) != flag) {
            ready = false;
            uint64_t const *src =
                ll_row(c.arena, buf, sender, t) + (c.tid + kConsumerThreads * j) * 4 + 2 * h;
            ld_ll2(src, raw[j][sender][2 * h], raw[j][sender][2 * h + 1]);
          }
        }
      }
    }
    if (ready)
      break;
    if (start == 0)
      start = globaltimer();
    else
      check_deadline(start);
  }
  float x[3][8];
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    uint32_t const res[4] = {uint32_t(rv[j].x), uint32_t(rv[j].y), uint32_t(rv[j].z),
                             uint32_t(rv[j].w)};
    uint32_t keep[4];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int const w = e >> 1, shift = (e & 1) * 16;
      float sum = bf2f(uint16_t(raw[j][0][w] >> shift));
#pragma unroll
      for (int sender = 1; sender < kTp; ++sender)
        sum += bf2f(uint16_t(raw[j][sender][w] >> shift));
      // The embedding (stage 0) starts the residual stream; every later stage adds to it.
      float const value = stage == 0 ? bf16r(sum) : bf16r(sum) + bf2f(uint16_t(res[w] >> shift));
      x[j][e] = value;
      ss = fmaf(value, value, ss);
      uint32_t const bits = f2bf(value);
      keep[w] = (e & 1) ? (keep[w] | (bits << 16)) : bits;
    }
    if (!output)
      *reinterpret_cast<int4 *>(residual + (c.tid + kConsumerThreads * j) * 8) =
          make_int4(int(keep[0]), int(keep[1]), int(keep[2]), int(keep[3]));
  }
  for (int mask = 16; mask > 0; mask >>= 1)
    ss += __shfl_xor_sync(0xffffffffu, ss, mask);
  if (c.lane == 0)
    c.misc->red[c.warp] = ss;
  consumer_sync();
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < kConsumerWarps; ++w)
    total += c.misc->red[w];
  float const factor = rsqrtf(total / float(kHidden) + c.a.eps);
  if (output) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      uint32_t const wb[4] = {uint32_t(wv[j].x), uint32_t(wv[j].y), uint32_t(wv[j].z),
                              uint32_t(wv[j].w)};
      uint32_t o[4];
#pragma unroll
      for (int w = 0; w < 4; ++w)
        o[w] = pack_bf16(x[j][2 * w] * factor * (1.0f + bf2f(uint16_t(wb[w]))),
                         x[j][2 * w + 1] * factor * (1.0f + bf2f(uint16_t(wb[w] >> 16))));
      *reinterpret_cast<int4 *>(c.a.out + size_t(t) * kHidden +
                                (c.tid + kConsumerThreads * j) * 8) =
          make_int4(int(o[0]), int(o[1]), int(o[2]), int(o[3]));
    }
    return;
  }
  // The quantized row goes to shared memory first, then out as LL6 words (6 bytes each).
  uint8_t *act = c.smem + kSmemNormRow;
  uint8_t *act_sf = act + kHidden;
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    int const ch = c.tid + kConsumerThreads * j;
    uint32_t const wb[4] = {uint32_t(wv[j].x), uint32_t(wv[j].y), uint32_t(wv[j].z),
                            uint32_t(wv[j].w)};
    float y[8], amax = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      y[e] = bf16r(x[j][e] * factor * (1.0f + bf2f(uint16_t(wb[e >> 1] >> ((e & 1) * 16)))));
      amax = fmaxf(amax, fabsf(y[e]));
    }
    // A 32-value scale group is four consecutive chunks: four consecutive lanes.
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    float inv;
    int const scale = e8m0_from_amax(amax, &inv);
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      lo |= uint32_t(to_e4m3(y[e] * inv)) << (8 * e);
      hi |= uint32_t(to_e4m3(y[e + 4] * inv)) << (8 * e);
    }
    *reinterpret_cast<uint2 *>(act + ch * 8) = make_uint2(lo, hi);
    if ((c.tid & 3) == 0)
      act_sf[ch / 4] = uint8_t(scale);
  }
  consumer_sync();
  // Publish: the GEMM CTAs poll these words directly (no release, no counter).
  uint64_t *dst = act_ll(c.arena, stage) + size_t(t) * kActLLRowWords;
  uint32_t const tag = ll_tag(c.epoch);
  for (int i = c.tid; i < kActLLRowWords; i += kConsumerThreads) {
    // scales follow the bytes
    uint16_t const *src = reinterpret_cast<uint16_t const *>(act + 6 * i);
    st_relaxed_gpu_u64(
        dst + i,
        ll_word(uint64_t(src[0]) | (uint64_t(src[1]) << 16) | (uint64_t(src[2]) << 32), tag));
  }
  consumer_sync(); // the row buffer is reused by the next phase
}

// ---------------------------------------------------------------------------
__device__ __forceinline__ bool later_duplicate(SmemMisc const *misc, int t) {
  bool dup = false;
  for (int u = t + 1; u < kRows; ++u)
    dup |= misc->slot_mapping[u] == misc->slot_mapping[t];
  return dup;
}

// ---------------------------------------------------------------------------
// MXFP8 GEMM unit u: out[token][n] = sum_k act[token][k] * w[n][k] for 16 weight
// rows and the full K. Warp w takes the 32-wide scale groups 8w..8w+7 of every
// stage (4w..4w+3 of down's 1024-byte last stage): A = weights via ldmatrix from
// the swizzled stage, B = activation^T, the e4m3 product of each group as two
// exact f16 MMAs, then the group's two UE8M0 scales applied to the FP32 result
// before accumulating in FP32; the eight warp sums are added in warp order. qkv
// publishes the BF16 rows, gate_up applies SwiGLU-OAI to the BF16 gate/up rows
// and publishes the BF16 result, o_proj and down push their BF16 rows (this
// rank's TP4 partial) to every rank.
__device__ __forceinline__ float tall_sum(float const *red, int t, int r) {
  int const lane = (r & 7) * 4 + (t >> 1), j = (r >> 3) * 2 + (t & 1);
  float v = red[lane * 4 + j];
#pragma unroll
  for (int w = 1; w < kConsumerWarps; ++w)
    v += red[(w * 32 + lane) * 4 + j];
  return v;
}

__device__ void tall_unit(Ctx &c, int layer, int g, int u, int parity) {
  int const k = gemm_k(g), ksf = k / kScaleBlock;
  int const nst = (k + kStageRow - 1) / kStageRow;
  uint8_t const *s_act = c.smem + kSmemAct;
  uint8_t const *s_asf = c.smem + kSmemActSf;
  float *red = reinterpret_cast<float *>(c.smem + kSmemRed) + parity * kConsumerWarps * 32 * 4;
  int const lane = c.lane, g8 = lane >> 2, tig = lane & 3;
  // ldmatrix: lane supplies row lrow (rows 0-7 / 8-15 for matrices 0,2 / 1,3) and the 16-byte half
  // lhalf of a 32-byte group; a box is [K block][rows][128 B] with chunk c of row r at c ^ (r % 8).
  int const lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lhalf = lane >> 4;
  bool const pair = g == kGemmGateUp;
  uint32_t const row_off =
      pair ? (lrow >> 3) * (kSlotW / 2) + (lrow & 7) * kKBlock : lrow * kKBlock;
  // one K block of a box
  uint32_t const block_bytes = pair ? kPairRows * kKBlock : kTallRows * kKBlock;
  uint32_t const b_row = smem_u32(s_act + (g8 < kRows ? g8 : 0) * kActRow + tig * 4);
  bool const b_valid = g8 < kRows;
  float2 const *xf = reinterpret_cast<float2 const *>(c.smem + kSmemXf) + tig * kXfStride;
  // rows g8, g8 + 8; tokens 2 tig, + 1
  float2 acc01 = make_float2(0.0f, 0.0f), acc23 = make_float2(0.0f, 0.0f);
  for (int st = 0; st < nst; ++st) {
    uint32_t const slot = c.ring % kRing;
    mbar_wait(smem_u32(&c.misc->full[slot]), (c.ring / kRing) & 1u);
    uint32_t const slot_base = smem_u32(c.smem + kSmemRing) + slot * kSlotBytes;
    int const kc = k - st * kStageRow < kStageRow ? k - st * kStageRow : kStageRow;
    int const gw = kc / (kConsumerWarps * kScaleBlock);  // scale groups of this warp: 8, or 4
    int const j0 = c.warp * gw;                          // first group of this warp in the stage
    int const grp = st * (kStageRow / kScaleBlock) + j0; // first group of this warp in K
    uint32_t sw0[2], sw1[2];
    float2 xs[8];
    uint32_t const sw_base = slot_base + kSlotW;
    if (gw == 8) {
      lds_u64(sw_base + g8 * ksf + grp, sw0);
      lds_u64(sw_base + (g8 + 8) * ksf + grp, sw1);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float4 const v = *reinterpret_cast<float4 const *>(xf + grp + 2 * q);
        xs[2 * q] = make_float2(v.x, v.y);
        xs[2 * q + 1] = make_float2(v.z, v.w);
      }
    } else {
      sw0[0] = lds_u32(sw_base + g8 * ksf + grp);
      sw1[0] = lds_u32(sw_base + (g8 + 8) * ksf + grp);
      sw0[1] = sw1[1] = 0u;
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        float4 const v = *reinterpret_cast<float4 const *>(xf + grp + 2 * q);
        xs[2 * q] = make_float2(v.x, v.y);
        xs[2 * q + 1] = make_float2(v.z, v.w);
      }
#pragma unroll
      for (int q = 4; q < 8; ++q)
        xs[q] = make_float2(0.0f, 0.0f);
    }
    uint32_t const a_stage = slot_base + row_off;
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      if (jj < gw) {
        // group in the stage: K block jl / 4, 32-byte group jl % 4 of its 128 bytes
        int const jl = j0 + jj;
        uint32_t a[4];
        ldsm_x4(a, a_stage + (jl >> 2) * block_bytes +
                       (uint32_t((((jl & 3) * 2 + lhalf) ^ (lrow & 7))) << 4));
        uint32_t b[2] = {0u, 0u};
        if (b_valid) {
          b[0] = lds_u32(b_row + (grp + jj) * kScaleBlock);
          b[1] = lds_u32(b_row + (grp + jj) * kScaleBlock + 16);
        }
        float d[4];
        mma_e4m3_f16(d, a, b);
        // FP32 scale products and accumulation as packed pairs (IEEE, same as the scalar ops).
        float const w0 = ue8m0_byte(sw0[jj >> 2], jj & 3), w1 = ue8m0_byte(sw1[jj >> 2], jj & 3);
        float2 const p0 = __fmul2_rn(make_float2(w0, w0), xs[jj]);
        float2 const p1 = __fmul2_rn(make_float2(w1, w1), xs[jj]);
        acc01 = __ffma2_rn(make_float2(d[0], d[1]), p0, acc01);
        acc23 = __ffma2_rn(make_float2(d[2], d[3]), p1, acc23);
      }
    }
    __syncwarp();
    if (lane == 0)
      mbar_arrive(smem_u32(&c.misc->empty[slot]));
    ++c.ring;
  }
  *reinterpret_cast<float4 *>(red + (c.warp * 32 + lane) * 4) =
      make_float4(acc01.x, acc01.y, acc23.x, acc23.y);
  consumer_sync();
  // qkv and gate_up publish LL6 words that their readers poll directly. No barrier here:
  // the next unit reduces into the other buffer (see the o_proj / down case below).
  if (g == kGemmGateUp) {
    // Thread (t, k) produces word k of token t: SwiGLU values 3k .. 3k + 2 (word 2: 6, 7).
    if (c.tid < kRows * 3) {
      int const t = c.tid / 3, k = c.tid % 3;
      uint64_t payload = 0;
#pragma unroll
      for (int e = 0; e < 3; ++e) {
        int const i = 3 * k + e;
        if (i < kPairRows) {
          float const gv = fminf(bf16r(tall_sum(red, t, i)), kSwigluLimit);
          float const uv =
              fminf(fmaxf(bf16r(tall_sum(red, t, kPairRows + i)), -kSwigluLimit), kSwigluLimit);
          float const sigmoid = 1.0f / (1.0f + expf(-gv * kSwigluAlpha));
          payload |= uint64_t(f2bf(gv * sigmoid * (uv + 1.0f))) << (16 * e);
        }
      }
      st_relaxed_gpu_u64(swi_ll(c.arena, layer) + t * kSwiLLRowWords + u * 3 + k,
                         ll_word(payload, ll_tag(c.epoch)));
    }
    return;
  }
  if (g == kGemmQkv) {
    if (c.tid < kRows * kTallRows / 2) {
      int const t = c.tid / (kTallRows / 2), pair = c.tid % (kTallRows / 2);
      uint64_t const word = pack_bf16(tall_sum(red, t, 2 * pair), tall_sum(red, t, 2 * pair + 1));
      st_relaxed_gpu_u64(qkv_ll(c.arena, layer) + t * kQkvLLRowWords + u * (kTallRows / 2) + pair,
                         ll_word(word, ll_tag(c.epoch)));
    }
    return;
  }
  if (c.tid < kRows * kTallRows / 2) {
    int const t = c.tid / (kTallRows / 2), pair = c.tid % (kTallRows / 2);
    uint32_t const gen = sum_generation(c, g == kGemmO ? 2 * layer + 1 : 2 * layer + 2);
    uint64_t const word =
        uint64_t(pack_bf16(tall_sum(red, t, 2 * pair), tall_sum(red, t, 2 * pair + 1))) |
        (uint64_t(gen + 1u) << 32);
    for (int receiver = 0; receiver < kTp; ++receiver)
      st_ll(ll_row(c.a.arena[receiver], int(gen & 1u), c.a.rank, t) + u * (kTallRows / 2) + pair,
            word);
  }
  // No barrier: the next unit reduces into the other buffer, and the one after it
  // writes this buffer only after that unit's barrier, which follows these reads.
}

// down's input: the BF16 SwiGLU output from its LL6 words, MXFP8-quantized here
// (SGLang's quantizer: UE8M0 per 32 values, E4M3 RNE). Thread i < 224 owns the 32-value
// groups i % 32 + 32 m (m < 3) of token row i / 32: the 12 words of a group's four gate_up
// units are contiguous, every load is in flight at once, and every address is a per-thread
// base plus a constant.
__device__ void load_swiglu_ll(Ctx &c, int layer) {
  constexpr int kGroupsRow = kInter / kScaleBlock; // 96
  constexpr int kLanes = 32;
  constexpr int kPer = kGroupsRow / kLanes;                // 3
  constexpr int kGroupWords = kScaleBlock / kPairRows * 3; // 12
  static_assert(kGroupsRow % kLanes == 0 && kRows * kLanes <= kConsumerThreads);
  uint8_t *s_asf = c.smem + kSmemActSf;
  if (c.tid < kRows * kLanes) {
    int const t = c.tid / kLanes, g0 = c.tid % kLanes;
    uint64_t const *src = swi_ll(c.arena, layer) + t * kSwiLLRowWords + g0 * kGroupWords;
    uint32_t const tag = ll_tag(c.epoch);
    uint64_t w[kPer][kGroupWords];
#pragma unroll
    for (int k = 0; k < kPer; ++k)
#pragma unroll
      for (int e = 0; e < kGroupWords; e += 2)
        ld_relaxed_gpu_v2(src + k * kLanes * kGroupWords + e, w[k][e], w[k][e + 1]);
    unsigned long long start = 0;
    for (;;) {
      bool ready = true;
#pragma unroll
      for (int k = 0; k < kPer; ++k)
#pragma unroll
        for (int e = 0; e < kGroupWords; e += 2) {
          if (!ll_ok(w[k][e], tag) || !ll_ok(w[k][e + 1], tag)) {
            ready = false;
            ld_relaxed_gpu_v2(src + k * kLanes * kGroupWords + e, w[k][e], w[k][e + 1]);
          }
        }
      if (ready)
        break;
      if (start == 0)
        start = globaltimer();
      else
        check_deadline(start);
    }
    uint8_t *s_act = c.smem + kSmemAct + t * kActRow + g0 * kScaleBlock;
    uint8_t *s_sf = s_asf + t * kActSfRow + g0;
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
      // Value j of the group (unit j / 8, value i = j % 8) is half i % 3 of word 3 (j / 8) + i / 3.
      auto value = [&](int j) {
        int const unit = j / kPairRows, i = j % kPairRows;
        return bf2f(uint16_t(w[k][3 * unit + i / 3] >> (16 * (i % 3))));
      };
      float amax = 0.0f;
#pragma unroll
      for (int j = 0; j < kScaleBlock; ++j)
        amax = fmaxf(amax, fabsf(value(j)));
      float inv;
      int const scale = e8m0_from_amax(amax, &inv);
      uint32_t b[kScaleBlock / 4];
#pragma unroll
      for (int i = 0; i < kScaleBlock / 4; ++i)
        b[i] = uint32_t(to_e4m3(value(4 * i) * inv)) |
               (uint32_t(to_e4m3(value(4 * i + 1) * inv)) << 8) |
               (uint32_t(to_e4m3(value(4 * i + 2) * inv)) << 16) |
               (uint32_t(to_e4m3(value(4 * i + 3) * inv)) << 24);
      uint4 *dst = reinterpret_cast<uint4 *>(s_act + k * kLanes * kScaleBlock);
      dst[0] = make_uint4(b[0], b[1], b[2], b[3]);
      dst[1] = make_uint4(b[4], b[5], b[6], b[7]);
      s_sf[k * kLanes] = uint8_t(scale);
    }
  }
  for (int i = c.tid; i < kInter / kScaleBlock; i += kConsumerThreads)
    s_asf[kRows * kActSfRow + i] = 0x7f; // padding token 7: unit scale, its column is discarded
}

// The normed MXFP8 activation of norm stage `stage` from its LL6 words, polled directly
// and unpacked into the GEMM's shared-memory rows. Thread i owns the act words i + 256 m
// (m < 4) of every token row and, for i < 224, scale word i % 32 of row i / 32, so every
// address is a per-thread base plus a constant.
__device__ void load_act_ll(Ctx &c, int stage) {
  constexpr int kActWords = kHidden / 6;                // 1024 per row
  constexpr int kPerRow = kActWords / kConsumerThreads; // 4
  constexpr int kSfWords = kActLLRowWords - kActWords;  // 32 per row
  static_assert(kActWords % kConsumerThreads == 0 && kRows * kSfWords <= kConsumerThreads);
  uint64_t const *src = act_ll(c.arena, stage) + c.tid;
  uint64_t const *sf_src =
      act_ll(c.arena, stage) + (c.tid / kSfWords) * kActLLRowWords + kActWords + c.tid % kSfWords;
  bool const has_sf = c.tid < kRows * kSfWords;
  uint32_t const tag = ll_tag(c.epoch);
  uint64_t w[kRows][kPerRow], ws;
#pragma unroll
  for (int t = 0; t < kRows; ++t)
#pragma unroll
    for (int m = 0; m < kPerRow; ++m)
      w[t][m] = ld_relaxed_gpu_u64(src + t * kActLLRowWords + m * kConsumerThreads);
  ws = has_sf ? ld_relaxed_gpu_u64(sf_src) : ll_word(0, tag);
  unsigned long long start = 0;
  for (;;) {
    bool ready = ll_ok(ws, tag);
#pragma unroll
    for (int t = 0; t < kRows; ++t)
#pragma unroll
      for (int m = 0; m < kPerRow; ++m)
        ready &= ll_ok(w[t][m], tag);
    if (ready)
      break;
    if (start == 0)
      start = globaltimer();
    else
      check_deadline(start);
#pragma unroll
    for (int t = 0; t < kRows; ++t)
#pragma unroll
      for (int m = 0; m < kPerRow; ++m)
        if (!ll_ok(w[t][m], tag))
          w[t][m] = ld_relaxed_gpu_u64(src + t * kActLLRowWords + m * kConsumerThreads);
    if (!ll_ok(ws, tag))
      ws = ld_relaxed_gpu_u64(sf_src);
  }
  uint32_t const act = smem_u32(c.smem + kSmemAct) + 6 * c.tid;
#pragma unroll
  for (int t = 0; t < kRows; ++t)
#pragma unroll
    for (int m = 0; m < kPerRow; ++m)
      sts_u16x3(act + t * kActRow + 6 * m * kConsumerThreads, w[t][m]);
  if (has_sf)
    sts_u16x3(smem_u32(c.smem + kSmemActSf) + (c.tid / kSfWords) * kActSfRow +
                  6 * (c.tid % kSfWords),
              ws);
  // padding token 7: unit scale
  for (int i = c.tid; i < kHidden / kScaleBlock / 4; i += kConsumerThreads)
    *reinterpret_cast<uint32_t *>(c.smem + kSmemActSf + kRows * kActSfRow + i * 4) = 0x7f7f7f7fu;
}

// The o_proj activation from its LL words: every word is polled until it carries this
// call's generation (the combine's writes need no separate counter).
__device__ void load_attn_act(Ctx &c, int layer) {
  constexpr int kPer = (kAttnLLWords + kConsumerThreads - 1) / kConsumerThreads;
  uint64_t const *src = at<uint64_t>(c.arena, kAttnLL) + size_t(layer) * kAttnLLWords;
  uint64_t w[kPer];
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    int const i = c.tid + k * kConsumerThreads;
    w[k] = i < kAttnLLWords ? ld_relaxed_gpu_u64(src + i) : uint64_t(c.epoch) << 32;
  }
  unsigned long long start = 0;
  for (;;) {
    bool ready = true;
#pragma unroll
    for (int k = 0; k < kPer; ++k)
      ready &= uint32_t(w[k] >> 32) == c.epoch;
    if (ready)
      break;
    if (start == 0)
      start = globaltimer();
    else
      check_deadline(start);
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
      int const i = c.tid + k * kConsumerThreads;
      if (uint32_t(w[k] >> 32) != c.epoch)
        w[k] = ld_relaxed_gpu_u64(src + i);
    }
  }
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    int const i = c.tid + k * kConsumerThreads;
    if (i < kAttnActWords) {
      int const t = i / (kQSize / 4), j = i % (kQSize / 4);
      *reinterpret_cast<uint32_t *>(c.smem + kSmemAct + t * kActRow + j * 4) = uint32_t(w[k]);
    } else if (i < kAttnLLWords) {
      int const si = i - kAttnActWords, t = si / (kQSize / kScaleBlock),
                grp = si % (kQSize / kScaleBlock);
      c.smem[kSmemActSf + t * kActSfRow + grp] = uint8_t(w[k]);
    }
  }
  // padding token 7: unit scale
  for (int i = c.tid; i < kQSize / kScaleBlock; i += kConsumerThreads)
    c.smem[kSmemActSf + kRows * kActSfRow + i] = 0x7f;
}

__device__ void tall_phase(Ctx &c, int layer, int g) {
  int const units = gemm_units(g);
  if (c.grid - 1 - c.cta >= units)
    return;
  // Every input arrives as self-validating words, polled directly.
  if (g == kGemmO)
    load_attn_act(c, layer);
  else if (g == kGemmDown)
    load_swiglu_ll(c, layer);
  else
    load_act_ll(c, g == kGemmQkv ? 2 * layer : 2 * layer + 1);
  consumer_sync();
  // The activation scales of every group as float pairs (tokens 2 tig, 2 tig + 1; token 7 pads
  // with code 0x7f = 1.0), decoded once per phase instead of per group and unit.
  {
    int const ksf = gemm_k(g) / kScaleBlock;
    float2 *xf = reinterpret_cast<float2 *>(c.smem + kSmemXf);
    uint8_t const *s_asf = c.smem + kSmemActSf;
    for (int i = c.tid; i < 4 * ksf; i += kConsumerThreads) {
      int const tig = i / ksf, grp = i - tig * ksf;
      xf[tig * kXfStride + grp] = make_float2(ue8m0(s_asf[(2 * tig) * kActSfRow + grp]),
                                              ue8m0(s_asf[(2 * tig + 1) * kActSfRow + grp]));
    }
  }
  consumer_sync();
  int parity = 0;
  for (int u = c.grid - 1 - c.cta; u < units; u += c.grid, parity ^= 1)
    tall_unit(c, layer, g, u, parity);
  consumer_sync(); // the phase's shared memory is reused by the next phase
}

// ---------------------------------------------------------------------------
// Attention: every block row attends the same window ending at kv_len. One work
// item = one 64-key chunk for all 112 query rows; warp w < 7 owns query rows
// 16w..16w+15 (token w). Unnormalized output and (max, sum) per row go to the arena.
__device__ void attention_chunk(Ctx &c, int chunk, int keys, int first) {
  int const lane = c.lane, g = lane >> 2, tig = lane & 3;
  int const m0 = c.warp * 16;
  uint32_t const sq = smem_u32(c.smem + kSmemQ), sk = smem_u32(c.smem + kSmemK),
                 sv = smem_u32(c.smem + kSmemV);
  float s[kChunk / 8][4];
#pragma unroll
  for (int nt = 0; nt < kChunk / 8; ++nt)
    s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
  for (int ks = 0; ks < kHeadDim / 16; ++ks) {
    uint32_t qa[4];
    ldsm_x4(qa, sq + ((m0 + (lane & 15)) * kARow + ks * 16 + (lane >> 4) * 8) * 2);
#pragma unroll
    for (int np = 0; np < kChunk / 16; ++np) {
      uint32_t kb[4];
      ldsm_x4(kb, sk + ((np * 16 + (lane & 7) + (lane >> 4) * 8) * kARow + ks * 16 +
                        ((lane >> 3) & 1) * 8) *
                           2);
      mma_bf16(s[2 * np], qa, kb[0], kb[1]);
      mma_bf16(s[2 * np + 1], qa, kb[2], kb[3]);
    }
  }
  float row_max[2] = {kNegInf, kNegInf};
#pragma unroll
  for (int nt = 0; nt < kChunk / 8; ++nt) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int const key = first + nt * 8 + tig * 2 + (j & 1);
      s[nt][j] = key < keys ? s[nt][j] * kSoftmaxScaleLog2 : kNegInf;
      row_max[j >> 1] = fmaxf(row_max[j >> 1], s[nt][j]);
    }
  }
  float row_sum[2] = {0.0f, 0.0f};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    row_max[h] = fmaxf(row_max[h], __shfl_xor_sync(0xffffffffu, row_max[h], 1));
    row_max[h] = fmaxf(row_max[h], __shfl_xor_sync(0xffffffffu, row_max[h], 2));
  }
#pragma unroll
  for (int nt = 0; nt < kChunk / 8; ++nt) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      s[nt][j] = exp2f(s[nt][j] - row_max[j >> 1]);
      row_sum[j >> 1] += s[nt][j];
    }
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    row_sum[h] += __shfl_xor_sync(0xffffffffu, row_sum[h], 1);
    row_sum[h] += __shfl_xor_sync(0xffffffffu, row_sum[h], 2);
  }
  float o[kHeadDim / 8][4];
#pragma unroll
  for (int dt = 0; dt < kHeadDim / 8; ++dt)
    o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0.0f;
#pragma unroll
  for (int ks = 0; ks < kChunk / 16; ++ks) {
    uint32_t const a[4] = {pack_bf16(s[2 * ks][0], s[2 * ks][1]),
                           pack_bf16(s[2 * ks][2], s[2 * ks][3]),
                           pack_bf16(s[2 * ks + 1][0], s[2 * ks + 1][1]),
                           pack_bf16(s[2 * ks + 1][2], s[2 * ks + 1][3])};
#pragma unroll
    for (int dp = 0; dp < kHeadDim / 16; ++dp) {
      uint32_t vb[4];
      ldsm_x4_t(vb, sv + ((ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * kARow + dp * 16 +
                          (lane >> 4) * 8) *
                             2);
      mma_bf16(o[2 * dp], a, vb[0], vb[1]);
      mma_bf16(o[2 * dp + 1], a, vb[2], vb[3]);
    }
  }
  float *partial = at<float>(c.arena, kAttnPartial) + size_t(chunk) * kAttnRows * kHeadDim;
#pragma unroll
  for (int dt = 0; dt < kHeadDim / 8; ++dt) {
    int const dim = dt * 8 + tig * 2;
    *reinterpret_cast<float2 *>(partial + size_t(m0 + g) * kHeadDim + dim) =
        make_float2(o[dt][0], o[dt][1]);
    *reinterpret_cast<float2 *>(partial + size_t(m0 + g + 8) * kHeadDim + dim) =
        make_float2(o[dt][2], o[dt][3]);
  }
  if (tig == 0) {
    float *stats = at<float>(c.arena, kAttnStats) + size_t(chunk) * kAttnRows * 2;
    stats[(m0 + g) * 2] = row_max[0];
    stats[(m0 + g) * 2 + 1] = row_sum[0];
    stats[(m0 + g + 8) * 2] = row_max[1];
    stats[(m0 + g + 8) * 2 + 1] = row_sum[1];
  }
}

__device__ __forceinline__ int64_t window_slot(Args const &a, int pos) {
  return int64_t(a.page_table[pos / kPage]) * kPage + pos % kPage;
}

// The layer's attention operands in shared memory: q of every block row after
// the Gemma q norm + NeoX RoPE on the first 64 dims (SGLang's fused kernel
// order), and the block's fresh K (k norm + RoPE) and V rows, all BF16. The raw
// rows are polled directly from the qkv LL words: warp w owns rows w, w + 8, ...
// (q rows 0-111: token * 16 + head, k rows 112-118, v rows 119-125); lane l holds
// dims 2l, 2l + 1 (word l) and 64 + 2l, 65 + 2l (word 32 + l), so the RoPE partner
// of its first two dims sits in lane l ^ 16. The norm weights and cos/sin were
// staged in shared memory before.
__device__ void build_qkv(Ctx &c, int layer) {
  constexpr int kAllRows = kAttnRows + 2 * kRows;
  constexpr int kPer = (kAllRows + kConsumerWarps - 1) / kConsumerWarps;
  uint16_t *sq = reinterpret_cast<uint16_t *>(c.smem + kSmemQ);
  uint16_t *fresh = reinterpret_cast<uint16_t *>(c.smem + kSmemFresh);
  float const *s_qkw = reinterpret_cast<float const *>(c.smem + kSmemQkW);
  float const *s_cs = reinterpret_cast<float const *>(c.smem + kSmemCs);
  uint64_t const *src = qkv_ll(c.arena, layer);
  uint32_t const tag = ll_tag(c.epoch);
  int const lane = c.lane;
  auto row_words = [&](int row) -> uint64_t const * {
    int const t = row < kAttnRows ? row / kQHeads : (row - kAttnRows) % kRows;
    int const col = row < kAttnRows ? (row % kQHeads) * kHeadDim
                                    : kQSize + (row - kAttnRows) / kRows * kHeadDim;
    return src + t * kQkvLLRowWords + col / 2;
  };
  uint64_t w[kPer][2];
#pragma unroll
  for (int b = 0; b < kPer; ++b) {
    int const row = c.warp + kConsumerWarps * b;
    if (row < kAllRows) {
      uint64_t const *p = row_words(row);
      w[b][0] = ld_relaxed_gpu_u64(p + lane);
      w[b][1] = ld_relaxed_gpu_u64(p + 32 + lane);
    } else {
      w[b][0] = w[b][1] = ll_word(0, tag);
    }
  }
  unsigned long long start = 0;
  for (;;) {
    bool ready = true;
#pragma unroll
    for (int b = 0; b < kPer; ++b)
      ready &= ll_ok(w[b][0], tag) && ll_ok(w[b][1], tag);
    if (ready)
      break;
    if (start == 0)
      start = globaltimer();
    else
      check_deadline(start);
#pragma unroll
    for (int b = 0; b < kPer; ++b) {
      int const row = c.warp + kConsumerWarps * b;
#pragma unroll
      for (int h = 0; h < 2; ++h)
        if (!ll_ok(w[b][h], tag))
          w[b][h] = ld_relaxed_gpu_u64(row_words(row) + 32 * h + lane);
    }
  }
  consumer_sync(); // the staged norm weights and cos/sin are visible
  // cos/sin index of the lane's first two dims (mod the rotary half)
  int const ci = (2 * lane) & 31;
#pragma unroll
  for (int b = 0; b < kPer; ++b) {
    int const row = c.warp + kConsumerWarps * b;
    if (row >= kAllRows)
      continue;
    if (row >= kAttnRows + kRows) { // v: stored as is
      uint32_t *dst = reinterpret_cast<uint32_t *>(fresh + (row - kAttnRows) * kHeadDim);
      dst[lane] = uint32_t(w[b][0]);
      dst[32 + lane] = uint32_t(w[b][1]);
      continue;
    }
    bool const q = row < kAttnRows;
    int const t = q ? row / kQHeads : row - kAttnRows;
    float const x0 = bf2f(uint16_t(w[b][0])), x1 = bf2f(uint16_t(w[b][0] >> 16));
    float const x2 = bf2f(uint16_t(w[b][1])), x3 = bf2f(uint16_t(w[b][1] >> 16));
    float ss = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      ss += __shfl_xor_sync(0xffffffffu, ss, mask);
    float const inv_rms = rsqrtf(ss / float(kHeadDim) + c.a.eps);
    float const *wt = s_qkw + (q ? 0 : kHeadDim);
    float const n0 = x0 * inv_rms * wt[2 * lane];
    float const n1 = x1 * inv_rms * wt[2 * lane + 1];
    float const n2 = x2 * inv_rms * wt[kHeadDim / 2 + 2 * lane];
    float const n3 = x3 * inv_rms * wt[kHeadDim / 2 + 2 * lane + 1];
    float const p0 = __shfl_xor_sync(0xffffffffu, n0, 16),
                p1 = __shfl_xor_sync(0xffffffffu, n1, 16);
    float const *cs = s_cs + t * kRotaryDim;
    float const cos0 = cs[ci], cos1 = cs[ci + 1], sin0 = cs[kRotaryDim / 2 + ci],
                sin1 = cs[kRotaryDim / 2 + ci + 1];
    // dims d < 32: n[d] cos - n[d + 32] sin; dims d + 32: n[d + 32] cos + n[d] sin
    float const o0 = lane < 16 ? n0 * cos0 - p0 * sin0 : n0 * cos0 + p0 * sin0;
    float const o1 = lane < 16 ? n1 * cos1 - p1 * sin1 : n1 * cos1 + p1 * sin1;
    uint16_t *dst = q ? sq + row * kARow : fresh + t * kHeadDim;
    reinterpret_cast<uint32_t *>(dst)[lane] = pack_bf16(o0, o1);
    reinterpret_cast<uint32_t *>(dst + kHeadDim / 2)[lane] = pack_bf16(n2, n3);
  }
  consumer_sync();
}

// Attention: every block row attends the same window ending at kv_len. One work
// item = one 64-key chunk for all 112 query rows. Window rows are read from the
// cache as it was before this call; rows that are one of the block's slots take
// the block's fresh K/V (the last block row of a slot wins, as in the cache
// write), so the result equals writing the block first and reading afterwards.
// CTA 0 writes the block's K/V to the cache.
__device__ void attention_phase(Ctx &c, int layer) {
  LayerArgs const &layer_args = c.a.layer[layer];
  int const kv_len = c.misc->kv_len;
  int const keys = kv_len < kWindow ? kv_len : kWindow;
  int const start = kv_len - keys;
  int const chunks = (keys + kChunk - 1) / kChunk;
  uint16_t *sk = reinterpret_cast<uint16_t *>(c.smem + kSmemK);
  uint16_t *sv = reinterpret_cast<uint16_t *>(c.smem + kSmemV);
  uint16_t const *fresh = reinterpret_cast<uint16_t const *>(c.smem + kSmemFresh);
  bool built = false;
  for (int chunk = c.cta; chunk < chunks; chunk += c.grid) {
    int const first = chunk * kChunk;
    for (int i = c.tid; i < kChunk * (kHeadDim / 8); i += kConsumerThreads) {
      int const row = i / (kHeadDim / 8), part = i % (kHeadDim / 8);
      if (first + row < keys) {
        size_t const slot = size_t(window_slot(c.a, start + first + row));
        cp_async_16(smem_u32(sk + row * kARow + part * 8),
                    layer_args.k_cache + slot * kHeadDim + part * 8);
        cp_async_16(smem_u32(sv + row * kARow + part * 8),
                    layer_args.v_cache + slot * kHeadDim + part * 8);
      } else {
        *reinterpret_cast<int4 *>(sk + row * kARow + part * 8) = make_int4(0, 0, 0, 0);
        *reinterpret_cast<int4 *>(sv + row * kARow + part * 8) = make_int4(0, 0, 0, 0);
      }
    }
    if (c.tid < kChunk) {
      int fix = -1;
      if (first + c.tid < keys) {
        int64_t const slot = window_slot(c.a, start + first + c.tid);
        for (int t = 0; t < kRows; ++t)
          fix = c.misc->slot_mapping[t] == slot ? t : fix;
      }
      c.misc->fix[c.tid] = fix;
    }
    if (!built) {
      // The q/k norm weights and the block rows' cos/sin do not depend on the qkv GEMM:
      // stage them before polling its words.
      float *s_qkw = reinterpret_cast<float *>(c.smem + kSmemQkW);
      float *s_cs = reinterpret_cast<float *>(c.smem + kSmemCs);
      for (int i = c.tid; i < 2 * kHeadDim + kRows * kRotaryDim; i += kConsumerThreads) {
        if (i < 2 * kHeadDim) {
          s_qkw[i] =
              1.0f + bf2f(i < kHeadDim ? layer_args.q_norm[i] : layer_args.k_norm[i - kHeadDim]);
        } else {
          int const j = i - 2 * kHeadDim, t = j / kRotaryDim;
          s_cs[j] = c.a.cos_sin[size_t(c.misc->positions[t]) * kRotaryDim + j % kRotaryDim];
        }
      }
      cp_async_wait_all();
      build_qkv(c, layer);
      if (c.cta == 0) {
        for (int i = c.tid; i < 2 * kRows * (kHeadDim / 8); i += kConsumerThreads) {
          int const k = i / (kHeadDim / 8), part = i % (kHeadDim / 8);
          int const which = k / kRows, t = k % kRows;
          int64_t const slot = c.misc->slot_mapping[t];
          if (slot >= 0 && !later_duplicate(c.misc, t))
            *(reinterpret_cast<int4 *>((which ? layer_args.v_cache : layer_args.k_cache) +
                                       size_t(slot) * kHeadDim) +
              part) = *reinterpret_cast<int4 const *>(fresh + k * kHeadDim + part * 8);
        }
      }
      built = true;
    } else {
      cp_async_wait_all();
      consumer_sync();
    }
    for (int i = c.tid; i < kChunk * (kHeadDim / 8); i += kConsumerThreads) {
      int const row = i / (kHeadDim / 8), part = i % (kHeadDim / 8);
      int const t = c.misc->fix[row];
      if (t >= 0) {
        *reinterpret_cast<int4 *>(sk + row * kARow + part * 8) =
            *reinterpret_cast<int4 const *>(fresh + t * kHeadDim + part * 8);
        *reinterpret_cast<int4 *>(sv + row * kARow + part * 8) =
            *reinterpret_cast<int4 const *>(fresh + (kRows + t) * kHeadDim + part * 8);
      }
    }
    consumer_sync();
    if (c.warp < kRows)
      attention_chunk(c, chunk, keys, first);
    consumer_sync();
  }
  if (c.tid == 0)
    red_add_release_gpu(signal(c.arena, kSigAttn, layer, c.cta), 1u);
}

// Merge the chunks of one query row, round to BF16 (the attention output), and
// quantize it for o_proj. Thread (group cg, quad q) adds chunks cg, cg + 8, ...
// for dims 4q..4q+3; the eight group sums are then added in group order.
__device__ void combine_phase(Ctx &c, int layer) {
  int const kv_len = c.misc->kv_len;
  int const keys = kv_len < kWindow ? kv_len : kWindow;
  int const chunks = (keys + kChunk - 1) / kChunk;
  float *acc_s = reinterpret_cast<float *>(c.smem + kSmemCombAcc);
  bool waited = false;
  for (int row = c.cta; row < kAttnRows; row += c.grid) {
    if (!waited)
      wait_signal(c, kSigAttn, layer, uint32_t(c.grid));
    else
      consumer_sync();
    waited = true;
    int const quad = c.tid & 31, cg = c.tid >> 5;
    float const *partial = at<float>(c.arena, kAttnPartial) + size_t(row) * kHeadDim + quad * 4;
    float4 pv[kMaxChunks / 8];
#pragma unroll
    for (int k = 0; k < kMaxChunks / 8; ++k) {
      int const ch = cg + 8 * k;
      pv[k] = ch < chunks ? __ldcg(reinterpret_cast<float4 const *>(
                                partial + size_t(ch) * kAttnRows * kHeadDim))
                          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    float const *stats = at<float>(c.arena, kAttnStats);
    if (c.tid < chunks) {
      float2 const st =
          __ldcg(reinterpret_cast<float2 const *>(stats + (size_t(c.tid) * kAttnRows + row) * 2));
      c.misc->comb_stat[c.tid][0] = st.x;
      c.misc->comb_stat[c.tid][1] = st.y;
    }
    consumer_sync();
    if (c.warp == 0) {
      float m = kNegInf;
      for (int ch = c.lane; ch < chunks; ch += 32)
        m = fmaxf(m, c.misc->comb_stat[ch][0]);
      for (int mask = 16; mask > 0; mask >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, mask));
      for (int ch = c.lane; ch < chunks; ch += 32)
        c.misc->comb_w[ch] = exp2f(c.misc->comb_stat[ch][0] - m);
      __syncwarp();
      if (c.lane == 0) {
        float l = 0.0f;
        for (int ch = 0; ch < chunks; ++ch)
          l = fmaf(c.misc->comb_stat[ch][1], c.misc->comb_w[ch], l);
        c.misc->comb_l = l;
      }
    }
    consumer_sync();
    float a4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int k = 0; k < kMaxChunks / 8; ++k) {
      int const ch = cg + 8 * k;
      if (ch < chunks) {
        float const w = c.misc->comb_w[ch];
        a4[0] = fmaf(pv[k].x, w, a4[0]);
        a4[1] = fmaf(pv[k].y, w, a4[1]);
        a4[2] = fmaf(pv[k].z, w, a4[2]);
        a4[3] = fmaf(pv[k].w, w, a4[3]);
      }
    }
    *reinterpret_cast<float4 *>(acc_s + cg * kHeadDim + quad * 4) =
        make_float4(a4[0], a4[1], a4[2], a4[3]);
    consumer_sync();
    if (c.warp == 0) {
      int const q = c.lane; // dims 4q..4q+3; a 32-dim scale group is eight lanes
      float value[4], amax = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float total = acc_s[4 * q + i];
#pragma unroll
        for (int k = 1; k < 8; ++k)
          total += acc_s[k * kHeadDim + 4 * q + i];
        value[i] = bf16r(total / c.misc->comb_l);
        amax = fmaxf(amax, fabsf(value[i]));
      }
      for (int mask = 1; mask < 8; mask <<= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
      float inv;
      int const scale = e8m0_from_amax(amax, &inv);
      int const t = row / kQHeads, head = row % kQHeads;
      uint64_t *ll = at<uint64_t>(c.arena, kAttnLL) + size_t(layer) * kAttnLLWords;
      uint64_t const tag = uint64_t(c.epoch) << 32;
      uint32_t bytes = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        bytes |= uint32_t(to_e4m3(value[i] * inv)) << (8 * i);
      st_relaxed_gpu_u64(ll + t * (kQSize / 4) + head * (kHeadDim / 4) + q, tag | bytes);
      if ((q & 7) == 0)
        st_relaxed_gpu_u64(ll + kAttnActWords + t * (kQSize / kScaleBlock) +
                               head * (kHeadDim / kScaleBlock) + q / 8,
                           tag | uint32_t(scale));
    }
    consumer_sync();
  }
}

// Latency-critical inputs go to L2 before the weight stream starts, so they never
// queue behind it: every layer's norm weights, q/k norm weights, the block's
// cos/sin rows and every layer's K/V rows of this CTA's attention chunk (the
// window and its page-table entries are the same for all layers).
__device__ void entry_prefetch(Ctx &c) {
  Args const &a = c.a;
  int const kv_len = c.misc->kv_len;
  int const keys = kv_len < kWindow ? kv_len : kWindow;
  int const start = kv_len - keys;
  if (c.warp == 0) {
    int const i = c.lane; // 32 small tensors
    if (i < 2 * kLayers) {
      LayerArgs const &layer = a.layer[i >> 1];
      bulk_prefetch_l2((i & 1) ? layer.post_norm : layer.input_norm, kHidden * 2);
    } else if (i == 2 * kLayers) {
      bulk_prefetch_l2(a.final_norm, kHidden * 2);
    } else if (i < 3 * kLayers + 1) {
      LayerArgs const &layer = a.layer[i - (2 * kLayers + 1)];
      bulk_prefetch_l2(layer.q_norm, kHeadDim * 2);
      bulk_prefetch_l2(layer.k_norm, kHeadDim * 2);
    } else if (i < 3 * kLayers + 1 + kRows) {
      bulk_prefetch_l2(a.cos_sin + size_t(c.misc->positions[i - (3 * kLayers + 1)]) * kRotaryDim,
                       kRotaryDim * 4);
    }
  } else if (c.warp == 1 && c.lane < 2 * kLayers) {
    // K/V rows of this CTA's chunk: at most two contiguous runs (pages) of slots.
    int const first = c.cta * kChunk;
    if (first < keys) {
      int const last = (first + kChunk < keys ? first + kChunk : keys) - 1;
      LayerArgs const &layer = a.layer[c.lane >> 1];
      uint16_t const *cache = (c.lane & 1) ? layer.v_cache : layer.k_cache;
      int pos = start + first;
      int const end = start + last;
      while (pos <= end) {
        int const page_end = (pos / kPage) * kPage + kPage - 1;
        int const run_end = page_end < end ? page_end : end;
        size_t const slot = size_t(a.page_table[pos / kPage]) * kPage + pos % kPage;
        bulk_prefetch_l2(cache + slot * kHeadDim, uint32_t(run_end - pos + 1) * kHeadDim * 2);
        pos = run_end + 1;
      }
    }
  }
}

__device__ void consumer_main(Ctx &c) {
  entry_prefetch(c);
  embedding_phase(c);
  consumer_sync();
  if (c.tid == 0)
    mbar_arrive(smem_u32(&c.misc->go));
  norm_phase(c, 0, c.a.layer[0].input_norm, false);
  for (int l = 0; l < kLayers; ++l) {
    tall_phase(c, l, kGemmQkv);
    attention_phase(c, l);
    combine_phase(c, l);
    tall_phase(c, l, kGemmO);
    norm_phase(c, 2 * l + 1, c.a.layer[l].post_norm, false);
    tall_phase(c, l, kGemmGateUp);
    tall_phase(c, l, kGemmDown);
    bool const last = l + 1 == kLayers;
    norm_phase(c, 2 * l + 2, last ? c.a.final_norm : c.a.layer[l + 1].input_norm, last);
  }
  // Every CTA read this call's generation at entry; the next call starts after this one ends.
  if (c.cta == 0 && c.tid == 0) {
    wait_count(at<uint32_t>(c.arena, kStartWord), c.epoch * uint32_t(c.grid));
    *reinterpret_cast<uint32_t volatile *>(c.arena + kCallWord) = c.epoch;
  }
}

__global__ void __launch_bounds__(kThreads, 1)
    draft_backbone_kernel(const __grid_constant__ Args a, const __grid_constant__ WeightMaps maps) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  // The swizzled TMA boxes need a 1024-byte aligned ring.
  uint8_t *smem = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  SmemMisc *misc = reinterpret_cast<SmemMisc *>(smem + kSmemMisc);
  uint8_t *arena = a.arena[a.rank];
  int const tid = int(threadIdx.x);
  if (tid == 0) {
    for (int i = 0; i < kRing; ++i) {
      mbar_init(smem_u32(&misc->full[i]), 1);
      mbar_init(smem_u32(&misc->empty[i]), kConsumerWarps);
    }
    mbar_init(smem_u32(&misc->go), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    misc->epoch = *reinterpret_cast<uint32_t volatile *>(arena + kCallWord) + 1u;
    misc->kv_len = *a.kv_len;
    atomicAdd(at<uint32_t>(arena, kStartWord), 1u);
  }
  if (tid < kRows) {
    misc->slot_mapping[tid] = a.slot_mapping[tid];
    misc->positions[tid] = a.positions[tid];
    misc->input_ids[tid] = a.input_ids[tid];
  }
  __syncthreads();
  if (tid >= kConsumerThreads) {
    producer(a, maps, smem, misc, int(blockIdx.x), int(gridDim.x), tid - kConsumerThreads);
    return;
  }
  Ctx c{a,   arena,    smem,     misc, misc->epoch, int(blockIdx.x), int(gridDim.x),
        tid, tid / 32, tid % 32, 0u};
  consumer_main(c);
}

// The plan state is 64-byte aligned; the maps sit at the first 128-byte boundary inside it
// (the state's address is fixed for the plan, so prepare_plan and launch find the same copy).
WeightMaps *plan_maps(void const *state) {
  uintptr_t const raw =
      reinterpret_cast<uintptr_t>(static_cast<PlanState const *>(state)->maps_raw);
  return reinterpret_cast<WeightMaps *>((raw + 127) & ~uintptr_t(127));
}
} // namespace

cudaError_t configure_kernel(int device, int *needed_ctas, int *resident_ctas) {
  cudaError_t status = cudaFuncSetAttribute(
      draft_backbone_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  if (status != cudaSuccess)
    return status;
  cudaDeviceProp properties{};
  if ((status = cudaGetDeviceProperties(&properties, device)) != cudaSuccess)
    return status;
  int per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, draft_backbone_kernel, kThreads,
                                                         kSmemBytes);
  if (status != cudaSuccess)
    return status;
  *needed_ctas = properties.multiProcessorCount;
  *resident_ctas = properties.cooperativeLaunch ? per_sm * properties.multiProcessorCount : 0;
  return cudaSuccess;
}

// The grid (one CTA per SM) and the weight tensor maps (pointers and shapes only).
cudaError_t prepare_plan(Args const &args, void *state) {
  PlanState *plan = static_cast<PlanState *>(state);
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess)
    return status;
  if ((status = cudaDeviceGetAttribute(&plan->grid, cudaDevAttrMultiProcessorCount, device)) !=
      cudaSuccess)
    return status;
  void *entry = nullptr;
  cudaDriverEntryPointQueryResult query = cudaDriverEntryPointSymbolNotFound;
  if ((status = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &entry, 12000,
                                                 cudaEnableDefault, &query)) != cudaSuccess)
    return status;
  if (entry == nullptr || query != cudaDriverEntryPointSuccess)
    return cudaErrorNotSupported;
  auto const encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(entry);
  WeightMaps *maps = plan_maps(state);
  for (int l = 0; l < kLayers; ++l) {
    LayerArgs const &layer = args.layer[l];
    for (int kind = 0; kind < kMapKinds; ++kind) {
      uint8_t const *w = kind == kMapQkv      ? layer.qkv
                         : kind == kMapO      ? layer.o
                         : kind == kMapGateUp ? layer.gate_up
                                              : layer.down;
      int const rows = kind == kMapQkv ? kQkvRows : kind == kMapGateUp ? kGateUpRows : kHidden;
      int const k = kind == kMapQkv || kind == kMapGateUp ? kHidden
                    : kind == kMapO                       ? kQSize
                                                          : kInter;
      cuuint64_t const dims[3] = {cuuint64_t(kKBlock), cuuint64_t(rows), cuuint64_t(k / kKBlock)};
      cuuint64_t const strides[2] = {cuuint64_t(k), cuuint64_t(kKBlock)};
      cuuint32_t const box[3] = {cuuint32_t(kKBlock),
                                 kind == kMapGateUp ? cuuint32_t(kPairRows) : cuuint32_t(kTallRows),
                                 kind == kMapDownTail ? cuuint32_t(kStageRow / kKBlock / 2)
                                                      : cuuint32_t(kStageRow / kKBlock)};
      cuuint32_t const elements[3] = {1, 1, 1};
      if (encode(&maps->map[l][kind], CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<uint8_t *>(w),
                 dims, strides, box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE,
                 CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
        return cudaErrorInvalidValue;
    }
  }
  return cudaSuccess;
}

cudaError_t launch(Args const &args, void const *state, cudaStream_t stream) {
  PlanState const *plan = static_cast<PlanState const *>(state);
  void *parameters[] = {const_cast<Args *>(&args), plan_maps(state)};
  return cudaLaunchCooperativeKernel(reinterpret_cast<void *>(draft_backbone_kernel),
                                     dim3(plan->grid), dim3(kThreads), parameters, kSmemBytes,
                                     stream);
}
} // namespace m31_draft
