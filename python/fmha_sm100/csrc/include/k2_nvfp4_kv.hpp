// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>

#ifndef MSA_NVFP4_KV_MODE
#define MSA_NVFP4_KV_MODE 0
#endif

namespace k2 {

enum class KvMode : int { Off = 0, WgOnly = 1, Passthru = 2, Nvfp4 = 3 };

static constexpr int    kModeInt  = MSA_NVFP4_KV_MODE;
static constexpr KvMode kMode     = static_cast<KvMode>(MSA_NVFP4_KV_MODE);

static constexpr bool kHasDequantWarpgroup = (kModeInt >= 1);
static constexpr bool kDequantProducesKV   = (kModeInt >= 2);
static constexpr bool kIngressIsNvfp4      = (kModeInt >= 3);

static constexpr int kNumWarpsDequant   = kHasDequantWarpgroup ? 4 : 0;
static constexpr int kNumThreadsDequant = kNumWarpsDequant * 32;
static constexpr int kNumActiveWarpsDequant =
    kHasDequantWarpgroup ? 4 : 0;
static_assert(kNumActiveWarpsDequant <= kNumWarpsDequant,
              "Active dequant warps cannot exceed allocated dequant warps");
static_assert(kNumWarpsDequant == 0 || kNumActiveWarpsDequant == 1 ||
              kNumActiveWarpsDequant == 2 || kNumActiveWarpsDequant == 4 ||
              kNumActiveWarpsDequant == 8,
              "k1::dequant_page_tile is instantiated for kNumWarps in {1,2,4,8}");

static constexpr int kStageBytesNvfp4   = 8192 + 1024;
static constexpr int kStageBytesPassthru = 16384;
static constexpr int kStageBytes = kIngressIsNvfp4 ? kStageBytesNvfp4 : kStageBytesPassthru;

static constexpr int kRingDepth      = kIngressIsNvfp4 ? 8 : 4;
static constexpr int kStageCountKvK2 = kIngressIsNvfp4 ? 6   : 4;

struct alignas(8) StageTag {
  uint32_t w0;   // bit 0 : is_v.  bits 1..31 : seq, the CTA-local tile counter.
  uint32_t w1;   // physical_page, for debugging and for the mode-3 gmem address.

  __host__ __device__ __forceinline__ bool     is_v() const { return (w0 & 1u) != 0u; }
  __host__ __device__ __forceinline__ uint32_t seq () const { return w0 >> 1; }
  __host__ __device__ __forceinline__ uint32_t page() const { return w1; }

  __host__ __device__ __forceinline__ static StageTag make(bool is_v, uint32_t seq,
                                                           uint32_t page) {
    return StageTag{(seq << 1) | (is_v ? 1u : 0u), page};
  }
};

struct Nvfp4KvViews {
  const uint8_t* k_data  = nullptr;
  const uint8_t* k_scale = nullptr;
  const uint8_t* v_data  = nullptr;
  const uint8_t* v_scale = nullptr;
  int64_t page_stride  = 0;                       // bytes between consecutive PHYSICAL pages
  int head_stride_data  = 8192;               // bytes between kv heads, data region
  int head_stride_scale = 1024;               // bytes between kv heads, scale region

  __host__ __device__ __forceinline__
  const uint8_t* data(bool is_v, int page, int head) const {
    return (is_v ? v_data : k_data) + (size_t)page * page_stride + (size_t)head * head_stride_data;
  }
  __host__ __device__ __forceinline__
  const uint8_t* scale(bool is_v, int page, int head) const {
    return (is_v ? v_scale : k_scale) + (size_t)page * page_stride + (size_t)head * head_stride_scale;
  }
};

#ifndef MSA_NVFP4_KV_CHECK
#define MSA_NVFP4_KV_CHECK 0
#endif

#if MSA_NVFP4_KV_CHECK
#define K2_STAGE_ASSERT(cond, ...)                                              \
  do {                                                                          \
    if (!(cond)) {                                                              \
      printf("K2 STAGE-IDENTITY FAILURE blk=%d tid=%d: " __VA_ARGS__);          \
      __trap();                                                                 \
    }                                                                           \
  } while (0)
#else
#define K2_STAGE_ASSERT(cond, ...) do { } while (0)
#endif

}  // namespace k2
