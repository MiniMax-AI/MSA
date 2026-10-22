#pragma once

#include <cstddef>
#include <cstdint>

namespace mega_atten::m3_decode::tp4_nvls {

inline constexpr int kWorldSize = 4;
inline constexpr int kTokenCount = 8;
inline constexpr int kHiddenSize = 6144;
inline constexpr int kOutputTileChannels = 128;
inline constexpr int kOutputTileCount = kHiddenSize / kOutputTileChannels;

inline constexpr size_t kPayloadBytes = kTokenCount * kHiddenSize * sizeof(unsigned short);
inline constexpr int kPayloadBufferCount = 2;
inline constexpr size_t kControlOffset = kPayloadBufferCount * kPayloadBytes;
inline constexpr size_t kControlBytes = 4096;

inline constexpr int kOutputProjectionSystemEntryWord = 0;
inline constexpr int kOutputProjectionGpuEntryWord = 1;
inline constexpr int kOutputProjectionCtaEntryWord = 16;
// Words [64, 607] are owned by Shared Fusion Control and the dynamic
// scheduler. Routed Expert FC2 synchronization lives after both regions.
inline constexpr int kRoutedExpertFc2GroupEntryWord = 688;
inline constexpr int kRoutedExpertFc2GroupLocalWord = 700;

inline constexpr size_t kNextHiddenOffset = kControlOffset + kControlBytes;
inline constexpr size_t kNextHiddenBytes = kPayloadBytes;
// Fused Indexer score scratch is internal to the one-kernel path. It follows
// next_hidden so the existing TP4 payload/control ABI remains unchanged.
inline constexpr int kIndexerRows = 8;
inline constexpr int kIndexerMaxPages = 8192;
inline constexpr size_t kIndexerScoresOffset = kNextHiddenOffset + kNextHiddenBytes;
inline constexpr size_t kIndexerScoresBytes = kIndexerRows * kIndexerMaxPages * sizeof(float);
inline constexpr size_t kIndexerControlOffset = kIndexerScoresOffset + kIndexerScoresBytes;
inline constexpr size_t kIndexerControlBytes = 256;
// Legacy operators keep their existing control region. MegaAttention owns
// a separate monotonic counter so Graph replays need neither grid IDs nor resets.
struct alignas(128) RuntimeIndexerControl {
  uint64_t arrivals;
  // Keep the workspace layout stable; invocation epochs derive from arrivals.
  alignas(128) uint64_t reserved[144];
};
inline constexpr size_t kRuntimeIndexerControlOffset = kIndexerControlOffset + kIndexerControlBytes;
inline constexpr size_t kIndexerWorkspaceBytes =
    (kRuntimeIndexerControlOffset + sizeof(RuntimeIndexerControl) + 127) / 128 * 128;
inline constexpr size_t kMinimumWorkspaceBytes = kNextHiddenOffset + kNextHiddenBytes;

static_assert(kPayloadBytes == 98304);
static_assert(kControlOffset == 196608);
static_assert(kNextHiddenOffset == 200704);
static_assert(kMinimumWorkspaceBytes == 299008);
static_assert(kIndexerScoresOffset == 299008);
static_assert(kIndexerControlOffset == 561152);
static_assert(sizeof(RuntimeIndexerControl) == 1280);
static_assert(kIndexerWorkspaceBytes == 562688);
static_assert(kRoutedExpertFc2GroupEntryWord + 2 <= kRoutedExpertFc2GroupLocalWord);
static_assert((kRoutedExpertFc2GroupLocalWord + 1) * sizeof(unsigned int) <= kControlBytes);
static_assert((kNextHiddenOffset % 16) == 0);
static_assert((kIndexerScoresOffset % 128) == 0);
static_assert((kIndexerControlOffset % 128) == 0);

} // namespace mega_atten::m3_decode::tp4_nvls
