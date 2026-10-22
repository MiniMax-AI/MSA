#pragma once

#include "common/kernel/include/tp4_nvls_layout.hpp"

namespace mega_atten::m3_decode::tp4_mailbox {

inline constexpr int kWorldSize = tp4_nvls::kWorldSize;
inline constexpr int kTokenCount = tp4_nvls::kTokenCount;
inline constexpr int kHiddenSize = tp4_nvls::kHiddenSize;
inline constexpr int kGenerationCount = 3;
inline constexpr int kClusterCtas = 8;
inline constexpr unsigned short kEmpty = 0x8000;
// Keep the shared workspace prefix available to the Indexer and MoE paths.
inline constexpr size_t kPayloadOffset = tp4_nvls::kIndexerWorkspaceBytes;
inline constexpr size_t kPayloadBytes = kGenerationCount * kWorldSize * tp4_nvls::kPayloadBytes;
inline constexpr size_t kPublishEpochOffset = kPayloadOffset + kPayloadBytes;
inline constexpr size_t kConsumeEpochOffset =
    kPublishEpochOffset + tp4_nvls::kOutputTileCount * sizeof(unsigned int);
inline constexpr size_t kWorkspaceBytes =
    kConsumeEpochOffset + kTokenCount * kClusterCtas * sizeof(unsigned int);

static_assert(kPayloadOffset % 128 == 0);
static_assert(kPublishEpochOffset % 16 == 0);
static_assert(kConsumeEpochOffset % 16 == 0);

} // namespace mega_atten::m3_decode::tp4_mailbox
