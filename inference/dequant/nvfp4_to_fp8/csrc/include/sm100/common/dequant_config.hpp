#pragma once

#include <cstdint>

namespace minimax::inference::dequant::sm100 {

constexpr int64_t kHeadDim = 128;
constexpr int64_t kPackedHeadDim = kHeadDim / 2;
constexpr int64_t kScaleGroups = kHeadDim / 16;
constexpr int kThreads = 256;
constexpr int kWarpsPerBlock = kThreads / 32;
constexpr int kRowsPerWarp = 4;
constexpr int kRowsPerBlock = kWarpsPerBlock * kRowsPerWarp;

} // namespace minimax::inference::dequant::sm100
