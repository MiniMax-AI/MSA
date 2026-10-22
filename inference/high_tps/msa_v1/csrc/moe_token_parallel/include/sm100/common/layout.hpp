#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda.h>

#include "include/sm100/common/params.hpp"

// Launch geometry, private arena layout and launch arguments, shared by the
// kernel and the host plan.
namespace mega_moe::token_parallel {
constexpr int kRouted = 128, kShared = 16, kGrid = kRouted + kShared;
constexpr int kConsumers = 256, kProducers = 128, kThreads = 384;
// Dynamic shared memory of the kernel: the full opt-in 227 KiB.
constexpr int kSmemBytes = 232448;
static_assert(kRouted == 128 && kShared == 16 && kGrid == 144);
static_assert(kSmemBytes == 227 * 1024);

// Private 16 MiB arena per rank, zeroed once at workspace creation. Offset zero
// is the completion generation; every flag and record carries a call
// generation, so no per-call clearing is required.
constexpr size_t kArenaBytes = size_t(16) << 20;
constexpr size_t kInputReady = 256, kGuReady = 2048, kDownReady = 4352;
static_assert(kInputReady + size_t(T) * 16 <= kGuReady);
static_assert(kGuReady + size_t(kGrid) * 16 <= kDownReady);
constexpr size_t kSharedPartials = 843776;
static_assert(kDownReady + size_t(kGrid) * 16 <= kSharedPartials);
static_assert(kSharedPartials + size_t(T) * H * sizeof(float) <= (size_t(1) << 20));
// Activation hand-offs between CTAs of one call. They are written and read
// only inside the kernel and never exposed to the caller.
constexpr size_t kSharedInput = size_t(12) << 20;
constexpr size_t kSharedInputScale = kSharedInput + size_t(T) * H;
constexpr size_t kRoutedInput = kSharedInputScale + size_t(T) * (H / 32);
constexpr size_t kRoutedInputScale = kRoutedInput + size_t(T) * (H / 2);
constexpr size_t kSharedMid = kRoutedInputScale + size_t(T) * (H / 16);
constexpr size_t kSharedMidScale = kSharedMid + size_t(T) * I;
constexpr size_t kRoutedMid = kSharedMidScale + size_t(T) * (I / 32);
constexpr size_t kRoutedMidScale = kRoutedMid + size_t(R) * (I / 2);
constexpr size_t kActivationEnd = kRoutedMidScale + size_t(R) * (I / 16);
static_assert(kSharedInputScale % 16 == 0 && kRoutedInput % 16 == 0 &&
              kRoutedInputScale % 16 == 0 && kSharedMid % 16 == 0 && kSharedMidScale % 16 == 0 &&
              kRoutedMid % 16 == 0 && kRoutedMidScale % 16 == 0);
static_assert(kActivationEnd <= (size_t(13) << 20));

// Tensor maps encoded once per plan. All public tensors are contiguous and
// 16-byte aligned, and the private hand-off buffers live in the arena.
enum MapId {
  RoutedFc1Value,
  RoutedFc1Scale,
  SharedFc1Value,
  SharedFc2Value,
  SharedInputValue,
  SharedMidValue,
  MapCount
};
struct Maps {
  CUtensorMap map[MapCount];
  // Three K256 routed FC2 value panels concatenated into one request.
  CUtensorMap routed_fc2_full;
};
struct KernelArgs {
  Args a;
  Maps maps;
};
static_assert(sizeof(CUtensorMap) == 128);
} // namespace mega_moe::token_parallel
