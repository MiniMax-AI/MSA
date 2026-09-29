// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

namespace cutlass::fmha::kernel {

template <auto kTag, typename Default, typename... Options>
struct find_option;

template <auto kTag, typename Default>
struct find_option<kTag, Default> {
  using option_value = Default;
};

template <auto kTag, typename Default, typename Option, typename... Options>
struct find_option<kTag, Default, Option, Options...>
    : std::conditional_t<Option::tag == kTag,
                         Option,
                         find_option<kTag, Default, Options...>> {};

template <auto kTag, typename Default, typename... Options>
using find_option_t =
    typename find_option<kTag, Default, Options...>::option_value;

enum class Tag {
  kIsPersistent,
  kNumMmaWarpGroups,
  kLoadsQSeparately,
  kIsMainloopLocked,
  kIsEpilogueLocked,
  kStagesQ,
  kStagesKV,
  kEpilogueKind,
  kBlocksPerSM,
  kClusterM,
  kAccQK
};

template <auto kTag, class Value>
struct Option {
  static constexpr auto tag = kTag;
  using option_value = Value;
};

}  // namespace cutlass::fmha::kernel
