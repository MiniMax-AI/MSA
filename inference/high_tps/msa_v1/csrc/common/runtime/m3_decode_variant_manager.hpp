#pragma once

#include <tvm/ffi/function.h>

#include <array>
#include <mutex>
#include <optional>

namespace mega_atten::m3_decode {

class M3DecodeVariantManager {
public:
  static M3DecodeVariantManager &instance();

  tvm::ffi::Function get_m3_decode_mega_atten_variant(bool wait_deadline, bool enable_placement);

private:
  M3DecodeVariantManager() = default;

  std::array<std::optional<tvm::ffi::Function>, 4> mega_atten_variants_;
  std::mutex mutex_;
};

} // namespace mega_atten::m3_decode
