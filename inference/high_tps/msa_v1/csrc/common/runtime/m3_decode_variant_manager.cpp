#include "common/runtime/m3_decode_variant_manager.hpp"

#include <torch/torch.h>

namespace mega_atten::m3_decode {

M3DecodeVariantManager &M3DecodeVariantManager::instance() {
  static auto *manager = new M3DecodeVariantManager();
  return *manager;
}

tvm::ffi::Function M3DecodeVariantManager::get_m3_decode_mega_atten_variant(bool wait_deadline,
                                                                            bool enable_placement) {
  int const index = (wait_deadline ? 1 : 0) + (enable_placement ? 2 : 0);
  std::lock_guard<std::mutex> lock(mutex_);
  if (!mega_atten_variants_[index].has_value()) {
    auto jit_fn = tvm::ffi::Function::GetGlobalRequired(
        "msa_v1_m3_decode.jit_get_m3_decode_mega_atten_variant");
    mega_atten_variants_[index] =
        jit_fn(wait_deadline, enable_placement).cast<tvm::ffi::Function>();
  }
  return mega_atten_variants_[index].value();
}

} // namespace mega_atten::m3_decode
