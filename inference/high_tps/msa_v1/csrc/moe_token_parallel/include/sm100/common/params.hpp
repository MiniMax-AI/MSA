#pragma once
#include <cstddef>
#include <cstdint>

namespace mega_moe {
// MiniMax-M3 decode MoE on TP4: eight tokens, hidden 6144, local intermediate
// 768, 128 routed experts with top-4 routing.
constexpr int T = 8, H = 6144, I = 768, E = 128, K = 4, R = T * K, WORLD = 4;
// FC1 rows: [gate I rows, up I rows]. Scales are logical row-major arrays.
struct Args {
  // The caller supplies the normalized BF16 hidden state. Never renormalize.
  const uint16_t *post_attention_norm, *attention_residual;
  const float *router_weight, *router_bias;
  const uint8_t *shared_fc1, *shared_fc1_sf, *shared_fc2, *shared_fc2_sf;
  const uint8_t *routed_fc1, *routed_fc1_sf, *routed_fc2, *routed_fc2_sf;
  const float *fc1_alpha, *fc2_alpha;
  // Activation hand-offs inside the private arena of this rank.
  uint8_t *shared_input, *shared_input_sf, *routed_input, *routed_input_sf;
  uint8_t *shared_mid, *shared_mid_sf, *routed_mid, *routed_mid_sf;
  uint16_t *out;
  // Private arenas of all four ranks; the first word of each is the
  // completion generation of its rank.
  uint32_t *ready[WORLD];
  int32_t rank;
};
} // namespace mega_moe
