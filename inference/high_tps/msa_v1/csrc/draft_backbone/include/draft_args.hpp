#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

// Shapes, operands and entry points shared by the host plan and the kernel. The
// layout inside the private arena and the plan state belongs to the kernel.
namespace m31_draft {
constexpr int kLayers = 5;
constexpr int kTp = 4;
constexpr int kRows = 7; // one request's draft block
constexpr int kHidden = 6144;
constexpr int kHeadDim = 128;
constexpr int kRotaryDim = 64;
constexpr int kQHeads = 16;                     // per rank; one KV head per rank
constexpr int kQSize = kQHeads * kHeadDim;      // 2048
constexpr int kQkvRows = kQSize + 2 * kHeadDim; // 2304: q | k | v
constexpr int kInter = 3072;                    // per rank
constexpr int kGateUpRows = 2 * kInter;         // gate | up
constexpr int kWindow = 4096;                   // keys a row attends: window_left 4095
constexpr int kPage = 128;
constexpr int kScaleBlock = 32;
constexpr float kSwigluAlpha = 1.702f;
constexpr float kSwigluLimit = 7.0f;
// Private arena per rank, zeroed once at workspace creation and mapped by every
// peer. Nothing in it is cleared per call.
constexpr size_t kArenaBytes = size_t(32) << 20;

struct LayerArgs {
  uint16_t const *input_norm;          // bf16 [kHidden]: raw Gemma w (applied as 1 + w in FP32)
  uint16_t const *post_norm;           // bf16 [kHidden]: raw Gemma w
  uint16_t const *q_norm;              // bf16 [kHeadDim]: raw Gemma w
  uint16_t const *k_norm;              // bf16 [kHeadDim]: raw Gemma w
  uint8_t const *qkv, *qkv_sf;         // e4m3 [kQkvRows][kHidden], ue8m0 [kQkvRows][kHidden / 32]
  uint8_t const *o, *o_sf;             // [kHidden][kQSize], [kHidden][kQSize / 32]
  uint8_t const *gate_up, *gate_up_sf; // [kGateUpRows][kHidden], [kGateUpRows][kHidden / 32]
  uint8_t const *down, *down_sf;       // [kHidden][kInter], [kHidden][kInter / 32]
  uint16_t *k_cache, *v_cache;         // bf16 [slots][kHeadDim]: slot = page * 128 + row
};

struct Args {
  LayerArgs layer[kLayers];
  int64_t const *input_ids;    // [kRows]: the block's tokens
  uint16_t const *embed;       // bf16 [vocab_rows][kHidden]: this rank's vocab shard
  uint16_t *out;               // bf16 [kRows][kHidden]: final-normed hidden
  uint16_t const *final_norm;  // bf16 [kHidden]: raw Gemma w
  int64_t const *positions;    // [kRows]
  int64_t const *slot_mapping; // [kRows]: cache slot of each block row, -1 skips the write
  int32_t const *page_table;   // [max_pages]: page of every 128 cached tokens
  int32_t const *kv_len;       // [1]: cached tokens including the block
  float const *cos_sin;        // f32 [positions][kRotaryDim]: cos | sin
  uint8_t *arena[kTp];         // every rank's arena, this rank's own at [rank]
  int64_t vocab_start;         // first token id of this rank's shard
  int64_t vocab_rows;          // tokens in this rank's shard
  int rank;
  int max_pages;
  float eps;
};

// Per-plan launch metadata (for example tensor maps), 64-byte aligned and zeroed
// before prepare_plan fills it.
constexpr size_t kPlanStateBytes = 16384;

// Prepare the kernel's attributes; reports the co-resident CTA count it needs
// and the device can hold (0 without cooperative launch support).
cudaError_t configure_kernel(int device, int *needed_ctas, int *resident_ctas);
// Once per plan, outside capture: derive launch metadata from the bound pointers
// and shapes only. Tensor contents may change between plan creation and every call.
cudaError_t prepare_plan(Args const &args, void *state);
// Enqueue one complete draft backbone on the stream.
cudaError_t launch(Args const &args, void const *state, cudaStream_t stream);
} // namespace m31_draft
