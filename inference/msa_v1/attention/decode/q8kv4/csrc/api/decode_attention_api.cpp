// C++ implementation of the Q8KV4 decode attention API.
//
// Kernel calls go through tvm_ffi; tensor management uses ATen (libtorch).

#include "decode_attention_api.hpp"

#include <ATen/DLConvertor.h>
#include <ATen/cuda/CUDAContext.h>
#include <Python.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace minimax::msa_v1::attention::decode::q8kv4 {

// ============================================================================
// Initialization — ensures tvm_ffi functions are registered.
// Called lazily on first API use. Safe from any context.
// ============================================================================

static std::once_flag g_init_flag;
static bool g_initialized = false;

void initialize() {
  std::call_once(g_init_flag, []() {
    if (!Py_IsInitialized()) {
      Py_Initialize();
    }
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject *modules = PySys_GetObject("modules");
    constexpr char kModuleName[] = "inference.msa_v1.attention.decode.q8kv4";
    if (!modules || !PyDict_GetItemString(modules, kModuleName)) {
      int const status = PyRun_SimpleString("import inference.msa_v1.attention.decode.q8kv4");
      TORCH_CHECK(status == 0, "failed to import inference.msa_v1.attention.decode.q8kv4");
    }
    PyGILState_Release(gstate);
    g_initialized = true;
  });
}

static inline void ensure_initialized() {
  if (!g_initialized)
    initialize();
}

static void fix_size_one_strides(DLManagedTensor *dl) {
  if (!dl || !dl->dl_tensor.strides)
    return;
  for (int i = 0; i < dl->dl_tensor.ndim; ++i) {
    if (dl->dl_tensor.shape[i] == 1) {
      int64_t compact = 1;
      for (int j = i + 1; j < dl->dl_tensor.ndim; ++j)
        compact *= dl->dl_tensor.shape[j];
      dl->dl_tensor.strides[i] = compact;
    }
  }
}

static tvm::ffi::Tensor torch_to_tvm(at::Tensor t) {
  bool native_fp8 = (t.scalar_type() == at::kFloat8_e4m3fn);
  if (native_fp8)
    t = t.view(at::kByte);
  auto *dl = at::toDLPack(t);
  fix_size_one_strides(dl);
  auto tvm_t = tvm::ffi::Tensor::FromDLPack(dl);
  if (native_fp8) {
    const_cast<DLTensor *>(tvm_t.GetDLTensorPtr())->dtype = {10, 8, 1};
  }
  return tvm_t;
}

static tvm::ffi::Tensor optional_to_tvm(const std::optional<at::Tensor> &t) {
  if (!t.has_value() || !t->defined())
    return tvm::ffi::Tensor(nullptr);
  return torch_to_tvm(*t);
}

static tvm::ffi::Tensor tensor_or_null(const at::Tensor &t) {
  if (!t.defined())
    return tvm::ffi::Tensor(nullptr);
  return torch_to_tvm(t);
}

static at::Tensor allocate_plan_buffer(int64_t size, int device, at::ScalarType dtype) {
  return torch::empty({size}, torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device));
}

template <typename T>
static at::Tensor copy_typed_vector_to_gpu(const std::vector<T> &source, at::ScalarType dtype,
                                           int device,
                                           std::vector<at::Tensor> &host_staging_buffers) {
  int64_t size = static_cast<int64_t>(source.size());
  auto host = torch::empty({size}, torch::TensorOptions().dtype(dtype).pinned_memory(true));
  std::memcpy(host.data_ptr(), source.data(), size * sizeof(T));
  auto gpu = allocate_plan_buffer(size, device, dtype);
  gpu.copy_(host, /*non_blocking=*/true);
  host_staging_buffers.push_back(std::move(host));
  return gpu;
}

static at::Tensor copy_vector_to_gpu(const std::vector<int32_t> &source, int device,
                                     std::vector<at::Tensor> &host_staging_buffers) {
  return copy_typed_vector_to_gpu<int32_t>(source, torch::kInt32, device, host_staging_buffers);
}

// ============================================================================
// Balanced (stream-K style) schedule for uniform TopK work items
// ============================================================================
//
// A work item is one (batch, q token, KV head) triple covering up to TopK pages, one kernel tile
// per page; a request with fewer pages than TopK (or a causally trimmed MTP token) covers fewer.
// Items that fill whole waves are assigned round-robin, one CTA per SM. Every item of the
// fractional tail wave is cut into the same number of equal KV segments over its valid tiles, one
// segment per CTA and no CTA spanning two items, so the tail is spread over the machine instead
// of running as a second wave of whole items. The segments of an item are numbered in KV order
// and merged in that order, which keeps the result deterministic. Ranges are in kernel tiles; the
// worklist format is the device plan kernel's.
//
// The kernel stages every slot of an item in shared memory for the merge, so an item is cut into
// at most kStreamKMergeMaxSlots segments (Traits::kMergeMaxSlots); a slot is the 16 x 256 B
// partial O block plus 16 LSE floats (Traits::kMergeSlotBytes).
constexpr int kStreamKMergeMaxSlots = 8;
constexpr int64_t kStreamKMergeBytesPerHead = 128 * sizeof(at::BFloat16) + sizeof(float);

struct StreamKSchedule {
  std::vector<int64_t> work_range; // [num_ctas] packed (begin, end) into the work arrays
  std::vector<int64_t> work_info;  // packed (q tile, head, batch) per work
  std::vector<int32_t> kv_begin, kv_end, split_idx; // per work, kernel-tile units
  std::vector<int32_t> split_count;                 // per work, segments of its item
  int max_splits = 1;
  int first_tail_item = 0;
  int tail_items = 0;
};

static inline uint64_t pack_work_info_host(int qo_tile, int head, int batch) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(qo_tile)) << 32) |
         (static_cast<uint64_t>(head & 0xffff) << 16) | static_cast<uint64_t>(batch & 0xffff);
}

static inline uint64_t pack_work_range_host(int begin, int end) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(end)) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(begin));
}

static StreamKSchedule build_stream_k_schedule(const std::vector<int32_t> &packed_qo_lens,
                                               const std::vector<int32_t> &kv_lens, int num_heads,
                                               int num_ctas, int kv_iters, int pack_factor,
                                               int page_size, int topk_pages, int kv_tile_size) {
  // Kernel tiles per item (one per page) and per plan tile.
  int const kernel_tile = 128;
  int const tiles_per_plan_tile = kv_tile_size / kernel_tile;
  int const tiles_per_item = kv_iters * tiles_per_plan_tile;
  struct Item {
    int batch, q_tile, head, tiles;
  };
  struct Work {
    int item, begin, end, split;
  };
  std::vector<Item> items;
  for (int batch = 0; batch < static_cast<int>(packed_qo_lens.size()); ++batch) {
    int const q_tiles = (packed_qo_lens[batch] + pack_factor - 1) / pack_factor;
    for (int q_tile = 0; q_tile < q_tiles; ++q_tile) {
      // Valid tiles of the item as the kernel computes them: the token's causally visible KV
      // length, in pages, capped at TopK.
      int const visible = std::max(0, kv_lens[batch] - ((q_tiles - 1) - q_tile));
      int const pages = std::min((visible + page_size - 1) / page_size, topk_pages);
      int const tiles = std::min(tiles_per_item, std::max(1, pages * page_size / kernel_tile));
      for (int head = 0; head < num_heads; ++head)
        items.push_back({batch, q_tile, head, tiles});
    }
  }
  int const num_items = static_cast<int>(items.size());
  int const full_waves = num_items / num_ctas;
  int const first_tail_item = full_waves * num_ctas;
  int const tail_items = num_items - first_tail_item;

  // Tail wave: every tail item is cut into the same number of equal KV ranges over its valid
  // tiles, one piece per CTA, as many pieces as the free CTAs allow (at most the merge staging
  // capacity), so no CTA crosses an item boundary inside the tail (a crossing costs a pipeline
  // drain and refill). With one piece the tail is a plain wave and the caller falls back.
  std::vector<std::vector<Work>> per_cta(num_ctas);
  std::vector<int> segments(tail_items, 0);
  int const pieces =
      tail_items > 0
          ? std::max(1, std::min({tiles_per_item, kStreamKMergeMaxSlots, num_ctas / tail_items}))
          : 0;
  bool const split_tail = pieces > 1;
  for (int local = 0; split_tail && local < tail_items; ++local) {
    int const tiles = items[first_tail_item + local].tiles;
    int const item_pieces = std::min(pieces, tiles);
    for (int piece = 0; piece < item_pieces; ++piece) {
      int const begin = (piece * tiles) / item_pieces;
      int const end = ((piece + 1) * tiles) / item_pieces;
      if (end <= begin)
        continue;
      per_cta[local * pieces + piece].push_back(
          {first_tail_item + local, begin, end, segments[local]++});
    }
  }
  if (!split_tail) {
    for (int local = 0; local < tail_items; ++local) {
      per_cta[(first_tail_item + local) % num_ctas].push_back(
          {first_tail_item + local, 0, tiles_per_item, segments[local]++});
    }
  }
  // Whole items first: the tail piece and its merge run at the end of the CTA.
  for (int item = 0; item < first_tail_item; ++item) {
    std::vector<Work> &works = per_cta[item % num_ctas];
    Work const whole{item, 0, tiles_per_item, 0};
    size_t insert_at = 0;
    while (insert_at < works.size() && works[insert_at].item < first_tail_item)
      ++insert_at;
    works.insert(works.begin() + insert_at, whole);
  }

  StreamKSchedule schedule;
  schedule.first_tail_item = first_tail_item;
  schedule.tail_items = tail_items;
  schedule.work_range.resize(num_ctas);
  int offset = 0;
  for (int cta = 0; cta < num_ctas; ++cta) {
    int const begin = offset;
    for (Work const &work : per_cta[cta]) {
      Item const &item = items[work.item];
      schedule.work_info.push_back(pack_work_info_host(item.q_tile, item.head, item.batch));
      schedule.kv_begin.push_back(work.begin);
      schedule.kv_end.push_back(work.end);
      schedule.split_idx.push_back(work.split);
      schedule.split_count.push_back(
          work.item >= first_tail_item ? std::max(1, segments[work.item - first_tail_item]) : 1);
      ++offset;
    }
    schedule.work_range[cta] = pack_work_range_host(begin, offset);
  }
  for (int count : segments) {
    schedule.max_splits = std::max(schedule.max_splits, count);
  }
  return schedule;
}

static int get_runtime_sm_count(int device) {
  if (device < 0) {
    cudaError_t cur_err = cudaGetDevice(&device);
    TORCH_CHECK(cur_err == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(cur_err));
  }
  int sm_count = 0;
  cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  TORCH_CHECK(err == cudaSuccess,
              "cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount) failed for device ", device,
              ": ", cudaGetErrorString(err));
  return sm_count;
}

static std::unordered_map<int, int> g_num_cta;
static std::mutex g_num_cta_mutex;

static int get_num_cta(int device) {
  std::lock_guard<std::mutex> lock(g_num_cta_mutex);
  auto [it, inserted] = g_num_cta.try_emplace(device, -1);
  if (inserted)
    it->second = get_runtime_sm_count(device);
  return it->second;
}

static std::vector<int32_t> tensor_to_vec_i32(const at::Tensor &t) {
  auto cpu = t.cpu().contiguous().to(torch::kInt32);
  auto ptr = cpu.data_ptr<int32_t>();
  return std::vector<int32_t>(ptr, ptr + cpu.numel());
}

// ============================================================================
// VariantManager — loads and caches JIT-compiled kernel .so files
// ============================================================================
class VariantManager {
public:
  static VariantManager &instance() {
    static auto *mgr = new VariantManager();
    return *mgr;
  }

  tvm::ffi::Function get_plan_fn(int device) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = plan_fn_cache_.find(device);
    if (it != plan_fn_cache_.end())
      return it->second;
    auto jit_fn =
        tvm::ffi::Function::GetGlobalRequired("msa_v1.attention.decode.q8kv4.jit_get_plan");
    auto fn = jit_fn((int64_t)device).cast<tvm::ffi::Function>();
    plan_fn_cache_[device] = fn;
    return fn;
  }
  tvm::ffi::Function get_reduction_fn(int device) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = reduction_fn_cache_.find(device);
    if (it != reduction_fn_cache_.end())
      return it->second;
    auto jit_fn =
        tvm::ffi::Function::GetGlobalRequired("msa_v1.attention.decode.q8kv4.jit_get_reduction");
    auto fn = jit_fn((int64_t)device).cast<tvm::ffi::Function>();
    reduction_fn_cache_[device] = fn;
    return fn;
  }
  tvm::ffi::Function get_fmha_fwd_sparse_variant(int topk, bool split_kv, int device,
                                                 int gqa_ratio, bool output_mxfp8) {
    uint64_t const split_key = split_kv ? 1 : 0;
    uint64_t const key = (static_cast<uint64_t>(device) << 33) |
                         (static_cast<uint64_t>(gqa_ratio) << 17) |
                         (static_cast<uint64_t>(topk) << 2) |
                         (static_cast<uint64_t>(output_mxfp8) << 1) | split_key;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = fmha_fwd_sparse_variant_cache_.find(key);
      if (it != fmha_fwd_sparse_variant_cache_.end())
        return it->second;
    }
    auto jit_fn = tvm::ffi::Function::GetGlobalRequired(
        "msa_v1.attention.decode.q8kv4.jit_get_fmha_fwd_sparse_variant");
    auto fn = jit_fn((int64_t)topk, split_kv, (int64_t)device, (int64_t)gqa_ratio,
                     output_mxfp8 ? "mxfp8" : "bf16")
                  .cast<tvm::ffi::Function>();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      fmha_fwd_sparse_variant_cache_[key] = fn;
    }
    return fn;
  }

private:
  VariantManager() {};

  std::unordered_map<int, tvm::ffi::Function> plan_fn_cache_;
  std::unordered_map<int, tvm::ffi::Function> reduction_fn_cache_;
  std::unordered_map<uint64_t, tvm::ffi::Function> fmha_fwd_sparse_variant_cache_;

  std::mutex mutex_;
};

static void call_plan(at::Tensor qo_segment_offsets, at::Tensor qo_segment_lens,
                      at::Tensor kv_segment_lens, at::Tensor packed_work_range,
                      at::Tensor packed_work_info, int qo_tile_size, int kv_tile_size,
                      int num_qo_heads, int num_ctas, bool causal,
                      const std::optional<at::Tensor> &qo_offset, int num_kv_splits,
                      at::Tensor kv_tile_begin_indices, at::Tensor kv_tile_end_indices,
                      at::Tensor kv_split_indices, at::Tensor num_kv_splits_per_row,
                      at::Tensor workspace_lse, int64_t lse_total_size, int pack_factor,
                      int64_t stream_int, int device) {
  auto &mgr = VariantManager::instance();
  auto plan_fn = mgr.get_plan_fn(device);

  plan_fn(torch_to_tvm(qo_segment_offsets), torch_to_tvm(qo_segment_lens),
          torch_to_tvm(kv_segment_lens), torch_to_tvm(packed_work_range),
          torch_to_tvm(packed_work_info), (int64_t)qo_tile_size, (int64_t)kv_tile_size,
          (int64_t)num_qo_heads, (int64_t)num_ctas, causal, optional_to_tvm(qo_offset),
          (int64_t)num_kv_splits, tensor_or_null(kv_tile_begin_indices),
          tensor_or_null(kv_tile_end_indices), tensor_or_null(kv_split_indices),
          tensor_or_null(num_kv_splits_per_row), stream_int, tensor_or_null(workspace_lse),
          lse_total_size, (int64_t)pack_factor);
}

PlanInfo _make_decode_plan_impl(at::Tensor qo_segment_lens, at::Tensor kv_segment_lens,
                                int num_qo_heads, int num_kv_heads,
                                std::optional<at::Tensor> qo_offset, int num_kv_splits,
                                int page_size, int kv_block_num, int usable_SM_count,
                                std::optional<int> device_opt, const std::string &split_mode) {
  // Every query owns an independent TopK list. Plan its full sparse capacity; the attention
  // kernel applies the causal mask using the actual sequence length and query position.
  constexpr bool kPlanCausal = false;
  TORCH_CHECK(split_mode == "legacy" || split_mode == "streamk",
              "split_mode must be 'legacy' or 'streamk', got ", split_mode);
  int device = device_opt.has_value() ? device_opt.value() : at::cuda::current_device();
  c10::cuda::CUDAGuard device_guard(device);
  int64_t stream_int = reinterpret_cast<int64_t>(at::cuda::getCurrentCUDAStream().stream());

  TORCH_CHECK(!qo_segment_lens.is_cuda() && !kv_segment_lens.is_cuda(),
              "prepare metadata must be host tensors; device-to-host planning is forbidden");

  auto qo_lens_list = tensor_to_vec_i32(qo_segment_lens);
  std::vector<int32_t> qo_lens = qo_lens_list;
  int max_qo_len_orig = *std::max_element(qo_lens.begin(), qo_lens.end());

  TORCH_CHECK(max_qo_len_orig > 0,
              "decode attention requires positive query lengths; got max_qo_len=", max_qo_len_orig);
  TORCH_CHECK(page_size == 128 && kv_block_num == 16,
              "Q8KV4 sparse decode requires page_size=128 and TopK=16");
  TORCH_CHECK(num_kv_heads > 0 && num_qo_heads % num_kv_heads == 0 &&
                  (num_qo_heads / num_kv_heads == 8 || num_qo_heads / num_kv_heads == 16),
              "Q8KV4 sparse decode requires 8 or 16 Q heads per KV head");

  PlanInfo info;

  auto cute_workspace = allocate_plan_buffer(32 * 1024 * 1024, device, torch::kUInt8);

  int num_ctas = get_num_cta(device);
  if (usable_SM_count > 0)
    num_ctas = std::min(usable_SM_count, num_ctas);

  int orig_num_qo_heads = num_qo_heads;
  int const pack_factor = num_qo_heads / num_kv_heads;
  bool qo_len_uniform = *std::min_element(qo_lens.begin(), qo_lens.end()) == max_qo_len_orig;
  num_qo_heads /= pack_factor;

  auto kv_lens = tensor_to_vec_i32(kv_segment_lens);

  std::vector<int32_t> qo_off_in;
  if (qo_offset.has_value() && qo_offset->defined()) {
    qo_off_in = tensor_to_vec_i32(*qo_offset);
  } else {
    qo_off_in.resize(qo_lens.size());
    for (size_t i = 0; i < qo_lens.size(); ++i)
      qo_off_in[i] = kv_lens[i] - qo_lens[i];
  }
  for (auto &q : qo_lens)
    q *= pack_factor;
  int const packed_max_qo_len = *std::max_element(qo_lens.begin(), qo_lens.end());
  int const q_tokens_per_batch = packed_max_qo_len / pack_factor;

  // Build offsets
  int batch_len = qo_lens.size();
  std::vector<int32_t> qo_offsets(batch_len + 1);
  qo_offsets[0] = 0;
  for (int i = 0; i < batch_len; ++i)
    qo_offsets[i + 1] = qo_offsets[i] + qo_lens[i];
  std::vector<int32_t> kv_offsets(batch_len + 1);
  kv_offsets[0] = 0;
  for (int i = 0; i < batch_len; ++i)
    kv_offsets[i + 1] = kv_offsets[i] + kv_lens[i];

  auto qo_offset_list = qo_off_in;

  auto qo_segment_offsets = copy_vector_to_gpu(qo_offsets, device, info.host_staging_buffers);
  auto kv_segment_offsets = copy_vector_to_gpu(kv_offsets, device, info.host_staging_buffers);
  auto qo_offset_gpu = copy_vector_to_gpu(qo_offset_list, device, info.host_staging_buffers);
  auto kv_seg_lens_gpu = copy_vector_to_gpu(kv_lens, device, info.host_staging_buffers);

  // Plan KV lens for sparse per-token mode
  std::vector<int32_t> plan_kv_lens_list(kv_lens.size(), kv_block_num * page_size);
  std::optional<at::Tensor> plan_qo_offset = std::nullopt;

  int total_qo_len = qo_offsets.back();
  int const max_qo_len = pack_factor;
  constexpr int qo_tile_size = 128;
  constexpr int kv_tile_size = 256;
  int const plan_qo_tile_size = pack_factor;

  // Split selection is performed by prepare() from host-known q_lengths and
  // launch geometry. The C++ planner never reads device cost state to host.
  TORCH_CHECK(num_kv_splits == 1 || num_kv_splits == 2 || num_kv_splits == 4 || num_kv_splits == 8,
              "Q8KV4 sparse decode split count must be 1, 2, 4, or 8");

  auto checked_mul = [](int64_t lhs, int64_t rhs, char const *label) {
    TORCH_CHECK(lhs >= 0 && rhs >= 0 &&
                    (lhs == 0 || rhs <= std::numeric_limits<int64_t>::max() / lhs),
                label, " size overflows int64");
    return lhs * rhs;
  };

  auto qo_seg_lens_gpu = copy_vector_to_gpu(qo_lens, device, info.host_staging_buffers);

  at::Tensor packed_work_range, packed_work_info;
  at::Tensor kv_tile_begin, kv_tile_end, kv_split_idx, nkv_per_row;
  at::Tensor workspace_o, ws_lse;
  int64_t lse_total = 0;
  bool use_stream_k = false;

  if (split_mode == "streamk") {
    // The kernel clamps every segment to the pages a request actually has; a segment past the
    // end runs empty and publishes a zero partial with -inf LSEs, so the fold needs no validity
    // test.
    int const kv_iters = kv_block_num * page_size / kv_tile_size;
    StreamKSchedule schedule =
        build_stream_k_schedule(qo_lens, kv_lens, num_qo_heads, num_ctas, kv_iters, pack_factor,
                                page_size, kv_block_num, kv_tile_size);
    // A whole-wave-only schedule has nothing to merge; the direct grid is cheaper.
    use_stream_k = schedule.max_splits > 1;
    if (use_stream_k) {
      TORCH_CHECK(schedule.max_splits <= kStreamKMergeMaxSlots,
                  "stream-K schedule cut an item into ", schedule.max_splits,
                  " segments; the in-kernel merge stages at most ", kStreamKMergeMaxSlots);
      num_kv_splits = schedule.max_splits;
      info.merge_item_base = schedule.first_tail_item;
      packed_work_range = copy_typed_vector_to_gpu<int64_t>(schedule.work_range, torch::kInt64,
                                                            device, info.host_staging_buffers);
      packed_work_info = copy_typed_vector_to_gpu<int64_t>(schedule.work_info, torch::kInt64,
                                                           device, info.host_staging_buffers);
      kv_tile_begin = copy_vector_to_gpu(schedule.kv_begin, device, info.host_staging_buffers);
      kv_tile_end = copy_vector_to_gpu(schedule.kv_end, device, info.host_staging_buffers);
      kv_split_idx = copy_vector_to_gpu(schedule.split_idx, device, info.host_staging_buffers);
      info.kv_split_count =
          copy_vector_to_gpu(schedule.split_count, device, info.host_staging_buffers);
      // Only the tail wave publishes partials; full-wave items store directly to the output.
      std::vector<int32_t> counters(schedule.tail_items, 0);
      info.merge_counter = copy_vector_to_gpu(counters, device, info.host_staging_buffers);
      int64_t const merge_bytes =
          checked_mul(schedule.tail_items,
                      static_cast<int64_t>(num_kv_splits) * pack_factor * kStreamKMergeBytesPerHead,
                      "merge workspace");
      workspace_o =
          allocate_plan_buffer(merge_bytes / sizeof(at::BFloat16), device, torch::kBFloat16);
    }
    // Otherwise the caller's split choice stands and the legacy plan below is used.
  }

  if (!use_stream_k) {
    // ---- Fixed split or no-split plan ----
    packed_work_range = allocate_plan_buffer(num_ctas, device, torch::kInt64);
    int64_t query_tiles = 0;
    for (int32_t packed_q_len : qo_lens) {
      query_tiles +=
          (static_cast<int64_t>(packed_q_len) + plan_qo_tile_size - 1) / plan_qo_tile_size;
    }
    int64_t max_work_items = checked_mul(checked_mul(query_tiles, num_qo_heads, "decode worklist"),
                                         std::max(num_kv_splits, 1), "decode split worklist");
    max_work_items = std::max<int64_t>(max_work_items, 1);
    packed_work_info = allocate_plan_buffer(max_work_items, device, torch::kInt64);

    if (num_kv_splits > 1) {
      kv_tile_begin = allocate_plan_buffer(max_work_items, device, torch::kInt32);
      kv_tile_end = allocate_plan_buffer(max_work_items, device, torch::kInt32);
      kv_split_idx = allocate_plan_buffer(max_work_items, device, torch::kInt32);
      nkv_per_row = allocate_plan_buffer(total_qo_len, device, torch::kInt32);
      int64_t workspace_rows =
          checked_mul(checked_mul(total_qo_len, num_kv_splits, "decode workspace"), num_qo_heads,
                      "decode workspace heads");
      workspace_o = allocate_plan_buffer(
          checked_mul(workspace_rows, 128, "decode output workspace"), device, torch::kBFloat16);
      lse_total = workspace_rows;
      ws_lse = allocate_plan_buffer(lse_total, device, torch::kFloat32);
    }

    auto plan_kv_lens_gpu =
        copy_vector_to_gpu(plan_kv_lens_list, device, info.host_staging_buffers);

    call_plan(qo_segment_offsets, qo_seg_lens_gpu, plan_kv_lens_gpu, packed_work_range,
              packed_work_info, plan_qo_tile_size, kv_tile_size, num_qo_heads, num_ctas,
              kPlanCausal, plan_qo_offset, num_kv_splits, kv_tile_begin, kv_tile_end, kv_split_idx,
              nkv_per_row, ws_lse, lse_total, pack_factor, stream_int, device);
  }
  info.stream_k = use_stream_k;

  info.packed_work_range = packed_work_range;
  info.packed_work_info = packed_work_info;
  info.kv_tile_begin_indices = kv_tile_begin;
  info.kv_tile_end_indices = kv_tile_end;
  info.kv_split_indices = kv_split_idx;
  info.num_kv_splits = num_kv_splits;
  info.workspace_o = workspace_o;
  info.workspace_lse = ws_lse;
  info.max_qo_len = max_qo_len;
  info.qo_tile_size = qo_tile_size;
  info.num_kv_splits_per_row = nkv_per_row;
  info.qo_segment_offsets = qo_segment_offsets;
  info.kv_segment_offsets = kv_segment_offsets;
  info.qo_segment_lens = qo_seg_lens_gpu;
  info.kv_segment_lens = kv_seg_lens_gpu;
  info.qo_offset = qo_offset_gpu;
  info.pack_factor = pack_factor;
  info.orig_num_qo_heads = orig_num_qo_heads;
  info.q_tokens_per_batch = q_tokens_per_batch;
  info.qo_len_uniform = qo_len_uniform;
  info.kv_block_num = kv_block_num;
  info.cute_workspace_buffer = cute_workspace;
  return std::move(info);
}

// ============================================================================
// fmha_fwd run
// ============================================================================

at::Tensor _run_decode_impl(at::Tensor q, at::Tensor k, at::Tensor v, PlanInfo &plan,
                            at::Tensor seq_lens, at::Tensor page_table, at::Tensor topk_indices,
                            at::Tensor k_scale, at::Tensor v_scale, at::Tensor out,
                            at::Tensor out_scale, float sm_scale) {
  c10::cuda::CUDAGuard device_guard(q.device());
  int device = q.get_device();
  int64_t nnz_qo = q.size(0);
  int num_qo_heads = q.size(1);
  int head_dim_qk = q.size(2);
  int head_dim_vo = head_dim_qk;
  int num_kv_heads = k.size(1);
  int page_size = k.size(2);

  int64_t stream_int = reinterpret_cast<int64_t>(at::cuda::getCurrentCUDAStream().stream());

  int64_t qo_total_len = nnz_qo;
  int batch_size = plan.qo_segment_lens.size(0);
  TORCH_CHECK(seq_lens.is_cuda() && seq_lens.scalar_type() == at::kInt && seq_lens.dim() == 1 &&
                  seq_lens.size(0) == batch_size,
              "seq_lens must be a CUDA int32 tensor with shape [batch]");

  int pack_factor = plan.pack_factor;
  int orig_num_qo_heads = plan.orig_num_qo_heads > 0 ? plan.orig_num_qo_heads : num_qo_heads;
  if (pack_factor > 1 && plan.orig_num_qo_heads > 0) {
    num_qo_heads = orig_num_qo_heads / pack_factor;
    qo_total_len = nnz_qo * pack_factor;
  }

  int max_qo_len = plan.max_qo_len;
  int qo_tile_size = plan.qo_tile_size;

  bool use_split_kv = (plan.num_kv_splits > 1 && plan.workspace_o.defined());
  // Balanced schedule: split items merge in the kernel and whole items store directly, so the
  // reduction launch is skipped.
  bool const in_kernel_merge = plan.stream_k;
  bool const output_mxfp8 = out_scale.numel() != 0;
  TORCH_CHECK(!output_mxfp8 || !use_split_kv || in_kernel_merge,
              "MXFP8 output requires Stream-K for multiple splits");
  auto &mgr = VariantManager::instance();
  int64_t run_kv_page_stride = page_table.size(1);

  auto call_fmha_variant = [&](const tvm::ffi::Function &variant_fn, int64_t run_num_kv_splits,
                               bool in_kernel_split_kv = false,
                               bool use_uniform_full_page_kv_len = false) {
    variant_fn(
        torch_to_tvm(plan.cute_workspace_buffer), torch_to_tvm(q), torch_to_tvm(k), torch_to_tvm(v),
        torch_to_tvm(plan.qo_segment_lens), torch_to_tvm(seq_lens),
        torch_to_tvm(plan.qo_segment_offsets), torch_to_tvm(plan.kv_segment_offsets),
        torch_to_tvm(plan.packed_work_range), torch_to_tvm(plan.packed_work_info),
        torch_to_tvm(out), tensor_or_null(out_scale), (double)sm_scale, (int64_t)max_qo_len,
        tensor_or_null(plan.qo_offset),
        run_num_kv_splits,
        in_kernel_split_kv ? tvm::ffi::Tensor(nullptr) : tensor_or_null(plan.kv_tile_begin_indices),
        in_kernel_split_kv ? tvm::ffi::Tensor(nullptr) : tensor_or_null(plan.kv_tile_end_indices),
        in_kernel_split_kv ? tvm::ffi::Tensor(nullptr) : tensor_or_null(plan.kv_split_indices),
        in_kernel_split_kv ? tvm::ffi::Tensor(nullptr) : tensor_or_null(plan.workspace_o),
        in_kernel_split_kv ? tvm::ffi::Tensor(nullptr) : tensor_or_null(plan.workspace_lse),
        in_kernel_split_kv ? tvm::ffi::Tensor(nullptr) : tensor_or_null(plan.num_kv_splits_per_row),
        (int64_t)qo_tile_size, torch_to_tvm(page_table), run_kv_page_stride,
        torch_to_tvm(topk_indices), torch_to_tvm(k_scale), torch_to_tvm(v_scale),
        (int64_t)pack_factor, (int64_t)plan.q_tokens_per_batch, plan.qo_len_uniform,
        use_uniform_full_page_kv_len, tensor_or_null(plan.kv_split_count),
        tensor_or_null(plan.merge_counter), plan.merge_item_base, stream_int);
  };

  int fmha_fwd_runtime_topk = static_cast<int>(topk_indices.size(2));
  bool fmha_fwd_uniform_full_pages = false;
  bool fmha_fwd_scale_layout_ok = k_scale.dim() == 4 && v_scale.dim() == 4;
  int fmha_fwd_q_tokens = 0;
  if (pack_factor == 8 || pack_factor == 16) {
    if (plan.q_tokens_per_batch > 0) {
      fmha_fwd_q_tokens = plan.q_tokens_per_batch;
    } else {
      int64_t const packed_rows_per_batch = static_cast<int64_t>(batch_size) * pack_factor;
      if (packed_rows_per_batch > 0 && qo_total_len % packed_rows_per_batch == 0) {
        fmha_fwd_q_tokens = static_cast<int>(qo_total_len / packed_rows_per_batch);
      }
    }
  }
  bool fmha_fwd_sparse_candidate =
      (q.scalar_type() == at::kFloat8_e4m3fn || q.scalar_type() == at::kByte) && page_size == 128 &&
      head_dim_qk == 128 && head_dim_vo == 128 && (pack_factor == 8 || pack_factor == 16) &&
      fmha_fwd_q_tokens > 0 && num_kv_heads > 0 &&
      orig_num_qo_heads == num_kv_heads * pack_factor && fmha_fwd_scale_layout_ok &&
      fmha_fwd_runtime_topk == 16 && plan.kv_block_num == 16;

  int fmha_fwd_run_kv_splits = use_split_kv ? plan.num_kv_splits : 1;
  bool fmha_fwd_in_kernel_split = false;

  if (fmha_fwd_sparse_candidate) {
    auto variant_fn =
        mgr.get_fmha_fwd_sparse_variant(fmha_fwd_runtime_topk, use_split_kv, device, pack_factor,
                                        output_mxfp8);
    call_fmha_variant(variant_fn, fmha_fwd_run_kv_splits, fmha_fwd_in_kernel_split,
                      fmha_fwd_uniform_full_pages);
  } else {
    TORCH_CHECK(false, "input does not match the Q8KV4 sparse TopK16 decode domain: ",
                "sparse_candidate=", fmha_fwd_sparse_candidate, " page_size=", page_size,
                " head_dim_qk=", head_dim_qk, " head_dim_vo=", head_dim_vo,
                " pack_factor=", pack_factor, " q_tokens=", fmha_fwd_q_tokens,
                " num_kv_splits=", plan.num_kv_splits, " use_split_kv=", use_split_kv,
                " scale_layout_ok=", fmha_fwd_scale_layout_ok, " topk=", fmha_fwd_runtime_topk,
                " hq=", orig_num_qo_heads, " hk=", num_kv_heads);
  }

  // Split-KV reduction (separate launch); the balanced schedule merges in the kernel.
  if (use_split_kv && !in_kernel_merge) {
    float log2_e = std::log2(std::exp(1.0f));
    float scale_softmax_log2 = sm_scale * log2_e;

    auto reduction_fn = mgr.get_reduction_fn(device);
    reduction_fn(
        torch_to_tvm(plan.workspace_o), torch_to_tvm(out), torch_to_tvm(plan.workspace_lse),
        torch_to_tvm(plan.num_kv_splits_per_row), (double)scale_softmax_log2, 1.0,
        (int64_t)plan.num_kv_splits, (int64_t)qo_total_len, (int64_t)num_qo_heads,
        (int64_t)head_dim_vo, (int64_t)(num_qo_heads * head_dim_vo), (int64_t)head_dim_vo,
        (int64_t)(num_qo_heads * head_dim_vo), (int64_t)head_dim_vo, (int64_t)orig_num_qo_heads,
        (int64_t)num_kv_heads, (int64_t)pack_factor, stream_int);
  }

  return out;
}

std::unique_ptr<PlanInfo> make_decode_plan(at::Tensor qo_segment_lens, at::Tensor kv_segment_lens,
                                           int num_qo_heads, int num_kv_heads, int num_kv_splits,
                                           int page_size, int topk, int usable_sm_count,
                                           std::optional<int> device,
                                           const std::string &split_mode) {
  ensure_initialized();
  TORCH_CHECK(!qo_segment_lens.is_cuda() && !kv_segment_lens.is_cuda(),
              "make_decode_plan requires host query lengths and planning KV capacity");
  TORCH_CHECK(topk == 16, "Q8KV4 decode requires TopK 16");
  auto qo_offset = kv_segment_lens - qo_segment_lens;
  return std::make_unique<PlanInfo>(_make_decode_plan_impl(
      qo_segment_lens, kv_segment_lens, num_qo_heads, num_kv_heads, qo_offset, num_kv_splits,
      page_size, topk, usable_sm_count, device, split_mode));
}

at::Tensor run_decode(at::Tensor q, at::Tensor k, at::Tensor v, PlanInfo &plan_info,
                      at::Tensor seq_lens, at::Tensor page_table, at::Tensor topk_indices,
                      at::Tensor k_scale, at::Tensor v_scale, at::Tensor out,
                      at::Tensor out_scale, float sm_scale) {
  ensure_initialized();
  return _run_decode_impl(q, k, v, plan_info, seq_lens, page_table, topk_indices, k_scale, v_scale,
                          out, out_scale, sm_scale);
}

} // namespace minimax::msa_v1::attention::decode::q8kv4
