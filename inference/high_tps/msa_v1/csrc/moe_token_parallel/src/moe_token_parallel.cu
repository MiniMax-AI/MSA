#include <cuda.h>
#include <cuda_runtime.h>

#include "include/sm100/collective/op_combine_allreduce.cuh"
#include "include/sm100/collective/op_routed_down.cuh"
#include "include/sm100/collective/op_routed_upgate.cuh"
#include "include/sm100/collective/op_router_pack.cuh"
#include "include/sm100/collective/op_shared_down.cuh"
#include "include/sm100/collective/op_shared_upgate.cuh"
#include "include/sm100/collective/router_fp32.cuh"
#include "src/moe_token_parallel.hpp"

namespace mega_moe::token_parallel {
// One persistent cooperative launch: 128 routed CTAs (token x top-k x slab)
// and 16 shared CTAs. Router jobs run on routed CTAs 8..103 before their own
// expert work; the first eight routed CTAs also pack one token's input.
__global__ __launch_bounds__(kThreads, 1) void moe_kernel(const __grid_constant__ KernelArgs k) {
  extern __shared__ __align__(1024) uint8_t smem[];
  auto const &a = k.a;
  int const role = int(blockIdx.x), tid = int(threadIdx.x);
  // Each routed producer thread owns one expert during route selection. Its
  // router bias is an input independent of every stage: request it now so
  // the selection after the router hand-off finds it in a register.
  float route_bias = 0.f;
  if (role < kRouted && tid >= kConsumers)
    route_bias = a.router_bias[tid - kConsumers];
  if (role < kRouted && tid == kConsumers + 96) {
    // Warm the routed descriptors before route selection issues the first
    // FC1 transfer and before FC2 begins.
    prefetch_map(&k.maps.map[RoutedFc1Value]);
    prefetch_map(&k.maps.map[RoutedFc1Scale]);
    prefetch_map(&k.maps.routed_fc2_full);
  }
  if (role >= kRouted && tid == kConsumers + 96) {
    prefetch_map(&k.maps.map[SharedFc1Value]);
    prefetch_map(&k.maps.map[SharedInputValue]);
  }
  if (role >= kRouted && tid == kConsumers + 64) {
    // The shared up/gate weights depend on no stage of this invocation.
    // Warm L2 at entry rather than after the startup join: these bytes
    // ride the otherwise idle HBM window before routes are known, and the
    // demand loads later read the same public bytes, merely L2-resident.
    for (int g = 0; g < H / (128 * kSharedGuPagesPerSlot); ++g)
      prefetch_shared16_pair(&k.maps.map[SharedFc1Value], role - kRouted,
                             g * kSharedGuPagesPerSlot);
  }
  // Router pages depend on nothing: router jobs stage weights and BF16 token
  // vectors at entry, overlapping the startup join and TMEM allocation.
  if (role >= 8 && role < 8 + kRouterJobs && tid >= kConsumers)
    produce_router_early(smem, k, role - 8, tid - kConsumers);
  Ring const ring = make_ring(smem);
  if (role < kRouted) {
    if (tid < kUpgateMegaSlots) {
      // Four producer arrivals plus one native scale-copy completion from
      // each K256 formatter. Values and scales share one readiness acquire.
      init_bar(smem_address(smem + kUpgateFull + tid * 8), 4 + kUpgateK / 256);
      // Ordinary pages let tcgen05.commit release the slot directly. Tile
      // endings commit to one-shot ready barriers shared by retire/epilogue.
      init_bar(smem_address(smem + kUpgateEmpty + tid * 8), 1);
      init_bar(smem_address(smem + kUpgateScaleReady + tid * 8), 1);
      if (tid == 0)
        init_bar(smem_address(smem + kUpgateInputBar), 1);
      if (tid == 0)
        init_bar(smem_address(smem + kUpgateScalesBar), 1);
    }
    if (tid < 3)
      init_bar(smem_address(smem + kUpgateTileReady + tid * 8), 1);
    if (tid == 4)
      init_bar(smem_address(smem + kRouteBar), 1);
    if (tid == 7)
      init_bar(smem_address(smem + kSharedSeenBar), 1);
    if (tid >= 8 && tid < 8 + kDownSlots) {
      int const s = tid - 8;
      init_bar(smem_address(smem + kOvFull + s * 8), 1);
      init_bar(smem_address(smem + kOvEmpty + s * 8), 1);
      init_bar(smem_address(smem + kOvDone + s * 8), 1);
      init_bar(smem_address(smem + kOvLoaded + s * 8), 3);
      if (s < 3)
        init_bar(smem_address(smem + kOvRetired + s * 8), 1);
    }
    if (tid < kUpgateMegaSlots || tid < 3 || tid == 4 || tid == 7 ||
        (tid >= 8 && tid < 8 + kDownSlots))
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // A router job's consumer warps run it before the startup join: its pages
  // were requested at entry and it needs only its own generation snapshot.
  if (role >= 8 && role < 8 + kRouterJobs && tid < kConsumers)
    op_router_prejoin(smem, a, role - 8, tid);
  // One launch per call advances the completion generation exactly once:
  // the final combine arrival publishes ready = this call's generation, and
  // the next call is ordered after it by the stream's kernel boundary. Every
  // CTA snapshots before any of its own contributions, so it never observes
  // this call's final publication.
  if (tid == 0) {
    uint32_t const generation = load_relaxed_gpu(a.ready[a.rank]) + 1u;
    *reinterpret_cast<uint32_t *>(smem + 8) = generation;
    *reinterpret_cast<uint32_t *>(smem + 12) =
        role < kRouted && selected(role) == 0 ? generation : 0u;
  }
  // Shared up/gate advances a private sixteen-ticket generation. The counter
  // persists across calls; no caller-side clearing is required.
  if (role >= kRouted && tid == 0) {
    auto counter = reinterpret_cast<unsigned long long *>(arena(a, a.rank) + kShared16Counter);
    auto ticket = atomicAdd(counter, 1ull);
    *reinterpret_cast<uint32_t *>(smem + 4) = uint32_t(ticket / kShared) + 1u;
  }
  int const tmem_owner_base = role < kRouted ? kConsumers + 64 : 0;
  if (tid >= tmem_owner_base && tid < tmem_owner_base + 32) {
    if (role < kRouted)
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(
                       smem_address(smem))
                   : "memory");
    else
      shared_tc::tmem_alloc(smem_address(smem), kSharedGuTmemColumns);
    shared_tc::tmem_relinquish();
  }
  // Input-pack rows depend on nothing: issue their loads before the CTA join
  // so the global latency hides under it.
  InputLoad input_load;
  if (role < T && tid < kConsumers)
    op_input_load(a, role, tid, input_load);
  __syncthreads();
  Context c{tid,
            *reinterpret_cast<uint32_t *>(smem + 8),
            0u,
            *reinterpret_cast<uint32_t *>(smem),
            smem,
            reinterpret_cast<float *>(smem + kBridgeOffset),
            Producer{ring, 0u, tid - kConsumers},
            Consumer{ring, 0u, tid},
            nullptr};
  if (c.produces()) {
    if (role < kRouted) {
      select_routes(c, a, role, route_bias);
      produce_routed_upgate(c, k, role);
      // FC2 streams into retiring FC1 slots without a CTA-wide reset.
      if (c.producer.ptid == 0)
        produce_down_overlap(c, k, role);
    } else {
      produce_shared_upgate(c, k, role);
      produce_shared_down(c, k, role);
    }
  } else {
    if (role < 8)
      op_input_pack(c, a, role, input_load);
    if (role < kRouted) {
      op_routed_upgate(c, a, role);
      op_routed_down_overlap(c, a, role);
      if (selected(role) == 0) {
        c.comm_epoch = *reinterpret_cast<uint32_t *>(smem + 12);
        c.shared_down = shared_partials(a);
        op_combine_allreduce(c, a, token(role), slab(role));
      }
    } else {
      op_shared_upgate(c, a, role);
      c.shared_down = shared_partials(a);
      op_shared_down(c, a, role);
    }
  }
  // Release TMEM after shared down has consumed the shared intermediate.
  __syncthreads();
  if (tid < 32) {
    if (role < kRouted)
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(c.tmem)
                   : "memory");
    else
      shared_tc::tmem_free(c.tmem, kSharedGuTmemColumns);
  }
}

namespace {
CUresult encode(CUtensorMap *map, int rank, void const *pointer, uint64_t const *dims,
                uint64_t const *strides, uint32_t const *box, CUtensorMapSwizzle swizzle,
                CUtensorMapL2promotion promotion) {
  uint32_t const step[5] = {1, 1, 1, 1, 1};
  return cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, const_cast<void *>(pointer), dims, strides, box,
      step, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, promotion, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
} // namespace

cudaError_t encode_maps(Args const &a, Maps *maps) {
  CUresult status = CUDA_SUCCESS;
  auto check = [&](CUresult result) {
    if (status == CUDA_SUCCESS)
      status = result;
  };
  {
    // Routed FC1 values: K256 panels adjacent within each eight-row swizzle
    // atom; the native descriptor strides over all K panels to the next atom.
    uint64_t dims[5] = {128u, 8u, uint64_t(H / 256), uint64_t(I / 8), uint64_t(2 * E)};
    uint64_t strides[4] = {uint64_t(H / 2), 128u, uint64_t(8) * (H / 2), uint64_t(I) * (H / 2)};
    uint32_t box[5] = {128u, 8u, uint32_t(kUpgateK / 256), 8u, 2u};
    check(encode(&maps->map[RoutedFc1Value], 5, a.routed_fc1, dims, strides, box,
                 CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B));
  }
  {
    // Routed FC1 scales: each K512 panel holds two adjacent 16-byte slices per
    // row. Rows are 384 bytes apart, so each request reads one 32-byte sector
    // per row; a wider promotion would fetch later K pages' sectors that the
    // concurrent weight stream evicts before their use.
    uint64_t dims[4] = {32u, uint64_t(I), uint64_t(2 * E), uint64_t(H / 512)};
    uint64_t strides[3] = {uint64_t(H / 16), uint64_t(I) * (H / 16), 32u};
    uint32_t box[4] = {32u, 64u, 2u, uint32_t(kUpgateK / 512)};
    check(encode(&maps->map[RoutedFc1Scale], 4, a.routed_fc1_sf, dims, strides, box,
                 CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE));
  }
  {
    // Routed FC2 values: three SW128 K256 panels concatenated in one view.
    uint64_t dims[3] = {128u, uint64_t(E) * H, 3u}, strides[2] = {I / 2, 128u};
    uint32_t box[3] = {128u, 128u, 3u};
    check(encode(&maps->routed_fc2_full, 3, a.routed_fc2, dims, strides, box,
                 CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE));
  }
  {
    // Shared FC1: each CTA's channel dimension is bounded to 48. TMA fills its
    // sixteen unused native rows with zero instead of fetching the next CTA's.
    uint64_t dims[5] = {128, 48, 2, kShared, H / 128};
    uint64_t strides[4] = {H, uint64_t(H) * I, 48u * H, 128};
    uint32_t box[5] = {128, 64, 2, 1, kSharedGuPagesPerSlot};
    check(encode(&maps->map[SharedFc1Value], 5, a.shared_fc1, dims, strides, box,
                 CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B));
  }
  {
    uint64_t dims[3] = {128, T, H / 128}, strides[2] = {H, 128};
    uint32_t box[3] = {128, T, kSharedGuPagesPerSlot};
    check(encode(&maps->map[SharedInputValue], 3, a.shared_input, dims, strides, box,
                 CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B));
  }
  auto encode_down = [&](MapId id, void const *pointer, int rows, int tile_rows) {
    uint64_t dims[3] = {128, uint64_t(rows), 6}, strides[2] = {I, 128};
    uint32_t box[3] = {128, uint32_t(tile_rows), 6};
    check(encode(&maps->map[id], 3, pointer, dims, strides, box, CU_TENSOR_MAP_SWIZZLE_128B,
                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B));
  };
  encode_down(SharedFc2Value, a.shared_fc2, H, 128);
  encode_down(SharedMidValue, a.shared_mid, T, 8);
  return status == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t configure_kernel(int device, int *resident_ctas) {
  cudaError_t status =
      cudaFuncSetAttribute(moe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  if (status != cudaSuccess)
    return status;
  cudaDeviceProp properties{};
  if ((status = cudaGetDeviceProperties(&properties, device)) != cudaSuccess)
    return status;
  int per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, moe_kernel, kThreads, kSmemBytes);
  if (status != cudaSuccess)
    return status;
  *resident_ctas = properties.cooperativeLaunch ? per_sm * properties.multiProcessorCount : 0;
  return cudaSuccess;
}

cudaError_t launch(KernelArgs const &arguments, cudaStream_t stream) {
  void *parameters[] = {const_cast<KernelArgs *>(&arguments)};
  return cudaLaunchCooperativeKernel(reinterpret_cast<void *>(moe_kernel), dim3(kGrid),
                                     dim3(kThreads), parameters, kSmemBytes, stream);
}
} // namespace mega_moe::token_parallel
