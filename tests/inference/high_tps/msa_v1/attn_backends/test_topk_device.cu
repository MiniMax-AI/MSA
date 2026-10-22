#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <limits>
#include <numeric>
#include <random>
#include <unistd.h>
#include <vector>

#include "attn/kernel/include/sm100_m3_decode_topk_runtime.hpp"

#define CHECK(call)                                                                                \
  do {                                                                                             \
    cudaError_t e = (call);                                                                        \
    if (e != cudaSuccess) {                                                                        \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));              \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

__global__ void select_row(float const *scores, int const *pages, int *output) {
  __shared__ cutlass::m3_decode::collective::runtime_topk::Scratch scratch;
  // Deliberately keep half of this 512-thread CTA outside TopK's barriers.
  if (threadIdx.x < 256) {
    cutlass::m3_decode::collective::runtime_topk::select(scores, *pages, scratch, output);
  }
}

std::array<int, 16> reference(std::vector<float> const &scores, int pages) {
  std::array<int, 16> result;
  result.fill(-1);
  if (pages <= 16) {
    std::iota(result.begin(), result.begin() + pages, 0);
    return result;
  }
  std::vector<int> ids(pages - 1);
  std::iota(ids.begin(), ids.end(), 0);
  std::sort(ids.begin(), ids.end(), [&](int a, int b) {
    return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
  });
  std::copy_n(ids.begin(), 15, result.begin());
  result[15] = pages - 1;
  std::sort(result.begin(), result.end());
  return result;
}

int main() {
  float *device_scores;
  int *device_pages, *device_output;
  CHECK(cudaMalloc(&device_scores, 8192 * sizeof(float)));
  CHECK(cudaMalloc(&device_pages, sizeof(int)));
  CHECK(cudaMalloc(&device_output, 16 * sizeof(int)));
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  select_row<<<1, 512, 0, stream>>>(device_scores, device_pages, device_output);
  CHECK(cudaStreamEndCapture(stream, &graph));
  size_t nodes = 0;
  CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
  if (nodes != 1)
    return 2;
  CHECK(cudaGraphInstantiate(&executable, graph, 0));
  cudaFuncAttributes attributes;
  CHECK(cudaFuncGetAttributes(&attributes, select_row));
  std::printf("TopK test kernel: registers=%d static_smem=%zu local_bytes=%zu\n",
              attributes.numRegs, attributes.sharedSizeBytes, attributes.localSizeBytes);
  std::mt19937 rng(20261021);
  std::normal_distribution<float> normal;
  std::vector<int> sizes = {8,    1024, 8192, 256,  8,    1,    16,   17,   31,   32,
                            33,   64,   65,   128,  129,  255,  257,  258,  511,  512,
                            513,  1023, 1025, 2047, 2048, 2049, 3072, 3073, 4096, 4097,
                            5120, 5121, 6144, 6145, 7168, 7169, 8191};
  int checked = 0;
  for (int pages : sizes) {
    for (int pattern = 0; pattern < 7; ++pattern) {
      std::vector<float> scores(8192, 1.0e30f); // Poison unused capacity.
      for (int i = 0; i < pages; ++i) {
        scores[i] = pattern == 0   ? normal(rng)
                    : pattern == 1 ? 0.5f
                    : pattern == 2 ? (i & 1 ? -0.0f : 0.0f)
                    : pattern == 3 ? float(i % 29)
                    : pattern == 4 ? -float(i % 43)
                    : pattern == 5 ? float(i % 23) * 1.0e-40f
                                   : normal(rng);
      }
      if (pattern == 6 && pages >= 18) {
        std::fill_n(scores.begin(), pages, 0.0f);
        for (int i = 0; i < 14; ++i)
          scores[i] = 1.0f - i * 0.01f;
        scores[14] = 0.5f;
        scores[15] = 0.500001f; // Same 16-bit bucket, different FP32 order.
      }
      auto const expected = reference(scores, pages);
      CHECK(cudaMemcpyAsync(device_scores, scores.data(), 8192 * sizeof(float),
                            cudaMemcpyHostToDevice, stream));
      CHECK(cudaMemcpyAsync(device_pages, &pages, sizeof(int), cudaMemcpyHostToDevice, stream));
      CHECK(cudaMemsetAsync(device_output, 0xa5, 16 * sizeof(int), stream));
      alarm(10);
      CHECK(cudaGraphLaunch(executable, stream));
      std::array<int, 16> observed;
      CHECK(cudaMemcpyAsync(observed.data(), device_output, sizeof(observed),
                            cudaMemcpyDeviceToHost, stream));
      CHECK(cudaStreamSynchronize(stream));
      alarm(0);
      if (observed != expected) {
        std::fprintf(stderr, "FAIL pages=%d pattern=%d\n", pages, pattern);
        for (int i = 0; i < 16; ++i)
          std::fprintf(stderr, "%d: %d vs %d\n", i, observed[i], expected[i]);
        return 1;
      }
      ++checked;
    }
  }
  CHECK(cudaGraphExecDestroy(executable));
  CHECK(cudaGraphDestroy(graph));
  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(device_scores));
  CHECK(cudaFree(device_pages));
  CHECK(cudaFree(device_output));
  std::printf("PASS %d exact TopK cases, one binary and one replayed Graph\n", checked);
}
