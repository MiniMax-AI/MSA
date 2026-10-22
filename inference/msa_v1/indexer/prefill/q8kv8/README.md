# Q8KV8 Prefill Indexer

[简体中文](README.zh-CN.md)

## Purpose

True-varlen paged prefill indexer for SM100, SM103, and SM107. Q/K use E4M3, and the output contains
selected logical page indices for every query, including the local page.

## Public API

```python
from inference.msa_v1.indexer.prefill.q8kv8 import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)

wrapper = BatchPrefillIndexerWithPagedKVCacheWrapper()
wrapper.plan(
    cu_seqlens_q,
    cu_seqlens_k,
    page_table,
    total_q=total_q,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    num_index_heads=4,  # Number of local index heads: 1, 2, or 4.
)
topk_indices = wrapper.run(q, paged_k_cache)
```

Callers may pass a preallocated output to `run()`. When request lengths change, call `plan()` or
the public `replan()` outside CUDA Graph capture.

## Data contract

The maximum total KV length per request is 2,097,152 tokens (16384 pages of 128 tokens),
including the current query chunk. Mixed request lengths are supported; selection uses each
query's valid page count and always includes its local page.

- `num_index_heads` defaults to 1 and accepts 1/2/4. Heads select TopK independently and share K cache.
- Q and output must be contiguous CUDA tensors; K must reside on the same device.
- H=1 also retains the output head dimension.

- `q`: `[total_q, H, 128]`, E4M3.
- `paged_k_cache`: `[physical_pages, 1, 128, 128]`, E4M3.
- `cu_seqlens_q` / `cu_seqlens_k`: `[B + 1]`, CUDA `torch.int32`.
- `page_table`: `[B, max_pages]`, mapping logical pages to physical pages.
- Output: `[H, total_q, 16]`, `torch.int32` logical page indices. Valid entries form a prefix, and
  the final valid entry must be the local page.

## Runtime requirements

- Bottom-right causal alignment is used, and `max_seqlen_k` must not be less than
  `max_seqlen_q`.
- Historical pages may map to scattered and unordered physical pages.
- Call `plan()` and `replan()` outside CUDA Graph capture and complete warmup before the first
  capture.
- SM100 and SM103 require `nvidia-cutlass-dsl[cu13]>=4.5.2`. SM107 requires a compatible
  CUDA toolkit and a CuTe DSL build exposing `cutlass.cute.nvgpu.tcgen05.SM107MmaFP8Op`.
  The PyTorch C++ extension builder must also recognize the `10.7a` CUDA architecture;
  builds without it fail with `Unknown CUDA arch (10.7a)`.
- Use the repository CUTLASS submodule or set `CUTLASS_ROOT` to public CUTLASS headers,
  including for wheel installations.
- The default AOT cache directory is `~/.cache/minfer/msa_v1`; override it with
  `MSA_V1_AOT_CACHE`. Set `MSA_V1_AOT_DISABLE=1` to disable the AOT cache. JIT is disabled by
  default on a cache miss. To enable JIT, explicitly set
  `FMHA_SM100_ALLOW_JIT=1` and finish compilation before CUDA Graph capture.

## Validation and performance testing

Run from the repository root; correctness covers H=1/2/4 automatically.

```bash
MINIMAX_INFERENCE_TEST_SUITE=smoke python -m pytest tests/inference/msa_v1/indexer/prefill/q8kv8/test_real_cases.py
MINIMAX_INFERENCE_TEST_SUITE=full python -m pytest tests/inference/msa_v1/indexer/prefill/q8kv8/test_real_cases.py
python -m benchmarks.inference.msa_v1.indexer.prefill.q8kv8.benchmark --suite full --num-index-heads 4 --verify
```

The benchmark measures the public `run()` CUDA Graph E2E time and aggregates
useful compute throughput with production weights. Pass
`--baseline <baseline-h1.json>` to check differences against the baseline.
See the [shared benchmark contract](../../../../../datas/inference/README.md)
for metric definitions.

## Cache layout

Non-contiguous cache support has been verified on SM100/SM103, not SM107.

Cache data must be contiguous along the last (dim) storage dimension (`stride(-1) == 1`).
Cache data pointers and non-singleton outer byte strides must be multiples of 16.
Byte strides equal tensor element strides multiplied by `element_size()`. Broadcast,
overlapping, and negative-stride caches are outside the supported contract. Q, metadata,
and output must follow the layout requirements above.

After CUDA Graph capture, changing a cache
address, shape, or stride requires recapture. Separate wrappers retain separate workspaces.
