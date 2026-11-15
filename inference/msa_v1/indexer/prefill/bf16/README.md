# BF16 Paged Prefill Indexer

[简体中文](README.zh-CN.md)

## Purpose

BF16 paged prefill indexer for SM100 and SM103. It selects logical pages for each query and
guarantees that the result includes the local page.

## Public API

```python
import torch
from inference.msa_v1.indexer.prefill.bf16 import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)

q = torch.randn(5, 2, 128, device="cuda", dtype=torch.bfloat16)
k_cache = torch.randn(4, 1, 128, 128, device="cuda", dtype=torch.bfloat16)
cu_seqlens_q = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
cu_seqlens_k = torch.tensor([0, 128, 384], device="cuda", dtype=torch.int32)
page_table = torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32)
out = torch.empty(2, 5, 16, device="cuda", dtype=torch.int32)
wrapper = BatchPrefillIndexerWithPagedKVCacheWrapper()
wrapper.plan(
    cu_seqlens_q, cu_seqlens_k, page_table,
    total_q=5, max_seqlen_q=3, max_seqlen_k=256, num_index_heads=2,
)
indices = wrapper.run(q, k_cache, out=out)
assert indices.shape == (2, 5, 16)
```

`BatchPrefillIndexerWithPagedKVCacheWrapper` provides a `plan()` / `run()` lifecycle. `plan()`
accepts varlen and page-table metadata, and `run()` accepts BF16 Q and paged BF16 K tensors for
the current layer. Callers may pass a preallocated output to `run()`.

### Pooled-key block scores

`BatchPrefillBlockScoreWrapper` returns scores without TopK selection or a forced local
block. One stored key represents `pooling_factor` (1, 2, or 4) original tokens, so each
128-token score block stores `128 // pooling_factor` keys.

```python
from inference.msa_v1.indexer.prefill.bf16.interface import BatchPrefillBlockScoreWrapper

scorer = BatchPrefillBlockScoreWrapper()
scorer.plan(
    cu_seqlens_q, cu_seqlens_k, page_table,
    total_q=total_q, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
    num_index_heads=2, pooling_factor=4,
)
scores = scorer.run(q, pooled_k_cache, out=score_output)
```

## Data contract

The maximum total KV length per request is 2,097,152 tokens (16384 pages of 128 tokens),
including the current query chunk. Mixed request lengths are supported; selection uses each
query's valid page count and always includes its local page.

- Q and K use BF16.
- One, two, or four local index heads are supported.
- Output shape is `[num_index_heads, total_q, 16]` and values are logical page IDs. Valid entries
  form a prefix, and the final valid entry must be the current query's local page.
- The K cache uses one KV head, head dimension 128, and page size 128.
- Varlen metadata and the page table use CUDA `torch.int32`.

| Argument | Shape | Dtype |
| --- | --- | --- |
| `q` | `[total_q,H,128]` | `torch.bfloat16` |
| `paged_k_cache` | `[physical_pages,1,128,128]` | `torch.bfloat16` |
| `cu_seqlens_q`, `cu_seqlens_k` | `[B+1]` | `torch.int32` |
| `page_table` | `[B,max_pages]` | `torch.int32` |
| `out` | `[H,total_q,16]` | `torch.int32` |

All inputs and outputs must be on the same CUDA device. Q, metadata, and output must be contiguous and 16-byte aligned.
Cumulative lengths start at zero and are nondecreasing; the final Q offset equals `total_q`,
and each request's KV length must cover its query length. `max_seqlen_q`/`max_seqlen_k` are
host upper bounds on per-request lengths. The KV bound is at most 16384×128, and the page table
must cover that bound. `num_index_heads` defaults to 4 and supports H=1/2/4.
The valid page count follows the bottom-right causal position:

```text
query_position = kv_length - query_length + query_index
local_page = query_position // 128
```

A historical page's score is the maximum FP32 Q/K dot product across its tokens, scaled by
`1/sqrt(128)`. TopK uses approximate 16-bit quantized ordering; near-tied scores need not match
exact FP32 sorting.

### Pooled-key block scores

Q has the same BF16 `[total_q,H,128]` layout. K is BF16
`[physical_pages,1,page_size // pooling_factor,128]`; the default `page_size=128` stores
one score block per physical page. `page_size=64` uses two page-table entries per logical
128-token score block; with pooling factor 4, each 64-token physical page stores 16 keys.
Padded and non-contiguous outer cache strides follow the same cache-layout requirements
above.

Both cumulative length arrays count original tokens. Query i of request b has original
position `p = kv_length[b] - query_length[b] + i`. Stored key j becomes visible only when
`(j+1) * pooling_factor <= p+1`; an unfinished group contributes no score. The output is
unscaled FP32 `[H,total_q,ceil(max_seqlen_k/128)]`, including visible keys in the current
block. Blocks with no visible complete keys, future blocks, and padded query rows contain
`-inf`. Scaling, rounding, selection, and any partial-group policy belong to the caller.
The existing indexer wrapper retains its original behavior.

The page table must cover `ceil(max_seqlen_k/page_size)` physical pages per request.
Length metadata must be nonnegative and monotone, start at zero, end at or below the
planned total Q capacity, and satisfy `kv_length >= query_length`. Valid page-table entries
must address initialized cache pages. These device-value constraints are the caller's
responsibility; `run()` performs no device-to-host validation or synchronization.

## Runtime requirements

- Only SM100/SM103 and paged K caches are supported.
- Requires `nvidia-cutlass-dsl[cu13]>=4.5.2`; rebuild compiled caches after upgrading DSL.
- Logical pages may be scattered and unordered.
- Call `plan()` outside CUDA Graph capture and complete the first `run()` before capture.
- Q/K, metadata, workspace, and output must reside on the same CUDA device and satisfy the
  validated shape, stride, and alignment requirements.

The TopK wrapper owns its writable workspace and default output; later `run()` calls overwrite
the default output. Use separate wrappers for concurrent streams and events to order metadata
updates before consumers. With metadata addresses, shapes, total Q, and capacities unchanged,
call `replan()` outside capture after in-place length updates; finish in-place page mapping
updates before consumers run. Changes to addresses, shapes, head counts, or capacities require
`plan()` and a new capture. Keep the wrapper, inputs, and outputs alive while using its Graph.

### Pooled-key block scores

Call `plan()` outside capture and warm up `run()` before capture. With addresses, shapes,
head count, and capacities unchanged, lengths and page mappings may change in place;
`run()` rebuilds its metadata and overwrites the complete output on every call and Graph
replay. Use separate wrappers for concurrent execution and order updates with stream
or event dependencies. The wrapper allocates its default output lazily on the first call
without `out`.
Using only explicit `out` avoids a default score tensor; an existing default output
remains owned by the wrapper and is overwritten by later calls without `out`.
An explicit `out` must be contiguous, 16-byte aligned FP32 on the same device and have the
planned output shape. The maximum KV length is 2,097,152 original tokens; output capacity
must contain fewer than 2^31 elements.

## Validation commands

```bash
MINIMAX_INFERENCE_TEST_SUITE=smoke python -m pytest tests/inference/msa_v1/indexer/prefill/bf16
python -m benchmarks.inference.msa_v1.indexer.prefill.bf16.benchmark --suite full --num-index-heads 2 --verify --out prefill_bf16.json
```

The benchmark measures the public `run()` path inside CUDA Graphs. `--verify` checks all
valid historical scores and TopK outputs against an independent reference outside timing.

### Pooled-key block scores

```bash
python -m pytest tests/inference/msa_v1/indexer/prefill/bf16/test_pooled_block_scores.py --msa-inference-suite=full
```

## Cache layout

Cache data must be contiguous along the last (dim) storage dimension (`stride(-1) == 1`).
Cache data pointers and non-singleton outer byte strides must be multiples of 16.
Byte strides equal tensor element strides multiplied by `element_size()`. Broadcast,
overlapping, and negative-stride caches are outside the supported contract. Q, metadata,
and output must follow the layout requirements above.

After CUDA Graph capture, changing a cache
address, shape, or stride requires recapture. Separate wrappers retain separate workspaces.
