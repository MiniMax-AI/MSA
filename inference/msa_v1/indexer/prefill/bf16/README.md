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

## Data contract

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

All inputs and outputs must be contiguous, 16-byte aligned, and on the same CUDA device.
Cumulative lengths start at zero and are nondecreasing; the final Q offset equals `total_q`,
and each request's KV length must cover its query length. `max_seqlen_q`/`max_seqlen_k` are
host upper bounds on per-request lengths. The KV bound is at most 8192×128, and the page table
must cover that bound. `num_index_heads` defaults to 4 and supports H=1/2/4.
The valid page count follows the bottom-right causal position:

```text
query_position = kv_length - query_length + query_index
local_page = query_position // 128
```

A historical page's score is the maximum FP32 Q/K dot product across its tokens, scaled by
`1/sqrt(128)`. TopK uses approximate 16-bit quantized ordering; near-tied scores need not match
exact FP32 sorting.

## Runtime requirements

- Only SM100/SM103 and paged K caches are supported.
- Requires `nvidia-cutlass-dsl[cu13]>=4.5.2`; rebuild compiled caches after upgrading DSL.
- Logical pages may be scattered and unordered.
- Call `plan()` outside CUDA Graph capture and complete the first `run()` before capture.
- Q/K, metadata, workspace, and output must reside on the same CUDA device and satisfy the
  validated shape, stride, and alignment requirements.

Each wrapper owns its writable workspace and default output; later `run()` calls overwrite
the default output. Use separate wrappers for concurrent streams and events to order metadata
updates before consumers. With metadata addresses, shapes, total Q, and capacities unchanged,
call `replan()` outside capture after in-place length updates; finish in-place page mapping
updates before consumers run. Changes to addresses, shapes, head counts, or capacities require
`plan()` and a new capture. Keep the wrapper, inputs, and outputs alive while using its Graph.

## Validation commands

```bash
MINIMAX_INFERENCE_TEST_SUITE=smoke python -m pytest tests/inference/msa_v1/indexer/prefill/bf16
python -m benchmarks.inference.msa_v1.indexer.prefill.bf16.benchmark --suite full --num-index-heads 2 --verify --out prefill_bf16.json
```

The benchmark measures the public `run()` path inside CUDA Graphs. `--verify` checks all
valid historical scores and TopK outputs against an independent reference outside timing.
