# BF16 Decode Indexer

[简体中文](README.zh-CN.md)

## Purpose

Select up to 16 logical KV pages independently for each local index head, using BF16 queries
and a shared BF16 key cache. Supported configurations are H=1/2/4 and Q=1–16.

## Public API

```python
import torch
from inference.msa_v1.indexer.decode.bf16 import (
    BatchDecodeIndexerWithPagedKVCacheWrapper,
)

q = torch.randn(2, 2, 2, 128, device="cuda", dtype=torch.bfloat16)
paged_k_cache = torch.randn(4, 128, 128, device="cuda", dtype=torch.bfloat16)
page_table = torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32)
seq_lens = torch.tensor([129, 256], device="cuda", dtype=torch.int32)
out = torch.empty(2, 4, 16, device="cuda", dtype=torch.int32)
wrapper = BatchDecodeIndexerWithPagedKVCacheWrapper()
wrapper.plan(page_table, seq_lens, num_index_heads=q.shape[2], query_length=q.shape[1])
indices = wrapper.run(q, paged_k_cache, out=out)
assert indices.shape == (2, 4, 16)
```

Call `plan()` outside CUDA Graph capture. Warm up `run()` before capture and provide a
preallocated output. Call `plan()` again when request metadata changes, or bind an explicit
`BatchDecodeIndexerPlan` from `inference.msa_v1.indexer.decode` and call its `update()` before
consumers. Construction and binding happen outside capture; `update()` supports capture.
Order cross-stream consumers and subsequent updates with events. Keep wrappers and plans alive
while their Graphs are in use.

Omitting `out` returns a wrapper-owned default output that later `run()` calls overwrite.
Use a separate output to retain results, and separate wrappers and writable workspaces for
concurrent calls.

## Data contract

| Argument | Shape | Dtype |
| --- | --- | --- |
| `q` | `[B,Q,H,128]` | `torch.bfloat16` |
| `paged_k_cache` | `[physical_pages,128,128]` | `torch.bfloat16` |
| `page_table` | `[B,max_pages]` | `torch.int32` |
| `seq_lens` | `[B]` | `torch.int32` |
| `out` | `[H,B*Q,16]` | `torch.int32` |

`plan()` defaults to `num_index_heads=1` and `query_length=8`; page-table capacity is 1–8192 pages.
All tensors must be contiguous and reside on the same CUDA device. Q and K require 16-byte
alignment. All requests share Q; lengths include the query chunk and satisfy
`Q <= seq_lens[b] <= max_pages*128`.

The score of a historical page is the maximum Q/K dot product across its tokens, scaled by
`1/sqrt(128)`. Heads select pages independently. Each output row contains a valid prefix,
with its local page last and unused slots set to `-1`. Historical indices need not be sorted. Selection uses approximate 16-bit quantized score ordering;
near-tied scores need not produce the same indices as exact FP32 sorting.
H=1 retains the same four-dimensional input and three-dimensional output conventions.

## Runtime requirements

The implementation targets GB200/B200 (SM100) and GB300/B300 (SM103), with
`nvidia-cutlass-dsl[cu13]>=4.5.2`. Different Q/H configurations may require compilation;
warm up the configurations needed by your application before Graph capture.

The output is head-major. Attention adapters expecting `[B*Q,H,16]` require an explicit
metadata-layout conversion at the integration boundary.

## Validation commands

```bash
MINIMAX_INFERENCE_TEST_SUITE=smoke python -m pytest tests/inference/msa_v1/indexer/decode/bf16
python -m benchmarks.inference.msa_v1.indexer.decode.bf16.benchmark --suite full --num-index-heads 4 --query-length 8 --verify --out bf16.json
python -m benchmarks.inference.msa_v1.indexer.decode.bf16.benchmark --suite low-latency --num-index-heads 1 --query-length 1 --verify --out bf16_low_latency.json
python -m benchmarks.inference.msa_v1.indexer.decode.plan --precision bf16 --num-index-heads 4 --query-length 8 --layers 1 --out plan.json
```

The production suite uses the shared case weights; the low-latency suite uses equal weights.
Effective MBU counts required traffic using each dtype's storage size and includes the public
score and TopK call. It is a traffic-model metric, not measured DRAM utilization. BF16 and FP8
may have similar MBU while their absolute latencies differ.
