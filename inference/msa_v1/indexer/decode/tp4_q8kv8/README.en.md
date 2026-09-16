# TP4 Q8KV8 Decode Indexer

[简体中文](README.md)

## Purpose

Paged decode indexer for SM100/SM103. Q/K use E4M3, and the output contains selected logical page
indices for each query, including the local page. The API does not accept `k_scale`.

## Public API

```python
from inference.msa_v1.indexer.decode.tp4_q8kv8 import (
    BatchDecodeIndexerWithPagedKVCacheWrapper,
)

wrapper = BatchDecodeIndexerWithPagedKVCacheWrapper()
wrapper.plan(page_table, seq_lens)
topk_indices = wrapper.run(q, paged_k_cache)
```

Use `workspace_size(batch_size)` to query the required workspace size and optionally provide an
external workspace to the wrapper. CUDA Graph mode also requires address-stable
`page_table_buffer` and `seq_lens_buffer` tensors at wrapper construction and a preallocated
output passed to `run()`.

## Data contract

- `q`: `[B, 8, 128]`, E4M3.
- `paged_k_cache`: `[physical_pages, 128, 128]`, E4M3.
- `page_table`: `[B, max_pages]`, a CUDA `torch.int32` logical-to-physical page mapping.
- `seq_lens`: `[B]`, CUDA `torch.int32`, including the current eight-token MTP chunk.
- Output: `[B * 8, 16]`, `torch.int32` logical page indices. Valid entries form a prefix, and
  the final valid entry is the local page.

## Runtime requirements

- For query `q_idx`, the local page is `(seq_lens[b] - 8 + q_idx) // 128`.
- Historical pages may map to scattered and unordered physical pages.
- Call `plan()` outside CUDA Graph capture.
- B200/SM100 and B300/SM103 are supported, and `nvidia-cutlass-dsl[cu13]>=4.5.2` is required.
