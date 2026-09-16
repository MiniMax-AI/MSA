# TP4 Q8KV4 Decode Indexer

[简体中文](README.md)

## Purpose

Paged decode indexer for SM100 and SM103. Q uses E4M3, K uses packed E2M1 with E4M3 scales,
and the output contains selected logical page indices for each query, including the local page.

## Public API

```python
from inference.msa_v1.indexer.decode.tp4_q8kv4 import (
    BatchDecodeIndexerWithPagedKVCacheWrapper,
)

wrapper = BatchDecodeIndexerWithPagedKVCacheWrapper()
wrapper.plan(page_table, seq_lens)
topk_indices = wrapper.run(q, packed_k_cache, k_scale=k_scale)
```

Use `workspace_size(batch_size)` to query the required workspace size and optionally provide an
external workspace to the wrapper. CUDA Graph mode also requires address-stable
`page_table_buffer` and `seq_lens_buffer` tensors at wrapper construction and a preallocated
output passed to `run()`.

## Data contract

- `q`: `[B, 8, 128]`, E4M3.
- `packed_k_cache`: `[physical_pages, 128, 64]`, packed E2M1.
- `k_scale`: `[physical_pages, 128, 8]`, E4M3 in a linear, non-swizzled layout.
- `page_table`: `[B, max_pages]`, a CUDA `torch.int32` logical-to-physical page mapping.
- `seq_lens`: `[B]`, CUDA `torch.int32`, including the current eight-token MTP chunk.
- Output: `[B * 8, 16]`, `torch.int32` logical page indices. Valid entries form a prefix, and
  the final valid entry is the local page.

## Runtime requirements

- For query `q_idx`, the local page is `(seq_lens[b] - 8 + q_idx) // 128`.
- Historical pages may map to scattered and unordered physical pages.
- Call `plan()` outside CUDA Graph capture.
- Toolchains that support QMUL4 use it automatically; otherwise the exact FP16 fallback
  supported by CUDA Toolkit 12.9 is selected. Callers do not choose the backend.
