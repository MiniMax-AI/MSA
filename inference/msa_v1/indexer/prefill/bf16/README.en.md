# BF16 Paged Prefill Indexer

[简体中文](README.md)

## Purpose

BF16 paged prefill indexer for SM100 and SM103. It selects logical pages for each query and
guarantees that the result includes the local page.

## Public API

`BatchPrefillIndexerWithPagedKVCacheWrapper` provides a `plan()` / `run()` lifecycle. `plan()`
accepts varlen and page-table metadata, and `run()` accepts BF16 Q and paged BF16 K tensors for
the current layer. Callers may pass a preallocated output to `run()`.

## Data contract

- Q and K use BF16.
- One or four local index heads are supported.
- Output shape is `[num_index_heads, total_q, 16]` and values are logical page IDs. Valid entries
  form a prefix, and the final valid entry must be the current query's local page.
- The K cache uses one KV head, head dimension 128, and page size 128.
- Varlen metadata and the page table use CUDA `torch.int32`.

## Runtime requirements

- Only SM100/SM103 and paged K caches are supported.
- Requires `nvidia-cutlass-dsl[cu13]>=4.5.2`; rebuild compiled caches after upgrading DSL.
- Logical pages may be scattered and unordered.
- Call `plan()` outside CUDA Graph capture and complete the first `run()` before capture.
- Q/K, metadata, workspace, and output must reside on the same CUDA device and satisfy the
  validated shape, stride, and alignment requirements.
