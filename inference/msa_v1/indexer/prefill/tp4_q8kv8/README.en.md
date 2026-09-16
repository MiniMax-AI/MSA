# TP4 Q8KV8 Prefill Indexer

[简体中文](README.md)

## Purpose

True-varlen paged prefill indexer for SM100 and SM103. Q/K use E4M3, and the output contains
selected logical page indices for every query, including the local page.

## Public API

```python
from inference.msa_v1.indexer.prefill.tp4_q8kv8 import (
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
)
topk_indices = wrapper.run(q, paged_k_cache)
```

Callers may pass a preallocated output to `run()`. When request lengths change, call `plan()` or
the public `replan()` outside CUDA Graph capture.

## Data contract

- `q`: `[total_q, 1, 128]`, E4M3.
- `paged_k_cache`: `[physical_pages, 1, 128, 128]`, E4M3.
- `cu_seqlens_q` / `cu_seqlens_k`: `[B + 1]`, CUDA `torch.int32`.
- `page_table`: `[B, max_pages]`, mapping logical pages to physical pages.
- Output: `[total_q, 16]`, `torch.int32` logical page indices. Valid entries form a prefix, and
  the final valid entry must be the local page.

## Runtime requirements

- Bottom-right causal alignment is used, and `max_seqlen_k` must not be less than
  `max_seqlen_q`.
- Historical pages may map to scattered and unordered physical pages.
- Call `plan()` and `replan()` outside CUDA Graph capture and complete warmup before the first
  capture.
- SM100 and SM103 are supported, and `nvidia-cutlass-dsl[cu13]>=4.5.2` is required.
- The default AOT cache directory is `~/.cache/minfer/msa_v1`; override it with
  `MSA_V1_AOT_CACHE`. Set `MSA_V1_AOT_DISABLE=1` to disable the AOT cache. JIT is disabled by
  default on a cache miss. For development-time JIT, explicitly set
  `FMHA_SM100_ALLOW_JIT=1` and finish compilation before CUDA Graph capture.
