# Q8KV8 Prefill Indexer

[简体中文](README.md)

## Purpose

True-varlen paged prefill indexer for SM100 and SM103. Q/K use E4M3, and the output contains
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
    num_index_heads=4,
)
topk_indices = wrapper.run(q, paged_k_cache)
```

Callers may pass a preallocated output to `run()`. When request lengths change, call `plan()` or
the public `replan()` outside CUDA Graph capture.

## Data contract

- `num_index_heads` defaults to 1 and accepts 1/2/4. Heads select TopK independently and share K cache.
- Q/K and output must be contiguous CUDA tensors on the same device.
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
- SM100 and SM103 are supported, and `nvidia-cutlass-dsl[cu13]>=4.5.2` is required.
- Use the repository's CUTLASS submodule or set `CUTLASS_ROOT` to public CUTLASS headers.
  Set this environment variable when using an installed wheel.
- The default AOT cache directory is `~/.cache/minfer/msa_v1`; override it with
  `MSA_V1_AOT_CACHE`. Set `MSA_V1_AOT_DISABLE=1` to disable the AOT cache. JIT is disabled by
  default on a cache miss. To enable JIT, explicitly set
  `FMHA_SM100_ALLOW_JIT=1` and finish compilation before CUDA Graph capture.

## Validation commands

Run from the repository root; correctness covers H=1/2/4 automatically.

```bash
MINIMAX_INFERENCE_TEST_SUITE=smoke python -m pytest tests/inference/msa_v1/indexer/prefill/q8kv8/test_real_cases.py
MINIMAX_INFERENCE_TEST_SUITE=full python -m pytest tests/inference/msa_v1/indexer/prefill/q8kv8/test_real_cases.py
python -m benchmarks.inference.msa_v1.indexer.prefill.q8kv8.benchmark --suite full --num-index-heads 4 --verify
```

The benchmark measures the public `run()` CUDA Graph E2E time and aggregates
useful compute throughput with production weights. Pass
`--baseline <baseline.json>` to check differences against the baseline.
See the [shared benchmark contract](../../../../../datas/inference/README.en.md)
for data distributions and statistical methods.
