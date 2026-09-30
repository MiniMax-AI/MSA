# BF16 Paged Sparse Prefill Attention

[简体中文](README.md)

## Purpose

BF16 paged sparse causal prefill attention for SM100, SM103, and SM107. It configures GQA through local query/KV head counts and can optionally return FP32 LSE.

## Public API

```python
from inference.msa_v1.attention.prefill.bf16 import (
    BatchPrefillWithPagedKVCacheWrapper,
)

wrapper = BatchPrefillWithPagedKVCacheWrapper()
wrapper.plan(
    topk_indices,
    cu_seqlens_q,
    cu_seqlens_k,
    page_table,
    num_q_heads=64,
    num_kv_heads=4,
    total_k=total_k,
    total_rows=total_rows,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
)
out, lse = wrapper.run(q, (k_cache, v_cache), return_lse=True)
```

Callers may pass preallocated `out` and `lse` tensors to `run()`.

## Data contract

- Q/K/V and output use BF16; optional `lse` uses FP32.
- The head dimension and page size are both 128.
- GQA group sizes 8 and 16 are supported; the defaults are 64 query heads and 4 KV heads.
- K/V caches may be contiguous `[physical_pages, Hkv, 128, 128]` tensors or SGLang-style
  strided page views with the same logical dimensions, dtype, and alignment requirements.
- `topk_indices` contains logical page IDs. Valid entries form a prefix followed by `-1`; pages
  may be scattered and unordered, and the final valid entry must be the local page.
- `cu_seqlens_q` / `cu_seqlens_k` are CUDA `torch.int32` varlen metadata.

## Runtime requirements

- The input tensor device selects the implementation; SM107 uses the Rubin path.
  The FP8 Rubin path requires a CuTe DSL version providing `cutlass.utils.rubin_helpers`
  and a CUDA toolchain supporting SM107.


- Only paged KV and causal attention are supported; chunk prefill uses bottom-right causal
  alignment.
- `cu_seqlens_k` is the only source of KV lengths.
- Call `plan()` outside CUDA Graph capture and complete the first `run()` before capture.
- The wrapper does not silently copy, reorder, or convert runtime K/V tensors.
- All inputs must reside on the same CUDA device and satisfy the validated shape, stride, and
  alignment requirements.
