# Q8KV4 Prefill Attention

[简体中文](README.md)

## Purpose

Paged sparse causal prefill attention for SM100 and SM103. Q uses E4M3, K/V use packed E2M1
with E4M3 scales, the output uses BF16, and varlen chunk prefill is supported.

Attention probabilities use `E4M3(P * 448)` with compensation during normalization.
Returned LSE retains the natural-log semantics of the original logits. Probability
quantization follows decode semantics; final outputs are not guaranteed bitwise identical across paths.

## Public API

```python
from inference.msa_v1.attention.prefill.q8kv4 import (
    BatchPrefillWithPagedKVCacheWrapper,
)

wrapper = BatchPrefillWithPagedKVCacheWrapper()
wrapper.plan(
    topk_indices,
    cu_seqlens_q,
    cu_seqlens_k,
    page_table,
    total_k=total_k,
    total_rows=total_rows,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
)
out, lse = wrapper.run(
    q,
    (packed_k_cache, packed_v_cache),
    kv_cache_sf=(k_scale, v_scale),
    return_lse=True,
)
```

Callers may pass preallocated `out` and `lse` tensors to `run()`.

## Data contract

- `q`: `[total_q, Hq, 128]`, E4M3; `Hq = 16 * Hkv`.
- `packed_k_cache` / `packed_v_cache`: `[physical_pages, Hkv, 128, 64]`, packed E2M1.
- `k_scale` / `v_scale`: `[physical_pages, Hkv, 128, 8]`, E4M3 in a linear, non-swizzled
  layout.
- `topk_indices`: `[Hkv, total_q, 16]` logical page IDs. Valid entries form a prefix followed by
  `-1`; historical pages may be scattered and unordered, and the local page must be the final
  valid entry.
- `cu_seqlens_q` / `cu_seqlens_k`: `[B + 1]`, CUDA `torch.int32`.
- `page_table`: `[B, max_pages]`, mapping logical pages to physical pages.
- `out`: `[total_q, Hq, 128]`, BF16; `lse`: `[total_q, Hq]`, FP32.

## Runtime requirements

- Only paged KV and causal attention are supported; chunk prefill uses bottom-right causal
  alignment.
- Head counts are inferred from TopK's leading dimension. TP1's local 64/4 and TP4's 16/1
  heads are supported without head replication.
- `cu_seqlens_k` is the only source of KV lengths.
- Call `plan()` outside CUDA Graph capture and preallocate `out` and `lse` during capture.
- NVFP4 dequantization requires CUDA Toolkit 13.4 or newer.
