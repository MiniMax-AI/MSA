# Q8KV8 Prefill Attention

[简体中文](README.md)

## Purpose

Paged sparse causal prefill attention for SM100, SM103, and SM107. Q/K/V use E4M3, the output uses
BF16, varlen chunk prefill is supported, and no K/V scales are used.

Attention probabilities use `E4M3(P * 448)` with compensation during normalization.
Returned LSE retains the natural-log semantics of the original logits. Probability
quantization follows decode semantics; final outputs are not guaranteed bitwise identical across paths.
Hardware exp2 is used. Alignment with decode covers scaling
and compensation semantics, without requiring bitwise-identical P.

## Public API

```python
from inference.msa_v1.attention.prefill.q8kv8 import (
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

The wrapper constructor optionally accepts `enable_fp16_softmax` and `enable_2x_fp8`.
The default `None` enables these only for Rubin FP8 inputs on the actual device;
`False` disables the corresponding optimization. Explicit `True` raises an error
on unsupported architectures or input types.

## Data contract

- `q`: `[total_q, Hq, 128]`, E4M3.
- `k_cache` / `v_cache`: `[physical_pages, Hkv, 128, 128]`, E4M3.
- `topk_indices`: `[Hkv, total_q, 16]` logical page IDs. Valid entries form a prefix followed
  by `-1`; historical pages may be scattered and unordered, and the local page must be the final
  valid entry.
- `cu_seqlens_q` / `cu_seqlens_k`: `[B + 1]`, CUDA `torch.int32`.
- `page_table`: `[B, max_pages]`, mapping logical pages to physical pages.
- `out`: `[total_q, Hq, 128]`, BF16; `lse`: `[total_q, Hq]`, FP32.

## Runtime requirements

- The input tensor device selects the implementation; SM107 uses the Rubin path.
  The FP8 Rubin path requires a CuTe DSL version providing `cutlass.utils.rubin_helpers`
  and a CUDA toolchain supporting SM107.


- Only paged KV and causal attention are supported; chunk prefill uses bottom-right causal
  alignment.
- `Hq / Hkv` may be 1, 2, 4, 8, or 16; the defaults are `Hq=64` and `Hkv=4`.
- `cu_seqlens_k` is the only source of KV lengths; the API does not accept K/V scales.
- Call `plan()` outside CUDA Graph capture and preallocate `out` and `lse` during capture.
