# Q8KV4 Prefill Attention

[简体中文](README.zh-CN.md)

## Purpose

Paged sparse causal prefill attention for SM100, SM103 and SM107. Q uses E4M3, K/V use packed
E2M1 with E4M3 scales, the output uses BF16, and varlen chunk prefill is supported.

Attention probabilities use `E4M3(P * 448)` with compensation during normalization.
Returned LSE retains the natural-log semantics of the original logits. Probability
quantization follows decode semantics; final outputs are not guaranteed bitwise identical across paths.

## Public API

```python
from inference.msa_v1.attention.prefill.q8kv4 import (
    BatchPrefillWithPagedKVCacheWrapper,
    PrefillWorkspace,
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

### Split workspace

The forward writes one partial output and LSE per (query, head, selected page) and the combine
merges them. `PrefillWorkspace` owns those buffers. A wrapper creates its own in `plan()`;
wrappers that run in order on one stream may share one passed to the constructor, and
`partial_bytes` reserves the largest batch up front:

```python
workspace = PrefillWorkspace(device)
workspace = PrefillWorkspace(
    device, partial_bytes=workspace.required_bytes(max_total_q, num_q_heads)
)
wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace)
```

On a GPU with several memory locality domains (Rubin uGPU, Blackwell) the workspace stripes the
partial buffer over the domains and the combine reads every tile from the domain of its own SM;
the results are the same as with interleaved memory. A batch that outgrows a striped buffer
synchronizes the device before replacing it, so reserve the largest batch before CUDA Graph
capture. `MM_SPARSE_PREFILL_Q8KV4_LOCALITY_DOMAINS=0` keeps the buffer interleaved (for
comparisons). One workspace must not be used from two streams at once.

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
- Head counts are inferred from TopK's leading dimension. Local 64/4 and 16/1
  heads are supported without head replication.
- `cu_seqlens_k` is the only source of KV lengths.
- Call `plan()` outside CUDA Graph capture and preallocate `out` and `lse` during capture.
- CUDA Toolkit 13.4 or newer is required; SM107 requires 13.5 or newer. NVFP4 dequantization
  uses QMUL4 where the toolchain provides it and an exact packed-FP16 path otherwise (SM107).

## Cache layout

Cache data must be contiguous along the last (dim) storage dimension (`stride(-1) == 1`).
Cache data pointers and non-singleton outer byte strides must be multiples of 16.
Each scale page must have tightly packed `[Hkv, 128, 8]` storage. Only the page stride may include padding; it must cover a complete page and
be a multiple of 16 bytes. Scale pointers must be 16-byte aligned. Data and scale may
have different page strides. Scale token/head padding or outer-axis permutations are not supported.
K/V strides must match; K/V scale strides must also match.
Byte strides equal tensor element strides multiplied by `element_size()`. Broadcast,
overlapping, and negative-stride caches are outside the supported contract. Q, metadata,
and output must follow the layout requirements above.

After CUDA Graph capture, changing a cache
address, shape, or stride requires recapture. Separate wrappers retain separate workspaces.
