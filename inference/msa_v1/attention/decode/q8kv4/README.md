# Q8KV4 Decode Attention

[简体中文](README.zh-CN.md)

## Purpose

Paged sparse causal decode attention for SM100, SM103, and SM107. Q uses E4M3, K/V use packed E2M1
with E4M3 scales, and the output uses BF16.
GQA=8 and GQA=16 are supported on B200/B300; SM107 retains GQA=16 support.
Dispatch uses the actual `Hq/Hkv` ratio.

## Public API

```python
from inference.msa_v1.attention.decode.q8kv4 import (
    BatchDecodeWithPagedKVCacheWrapper,
)

wrapper = BatchDecodeWithPagedKVCacheWrapper()
wrapper.plan(
    topk_indices,
    page_table,
    seq_lens,
    q_len_per_req=8,
    num_q_heads=32,
    num_kv_heads=4,
)
out = wrapper.run(
    q,
    (packed_k_cache, packed_v_cache),
    kv_cache_sf=(k_scale, v_scale),
)
```

`plan()` accepts request-level metadata, and `run()` accepts Q/K/V tensors and quantization
scales for the current layer. Callers may pass a preallocated `out` to `run()`.

## Data contract

- `q`: `[B * q_len_per_req, Hq, 128]`, E4M3.
- `packed_k_cache` / `packed_v_cache`: `[physical_pages, Hkv, 128, 64]`, packed E2M1 stored
  as `torch.uint8`, with two elements per byte.
- `k_scale` / `v_scale`: `[physical_pages, Hkv, 128, 8]`, E4M3 in a linear, non-swizzled
  layout. Each scale covers 16 consecutive head-dimension elements; this quantization group
  does not change with the GQA ratio.
- `topk_indices`: `[B * q_len_per_req, Hkv, 16]` logical page IDs. Valid entries form a prefix
  followed by `-1`; historical pages may be scattered and unordered, and the local page must
  be the final valid entry.
- `page_table`: `[B, max_pages]`, mapping logical pages to physical pages.
- `seq_lens`: `[B]`, final KV lengths including the current decode/MTP query chunk.
- `out`: `[B * q_len_per_req, Hq, 128]`, BF16.

All tensors must be contiguous and reside on the same CUDA device. `topk_indices`, `page_table`,
and `seq_lens` use `torch.int32`; Q, K/V, scales, and output require 16-byte aligned addresses.

## Runtime requirements

- `q_len_per_req` is any positive runtime integer.
- On B200/B300, `Hq/Hkv` must be 8 or 16; the defaults remain `Hq=64` and `Hkv=4`.
  The GQA=8 example uses `Hq=32`, `Hkv=4`; configurations such as `8/1` and `16/2` also work.
- `q_len_per_req=8` represents one main token plus seven predicted tokens. Each query retains
  its own TopK and causal position.
- Call `plan()` outside CUDA Graph capture and provide a preallocated `out` during capture.
- For the same batch, query shape, and capacity, update `seq_lens`, `page_table`, and TopK
  in place to reuse the plan / Graph, including shorter requests, padding, and refilling.
  Tensor addresses, shapes, and dtypes must stay unchanged. Order updates and `run()` on
  the same stream; changed capacities or addresses require a new plan / capture.
- Queries with a negative causal position produce zero output and require all `-1` TopK entries.
- `num_kv_splits` accepts 1/2/4/8; an explicit value is honored, while the default selects
  automatically. All support this reuse. Each wrapper owns independent workspace that can be
  released when the wrapper is destroyed. Concurrent streams or CUDA Graphs must use separate
  wrappers.
- Processes may share `TORCH_EXTENSIONS_DIR` on a filesystem supporting POSIX file locks.
  Each process uses its own wrapper; cold builds serialize publication of the same artifact.
- SM100/SM103 require CUDA Toolkit 12.9 or newer; SM107 requires CUDA Toolkit 13.5 or newer.
  Toolchains that support QMUL4 use it automatically; otherwise the exact FP16 dequantization
  fallback is selected. Callers do not choose the backend.

## Validation commands

Run from the repository root with the dependencies listed above installed:

```bash
MINIMAX_INFERENCE_TEST_SUITE=full python -m pytest tests/inference/msa_v1/attention/decode/q8kv4
python -m benchmarks.inference.msa_v1.attention.decode.q8kv4.benchmark --suite full --num-q-heads 32 --num-kv-heads 4
```

Full correctness covers GQA=8 and GQA=16. The benchmark measures
the public `run()` CUDA Graph E2E latency. Use `--num-q-heads 64` to measure GQA=16.
