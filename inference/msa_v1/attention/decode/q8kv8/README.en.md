# Q8K8 Paged Sparse Decode Attention

[简体中文](README.md)

## Purpose

This module adapts MSA per-query sparse metadata to an external FlashInfer Q8K8 block-sparse
decode backend. Q/K/V use E4M3, and the output uses BF16. The API does not perform implicit FP4
dequantization.

## Public API

```python
from inference.msa_v1.attention.decode.q8kv8 import (
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
out = wrapper.run(q, (k_cache, v_cache))
```

Callers may pass a preallocated `out` to `run()`.

## Data contract

- `q`: `[B * q_len_per_req, Hq, 128]`, E4M3.
- `k_cache` / `v_cache`: `[physical_pages, Hkv, 128, 128]`, E4M3.
- `topk_indices`: `[B * q_len_per_req, Hkv, topk]`, where `topk <= 16`. Valid entries form a
  prefix followed by `-1`; historical pages may be scattered and unordered, and the local page
  must be the final valid entry.
- `page_table`: `[B, max_pages]`, mapping logical pages to physical pages.
- `seq_lens`: `[B]`, final KV lengths including the current decode/MTP query chunk.
- `out`: `[B * q_len_per_req, Hq, 128]`, BF16.

The default configuration is `Hq=64`, `Hkv=4`, head dimension 128, and page size 128.

`topk_indices`, `page_table`, and `seq_lens` must be contiguous CUDA `torch.int32` tensors.
Q and output must also be contiguous. All tensors reside on the same CUDA device.

## Runtime requirements

FlashInfer is an optional external dependency. This repository does not copy or distribute
FlashInfer source code or cubins; install a version with a compatible block-sparse decode backend
before use. Call `plan()` outside CUDA Graph capture, run one warmup `run()` before the first
capture, and use preallocated outputs during capture.

- GQA=8/16 is supported on B200/SM100 and B300/SM103; `q_len_per_req=8` means one main token plus seven MTP tokens.
- FlashInfer paths require `seq_lens[b] >= q_len_per_req` and unique historical TopK pages preceding the local page.
- FlashInfer `plan()` validates GPU metadata and may synchronize with the host. Re-plan after metadata contents change and recapture existing Graphs.
- Warm up `run()` before the first capture and preallocate `out` during capture. Concurrent streams/Graphs use separate wrappers.
- Q, K/V, and output must have 16-byte aligned addresses and reside on the same GPU.
- `Hq/Hkv` supports 8 or 16 through FlashInfer; the default remains `64/4`. The final two K/V dimensions must be contiguous; head/page strides may be non-contiguous, with matching K/V strides.

Install the optional dependency from the repository root while retaining CuTe DSL 4.5.2:

```bash
python -m pip install -e '.[flashinfer]'
```

This extra pins `flashinfer-python==0.6.17`. First use may download or build official kernels; a missing compatible backend/cubin raises an error.
The repository does not distribute FlashInfer source code or cubins.

## Validation

Run from the repository root. Tests cover `32/4` and `64/4`; benchmarks select configurations using actual head counts.

```bash
python -m pytest tests/inference/msa_v1/attention/decode/q8kv8 -q -s --msa-inference-suite=smoke
python -m pytest tests/inference/msa_v1/attention/decode/q8kv8 -q -s --msa-inference-suite=full
python -m pytest tests/inference/dequant/nvfp4_to_fp8/test_sparse_flashinfer.py -q -s
python -m benchmarks.inference.msa_v1.attention.decode.q8kv8.benchmark --suite full --num-q-heads 32 --num-kv-heads 4
```

Record the actual GPU used for validation. Report B200 and B300 results separately; measurements on one do not validate the other.
