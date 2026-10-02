# BF16 Sparse Decode Attention

[简体中文](README.zh-CN.md)

## Purpose

Paged sparse attention with BF16 Q/K/V and BF16 output, using the external FlashInfer
TRTLLM-GEN block-sparse backend. Each query and KV head has its own selected logical pages.

## Public API

```python
import torch
from inference.msa_v1.attention.decode.bf16 import BatchDecodeWithPagedKVCacheWrapper

query_length = 2
q = torch.randn(4, 16, 128, device="cuda", dtype=torch.bfloat16)
k_cache = torch.randn(4, 2, 128, 128, device="cuda", dtype=torch.bfloat16)
v_cache = torch.randn_like(k_cache)
page_table = torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32)
seq_lens = torch.tensor([256, 256], device="cuda", dtype=torch.int32)
topk_indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32).repeat(4, 2, 1)
out = torch.empty_like(q)
wrapper = BatchDecodeWithPagedKVCacheWrapper(enable_pdl=True)
wrapper.plan(
    topk_indices, page_table, seq_lens,
    q_len_per_req=query_length,
    num_q_heads=q.shape[1],
    num_kv_heads=k_cache.shape[1],
)
output = wrapper.run(q, (k_cache, v_cache), out=out)
assert output.shape == q.shape
```

`enable_pdl` is optional and defaults to `True`; set it to `False` to disable programmatic
dependent launch. `sm_scale` optionally overrides the default `1/sqrt(128)` attention scale.

Omitting `out` returns a wrapper-owned default output that later `run()` calls overwrite.
Use a separate output to retain results, and separate wrappers and writable workspaces for
concurrent calls.

## Data contract

| Argument | Shape | Dtype |
| --- | --- | --- |
| `q` | `[B*Q,num_q_heads,128]` | `torch.bfloat16` |
| `k_cache`, `v_cache` | `[physical_pages,num_kv_heads,128,128]` | `torch.bfloat16` |
| `topk_indices` | `[B*Q,num_kv_heads,K]`, K=1–16 | `torch.int32` |
| `page_table` | `[B,max_pages]` | `torch.int32` |
| `seq_lens` | `[B]` | `torch.int32` |
| `out` | Same shape as `q` | `torch.bfloat16` |

All tensors must reside on the same CUDA device. Q, metadata, and output must be contiguous.
K/V may be strided across pages and heads; their token and head-dimension strides must be 128 and 1, and K/V shapes and strides must match. Q/K/V and output require 16-byte alignment.
Each KV head serves 8 or 16 Q heads. Sequence lengths include the Q-token query chunk, shared
by all requests. Query position is `seq_lens[b] - Q + t`.

TopK contains logical page IDs, with a valid prefix and `-1` padding. Historical pages are
unique and may appear in any order. The final valid page must be the query's local page.
The indexer's head-major `[H,B*Q,16]` output requires an explicit conversion to the attention
metadata layout; Q/K/V data need no such conversion.

`plan()` defaults to `num_q_heads=64` and `num_kv_heads=4`; `q_len_per_req` is a required
positive integer, and every `seq_lens` entry must be at least Q. Each row must contain exactly
`min(K, local_page + 1)` valid TopK entries. Valid page-table mappings must refer to physical
pages within the K/V cache.

## Runtime requirements

Install the optional dependency with `python -m pip install -e '.[flashinfer]'`
(`flashinfer-python==0.6.17`). Its public TRTLLM-GEN kernel artifacts must
be available to the runtime. If `flashinfer-cubin` or `flashinfer-jit-cache` is installed, its version must match
the Python package; FlashInfer reports a version error on a mismatch. The optional
`flashinfer-cubin==0.6.17` wheel is available from the
[FlashInfer wheel index](https://flashinfer.ai/whl):

```bash
python -m pip install flashinfer-cubin==0.6.17 --index-url https://flashinfer.ai/whl
```
Target GPUs are GB200/B200 (SM100) and GB300/B300 (SM103).

Call `plan()` outside Graph capture whenever TopK, page mappings, or sequence lengths change.
Warm up `run()` before capture and supply a preallocated `out` during capture. Keep the wrapper
alive while its Graph is used. Use separate wrappers for concurrent independent requests or
streams. This interface returns the attention output only; it does not expose LSE.

## Validation commands

```bash
MINIMAX_INFERENCE_TEST_SUITE=smoke python -m pytest tests/inference/msa_v1/attention/decode/bf16
python -m benchmarks.inference.msa_v1.attention.decode.bf16.benchmark --suite full --out bf16_attention.json
```

The benchmark reports CUDA Graph latency for the public `run()` path, excluding planning, allocation, compilation, and warmup. It checks the complete output against an independent reference.
Pass `--disable-pdl` to measure the same path with programmatic dependent launch disabled.
