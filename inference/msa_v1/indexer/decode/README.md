# Decode Block Scores

[简体中文](README.zh-CN.md)

## Purpose

`BatchDecodeBlockScoreWrapper` returns FP32 scores for logical blocks of 128 original
tokens. It accepts per-token or pooled keys and performs neither TopK selection nor
forced local-block inclusion. Existing decode indexer wrappers retain their TopK behavior.

## Public API

```python
import torch
from inference.msa_v1.indexer.decode.interface import BatchDecodeBlockScoreWrapper

q = torch.randn(2, 2, 2, 128, device="cuda", dtype=torch.bfloat16)
k_cache = torch.randn(4, 32, 128, device="cuda", dtype=q.dtype)
page_table = torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32)
seq_lens = torch.tensor([129, 256], device="cuda", dtype=torch.int32)
out = torch.empty(2, 4, 2, device="cuda", dtype=torch.float32)
scorer = BatchDecodeBlockScoreWrapper(input_dtype=q.dtype, pooling_factor=4)
scorer.plan(page_table, seq_lens, num_index_heads=2, query_length=2)
scores = scorer.run(q, k_cache, out=out)
assert scores.shape == (2, 4, 2)
```

Construction defaults to `input_dtype=torch.float8_e4m3fn`, `pooling_factor=1`,
`page_size=128`, `include_partial_block=False`, and `sm_scale=None`.
`plan()` defaults to `num_index_heads=1` and `query_length=8`.

## Data contract

Let P be the physical page capacity, F the pooling factor, and S the page size in
original tokens. F is 1, 2, or 4; S is 64 or 128. All requests share Q in 1–16 and
H in 1/2/4; keys have one shared head.

| Argument | Shape | Dtype |
| --- | --- | --- |
| `q` | `[B,Q,H,128]` | BF16 or E4M3 |
| `k_cache` | `[physical_pages,S/F,128]` | Same as Q |
| `page_table` | `[B,P]` | `torch.int32` |
| `seq_lens` | `[B]` | `torch.int32` |
| `out` | `[H,B*Q,ceil(P*S/128)]` | `torch.float32` |

`seq_lens` counts original tokens, including the current query chunk. Query i in
request b is at `p = seq_lens[b] - Q + i`. Stored key j represents F original
tokens and becomes visible only when `(j+1)*F <= p+1`. Incomplete groups do not
contribute. With `include_partial_block=False`, only blocks strictly before
`p // 128` are scored. With `True`, completed visible groups in the current block
are also scored. Empty and invalid blocks contain `-inf` on every call.

For each head, query and visible block, the result is `max(Q dot K) * scale`:

| `sm_scale` | BF16 scale | E4M3 scale |
| --- | --- | --- |
| Omitted or `None` | `1/sqrt(128)` | `1` |
| Explicit finite positive value | That value | That value |

Pass `sm_scale=1.0` for unscaled scores. Scaling is applied once after FP32 dot
products and block maxima. There is no softmax or key-scale input.

The capacity is at most 16384 logical blocks (2,097,152 original tokens); P may be
up to 16384 for page128 or 32768 for page64. Page64 combines two independently
mapped physical pages per logical block, and an odd P gives a final half-block.
Valid page IDs must address initialized cache pages. Lengths and mappings must
stay within the planned capacity; empty requests have no visible scores.

All tensors must reside on the same CUDA device. Q, metadata and output must be
contiguous. Q/K pointers and non-singleton K outer byte strides must be multiples
of 16; K's last dimension must have unit stride. Broadcast, overlapping and
negative-stride caches are unsupported.

## Runtime and CUDA Graphs

Requires SM100/SM103 and `nvidia-cutlass-dsl[cu13]>=4.5.2`. Construct and call
`plan()` outside capture, then warm up `run()` with the intended Q/K layout.
Capture requires an explicit preallocated `out`; omitting it outside capture
allocates a new result on each call. Each call overwrites the complete output.

Call `plan()` again after length changes, or bind a `BatchDecodeIndexerPlan` from
`inference.msa_v1.indexer.decode`. The shared plan must use matching
`max_pages=P`, H, Q, F, `page_size`, and `include_partial_block`.
Call its `update()` after in-place length changes and before consumers;
`update()` can be captured. In-place page mapping updates must finish before
consumers execute. Order cross-stream updates and consumers with events.

```python
from inference.msa_v1.indexer.decode import BatchDecodeIndexerPlan

plan = BatchDecodeIndexerPlan(
    seq_lens, max_pages=2, num_index_heads=2, query_length=2,
    pooling_factor=4, page_size=128, include_partial_block=False,
)
plan.update()
scorer.plan(page_table, seq_lens, num_index_heads=2, query_length=2, shared_plan=plan)
scorer.run(q, k_cache, out=out)  # Warm up before capture.
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    plan.update()
    scorer.run(q, k_cache, out=out)
graph.replay()
```

Keep the wrapper, plan, tensors and output alive while their Graph is used.
Use separate wrappers and writable workspaces for concurrent execution. Changes
to batch, Q/H, geometry, metadata bindings, or captured tensor addresses/layouts
require matching planning and recapture.

## Validation commands

```bash
FMHA_SM100_ALLOW_JIT=1 python -m pytest tests/inference/msa_v1/indexer/decode/test_block_scores.py --msa-inference-suite=full
FMHA_SM100_ALLOW_JIT=1 python -m pytest tests/inference/msa_v1/indexer/decode --msa-inference-suite=full
python -m pytest tests/inference/test_blackwell_compile.py -k 'decode_indexer or decode_shared_plan'
```
