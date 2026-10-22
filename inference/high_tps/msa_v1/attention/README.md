# M3 Decode Attention

English | [简体中文](README.zh-CN.md)

The unified entry is `inference.high_tps.msa_v1.attention.create_attention_runner`. It provides the TP4 MegaAttention backend for MiniMax-M3.1 decode:

| Backend | Execution |
| --- | --- |
| `mega` (default) | One kernel from input normalization through post-attention norm |

The lower-level `create_attention_plan` and `m3_decode_mega_atten` entry points remain available.

`AttentionConfig(enable_placement=False)` is the default for `mega`; the lower-level
`m3_decode_mega_atten` accepts the same keyword. This mode requires canonical hardware
cluster placement. Concurrent GPU work can change placement; a mismatch reports
`M3_DECODE_CANONICAL_PLACEMENT_REQUIRED` and a CUDA launch failure. Use
`enable_placement=True` when non-canonical placement must be supported. After a launch
failure, terminate the TP worker group and recreate its CUDA contexts. SGLang's plugin
enables placement unless no other stream runs kernels during the forward.

Placement and diagnostic-timeout modes select separate compiled kernels. Choose and warm up
the mode before Graph capture; changing it requires a new runner/plan and Graph capture.
Within each static mode, all supported KV lengths and page capacities reuse one compiled
kernel, with one kernel launch per call. A workspace may be shared sequentially across modes.

`AttentionPlan.run()` returns `(attention_residual, post_attention_norm)`, both BF16 `[Q,6144]`. Physical input and output storage remains eight rows; the final Q rows are valid. Create `plan_info` for eight physical query rows, including when Q<8. For Q<8, set the leading `8-Q` slot mappings to `-1`; these padded rows must not update the cache. The caller owns consistency of token positions, slot mapping, and causal lengths. Compact Q<8 input storage is not accepted.

## Inputs and lifecycle

Requires one request, Q=1–8, TP4, hidden=6144, 16 Q heads, 1 KV head, head_dim=128, page_size=128, and TopK=16. The query buffers use FP8 E4M3 `[8,16,128]`; K/V/KI caches use NVFP4 packed uint8 `[pages,1,128,64]` with one FP8 scale per 16 values. The `mega` backend also accepts page-strided cache views (see the [root README](../README.md#paged-cache-layout)). `layer_input` is contiguous BF16 `[8,6144]`; projection/OProj weights use 128×4-swizzled MXFP8 scales. The output is written to caller-provided BF16 `[8,6144]` buffers.

Create weights and a TP4 context outside CUDA Graph capture, then bind each slot's existing cache, metadata, and preallocated buffers with `runner.prepare(...)`. Preallocate the outputs and warm up each path before capture; the runner does not deliberately allocate or prepack in `plan.run()`. `AttentionWeights`, `AttentionCache`, and `AttentionBuffers` define the required tensors. The [backend test](../../../../tests/inference/high_tps/msa_v1/attn_backends/test_backends.py) shows complete allocations and calls.

```python
from inference.high_tps.msa_v1.attention import create_attention_runner, create_attention_tp_context

tp_context = create_attention_tp_context(
    backend="mega", group_id="decode-attention"
)
runner = create_attention_runner(
    weights=attention_weights, tp_context=tp_context,
    backend="mega",
)
plan = runner.prepare(
    q_len=q_len, layer_input=layer_input, positions=positions,
    slot_mapping=slot_mapping, cos_sin_cache=cos_sin_cache,
    cache=attention_cache, buffers=attention_buffers, plan_info=plan_info,
)
attention_residual, post_attention_norm = plan.run()
# Close tp_context collectively after Graphs and outstanding GPU work finish.
```

All tensors reside on the rank-local CUDA device, selected before calling. All four ranks must create compatible TP4 contexts collectively. Prepare metadata and outputs, then warm up and capture the Graph. Calls may modify K/V/KI cache and outputs; the operator never modifies weights. After all in-flight calls finish, the `mega` backend permits in-place updates of directly consumed weights and scales, consistently across TP ranks. Replacing storage requires a new plan and Graph capture. Graph input/output addresses remain stable. Calls sharing a workspace must execute sequentially. Destroy Graphs and wait for outstanding GPU work before calling `tp_context.close()` collectively.

The `mega` backend uses a low-latency TP4 mailbox collective inside its single kernel. Each rank's OProj output is rounded to BF16, summed in FP32 in rank order, and added to the residual before BF16 rounding and FP32 RMSNorm. Negative-zero communication values are canonicalized to positive zero. There is no alternate NVLS reduction backend for `mega`; CUDA multicast and a single-node NVLink-connected TP4 group are still required. The context allocates and initializes communication storage outside capture. Never reset or reuse that storage while any rank has outstanding work, and use ordinary completion dependencies between complete Attention calls.

To check the reduction rounding contract, cache isolation, and repeated Graph execution:

```bash
ALLOW_JIT=1 torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.attn_mega.test_mailbox --output /path/to/results
```

`AttentionConfig(wait_timeout_s=...)` sets a diagnostic deadline in seconds
for waits on another CTA or TP rank. The default 0 disables it. A nonzero value
selects a separate diagnostic binary; expiration traps and requires terminating
the TP worker group and recreating CUDA contexts. `enable_pdl` defaults to False;
this single-kernel path does not require PDL from an external producer.

See the [root README](../../../../README.md) for build instructions. To check MegaAttention with generated test inputs on four GPUs:

```bash
ALLOW_JIT=1 torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.attn_backends.test_backends --q 8 --kv-len 8192
```
