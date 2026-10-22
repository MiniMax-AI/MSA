# MiniMax-M3.1 TP4 decode

[简体中文](README.zh-CN.md)

This package provides model-level decode paths for one request, physical
query capacity eight, hidden size 6144, TP4, sparse topK 16, and paged NVFP4
K/V/KI cache. It does not support EP or multiple independent requests in one
layer call. See the [Attention](attention/README.md) and [MoE](moe/README.md)
contracts for buffer shapes, weight layouts, and backend-specific dependencies.
The DSpark draft model's five-layer backbone (seven-row draft block, one
launch per rank) is documented in [draft](draft/README.md).

The public backends are TP4 MegaAttention (Q=1–8) and TP4 MegaMoE (Q=8),
each launching one kernel per rank. Their Q=8 decode-layer pipeline launches two kernels.

`create_decode_layer_runner(structure="module-pipeline", attention=..., moe=...)`
accepts the two prepared runners. `plan.run()` returns BF16 `[8,6144]`
`next_hidden`, owned by the MoE TP4 workspace. Standalone Attention with Q<8
uses eight physical rows and `slot_mapping=-1` for the leading `8-Q` rows.
All `positions` must index the supplied RoPE cache.

Prepare weights, TP4 communication, metadata, cache, and graph-stable buffers
outside capture. `runner.prepare(...)` binds one slot; `plan.run()` enqueues
only the selected CUDA work and updates the paged cache. The caller keeps
weights, plan, buffers, and TP4 context alive until graph destruction and
outstanding GPU work complete, then closes the context collectively. Do not
share a writable workspace between concurrent streams or graphs.

### Paged cache layout

`AttentionCache` (and the `k`/`v`/`indexer_k` operands of the Mega kernels)
holds `P` physical pages per component. `k`, `v`, and `ki` pages are
`[128,64]` packed E2M1 bytes (`[P,1,128,64]`; `ki` may also be `[P,128,64]`);
`k_scale`, `v_scale`, and `ki_scale` pages are `[128,8]` E4M3 bytes
(`[P,1,128,8]`; `ki_scale` may also be `[P,128,8]`). Rows are dense inside a
page. `stride(0)` is the page stride in bytes and may be any multiple of 16 in
`[dense page bytes, 2^40)` (dense page bytes: 8192 for data, 1024 for scales);
`k` and `v` additionally need a 32-byte multiple page stride and a 32-byte
aligned base (packed-FP4 TMA requirement). Strides are chosen
independently per component, so the caches may be views into larger per-page
slabs (for example one 18432-byte row holding k, k_scale, v, v_scale). Dense
tensors remain valid. MegaAttention write only the new
rows selected by `slot_mapping` (slot = page·128 + row) and never touch bytes
between components.

### Capacity-sized plans for CUDA graphs

MegaAttention reads the current KV length from `plan_info.kv_segment_lens` on
every call. Create the plan with the page-table capacity (up to 1048576 tokens),
then update that device tensor and the page mapping in place before replay.
Entries beyond the visible length are not read. All legal KV lengths and page
capacities reuse one compiled kernel for each static placement/diagnostic mode.
MegaAttention defaults to `AttentionConfig(enable_placement=False)`, requiring canonical
hardware placement; enable it for non-canonical placement, as the SGLang plugin does.
See the [Attention contract](attention/README.md) for failure handling and Graph mode changes.
Wave-balanced dispatch is disabled. Selection returns
the exact FP32 top 15 historical pages plus the local page, with page-ID tie
breaking and ascending valid IDs; short rows use `-1` padding.

One MegaAttention Graph node can be replayed repeatedly on the same workspace,
including long/short KV switches. Calls sharing a workspace must be serialized.
Q=1–8 remains supported through physical eight-row padding; padded slots must
be `-1`. Attention weights may be updated in place after all in-flight calls
finish. Replacing their storage requires a new binding and Graph capture.

The optional [SGLang plugin](../../../integrations/sglang/README.md) integrates
single-request eight-token chain verification. It does not support service-side
online weight updates.

The package requires a CUDA toolkit that accepts the QMUL4 instruction,
PyTorch, TVM-FFI, Jinja2, and the repository CUTLASS submodule.
TP4 MegaAttention and MegaMoE do not require FlashInfer. Import and plan construction are lazy with
respect to backend compilation. On a fresh installation, set `ALLOW_JIT=1`
while warming up a Mega backend; later runs may reuse its compiled cache.

The [TP4 test fixture](../../../tests/inference/high_tps/msa_v1/common/m3_decode_fixture.py)
contains complete tensor construction. Run the two-kernel integration smoke check with
four GPUs after installing the backend dependencies:

```bash
ALLOW_JIT=1 torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.integration.test_attn_mega_moe_mega --q 8 --kv-len 8192 --output /tmp/msa-tp4-check
```
