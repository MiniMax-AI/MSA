# Multi-head Q8KV4 Decode Indexer

[简体中文](README.zh-CN.md)

## Purpose

Paged decode indexer for SM100 and SM103. Q uses E4M3, K uses packed E2M1 with E4M3 scales,
and the output contains selected logical page indices for each query, including the local page.

## Public API

```python
from inference.msa_v1.indexer.decode.q8kv4 import (
    BatchDecodeIndexerWithPagedKVCacheWrapper,
)

wrapper = BatchDecodeIndexerWithPagedKVCacheWrapper()
wrapper.plan(page_table, seq_lens, num_index_heads=4, query_length=q.shape[1])
topk_indices = wrapper.run(q, packed_k_cache, k_scale=k_scale)
```

Use `workspace_size(batch_size)` to query the required workspace size and optionally provide an
external workspace. Set the current CUDA device to the target device before querying; capacity
includes that device's scheduling metadata. Query again after upgrading instead of reusing an old
fixed byte count. Without an explicit shared plan, call `plan()` again when lengths or page
mappings change. Shared-plan updates are described below.
CUDA Graph mode also requires address-stable
`page_table_buffer` and `seq_lens_buffer` tensors at wrapper construction and a preallocated
output passed to `run()`.

### Shared plan and Graph updates

`query_length` accepts integers from 1 through 16 and defaults to 8. All requests in a batch
use the same Q. Lengths must satisfy `Q <= seq_lens[b] <= max_pages * 128`.

Changing Q or H may trigger compilation; warm up the required configurations before Graph capture.

Import `BatchDecodeIndexerPlan` from `inference.msa_v1.indexer.decode` and construct it outside capture:

```python
from inference.msa_v1.indexer.decode import BatchDecodeIndexerPlan

shared_plan = BatchDecodeIndexerPlan(
    seq_lens, max_pages=page_table.shape[1], num_index_heads=4, query_length=q.shape[1]
)
wrapper.plan(
    page_table, seq_lens, num_index_heads=4, query_length=q.shape[1], shared_plan=shared_plan
)
shared_plan.update()
```

Construct the plan, bind wrappers, and compile the first `run()` outside capture.
Capture `shared_plan.update()` followed by every consuming layer's `run()` in a CUDA Graph.
Replay refreshes the plan after in-place changes to `seq_lens`. Layers may use different page tables,
but must share B, Q, H, page capacity, source-length address, and device.
Do not additionally pass `workspace_buffer` to a wrapper using a shared plan.
In shared mode, update a bound page table in place before consumers run; changing only its
contents does not require rebinding. Rebind outside capture when metadata addresses or shapes
change. Create a matching new plan when B, Q, H, or page capacity changes, and recapture Graphs
that used the previous bindings.

Call `update()` before the first consumer and once per step when lengths change; do not reuse stale
results across steps. For multiple streams, use events to order consumers after the update and the
next update after all consumers. Keep plans and wrappers alive for the lifetime of their Graphs.
Rebinding and allocation are not supported inside capture.

## Data contract

- `q`: `[B, Q, H, 128]`, E4M3.
- `packed_k_cache`: `[physical_pages, 128, 64]`, packed E2M1.
- `k_scale`: `[physical_pages, 128, 8]`, E4M3 in a linear, non-swizzled layout.
- `page_table`: `[B, max_pages]`, a CUDA `torch.int32` logical-to-physical page mapping.
- `seq_lens`: `[B]`, CUDA `torch.int32`, including the current Q-token query chunk.
- Output: `[H, B * Q, 16]`, `torch.int32` logical page indices. Valid entries form a prefix, and
  the final valid entry is the local page.

All input tensors must be contiguous and on the same CUDA device. A historical page score is the maximum dot product between its tokens and the corresponding Q head. Each head selects historical candidates independently; unused output slots are filled with `-1`.

`packed_k_cache` is stored as `torch.uint8`. Dequantization computes E2M1 × scale and rounds with saturation to E4M3 before the dot product, which uses FP32 accumulators.

## Runtime requirements

- For query `q_idx`, the local page is `(seq_lens[b] - Q + q_idx) // 128`.
- Historical pages may map to scattered and unordered physical pages.
- Call `plan()` outside CUDA Graph capture.
- Toolchains that support QMUL4 use it automatically; otherwise the exact FP16 fallback
  supported by CUDA Toolkit 12.9 is selected. Callers do not choose the backend.

`num_index_heads` accepts only 1/2/4; the default is 1. Heads independently select historical pages and share the single-head K cache. H=1 also requires four-dimensional Q and returns a three-dimensional output. Preallocated `out` must be contiguous int32 on the same device with shape `[H,B*Q,16]`.

## Validation and performance testing

```bash
MINIMAX_INFERENCE_TEST_SUITE=smoke python -m pytest tests/inference/msa_v1/indexer/decode/q8kv4
MINIMAX_INFERENCE_TEST_SUITE=full python -m pytest tests/inference/msa_v1/indexer/decode/q8kv4
python -m benchmarks.inference.msa_v1.indexer.decode.q8kv4.benchmark --suite full --num-index-heads 4 --verify --out result.json
```

### Low-latency benchmark

`--suite low-latency` independently covers batch={1,2,4} × KV length={1000,4000,8000,32000}:
12 cases with Q=8 by default, configurable through `--query-length`. Batch=1 uses the exact nominal length; batch=2/4 use
true variable-length metadata with the nominal mean. Both use the same public interface.

This suite is separate from the 28-case `full` production-weighted statistics.
`--baseline` compares results with matching H, device, DSL version, and timing protocol.
The suite uses disjoint-page cold-cache rotation (reuse distance≥2×L2), 5 warmups,
20 replays, and 120 calls per Graph. `--verify` checks all scores and TopK outputs.

```bash
python -m benchmarks.inference.msa_v1.indexer.decode.q8kv4.benchmark --suite low-latency --num-index-heads 1 --verify --out low_latency.json
python -m benchmarks.inference.msa_v1.indexer.decode.q8kv4.benchmark --suite low-latency --num-index-heads 1 --verify --baseline low_latency.json --out candidate.json
```

Use `--query-length` to select Q=1–16. Results for different Q values are reported separately.
Shared-plan tests and separate timing scopes:

```bash
python -m pytest tests/inference/msa_v1/indexer/decode/test_shared_plan.py
python -m benchmarks.inference.msa_v1.indexer.decode.plan --precision q8kv4 --num-index-heads 4 --query-length 8 --layers 1 --out plan.json
```

The shared-plan benchmark covers the same 12 low-latency cases and records host+device update,
Graph update, Graph run, and Graph update+run separately. `--layers` explicitly selects an operator
layer count, not a model configuration. Graph latency covers one complete invocation of these layers
and is not divided by the layer count.
Standalone update timing uses hot metadata caches. Run and update+run use disjoint K rotation
with a K reuse distance of at least 2×L2. These cache conditions are reported separately.
If host+device timing still exceeds CV=3% after three attempts, it is marked invalid and
the command fails. Independently valid Graph timings remain in the output file.
