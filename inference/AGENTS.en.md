# Inference Interface Guidelines (English)

The Simplified-Chinese [`AGENTS.md`](AGENTS.md) is normative. This English
companion must be updated whenever the normative contract changes.

## Scope and public interface

These rules apply to public Python wrappers, runtime integration, and GPU
kernel interfaces under `inference/`. Kernel algorithms, numerical precision,
source style, and validation also follow the repository-root `AGENTS.md`.

Inference attention, indexer, and KV-cache APIs should follow FlashInfer's
public lifecycle and parameter semantics whenever possible. Reuse an existing
wrapper when only dtype, quantization, or backend scheduling changes; do not
create a duplicate public API for the same operation. Backend kernel ABI does
not need to copy FlashInfer mechanically. Document unavoidable differences.

## `plan()` and `run()`

- Use a two-stage lifecycle when request structure or scheduling can be reused
  across transformer layers or calls.
- `plan()` receives reusable request metadata such as variable-length
  information, page tables, sparse TopK indices, mask/window configuration,
  workspace capacity, and maximum sequence lengths. It owns preparation,
  scheduling, workspace sizing, and backend selection. CSR, worklists,
  barriers, split/combine state, and other backend details remain opaque.
- `run()` receives data that changes per layer or call: Q/K/V, quantization
  scales, output tensors, and a small number of runtime scalars. It must not
  read device tensors back to the host for launch decisions. It supports
  preallocated outputs/workspaces and CUDA Graph capture.
- Any one-time host planning or D2H synchronization is outside capture and is
  documented. Prefer an asynchronous device prepare path when available.
- A stateless kernel with no reusable planning state need not expose a fake
  two-stage API.
- Formal performance includes only the full public `run()` device path replayed
  from a captured CUDA Graph. It excludes `plan()`, allocation, input
  generation, H2D copies, graph capture, correctness reference, and warmup.
  Private stage or main-kernel latency cannot replace this E2E metric.

Each metadata item has one authoritative source. Decode request lengths are
named `seq_lens`; varlen prefill uses `cu_seqlens_q` and `cu_seqlens_k`.
Do not scan TopK padding, page-table contents, or physical page order to derive
an explicitly supplied length. Paged KV always uses explicit logical-to-
physical mapping and cannot assume sorted, contiguous, or identity page IDs.

Public `run()` accepts preallocated output and auxiliary tensors. Allocation is
allowed as a convenience path only outside CUDA Graph capture. Plan/workspace
ownership must not create hidden allocation or synchronization inside `run()`.
Compile/cache keys contain only stable code-generation properties; request
sizes and tensor values stay runtime data.

## MSA v1 acceptance contract

The following correctness and benchmark rules apply to `inference/msa_v1/`.
They do not replace the training suites.

Formal performance for every MSA v1 operator is CUDA Graph E2E latency of the
public `run()` path. Run a fixed baseline before the candidate, with identical
inputs, graph configuration, cold-cache method, warmup, replay count, and
statistics. The target GPU must have no other compute process. Do not lock
clocks. A valid case has `CV <= 3%`; retry only that case, at most three times.
Every case may regress by at most 5%. Acceptance also requires the weighted
arithmetic mean of per-case E2E medians to be strictly better than baseline;
there is no additional minimum improvement percentage. The user decides
whether to accept and merge the complete result.

MBU and MFU are diagnostic roofline indicators. Use MBU for memory-bound
stages and MFU for compute-bound stages to explain the remaining ceiling, but
never substitute either for formal E2E acceptance.

## Shared MSA v1 prefill cases

- Correctness source of truth:
  `datas/inference/generate_cases.py` and its generated manifests.
- The formal MSA v1 suite has 512 deterministic cases; smoke is a stable
  32-case subset. Both use the same generator and seed.
- Compare the complete public output tensor and all required auxiliary outputs
  with an independent reference. Smoke does not require repeat execution.
  Deterministic formal cases run three times and require bitwise-identical
  complete outputs.
- The canonical prefill benchmark manifest contains 128 fixed production
  cases. Execute and report every case, then compute the production-weighted
  arithmetic mean of E2E medians.
- Cold cache uses a bank of disjoint input/output tensor sets whose touched
  footprint exceeds twice L2. Capture the public path for each slot, rotate
  through the bank before reuse, and exclude allocation/rotation setup from
  E2E.
- After a candidate passes the latency gate, independently validate every
  benchmark case and run smoke. Before final commit, run the 512-case full
  correctness suite.

## Shared MSA v1 decode cases

- Correctness source of truth: `tests/inference/msa_v1/decode/cases.py`.
- Full correctness contains exactly 256 deterministic cases with
  `seed=1701`: 75% production distribution and 25% robustness coverage. Smoke
  is a stable 96-case subset generated by the same source.
- Coverage includes `q_len_per_req` in `{1, 2, 4, 8, 16}`, batch samples over
  `[1, 512]`, sequence lengths from 1K to 512K with a 100K production hot spot,
  page/local-block boundaries, varlen requests, sparse unordered pages, TopK
  cardinalities, and edge values. Fixed-query-length operators reuse the
  batch/sequence/page metadata while retaining their public fixed Q length.
- Formal correctness compares the complete public outputs and necessary
  auxiliaries. Deterministic full cases run three times and require bitwise-
  identical complete outputs. Smoke still uses a whole-output independent
  reference, but does not run the three-repeat bitwise check.
- Benchmark source of truth: `benchmarks/inference/msa_v1/decode/cases.py`.
  It contains the full Cartesian product `batch_size in {8,32,64,128}` and
  nominal `seq_len in {1K,4K,5K,10K,50K,100K,200K}`, with fixed
  `q_len_per_req=8` and true per-request varlen lengths: 28 cases total.
- Production-frequency weights sum to 10,000 and encode the negative
  batch/sequence-length correlation used by rollout. All 28 cases have a
  positive weight, are reported individually, contribute to the weighted
  arithmetic mean, and obey the 5% per-case regression gate.
- Every decode benchmark uses cold KV data. Rotate disjoint physical-page
  regions with a total touched footprint greater than twice L2. If one graph
  cannot contain the complete bank while preserving the fixed calls-per-graph
  protocol, capture multiple graphs and treat one replay of the entire bank as
  one timing sample. Cold-cache setup is excluded from E2E.
- Use five warmup replays, 20 measured replay samples, and 120 public `run()`
  calls per captured graph. Report medians and CV; never use a best run.
- After a candidate passes the latency gate, independently validate complete
  outputs for all 28 benchmark cases and run smoke. Before final commit, run
  the 256-case full correctness suite.

The exact decode production-weight matrix is:

```text
SeqLen    B8    B32   B64   B128
1K        30    90    180   300
4K        40    120   240   400
5K        50    150   300   500
10K       140   350   490   420
50K       500   800   500   200
100K      1260  980   420   140
200K      910   350   112   28
```

## Q8KV4 paged prefill

The public wrapper is `BatchPrefillWithPagedKVCacheWrapper`. It owns an opaque
plan containing varlen query/KV metadata, page mapping, sparse indices, and
workspace. `run()` receives current Q, paged FP4 K/V data, K/V scales, and
preallocated output/LSE when capture is required. Tensor Core accumulators,
softmax, reduction, and correction are FP32; only dequantization may use half
precision. GMEM K/V data and scales follow the public linear, non-swizzled
layout; internal SMEM/TMEM transforms are allowed.

## Q8KV4 paged decode attention

The public wrapper is `BatchDecodeWithPagedKVCacheWrapper`. One invocation has
a single host scalar `q_len_per_req`; `seq_lens[b]` includes the current decode
or MTP chunk. For request `b` and query index `q_idx`:

```text
query_position = seq_lens[b] - q_len_per_req + q_idx
local_block = query_position // 128
valid_count = min(16, local_block + 1)
```

Valid TopK entries form a prefix, the final valid entry is `local_block`, and
padding is `-1`. Page IDs within the valid prefix may be sparse and unordered.
The kernel computes `valid_count` from `seq_lens`, `q_len_per_req`, and
`q_idx`; it must not scan TopK. Only `local_block` receives the causal/length
mask. The public contract is paged KV only, with linear non-swizzled FP4 data,
K scales, and V scales in GMEM.

## Decode indexer

Q8KV4 and Q8KV8 expose `BatchDecodeIndexerWithPagedKVCacheWrapper`. Planning
owns the page table, `seq_lens`, scheduler state, and workspace. `run()` owns
the public proxy-score plus forced-tail TopK path and accepts preallocated
output for graph capture. Proxy scores are a private intermediate: they may be
profiled separately for MBU/MFU diagnosis but are not a public performance
metric.

The public packages are `inference.msa_v1.indexer.decode.q8kv4` and `q8kv8`.
The keyword-only `plan()` argument `num_index_heads=1` accepts H=1/2/4.
The keyword-only `query_length` accepts any integer Q=1..16, shared by all
requests in the batch. Q may be compile-time: Q/H specialization is allowed when
it changes MMA tile shape or other generated-code configuration. Dimensions that
do not change code generation remain runtime. With a fixed static configuration,
changes to batch, KV length, and metadata must reuse compiled artifacts. Never
read device tensor contents to construct a compile key.
An explicit `BatchDecodeIndexerPlan` may share read-only metadata within the same
step. Its `update()` runs on the current stream and supports capture; construction
and binding happen outside capture. Callers order cross-stream consumers and
subsequent updates with events. Do not reuse stale metadata across steps.
Without an explicit shared plan, call `plan()` again when lengths or page mappings change.
With a shared plan, call `update()` after in-place length changes; bound page tables may be
updated in place before consumers run. Rebind outside capture when metadata addresses or
shapes change. Create a matching new plan when B, Q, H, or page capacity changes, and
recapture Graphs that used the previous bindings.
The internal TopK stage serves both precisions and all supported local index head counts;
it does not accept a head-count parameter.
Q is contiguous E4M3 with shape `[B, Q, H, 128]`; heads compute independently
and share the K cache. The public result has shape `[H, B * Q, 16]`, dtype
`int32`, and is contiguous. Valid entries form a prefix; the final valid output
is the local block and any suffix is `-1`. TopK ordering is score-descending
apart from the forced local tail contract. Q8KV8 does not expose an unused
`k_scale` argument. Avoid public aliases, legacy tuples, or historical modes
that have no production caller and no formal test.

## Migration and compatibility

New inference code follows these contracts directly. Existing code is cleaned
incrementally: remove dead compatibility fields and paths only with caller and
test evidence, update tests and README in the same change, and do not combine
an interface cleanup with an unrelated kernel schedule rewrite. Preserve the
real varlen/page/TopK semantics throughout migration.
