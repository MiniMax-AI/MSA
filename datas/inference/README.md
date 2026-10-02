# Inference workloads

[Simplified Chinese](README.zh-CN.md)

This directory stores normalized prefill workloads derived from local
inference shape dumps. It is the shared data source for MSA v1 Attention
and Indexer tests and benchmarks and does not contain tensor payloads.

## Data semantics

Only records with `batch_type == "prefill"` are processed. The original
`kv_lens` represents the cached prefix before the current prefill call:

```text
final_kv_lens[i] = prefix_lens[i] + query_lens[i]
query_position    = prefix_lens[i] + q_idx
local_page        = floor(query_position / 128)
```

The 10,937 original prefill records normalize to 10,562 unique shapes with a
total call weight of 14,805. Original dumps are excluded by `.gitignore`; the
committed manifests do not store DP ranks, source file names, or source file
hashes.

## Files and schema

- `prefill_cases_v1/part-*.jsonl`: unique shapes, call weights, and shape
  metrics, deterministically split into 4,096-line shards. Files are loaded in
  file-name order as one logical manifest, and each shard is smaller than
  5 MiB.
- `prefill_cases_v1.meta.json`: schema, shard list and checksums,
  normalization rules, and aggregate counts.
- `prefill_test_cases_v1.json`: smoke, full, CUDA Graph, exhaustive, and
  operator-specific test tiers.
- `prefill_benchmark_cases_v1.json`: fixed 128-case benchmark selection whose
  representative weights sum to 14,805.

Each manifest case contains:

```text
schema_version
case_id
batch_size
query_lens
prefix_lens
final_kv_lens
count
metrics
```

`case_id` is a SHA256 prefix of the normalized shape JSON and is independent
of input file order. `metrics` stores workspace, FLOPs, and shape-related audit
metrics.

## Test tiers

`prefill_test_cases_v1.json` defines these shared tiers:

- `static_all`: schema, lengths, FLOPs, uniqueness, and selection-reference
  checks for all 10,562 cases.
- `msa_v1_smoke`: 32 deterministic real cases for MSA v1 quick correctness checks.
- `msa_v1_full`: 512 real cases for full-output MSA v1 correctness checks.
- `gemm_topk_e2e`: 1,024 end-to-end Indexer cases.
- `cuda_graph`: 16 cases covering different batches, workspace sizes, and
  workload imbalance.
- `exhaustive`: GPU execution over all 10,562 cases.

Tests derive reproducible inputs and randomized, non-contiguous physical page
tables from `case_id`; data files do not store tensor payloads.

## Benchmark selection

The MSA v1 `full` benchmark contains 128 representative, hot, batch anchor, and
stress cases. Weighted statistics use its production-representative cases with
`representative_weight`, whose weights sum to 14,805. The complete dataset contains
10,562 shapes for broader coverage checks. Statistics are computed as follows:

```text
weighted_mean_latency =
    sum(representative_weight[i] * latency[i])
    / sum(representative_weight[i])

aggregate_useful_TFLOPS =
    sum(representative_weight[i] * useful_flops[i])
    / sum(representative_weight[i] * latency_seconds[i])
    / 1e12
```

Cases with `useful_flops == 0` report latency without an individual TFLOPS
value.

## FLOPs convention

For query row `q_idx` in sequence `i`, the workload counts complete historical
pages before the forced tail:

```text
historical_pages(i, q_idx) = floor((prefix_lens[i] + q_idx) / 128)

useful_flops =
    2 * 128 * 128
    * sum_i sum_q_idx historical_pages(i, q_idx)
```

The first 128 is the page token count and the second is the head dimension.
TopK is not included in the TFLOPS count.

## Generation and validation

```bash
python3 datas/inference/generate_cases.py
python3 datas/inference/generate_cases.py --check
```

Tests and benchmarks read data through `datas.inference.cases`.

This prefill benchmark measures the complete public `run()` E2E path under CUDA
Graph replay, excluding `plan()`. Component timings do not replace E2E.
Prefill uses disjoint multi-tensor rotation with a reuse
distance greater than twice L2, 5 warmups, 20 replays, and a per-case
`CV <= 3%` requirement.

```bash
python3 -m benchmarks.inference.msa_v1.indexer.prefill.q8kv8.benchmark \
  --suite full --out /path/to/baseline.json

python3 -m benchmarks.inference.msa_v1.indexer.prefill.q8kv8.benchmark \
  --suite full --baseline /path/to/baseline.json --out /path/to/candidate.json
```

Use `smoke` for a quick check; `full` uses the complete 128-case benchmark selection.

Q8KV8 prefill indexer accepts `--num-index-heads` 1/2/4 and returns `[H, total_q, 16]`.
Useful FLOPs equal the single-head count above multiplied by H. Production-weighted
throughput is the weighted FLOPs sum divided by the weighted E2E time sum.
`--verify` independently checks every
head/query outside timing. Baseline and candidate must use the same device, DSL
version, timing protocol, and cold-cache slot count.
