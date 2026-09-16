# MSA v1 Inference Operators

[简体中文](README.md)

## Overview

MSA v1 provides decode and prefill APIs for paged sparse attention and indexers. Every public
operator follows a `plan()` / `run()` lifecycle: `plan()` accepts request-level metadata, and
`run()` accepts tensors for the current layer and returns results.

## Public APIs

| Operator | Package | Public wrapper |
| --- | --- | --- |
| Decode attention | `inference.msa_v1.attention.decode.<dtype>` | `BatchDecodeWithPagedKVCacheWrapper` |
| Prefill attention | `inference.msa_v1.attention.prefill.<dtype>` | `BatchPrefillWithPagedKVCacheWrapper` |
| Decode indexer | `inference.msa_v1.indexer.decode.<variant>` | `BatchDecodeIndexerWithPagedKVCacheWrapper` |
| Prefill indexer | `inference.msa_v1.indexer.prefill.<variant>` | `BatchPrefillIndexerWithPagedKVCacheWrapper` |

Available formats and constraints:

- Attention: BF16 prefill, Q8KV4 decode/prefill, and Q8KV8 decode/prefill.
- Indexer: BF16 prefill, TP4 Q8KV4 decode, and TP4 Q8KV8 decode/prefill.
- BF16 paged prefill attention accepts contiguous and SGLang-style strided K/V views.
- BF16 paged prefill indexer supports one or four local index heads and returns
  `[num_index_heads, total_q, 16]`.
- Q8KV4 and Q8KV8 decode attention support GQA=8/16 on B200/SM100 and B300/SM103,
  dispatch by the actual `Hq/Hkv`, and return BF16. QLen=8 represents one main token and seven
  predicted tokens. Q8KV4 uses native CUTLASS C++ and retains SM107 GQA=16 support with
  CUDA 13.5 or newer.
- Q8KV8 prefill attention supports GQA group sizes 1, 2, 4, 8, and 16.

See the README in each operator directory for exact tensor shapes, dtypes, and usage examples.

## Common data contract

- Only paged KV caches are supported. Varlen requests must provide their real length metadata.
- TopK logical pages may be scattered and unordered; callers must not assume they are
  contiguous or sorted.
- Valid TopK entries form a prefix followed by `-1`, and the final valid entry must be the local
  page.
- Call `plan()` outside CUDA Graph capture. Paths used with graph capture must be warmed up
  first and use preallocated outputs during capture.
- Wrappers do not silently sort TopK entries or copy and reorder K/V to bypass input
  restrictions.

## Optional dependencies and format conversion

`inference.dequant` provides general NVFP4-to-E4M3 conversion, including dense conversion and
sparse K/V conversion that processes only TopK-selected pages.

Q8K8 decode attention uses an external FlashInfer backend. This repository does not distribute
FlashInfer source code or cubins. Install the pinned `flashinfer-python==0.6.17` dependency with
`python -m pip install -e '.[flashinfer]'`. The adapter does not perform implicit FP4 dequantization.

## Validation

- Decode correctness: a 96-case smoke suite and a 256-case full suite.
- Prefill correctness: a 32-case smoke suite and a 512-case full suite.
- Formal correctness compares every element of public outputs and required auxiliary outputs
  against an independent reference.
- Formal decode benchmarks use 28 RL-rollout cases; prefill uses 128 fixed production cases.
- The formal performance metric is end-to-end latency of the public `run()` path inside a CUDA
  Graph.

See the repository [README](../../README.en.md),
[inference data manifests](../../datas/inference/README.en.md), and
[inference rules](../AGENTS.en.md) for shared cases, commands, and acceptance rules.
