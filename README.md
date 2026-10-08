# MiniMax MSA

[Simplified Chinese](README.zh-CN.md)

## 1. Overview

MiniMax Sparse Attention (MSA) uses an Indexer to select TopK blocks for each
query, then computes Sparse Attention over those blocks. This branch provides
MSA v1 training and inference operators for NVIDIA Blackwell GPUs, covering
BF16, FP8 E4M3, and NVFP4 data paths.
Algorithm reference: [MiniMax Sparse Attention paper](docs/MiniMaxSparseAttention.pdf).

Training uses `msa_v1`, inference uses `inference.msa_v1`, and quantized-format
conversion uses `inference.dequant`. Training and inference have their own data
layouts and public interfaces. The supported operators are listed below.

### 1.1 Training operators

| Operator | Main inputs | Outputs | Supported scope |
| --- | --- | --- | --- |
| Sparse Attention | BF16 Q/K/V | BF16 O, FP32 LSE; BF16 dQ/dK/dV | Forward / Backward with variable-length sequences |
| Sparse Attention: FP8 forward | FP8 E4M3 Q/K/V | BF16 O, FP32 LSE | Forward only; no Backward for native FP8 Q/K/V |
| Sparse Attention: probability QAT | BF16 Q/K/V; `sparse_attn_p_mode="fp8"` | BF16 O, FP32 LSE; BF16 dQ/dK/dV | Forward / Backward with FP8 quantization-aware training of attention probabilities |
| Indexer (including TopK) | BF16 Q/K | INT32 TopK indices, FP32 selected LSE | Causal block selection for variable-length sequences |
| Tree Indexer (including TopK) | BF16 Q/K, precompiled mask plan | INT32 TopK indices, FP32 selected LSE | Batch-1 tree / custom visibility selection |
| Sparse KL Backward | BF16 teacher Q/K and indexer Q/K, FP32 LSE | BF16 indexer dQ/dK | Indexer gradients from the sparse KL objective |

LSE is returned according to interface options; the table does not imply that
every call returns LSE. Public training Attention uses head dimension=128,
64 Q heads, 4 KV heads, block size=128, and TopK=16.
See the [training interfaces](#31-msa-v1-training) for entry points.

### 1.2 Inference operators

| Operator | Supported input formats | Outputs | Use and constraints |
| --- | --- | --- | --- |
| Prefill Attention | BF16 Q/K/V; Q8KV8; Q8KV4 | BF16 O, optional FP32 LSE | Paged sparse causal attention with variable-length chunk prefill |
| Decode Attention | BF16 Q/K/V; Q8KV8; Q8KV4 | BF16 O by default; Q8KV4 optionally emits MXFP8 O | Paged sparse decode / MTP; BF16 and Q8KV8 require the optional FlashInfer dependency |
| Prefill Indexer (including TopK) | BF16 Q/K; FP8 E4M3 Q/K | INT32 logical page indices | BF16 and FP8 support 1/2/4 local index heads |
| Decode Indexer (including TopK) | BF16 Q/K; FP8 E4M3 Q/K; FP8 E4M3 Q + NVFP4 K | INT32 logical page indices | 1/2/4 local index heads and 1–16 queries per request |
| NVFP4 → FP8 conversion | Packed E2M1 data + E4M3 scales | FP8 E4M3 data | Dense conversion or conversion of only TopK-selected paged K/V |

See the [inference interfaces](#32-msa-v1-inference) for package paths and the
[inference documentation](inference/msa_v1/README.md) and its operator README
links for shape, scale, paging, and CUDA Graph contracts.

### 1.3 Data types

- **BF16**: `torch.bfloat16`.
- **FP8 E4M3**: `torch.float8_e4m3fn`; Q8/K8/V8 denotes floating-point FP8, not INT8.
- **NVFP4**: packed E2M1 data with E4M3 group scales; both must be supplied according
  to the operator contract.
- **Q8KV8**: FP8 E4M3 Q/K/V. **Q8KV4**: FP8 E4M3 Q with NVFP4 K/V.
  Indexers read only Q/K, not V.
- The tables describe public input/output formats. Tensor Core accumulation uses
  FP32; FP16 intermediate-storage options do not imply public FP16 Q/K/V support.

All varlen interfaces use real length metadata. Valid TopK entries retain sparse,
unordered semantics, with the local block in the last valid position.

## 2. Requirements and installation

| GPU | Architecture | Status |
| --- | --- | --- |
| NVIDIA GB200/B200 | SM100 | Supported |
| NVIDIA GB300/B300 | SM103 | Supported |

The repository's Blackwell kernels cover B200/SM100 and B300/SM103, including
inference and training. Runtime dispatch selects a compatible implementation;
each operator's existing dtype, shape, and data contracts still apply.

The minimum CuTe DSL version requirement is:

```text
nvidia-cutlass-dsl[cu13]>=4.5.2
```

Install the project with:

```bash
git submodule update --init --recursive
python -m pip install -e ".[test]"
```

When using a regular package installation, the C++ operators and Q8KV8 prefill indexer
still require public CUTLASS headers. Keep a CUTLASS checkout and set `CUTLASS_ROOT`.
For example, run from the repository root:

```bash
export CUTLASS_ROOT="$PWD/third_party/cutlass"
python -m pip install .
```

Rebuild AOT artifacts after upgrading DSL; do not reuse compiled
caches across DSL or CUDA backend versions. This project installs the cu13 backend
by default, including for CuTe DSL 4.5.2. Install and verify the loaded versions with:

```bash
python -m pip install "nvidia-cutlass-dsl[cu13]>=4.5.2"
python -c 'import cutlass; print(cutlass.__version__, cutlass.CUDA_VERSION)'
```

For 4.5.2, if verification still reports CUDA 12.9, reinstall its cu13 wheel last:

```bash
python -m pip install --force-reinstall --no-deps "nvidia-cutlass-dsl-libs-cu13==4.5.2"
python -c 'import cutlass; print(cutlass.__version__, cutlass.CUDA_VERSION)'
```

Q8KV4 C++ operators use the standard `CUDA_HOME`, `CUTLASS_ROOT`, and
`TORCH_EXTENSIONS_DIR` configuration. Runtime JIT always selects `sm_100a` or
`sm_103a` from the CUDA device of the input tensor. Tensor-less AOT/offline
builds use `MM_SPARSE_TARGET_ARCH=100a|103a` to select a target (default:
`103a`). `TORCH_CUDA_ARCH_LIST` is internal compiler plumbing rather than the
runtime architecture-selection interface. Q8KV4 Decode Attention, Decode
Indexer, and the shared dequant package support CUDA Toolkit 12.9 or newer.
They use QMUL4 when the selected toolchain supports it and otherwise select an
exact fallback automatically. Q8KV4 Prefill Attention requires CUDA Toolkit
13.4 or newer.

BF16 and Q8KV8 sparse decode are provided through an optional external FlashInfer
TRTLLM-GEN backend. This repository does not include FlashInfer source code or
cubins. Install `flashinfer-python==0.6.17` with `python -m pip install -e '.[flashinfer]'`.
Q8KV4 decode uses native CUTLASS C++. All three decode attention formats support GQA=8/16 and BF16
output on B200/B300, selecting the implementation by the actual `Hq/Hkv`.
Q8KV4 decode also retains SM107 GQA=16 support, requiring CUDA Toolkit 13.5 or newer.

Rubin (SM107) retains BF16/Q8KV8 prefill attention and Q8KV4 decode attention paths,
selected from the input tensor device. See the operator READMEs for dependencies and restrictions.
This does not imply Rubin support for the other operators.

## 3. Main interfaces

### 3.1 MSA v1 training

```python
from msa_v1 import attention, indexer, indexer_tree, kl

topk_indices, indexer_lse = indexer.forward(...)
metadata = attention.prepare(...)
out, lse = attention.forward(..., metadata, return_softmax_lse=True)
dq, dk, dv = attention.backward(..., out, lse, metadata)
dqi, dki = kl.backward(..., indexer_lse, metadata)
```

Each layer must call `attention.prepare()` with its current `topk_indices`. The
returned `AttentionMetadata` is reused only by the corresponding Attention
forward, Attention backward, and KL backward. Run `prepare()` again after TopK
changes. If TopK changes between CUDA Graph replays, capture `prepare()` so each
replay rebuilds the metadata.

MSA v1 Attention E4M3 probability paths, including `sparse_attn_p_mode="fp8"` QAT,
use `E4M3(P * 448)` with compensation in normalization and backward probability
reconstruction. LSE retains the natural-log semantics of the original logits, and
the logical softmax STE definition is unchanged. This contract does not guarantee
bitwise-identical final training and inference outputs.
QAT/FP8 paths and decode use hardware exp2; ordinary BF16 paths retain their existing
emulation settings. Scaling and compensation semantics are aligned; quantized P need not match bitwise across paths.

| Module | Public API |
| --- | --- |
| `msa_v1.indexer` | `prepare_indexer_schedule`, `forward` |
| `msa_v1.attention` | `prepare`, `forward`, `backward` |
| `msa_v1.indexer_tree` | `compile_plan`, `forward` |
| `msa_v1.kl` | `backward` |

See
[`training/msa_v1/indexer_tree/tree_indexer_usage.md`](training/msa_v1/indexer_tree/tree_indexer_usage.md)
for Tree Indexer usage.

### 3.2 MSA v1 inference

Decode indexers also expose `BatchDecodeIndexerPlan`: construct and bind it outside capture.
Its `update()` can run inside a Graph to refresh length metadata shared by layers in the same step.

MSA v1 inference operators use a common `plan()` / `run()` lifecycle.
`plan()` accepts request-level metadata, while `run()` accepts per-layer data.
Complete `plan()` and any required warmup before CUDA Graph capture, and use
preallocated outputs during capture.

| Operator | Package | Input format |
| --- | --- | --- |
| Decode Attention | [`inference.msa_v1.attention.decode.bf16`](inference/msa_v1/attention/decode/bf16/README.md) | BF16 Q/K/V; GQA=8/16 |
| Decode Attention | `inference.msa_v1.attention.decode.q8kv4` | E4M3 Q + NVFP4 K/V |
| Decode Attention | `inference.msa_v1.attention.decode.q8kv8` | E4M3 Q/K/V |
| Prefill Attention | `inference.msa_v1.attention.prefill.bf16` | BF16 Q/K/V |
| Prefill Attention | `inference.msa_v1.attention.prefill.q8kv4` | E4M3 Q + NVFP4 K/V |
| Prefill Attention | `inference.msa_v1.attention.prefill.q8kv8` | E4M3 Q/K/V |
| Decode Indexer | [`inference.msa_v1.indexer.decode.bf16`](inference/msa_v1/indexer/decode/bf16/README.md) | BF16 Q/K; H=1/2/4, Q=1–16 |
| Decode Indexer | `inference.msa_v1.indexer.decode.q8kv4` | E4M3 Q + NVFP4 K |
| Decode Indexer | `inference.msa_v1.indexer.decode.q8kv8` | E4M3 Q/K |
| Prefill Indexer | `inference.msa_v1.indexer.prefill.bf16` | BF16 Q/K |
| Prefill Indexer | `inference.msa_v1.indexer.prefill.q8kv8` | E4M3 Q/K |

Q8KV4 Decode Attention example:

```python
from inference.msa_v1.attention.decode.q8kv4 import (
    BatchDecodeWithPagedKVCacheWrapper,
)

wrapper = BatchDecodeWithPagedKVCacheWrapper()
wrapper.plan(
    topk_indices,
    page_table,
    seq_lens,
    q_len_per_req=q_len_per_req,
    num_q_heads=64,
    num_kv_heads=4,
)
out = wrapper.run(
    q,
    (packed_k_cache, packed_v_cache),
    kv_cache_sf=(k_scale, v_scale),
)
```

See each operator's README for its shape, dtype, TopK/page contract, and
optional arguments.

## 4. Correctness tests

Run from the repository root. Set `FMHA_SM100_ALLOW_JIT=1` to enable initial compilation; compilation is excluded from kernel execution timing.

```bash
export FMHA_SM100_ALLOW_JIT=1
python -m pytest tests/test_package_layout.py tests/test_training_workloads.py -q
python -m pytest tests/training/msa_v1/cute -q -s --msa-v1-suite=smoke
python -m pytest tests/training/msa_v1/cute -q -s --msa-v1-suite=full
python -m pytest tests/inference -q -s --msa-inference-suite=smoke
python -m pytest tests/inference -q -s --msa-inference-suite=full
```

## 5. Benchmark

### 5.1 Datasets and workloads

| Scenario | Data source | Benchmark scope |
| --- | --- | --- |
| Training | [Training workloads](datas/training/README.md) | Selected with the training benchmark's `--case-suite` |
| Inference Prefill | [Inference workloads](datas/inference/README.md): 10,562 sanitized unique shapes | `--suite full` runs a fixed 128-case selection; representative weights sum to 14,805 |
| Inference Decode / MTP | [Fixed workload configuration](benchmarks/inference/msa_v1/decode/cases.py) | `--suite full` runs 28 cases: 4 batch sizes × 7 nominal sequence lengths |

The Prefill manifest stores query/prefix/KV lengths and call weights. Attention
and Indexer share these shapes; tensor values and non-contiguous physical page
tables are generated reproducibly at runtime. Decode uses batch=8/32/64/128 and
nominal sequence lengths of 1,000/4,000/5,000/10,000/50,000/100,000/200,000, with
8 queries per request (1 main token + 7 draft tokens). Using base seed 1701,
actual sequence lengths vary within each batch around the nominal length while
preserving that exact mean. Decode is a fixed synthetic workload suite.
These data contain no tensor payloads or raw business data.

Indexers accept Q=1–16 through `--query-length` and H=1/2/4 through `--num-index-heads`.
The separate `--suite low-latency` covers batch=1/2/4 × KV length=1,000/4,000/8,000/32,000
and reports latency for these 12 cases independently.

### 5.2 Examples

Run from the repository root and enable initial JIT compilation as described
in the preceding section. Training smoke benchmark:

```bash
python benchmarks/training/msa_v1/benchmark.py --kernel indexer --case-suite smoke
```

Inference Prefill Indexer: smoke and complete 128-case benchmark:

```bash
python -m benchmarks.inference.msa_v1.indexer.prefill.bf16.benchmark \
  --suite smoke --out agent/agent_benchmark/prefill_bf16_smoke.json
python -m benchmarks.inference.msa_v1.indexer.prefill.bf16.benchmark \
  --suite full --out agent/agent_benchmark/prefill_bf16_full.json
```

Inference Decode Indexer: complete 28-case benchmark:

```bash
python -m benchmarks.inference.msa_v1.indexer.decode.q8kv8.benchmark \
  --suite full --graph-calls 120 --warmup 5 --replays 20 \
  --out agent/agent_benchmark/decode_q8kv8_full.json
```

These are executable Indexer examples. Attention benchmarks are under
[`benchmarks/inference/msa_v1/attention/`](benchmarks/inference/msa_v1/attention/);
select the entry point for the decode/prefill phase and dtype. See the
[training benchmark documentation](benchmarks/training/msa_v1/README.md)
for full training usage.

### 5.3 Timing protocol

Input construction, compilation, planning, and warmup are excluded from benchmark
metrics. Inference measures the complete public `run()` E2E path inside CUDA
Graph; Indexer timing includes score computation and TopK. Prefill rotates
disjoint tensor sets with a reuse distance greater than twice L2 capacity.
Decode defaults to 120 calls per Graph and rotates inputs for cold-cache timing.
Defaults are 5 warmup replays and 20 timed replays, reporting median latency per
call and CV. Prefill calls per Graph depend on tensor slots and are not fixed
at 120.

Use `smoke` for quick checks and unfiltered `full` to compare results across the
complete suite. The inference Indexer entry points above accept
`--baseline <baseline.json>` to compare E2E performance on the same workload.
The examples save results to the Git-ignored `agent/` directory; `--out` selects the output path.

## 6. Third-party licenses

This repository uses the [MIT license](LICENSE). Third-party attribution and license notices are documented in [NOTICE](NOTICE); original source copyright/license headers are preserved.
The CUTLASS submodule is at `third_party/cutlass` (BSD-3-Clause).
Shared CuTe training and inference components are adapted from FlashAttention; its full BSD-3-Clause license is reproduced in [NOTICE](NOTICE).
FlashInfer-derived C++ sources retain their Apache-2.0 notices; other inherited upstream attributions remain in NOTICE. Python runtime dependencies retain their distributed licenses.
