# MiniMax Sparse Attention (MSA)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.10-blue.svg)](pyproject.toml)
[![GPU](https://img.shields.io/badge/NVIDIA-SM100-76b900.svg)](#requirements)
[![Stack: CuTe-DSL + Cuda](https://img.shields.io/badge/stack-CuTe--DSL%20%2B%20Cuda-purple.svg)](#stacks)

**MSA** (`fmha_sm100`) ships dense FlashAttention and sparse top-k attention
kernels for **NVIDIA SM100**. Two JIT-compiled stacks
share one Python package:

![MSA architecture](docs/architecture.png)

> Algorithm reference: [MiniMax Sparse Attention paper](docs/MiniMaxSparseAttention.pdf).

| Stack | Path | What it gives you |
|---|---|---|
| **csrc JIT** | `python/fmha_sm100/csrc/` | Dense FMHA (`fmha_sm100`, `fmha_sm100_plan`) + `sparse_topk_select` indexer, compiled from Jinja templates by `jit.py` at runtime. |
| **CuTe-DSL** | `python/fmha_sm100/cute/` | Full sparse attention (forward + paged FP8 decode, BF16 / FP8 / NVFP4 / FP4), compiled at runtime via `cute.compile`. |
| **Bridge** | `python/fmha_sm100/sparse_fmha_adapter.py` | Adapts the `fmha_sm100` API to call `sparse_atten_func` for sparse prefill paths. |

> **License: MIT.** Self-authored files carry `SPDX-License-Identifier: MIT`.
> See [LICENSE](LICENSE) and [NOTICE](NOTICE). Bundled / derived third-party
> code retains its own license — see [Third-party licenses](#third-party-licenses).

## Requirements

- **GPU**: NVIDIA SM100 (B200) or SM103 (B300); the Q8KV4 NVFP4 decode kernel also builds for SM107.
- **Toolchain**: CUDA Toolkit with `nvcc` on `PATH` (or `CUDA_HOME` / `CUDA_PATH` set).
- **Python**: ≥ 3.10.
- **OS**: Linux x86_64 (aarch64 untested; JIT builds may need small Makefile edits on WSL).

Quick sanity check before installing:

```bash
nvcc --version                # expect ≥ 12.x
nvidia-smi --query-gpu=compute_cap --format=csv | grep -E "10.[037]"  # confirm SM100/SM103/SM107
python -c "import sys; print(sys.version_info[:2])"              # ≥ (3, 10)
```

## Install

```bash
# --recursive pulls the NVIDIA CUTLASS submodule (python/fmha_sm100/cutlass/),
# whose headers are required for JIT/AOT compilation.
git clone --recursive https://github.com/MiniMax-AI/MSA.git msa
cd msa
# If you cloned without --recursive:
#   git submodule update --init --recursive
pip install .           # standard install (works from a wheel too)
# or
pip install -e .        # editable install for development
```

This pulls in the CuTe-DSL stack via `nvidia-cutlass-dsl` and `quack-kernels`;
the csrc kernels are JIT-compiled at first import from sources shipped inside
the package.

## Verify

Run a small CUDA smoke test. **The first run JIT-compiles `sparse_topk_select`,
which takes 30 s – a few minutes on a cold nvcc cache** — this is normal, not
a hang. Subsequent runs hit the JIT cache and finish in seconds.

```bash
python tests/smoke/test_sparse_topk_forced.py
```

## Usage

```python
import torch
from fmha_sm100 import fmha_sm100, fmha_sm100_plan, sparse_topk_select

# Page size and top-k for the sparse prefill path.
page_size, topk = 128, 16

# Dense proxy pass: compute per-block max score from a cheap Q slice.
proxy_plan = fmha_sm100_plan(
    qo_lens, kv_lens, proxy_q.shape[1],
    num_kv_heads=1,
    page_size=page_size,
    output_maxscore=True,
)
_, max_score = fmha_sm100(
    proxy_q, proxy_k_pages, proxy_v_pages, proxy_plan,
    kv_indices=kv_indices,
    output_o=False,
    output_maxscore=True,
)

# max_score -> sparse KV block indexes.
kv_block_indexes = sparse_topk_select(
    max_score.contiguous(), topk, num_valid_pages=num_pages,
)

# Sparse attention with the selected blocks.
sparse_plan = fmha_sm100_plan(
    qo_lens, kv_lens, q.shape[1],
    num_kv_heads=k_pages.shape[1],
    page_size=page_size,
    kv_block_num=topk,
)
out, _ = fmha_sm100(
    q, k_pages, v_pages, sparse_plan,
    kv_indices=kv_indices,
    kv_block_indexes=kv_block_indexes,
)
```

For block-sparse prefill with CSR metadata, the FP4 indexer, the Q8KV4/Q8KV8
paged indexers over the vLLM index-K cache, NVFP4 K/V, and the paged FP8
decode wrapper, see the **CuTe-DSL deep dive**:

- [`python/fmha_sm100/cute/README.md`](python/fmha_sm100/cute/README.md)

Sparse decode on an NVFP4 KV cache (uint8 `[pages, Hkv, 128, 72]` K/V with fp32
global scales) runs on the Q8KV4 kernel when the batch fits it; `fmha_sm100_plan`
takes `decode_backend`, `kv_dtype` and `block_scale_shift` to steer that. QMUL4
dequantization needs CUDA 13.4 or newer, older toolkits build an FP16 fallback.
See:

- [`python/fmha_sm100/decode_q8kv4/README.md`](python/fmha_sm100/decode_q8kv4/README.md)

## Test

```bash
# Fast smoke tests.
python -m pytest tests/smoke -q

# API and end-to-end integration tests.
python -m pytest tests/integration -q
python tests/integration/test_proxy_kv_e2e.py

# Large regression suites.
python tests/regression/test_correctness.py
python tests/regression/test_sparse_attn.py

# Q8KV4 NVFP4 sparse decode: smoke set, full matrices, FP16 dequant fallback.
python -m pytest tests/q8kv4 -q -m "not full"
python -m pytest tests/q8kv4 -q
FMHA_SM100_DECODE_Q8KV4_DISABLE_QMUL4=1 python -m pytest tests/q8kv4 -q

# CuTe-DSL forward-only sparse attention.
cd python/fmha_sm100/cute
python -m pytest test_sparse_atten.py -q
```

## Benchmark

`benchmarks/bench_sparse_attention_ops.py` covers dense prefill, paged
prefill, sparse prefill, dense decode, paged decode, sparse decode, in
`fp8` and `bf16` (`nvfp4` is sparse-prefill only).

```bash
python benchmarks/bench_sparse_attention_ops.py --help     # full flag list
```

Common invocations (output is TSV):

| Goal | Command |
|---|---|
| FP8 full sweep | `python benchmarks/bench_sparse_attention_ops.py --dtype fp8 --sections all --output_mode o -o /tmp/msa_fp8.tsv` |
| BF16 full sweep | `python benchmarks/bench_sparse_attention_ops.py --dtype bf16 --sections all --output_mode o -o /tmp/msa_bf16.tsv` |
| NVFP4 sparse prefill | `python benchmarks/bench_sparse_attention_ops.py --dtype nvfp4 --sections sparse_prefill --output_mode o -o /tmp/msa_nvfp4.tsv` |
| Quick CI smoke | `python benchmarks/bench_sparse_attention_ops.py --dtype fp8 --sections prefill,decode,sparse_decode --seqs 8192,16384 --tp 1,4 --decode-k 8192,131072 --decode-b 32 --dry-run-ms 50 --repeat-ms 200 -o /tmp/msa_smoke.tsv` |
| Output-mode checks (dense/paged) | `--output_mode maxscore` or `--output_mode full` |

`benchmarks/bench_q8kv4_decode.py` times NVFP4 sparse decode through the
`fmha_sm100` API on the Q8KV4 kernel and on the kv_mode 3 kernel, replaying CUDA
graphs over page regions larger than twice the L2 (exclusive GPU required):

```bash
python benchmarks/bench_q8kv4_decode.py --suite mtp --backends q8kv4,kv_mode3
python benchmarks/bench_q8kv4_decode.py --suite full --gqa 16 --shift 3 --output /tmp/q8kv4_full.json
```

## Layout

```
python/fmha_sm100/                  Python package
  __init__.py                       Public re-exports (lazy for the CuTe-DSL stack)
  api.py                            fmha_sm100 / fmha_sm100_plan / sparse_topk_select
  jit.py                            Runtime JIT (nvcc + ninja) for the csrc stack
  sparse.py                         Lazy shim that loads the cute/ stack
  sparse_fmha_adapter.py            Bridge: fmha_sm100 API → sparse_atten_func
  q8kv4_decode_adapter.py           Bridge: fmha_sm100 API → decode_q8kv4 (NVFP4 sparse decode)
  decode_q8kv4/                     Q8KV4 paged sparse decode (own csrc + JIT, see its README)
  csrc/                             CUDA kernels + Jinja templates (JIT-compiled)
    include/                        Vendored FlashInfer / CUTLASS-derived / TRT-LLM headers
  cutlass/                          NVIDIA CUTLASS git submodule (include/ + tools/util/include/)
  cute/                             CuTe-DSL sparse attention (loaded via sys.path)
tests/                              Correctness tests
  smoke/  integration/  regression/  q8kv4/
scripts/                            Warmup + cache-management helpers
benchmarks/                         bench_sparse_attention_ops.py, bench_q8kv4_decode.py
```

## Stacks

- **csrc JIT** — dense FlashAttention, page KV, and `sparse_topk_select`
  indexer. Compiled at runtime from `csrc/*.cu.jinja` plus
  `csrc/include/`. Public entry: `fmha_sm100.plan → run`.
- **CuTe-DSL** — block-sparse prefill, FP8 / NVFP4 / FP4 quantization, paged
  FP8 decode (`SparseDecodePagedAttentionWrapper`), FP4 block-score indexer.
  Public entry: `fmha_sm100.sparse_atten_func`,
  `fmha_sm100.sparse_decode_atten_func`, `fmha_sm100.fp4_indexer_block_scores`.
- **Q8KV4 decode** — CUTLASS C++ paged sparse decode on the NVFP4 cache
  (`fmha_sm100.decode_q8kv4`), reached through `fmha_sm100` for uint8 caches
  or directly via `plan_decode` / `run_decode`.
- **Bridge** — `sparse_fmha_plan` / `sparse_fmha` adapt the dense-API call
  site to the sparse backend for prefill paths; useful when you already
  drive the dense kernel and want a one-line swap to sparse.

## Third-party licenses

`fmha_sm100` bundles, derives from, or depends on the third-party components
below. Each retains its original license; this section summarizes them.
Authoritative text is shipped with each component.

### Vendored / derived source (shipped in this repo)

| Component | License | Where |
|---|---|---|
| **NVIDIA CUTLASS** | BSD-3-Clause | Git submodule at `python/fmha_sm100/cutlass/` (provides `include/` + `tools/util/include/`), plus BSD-3-tagged headers under `python/fmha_sm100/csrc/include/` and `python/fmha_sm100/decode_q8kv4/csrc/include/` (e.g. `fmha_fusion.hpp`). The SM100 MMA descriptor encodings in `python/fmha_sm100/cute/src/common/mma_sm100_desc.py` mirror CUTLASS hardware descriptors. Copyright (c) 2017–2025 NVIDIA CORPORATION & AFFILIATES. |
| **FlashInfer** | Apache-2.0 | Headers and sources under `python/fmha_sm100/csrc/`, `python/fmha_sm100/csrc/include/` and `python/fmha_sm100/decode_q8kv4/csrc/` that carry a `Copyright (c) <year> by FlashInfer team` line (e.g. `allocator.h`, `exception.h`, `utils.cuh`, `cutlass_utils.cuh`, `fmha_cutlass_sm100.cuh`, `sparse_topk_select.cuh`, `plan.cuh`, `sm100_fmha_reduction.hpp`, `tvm_ffi_utils.h`). Project: <https://github.com/flashinfer-ai/flashinfer>. |
| **NVIDIA TensorRT-LLM + NAVER Corp (CLOVA)** | Apache-2.0 | Portions of `python/fmha_sm100/csrc/include/sparse_topk_select.cuh` — `indexerTopK` histogram-step + insertion-sort derived from `tensorrt_llm/cpp/tensorrt_llm/kernels/indexerTopK.cu`. Copyright (c) 2019–2026 NVIDIA CORPORATION; Copyright (c) 2021 NAVER Corp. The per-file header in `sparse_topk_select.cuh` includes a function-level provenance map. |

### Runtime dependencies (installed via pip)

| Package | Upstream | License |
|---|---|---|
| `quack-kernels` | <https://github.com/Dao-AILab/quack> | Apache-2.0 |
| `nvidia-cutlass-dsl` | NVIDIA CUTLASS Python DSL | NVIDIA / BSD-3-Clause (see package) |
| `apache-tvm-ffi` | Apache TVM FFI | Apache-2.0 |
| `cuda-python` | NVIDIA | NVIDIA / see package |
| `torch` | <https://github.com/pytorch/pytorch> | BSD-3-Clause |
| `jinja2` | <https://github.com/pallets/jinja> | BSD-3-Clause |
| `ninja` | <https://github.com/ninja-build/ninja> | Apache-2.0 |
| `pybind11` | <https://github.com/pybind/pybind11> | BSD-3-Clause |

The exact license of each installed package is distributed with that package;
consult its metadata (`pip show <pkg>`) for the authoritative text.

## Citation

If MSA helps your research, please cite it. (BibTeX entry coming once the
companion paper / technical report has a stable identifier — placeholder.)
The algorithmic reference is shipped at
[`docs/MiniMaxSparseAttention.pdf`](docs/MiniMaxSparseAttention.pdf).

```bibtex
@software{msa2026,
  title  = {MiniMax Sparse Attention (MSA): FlashAttention and block-sparse
            attention kernels for NVIDIA SM100},
  author = {{MiniMax}},
  year   = {2026},
  url    = {https://github.com/MiniMax-AI/MSA}
}
```

## Contributing

Issues and PRs welcome on the
[issue tracker](https://github.com/MiniMax-AI/MSA/issues). For kernel or
runtime-contract changes, open an issue first to align on the public
surface — `fmha_sm100.api`, `fmha_sm100.sparse` and
`cute.interface` are the stable entry points; everything else
is internal and may change without notice.
