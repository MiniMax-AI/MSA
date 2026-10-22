# Correctness checks

English | [简体中文](README.zh-CN.md)

Install dependencies and set PYTHONPATH as described in the root README. The execution-path entries below run with torchrun on single-node TP4.

| Path | Module entry |
|---|---|
| `attn_mega` | `tests.inference.high_tps.msa_v1.attn_mega.test_attn_mega` |
| `moe_mega` | `tests.inference.high_tps.msa_v1.moe_mega.test_moe_mega` |
| `attn_mega_moe_mega` | `tests.inference.high_tps.msa_v1.integration.test_attn_mega_moe_mega` |

```bash
torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.integration.test_attn_mega_moe_mega --q 8 --output agents/tests/integration
```

Entries using the MoE `mega` backend (`moe_mega`, `attn_mega_moe_mega`) accept only Q=8; the other entries accept Q=1–8. Attention execute 8 physical rows; Q identifies the last Q valid output rows. The integration check includes the complete two-kernel pipeline.

## Coverage

Default zero-weight checks use an independent analytic oracle: MoE output equals the residual; Attention output is the input Gemma-RMSNorm. Checks cover finite values, Graph node counts for 10 complete operations, repeated replay, and reverse-Q plan reuse. GPU execution exceeding 10 seconds fails; compilation is excluded.

`--nonzero` uses nonzero weights and saves per-rank `.pt` outputs for identical-input cross-version comparison; finite checks alone are not an independent correctness proof.

`moe_mega/test_reference.py` checks the Q8 `mega` MoE against an independent FP32 reference built from the msa_v1 quantization and rounding rules, over balanced and random routes with several active-expert counts. It also checks bitwise repeatability, Graph replay with changed inputs, alternating plans, zero scales and invalid usage.

```bash
torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.moe_mega.test_reference
```

Each rank writes JSON, tensor dumps, and Graph DOT. Failures return a nonzero exit code. These tests do not report formal latency.


## Cache and conversion checks

The cache dependency test requires no GPU. The QMUL4 conversion check requires
one supported GPU and compares all 65,536 four-lane codes across 127 finite
nonnegative scale rotations against an FP32 reference.

```bash
python -m pytest -q tests/inference/high_tps/msa_v1/common/test_build_cache.py
python -m tests.inference.high_tps.msa_v1.common.check_qmul
```

## Runtime KV and serving contracts

The runtime-KV test compares selected pages with an independent FP32 oracle,
checks the complete output with the existing FP8 storage tolerance, and replays
one Graph across short/long lengths up to 1M without recompilation. The stride
test compares contiguous and padded slab views, including cache-write canaries.

```bash
torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.attn_backends.test_runtime_kv
torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.mega.test_strided_cache --q 1 2 3 4 5 6 7 8
torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.moe_mega.test_weight_refresh --backend mega
```

The test updates original weights and scales, refreshes the runner, and compares
the retained Graph with a freshly prepared runner. Startup and metadata tests for
the SGLang plugin are documented in its [guide](../../../../integrations/sglang/README.md).

The standalone device-selection regression checks ties, quantization-boundary
collisions, internal widths, and named-barrier participation inside a 512-thread
CTA. Build outside the source tree; use `sm_100a` for B200 or `sm_103a` for B300:

```bash
nvcc -O3 -std=c++17 -arch=sm_103a \
  -I inference/high_tps/msa_v1/csrc \
  tests/inference/high_tps/msa_v1/attn_backends/test_topk_device.cu \
  -o /tmp/msa_topk_test
/tmp/msa_topk_test
```
