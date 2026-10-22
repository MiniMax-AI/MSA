# M3.1 DSpark draft backbone on TP4

English | [简体中文](README.zh-CN.md)

One kernel launch per rank for the MiniMax-M3.1 DSpark draft model's backbone on one four-GPU NVLink
node: one request's seven-row draft block (row 0 the bonus token, rows 1-6 the mask token, or any
tokens), the vocab-parallel embedding, the five dense decoder layers and the final norm. It replaces the
per-layer operator sequence of SGLang's standard draft path. The draft head (lm_head, sampling) is not
part of this call.

Per layer, with SGLang's rounding points: Gemma RMSNorm, MXFP8 QKV (16 Q heads and one KV head per
rank, head dim 128), per-head Gemma q/k norm and NeoX RoPE on the first 64 dims, the block's K/V written
to the layer's BF16 paged cache, sliding-window attention over the last 4096 keys (every block row sees
the whole block), MXFP8 o_proj, TP4 sum + residual + Gemma RMSNorm, MXFP8 gate_up (3072 + 3072 rows per
rank), SwiGLU-OAI (alpha 1.702, limit 7), MXFP8 down, TP4 sum + residual + the next norm. Activations are
quantized to MXFP8 per 32 values (UE8M0 scale, E4M3 RNE); GEMMs, norms, softmax statistics and the TP4
sums accumulate in FP32. Each TP4 sum adds the four ranks' BF16 partials in rank order in FP32 and
rounds once, so all ranks return bitwise identical output and repeated calls are bitwise identical.

The whole backbone, including the embedding's and each layer's two TP4 sums, runs inside one
cooperative launch per rank; the ranks exchange their partials through the context's IPC arenas,
without NCCL or a separate all-reduce.

## Inputs and lifecycle

```python
from inference.high_tps.msa_v1.draft import (
    DraftBackboneInputs, DraftBackbonePlan, DraftBackboneWeights, DraftLayerWeights, create_draft_tp_context)

context = create_draft_tp_context(tp_group)          # collective; one 32 MiB IPC arena per rank
weights = DraftBackboneWeights(layers=[DraftLayerWeights(...) for _ in range(5)],
                               embed=embed_shard, vocab_start=rank * 50016, final_norm=final_norm, eps=1e-6)
inputs = DraftBackboneInputs(input_ids=..., positions=..., slot_mapping=..., page_table=..., kv_len=...,
                             cos_sin=..., k_caches=[...], v_caches=[...])
plan = DraftBackbonePlan(context, weights, inputs, out)   # out: BF16 [7, 6144]
plan.run()                                                # capturable; returns out
```

| Operand (per rank) | dtype | shape |
|---|---|---|
| `qkv` / `qkv_sf` | E4M3 / UE8M0 | `[2304,6144]` / `[2304,192]` (q 16 heads, k, v) |
| `o` / `o_sf` | E4M3 / UE8M0 | `[6144,2048]` / `[6144,64]` |
| `gate_up` / `gate_up_sf` | E4M3 / UE8M0 | `[6144,6144]` / `[6144,192]` (gate 3072, up 3072) |
| `down` / `down_sf` | E4M3 / UE8M0 | `[6144,3072]` / `[6144,96]` |
| norm weights | BF16 raw Gemma `w` (applied as `1 + w`) | `[6144]`, q/k `[128]` |
| `embed` | BF16 | `[50016,6144]`: rows `[50016 r, 50016 r + 50016)` of the vocabulary |
| `k_caches`, `v_caches` | BF16, one per layer | `[slots,128]`, slot = page * 128 + row |
| `input_ids`, `positions`, `slot_mapping` | int64 | `[7]` (`slot_mapping` -1 skips the row's write) |
| `page_table`, `kv_len` | int32 | `[max_pages]`, `[1]` (cached tokens including the block) |
| `cos_sin` | FP32 | `[positions,64]` (cos, sin) |

Weights use the checkpoint's row-major layouts (the MXFP8 scales are the row-major `weight_scale_inv`,
not a swizzled copy). Every operand is bound at plan creation; weights, caches and step inputs may change
in place between calls (rewrite the step inputs before each graph replay). Any `kv_len` from 7 to the page
table's capacity is supported, and pages need not be contiguous. Each call writes only the block's slots
of each cache and the output.

Create contexts and plans on every rank in the same order, outside graph capture, and warm each plan up
before capturing `plan.run()`. Plans of one context run in stream order; every rank must issue every
call (if a rank misses one, the other ranks' kernels trap after about 10 s instead of hanging).
Destroy graphs and finish outstanding work before calling `context.close()` collectively. Requires
SM100/SM103, four distinct devices of one node with CUDA IPC peer access, and a CUDA toolkit; the first
use compiles the extension (set `ALLOW_JIT=1`). Runtime validation covers SM103 (B300); the SM100
build is compile-checked only.

## Validation and performance testing

```bash
ALLOW_JIT=1 torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.draft.test_reference
```

The test compares every call with an FP64 reference. The MXFP8 requantization before every GEMM turns
any legitimate rounding difference (accumulation order, BF16 attention probabilities) into E4M3 code
changes that compound over the five layers, so the bound is relative: the output and every layer's
written K/V must be within 1.5x the distance of an independent FP32 implementation to the reference
(and cosine >= 0.99), and layer 0's K/V within one BF16 ulp of each row's largest value. It also
requires bitwise identical ranks, no cache writes outside the block's slots, CUDA Graph replay equal
to eager and correct after changed inputs, bitwise repeats, weights changed in place after plan
creation, alternating plans and large norm weights, over kv lengths 7-65536 with mask and token
blocks.

Latency on 4x B300 (SM103, default clocks), us per call, against the same five layers built from
SGLang's operators (FlashInfer MXFP8 GEMMs, its fused q/k norm + RoPE, TRT-LLM decode attention, fused
SwiGLU-MXFP8 and the fused TP4 all-reduce + residual + norm) with the same weights, caches and inputs:
each graph replays 16 calls over 4 rotated weight sets per rank, each followed by a non-PDL 16 MiB copy
(timed alone and subtracted), slowest rank, implementations interleaved, interquartile mean of 12
samples. A second capture of the SGLang path measured against the first gives 0.98-1.02.

| kv length | block | norm weights | SGLang operators | This kernel | Speedup |
|---|---|---|---|---|---|
| 7 | mask | 0.1 | 343.8 | 249.3 | 1.38 |
| 1000 | tokens | 0.1 | 355.0 | 256.1 | 1.39 |
| 1024 | mask | 0.1 | 355.1 | 254.2 | 1.40 |
| 4096 | mask | 0.1 | 350.4 | 261.5 | 1.34 |
| 4103 | mask | 0.1 | 359.8 | 270.6 | 1.33 |
| 8192 | mask | 0.1 | 358.1 | 261.4 | 1.37 |
| 8192 | tokens | 0.1 | 357.4 | 266.2 | 1.34 |
| 8192 | tokens | 4 | 359.8 | 261.5 | 1.38 |
| 40000 | mask | 0.1 | 360.5 | 260.5 | 1.38 |
| 131072 | mask | 0.1 | 352.8 | 258.3 | 1.37 |

Over 18 measured points (kv 7-131072, mask and token blocks, norm weight scales 0.1/1.5/4, two weight
seeds) the speedup is 1.31-1.41, geomean 1.36. Against the FP64 reference the output's cosine is within
1e-4 of the SGLang operator path's on every point and higher on 16 of 18 (by up to 3.5e-3 with large
norm weights).
