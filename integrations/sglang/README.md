# sglang-msa-mono

[简体中文](README.zh-CN.md)

SGLang monokernel plugin for MSA-v1 MegaAttention and the token-parallel MoE
kernel (`inference.high_tps.msa_v1`), and for the DSpark draft backbone
(`inference.high_tps.msa_v1.draft`). It registers `--monokernel-backend msa` for
MiniMax-M3.1 (`MiniMaxM3SparseForConditionalGeneration`, as released in
`MiniMaxAI/MiniMax-M3.1-Flash`, and `MiniMaxM3TinyForConditionalGeneration`,
`MiniMaxM3SparseForCausalLM`, `MiniMaxM3TinyForCausalLM`) and for its DSpark draft
(`DSparkMiniMaxDraftModel`) on SM100 / SM103 with TP4.

Both architectures are compilation targets. Runtime integration checks currently
cover SM103; SM100 runtime and complete-server performance remain unqualified.


This is an optional integration. The pinned fork currently requires access permission;
its commit is not anonymously accessible, and public SGLang releases do not provide
the required `MiniMaxM3Nvfp4KVPool` interface. Installing the plugin alone therefore
does not provide a publicly reproducible server setup. The standalone MSA and draft
APIs do not require this fork.

## Install

```bash
pip install -e .                     # the MSA kernels (minimax-msa)
pip install --no-deps -e integrations/sglang
```

SGLang loads the package as a general plugin (`sglang.srt.plugins` entry point).

## Use

```bash
# MiniMaxAI/MiniMax-M3.1-Flash ships its DSpark draft in the dspark/ subdirectory
SGLANG_FORCE_COARSE_WAR_BARRIER=1 \
python -m sglang.launch_server --model-path MiniMax-M3.1-Flash --trust-remote-code \
  --tp-size 4 --page-size 128 \
  --speculative-algorithm DSPARK --speculative-draft-model-path MiniMax-M3.1-Flash/dspark \
  --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8 \
  --speculative-dspark-block-size 7 \
  --monokernel-backend msa --disable-attn-tp-gather
```

Only DSpark/EAGLE chain verify steps of one request with exactly 8 draft tokens
run the target kernels, because the token-parallel MoE kernel has a fixed 8 rows. The
embedding and dense layers 0-2 run SGLang's modules. Each MoE layer then runs
as two persistent launches: MegaAttention, then the token-parallel MoE, each
with its TP4 all-reduce in its own NVLS workspace. SGLang's final norm and
logits processor end the step. Prefill, plain decode and batches of more than one
request take the standard path. A server without
speculative decoding, or with another draft token count, fails at startup and
lists the reasons.

The DSpark draft's one-request block step (7 rows: the bonus token and six mask
tokens) runs the draft backbone: the vocab-parallel embedding, the five dense
layers and the final norm in one cooperative launch per rank, with every TP4 sum
inside it. SGLang's draft head (lm_head, Markov head, sampling) and its injection
of the target's hidden states into the draft KV cache are unchanged. The draft
graphs must hold one request on their own: with an a2a MoE backend (M3.1 NVFP4
forces `megamoe`), SGLang pads draft rows to a multiple of attention TP, so pass
`--disable-attn-tp-gather`; without it the draft spec refuses the server at
startup.

## Contract

- Cluster placement: MegaAttention's canonical-placement mode is about 3.4 µs faster
  per launch, but traps (`M3_DECODE_CANONICAL_PLACEMENT_REQUIRED`, then a CUDA launch
  failure that ends the TP worker group) unless every SM is free when it launches.
  Under overlap scheduling the scheduler prepares the next batch on its own stream
  and by default waits only for an event inside the previous forward, so its small
  kernels can hold SMs at a MegaAttention launch. The plugin therefore enables
  placement unless no other stream can run kernels during the forward:
  `SGLANG_FORCE_COARSE_WAR_BARRIER=1` (the scheduler stream waits for the whole
  forward) or `--disable-overlap-schedule`, and `SGLANG_ENABLE_OVERLAP_PLAN_STREAM`
  unset. The server log states the choice (`cluster placement on` or
  `off (canonical)`). With placement off, the GPUs must not run other processes.
- Checkpoint: MiniMax-M3.1's NVFP4 release (`quantization_config` declares
  `mxfp8` with `moe_quant_algo: NVFP4`, which SGLang serves as
  `minimax_m3_nvfp4`): MXFP8 attention and shared expert, NVFP4 routed
  experts, FP32 router. Within each expert, the w1 and w3 weight global scales
  must match (the kernel has one FC1 alpha), and every input global scale must
  be 1. Every MoE layer must be an MSA sparse attention layer; the dense layers
  may use either attention.
- KV cache: SGLang's MSA-v1 NVFP4 pool, used in place through its
  page-strided views (see [Paged cache layout](../../inference/high_tps/msa_v1/README.md#paged-cache-layout)). It must not
  be compact or share K/V, and pages are 128 tokens.
- Context: at most 1048576 tokens (8192 pages). Each graph's plan uses the
  page table's capacity, and the runtime KV length is written into
  `plan_info.kv_segment_lens` on every step.
- Weights: routed experts are written into the `MoeWeights` logical layout
  (gate rows then up rows, row-major scales) as they are loaded. Each expert is
  stored once. All other operands are copied from the model after loading.
  Online weight updates are refused (SGLang's monokernel framework has no
  refresh hook).
- SGLang: monokernel API **1** and the M3.1 NVFP4 cache interface
  (`MiniMaxM3Nvfp4KVPool`). The integration is pinned for validation to
  [`4143ef163183ed759abc613b2bd6f872c8cf16f4`](https://github.com/ispobock/sglang-minimax-collab/commit/4143ef163183ed759abc613b2bd6f872c8cf16f4).
  A different monokernel API is rejected. This does not establish compatibility
  with arbitrary SGLang releases or real-checkpoint accuracy.
- `--fp8-gemm-runner-backend flashinfer_trtllm` is refused because it
  shuffles MXFP8 weights in place.
- Draft: the draft config SGLang derives from the checkpoint's `dspark_*` fields
  (5 sliding-attention layers of window 4096, dense FFN 12288, Gemma norms,
  `sample_from_anchor` on), MXFP8 linears in checkpoint layout, SGLang's plain
  BF16 draft KV pool with 128-token pages (not a fused or quantized draft cache),
  and the target's vocab-parallel embedding without added vocabulary.
  `SGLANG_DSPARK_EMBED_IN_GRAPH` must stay on: the kernel embeds token ids.

## Tests

```bash
PYTHONPATH=.:integrations/sglang:<sglang>/python pytest integrations/sglang/tests
```

The weight-layout and startup-contract tests run on CPU; the step-metadata
prepare tests require CUDA. These tests do not establish real-checkpoint model
quality, acceptance length, throughput, latency or long-running service stability.
SM100 and SM103 runtime qualification must be recorded separately.

A checkpoint-free TP4 binding check uses synthetic nonzero weights, real cache
views and both production runners. It checks a two-kernel Graph while switching
KV lengths through 1K, 128K, 1M, 32K and 1K:

```bash
PYTHONPATH=.:integrations/sglang ALLOW_JIT=1 torchrun --standalone --nproc-per-node=4 integrations/sglang/tests/test_minimax_m3_binding.py
```

This check exercises one MoE layer's binding; it does not run a complete server
or measure end-to-end serving performance.
