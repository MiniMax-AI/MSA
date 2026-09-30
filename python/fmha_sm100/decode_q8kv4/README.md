# Q8KV4 paged sparse decode

Decode attention for the MiniMax sparse-attention (MSA) NVFP4 KV cache on SM100, SM103 and
SM107: E4M3 queries, E2M1 keys and values with E4M3 block scales, TopK-selected pages, causal
masking of the query's own page. `fmha_sm100_plan` / `fmha_sm100` route NVFP4 sparse decode here
automatically; this page documents that routing and the package's own API.

## Requirements

- CUDA Toolkit 12.9 or newer for SM100/SM103, 13.5 or newer for SM107. The fast dequantization
  path uses the QMUL4 instruction, which `ptxas` accepts from CUDA 13.4 on; older toolkits build
  an FP16 dequantization fallback automatically (about 1.3x slower on B300). SM107 always uses
  the fallback.
- CUTLASS headers: the repository submodule (4.3) serves SM100/SM103; SM107 needs CUTLASS 4.8 or
  newer, pointed to by `CUTLASS_ROOT`.
- Kernels are JIT-compiled on first use into `MINFER_FMHA_CACHE_DIR` (or `TORCH_EXTENSIONS_DIR`,
  or `~/.cache/minfer/fmha_sm100`), one binary per GQA ratio, split mode and block-scale shift.

Environment switches: `FMHA_SM100_DECODE_Q8KV4_ARCH` selects the target for offline builds
without a device; `FMHA_SM100_DECODE_Q8KV4_DISABLE_QMUL4=1` builds the FP16 fallback on a
toolchain that supports QMUL4.

## Through `fmha_sm100`

```python
plan = fmha_sm100_plan(qo_lens, kv_lens, num_q_heads, num_kv_heads=num_kv_heads,
                       page_size=128, kv_block_num=topk,
                       decode_backend="auto",      # or "q8kv4" / "kv_mode3"
                       block_scale_shift=3)        # TransformerEngine block-scale convention
out, _ = fmha_sm100(q, k_cache, v_cache, plan, kv_indices=kv_indices,
                    kv_block_indexes=kv_block_indexes, k_scale=k_global, v_scale=v_global)
```

`k_cache` / `v_cache` are the K and V slot views `cache[:, 0::2]` / `cache[:, 1::2]` of a
uint8 `[pages, 2 * Hkv, 128, 72]` NVFP4 cache of per-head K/V slots; `k_scale` /
`v_scale` are one-element fp32 CUDA tensors. The plan carries a Q8KV4 schedule when the batch
fits this kernel (page size 128, 8 or 16 Q heads per KV head, uniform query lengths, 1 to 64
blocks, causal, no max-score output) and `fmha_sm100` uses it for uint8 caches when the call fits
too (E4M3 Q, block indexes present, no max-score output, no `q_offset_override`, `o_scale` of 1).
Everything else runs on the kv_mode 3 kernel; `decode_backend="q8kv4"` raises instead of falling
back, `"kv_mode3"` never plans this kernel, and `kv_dtype="fp8"` skips the plan for FP8 caches.

## Direct API

```python
from fmha_sm100.decode_q8kv4 import plan_decode, run_decode, interleave_v_scales

plan = plan_decode(batch_size=B, q_len_per_req=q_len, topk=topk, device=device,
                   num_q_heads=64, num_kv_heads=4, block_scale_shift=0)
out = run_decode(plan, q, (k_data, v_data), kv_cache_sf=(k_scale, v_scale),
                 seq_lens=seq_lens, kv_indices=kv_indices, kv_indptr=kv_indptr,
                 topk_indices=topk_indices, kv_global_scale=(k_global, v_global), out=out)
```

`plan_decode` depends only on the batch shape and is built outside CUDA Graph capture; one plan
serves every layer of a step. `run_decode` takes the tensors per call and is allocation-free when
`out` is given. `BatchDecodeWithPagedKVCacheWrapper` bundles a plan with one set of metadata
(`plan(topk_indices, page_table, seq_lens, ...)` then `run(q, (k, v), kv_cache_sf=...)`) for
callers that keep them together; its `run` accepts a `topk_indices` override.

## Data contract

- `q`: `[B * q_len_per_req, Hq, 128]` E4M3, the `q_len_per_req` tokens of a request last in its
  KV sequence, in order.
- `k_data` / `v_data`: `[pages, Hkv, 128, 64]` uint8, two E2M1 values per byte; token rows
  contiguous, page and head strides free (multiples of 16 bytes, data heads at least 8192 bytes
  apart), so packed pages are read in place.
- `k_scale` / `v_scale`: `[pages, Hkv, 128, 8]` E4M3 block scales (one per 16 values), same
  stride rules. K blocks are linear (`token * 8 + group`); V blocks are in token-quad order
  (`(token // 4) * 32 + group * 4 + token % 4`, vLLM's layout); `interleave_v_scales` converts a
  linear tensor.
- Global scales: `value = code x block_scale x global_scale`; the kernels read the one-element
  fp32 tensors on the device, so captured graphs follow updated tensors. Omitted means 1.0.
- `block_scale_shift`: 0 when `code x block_scale` already fits E4M3 (products up to 448); 3 when
  block scales use the full E4M3 range next to a global scale (products up to 6 x 448). The kernel
  divides the block scales by `2 ** shift` before the product and folds the factor back exactly;
  block scales below `2 ** (shift - 6)` round to E4M3 subnormals. Each shift is a separate binary.
- `seq_lens`: `[B]` int32 KV lengths; `kv_indices`: flat int32 physical page list with
  `kv_indptr` (`[B + 1]`) giving each request's first entry, one page at least per request.
- `topk_indices`: `[B * q_len_per_req, Hkv, topk]` int32 logical page ids relative to the request,
  `1 <= topk <= 64`. The kernel uses the leading entries that lie in `[0, own page]` and stops at
  the first other entry (`-1` padding or a later page); it masks the query's own page to the
  tokens up to the query and reads every other selected page in full. Order does not matter.
- `out`: `[B * q_len_per_req, Hq, 128]` bf16.

Supported: head dim 128, page size 128, 8 or 16 Q heads per KV head, causal only. Runs are
bitwise deterministic for identical inputs.

## Tests and benchmark

```bash
python -m pytest tests/q8kv4 -q -m "not full"   # smoke set
python -m pytest tests/q8kv4 -q                  # full matrices
FMHA_SM100_DECODE_Q8KV4_DISABLE_QMUL4=1 python -m pytest tests/q8kv4 -q   # FP16 fallback path
python benchmarks/bench_q8kv4_decode.py --suite mtp --backends q8kv4,kv_mode3
```

The tests generate their inputs and carry an fp32 reference; the benchmark replays CUDA graphs
over page regions larger than twice the L2 and needs an exclusive GPU.
