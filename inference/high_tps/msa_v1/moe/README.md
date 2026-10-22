# Decode MoE interface

English | [简体中文](README.zh-CN.md) | [Project setup](../../../../README.md)

`create_moe_runner` exposes one MoE boundary: post-attention normalized hidden
states and the attention residual enter; BF16 hidden states after one TP sum and
one residual addition leave. Do not apply another reduction or residual addition.
Attention, input projection, final norm, LM head and sampling remain outside this
interface. TP4 MegaMoE launches one kernel per rank.

## Scope and backend selection

The first integration supports one request, q=8, hidden size 6144, TP4 on one
NVLink node, 128 routed experts with top-4 routing, one shared expert, and local
intermediate size 768 (global 3072). EP, multiple requests, variable per-request
lengths and other model dimensions are unsupported. Router weights and logits
are FP32. Shared experts use MXFP8; routed experts use NVFP4. Accumulation uses
FP32. The fixed activation is SwiGLU with alpha=1.702, limit=7 and beta=1.

The only backend is `mega`, also the default, and it accepts only q=8.
Unsupported shapes or backend names raise errors.

## Dependencies

MegaMoE targets SM100/SM103. It is runtime-validated on SM103 (GB300);
SM100 has compile validation only. Install CUDA-enabled PyTorch, a CUDA toolkit
supporting QMUL4, TVM-FFI, and CUTLASS. The four devices must be on one NVLink node
with CUDA NVLS multicast support. FlashInfer is not required.
Compilation and weight preparation must finish before Graph capture.

## Weight contract

`MoeWeights` holds one rank's weights, stable between explicit refreshes. All tensors are contiguous,
16-byte aligned CUDA tensors on the same device, without gradients. FC1 rows
are **all gate rows followed by all up rows**, along dimension -2. Scale tensors
are logical row-major arrays, not pre-swizzled storage.

| Field | Shape | Dtype / meaning |
|---|---|---|
| `router_weight` | `[128,6144]` | FP32, raw router weight |
| `router_correction_bias` | `[128]` | FP32, expert selection bias |
| `shared_fc1_weight` | `[1536,6144]` | FP8 E4M3 |
| `shared_fc1_weight_sf` | `[1536,192]` | uint8 E8M0, one scale per 32 values |
| `shared_fc2_weight` | `[6144,768]` | FP8 E4M3 |
| `shared_fc2_weight_sf` | `[6144,24]` | uint8 E8M0 |
| `routed_fc1_weight` | `[128,1536,3072]` | uint8, two packed E2M1 values per byte |
| `routed_fc1_weight_sf` | `[128,1536,384]` | FP8 E4M3, one scale per 16 values |
| `routed_fc1_alpha` | `[128]` | positive finite FP32 global weight scale |
| `routed_fc2_weight` | `[128,6144,384]` | uint8, packed E2M1 |
| `routed_fc2_weight_sf` | `[128,6144,48]` | FP8 E4M3 |
| `routed_fc2_alpha` | `[128]` | positive finite FP32 global weight scale |

Packed nibbles use the existing low-level weight ABI (the first logical value is
the low nibble). Gate/up global scales must be equal within each expert. Routed
activation global scales are fixed at 1. The FC2 alpha here **excludes** the routing
multiplier; `MoeConfig(routed_scaling_factor=2.0, enable_pdl=True)` applies it once.
The correction bias affects expert selection, not the normalized route weights.
Quantization, BF16 intermediate stores and parallel reduction order can produce
small numerical differences; bitwise identity is not promised.

For callers already using the low-level MegaMoe weight arguments:

```python
weights = MoeWeights.from_mega(
    routed_scaling_factor=2.0,
    **existing_weight_arguments,
)
```

Pass exactly the twelve weight arguments in the table, using their existing
low-level layouts. The conversion undoes the FC1 tile row ordering and scale
packing and removes the routing multiplier already folded into low-level FC2
alpha. Use the same multiplier in `MoeConfig`. Input tensors are never modified.
Creating a runner prepares backend-specific weight storage once, shared by all
of its plans. Account for prepared weights in resident memory. Preparation may synchronize.

## Lifecycle and integration

Initialize a single-node four-rank `torch.distributed` group and select each
rank's CUDA device first. Every rank must create and close contexts in the same
order, using matching `group_id`, `slot_id` and `backend`. IDs must uniquely identify
live contexts. `group=None` uses the default process group.

```python
from inference.high_tps.msa_v1.moe import MoeConfig, create_moe_runner, create_moe_tp_context

context = create_moe_tp_context(group=tp_group, group_id="decoder-moe", slot_id=0)
runner = create_moe_runner(weights=weights, config=MoeConfig(),
                           tp_context=context, backend="mega")
plan = runner.prepare(q_len=8, hidden=attention_hidden,
                      residual=attention_residual)
attention_run()
output = plan.run()
```

`hidden` and `residual` must be contiguous BF16 `[8,6144]`, 16-byte aligned, on
the weights' device, without gradients. They are read-only. The attention producer
must finish writing them before consumption on the current stream. Tail slices
of preallocated `[8,6144]` buffers bind directly, with no copy or padding kernel.
`output` contains exactly eight valid rows; downstream code consumes `[8,6144]`.

The plan owns its output and writable buffers. Mega output aliases
`plan.mega_workspace.next_hidden`; inputs must not overlap that workspace.
Its contents can be overwritten by any subsequent operation using that workspace,
including attention. Retain or consume output before reuse. There is no
arbitrary caller-provided `out` argument or hidden output copy.

Create Q=8 bound plans collectively, in the same order on all ranks.
All mega plans of a context share its communication workspace and its output,
which is also exposed as `context.mega_workspace` for attention. Warm each plan
up with valid inputs on **every rank**, and
synchronize before capture. Capture `attention_run(); plan.run()` in the caller's
CUDA Graph. The plan does not capture or replay a nested graph. Q is fixed at 8;
changing tensor addresses requires preparation and recapture. The current CUDA device must match the plan. All ranks must execute
identical q/backend/collective sequences. PDL is requested on supporting operators;
it is not a promise that every operator supports PDL.

For an online weight update, rewrite the `MoeWeights` tensors in place while no
plan is executing, then call `runner.refresh_weights()` outside capture. It
re-derives each backend's prepared storage in place. Plans and captured graphs
keep their addresses and need neither re-preparation nor recapture. The caller must wait
for all in-flight work, finish updates and refreshes on all TP4 ranks, and order
subsequent execution after refresh completion. Shape, dtype, device, stride and
storage identity must stay fixed. Storage replacement requires a new runner and
Graph capture. A failed refresh invalidates all runner plans until a successful
refresh; do not replay a previously captured Graph after failure, because direct
`graph.replay()` bypasses Python validation. This operator API does not enable
SGLang service-side online updates.

JIT compilation, offline weight packing, and plan/context construction finish
before capture. The prepared plan launches one kernel with fixed buffer addresses.

Plans sharing a context must execute sequentially. For concurrent execution, use
independent contexts, plans and writable attention buffers. Keep runners, plans,
inputs and contexts alive until graphs and outstanding GPU work are finished.
Then destroy graphs, collectively call `context.close()`, and finally destroy the
process group. Python object destruction does not collectively release resources.

Invalid q, shape, dtype, device, alignment, output overlap or context state raises
an error before execution where host metadata permits. Missing native extensions
or incompatible operator versions may fail during preparation or warmup. A rank
failure requires the application to abort peer collective execution.

## Minimal correctness check

Save the following as `moe_smoke.py`, then run `ALLOW_JIT=1 torchrun --standalone --nproc-per-node=4 moe_smoke.py` from an environment where `inference.high_tps.msa_v1` is importable. It creates zero weights, repeatedly replays the Q=8 graph, and expects output to equal the residual exactly. This checks the API and collective lifecycle, not nonzero-model numerical accuracy or performance.

```python
"""Standalone zero-weight oracle using only the public MoE interface."""

import os

import torch
import torch.distributed as dist

from inference.high_tps.msa_v1.moe import MoeWeights, create_moe_runner, create_moe_tp_context

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")


def full(shape, value, dtype):
    return torch.full(shape, value, dtype=dtype, device="cuda")


weights = MoeWeights(
    router_weight=full((128, 6144), 0, torch.float32),
    router_correction_bias=full((128,), 0, torch.float32),
    shared_fc1_weight=full((1536, 6144), 0, torch.float8_e4m3fn),
    shared_fc1_weight_sf=full((1536, 192), 127, torch.uint8),
    shared_fc2_weight=full((6144, 768), 0, torch.float8_e4m3fn),
    shared_fc2_weight_sf=full((6144, 24), 127, torch.uint8),
    routed_fc1_weight=full((128, 1536, 3072), 0, torch.uint8),
    routed_fc1_weight_sf=full((128, 1536, 384), 1, torch.float8_e4m3fn),
    routed_fc1_alpha=full((128,), 1, torch.float32),
    routed_fc2_weight=full((128, 6144, 384), 0, torch.uint8),
    routed_fc2_weight_sf=full((128, 6144, 48), 1, torch.float8_e4m3fn),
    routed_fc2_alpha=full((128,), 1, torch.float32),
)
context = create_moe_tp_context(group_id="moe-smoke")
runner = create_moe_runner(weights=weights, tp_context=context)
hidden = full((8, 6144), 0.5, torch.bfloat16)
residual = full((8, 6144), 1, torch.bfloat16)
plan = runner.prepare(q_len=8, hidden=hidden, residual=residual)
plan.run()
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    plan.run()
for _ in range(3):
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(plan.output, residual, rtol=0, atol=0)
del graph
context.close()
dist.destroy_process_group()
print(f"PASS rank={rank}", flush=True)
```

For performance measurements, use the project CUDA Graph and cold-L2 protocol: 1000 complete operations per graph, two warmup replays, five event-timed replays, independent rotating weight/input/scale slots and a slot-count stability check. Count all work in the complete logical operator; do not compare an untimed wrapper call against a graph latency.
