# AGENTS.md (English)

The Simplified-Chinese [`AGENTS.md`](AGENTS.md) is the normative source. This
file is its English companion and must be updated with it.

## Paths and scope

- This branch contains only MSA v1 training, inference, and required dependencies;
  do not import other model implementations or development-repository history.
- Repository paths in this document are relative to the repository root;
  Markdown links are relative to the containing document.
- External references use public Git links pinned to a verified commit, without
  depending on a developer's directory layout.

## Collaboration prerequisites

- Clarify requirements, design, objective, target architecture, dtypes,
  shapes, fixed/varlen path, correctness thresholds, and performance metric
  before implementation. Do not act while an ambiguity can change
  correctness, API, performance acceptance, or a destructive operation.
- The user must choose CuTe DSL or CUTLASS C++ before a new operator or kernel
  rewrite begins. Do not switch implementation technology autonomously.
- Reply in Simplified Chinese. Code comments are English; design documents are
  Simplified Chinese. Use plain-text formulas in terminal-oriented responses.

## Git and worktrees

- A feature means a new operator. New operators and performance optimization
  start from the latest user-designated development baseline (`nv_dev` for
  this branch) in an isolated worktree. Use
  `feature/<name>` for a new operator and `perf/<name>` for optimization.
- Small documentation, rule, configuration, or local maintenance changes do
  not require a worktree. Bug fixes stay in the requested worktree unless the
  user asks for isolation; isolated fixes use `fix/<name>`.
- Temporary development commits are allowed, but merge each final objective
  into the target worktree as exactly one squashed commit containing only scoped
  files. Preserve unrelated user changes.
- Push only after the user explicitly chooses whether and where to push.

## Toolchain and reproducibility

- All Blackwell kernels in the repository, including inference and training,
  must support both SM100/B200 and SM103/B300. Select architecture-specific
  instructions for the actual device and retain a compatible implementation
  for the other architecture. Interfaces, compilation targets, caches, tests,
  and public documentation must agree on support. Clearly distinguish
  compilation-only validation from execution on the corresponding GPU.
- Require `nvidia-cutlass-dsl[cu13]>=4.5.2`. Kernel compatibility changes must be
  validated with both 4.5.2 and the current latest stable release; prereleases
  are not the default acceptance target. Record the actually loaded version
  and do not claim validation for untested versions. Switch to a supported
  isolated environment if the loaded version is older than 4.5.2.
- Install and validate the cu13 backend by default; isolate compiled and AOT
  caches across CUDA backend versions.
- Do not reuse compile or AOT caches across DSL versions or mutate a shared
  Python environment. Fix the DSL version for performance comparisons and
  label cross-version comparisons separately.
- Product code must not depend on internal tuning knobs, closed headers, SASS
  patches, or unreproducible build steps. Public CUDA instructions such as
  QMUL4 are allowed.
- Public reference algorithms must be adapted to this project's contracts,
  revalidated, and attributed. Do not copy code with unverified licensing or
  code accessible only internally.
- Prefer standard `CUDA_HOME`, `CUTLASS_ROOT`, `TORCH_EXTENSIONS_DIR`, and
  `TORCH_CUDA_ARCH_LIST`; never hard-code developer or internal-tool paths.

## Engineering tools and references

- Before CUDA, CuTe DSL, or GPU performance work, read the most relevant skill
  if the current environment provides one. Skills are engineering references,
  do not override repository rules, and must not require private tools or
  internal knowledge bases.
- Reference code is read-only by default. Locate the relevant operator and
  implementation before reading; do not scan unrelated repositories.
- When the user requests an NCU report to inspect, use `ncu --set full`, save
  it at an accessible stable path, and provide a compressed copy and an `scp`
  command with the actual host and path.

## Data and harness contracts

- Varlen workloads must use the real varlen path; never silently substitute a
  fixed-length fast path.
- TopK page IDs may be sparse, unordered, and non-contiguous. Harnesses must
  not sort or reorder them.
- The last valid TopK entry is the local block. Only that page needs the local
  causal/length mask; selected history pages do not. Padding is not valid
  TopK.
- Preserve real page-table mapping, varlen boundaries, and public dispatch.
  Tests and references must not bypass the production path.

## Source organization and style

Interfaces, CuTe DSL kernels, warp specialization, and AOT follow FA4 and
the repository style. The following public examples are pinned to commit
`145b1010051dbfd4bdc41a0ae55d495b08d7a458`; they do not replace explicit repository rules:

- [interface.py](https://github.com/Dao-AILab/flash-attention/blob/145b1010051dbfd4bdc41a0ae55d495b08d7a458/flash_attn/cute/interface.py)
- [flash_fwd_sm100.py](https://github.com/Dao-AILab/flash-attention/blob/145b1010051dbfd4bdc41a0ae55d495b08d7a458/flash_attn/cute/flash_fwd_sm100.py)
- [flash_bwd_sm100.py](https://github.com/Dao-AILab/flash-attention/blob/145b1010051dbfd4bdc41a0ae55d495b08d7a458/flash_attn/cute/flash_bwd_sm100.py)


CuTe DSL follows the compact FA4 organization: `interface.py` is at the
operator root; device kernels and independent stages also live at that root
with descriptive `_sm90.py`, `_sm100.py`, `_preprocess.py`, `_combine.py`,
etc. suffixes. Reusable components used by at least two kernel files belong in
`common/`; one-kernel helpers remain private to that kernel. Do not create a
mechanical C++-style `kernel/` tree for CuTe DSL.

CUTLASS C++ uses:

```text
csrc/
├── api/                 # PyBind and public host API
├── src/                 # plan/reduction translation units
├── templates/           # JIT instantiation templates
└── include/sm100/
    ├── common/          # Params, Traits, basic helpers
    ├── collective/      # load/dequant/MMA/softmax/correction/mainloop
    ├── device/          # adapter, plan, reduction
    └── kernel/          # entry, grid, launch policy
```

C++ types/classes/Traits use PascalCase, functions and variables descriptive
snake_case, compile-time constants `kPascalCase`, two-space indentation, and
CUTLASS upstream style. The repository's only `.clang-format` is LLVM-based,
two-space indent, four-space continuation, and 100 columns. Format only scoped
hand-written C/C++ files. Generated output must be regenerated from its source
of truth.

CuTe DSL uses four-space indentation, descriptive snake_case helpers,
PascalCase classes, explicit `cute.Tensor`/`Optional[cute.Tensor]`/
`cutlass.Constexpr[...]` annotations, and the class order `__init__` →
`@cute.jit __call__` → `@cute.kernel`. Prefer existing tiled-copy and
`cute.copy` abstractions. Interfaces explicitly validate dtype, device, shape,
stride, alignment, and metadata; hidden input conversions must not mask kernel
limitations. Import order is `cutlass` → `cutlass.cute as cute` →
`cutlass.cute.nvgpu` → `cuda.bindings`. Tensor prefixes are `m*` (GMEM), `g*`
(GMEM tiles), `s*` (SMEM), `t*` (thread views), and `acc_*` (accumulators).
TopK indices, LSE, and `cu_seqlens` metadata need not use tiled copies.

## Compile keys

Compile keys contain only stable properties that change code generation:
architecture, dtype, head dimension, GQA ratio, tile/stage/thread/cluster
configuration, algorithm switches, optional tensor presence, and static
layout/broadcast patterns. Never include runtime batch size, total Q/KV,
sequence lengths, number or content of sequences, `cu_seqlens` values, TopK
values, tensor identity/pointer, or stream. Runtime tensor values must not
cause D2H synchronization or recompilation. Cache tests vary runtime sizes and
metadata while requiring reuse, and vary a real static configuration when a
new artifact is expected. Varlen metadata enters keys only through static
signature properties such as presence, never through tensors or values. New
specializations require evidence of different generated code and a bounded,
stable host-side enumeration, never a runtime size.

## Runtime state and workspace lifetime

- Process-global caches may contain only compiled artifacts and immutable host
  metadata that does not retain runtime CUDA tensors, device pointers, or
  workload state.
- Never cache CUDA tensors in process-global variables, static arrays, or
  dictionaries keyed by runtime shapes. This includes workspaces, intermediate
  outputs, dequantization buffers, counters, page metadata, and schedules.
- A runtime workspace must be exclusively owned by its wrapper, plan, request,
  or CUDA Graph instance and have an explicit lifetime. Destroying the owner
  must not leave its GPU memory retained by a global strong reference.
- Different plans, requests, CUDA streams, or CUDA Graphs on the same GPU must
  not share a writable workspace. A buffer pool is allowed only with bounded
  capacity, reclamation, stream/event-safe reuse, and concurrent correctness
  tests.
- A schedule or plan derived from indexer outputs, TopK indices, page mappings,
  sequence metadata, or other runtime data must not be reused across layers,
  steps, or workloads. It may only be passed explicitly from a forward
  operation to its corresponding backward operation.
- CUDA Graph pointer stability must use graph- or wrapper-owned buffers, not
  process-global shared CUDA tensors.
- Validation must confirm that varying runtime shapes does not cause unbounded
  retained GPU memory, destroying a wrapper or plan releases its workspace,
  and concurrent streams or instances do not alias writable workspaces or
  corrupt results.

## Numerical precision

- Tensor Core GEMM accumulators are FP32.
- CUDA Core multiply-add and transcendental functions execute in FP32. GMEM
  and SMEM may store BF16/FP16 or lower-precision quantized data, but arithmetic
  converts to FP32 first.
- Dequantization alone may use half precision. This exception does not apply
  to accumulation, softmax, reduction, or other CUDA Core arithmetic.
- Downcast only after FP32 computation according to the output contract.
- Valid parallel reduction, atomic, online-softmax, NVLS, or AllReduce order
  changes are not precision bugs when semantics, dtype, and FP32 accumulation
  remain intact. Identify the concrete source of nondeterministic accumulation
and use fixed agreed tolerances; do not explain failures away as rounding.
Deterministic paths require bitwise-identical output for the same input, seed,
configuration, and environment. Unexplained nondeterminism is an unknown bug;
find its cause instead of relaxing tolerances.

## Correctness and validation

- Formal correctness compares every public output element and required
  auxiliary result against an independent reference. The reference must model
  the real input/dequant semantics and must not reuse the tested kernel.
- Use nontrivial random values and varying scales plus separate zero/extreme
  boundary cases. Check NaN/Inf.
- Shared manifests define shapes, distributions, seed, and page/TopK
  contracts. Dtype/operator-specific generators, references, and tolerances
  stay in operator tests.
- New operators and bug fixes first pass smoke. Any new operator, kernel
  change, or bug fix passes the target full correctness suite before the final
  commit. Shared-component changes run all affected suites, or the full suite
  if the affected scope cannot be determined. Add coverage for features and
  fixes, preferably by extending existing tests. Deterministic paths include
  repeated execution checks; nondeterministic paths use fixed thresholds.
- Sparse/varlen tests cover irregular `cu_seqlens`, unordered pages,
  local-block-last, local partial pages, minimums, and upper bounds.
- A single kernel execution over 30 seconds is a deadlock. Compile and run
  timing are logged separately and synchronized around the run. Terminate and
  investigate a hung run; compilation is excluded from this timeout.
- Check the full diff, format/lint, relevant tests, README/API documentation,
  and absence of correctness/performance regressions before commit.

## Compile and run timing

Every `cute.compile(...)` and test/demo kernel run records separate host logger
timings. Synchronize the GPU before and after timing execution, for example:

```python
import logging
import time

import torch

logger = logging.getLogger(__name__)

t0 = time.time()
compiled = cute.compile(demo, ...)
logger.info("Compiled in %.3fs", time.time() - t0)

torch.cuda.synchronize()
t0 = time.time()
compiled(...)  # If this times out, treat it as a deadlock.
torch.cuda.synchronize()
logger.info("Ran in %.3fms", (time.time() - t0) * 1e3)
```

Other runtimes use their device synchronization equivalent. The 30-second
execution timeout excludes compilation. Formal benchmark statistics exclude
compilation and warmup.

## Code-clean gate

Before commit, remove device `printf`, temporary logs, debug branches,
commented-out code, unused imports/variables/helpers, experimental constants,
duplicate implementations, unreachable branches, and untested fallbacks.
Delete migrated dense/prefill/compatibility tuple/output modes and historical
parameters with neither production callers nor formal test coverage. Keep compatibility only with an explicit
condition and test.

Do not introduce hidden host-device synchronization, temporary allocation,
copy/reorder, repeated compilation, runtime-shape specialization, test-only
production branches, or harness bypasses. Use named constants, single-purpose
helpers, clear warp/barrier ownership, minimal scoped diffs, and update public
README/API documentation with code. Comments explain why rather than repeat
operations. Keep compile/run timing loggers. Migrated paths may remain when
covered by production callers or formal tests. Before commit, inspect the full
diff and run formatting/lint and relevant tests; compile-cache or performance
changes also require cache-reuse checks and an E2E benchmark.

## Performance work

- Each candidate records its reason, exact change, cases, environment, CuTe
  DSL version, before/after formal metric, cumulative change from the fixed
  initial baseline, and incremental change from the last accepted version.
  Record rejected, regressed, and incorrect candidates too.
- Formal benchmarks require exclusive GPU access, fixed inputs/warmup/repeats/
  statistics, and no clock locking. Stop collection if another compute process
  is detected. SM clock, temperature, and power records are not required.
  Do not select a best single run. Record each change separately; use ablation
  when changes must be combined so gains remain attributable.
- E2E acceptance is defined by the nearest directory `AGENTS.md`. TFLOPS,
  cycles, main-kernel latency, Tensor Core SOL, MBU, and MFU are diagnostic.
- Inspect scalar/non-coalesced loads, avoidable LDS/STS/LDG/STG, bank
  conflicts, register spills/LDL/STL, load/compute/softmax overlap, waits, and
  tail effects. Revalidate all reference-derived ideas on real varlen/TopK
  inputs. Dense-equivalent causal sparse FLOPs count actual causal work.

## User-facing documentation

- User-facing `README.md`, public API documentation, and formal product
  documentation are tracked and must change with the code. A new operator or
  any change to a public interface, data contract, dependency, supported
  configuration, invocation, or user-visible behavior updates the nearest
  operator README and any affected parent README in the same commit.
- Every tracked user-facing `README.md` is the normative Simplified-Chinese
  version and has a same-directory `README.en.md` companion. Both files link to
  each other at the top. Their heading structure, tables, examples, commands,
  public contracts, and limitations remain semantically aligned. Updating
  either language requires updating the other in the same commit.
- README files are for operator users. They document purpose, installation and
  dependencies, public APIs, input/output contracts, supported configurations,
  invocation examples, user-visible errors, and validation commands. They do
  not document warp/CTA decomposition, TMA/TMEM/SMEM layouts, pipelines and
  barriers, schedulers, private stages, internal workspace organization, SASS,
  or optimization history. Those details belong in local design/optimization
  documents, profile reports, or source comments.
- A README must not expose or recommend private modules, helpers, stages, or
  unexported parameters. Public examples match current package exports and
  interface signatures. Validate local links and commands before commit.

## Agent artifacts and documents

Temporary tests, benchmarks, profiles, logs, JSON/CSV, and NCU/NSYS reports go
under the Git-ignored repository-root `agent/` directory:

- tests: `agent/agent_tests/`
- benchmarks: `agent/agent_benchmark/`
- profiles: `agent/agent_space/`
- memory summaries: `agent/memory/`

Do not track these artifacts. Local `design.md`, `optimization.md`, and
`results.md` and optimization-process documents are also untracked. If already
tracked, remove them from the index while keeping local copies and ignore them.
