# MSA v1 training benchmark

[Simplified Chinese](README.md)

This benchmark covers Attention FWD, Attention BWD, Indexer, and KL backward.
By default it uses the real `192K / CP16 / 12 chunks per rank` workload. The
smoke suite runs one case per CP rank, while the 32 full-suite shape strata use
weights that represent 1,500 real calls.

## Usage

```bash
python benchmarks/training/msa_v1/benchmark.py --kernel attention-fwd
python benchmarks/training/msa_v1/benchmark.py --kernel attention-bwd
python benchmarks/training/msa_v1/benchmark.py --kernel indexer
python benchmarks/training/msa_v1/benchmark.py --kernel indexer --use-fp16-score
python benchmarks/training/msa_v1/benchmark.py --kernel kl
```

The default is `--case-suite smoke`. Use `--case-suite full` for the fixed
32-case selection, `--benchmark-case-id` for one fixed case, or
`--case-suite all` for every rank-local case. Legacy synthetic workloads are
available only through explicit arguments, for example:

```bash
python benchmarks/training/msa_v1/benchmark.py \
  --kernel attention-fwd \
  --synthetic-scenario 128k_cp16
```

## Timing and FLOPs

Input construction, JIT compilation, metadata allocation, and planning are
excluded from hot-path CUDA Event timing. Results also retain preprocess and
cold-E2E metrics and use `representative_weight` for the production-weighted
aggregate.

```text
FWD FLOPs = 2 * (Dqk + Dv) * Hq * sparse_elements
BWD FLOPs = 2 * (3 * Dqk + 2 * Dv) * Hq * sparse_elements
Indexer FLOPs = 2 * Di * Hi * causal_elements
KL FLOPs = 2 * (Dqk * Hq + 3 * Di * Hi) * sparse_elements
```

MSA v1 uses `Dqk=Dv=Di=128`, `Hq=64`, `Hi=4`, `block_size=128`, and
`topK=16`. Indexer option `--use-fp16-score` uses an FP16 score workspace
without changing the public output contract. Results and temporary profiles
are written to the agent workspace outside the repository.
