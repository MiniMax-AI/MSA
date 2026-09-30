# MSA v1 training benchmark

[English](README.en.md)

该 benchmark 覆盖 Attention FWD、Attention BWD、Indexer 和 KL backward。默认使用
真实 `192K / CP16 / 12 chunks per rank` workload。默认 smoke 运行每个 CP rank 一个
case；full 的 32 个 shape strata 权重合计代表 1500 条真实调用。

## 使用方式

```bash
python benchmarks/training/msa_v1/benchmark.py --kernel attention-fwd
python benchmarks/training/msa_v1/benchmark.py --kernel attention-bwd
python benchmarks/training/msa_v1/benchmark.py --kernel indexer
python benchmarks/training/msa_v1/benchmark.py --kernel indexer --use-fp16-score
python benchmarks/training/msa_v1/benchmark.py --kernel kl
```

默认 `--case-suite smoke`；`--case-suite full` 运行固定 32-case selection，
`--benchmark-case-id` 运行一个固定 case，`--case-suite all` 运行所有 rank-local case。
旧人工 workload 只能通过显式参数运行，例如：

```bash
python benchmarks/training/msa_v1/benchmark.py \
  --kernel attention-fwd \
  --synthetic-scenario 128k_cp16
```

## 计时与 FLOPs

输入构造、JIT、metadata 分配和 plan 不进入 hot-path CUDA Event 计时。输出同时保留
preprocess/cold-e2e 指标，并使用 `representative_weight` 计算真实分布的 weighted aggregate。

```text
FWD FLOPs = 2 * (Dqk + Dv) * Hq * sparse_elements
BWD FLOPs = 2 * (3 * Dqk + 2 * Dv) * Hq * sparse_elements
Indexer FLOPs = 2 * Di * Hi * causal_elements
KL FLOPs = 2 * (Dqk * Hq + 3 * Di * Hi) * sparse_elements
```

MSA v1 使用 `Dqk=Dv=Di=128`、`Hq=64`、`Hi=4`、`block_size=128`、`topK=16`。
Indexer 的 `--use-fp16-score` 使用 FP16 score workspace，不改变公开输出契约。
结果与临时 profile 写入仓库外的 agent 工作目录。
