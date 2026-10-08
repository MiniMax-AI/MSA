# MiniMax MSA

[English](README.md)

## 1. 概述

MiniMax Sparse Attention（MSA）通过 Indexer 为每个 query 选择 TopK blocks，
再由 Sparse Attention 在选中的 blocks 上计算注意力。本分支提供 MSA v1 的
训练和推理算子，面向 NVIDIA Blackwell GPU，覆盖 BF16、FP8 E4M3 和 NVFP4 数据路径。
算法参考：[MiniMax Sparse Attention 论文](docs/MiniMaxSparseAttention.pdf)。

训练包为 `msa_v1`，推理包为 `inference.msa_v1`，量化格式转换由 `inference.dequant`
提供。训练和推理使用各自的数据布局与公开接口；具体支持范围如下。

### 1.1 训练算子

| 算子 | 主要输入 | 输出 | 支持范围 |
| --- | --- | --- | --- |
| Sparse Attention | BF16 Q/K/V | BF16 O、FP32 LSE；BF16 dQ/dK/dV | Forward / Backward，支持变长序列 |
| Sparse Attention：FP8 forward | FP8 E4M3 Q/K/V | BF16 O、FP32 LSE | 仅 Forward，不支持原生 FP8 Q/K/V 的 Backward |
| Sparse Attention：概率 QAT | BF16 Q/K/V；`sparse_attn_p_mode="fp8"` | BF16 O、FP32 LSE；BF16 dQ/dK/dV | Forward / Backward；对注意力概率执行 FP8 量化感知训练 |
| Indexer（含 TopK） | BF16 Q/K | INT32 TopK indices、FP32 selected LSE | 因果、变长序列的 block 选择 |
| Tree Indexer（含 TopK） | BF16 Q/K、预先构建的 mask plan | INT32 TopK indices、FP32 selected LSE | Batch=1 的树状 / 自定义可见性选择 |
| Sparse KL Backward | BF16 teacher Q/K 与 indexer Q/K、FP32 LSE | BF16 indexer dQ/dK | 通过稀疏 KL 目标计算 indexer 梯度 |

LSE 可按接口选项返回；上表不表示每次调用都返回 LSE。
训练 Attention 的公开配置为 head dimension=128、64 个 Q heads、4 个 KV heads，
block size=128、TopK=16。调用入口见[训练接口](#31-msa-v1-训练)。

### 1.2 推理算子

| 算子 | 支持的输入格式 | 输出 | 用途与约束 |
| --- | --- | --- | --- |
| Prefill Attention | BF16 Q/K/V；Q8KV8；Q8KV4 | BF16 O，可选 FP32 LSE | Paged sparse causal attention，支持变长 chunk prefill |
| Decode Attention | BF16 Q/K/V；Q8KV8；Q8KV4 | 默认 BF16 O；Q8KV4 可选 MXFP8 O | Paged sparse decode / MTP；BF16 和 Q8KV8 需安装可选 FlashInfer 依赖 |
| Prefill Indexer（含 TopK） | BF16 Q/K；FP8 E4M3 Q/K | INT32 logical page indices | BF16 和 FP8 均支持 1/2/4 个本地 index heads |
| Decode Indexer（含 TopK） | BF16 Q/K；FP8 E4M3 Q/K；FP8 E4M3 Q + NVFP4 K | INT32 logical page indices | 1/2/4 个本地 index heads，每请求 1–16 个 query |
| NVFP4 → FP8 转换 | Packed E2M1 数据 + E4M3 scale | FP8 E4M3 数据 | Dense 转换或仅转换 TopK 选中的 paged K/V |

各算子的 package 路径见[推理接口](#32-msa-v1-推理)，shape、scale、分页和
CUDA Graph 契约见[推理算子文档](inference/msa_v1/README.zh-CN.md)及其链接的各算子 README。

### 1.3 数据类型说明

- **BF16**：`torch.bfloat16`。
- **FP8 E4M3**：`torch.float8_e4m3fn`；本文的 Q8/K8/V8 表示浮点 FP8，不是 INT8。
- **NVFP4**：打包的 E2M1 数据及其 E4M3 分组 scale，必须按对应接口提供两者。
- **Q8KV8**：Q/K/V 均为 FP8 E4M3；**Q8KV4**：Q 为 FP8 E4M3，K/V 为 NVFP4。
  Indexer 只读取 Q/K，不读取 V。
- 上表列出的是公开输入输出格式。Tensor Core 累加使用 FP32；FP16 中间存储选项
  不代表公开接口接受 FP16 Q/K/V。

所有 varlen 接口都使用真实长度 metadata；TopK 的有效项保持离散无序语义，
最后一个有效 block 为 local block。

## 2. 系统要求与安装

| GPU | Architecture | 状态 |
| --- | --- | --- |
| NVIDIA GB200/B200 | SM100 | 支持 |
| NVIDIA GB300/B300 | SM103 | 支持 |

仓库的 Blackwell kernel 覆盖 B200/SM100 与 B300/SM103，包括 inference 和 training。
运行时按设备架构选择兼容实现；各算子既有的 dtype、shape 和数据契约仍适用。

本项目的 CuTe DSL 最低版本要求为：

```text
nvidia-cutlass-dsl[cu13]>=4.5.2
```

安装项目：

```bash
git submodule update --init --recursive
python -m pip install -e ".[test]"
```

作为普通安装包使用时，C++ 算子和 Q8KV8 prefill indexer 仍需要公开 CUTLASS headers。
保留 CUTLASS checkout，并通过 `CUTLASS_ROOT` 指定位置。例如在仓库根目录执行：

```bash
export CUTLASS_ROOT="$PWD/third_party/cutlass"
python -m pip install .
```

升级 DSL 后需要重新构建 AOT 产物，
不得跨 DSL 或 CUDA backend 版本复用编译缓存。本项目默认安装 cu13 backend，
包括 CuTe DSL 4.5.2。可用以下命令安装并核验实际加载版本：

```bash
python -m pip install "nvidia-cutlass-dsl[cu13]>=4.5.2"
python -c 'import cutlass; print(cutlass.__version__, cutlass.CUDA_VERSION)'
```

固定使用 4.5.2 时，若核验仍显示 CUDA 12.9，需最后重新安装同版本 cu13 wheel：

```bash
python -m pip install --force-reinstall --no-deps "nvidia-cutlass-dsl-libs-cu13==4.5.2"
python -c 'import cutlass; print(cutlass.__version__, cutlass.CUDA_VERSION)'
```

Q8KV4 C++ 算子使用标准的 `CUDA_HOME`、`CUTLASS_ROOT` 和 `TORCH_EXTENSIONS_DIR`
配置。运行时 JIT 始终根据输入 tensor 所在的 CUDA device 选择 `sm_100a` 或
`sm_103a`；无 tensor 的 AOT/离线构建使用 `MM_SPARSE_TARGET_ARCH=100a|103a` 指定
目标（默认 `103a`）。`TORCH_CUDA_ARCH_LIST` 只由内部编译流程设置，不作为运行时
架构选择接口。Q8KV4 Decode Attention、Decode Indexer 和共享 dequant 支持 CUDA
Toolkit 12.9 或更高版本；工具链支持 QMUL4 时使用该路径，否则自动使用精确
fallback。Q8KV4 Prefill Attention 要求 CUDA Toolkit 13.4 或更高版本。

BF16 和 Q8KV8 sparse decode 通过可选的外部 FlashInfer TRTLLM-GEN backend 提供。本仓库不包含
FlashInfer 源码或 cubin；使用 `python -m pip install -e '.[flashinfer]'` 安装
`flashinfer-python==0.6.17`。Q8KV4 decode 使用原生 CUTLASS C++；三种 decode attention
均支持 B200/B300 上的 GQA=8/16 和 BF16 输出，按实际 `Hq/Hkv` 选择实现。
Q8KV4 decode 另保留 SM107 的 GQA=16 支持，需要 CUDA Toolkit 13.5 或更新版本。

Rubin（SM107）保留 BF16/Q8KV8 prefill attention 和 Q8KV4 decode attention 路径，
通过输入 tensor 所在设备自动选择；具体依赖和限制见各算子 README。其余算子不据此声明 Rubin 支持。

## 3. 主要接口

### 3.1 MSA v1 训练

```python
from msa_v1 import attention, indexer, indexer_tree, kl

topk_indices, indexer_lse = indexer.forward(...)
metadata = attention.prepare(...)
out, lse = attention.forward(..., metadata, return_softmax_lse=True)
dq, dk, dv = attention.backward(..., out, lse, metadata)
dqi, dki = kl.backward(..., indexer_lse, metadata)
```

每一层必须使用该层当前的 `topk_indices` 调用 `attention.prepare()`。返回的
`AttentionMetadata` 只在对应的 Attention forward、Attention backward 和 KL backward
之间复用；TopK 更新后必须重新执行 `prepare()`。若 TopK 会在 CUDA Graph replay 之间
变化，Graph 必须捕获 `prepare()`，使其在每次 replay 时重新生成 metadata。

MSA v1 Attention 的 E4M3 概率路径（含 `sparse_attn_p_mode="fp8"` QAT）采用
`E4M3(P × 448)`，归一化和 backward 概率重建补偿该缩放。LSE 保持原始 logits
的自然对数语义，QAT 的逻辑 softmax STE 定义不变。该约定不承诺训练与推理最终输出逐位一致。
QAT/FP8 路径及 decode 使用硬件 exp2，普通 BF16 路径保留原有模拟设置；跨路径对齐缩放及补偿语义，
不要求量化后的 P 逐位一致。

| Module | 公开 API |
| --- | --- |
| `msa_v1.indexer` | `prepare_indexer_schedule`、`forward` |
| `msa_v1.attention` | `prepare`、`forward`、`backward` |
| `msa_v1.indexer_tree` | `compile_plan`、`forward` |
| `msa_v1.kl` | `backward` |

Tree Indexer 的用法见
[`training/msa_v1/indexer_tree/tree_indexer_usage.md`](training/msa_v1/indexer_tree/tree_indexer_usage.md)。

### 3.2 MSA v1 推理

MSA v1 推理算子统一使用 `plan()` / `run()` 生命周期。`plan()` 接收请求级 metadata，
`run()` 接收逐层数据。进入 CUDA Graph capture 前必须完成 `plan()` 和所需 warmup；
capture 期间应使用预分配输出。Decode indexer 另提供显式 `BatchDecodeIndexerPlan`：
在 capture 外构造和绑定，`update()` 可在 Graph 内更新同一 step 各层共享的长度 metadata。

| 算子 | Package | 输入格式 |
| --- | --- | --- |
| Decode Attention | [`inference.msa_v1.attention.decode.bf16`](inference/msa_v1/attention/decode/bf16/README.zh-CN.md) | BF16 Q/K/V; GQA=8/16 |
| Decode Attention | `inference.msa_v1.attention.decode.q8kv4` | E4M3 Q + NVFP4 K/V |
| Decode Attention | `inference.msa_v1.attention.decode.q8kv8` | E4M3 Q/K/V |
| Prefill Attention | `inference.msa_v1.attention.prefill.bf16` | BF16 Q/K/V |
| Prefill Attention | `inference.msa_v1.attention.prefill.q8kv4` | E4M3 Q + NVFP4 K/V |
| Prefill Attention | `inference.msa_v1.attention.prefill.q8kv8` | E4M3 Q/K/V |
| Decode Indexer | [`inference.msa_v1.indexer.decode.bf16`](inference/msa_v1/indexer/decode/bf16/README.zh-CN.md) | BF16 Q/K; H=1/2/4, Q=1–16 |
| Decode Indexer | `inference.msa_v1.indexer.decode.q8kv4` | E4M3 Q + NVFP4 K |
| Decode Indexer | `inference.msa_v1.indexer.decode.q8kv8` | E4M3 Q/K |
| Prefill Indexer | `inference.msa_v1.indexer.prefill.bf16` | BF16 Q/K |
| Prefill Indexer | `inference.msa_v1.indexer.prefill.q8kv8` | E4M3 Q/K |

Q8KV4 Decode Attention 示例：

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

各算子的 shape、dtype、TopK/page 契约和可选参数见对应目录的 README。

## 4. 正确性测试

在仓库根目录运行。设置 `FMHA_SM100_ALLOW_JIT=1` 允许首次编译；首次编译不计入 kernel 执行耗时。

```bash
export FMHA_SM100_ALLOW_JIT=1
python -m pytest tests/test_package_layout.py tests/test_training_workloads.py -q
python -m pytest tests/training/msa_v1/cute -q -s --msa-v1-suite=smoke
python -m pytest tests/training/msa_v1/cute -q -s --msa-v1-suite=full
python -m pytest tests/inference -q -s --msa-inference-suite=smoke
python -m pytest tests/inference -q -s --msa-inference-suite=full
```

## 5. Benchmark

### 5.1 数据集与 workload

| 场景 | 数据来源 | Benchmark 范围 |
| --- | --- | --- |
| 训练 | [训练 workload](datas/training/README.zh-CN.md) | 按训练 benchmark 的 `--case-suite` 选择 |
| 推理 Prefill | [推理 workload](datas/inference/README.zh-CN.md)：10,562 个去敏唯一 shape | `--suite full` 固定运行 128 个 case；代表性权重合计 14,805 |
| 推理 Decode / MTP | [固定 workload 配置](benchmarks/inference/msa_v1/decode/cases.py) | `--suite full` 运行 28 个 case：4 个 batch size × 7 个标称序列长度 |

Prefill manifest 保存 query/prefix/KV 长度及调用权重，Attention 与 Indexer 共用这些
shape；tensor 数据和非连续 physical page table 在运行时可复现生成。
Decode 配置使用 batch=8/32/64/128，标称序列长度为
1,000/4,000/5,000/10,000/50,000/100,000/200,000，每个请求包含 8 个 query
（1 个主 token + 7 个 draft token）。以 1701 为基础 seed，在 batch 内生成围绕标称
长度变化的真实 seqlen，并保持平均长度等于标称值；它是一套固定合成 workload。
Indexer 可通过 `--query-length` 选择 Q=1–16，通过 `--num-index-heads` 选择 H=1/2/4。
独立的 `--suite low-latency` 覆盖 batch=1/2/4 × KV length=1,000/4,000/8,000/32,000，
共 12 个 case，单独报告延迟。
这些数据不包含 tensor payload 或原始业务数据。

### 5.2 运行示例

在仓库根目录运行，先按上节说明启用首次 JIT 编译。训练快速检查：

```bash
python benchmarks/training/msa_v1/benchmark.py --kernel indexer --case-suite smoke
```

推理 Prefill Indexer：快速检查与完整 128-case benchmark：

```bash
python -m benchmarks.inference.msa_v1.indexer.prefill.bf16.benchmark \
  --suite smoke --out agent/agent_benchmark/prefill_bf16_smoke.json
python -m benchmarks.inference.msa_v1.indexer.prefill.bf16.benchmark \
  --suite full --out agent/agent_benchmark/prefill_bf16_full.json
```

推理 Decode Indexer：完整 28-case benchmark：

```bash
python -m benchmarks.inference.msa_v1.indexer.decode.q8kv8.benchmark \
  --suite full --graph-calls 120 --warmup 5 --replays 20 \
  --out agent/agent_benchmark/decode_q8kv8_full.json
```

以上是 Indexer 的可执行示例；Attention benchmark 位于
[`benchmarks/inference/msa_v1/attention/`](benchmarks/inference/msa_v1/attention/)，
按 decode/prefill 与 dtype 选择对应入口。训练的完整用法见
[训练 benchmark](benchmarks/training/msa_v1/README.zh-CN.md)。

### 5.3 计时口径

输入构造、编译、plan 和 warmup 不计入 benchmark 指标。推理计时覆盖 CUDA Graph 内公开
`run()` 的完整 E2E 路径；Indexer 包含评分计算与 TopK。Prefill 通过不相交 tensor
轮换建立超过 2 倍 L2 的复用距离；Decode 默认每张 Graph 调用 120 次，并轮换输入
实现 cold-cache。默认 5 次 warmup、20 次 replay，报告每次调用的 median latency 和 CV。
Prefill 的每张 Graph 调用次数由 tensor slots 决定，不固定为 120。

`smoke` 用于快速检查；比较完整测试集的聚合结果时，使用不带 case 筛选的 `full`。
上述推理 Indexer 入口支持 `--baseline <baseline.json>`，用于比较相同 workload 的
E2E 性能。示例将结果保存至 Git 忽略的 `agent/` 目录；可通过 `--out` 选择输出路径。

## 6. 第三方许可证

本仓库使用 [MIT 许可证](LICENSE)。第三方来源与许可证声明见 [NOTICE](NOTICE)；
原有源码头部的版权及许可证不变。CUTLASS 子模块位于 `third_party/cutlass`（BSD-3-Clause）。
CuTe 训练和推理公共组件参考 FlashAttention，其 BSD-3-Clause 许可证全文收录于 [NOTICE](NOTICE)。
FlashInfer 派生 C++ 文件保留原 Apache-2.0 声明；上游继承的其他来源署名保留在 NOTICE 中。Python 运行时依赖各自遵循其发行包许可证。
