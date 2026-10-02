# Inference workloads

[English](README.md)

本目录保存从本地 inference shape dump 归一化得到的 prefill workload。它是 MSA v1
Attention 和 Indexer 测试、benchmark 共用的数据源，不包含 tensor payload。

## 数据语义

只处理 `batch_type == "prefill"`。原始 `kv_lens` 表示本次 prefill 前的 cached prefix：

```text
final_kv_lens[i] = prefix_lens[i] + query_lens[i]
query_position    = prefix_lens[i] + q_idx
local_page        = floor(query_position / 128)
```

10,937 条原始 prefill 记录归一化为 10,562 个唯一 shape，总调用权重为 14,805。原始
dump 由仓库 `.gitignore` 排除；提交的 manifest 不保存 DP rank、源文件名或源文件 hash。

## 文件与 schema

- `prefill_cases_v1/part-*.jsonl`：按 4,096 行确定性分片的唯一 shape、调用权重和
  shape metrics；加载时按文件名顺序组成一个逻辑 manifest，每个 shard 小于 5 MiB。
- `prefill_cases_v1.meta.json`：schema、分片清单与校验值、规范化规则和汇总计数。
- `prefill_test_cases_v1.json`：smoke、full、CUDA Graph、exhaustive 和算子专用测试层。
- `prefill_benchmark_cases_v1.json`：固定 128-case benchmark selection；代表性权重合计
  14,805。

每个 manifest case 包含：

```text
schema_version
case_id
batch_size
query_lens
prefix_lens
final_kv_lens
count
metrics
```

`case_id` 是规范化 shape JSON 的 SHA256 前缀，与输入文件顺序无关。`metrics` 保存
workspace、FLOPs 和 shape 相关的审计指标。

## 测试分层

`prefill_test_cases_v1.json` 定义以下通用 tier：

- `static_all`：全部 10,562 case 的 schema、长度、FLOPs、唯一性和 selection 引用检查。
- `msa_v1_smoke`：32 个确定性真实 case，用于 MSA v1 快速正确性检查。
- `msa_v1_full`：512 个真实 case，用于 MSA v1 全量输出数值正确性检查。
- `gemm_topk_e2e`：1,024 个 Indexer 端到端 case。
- `cuda_graph`：16 个覆盖不同 batch、workspace 和调度不均衡程度的 case。
- `exhaustive`：全部 10,562 case 的 GPU 结构执行层。

测试根据 `case_id` 生成可复现输入和随机、非连续 physical page table；数据文件不保存
tensor payload。

## Benchmark 选择

MSA v1 的 `full` benchmark 包含 128 个 case，覆盖 representative、hot、batch anchor
和多类 stress shape。加权统计使用其中带 `representative_weight` 的生产代表 case，
其权重总和为 14,805。完整数据集包含 10,562 个 shape，可用于更广泛的覆盖检查。
统计公式如下：

```text
weighted_mean_latency =
    sum(representative_weight[i] * latency[i])
    / sum(representative_weight[i])

aggregate_useful_TFLOPS =
    sum(representative_weight[i] * useful_flops[i])
    / sum(representative_weight[i] * latency_seconds[i])
    / 1e12
```

`useful_flops == 0` 的 case 只报告 latency，不单独计算 TFLOPS。

## FLOPs 口径

对 sequence `i` 的 query row `q_idx`，kernel 只计算 forced tail 前的完整历史 page：

```text
historical_pages(i, q_idx) = floor((prefix_lens[i] + q_idx) / 128)

useful_flops =
    2 * 128 * 128
    * sum_i sum_q_idx historical_pages(i, q_idx)
```

其中第一个 128 是 page token 数，第二个 128 是 head dimension。TopK 不折算为
TFLOPS。

## 生成与检查

```bash
python3 datas/inference/generate_cases.py
python3 datas/inference/generate_cases.py --check
```

测试和 benchmark 统一从 `datas.inference.cases` 读取数据。

此 prefill benchmark 计时 CUDA Graph 中公开 `run()` 的完整 E2E 路径，
不包含 `plan()`，也不使用单个计算阶段的时间代替 E2E。Prefill 使用 disjoint multi-tensor
rotation 建立超过 2 倍 L2 的 reuse distance；5 次 warmup、20 次 replay，每个 case
`CV <= 3%`。

```bash
python3 -m benchmarks.inference.msa_v1.indexer.prefill.q8kv8.benchmark \
  --suite full --out /path/to/baseline.json

python3 -m benchmarks.inference.msa_v1.indexer.prefill.q8kv8.benchmark \
  --suite full --baseline /path/to/baseline.json --out /path/to/candidate.json
```

`smoke` 用于快速检查；`full` 使用完整的 128-case benchmark selection。

Q8KV8 prefill indexer 的 `--num-index-heads` 支持 1/2/4，输出为 `[H, total_q, 16]`。
有效 FLOPs 为上述单 head 计数乘 H；生产加权吞吐为加权 FLOPs 总和除以加权 E2E
时间总和。`--verify` 在计时外对全部 head/query 执行独立 reference 检查。基线与候选
必须使用相同设备、DSL 版本、计时协议和 cold-cache slot 数。
