# BF16 Decode Indexer

[English](README.md)

## 功能

使用 BF16 query 和共享 BF16 K cache，为每个本地 index head 独立选择最多 16 个
logical KV page。支持 H=1/2/4、Q=1–16。

## 公开接口

```python
import torch
from inference.msa_v1.indexer.decode.bf16 import (
    BatchDecodeIndexerWithPagedKVCacheWrapper,
)

q = torch.randn(2, 2, 2, 128, device="cuda", dtype=torch.bfloat16)
paged_k_cache = torch.randn(4, 128, 128, device="cuda", dtype=torch.bfloat16)
page_table = torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32)
seq_lens = torch.tensor([129, 256], device="cuda", dtype=torch.int32)
out = torch.empty(2, 4, 16, device="cuda", dtype=torch.int32)
wrapper = BatchDecodeIndexerWithPagedKVCacheWrapper()
wrapper.plan(page_table, seq_lens, num_index_heads=q.shape[2], query_length=q.shape[1])
indices = wrapper.run(q, paged_k_cache, out=out)
assert indices.shape == (2, 4, 16)
```

在 CUDA Graph capture 外调用 `plan()`，capture 前先 warmup `run()`，并预分配输出。
请求 metadata 变化后重新 `plan()`，或绑定从 `inference.msa_v1.indexer.decode` 导入的
`BatchDecodeIndexerPlan`，在消费前调用其 `update()`。构造和绑定须在 capture 外，
`update()` 支持 capture。跨 stream 消费及再次更新必须用 event 排序；Graph 使用期间
保持 wrapper 和 plan 存活。

省略 `out` 时返回 wrapper 持有的默认输出，后续 `run()` 会覆盖它；需要保留结果时
使用独立的输出。并发调用须使用独立 wrapper 和可写 workspace。

## 数据契约

| 参数 | Shape | Dtype |
| --- | --- | --- |
| `q` | `[B,Q,H,128]` | `torch.bfloat16` |
| `paged_k_cache` | `[physical_pages,128,128]` | `torch.bfloat16` |
| `page_table` | `[B,max_pages]` | `torch.int32` |
| `seq_lens` | `[B]` | `torch.int32` |
| `out` | `[H,B*Q,16]` | `torch.int32` |

`plan()` 默认 `num_index_heads=1`、`query_length=8`；page table 容量为 1–8192 页。
所有 tensor 必须 contiguous 且位于同一 CUDA device，Q/K 须 16-byte 对齐。
所有请求共用 Q；长度包含当前 query chunk，满足 `Q <= seq_lens[b] <= max_pages*128`。

历史页评分为该页各 token 的 Q/K 点积最大值，乘以 `1/sqrt(128)`。
各 head 独立选择页面。输出有效项构成前缀，最后一项为 local page，无效后缀填 `-1`；
历史索引不保证排序。选择使用 16-bit 量化近似排序，分数接近时不保证与精确 FP32
排序产生相同索引。H=1 也保留四维输入与三维输出。

## 运行约束

支持 GB200/B200（SM100）和 GB300/B300（SM103），要求
`nvidia-cutlass-dsl[cu13]>=4.5.2`。不同 Q/H 配置可能触发编译，应在 Graph capture 前
warmup 应用需要的配置。

输出为 head-major；与要求 `[B*Q,H,16]` 的 attention adapter 集成时，须显式转换
metadata 布局。

## 验证命令

```bash
MINIMAX_INFERENCE_TEST_SUITE=smoke python -m pytest tests/inference/msa_v1/indexer/decode/bf16
python -m benchmarks.inference.msa_v1.indexer.decode.bf16.benchmark --suite full --num-index-heads 4 --query-length 8 --verify --out bf16.json
python -m benchmarks.inference.msa_v1.indexer.decode.bf16.benchmark --suite low-latency --num-index-heads 1 --query-length 1 --verify --out bf16_low_latency.json
python -m benchmarks.inference.msa_v1.indexer.decode.plan --precision bf16 --num-index-heads 4 --query-length 8 --layers 1 --out plan.json
```

生产 suite 使用共享 case 权重，low-latency suite 使用等权重。有效 MBU 按各 dtype
存储字节数计算必要流量，并覆盖公开评分与 TopK 调用；它是流量模型指标，不是实测
DRAM 利用率。BF16 与 FP8 的 MBU 接近并不意味着绝对延迟相同。
