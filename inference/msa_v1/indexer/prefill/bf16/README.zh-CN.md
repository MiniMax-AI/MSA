# BF16 Paged Prefill Indexer

[English](README.md)

## 功能

面向 SM100/SM103 的 BF16 paged prefill indexer。它为每个 query 选择 logical pages，
并保证结果包含 local page。

## 公开接口

```python
import torch
from inference.msa_v1.indexer.prefill.bf16 import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)

q = torch.randn(5, 2, 128, device="cuda", dtype=torch.bfloat16)
k_cache = torch.randn(4, 1, 128, 128, device="cuda", dtype=torch.bfloat16)
cu_seqlens_q = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
cu_seqlens_k = torch.tensor([0, 128, 384], device="cuda", dtype=torch.int32)
page_table = torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32)
out = torch.empty(2, 5, 16, device="cuda", dtype=torch.int32)
wrapper = BatchPrefillIndexerWithPagedKVCacheWrapper()
wrapper.plan(
    cu_seqlens_q, cu_seqlens_k, page_table,
    total_q=5, max_seqlen_q=3, max_seqlen_k=256, num_index_heads=2,
)
indices = wrapper.run(q, k_cache, out=out)
assert indices.shape == (2, 5, 16)
```

`BatchPrefillIndexerWithPagedKVCacheWrapper` 提供统一的 `plan()` / `run()` 生命周期。
`plan()` 接收 varlen 与 page-table metadata，`run()` 接收当前层的 BF16 Q 和 paged BF16
K cache。调用方可以向 `run()` 传入预分配输出。

### 池化 Key 的 Block 分数

`BatchPrefillBlockScoreWrapper` 只返回分数，不执行 TopK，也不强制加入 local block。
每个存储 key 表示 `pooling_factor`（1、2 或 4）个原始 token，因此每个 128-token
分数 block 存储 `128 // pooling_factor` 个 key。

```python
from inference.msa_v1.indexer.prefill.bf16.interface import BatchPrefillBlockScoreWrapper

scorer = BatchPrefillBlockScoreWrapper()
scorer.plan(
    cu_seqlens_q, cu_seqlens_k, page_table,
    total_q=total_q, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
    num_index_heads=2, pooling_factor=4,
)
scores = scorer.run(q, pooled_k_cache, out=score_output)
```

## 数据契约

每个请求的总 KV 长度最多为 2,097,152 tokens（16384 页，每页 128 tokens），
包含当前 query chunk。支持 batch 内请求长度不同；每个 query 按自己的有效页数选择，
并始终保留其 local page。

- Q 和 K 为 BF16。
- 支持 1、2 或 4 个本地 index head。
- 输出为 `[num_index_heads, total_q, 16]`，内容为 logical page ID。有效项位于前缀，
  最后一个有效项必须是当前 query 的 local page。
- K cache 使用单个 KV head、128 的 head dimension 和 128 的 page size。
- Varlen metadata 和 page table 使用 CUDA `torch.int32`。

| 参数 | Shape | Dtype |
| --- | --- | --- |
| `q` | `[total_q,H,128]` | `torch.bfloat16` |
| `paged_k_cache` | `[physical_pages,1,128,128]` | `torch.bfloat16` |
| `cu_seqlens_q`, `cu_seqlens_k` | `[B+1]` | `torch.int32` |
| `page_table` | `[B,max_pages]` | `torch.int32` |
| `out` | `[H,total_q,16]` | `torch.int32` |

所有输入和输出须位于同一 CUDA device；Q、metadata 和输出须 contiguous、16-byte 对齐。累积长度从 0
开始且非递减，Q 的末项等于 `total_q`，每个请求的 KV 长度须覆盖 query 长度。
`max_seqlen_q`/`max_seqlen_k` 是各请求长度的 host 上界；KV 上界最多为 16384×128，
page table 必须覆盖该上界。默认 `num_index_heads=4`，支持 H=1/2/4。
每行有效页数由 bottom-right causal 位置决定：

```text
query_position = kv_length - query_length + query_index
local_page = query_position // 128
```

历史页评分为 FP32 Q/K 点积在该页 token 上的最大值乘以 `1/sqrt(128)`；
TopK 使用 16-bit 量化近似排序，分数接近时不保证与精确 FP32 排序相同。

### 池化 Key 的 Block 分数

Q 仍为 BF16 `[total_q,H,128]`。K 为 BF16 `[physical_pages,1,page_size // pooling_factor,128]`；
默认 `page_size=128` 时每个物理 page 对应一个分数 block。`page_size=64` 时，每个逻辑
128-token 分数 block 使用两个 page-table 项；pooling factor 为 4 时，每个 64-token
物理 page 存储 16 个 key。
带 padding 和非连续外层 stride 的 cache 遵循上文相同的布局约束。

两组累积长度都以原始 token 为单位。请求 b 的 query i 的原始位置为
`p = kv_length[b] - query_length[b] + i`。只有 `(j+1) * pooling_factor <= p+1`
时，存储 key j 才可见；未完成的 group 不贡献分数。输出为未缩放 FP32
`[H,total_q,ceil(max_seqlen_k/128)]`，包括当前 block 中已可见的 key。
没有可见完整 key 的 block、未来 block 和 padding query 行均为 `-inf`。
缩放、舍入、选择和不完整 group 策略由调用方负责。既有 indexer wrapper 的行为保持不变。

每个请求的 page table 容量必须覆盖 `ceil(max_seqlen_k/page_size)` 个物理 page。
长度必须非负、单调、从零开始，最终 Q offset 不超过计划的 total Q 容量，且
`kv_length >= query_length`。有效 page-table 项必须指向已初始化的 cache page。
这些设备值约束由调用方负责；`run()` 不做 device-to-host 校验或同步。

## 运行约束

- 仅支持 SM100/SM103 和 paged K cache。
- 要求 `nvidia-cutlass-dsl[cu13]>=4.5.2`；升级 DSL 后需要重新构建编译缓存。
- Logical page 可以离散、无序。
- `plan()` 必须在 CUDA Graph capture 外调用；首次 `run()` 也必须在 capture 外完成。
- Q/K、metadata、workspace 和输出必须位于同一 CUDA device，并满足接口校验的 shape、
  stride 和 alignment。

TopK wrapper 独占可写 workspace 和默认输出；默认输出会被后续 `run()` 覆盖。并发 stream
使用独立 wrapper，并用 event 保证 metadata 更新先于消费。保持 metadata 地址、shape、
总 Q 和容量不变时，长度原地更新后在 capture 外调用 `replan()`；page mapping 原地更新
须在消费前完成。地址、shape、head 数或容量变化时重新 `plan()` 并重新 capture。
Graph 使用期间保持 wrapper、输入和输出存活。

### 池化 Key 的 Block 分数

在 capture 外调用 `plan()`，并在 capture 前预热 `run()`。地址、shape、head 数和容量不变时，
长度和 page mapping 可以原地更新；每次调用和 Graph replay 都重建 metadata 并覆盖全部输出。
并发执行使用独立 wrapper，并通过 stream/event 排序 metadata 更新。
首次不传 `out` 的调用才分配 wrapper 持有的默认输出。
仅使用显式 `out` 时不分配默认分数 tensor；已分配的默认输出仍由 wrapper 持有，
并由后续不传 `out` 的调用覆盖。显式 `out` 必须为同设备、连续、16 字节对齐的
FP32，shape 与计划输出一致。KV 长度上限为 2,097,152 个原始 token；输出容量必须少于
2^31 个元素。

## 验证命令

```bash
MINIMAX_INFERENCE_TEST_SUITE=smoke python -m pytest tests/inference/msa_v1/indexer/prefill/bf16
python -m benchmarks.inference.msa_v1.indexer.prefill.bf16.benchmark --suite full --num-index-heads 2 --verify --out prefill_bf16.json
```

Benchmark 测量 CUDA Graph 中公开 `run()` 路径的耗时。`--verify` 在计时之外使用独立
reference，全量检查有效历史 scores 和 TopK 输出。

### 池化 Key 的 Block 分数

```bash
python -m pytest tests/inference/msa_v1/indexer/prefill/bf16/test_pooled_block_scores.py --msa-inference-suite=full
```

## Cache 布局

Cache 数据的最后一个存储维度（dim）须连续（`stride(-1) == 1`）。
Cache 数据地址及非单例外层维度的字节 stride 必须为 16 的倍数。
字节 stride 等于 tensor 元素 stride 乘以 `element_size()`。广播、重叠和负 stride cache
不在支持范围内。Q、metadata 和输出的布局要求见上文。

CUDA Graph capture 后，cache 地址、shape
或 stride 改变时须重新 capture。不同 wrapper 保持独立 workspace。
