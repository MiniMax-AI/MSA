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

## 数据契约

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

所有输入和输出须 contiguous、16-byte 对齐且位于同一 CUDA device。累积长度从 0
开始且非递减，Q 的末项等于 `total_q`，每个请求的 KV 长度须覆盖 query 长度。
`max_seqlen_q`/`max_seqlen_k` 是各请求长度的 host 上界；KV 上界最多为 8192×128，
page table 必须覆盖该上界。默认 `num_index_heads=4`，支持 H=1/2/4。
每行有效页数由 bottom-right causal 位置决定：

```text
query_position = kv_length - query_length + query_index
local_page = query_position // 128
```

历史页评分为 FP32 Q/K 点积在该页 token 上的最大值乘以 `1/sqrt(128)`；
TopK 使用 16-bit 量化近似排序，分数接近时不保证与精确 FP32 排序相同。

## 运行约束

- 仅支持 SM100/SM103 和 paged K cache。
- 要求 `nvidia-cutlass-dsl[cu13]>=4.5.2`；升级 DSL 后需要重新构建编译缓存。
- Logical page 可以离散、无序。
- `plan()` 必须在 CUDA Graph capture 外调用；首次 `run()` 也必须在 capture 外完成。
- Q/K、metadata、workspace 和输出必须位于同一 CUDA device，并满足接口校验的 shape、
  stride 和 alignment。

Wrapper 独占可写 workspace 和默认输出；默认输出会被后续 `run()` 覆盖。并发 stream
使用独立 wrapper，并用 event 保证 metadata 更新先于消费。保持 metadata 地址、shape、
总 Q 和容量不变时，长度原地更新后在 capture 外调用 `replan()`；page mapping 原地更新
须在消费前完成。地址、shape、head 数或容量变化时重新 `plan()` 并重新 capture。
Graph 使用期间保持 wrapper、输入和输出存活。

## 验证命令

```bash
MINIMAX_INFERENCE_TEST_SUITE=smoke python -m pytest tests/inference/msa_v1/indexer/prefill/bf16
python -m benchmarks.inference.msa_v1.indexer.prefill.bf16.benchmark --suite full --num-index-heads 2 --verify --out prefill_bf16.json
```

Benchmark 测量 CUDA Graph 中公开 `run()` 路径的耗时。`--verify` 在计时之外使用独立
reference，全量检查有效历史 scores 和 TopK 输出。
