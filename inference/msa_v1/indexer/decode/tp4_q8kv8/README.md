# TP4 Q8KV8 Decode Indexer

[English](README.en.md)

## 功能

面向 SM100/SM103 的 paged decode indexer。Q/K 均使用 E4M3，输出每个 query 选中的 logical
page indices，并保证包含 local page。接口不接收 `k_scale`。

## 公开接口

```python
from inference.msa_v1.indexer.decode.tp4_q8kv8 import (
    BatchDecodeIndexerWithPagedKVCacheWrapper,
)

wrapper = BatchDecodeIndexerWithPagedKVCacheWrapper()
wrapper.plan(page_table, seq_lens)
topk_indices = wrapper.run(q, paged_k_cache)
```

可通过 `workspace_size(batch_size)` 查询 workspace 大小并向 wrapper 传入外部 workspace。
CUDA Graph 模式还需要在构造 wrapper 时传入地址稳定的 `page_table_buffer` 和
`seq_lens_buffer`，并向 `run()` 传入预分配输出。

## 数据契约

- `q`：`[B, 8, 128]`，E4M3。
- `paged_k_cache`：`[physical_pages, 128, 128]`，E4M3。
- `page_table`：`[B, max_pages]`，CUDA `torch.int32` logical-to-physical page mapping。
- `seq_lens`：`[B]`，CUDA `torch.int32`，包含当前 8-token MTP chunk。
- 输出：`[B * 8, 16]`，`torch.int32` logical page indices。有效项位于前缀，最后
  一个有效项为 local page。

## 运行约束

- 对 query `q_idx`，local page 为 `(seq_lens[b] - 8 + q_idx) // 128`。
- 历史 page 的物理映射允许乱序和不连续。
- `plan()` 必须在 CUDA Graph capture 外调用。
- 支持 B200/SM100 和 B300/SM103，并要求 `nvidia-cutlass-dsl[cu13]>=4.5.2`。
