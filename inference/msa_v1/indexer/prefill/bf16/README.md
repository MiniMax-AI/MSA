# BF16 Paged Prefill Indexer

[English](README.en.md)

## 功能

面向 SM100/SM103 的 BF16 paged prefill indexer。它为每个 query 选择 logical pages，
并保证结果包含 local page。

## 公开接口

`BatchPrefillIndexerWithPagedKVCacheWrapper` 提供统一的 `plan()` / `run()` 生命周期。
`plan()` 接收 varlen 与 page-table metadata，`run()` 接收当前层的 BF16 Q 和 paged BF16
K cache。调用方可以向 `run()` 传入预分配输出。

## 数据契约

- Q 和 K 为 BF16。
- 支持 1 或 4 个本地 index head。
- 输出为 `[num_index_heads, total_q, 16]`，内容为 logical page ID。有效项位于前缀，
  最后一个有效项必须是当前 query 的 local page。
- K cache 使用单个 KV head、128 的 head dimension 和 128 的 page size。
- Varlen metadata 和 page table 使用 CUDA `torch.int32`。

## 运行约束

- 仅支持 SM100/SM103 和 paged K cache。
- 要求 `nvidia-cutlass-dsl[cu13]>=4.5.2`；升级 DSL 后需要重新构建编译缓存。
- Logical page 可以离散、无序。
- `plan()` 必须在 CUDA Graph capture 外调用；首次 `run()` 也必须在 capture 外完成。
- Q/K、metadata、workspace 和输出必须位于同一 CUDA device，并满足接口校验的 shape、
  stride 和 alignment。
