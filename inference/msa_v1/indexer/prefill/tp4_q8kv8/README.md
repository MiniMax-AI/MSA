# TP4 Q8KV8 Prefill Indexer

[English](README.en.md)

## 功能

面向 SM100/SM103 的 true-varlen paged prefill indexer。Q/K 均使用 E4M3，输出每个
query 选中的 logical page indices，并保证包含 local page。

## 公开接口

```python
from inference.msa_v1.indexer.prefill.tp4_q8kv8 import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)

wrapper = BatchPrefillIndexerWithPagedKVCacheWrapper()
wrapper.plan(
    cu_seqlens_q,
    cu_seqlens_k,
    page_table,
    total_q=total_q,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
)
topk_indices = wrapper.run(q, paged_k_cache)
```

调用方可以向 `run()` 传入预分配输出。请求长度发生变化时，应在 CUDA Graph capture
外重新调用 `plan()` 或公开的 `replan()`。

## 数据契约

- `q`：`[total_q, 1, 128]`，E4M3。
- `paged_k_cache`：`[physical_pages, 1, 128, 128]`，E4M3。
- `cu_seqlens_q` / `cu_seqlens_k`：`[B + 1]`，CUDA `torch.int32`。
- `page_table`：`[B, max_pages]`，logical-to-physical page mapping。
- 输出：`[total_q, 16]`，`torch.int32` logical page indices。有效项位于前缀，最后
  一个有效项必须是 local page。

## 运行约束

- 使用 bottom-right causal 对齐，`max_seqlen_k` 必须不小于 `max_seqlen_q`。
- 历史 page 的物理映射允许乱序和不连续。
- `plan()` 和 `replan()` 必须在 CUDA Graph capture 外调用；首次 capture 前必须完成
  warmup。
- 支持 SM100/SM103，并要求 `nvidia-cutlass-dsl[cu13]>=4.5.2`。
- 默认 AOT cache 目录为 `~/.cache/minfer/msa_v1`，可通过 `MSA_V1_AOT_CACHE`
  覆盖。设置 `MSA_V1_AOT_DISABLE=1` 可禁用 AOT cache。Cache miss 时默认禁止 JIT；
  开发环境需要 JIT 时，显式设置 `FMHA_SM100_ALLOW_JIT=1`，并在 CUDA Graph capture
  前完成编译。
