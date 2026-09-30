# Q8KV8 Prefill Indexer

[English](README.en.md)

## 功能

面向 SM100/SM103 的 true-varlen paged prefill indexer。Q/K 均使用 E4M3，输出每个
query 选中的 logical page indices，并保证包含 local page。

## 公开接口

```python
from inference.msa_v1.indexer.prefill.q8kv8 import (
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
    num_index_heads=4,
)
topk_indices = wrapper.run(q, paged_k_cache)
```

调用方可以向 `run()` 传入预分配输出。请求长度发生变化时，应在 CUDA Graph capture
外重新调用 `plan()` 或公开的 `replan()`。

## 数据契约

- `num_index_heads` 默认为 1，支持 1/2/4。各 head 独立选择 TopK，共享 K cache。
- Q/K 和输出必须为同设备 CUDA contiguous tensor。
- H=1 的输出也保留 head 维。

- `q`：`[total_q, H, 128]`，E4M3。
- `paged_k_cache`：`[physical_pages, 1, 128, 128]`，E4M3。
- `cu_seqlens_q` / `cu_seqlens_k`：`[B + 1]`，CUDA `torch.int32`。
- `page_table`：`[B, max_pages]`，logical-to-physical page mapping。
- 输出：`[H, total_q, 16]`，`torch.int32` logical page indices。有效项位于前缀，最后
  一个有效项必须是 local page。

## 运行约束

- 使用 bottom-right causal 对齐，`max_seqlen_k` 必须不小于 `max_seqlen_q`。
- 历史 page 的物理映射允许乱序和不连续。
- `plan()` 和 `replan()` 必须在 CUDA Graph capture 外调用；首次 capture 前必须完成
  warmup。
- 支持 SM100/SM103，并要求 `nvidia-cutlass-dsl[cu13]>=4.5.2`。
- 使用仓库的 CUTLASS 子模块，或通过 `CUTLASS_ROOT` 指向公开 CUTLASS headers；
  安装 wheel 后应设置此环境变量。
- 默认 AOT cache 目录为 `~/.cache/minfer/msa_v1`，可通过 `MSA_V1_AOT_CACHE`
  覆盖。设置 `MSA_V1_AOT_DISABLE=1` 可禁用 AOT cache。Cache miss 时默认禁止 JIT；
  需要 JIT 时，显式设置 `FMHA_SM100_ALLOW_JIT=1`，并在 CUDA Graph capture
  前完成编译。

## 验证命令

在仓库根目录运行；正确性自动覆盖 H=1/2/4。

```bash
MINIMAX_INFERENCE_TEST_SUITE=smoke python -m pytest tests/inference/msa_v1/indexer/prefill/q8kv8/test_real_cases.py
MINIMAX_INFERENCE_TEST_SUITE=full python -m pytest tests/inference/msa_v1/indexer/prefill/q8kv8/test_real_cases.py
python -m benchmarks.inference.msa_v1.indexer.prefill.q8kv8.benchmark --suite full --num-index-heads 4 --verify
```

Benchmark 使用公开 `run()` 的 CUDA Graph E2E 时间，按生产权重汇总有效计算吞吐。

传入 `--baseline <baseline.json>` 可检查与基线的差异。数据分布和统计方法见
[共享 benchmark 契约](../../../../../datas/inference/README.md)。
