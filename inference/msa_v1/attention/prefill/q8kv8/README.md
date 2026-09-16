# Q8KV8 Prefill Attention

[English](README.en.md)

## 功能

面向 SM100/SM103 的 paged sparse causal prefill attention。Q/K/V 均使用 E4M3，输出为
BF16，支持 varlen chunk prefill，且不使用 K/V scale。

Attention 概率按 `E4M3(P × 448)` 量化，并在归一化时补偿该缩放；返回的 LSE
保持原始 logits 的自然对数语义。概率量化语义与 decode 对齐，但不承诺最终输出跨路径逐位一致。
使用硬件 exp2；与 decode 的对齐范围为缩放和补偿语义，不要求 P 逐位一致。

## 公开接口

```python
from inference.msa_v1.attention.prefill.q8kv8 import (
    BatchPrefillWithPagedKVCacheWrapper,
)

wrapper = BatchPrefillWithPagedKVCacheWrapper()
wrapper.plan(
    topk_indices,
    cu_seqlens_q,
    cu_seqlens_k,
    page_table,
    num_q_heads=64,
    num_kv_heads=4,
    total_k=total_k,
    total_rows=total_rows,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
)
out, lse = wrapper.run(q, (k_cache, v_cache), return_lse=True)
```

调用方可以向 `run()` 传入预分配的 `out` 和 `lse`。

## 数据契约

- `q`：`[total_q, Hq, 128]`，E4M3。
- `k_cache` / `v_cache`：`[physical_pages, Hkv, 128, 128]`，E4M3。
- `topk_indices`：`[Hkv, total_q, 16]`，logical page ID。有效项位于前缀，无效后缀
  为 `-1`；历史页允许离散无序，local page 必须是最后一个有效项。
- `cu_seqlens_q` / `cu_seqlens_k`：`[B + 1]`，CUDA `torch.int32`。
- `page_table`：`[B, max_pages]`，logical-to-physical page mapping。
- `out`：`[total_q, Hq, 128]`，BF16；`lse`：`[total_q, Hq]`，FP32。

## 运行约束

- 仅支持 paged KV 和 causal attention；chunk prefill 使用 bottom-right causal 对齐。
- `Hq / Hkv` 支持 1、2、4、8 或 16；默认 `Hq=64`、`Hkv=4`。
- `cu_seqlens_k` 是 KV 长度的唯一来源；接口不接收 K/V scale。
- `plan()` 必须在 CUDA Graph capture 外调用；capture 时应预分配 `out` 和 `lse`。
