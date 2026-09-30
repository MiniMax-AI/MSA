# Q8KV4 Prefill Attention

[English](README.en.md)

## 功能

面向 SM100/SM103 的 paged sparse causal prefill attention。Q 使用 E4M3，K/V 使用
packed E2M1 和 E4M3 scale，输出为 BF16，并支持 varlen chunk prefill。

Attention 概率按 `E4M3(P × 448)` 量化，并在归一化时补偿该缩放；返回的 LSE
保持原始 logits 的自然对数语义。概率量化语义与 decode 对齐，但不承诺最终输出跨路径逐位一致。

## 公开接口

```python
from inference.msa_v1.attention.prefill.q8kv4 import (
    BatchPrefillWithPagedKVCacheWrapper,
)

wrapper = BatchPrefillWithPagedKVCacheWrapper()
wrapper.plan(
    topk_indices,
    cu_seqlens_q,
    cu_seqlens_k,
    page_table,
    total_k=total_k,
    total_rows=total_rows,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
)
out, lse = wrapper.run(
    q,
    (packed_k_cache, packed_v_cache),
    kv_cache_sf=(k_scale, v_scale),
    return_lse=True,
)
```

调用方可以向 `run()` 传入预分配的 `out` 和 `lse`。

## 数据契约

- `q`：`[total_q, Hq, 128]`，E4M3；`Hq = 16 * Hkv`。
- `packed_k_cache` / `packed_v_cache`：`[physical_pages, Hkv, 128, 64]`，packed E2M1。
- `k_scale` / `v_scale`：`[physical_pages, Hkv, 128, 8]`，E4M3，线性非 swizzle 布局。
- `topk_indices`：`[Hkv, total_q, 16]`，logical page ID。有效项位于前缀，无效后缀为
  `-1`；历史页允许离散无序，local page 必须是最后一个有效项。
- `cu_seqlens_q` / `cu_seqlens_k`：`[B + 1]`，CUDA `torch.int32`。
- `page_table`：`[B, max_pages]`，logical-to-physical page mapping。
- `out`：`[total_q, Hq, 128]`，BF16；`lse`：`[total_q, Hq]`，FP32。

## 运行约束

- 仅支持 paged KV 和 causal attention；chunk prefill 使用 bottom-right causal 对齐。
- head 数由 TopK 的首维确定，支持 64/4 和 16/1 本地 query/KV heads，无需复制 heads。
- `cu_seqlens_k` 是 KV 长度的唯一来源。
- `plan()` 必须在 CUDA Graph capture 外调用；capture 时应预分配 `out` 和 `lse`。
- NVFP4 dequant 要求 CUDA Toolkit 13.4 或更高版本。
