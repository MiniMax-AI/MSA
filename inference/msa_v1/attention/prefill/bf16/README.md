# BF16 Paged Sparse Prefill Attention

[English](README.en.md)

## 功能

面向 SM100/SM103 的 BF16 paged sparse causal prefill attention，支持 TP1 和 TP4
下的 GQA 配置，并可选输出 FP32 LSE。

## 公开接口

```python
from inference.msa_v1.attention.prefill.bf16 import (
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

- Q/K/V 和输出均为 BF16；可选 `lse` 为 FP32。
- Head dimension 和 page size 均为 128。
- 支持 8 和 16 的 GQA group size；默认使用 64 个 query heads 和 4 个 KV heads。
- K/V cache 可以是连续的 `[physical_pages, Hkv, 128, 128]`，也可以是满足相同逻辑
  维度、dtype 和对齐约束的 SGLang-style strided page view。
- `topk_indices` 使用 logical page ID。有效项位于前缀，无效后缀为 `-1`；page 可以
  离散无序，最后一个有效项必须是 local page。
- `cu_seqlens_q` / `cu_seqlens_k` 为 CUDA `torch.int32` varlen metadata。

## 运行约束

- 仅支持 paged KV 和 causal attention；chunk prefill 使用 bottom-right causal 对齐。
- `cu_seqlens_k` 是 KV 长度的唯一来源。
- `plan()` 必须在 CUDA Graph capture 外调用；首次 `run()` 也必须在 capture 外完成。
- Runtime K/V 不会被 wrapper 隐式复制、重排或转换。
- 所有输入必须位于同一 CUDA device，并满足接口校验的 shape、stride 和 alignment。
