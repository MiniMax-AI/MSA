# Q8KV4 Prefill Attention

[English](README.md)

## 功能

面向 SM100/SM103/SM107 的 paged sparse causal prefill attention。Q 使用 E4M3，K/V 使用
packed E2M1 和 E4M3 scale，输出为 BF16，并支持 varlen chunk prefill。

Attention 概率按 `E4M3(P × 448)` 量化，并在归一化时补偿该缩放；返回的 LSE
保持原始 logits 的自然对数语义。概率量化语义与 decode 对齐，但不承诺最终输出跨路径逐位一致。

## 公开接口

```python
from inference.msa_v1.attention.prefill.q8kv4 import (
    BatchPrefillWithPagedKVCacheWrapper,
    PrefillWorkspace,
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

### Split workspace

forward 为每个 (query, head, 选中页) 写出一份 partial output 和 LSE，由 combine 合并。
`PrefillWorkspace` 持有这些 buffer。wrapper 默认在 `plan()` 中创建自己的 workspace；在同一
stream 上顺序执行的多个 wrapper 可以共享构造时传入的同一个 workspace，`partial_bytes` 可按
最大 batch 预留：

```python
workspace = PrefillWorkspace(device)
workspace = PrefillWorkspace(
    device, partial_bytes=workspace.required_bytes(max_total_q, num_q_heads)
)
wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace)
```

在具有多个 memory locality domain 的 GPU（Rubin uGPU、Blackwell）上，workspace 将 partial
buffer 按 domain 条带化，combine 的每个 tile 都从其所在 SM 的 domain 读取；结果与 interleaved
内存完全一致。batch 超出已条带化的 buffer 时会先同步设备再替换，因此应在 CUDA Graph capture
前按最大 batch 预留。`MM_SPARSE_PREFILL_Q8KV4_LOCALITY_DOMAINS=0` 保持 interleaved 布局
（用于对比）。同一个 workspace 不得同时用于两个 stream。

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
- 要求 CUDA Toolkit 13.4 或更高版本；SM107 要求 13.5 或更高版本。NVFP4 dequant 在工具链
  支持时使用 QMUL4，否则（SM107）使用精确的 packed-FP16 路径。

## Cache 布局

Cache 数据的最后一个存储维度（dim）须连续（`stride(-1) == 1`）。
Cache 数据地址及非单例外层维度的字节 stride 必须为 16 的倍数。
Scale 每页内部的 `[Hkv, 128, 8]` 必须紧密排列。
仅页间 stride 可以包含填充；它必须覆盖完整一页，且为 16 字节的倍数。Scale 起始地址须
16 字节对齐。数据与 scale 可以使用不同的 page stride。Scale 不支持 token/head 间填充
或外层维度置换。
K/V 的 stride 须一致；K/V scale 的 stride 也须一致。
字节 stride 等于 tensor 元素 stride 乘以 `element_size()`。广播、重叠和负 stride cache
不在支持范围内。Q、metadata 和输出的布局要求见上文。

CUDA Graph capture 后，cache 地址、shape
或 stride 改变时须重新 capture。不同 wrapper 保持独立 workspace。
