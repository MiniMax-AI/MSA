# Q8K8 Paged Sparse Decode Attention

[English](README.en.md)

## 功能

该模块把 MSA 的 per-query sparse metadata 适配到外部 FlashInfer Q8K8 block-sparse
decode backend。Q/K/V 使用 E4M3，输出为 BF16。该接口不会隐式执行 FP4 dequant。

## 公开接口

```python
from inference.msa_v1.attention.decode.q8kv8 import (
    BatchDecodeWithPagedKVCacheWrapper,
)

wrapper = BatchDecodeWithPagedKVCacheWrapper()
wrapper.plan(
    topk_indices,
    page_table,
    seq_lens,
    q_len_per_req=8,
    num_q_heads=32,
    num_kv_heads=4,
)
out = wrapper.run(q, (k_cache, v_cache))
```

调用方可以向 `run()` 传入预分配的 `out`。

## 数据契约

- `q`：`[B * q_len_per_req, Hq, 128]`，E4M3。
- `k_cache` / `v_cache`：`[physical_pages, Hkv, 128, 128]`，E4M3。
- `topk_indices`：`[B * q_len_per_req, Hkv, topk]`，其中 `topk <= 16`。有效项位于
  前缀，无效后缀为 `-1`；历史页允许离散无序，local page 必须是最后一个有效项。
- `page_table`：`[B, max_pages]`，logical-to-physical page mapping。
- `seq_lens`：`[B]`，包含当前 decode/MTP query chunk 的最终 KV 长度。
- `out`：`[B * q_len_per_req, Hq, 128]`，BF16。

默认配置为 `Hq=64`、`Hkv=4`、head dimension 128 和 page size 128。

`topk_indices`、`page_table` 和 `seq_lens` 必须为 contiguous 的 CUDA `torch.int32`
tensor；Q 和输出也必须 contiguous。所有 tensor 位于同一 CUDA device。

## 运行约束

FlashInfer 是可选外部依赖。仓库不复制或分发 FlashInfer 源码与 cubin；运行前必须安装
包含兼容 block-sparse decode backend 的 FlashInfer 版本。`plan()` 必须在 CUDA Graph
capture 外调用，并在首次 capture 前完成一次 `run()` warmup；capture 时应使用预分配输出。

- B200/SM100、B300/SM103 支持 GQA=8/16；`q_len_per_req=8` 表示 1 个主 token 加 7 个 MTP token。
- 使用 FlashInfer 的路径要求 `seq_lens[b] >= q_len_per_req`，历史 TopK 必须唯一且位于 local page 之前。
- FlashInfer 路径的 `plan()` 会校验 GPU metadata，并可能同步到 host；metadata 内容变化后必须重新 `plan()`，已有 Graph 需要重新 capture。
- 首次 capture 前完成一次 `run()` warmup，capture 时预分配 `out`。不同并发 stream/Graph 使用独立 wrapper。
- Q、K/V 和输出的起始地址必须 16-byte aligned，所有张量位于同一 GPU。
- `Hq/Hkv` 支持 8 或 16，均使用 FlashInfer；默认仍为 `64/4`。K/V 最后两维必须连续，head/page 维允许非连续且 K/V stride 必须一致。

在仓库根目录安装可选依赖，保持项目 CuTe DSL 4.5.2：

```bash
python -m pip install -e '.[flashinfer]'
```

该 extra 固定 `flashinfer-python==0.6.17`。首次使用可能下载或构建官方 kernel；缺少兼容 backend/cubin 时会报错。
仓库不分发 FlashInfer 源码或 cubin。

## 验证命令

在仓库根目录执行；测试自动覆盖 `32/4` 和 `64/4`，benchmark 通过实际 head 数选择配置。

```bash
python -m pytest tests/inference/msa_v1/attention/decode/q8kv8 -q -s --msa-inference-suite=smoke
python -m pytest tests/inference/msa_v1/attention/decode/q8kv8 -q -s --msa-inference-suite=full
python -m pytest tests/inference/dequant/nvfp4_to_fp8/test_sparse_flashinfer.py -q -s
python -m benchmarks.inference.msa_v1.attention.decode.q8kv8.benchmark --suite full --num-q-heads 32 --num-kv-heads 4
```
