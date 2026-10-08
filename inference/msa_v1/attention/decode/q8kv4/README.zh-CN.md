# Q8KV4 Decode Attention

[English](README.md)

## 功能

面向 SM100/SM103/SM107 的 paged sparse causal decode attention。Q 使用 E4M3，K/V 使用
packed E2M1 和 E4M3 scale。默认输出为 BF16；可选 Stream-K 模式输出供兼容 OProj
消费的 MXFP8 activation 和 scale。
支持 B200/B300 上的 GQA=8 和 GQA=16；SM107 保留 GQA=16 支持。
后端按实际 `Hq/Hkv` 选择。

## 公开接口

```python
from inference.msa_v1.attention.decode.q8kv4 import (
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
out = wrapper.run(
    q,
    (packed_k_cache, packed_v_cache),
    kv_cache_sf=(k_scale, v_scale),
)
```

`plan()` 接收请求级 metadata；`run()` 接收当前层的 Q/K/V 和量化 scale。调用方可以
向 `run()` 传入预分配的 `out`。

要输出 MXFP8，在 `plan()` 中设置 `output_mode="mxfp8"`。此时 `run()` 返回
`(data, scales)`：`data` 为 E4M3 `[B * q_len_per_req, Hq * 128]`，`scales` 为
FlashInfer 128x4 swizzled 布局的一维 `torch.uint8` UE8M0 tensor，长度为
`ceil(B * q_len_per_req / 128) * Hq * 512`。数值契约是先得到 BF16 Attention
输出，再以 FP32 按每 32 个元素进行 MXFP8 量化。CUDA Graph capture 时传入预分配
的 `out` 和 `out_scale`。默认 `output_mode="bf16"` 保持原有输出形状和返回类型。
wrapper 的 scale buffer 在 `plan()` 时清零。若传入独立的 `out_scale`，应在 capture 前
将其 padding 行清零；kernel 只写有效输出行的 scale。

## 数据契约

- `q`：`[B * q_len_per_req, Hq, 128]`，E4M3。
- `packed_k_cache` / `packed_v_cache`：`[physical_pages, Hkv, 128, 64]`，`torch.uint8`
  存储 packed E2M1，每个 byte 包含两个元素。
- `k_scale` / `v_scale`：`[physical_pages, Hkv, 128, 8]`，E4M3，线性非 swizzle 布局，
  每 16 个连续 head-dimension 元素共享一个 scale；该量化分组不随 GQA ratio 改变。
- `topk_indices`：`[B * q_len_per_req, Hkv, 16]`，logical page ID。有效项位于前缀，
  无效后缀为 `-1`；历史页允许乱序和不连续，local page 必须是最后一个有效项。
- `page_table`：`[B, max_pages]`，logical-to-physical page mapping。
- `seq_lens`：`[B]`，包含当前 decode/MTP query chunk 的最终 KV 长度。
- `out`：默认 `[B * q_len_per_req, Hq, 128]`，BF16；MXFP8 模式见上文。

所有 tensor 必须 contiguous 且位于同一 CUDA device。`topk_indices`、`page_table` 和
`seq_lens` 的 dtype 为 `torch.int32`；Q、K/V、scale 和输出的起始地址必须 16-byte aligned。

## 运行约束

- `q_len_per_req` 是任意正整数的运行时参数。
- B200/B300 上 `Hq/Hkv` 为 8 或 16；默认仍为 `Hq=64`、`Hkv=4`。
  GQA=8 示例使用 `Hq=32`、`Hkv=4`，也支持 `8/1` 和 `16/2` 等配置。
- `q_len_per_req=8` 对应 1 个主 token 加 7 个预测 token；各 query 保持独立的
  TopK 和 causal 位置。
- `plan()` 必须在 CUDA Graph capture 外调用；capture 时应传入预分配的 `out`。
- 相同 batch、query shape 和容量下，可通过原位更新 `seq_lens`、`page_table`、TopK
  复用已有 plan / Graph，包括请求变短、padding 和重新填充。tensor 地址、shape、dtype
  必须保持不变，更新与 `run()` 必须在同一 stream 上有序执行；改变容量或地址后需重新 plan / capture。
- query 的 causal 位置小于零时输出为零，TopK 整行须为 `-1`。
- `num_kv_splits` 可选 1/2/4/8；显式指定时严格使用该 split 数，默认自动选择。
  所有选项均支持上述复用。不同 wrapper 的 workspace 独立，销毁 wrapper 后可以释放；
  并发 stream 或 CUDA Graph 应各自使用独立 wrapper。
- MXFP8 模式在自动选择 split 数时使用 Stream-K。显式指定大于 1 的 split 数会选择
  legacy multi-split 路径，MXFP8 模式会明确拒绝，不会增加辅助 reduction kernel。
- 多个进程可以共享 `TORCH_EXTENSIONS_DIR`，缓存所在文件系统须支持 POSIX 文件锁。
  每个进程使用独立的 wrapper；冷编译会串行发布同一编译产物。
- SM100/SM103 最低支持 CUDA Toolkit 12.9；SM107 要求 CUDA Toolkit 13.5 或更新版本。
  支持 QMUL4 的工具链会自动使用该路径，否则自动使用精确的 FP16 dequant fallback；
  调用方不需要选择后端。

## 验证命令

安装上述依赖后，在仓库根目录执行：

```bash
MINIMAX_INFERENCE_TEST_SUITE=full python -m pytest tests/inference/msa_v1/attention/decode/q8kv4
python -m benchmarks.inference.msa_v1.attention.decode.q8kv4.benchmark --suite full --num-q-heads 32 --num-kv-heads 4
```

完整正确性测试包含 GQA=8 和 GQA=16。benchmark 测量公开 `run()` 的
CUDA Graph E2E 延迟；将 `--num-q-heads` 改为 64 可测 GQA=16。
