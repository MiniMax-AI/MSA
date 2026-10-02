# BF16 Sparse Decode Attention

[English](README.md)

## 功能

使用外部 FlashInfer TRTLLM-GEN block-sparse backend 的 paged sparse attention。
Q/K/V 与输出均为 BF16，每个 query 和 KV head 有独立的 logical page 选择。

## 公开接口

```python
import torch
from inference.msa_v1.attention.decode.bf16 import BatchDecodeWithPagedKVCacheWrapper

query_length = 2
q = torch.randn(4, 16, 128, device="cuda", dtype=torch.bfloat16)
k_cache = torch.randn(4, 2, 128, 128, device="cuda", dtype=torch.bfloat16)
v_cache = torch.randn_like(k_cache)
page_table = torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32)
seq_lens = torch.tensor([256, 256], device="cuda", dtype=torch.int32)
topk_indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32).repeat(4, 2, 1)
out = torch.empty_like(q)
wrapper = BatchDecodeWithPagedKVCacheWrapper(enable_pdl=True)
wrapper.plan(
    topk_indices, page_table, seq_lens,
    q_len_per_req=query_length,
    num_q_heads=q.shape[1],
    num_kv_heads=k_cache.shape[1],
)
output = wrapper.run(q, (k_cache, v_cache), out=out)
assert output.shape == q.shape
```

`enable_pdl` 默认为 `True`，设为 `False` 可禁用 programmatic dependent launch。
`sm_scale` 可覆盖默认的 `1/sqrt(128)` attention scale。

省略 `out` 时返回 wrapper 持有的默认输出，后续 `run()` 会覆盖它；需要保留结果时
使用独立的输出。并发调用须使用独立 wrapper 和可写 workspace。

## 数据契约

| 参数 | Shape | Dtype |
| --- | --- | --- |
| `q` | `[B*Q,num_q_heads,128]` | `torch.bfloat16` |
| `k_cache`, `v_cache` | `[physical_pages,num_kv_heads,128,128]` | `torch.bfloat16` |
| `topk_indices` | `[B*Q,num_kv_heads,K]`，K=1–16 | `torch.int32` |
| `page_table` | `[B,max_pages]` | `torch.int32` |
| `seq_lens` | `[B]` | `torch.int32` |
| `out` | 与 `q` 相同 | `torch.bfloat16` |

所有 tensor 必须在同一 CUDA device。Q、metadata 和输出须 contiguous；K/V 的页与
head 维度可有 stride，token/head-dim stride 必须为 128/1，K/V 的 shape 和 stride
须相同。Q/K/V 和输出须 16-byte 对齐。每个 KV head 对应 8 或 16 个 Q head。
所有请求共用 Q，序列长度包含 Q-token query chunk，query 位置为 `seq_lens[b] - Q + t`。

TopK 保存 logical page ID，有效前缀后的 padding 为 `-1`。历史页不能重复，可以无序；
最后一个有效页必须为该 query 的 local page。Indexer 的 `[H,B*Q,16]` head-major 输出
需要显式转换为 attention metadata 布局，Q/K/V 无需此转换。

`plan()` 默认 `num_q_heads=64`、`num_kv_heads=4`；`q_len_per_req` 为必填正整数，
每个 `seq_lens` 必须至少为 Q。每行有效 TopK 项数必须为 `min(K, local_page + 1)`，
page table 的有效映射须指向 K/V cache 范围内的物理页。

## 运行约束

通过 `python -m pip install -e '.[flashinfer]'` 安装可选依赖
`flashinfer-python==0.6.17`，运行环境需要可用的公开 TRTLLM-GEN kernel artifacts。
若安装 `flashinfer-cubin` 或 `flashinfer-jit-cache`，版本必须与 Python 包匹配；
不匹配时 FlashInfer 会报版本错误。`flashinfer-cubin==0.6.17` 可选 wheel 来自
[FlashInfer wheel index](https://flashinfer.ai/whl)：

```bash
python -m pip install flashinfer-cubin==0.6.17 --index-url https://flashinfer.ai/whl
```

支持 GB200/B200（SM100）和 GB300/B300（SM103）。

TopK、page mapping 或长度变化后，须在 Graph capture 外重新 `plan()`。
Capture 前 warmup `run()`，capture 时提供预分配的 `out`。Graph 使用期间保持 wrapper
存活，并发请求或 stream 使用独立 wrapper。本接口仅返回 attention 输出，不返回 LSE。

## 验证命令

```bash
MINIMAX_INFERENCE_TEST_SUITE=smoke python -m pytest tests/inference/msa_v1/attention/decode/bf16
python -m benchmarks.inference.msa_v1.attention.decode.bf16.benchmark --suite full --out bf16_attention.json
```

Benchmark 报告公开 `run()` 的 CUDA Graph latency，不包含 plan、分配、编译和 warmup，
并使用独立 reference 检查整个输出。传入 `--disable-pdl` 可测量禁用 PDL 后的同一路径。
