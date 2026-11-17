# Decode Block 分数

[English](README.md)

## 功能

`BatchDecodeBlockScoreWrapper` 为每个覆盖 128 个原始 token 的逻辑 block 返回 FP32
分数。支持逐 token 或池化 Key，不执行 TopK，也不强制加入 local block。
既有 decode indexer wrapper 保持原有 TopK 行为。

## 公开 API

```python
import torch
from inference.msa_v1.indexer.decode.interface import BatchDecodeBlockScoreWrapper

q = torch.randn(2, 2, 2, 128, device="cuda", dtype=torch.bfloat16)
k_cache = torch.randn(4, 32, 128, device="cuda", dtype=q.dtype)
page_table = torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32)
seq_lens = torch.tensor([129, 256], device="cuda", dtype=torch.int32)
out = torch.empty(2, 4, 2, device="cuda", dtype=torch.float32)
scorer = BatchDecodeBlockScoreWrapper(input_dtype=q.dtype, pooling_factor=4)
scorer.plan(page_table, seq_lens, num_index_heads=2, query_length=2)
scores = scorer.run(q, k_cache, out=out)
assert scores.shape == (2, 4, 2)
```

构造参数默认为 `input_dtype=torch.float8_e4m3fn`、`pooling_factor=1`、
`page_size=128`、`include_partial_block=False`、`sm_scale=None`。
`plan()` 默认为 `num_index_heads=1` 和 `query_length=8`。

## 数据契约

P 表示物理 page 容量，F 表示池化比例，S 表示以原始 token 计数的 page 大小。
F 为 1、2 或 4，S 为 64 或 128。所有请求共享 Q（1–16）和 H（1/2/4）；
Key 只有一个共享 head。

| 参数 | Shape | Dtype |
| --- | --- | --- |
| `q` | `[B,Q,H,128]` | BF16 或 E4M3 |
| `k_cache` | `[physical_pages,S/F,128]` | 与 Q 一致 |
| `page_table` | `[B,P]` | `torch.int32` |
| `seq_lens` | `[B]` | `torch.int32` |
| `out` | `[H,B*Q,ceil(P*S/128)]` | `torch.float32` |

`seq_lens` 以原始 token 计数，包含当前 query chunk。请求 b 的 query i 位于
`p = seq_lens[b] - Q + i`。存储 key j 代表 F 个原始 token，只有
`(j+1)*F <= p+1` 时才可见；未完成的 group 不贡献分数。
`include_partial_block=False` 时只计算编号严格小于 `p // 128` 的历史 block；
为 `True` 时还计算当前 block 内已完成且可见的 group。
每次调用后，空 block 和无效 block 的分数均为 `-inf`。

每个 head、query 和可见 block 的输出为 `max(Q dot K) * scale`：

| `sm_scale` | BF16 的 scale | E4M3 的 scale |
| --- | --- | --- |
| 省略或 `None` | `1/sqrt(128)` | `1` |
| 显式有限正数 | 指定的值 | 指定的值 |

传入 `sm_scale=1.0` 可获得未缩放分数。FP32 点积和 block 最大值计算完成后只缩放一次。
接口不执行 softmax，也不接收 Key scale。

最多支持 16384 个逻辑 block（2,097,152 个原始 token）；page128 的 P 上限为 16384，
page64 的 P 上限为 32768。page64 的每个逻辑 block 合并两个独立映射的物理 page；
P 为奇数时最后一个 block 只有半块容量。有效 page ID 必须指向已初始化的 cache page。
长度和映射不得超过计划容量；空请求没有可见分数。

所有 tensor 必须位于同一 CUDA device。Q、metadata 和输出必须连续。
Q/K 指针及 K 的非 singleton 外层字节 stride 必须为 16 的倍数；K 最后一维 stride
必须为 1。不支持广播、重叠或负 stride 的 cache。

## 运行与 CUDA Graph

要求 SM100/SM103 和 `nvidia-cutlass-dsl[cu13]>=4.5.2`。在 capture 外构造对象并调用
`plan()`，随后用实际 Q/K 布局预热 `run()`。Capture 时必须显式传入预分配的 `out`；
capture 外省略 `out` 会在每次调用时分配新结果。每次调用覆盖完整输出。

长度变化后重新调用 `plan()`，或绑定 `inference.msa_v1.indexer.decode` 导出的
`BatchDecodeIndexerPlan`。共享 plan 的 `max_pages=P`、H、Q、F、`page_size` 和
`include_partial_block` 必须与 wrapper 匹配。长度原地变化后、消费者执行前调用
共享 plan 的 `update()`；它支持 capture。原地 page mapping 更新必须在消费者执行前
完成。跨 stream 的更新和消费通过 event 排序。

```python
from inference.msa_v1.indexer.decode import BatchDecodeIndexerPlan

plan = BatchDecodeIndexerPlan(
    seq_lens, max_pages=2, num_index_heads=2, query_length=2,
    pooling_factor=4, page_size=128, include_partial_block=False,
)
plan.update()
scorer.plan(page_table, seq_lens, num_index_heads=2, query_length=2, shared_plan=plan)
scorer.run(q, k_cache, out=out)  # Warm up before capture.
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    plan.update()
    scorer.run(q, k_cache, out=out)
graph.replay()
```

Graph 使用期间保持 wrapper、plan、输入 tensor 和输出存活。并发执行使用独立 wrapper
及可写 workspace。batch、Q/H、几何参数、metadata 绑定或捕获的 tensor 地址/布局变化时，
需要匹配的新计划和重新 capture。

## 验证命令

```bash
FMHA_SM100_ALLOW_JIT=1 python -m pytest tests/inference/msa_v1/indexer/decode/test_block_scores.py --msa-inference-suite=full
FMHA_SM100_ALLOW_JIT=1 python -m pytest tests/inference/msa_v1/indexer/decode --msa-inference-suite=full
python -m pytest tests/inference/test_blackwell_compile.py -k 'decode_indexer or decode_shared_plan'
```
