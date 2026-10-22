# M3 Decode Attention

[English](README.md) | 简体中文

统一入口为 `inference.high_tps.msa_v1.attention.create_attention_runner`。提供 MiniMax-M3.1 decode 的 TP4 MegaAttention backend：

| Backend | 执行方式 |
| --- | --- |
| `mega`（默认） | 从输入归一化到 post-attention norm 的单 kernel |

底层入口 `create_attention_plan` 和 `m3_decode_mega_atten` 保持可用。

`mega` 默认使用 `AttentionConfig(enable_placement=False)`；底层
`m3_decode_mega_atten` 接受同名参数。该模式要求 canonical 硬件 cluster 排布。
并发 GPU 工作可能改变排布；不匹配时会报告 `M3_DECODE_CANONICAL_PLACEMENT_REQUIRED`
和 CUDA launch failure。需要兼容非 canonical 排布时使用 `enable_placement=True`。
launch failure 后须终止整组 TP worker，并重建 CUDA context。SGLang 插件默认开启
placement，仅在 forward 期间没有其他 stream 运行 kernel 时关闭。

Placement 和诊断超时模式选择独立编译产物。须在 Graph capture 前选择模式并预热；
切换模式需要新建 runner/plan 并重新捕获 Graph。固定静态模式后，全部合法 KV 长度和
page capacity 复用一个编译产物，每次调用仍只有一个 kernel。不同模式可串行共享 workspace。

`AttentionPlan.run()` 返回 `(attention_residual, post_attention_norm)`，均为 BF16 `[Q,6144]`。物理输入和输出存储仍为 8 行，末尾 Q 行有效。包括 Q<8 在内，`plan_info` 均须按 8 个物理 query 行创建。Q<8 时须将前 `8-Q` 行的 slot mapping 设为 `-1`，避免填充行修改 cache。调用方负责 token position、slot mapping 与因果长度一致；不能直接传入 Q<8 的紧凑输入存储。

## 输入与生命周期

支持单请求、Q=1–8、TP4、hidden=6144、16 Q heads、1 KV head、head_dim=128、page_size=128 和 TopK=16。Query buffer 为 FP8 E4M3 `[8,16,128]`；K/V/KI cache 为 NVFP4 packed uint8 `[pages,1,128,64]`，每 16 个值对应一个 FP8 scale。`mega` backend 也接受按 page 跨步的 cache 视图（见[根目录 README](../README.zh-CN.md#paged-cache-布局)）。`layer_input` 为连续 BF16 `[8,6144]`；Projection/OProj weight 使用 128×4 swizzled MXFP8 scale。输出写入调用方提供的 BF16 `[8,6144]` buffer。

在 CUDA Graph capture 外创建 weight 和 TP4 context，再用 `runner.prepare(...)` 绑定每个 slot 的 cache、metadata 和预分配 buffer。输出须预分配，并在 capture 前预热；runner 不会在 `plan.run()` 中主动分配或 prepack。`AttentionWeights`、`AttentionCache` 和 `AttentionBuffers` 定义所需张量。[公开测试](../../../../tests/inference/high_tps/msa_v1/attn_backends/test_backends.py)展示完整分配和调用。

```python
from inference.high_tps.msa_v1.attention import create_attention_runner, create_attention_tp_context

tp_context = create_attention_tp_context(
    backend="mega", group_id="decode-attention"
)
runner = create_attention_runner(
    weights=attention_weights, tp_context=tp_context,
    backend="mega",
)
plan = runner.prepare(
    q_len=q_len, layer_input=layer_input, positions=positions,
    slot_mapping=slot_mapping, cos_sin_cache=cos_sin_cache,
    cache=attention_cache, buffers=attention_buffers, plan_info=plan_info,
)
attention_residual, post_attention_norm = plan.run()
# Close tp_context collectively after Graphs and outstanding GPU work finish.
```

所有张量必须位于调用前选定的本 rank CUDA device。四个 rank 必须集体创建兼容的 TP4 context。准备 metadata 和输出后，再预热并捕获 Graph。调用会修改 K/V/KI cache 和输出；算子不修改 weight。所有在途调用结束后，`mega` backend 允许原地更新直接读取的权重和 scale；TP 各 rank 必须一致完成更新。替换 storage 时必须重新绑定 plan 并捕获 Graph。Graph 生命周期内输入输出地址保持稳定。共享 workspace 的调用必须顺序执行。销毁 Graph 并等待 GPU 工作结束后，集体调用 `tp_context.close()`。

`mega` backend 在单 kernel 内使用低延迟 TP4 mailbox 通信。各 rank 的 OProj 输出先舍入为 BF16，按 rank 顺序用 FP32 累加，再与 residual 相加、舍入为 BF16，最后执行 FP32 RMSNorm。通信值中的负零会规范化为正零。`mega` 不提供另一套 NVLS 归约 backend；仍要求 CUDA multicast 和单节点 NVLink 互联的 TP4 组。Context 在 capture 外分配并初始化通信存储。任何 rank 尚有未完成工作时，不得重置或复用该存储；完整 Attention 调用之间必须使用普通完成依赖。

检查归约舍入契约、cache 写入隔离和多次 Graph 重放：

```bash
ALLOW_JIT=1 torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.attn_mega.test_mailbox --output /path/to/results
```

`AttentionConfig(wait_timeout_s=...)` 为跨 CTA 与跨 rank 等待设置诊断超时秒数；
默认 0 表示关闭。非零值选择独立诊断编译产物，超时会触发 CUDA trap，需终止 TP worker
并重建 CUDA context。`enable_pdl` 默认为 False；此单 kernel 路径不依赖外部 producer 的 PDL。

构建步骤见[根 README](../../../../README.zh-CN.md)。在四张 GPU 上用自动生成的测试输入检查 MegaAttention：

```bash
ALLOW_JIT=1 torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.attn_backends.test_backends --q 8 --kv-len 8192
```
