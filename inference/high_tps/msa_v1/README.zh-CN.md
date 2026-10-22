# MiniMax-M3.1 TP4 Decode

[English](README.md)

本包提供单请求、物理 query 容量为 8、hidden size 6144、TP4、
sparse topK 16 与 paged NVFP4 K/V/KI cache 的模型级 decode 路径。
不支持 EP，也不支持一次层调用处理多个独立请求。buffer shape、权重布局和
各 backend 的依赖见 [Attention](attention/README.zh-CN.md) 与
[MoE](moe/README.zh-CN.md) 契约。
DSpark draft 模型的 5 层 backbone（7 行 draft block，每 rank 一次 launch）见
[draft](draft/README.zh-CN.md)。

公开 backend 仅包含 TP4 MegaAttention（Q=1–8）和 TP4 MegaMoE（Q=8），
各自每 rank 一次 kernel launch。二者组成的 Q=8 decode layer 每次调用两个 kernel。

`create_decode_layer_runner(structure="module-pipeline", attention=..., moe=...)`
接收已准备好的两个 runner；`plan.run()` 返回 BF16 `[8,6144]` 的 `next_hidden`，
该输出由 MoE 的 TP4 workspace 持有。独立 Attention 在 Q<8 时使用八行物理存储，
前 `8-Q` 行的 `slot_mapping` 设为 `-1`。所有 `positions` 必须位于 RoPE cache 范围内。

权重、TP4 通信、metadata、cache 和 Graph 稳定的 buffer 均须在 capture 外准备。
`runner.prepare(...)` 绑定一个 slot；`plan.run()` 只提交选定的 CUDA 工作，
并更新 paged cache。调用方须让权重、plan、buffer 和 TP4 context 存活到 Graph
销毁及 GPU 工作完成，再集体关闭 context。并发 stream 或 Graph 不得共用可写
workspace。

### Paged cache 布局

`AttentionCache`（以及 Mega kernel 的 `k`/`v`/`indexer_k` 参数）每个分量含 `P`
个物理 page。`k`、`v`、`ki` 的每页为 `[128,64]` packed E2M1 字节
（`[P,1,128,64]`；`ki` 也可为 `[P,128,64]`）；`k_scale`、`v_scale`、`ki_scale`
的每页为 `[128,8]` E4M3 字节（`[P,1,128,8]`；`ki_scale` 也可为 `[P,128,8]`）。
page 内各行紧密排列。`stride(0)` 为以字节计的 page stride，可取
`[紧密 page 字节数, 2^40)` 内任意 16 的倍数（紧密 page 字节数：数据 8192，scale
1024）；`k` 与 `v` 还要求 page stride 为 32 的倍数且起始地址 32 字节对齐（packed
FP4 TMA 的要求）。各分量 stride 可不同，因此 cache 可以是更大 per-page slab 的视图（例如一行
18432 字节同时存放 k、k_scale、v、v_scale）。紧密张量仍然有效。MegaAttention只写入 `slot_mapping` 选中的新行（slot = page·128 + row），
不会改动分量之间的字节。

### CUDA Graph 的容量 plan

MegaAttention 每次调用都从 `plan_info.kv_segment_lens` 读取实际 KV 长度。
以 page table 容量（最高 1048576 tokens）创建 plan，replay 前原地更新设备长度
与 page mapping；不读取超出可见长度的条目。所有合法 KV 长度与 page capacity
在固定 placement/诊断静态模式下复用一个编译产物。
MegaAttention 默认 `AttentionConfig(enable_placement=False)`，要求 canonical 硬件排布；
需要非 canonical 排布时须开启；SGLang 插件仅在 forward 期间没有其他 stream 运行 kernel 时
关闭（见其 README）。错误处理与 Graph 模式切换
见 [Attention 契约](attention/README.zh-CN.md)。关闭 wave-balanced 选型。
选页使用原始 FP32 分数精确选择 15 个历史页并保留 local page，同分按 page ID
决定，有效 ID 升序排列，短行后缀填 `-1`。

同一 workspace 上只含一个 MegaAttention 节点的 Graph 可以连续重放，并在长短
KV 间切换。共享 workspace 的调用必须串行。Q=1–8 沿用八行物理 padding，
padding 对应的 slot 必须为 `-1`。Attention 权重可在所有在途调用结束后原地更新；
替换 storage 必须重新绑定并捕获 Graph。

可选的 [SGLang 插件](../../../integrations/sglang/README.zh-CN.md) 接入单请求
八 token 链式验证，不支持服务端在线权重更新。

依赖支持 QMUL4 指令的 CUDA Toolkit、PyTorch、TVM-FFI、Jinja2 与仓库中的
CUTLASS submodule。TP4 MegaAttention 与 MegaMoE 不依赖 FlashInfer。
导入时不会立即编译所有 backend。首次预热 Mega backend 时设置 `ALLOW_JIT=1`；
之后可复用匹配的编译缓存。

[TP4 测试 fixture](../../../tests/inference/high_tps/msa_v1/common/m3_decode_fixture.py)
包含完整张量构造。安装依赖后，在四张 GPU 上运行两 kernel 集成冒烟测试：

```bash
ALLOW_JIT=1 torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.integration.test_attn_mega_moe_mega --q 8 --kv-len 8192 --output /tmp/msa-tp4-check
```
