# Decode MoE 接口

[English](README.md) | 简体中文 | [项目安装](../../../../README.zh-CN.md)

`create_moe_runner` 封装完整的 MoE 边界：输入 post-attention norm 后的 hidden
和 attention residual，输出已经过一次 TP 求和及一次 residual add 的 BF16 hidden。
下游不能再次做相同的归约或 residual add。Attention、输入投影、final norm、LM head
和 sampling 均在接口之外。TP4 MegaMoE 每 rank 只启动一个 kernel。

## 范围与 backend 选择

首个集成支持单请求、q=8、hidden size 6144、单 NVLink 节点 TP4、128 个 routed
experts、top-4 路由、一个 shared expert、本地 intermediate size 768（全局 3072）。
不支持 EP、多请求、请求间变长以及其他模型维度。Router 权重及 logits 使用 FP32；
shared expert 使用 MXFP8；routed expert 使用 NVFP4；累加使用 FP32。固定激活函数
为 SwiGLU，alpha=1.702、limit=7、beta=1。

唯一 backend 为 `mega`，也是默认值，只接受 q=8。不支持的 shape 或 backend 会报错。

## 依赖

面向 SM100/SM103。MegaMoE 已在 SM103（GB300）上通过运行验证；SM100 仅有编译验证。
安装项目所需的 CUDA 版 PyTorch、支持 QMUL4 的 CUDA Toolkit、TVM-FFI 与 CUTLASS。
所选四卡必须位于同一 NVLink 节点并支持 CUDA NVLS multicast，不需要 FlashInfer。
编译和权重准备必须在 Graph capture 前完成。

## 权重契约

`MoeWeights` 保存当前 rank 的只读权重。所有 tensor 都必须是连续、16-byte 对齐、
位于同一 CUDA device、无需梯度的 tensor。FC1 在倒数第二维上按**全部 gate 行，
然后全部 up 行**排列。Scale 是逻辑行主序数组，不是预先 swizzle 后的存储。

| 字段 | Shape | Dtype / 语义 |
|---|---|---|
| `router_weight` | `[128,6144]` | FP32，原始 router 权重 |
| `router_correction_bias` | `[128]` | FP32，expert 选择 bias |
| `shared_fc1_weight` | `[1536,6144]` | FP8 E4M3 |
| `shared_fc1_weight_sf` | `[1536,192]` | uint8 E8M0，每 32 个值一个 scale |
| `shared_fc2_weight` | `[6144,768]` | FP8 E4M3 |
| `shared_fc2_weight_sf` | `[6144,24]` | uint8 E8M0 |
| `routed_fc1_weight` | `[128,1536,3072]` | uint8，每 byte 两个 packed E2M1 值 |
| `routed_fc1_weight_sf` | `[128,1536,384]` | FP8 E4M3，每 16 个值一个 scale |
| `routed_fc1_alpha` | `[128]` | 正有限 FP32 全局权重 scale |
| `routed_fc2_weight` | `[128,6144,384]` | uint8，packed E2M1 |
| `routed_fc2_weight_sf` | `[128,6144,48]` | FP8 E4M3 |
| `routed_fc2_alpha` | `[128]` | 正有限 FP32 全局权重 scale |

Packed nibble 沿用现有底层权重 ABI，第一个逻辑值放在低 4 bit。同一 expert 的 gate/up
全局 scale 必须相等。Routed activation 全局 scale 固定为 1。此处 FC2 alpha **不含**
路由倍率；`MoeConfig(routed_scaling_factor=2.0, enable_pdl=True)` 将倍率应用一次。
Correction bias 影响 expert 选择，不参与归一化后的路由权重。量化、BF16 中间存储、
并行归约顺序可能产生小幅数值差异，不承诺 bitwise 相同。

已有底层 MegaMoe 权重参数的调用者可使用：

```python
weights = MoeWeights.from_mega(
    routed_scaling_factor=2.0,
    **existing_weight_arguments,
)
```

仅传入上表十二个权重参数，使用现有底层布局。转换会还原 FC1 的 tile 行顺序和 scale
packing，并移除已经折入底层 FC2 alpha 的路由倍率。`MoeConfig` 必须使用相同倍率。
输入 tensor 不会被修改。创建 runner 时一次性准备各 backend 的权重存储，其所有 plan
共享这些权重。准备阶段可能发生同步，需计入准备后权重的常驻显存。

## 生命周期与集成

先初始化单节点四 rank 的 `torch.distributed` group，并为每个 rank 选择 CUDA device。
各 rank 必须以相同顺序创建和关闭 context，使用一致的 `group_id`、`slot_id`、
`backend`。ID 必须能唯一区分同时存活的 context。`group=None` 使用默认进程组。

```python
from inference.high_tps.msa_v1.moe import MoeConfig, create_moe_runner, create_moe_tp_context

context = create_moe_tp_context(group=tp_group, group_id="decoder-moe", slot_id=0)
runner = create_moe_runner(weights=weights, config=MoeConfig(),
                           tp_context=context, backend="mega")
plan = runner.prepare(q_len=8, hidden=attention_hidden,
                      residual=attention_residual)
attention_run()
output = plan.run()
```

`hidden`、`residual` 必须是连续、16-byte 对齐、无需梯度的 BF16 `[8,6144]`，并与
权重位于同一设备；两者只读。Attention producer 必须在当前 stream 的消费之前写完
这些数据。对预分配 `[8,6144]` buffer 取尾部切片可直接绑定，无 copy 或 padding kernel。
`output` 恰好包含 8 行有效数据；下游消费 `[8,6144]`。

Plan 拥有输出和可写 buffer。Mega 输出与 `plan.mega_workspace.next_hidden`
共享存储；输入不得覆盖该 workspace。之后任何使用该 workspace 的操作，包括 attention，
都可能覆盖其内容。复用前必须完成
输出消费或保存。接口没有任意调用方 `out` 参数，也不会隐式复制输出。

所有 rank 按相同顺序集体创建 Q=8 的绑定 plan。同一 context 的所有 Mega plan
共享其通信 workspace 和输出，该 workspace 同时通过 `context.mega_workspace` 暴露给
attention。所有 rank 使用有效输入执行 warmup，并在 capture 前同步。
在调用方 CUDA Graph 内 capture `attention_run(); plan.run()`；plan 内部不创建或 replay
嵌套 Graph。Q 固定为 8；改变 tensor 地址需要重新准备和 capture。当前 CUDA device 必须匹配 plan。所有 rank 必须执行一致的 q/backend/collective
序列。在支持的算子上请求 PDL，不表示每个算子都支持 PDL。

在线更新权重时，先在没有 plan 执行时原地改写 `MoeWeights` 的各个 tensor，再在 capture
之外调用 `runner.refresh_weights()`。它会原地重新生成各 backend 准备好的权重存储；plan
和已 capture 的 Graph 地址不变，不需要重新准备或重新 capture。
调用方必须等待所有在途工作结束，在 TP4 所有 rank 完成一致更新与刷新后，才恢复执行，
并保证后续调用等待刷新完成。shape、dtype、device、stride 和 storage 身份必须不变；
替换 storage 需要新建 runner 并重新捕获 Graph。刷新失败会使所有 plan 失效，直到成功
刷新；失败后也不得直接 replay 旧 Graph，因为 `graph.replay()` 绕过 Python 校验。
此算子接口不会启用 SGLang 服务端在线更新。

JIT 编译、离线权重 packing、plan/context 创建均在 capture 前完成。
准备好的 plan 使用固定 buffer 地址，每次调用一个 kernel。

共享 context 的 plan 必须顺序执行。并发执行需使用独立 context、plan 和可写 attention
buffer。Graph 和未完成 GPU 操作结束前，必须保持 runner、plan、输入和 context 存活。
之后先销毁 Graph，再由各 rank 集体调用 `context.close()`，最后销毁进程组。
Python 对象析构不会集体释放通信资源。

对于非法 q、shape、dtype、device、alignment、输出重叠或 context 状态，接口会在 host
metadata 可判断的范围内于执行前报错。缺少原生扩展或算子版本不兼容可能在准备或 warmup
时失败。某个 rank 失败时，应用必须中止其他 rank 的 collective 执行。

## 最小正确性检查

将以下代码保存为 `moe_smoke.py`，在可以导入 `inference.high_tps.msa_v1` 的环境中运行 `ALLOW_JIT=1 torchrun --standalone --nproc-per-node=4 moe_smoke.py`。它创建零权重，重复 replay Q=8 的 Graph，预期输出与 residual 完全相同。该检查覆盖 API 和 collective 生命周期，不代表非零模型的数值精度验证或性能测试。

```python
"""Standalone zero-weight oracle using only the public MoE interface."""

import os

import torch
import torch.distributed as dist

from inference.high_tps.msa_v1.moe import MoeWeights, create_moe_runner, create_moe_tp_context

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")


def full(shape, value, dtype):
    return torch.full(shape, value, dtype=dtype, device="cuda")


weights = MoeWeights(
    router_weight=full((128, 6144), 0, torch.float32),
    router_correction_bias=full((128,), 0, torch.float32),
    shared_fc1_weight=full((1536, 6144), 0, torch.float8_e4m3fn),
    shared_fc1_weight_sf=full((1536, 192), 127, torch.uint8),
    shared_fc2_weight=full((6144, 768), 0, torch.float8_e4m3fn),
    shared_fc2_weight_sf=full((6144, 24), 127, torch.uint8),
    routed_fc1_weight=full((128, 1536, 3072), 0, torch.uint8),
    routed_fc1_weight_sf=full((128, 1536, 384), 1, torch.float8_e4m3fn),
    routed_fc1_alpha=full((128,), 1, torch.float32),
    routed_fc2_weight=full((128, 6144, 384), 0, torch.uint8),
    routed_fc2_weight_sf=full((128, 6144, 48), 1, torch.float8_e4m3fn),
    routed_fc2_alpha=full((128,), 1, torch.float32),
)
context = create_moe_tp_context(group_id="moe-smoke")
runner = create_moe_runner(weights=weights, tp_context=context)
hidden = full((8, 6144), 0.5, torch.bfloat16)
residual = full((8, 6144), 1, torch.bfloat16)
plan = runner.prepare(q_len=8, hidden=hidden, residual=residual)
plan.run()
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    plan.run()
for _ in range(3):
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(plan.output, residual, rtol=0, atol=0)
del graph
context.close()
dist.destroy_process_group()
print(f"PASS rank={rank}", flush=True)
```

性能测量应采用项目规定的 CUDA Graph 与 cold-L2 协议：每张 Graph 包含 1000 次完整操作，warmup 两次，CUDA Event 计时五次，轮换地址独立的 weight/input/scale，并检查增加 slot 数后的稳定性。完整逻辑算子的全部工作都要计入；不能把未计时的 wrapper 调用与 Graph latency 比较。
