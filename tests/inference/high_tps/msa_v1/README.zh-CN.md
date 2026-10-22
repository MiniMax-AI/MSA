# 正确性检查

[English](README.md) | 简体中文

先按根 README 安装依赖并设置 PYTHONPATH。下列执行路径入口通过 torchrun 在单节点 TP4 上运行。

| 路径 | 模块入口 |
|---|---|
| `attn_mega` | `tests.inference.high_tps.msa_v1.attn_mega.test_attn_mega` |
| `moe_mega` | `tests.inference.high_tps.msa_v1.moe_mega.test_moe_mega` |
| `attn_mega_moe_mega` | `tests.inference.high_tps.msa_v1.integration.test_attn_mega_moe_mega` |

```bash
torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.integration.test_attn_mega_moe_mega --q 8 --output agents/tests/integration
```

使用 MoE `mega` backend 的入口（`moe_mega`、`attn_mega_moe_mega`）只接受 Q=8，其余入口接受 Q=1～8。Attention执行物理 8 行，Q 表示最后 Q 行有效输出。集成检查包含完整的两个 kernel 串联。

## 覆盖范围

默认零权重测试使用独立解析参考：MoE 输出应等于 residual；Attention 输出应为输入的 Gemma-RMSNorm。检查有限值、10 次完整操作的 Graph 节点数、重复 replay 和倒序 Q plan 复用。GPU 执行超过 10 秒视为失败，编译时间不计入。

`--nonzero` 使用非零权重并保存各 rank 的输出 `.pt`，用于同输入跨版本比较；有限值检查不等同于独立正确性证明。

`moe_mega/test_reference.py` 以按 msa_v1 量化与舍入规则构建的独立 FP32 参考检查 Q8 `mega` MoE，覆盖均衡与随机路由及多种激活专家数，并检查逐位可重复、改变输入后的 Graph replay、交替 plan、零 scale 和非法用法。

```bash
torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.moe_mega.test_reference
```

每个 rank 输出 JSON、tensor dump 和 Graph DOT；脚本失败会返回非零退出码。测试不输出正式 latency。


## 缓存与转换检查

缓存依赖测试不需要 GPU。QMUL4 转换检查需要一张受支持的 GPU，并将全部
65,536 种四 lane 编码在 127 组有限非负 scale 轮换下的结果与 FP32 参考比较。

```bash
python -m pytest -q tests/inference/high_tps/msa_v1/common/test_build_cache.py
python -m tests.inference.high_tps.msa_v1.common.check_qmul
```

## Runtime KV 与服务集成契约

Runtime-KV 测试使用独立 FP32 参考核对选页，按既有 FP8 存储容差比较完整输出，
并在最高 1M 的长短长度之间重放同一 Graph，检查没有重新编译。Stride 测试比较
连续布局和带 padding 的 slab view，并检查 cache 写回 canary。

```bash
torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.attn_backends.test_runtime_kv
torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.mega.test_strided_cache --q 1 2 3 4 5 6 7 8
torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.moe_mega.test_weight_refresh --backend mega
```

测试原地更新原始权重与 scale、刷新 runner，并将保留的 Graph 与新建 runner 对比。
SGLang 插件的启动和 metadata 测试见其[指南](../../../../integrations/sglang/README.zh-CN.md)。

独立 device 选择测试覆盖同分、量化边界碰撞、内部宽度，以及 512 线程 CTA 中的
named-barrier 参与范围。在源码树外生成可执行文件；B200 使用 `sm_100a`，B300 使用 `sm_103a`：

```bash
nvcc -O3 -std=c++17 -arch=sm_103a \
  -I inference/high_tps/msa_v1/csrc \
  tests/inference/high_tps/msa_v1/attn_backends/test_topk_device.cu \
  -o /tmp/msa_topk_test
/tmp/msa_topk_test
```
