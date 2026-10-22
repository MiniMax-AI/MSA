# TP4 上的 M3.1 DSpark draft backbone

[English](README.md) | 简体中文

在单个 4 卡 NVLink 节点上，每个 rank 用一次 kernel launch 完成 MiniMax-M3.1 DSpark draft 模型的
backbone：一个请求的 7 行 draft block（第 0 行为 bonus token，第 1-6 行为 mask token，也可以是任意
token）的 vocab-parallel embedding、5 个 dense decoder 层和 final norm。它替换 SGLang 标准 draft
路径中逐层调用的算子序列。draft head（lm_head、采样）不在本次调用内。

每层的计算及其舍入点与 SGLang 一致：Gemma RMSNorm、MXFP8 QKV（每 rank 16 个 Q head、1 个 KV head，
head dim 128）、逐 head 的 Gemma q/k norm 与前 64 维的 NeoX RoPE、把 block 的 K/V 写入该层的 BF16
分页 cache、对最近 4096 个 key 的滑动窗口 attention（block 内每行可见整个 block）、MXFP8 o_proj、
TP4 求和 + 残差 + Gemma RMSNorm、MXFP8 gate_up（每 rank 3072 + 3072 行）、SwiGLU-OAI（alpha 1.702，
limit 7）、MXFP8 down、TP4 求和 + 残差 + 下一个 norm。激活按每 32 个值量化为 MXFP8（UE8M0 scale，
E4M3 RNE）；GEMM、norm、softmax 统计量和 TP4 求和都以 FP32 累加。每次 TP4 求和按 rank 顺序以 FP32
累加 4 个 rank 的 BF16 partial 并只舍入一次，因此所有 rank 输出逐位一致，重复调用也逐位一致。

整个 backbone（包括 embedding 与每层两次 TP4 求和）在每个 rank 的一次 cooperative launch 内完成；
各 rank 通过 context 的 IPC arena 交换 partial，不使用 NCCL 或单独的 all-reduce。

## 输入与生命周期

```python
from inference.high_tps.msa_v1.draft import (
    DraftBackboneInputs, DraftBackbonePlan, DraftBackboneWeights, DraftLayerWeights, create_draft_tp_context)

context = create_draft_tp_context(tp_group)          # 集合调用；每 rank 一个 32 MiB IPC arena
weights = DraftBackboneWeights(layers=[DraftLayerWeights(...) for _ in range(5)],
                               embed=embed_shard, vocab_start=rank * 50016, final_norm=final_norm, eps=1e-6)
inputs = DraftBackboneInputs(input_ids=..., positions=..., slot_mapping=..., page_table=..., kv_len=...,
                             cos_sin=..., k_caches=[...], v_caches=[...])
plan = DraftBackbonePlan(context, weights, inputs, out)   # out：BF16 [7, 6144]
plan.run()                                                # 可被 CUDA Graph 捕获；返回 out
```

| 操作数（每 rank） | dtype | shape |
|---|---|---|
| `qkv` / `qkv_sf` | E4M3 / UE8M0 | `[2304,6144]` / `[2304,192]`（q 16 个 head、k、v） |
| `o` / `o_sf` | E4M3 / UE8M0 | `[6144,2048]` / `[6144,64]` |
| `gate_up` / `gate_up_sf` | E4M3 / UE8M0 | `[6144,6144]` / `[6144,192]`（gate 3072、up 3072） |
| `down` / `down_sf` | E4M3 / UE8M0 | `[6144,3072]` / `[6144,96]` |
| norm 权重 | BF16 原始 Gemma `w`（按 `1 + w` 使用） | `[6144]`，q/k 为 `[128]` |
| `embed` | BF16 | `[50016,6144]`：词表第 `[50016 r, 50016 r + 50016)` 行 |
| `k_caches`、`v_caches` | BF16，每层一个 | `[slots,128]`，slot = page * 128 + row |
| `input_ids`、`positions`、`slot_mapping` | int64 | `[7]`（`slot_mapping` 为 -1 时跳过该行写入） |
| `page_table`、`kv_len` | int32 | `[max_pages]`、`[1]`（含本 block 的已缓存 token 数） |
| `cos_sin` | FP32 | `[positions,64]`（cos、sin） |

权重使用 checkpoint 的行主序布局（MXFP8 scale 为行主序的 `weight_scale_inv`，不是 swizzle 后的副本）。
所有操作数在创建 plan 时绑定；权重、cache 和 step 输入可以在两次调用之间原地修改（每次 graph replay
之前重写 step 输入）。支持从 7 到 page table 容量之间的任意 `kv_len`，page 不必连续。每次调用只写
各层 cache 中本 block 的 slot 以及输出。

在每个 rank 上以相同顺序、在 graph 捕获之外创建 context 和 plan，并在捕获 `plan.run()` 之前预热
每个 plan。同一 context 的 plan 按 stream 顺序执行；每个 rank 都必须发出每一次调用（若某个 rank
漏发一次调用，其余 rank 的 kernel 会在约 10 s 后 trap，而不是一直挂起）。先销毁 graph 并
等待未完成的 GPU 工作，再集合调用 `context.close()`。需要 SM100/SM103、同一节点上 4 个不同的设备
（支持 CUDA IPC peer access）以及 CUDA toolkit；首次使用时编译扩展（设置 `ALLOW_JIT=1`）。运行验证覆盖 SM103（B300）；SM100 仅做了编译验证。

## 验证与性能测试

```bash
ALLOW_JIT=1 torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.draft.test_reference
```

测试把每次调用与 FP64 参考比较。每个 GEMM 之前的 MXFP8 重量化会把任何合理的舍入差异（累加顺序、
BF16 attention 概率）变成 E4M3 码值的变化，并在 5 层之间累积，因此精度界是相对的：输出与每层写入的
K/V 与参考的距离不超过一个独立 FP32 实现与参考距离的 1.5 倍（且 cosine >= 0.99），第 0 层 K/V 与参考
相差不超过该行最大值的 1 个 BF16 ulp。测试还要求各 rank 逐位一致、不写 block slot 之外的 cache、CUDA Graph replay
与 eager 一致且输入变化后仍正确、重复调用逐位一致、创建 plan 后原地修改权重、交替执行两个 plan、
较大的 norm 权重，覆盖 kv 长度 7-65536 以及 mask 和 token 两种 block。

4x B300（SM103，默认频率）上的时延，单位为每次调用的 us，对比对象是用 SGLang 算子搭建的同样 5 层
（FlashInfer MXFP8 GEMM、其融合的 q/k norm + RoPE、TRT-LLM decode attention、融合的 SwiGLU-MXFP8、
融合的 TP4 all-reduce + 残差 + norm），使用相同的权重、cache 和输入：每个 graph 在每个 rank 上轮换
4 组权重共 16 次调用，每次调用后接一个非 PDL 的 16 MiB copy（单独计时后扣除），取最慢 rank，各实现
交错采样，取 12 个样本的四分位均值。SGLang 路径的第二次独立捕获与第一次相比为 0.98-1.02。

| kv 长度 | block | norm 权重量级 | SGLang 算子路径 | 本 kernel | 加速比 |
|---|---|---|---|---|---|
| 7 | mask | 0.1 | 343.8 | 249.3 | 1.38 |
| 1000 | token | 0.1 | 355.0 | 256.1 | 1.39 |
| 1024 | mask | 0.1 | 355.1 | 254.2 | 1.40 |
| 4096 | mask | 0.1 | 350.4 | 261.5 | 1.34 |
| 4103 | mask | 0.1 | 359.8 | 270.6 | 1.33 |
| 8192 | mask | 0.1 | 358.1 | 261.4 | 1.37 |
| 8192 | token | 0.1 | 357.4 | 266.2 | 1.34 |
| 8192 | token | 4 | 359.8 | 261.5 | 1.38 |
| 40000 | mask | 0.1 | 360.5 | 260.5 | 1.38 |
| 131072 | mask | 0.1 | 352.8 | 258.3 | 1.37 |

在 18 个测量点上（kv 7-131072、mask 和 token block、norm 权重量级 0.1/1.5/4、两组权重种子），加速比为
1.31-1.41，几何平均 1.36。与 FP64 参考相比，输出的 cosine 在每个点上与 SGLang 算子路径相差不超过 1e-4，
其中 16/18 个点更高（norm 权重较大时最多高 3.5e-3）。
