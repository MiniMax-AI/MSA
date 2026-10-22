# sglang-msa-mono

[English](README.md)

SGLang monokernel 插件，通过 `inference.high_tps.msa_v1` 调用 MegaAttention 和
token-parallel MoE，并通过 `inference.high_tps.msa_v1.draft` 调用 DSpark draft
backbone。注册 `--monokernel-backend msa`，面向 TP4、SM100/SM103 的
MiniMax-M3.1：`MiniMaxM3SparseForConditionalGeneration`（发布版
`MiniMaxAI/MiniMax-M3.1-Flash` 使用此架构）、`MiniMaxM3TinyForConditionalGeneration`、
`MiniMaxM3SparseForCausalLM`、`MiniMaxM3TinyForCausalLM`，以及其 DSpark draft
`DSparkMiniMaxDraftModel`。

两种架构均为编译目标。当前集成运行检查覆盖 SM103；SM100 实机运行和完整服务性能
尚未完成验证。


这是可选集成。固定的 fork 当前需要访问权限，其 commit 无法匿名获取；公开 SGLang
版本也未提供所需的 `MiniMaxM3Nvfp4KVPool` 接口。因此，仅安装插件不能获得公开可复现的
服务部署环境。独立 MSA 与 draft API 不依赖该 fork。

## 安装

```bash
pip install -e .                     # MSA kernels（minimax-msa）
pip install --no-deps -e integrations/sglang
```

SGLang 通过 `sglang.srt.plugins` entry point 加载此通用插件。

## 使用

```bash
# MiniMaxAI/MiniMax-M3.1-Flash 的 DSpark draft 位于 dspark/ 子目录
python -m sglang.launch_server --model-path MiniMax-M3.1-Flash --trust-remote-code \
  --tp-size 4 --page-size 128 \
  --speculative-algorithm DSPARK --speculative-draft-model-path MiniMax-M3.1-Flash/dspark \
  --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8 \
  --speculative-dspark-block-size 7 \
  --monokernel-backend msa --disable-attn-tp-gather
```

仅单请求、恰好 8 个 draft token 的 DSpark/EAGLE 链式 verify 调用 target kernel，
因为 token-parallel MoE 固定为 8 行。embedding 与第 0–2 个 dense 层沿用
SGLang 模块；随后每个 MoE 层调用 MegaAttention 和 token-parallel MoE 两个
kernel，各自使用独立 NVLS workspace 完成 TP4 all-reduce，最后沿用 SGLang 的
final norm 和 logits processor。prefill、普通 decode 和多请求 batch 走标准路径。
没有启用投机解码，或 draft token 数不匹配时，服务启动会列出原因并失败。

DSpark draft 的单请求 block 步骤（7 行：bonus token 加 6 个 mask token）调用 draft
backbone：vocab-parallel embedding、5 个 dense 层和 final norm 在每个 rank 上合为一次
cooperative launch，全部 TP4 求和都在 kernel 内完成。SGLang 的 draft head（lm_head、
Markov head、采样）以及把 target hidden states 注入 draft KV cache 的步骤保持不变。
draft graph 必须只装一个请求：使用 a2a MoE backend 时（M3.1 NVFP4 强制 `megamoe`），
SGLang 会把 draft 行数补齐到 attention TP 的倍数，因此需要传
`--disable-attn-tp-gather`；未传时 draft spec 会在启动时拒绝该服务。

## 契约

- 插件显式使用 `AttentionConfig(enable_placement=True)`，兼容非 canonical 硬件
  cluster 排布。独立 MegaAttention API 默认为 `False`；其 canonical 排布限制
  不作为插件默认设置。
- Checkpoint 为 MiniMax-M3.1 的 NVFP4 发布版（`quantization_config` 声明 `mxfp8`
  并带 `moe_quant_algo: NVFP4`，SGLang 按 `minimax_m3_nvfp4` 加载）：attention 和
  shared expert 使用 MXFP8，routed experts 使用 NVFP4，router 使用 FP32。同一
  expert 的 w1/w3 global scale 必须相同，所有 input global scale 必须为 1。所有
  MoE 层必须是 MSA 稀疏 attention 层；dense 层可以使用任一种 attention。
- KV cache 为 SGLang MSA-v1 NVFP4 pool，直接消费 page-strided view，见
  [Paged cache 布局](../../inference/high_tps/msa_v1/README.zh-CN.md#paged-cache-布局)。
  不支持 compact 或 K/V 共享，page size 必须为 128。
- Context 上限 1048576 tokens，即 8192 pages。Graph 的 plan 按 page table
  容量创建，每步将真实 KV 长度写入 `plan_info.kv_segment_lens`。
- 加载时直接将 routed experts 写入 `MoeWeights` 逻辑布局：gate 行在前、up 行
  在后，scale 按行排列；每个 expert 仅保存一份该布局。其他操作数在模型加载后
  从模型复制。服务拒绝在线权重更新，SGLang monokernel 框架尚无刷新 hook。
- SGLang 要求 monokernel API **1** 和 M3.1 NVFP4 cache 接口
  `MiniMaxM3Nvfp4KVPool`；本次集成固定
  [`4143ef163183ed759abc613b2bd6f872c8cf16f4`](https://github.com/ispobock/sglang-minimax-collab/commit/4143ef163183ed759abc613b2bd6f872c8cf16f4)
  作为验证版本。其他 monokernel API 会被拒绝。这不代表任意 SGLang 发行版兼容，
  也不代表已通过真实 checkpoint 的准确率验收。
- 拒绝 `--fp8-gemm-runner-backend flashinfer_trtllm`，因为它会原地重排 MXFP8 权重。
- Draft：SGLang 从 checkpoint 的 `dspark_*` 字段推导出的 draft 配置（5 个窗口 4096 的
  sliding attention 层、dense FFN 12288、Gemma norm、开启 `sample_from_anchor`），
  checkpoint 布局的 MXFP8 linear，SGLang 普通 BF16 draft KV pool 且 page 为 128 tokens
  （不支持融合或量化的 draft cache），以及不含额外词表的 target vocab-parallel embedding。
  `SGLANG_DSPARK_EMBED_IN_GRAPH` 必须保持开启：kernel 内部完成 token id 的 embedding。

## 测试

```bash
PYTHONPATH=.:integrations/sglang:<sglang>/python pytest integrations/sglang/tests
```

权重布局和启动契约测试在 CPU 上运行；每步 metadata 准备测试需要 CUDA。
这些测试不能证明真实模型质量、accept length、吞吐、延迟或服务长期稳定性。
SM100 与 SM103 的实机验收需要分别记录。

无需 checkpoint 的 TP4 binding 检查使用非零合成权重、真实 cache view 和两个生产
runner。它检查包含两个 kernel 的 Graph，并按 1K、128K、1M、32K、1K 切换 KV 长度：

```bash
PYTHONPATH=.:integrations/sglang ALLOW_JIT=1 torchrun --standalone --nproc-per-node=4 integrations/sglang/tests/test_minimax_m3_binding.py
```

该检查覆盖一个 MoE 层的 binding，不运行完整服务，也不测量端到端服务性能。
