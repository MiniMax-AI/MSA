# MSA v1 推理算子

[English](README.en.md)

## 概览

MSA v1 提供 paged sparse attention 和 indexer 的 decode/prefill 接口。所有公开算子
均采用 `plan()` / `run()` 生命周期：`plan()` 接收请求级 metadata，`run()` 接收当前层
的张量并返回结果。

## 公开接口

| 算子 | 包 | 公开 wrapper |
| --- | --- | --- |
| Decode attention | `inference.msa_v1.attention.decode.<dtype>` | `BatchDecodeWithPagedKVCacheWrapper` |
| Prefill attention | `inference.msa_v1.attention.prefill.<dtype>` | `BatchPrefillWithPagedKVCacheWrapper` |
| Decode indexer | `inference.msa_v1.indexer.decode.<variant>` | `BatchDecodeIndexerWithPagedKVCacheWrapper` |
| Prefill indexer | `inference.msa_v1.indexer.prefill.<variant>` | `BatchPrefillIndexerWithPagedKVCacheWrapper` |

可用数据格式与约束：

- Attention：BF16 prefill、Q8KV4 decode/prefill 和 Q8KV8 decode/prefill。
- Indexer：BF16 prefill、H=1/2/4 Q8KV4/Q8KV8 decode，以及 H=1/2/4 Q8KV8 prefill。
- Q8KV4/Q8KV8 decode indexer 支持 Q=1–16，输出 `[H,B*Q,16]`；
  `inference.msa_v1.indexer.decode.BatchDecodeIndexerPlan` 支持跨层共享及 Graph 内更新。
- BF16 paged prefill attention 支持 contiguous 和 SGLang-style strided K/V view。
- BF16 paged prefill indexer 支持 1 或 4 个本地 index head，输出为
  `[num_index_heads, total_q, 16]`。
- Q8KV4 与 Q8KV8 decode attention 在 B200/SM100、B300/SM103 上支持 GQA=8/16，
  按实际 `Hq/Hkv` dispatch，输出 BF16；query length 由公开接口指定。
  Q8KV4 使用原生 CUTLASS C++，并保留 CUDA 13.5 或更新版本上的 SM107 GQA=16 支持。
- Q8KV8 prefill attention 支持 GQA group size 1、2、4、8 或 16。

BF16/Q8KV8 prefill attention 还保留 SM107 Rubin 路径，由实际设备架构选择。

具体张量 shape、dtype 和调用示例见各算子目录中的 README。

## 通用数据契约

- 仅支持 paged KV cache；varlen 请求必须提供真实的长度 metadata。
- TopK logical page 可以离散、无序，不能假设连续或排序。
- TopK 有效项位于前缀，无效后缀为 `-1`；最后一个有效项必须是 local page。
- `plan()` 必须在 CUDA Graph capture 外调用。需要 graph capture 的路径应先 warmup，
  并在 capture 时使用预分配输出。
- Wrapper 不会为了绕过输入限制而隐式排序 TopK、复制或重排 K/V。

## 可选依赖与格式转换

通用 NVFP4 到 E4M3 转换由 `inference.dequant` 提供，包括 dense 转换和仅处理 TopK
选中页的 sparse K/V 转换。

Q8K8 decode attention 使用外部 FlashInfer backend。仓库不分发 FlashInfer 源码或
cubin；通过 `python -m pip install -e '.[flashinfer]'` 安装固定的
`flashinfer-python==0.6.17`。该 adapter 不会隐式执行 FP4 dequant。

## 验证

- Decode correctness：96-case smoke suite 和 256-case full suite。
- Prefill correctness：32-case smoke suite 和 512-case full suite。
- 正确性测试对公开输出和必要辅助输出做全量 reference 比较。
- Decode full benchmark 使用 28 个 RL rollout case；Prefill 使用固定 128 个生产 case。
- Benchmark 报告 CUDA Graph 中公开 `run()` 全路径的 E2E latency。

共享 case、运行命令和统计方法见仓库根目录 [README](../../README.md)、
[inference data manifests](../../datas/inference/README.md)。
