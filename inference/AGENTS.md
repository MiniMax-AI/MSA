# Inference Interface Guidelines

[English companion](AGENTS.en.md). 本文件为权威版本；英文版必须随本文件同步更新。

本文件适用于 `inference/` 目录下的公开 Python wrapper、runtime integration 和 GPU kernel
接口。底层 kernel 实现仍需遵循仓库根目录 `AGENTS.md` 的算法、精度、编码和验证规范。

## FlashInfer 接口对齐原则

- Inference attention、indexer 和 KV-cache 相关的公开接口，应尽可能遵循 FlashInfer 的接口
  设计、参数语义和生命周期。
- 这里的“遵循”主要指公开 wrapper 的职责划分和调用方式。底层 CuTe DSL、CUDA 或 CUTLASS
  kernel ABI 不要求机械复制 FlashInfer。
- 新 backend 应优先接入已有 wrapper，不应仅因 dtype、量化格式或调度实现不同而重复创建
  语义相同的公开 API。
- 如果无法遵循 FlashInfer 的既有接口，应在接口文档中明确说明差异、原因以及未来向
  FlashInfer 适配时所需的转换。

## `plan()` / `run()` 生命周期

- 当请求结构或调度结果能够跨 transformer layer 或多次调用复用时，公开 wrapper 应采用
  `plan()` / `run()` 两阶段接口。
- `plan()` 接收请求级且可复用的 metadata，例如：
  - query/KV 变长信息；
  - page table；
  - sparse TopK indices；
  - mask 或 window 配置；
  - host 已知的 workspace capacity 和最大序列长度。
- `plan()` 负责 prepare、schedule、workspace sizing 和 backend 选择，并将其结果封装为
  wrapper 的内部 plan state。CSR、worklist、split/combine workspace 等内部结构不得要求用户
  在每次 `run()` 时重复传入。
- `run()` 只接收当前层或当前调用变化的数据，例如 Q/K/V、量化 scale、输出 tensor 和少量
  runtime scalar。
- `run()` 不得读取 device tensor 到 host 决定控制流、调度或 launch geometry，并应支持预分配
  output/workspace 和 CUDA Graph capture。
- `plan()` 如需一次性 host planning 或 D2H 同步，必须明确记录该行为，并且不得在 CUDA Graph
  capture 内调用。能够复用训练路径的纯 device prepare 时，优先保留异步实现。
- 没有可复用 planning 状态的简单 stateless kernel 不强制包装成两阶段接口。
- 正式性能范围仅包含 CUDA Graph capture 后公开 `run()` 的完整 device 路径。`plan()`、内存
  allocation、输入准备和 Graph capture 均不属于正式 E2E latency，也不得用私有 kernel 或
  stage latency 替代公开 `run()` 的 E2E 结果。

## 参数和 metadata 约定

- 参数语义与 FlashInfer 相同时，优先使用 FlashInfer 的 shape、dtype、layout 和命名约定。
- 因复用训练路径而保留仓库既有命名时，应在 wrapper 边界进行一次明确转换，不得同时提供
  多个名字表达同一语义。
- 同一项 metadata 只能有一个权威来源，并优先采用对应 FlashInfer 接口的命名。Decode 中
  独立的 `[B]` 实际序列长度统一命名为 `seq_lens`，不得在同一接口中同时提供 `seq_lens`、
  `kv_lengths` 或 `seqused_k`。Varlen prefill 的累积长度继续按其接口契约使用
  `cu_seqlens_q` / `cu_seqlens_k`。
- 不得通过扫描 TopK 后缀、page table 内容或物理 page 排列推导已有的显式长度或有效项数。
- Paged KV 必须通过明确的 logical-to-physical page mapping 访问。不得假设 physical page IDs
  有序、连续或与 logical page IDs 相同。
- Backend-specific schedule、CSR、barrier state 和临时 workspace 应保持 opaque，不得成为公共
  kernel API 的稳定 ABI。
- 不提供仅为兼容命名而存在的公开别名，例如同时导出 `prepare`/`plan` 或 `forward`/`run`。
  如需兼容旧接口，应使用独立、可删除并有弃用计划的 adapter。

## 输出和 workspace

- `run()` 应允许调用者传入 `out`；需要 LSE 或其他辅助输出时，也应允许调用者预分配对应
  tensor。
- 默认分配输出可以作为易用路径，但 CUDA Graph 路径不得依赖运行时动态分配。
- Split-KV、partial output、partial LSE 和 combine workspace 由 wrapper/plan 管理，不作为每层
  调用的业务参数。
- 多层复用时，plan state 和 workspace 的生命周期必须独立于单层 Q/K/V tensor。

## MSA v1 正确性与性能验收

- CUDA Graph E2E latency 是 `inference/msa_v1/` 下所有 op 的唯一正式性能指标。Graph 必须 capture
  公开生产 `run()` 中的全部 device kernel；主 kernel、私有 stage、TFLOPS、cycles、MBU 和
  MFU 只用于瓶颈诊断，不能替代 E2E 结论。
- 输入 tensor 和 metadata 必须在 Graph capture 前准备在 GPU 上。编译、内存 allocation、
  `plan()`、输入生成、H2D copy、Graph capture、correctness reference 和 warmup 均不计入
  E2E，也不要求汇报 `plan()` latency。
- 性能优化候选允许先运行正式 benchmark 做筛选。只有正式加权 E2E median 严格优于固定
  baseline 且每个有效测量的 `CV <= 3%` 时，才继续运行 benchmark case 的完整精度验证和
  目标 op 的 smoke correctness；无提升的候选直接拒绝，不要求运行精度验证。
- 新增 op 或 bug fix 必须先通过目标 op 的 smoke correctness；新增 op、kernel 修改或 bug fix
  在最终提交前必须通过目标 op 的 full correctness。
- Baseline 与 candidate 必须使用相同输入、Graph 配置、cold-cache 策略、warmup 和统计方式。
  一次优化目标中的所有 candidate 都与开发分支起点的固定 baseline commit 比较，同时记录
  相对上一个有效版本的增量变化。
- 正式 benchmark 必须独占目标 GPU。某个 case 的 `CV > 3%` 时只重跑该 case，最多重试 3 次；
  仍不稳定则本次结果无效，禁止挑选最好的一次。Baseline 的全部 case 先运行，随后运行
  candidate，不要求交错执行。

## MSA v1 共享 Prefill correctness case

- Prefill correctness 的唯一 source of truth 为
  `datas/inference/prefill_test_cases_v1.json`，通过 `datas.inference.cases` 加载。所有 prefill op
  必须直接复用该 manifest，不得维护 op 私有的通用 shape 列表。
- Full suite 固定为 512 个 case；smoke suite 固定为 full 的 32 个稳定子集。二者必须使用同一
  确定性选择方法、稳定 case ID 和 manifest 中记录的 seed。
- Full 与 smoke correctness 都必须对整个 output tensor 和所有必要辅助输出（例如 LSE）做
  全量 reference 比较，不得抽样 query、head、token 或 output tile。Smoke suite 只免除
  3 次 bitwise-identical 重复执行，不能降低输出覆盖范围。
- 修改共享 prefill manifest 后，必须运行所有受影响 prefill op 的 full correctness，不得只
  验证当前 op。全部 10,562 个生产 shape 作为 exhaustive audit，不属于每次提交的 full suite。

## MSA v1 共享 Prefill benchmark case

- Prefill benchmark 的唯一 source of truth 为
  `datas/inference/prefill_benchmark_cases_v1.json`，通过
  `benchmarks/inference/msa_v1/attention/prefill/cases.py` 加载。Q8KV4、Q8KV8 及其他 prefill
  op 必须直接复用，不得复制通用 case 定义。
- 正式 benchmark 固定为 128 个生产 case。所有 case 必须执行并逐项报告；正式总分使用 manifest
  中的生产频率权重计算 CUDA Graph E2E median 的加权算术平均。
- 任一 case 相对固定 baseline 的 E2E latency 回退不得超过 5%，每个有效测量均要求
  `CV <= 3%`。不设置统一的最低整体提升百分比；完整结果是否接受并合入由用户决定。
- Prefill cold-cache 使用 disjoint multi-tensor rotation：在 Graph capture 前分配地址互异的
  input、output 和必要辅助 tensor bank，每个 Graph slot 调用一次公开 `run()` 并写入独立
  output/LSE。计时 replay 按固定顺序轮换全部 slot，使同一地址再次使用前的真实工作集超过
  2 倍 L2。allocation 和 capture 不计入 E2E，不得加入 L2 flush kernel；总 Graph 时间除以
  slot 数得到 per-call latency。
- 性能候选通过初筛后，必须对 128 个 benchmark case 的整个 output tensor 和必要辅助输出执行
  一次独立 reference 校验，并运行 32 个 smoke correctness case；最终提交前仍须运行目标
  prefill op 的 512 个 full correctness case。

## MSA v1 共享 Decode correctness case

- Decode correctness 的唯一 source of truth 为
  `tests/inference/msa_v1/decode/cases.py`。所有 decode op 必须直接复用该 manifest，
  不得复制或维护 op 私有的通用 case 列表。
- Full suite 固定为 256 个 case；smoke suite 固定为 full 的 96 个稳定子集。二者
  使用相同的确定性生成器、稳定 case 名称和统一 `seed=1701`，不得手写 256 条
  静态列表。
- Full suite 由 75% 生产分布 case 和 25% 鲁棒性 case 组成。具体 batch、SeqLen、
  QLen、分布、边界和权重只在共享 manifest 中维护，`AGENTS.md` 不重复列举。
- 通用 decode attention 路径覆盖 `q_len_per_req=1/2/4/8/16`。接口静态固定 query length 的
  op 仍须复用共享 case 的 batch、SeqLen、seed 和 page/TopK metadata 分布，但按其公开契约
  使用固定 query length，不得复制一份私有 shape 列表。
- 鲁棒性 case 必须覆盖离散 physical pages、shared-prefix physical pages、随机
  page-table permutation、离散无序 TopK、local-block-last 和 local-page mask 边界。
- Full suite 中标记为确定性计算的 case 必须连续执行 3 次，整个输出均须 bitwise
  identical；smoke suite 不做重复执行检查。合法非确定性路径仍按固定数值阈值验收。
- 修改共享 decode manifest 后，必须运行所有受影响 decode op 的 full correctness，
  不得只验证当前 op。

## MSA v1 共享 Decode benchmark case

- Decode benchmark 的唯一 source of truth 为
  `benchmarks/inference/msa_v1/decode/cases.py`。所有 decode op 必须直接复用该
  manifest，不得复制或维护 op 私有的通用 benchmark case。
- Full benchmark 固定为 28 个 case，具体 batch、SeqLen、QLen、varlen 生成方式和
  RL rollout 联合权重只在 manifest 中维护。所有 case 必须执行并逐项报告。
- 正式总分为各 case CUDA Graph E2E median 的加权算术平均，不使用加权几何平均。
  同时报告每个 case 对总 latency 变化的加权贡献。所有 28 个 case 均有生产权重并受
  单 case 回退门禁约束。
- 任一 case 相对固定 baseline 的 E2E latency 回退不得超过 5%。不设置统一的最低
  整体提升阈值；完整结果是否接受并合入由用户决定。
- Baseline 与 candidate 必须使用相同输入、`seed=1701`、Graph 配置、cold-cache
  策略、warmup 和统计方式。
- 正式计时固定使用 5 次 Graph warmup 和 20 次 timed replay；每张 Graph 包含
  120 个 decode calls。小工作集需要多张 Graph bank 时，每次 replay 连续执行完整 bank，
  用总耗时除以总 call 数得到一个 per-call 样本；仍只产生 20 个样本，并以 median 作为该
  case 结果，要求 `CV <= 3%`。不以单次最好结果作为结论。
- 所有 decode benchmark case 均强制 cold-cache。使用 disjoint physical-page
  rotation，使轮换 working set 至少达到设备 L2 容量的 2 倍；不得把 L2 flush
  kernel 放入 E2E。小 working set 可以在 benchmark harness 中复制 physical-page
  bank，但不得改变单次调用的数据契约、工作量或读取字节数。
- 性能候选通过初筛后，必须对 28 个 benchmark case 的整个输出 tensor 做一次独立
  reference 校验，并运行 96 个 smoke correctness case；最终提交前仍须运行目标
  decode op 的 256 个 full correctness case。

## Q8KV4 Paged Prefill 当前契约

`inference/msa_v1/attention/prefill/q8kv4` 采用 FlashInfer 风格的 stateful wrapper：

```text
plan:
    topk_indices
    cu_seqlens_q
    cu_seqlens_k
    page_table
    -> q2k-to-k2q prepare + schedule + combine workspace

run:
    q
    packed k_cache / v_cache
    k_scale / v_scale
    out / lse
```

具体约束：

- 仅支持 paged KV。
- `cu_seqlens_q` 和 `cu_seqlens_k` 均为 `[B + 1]`、CUDA、contiguous、`torch.int32`。
- K 长度的唯一来源是 `cu_seqlens_k`；接口不再接收 `kv_lengths` 或 `seqused_k`。
- `page_table` 使用当前 MSA inference/training paged 路径的 dense
  `[B, max_num_pages_per_seq]` logical-to-physical mapping。未来接入 FlashInfer 时，在 wrapper
  adapter 中与 `paged_kv_indptr + paged_kv_indices` 转换。
- `topk_indices` 保存 logical KV block/page IDs；无效后缀使用 `-1`，prepare 路径负责忽略无效项。
- Chunk prefill 使用 bottom-right causal 对齐：

```text
q_len[b] = cu_seqlens_q[b + 1] - cu_seqlens_q[b]
kv_len[b] = cu_seqlens_k[b + 1] - cu_seqlens_k[b]
query_position = kv_len[b] - q_len[b] + q_idx
```

- `plan()` 内部直接复用训练路径的 q2k-to-k2q prepare、schedule 和 combine 契约；这些内部
  tensor 不暴露为 `run()` 的重复参数。
- 同一个 plan 应能够跨 transformer layer 复用；page table、TopK 或变长 metadata 变化时必须
  重新 plan。

## Q8KV4 Paged Decode Attention 当前契约

`inference/msa_v1/attention/decode/q8kv4` 仅公开
`BatchDecodeWithPagedKVCacheWrapper`：

```text
plan:
    topk_indices
    page_table
    seq_lens
    q_len_per_req / sm_scale
    -> decode schedule + split/combine workspace + default output

run:
    q
    paged_kv_cache = (K, V)
    kv_cache_sf = (K scale, V scale)
    out
```

- `seq_lens` 是 `[B]` CUDA `torch.int32`，是 decode KV 实际长度的唯一公开名称。
- `topk_indices` 是 `[B * q_len_per_req, 4, 16]` logical page ID；历史 page 可以乱序且不连续。
- `q_len_per_req` 是任意正整数的 runtime 参数，不参与 JIT specialization。对请求内 query
  `q_idx`，causal 位置为：

```text
query_position = seq_lens[b] - q_len_per_req + q_idx
local_block = query_position // 128
```

- `plan()` 必须在 CUDA Graph capture 外执行；`run()` 使用 plan 持有的 metadata、workspace 和
  默认输出，允许传入预分配 `out`。
- `sm_scale` 遵循 FlashInfer 命名；底层 backend 的 `fmha_fwd_*` 名称不属于公开 Python API。
- page table、TopK、长度或 query shape 变化时必须重新 `plan()`；同一 plan 可以跨层复用。

## 多 Head Decode Indexer 契约

`plan()` 的 keyword-only `num_index_heads=1` 仅接受 H=1/2/4；Q 为
`[B, Q, H, 128]` contiguous E4M3，各 head 独立计算并共享 K cache。
`query_length` 是 keyword-only 参数，支持 Q=1..16 的任意整数，同一 batch 内 Q 相同。
Q 可以为编译期量；若 Q/H 改变 MMA tile shape 或其他实际 codegen 配置，允许相应静态特化。
不改变 codegen 的维度保持 runtime；固定静态配置时，batch、KV 长度和 metadata 变化应复用
编译产物。不得为构造 compile key 读取设备 tensor 内容。

`inference/msa_v1/indexer/decode/q8kv4` 和 `q8kv8` 采用以下公开 wrapper 契约：

```text
plan:
    page_table
    seq_lens
    -> scheduler prepare + opaque workspace

run:
    q
    k_cache (Q8KV4 packed; Q8KV8 E4M3)
    k_scale (Q8KV4 only)
    out: [H, B * Q, 16] contiguous int32
    -> forced-tail TopK logical page indices
```

具体约束：

- `seq_lens` 是 `[B]`、CUDA、contiguous、`torch.int32`，表示包含当前 Q-token MTP query chunk
  的最终 KV 长度；它是 decode indexer 中 K 长度的唯一权威来源。
- 公开 wrapper 不再使用 `kv_lengths` 或 `seqused_k` 表达该语义。底层 kernel ABI 可以暂时
  保留 `kv_lengths_ptr` 等内部名称，但必须在 wrapper 边界完成单向转换，内部名称不得成为
  公开接口。
- 当前 MSA dense paged 表达继续使用 `page_table: [B, max_num_pages_per_seq]` 和 `seq_lens`。
- 如果未来切换到 FlashInfer 原生 paged 表达，应整体改用
  `paged_kv_indptr + paged_kv_indices + paged_kv_last_page_len`；不得在同一个公开接口中同时接收
  该组合与 `page_table + seq_lens`。
- Query 位置和 local block 继续按以下规则计算：

```text
query_position = seq_lens[b] - Q + q_idx
local_block = query_position // 128
```

- `page_table`、`seq_lens` 和由它们生成的 scheduler state 属于 `plan()`；`q`、当前层的 packed
  K cache、`k_scale` 和 `out` 属于 `run()`。
- 未使用显式共享 plan 时，`page_table` 或 `seq_lens` 内容变化后重新调用 `plan()`。
  使用 `BatchDecodeIndexerPlan` 时，同一 step 内只读 metadata 可跨 transformer layer 共享；
  同地址长度变化后调用 `update()`，同地址 page table 可在消费前原地更新。
  metadata 地址或 shape 改变时，在 capture 外重新绑定；B、Q、H 或 page capacity 改变时，
  创建匹配的新 plan，并重新 capture 使用旧绑定的 Graph。
  `update()` 在当前 stream 执行且允许 capture；构造和绑定必须在 capture 外。
  跨 stream 消费及再次更新需由调用方用 event 排序，不允许跨 step 复用旧 metadata。
- Q8KV4 与 Q8KV8 的公开入口均为 `BatchDecodeIndexerWithPagedKVCacheWrapper`；Q8KV8 的
  `run()` 不接收 `k_scale`。公开 wrapper 内部顺序执行 proxy score 与 forced-tail TopK，
  不公开 score-only wrapper、模块级 `forward()` 或 GEMM wrapper 名称。
- 代理分数仅是内部中间量；性能分析可以单独计量私有 proxy-score stage，并按实际瓶颈评估
  MBU 或 MFU，但它只用于诊断 kernel 上限，不能替代公开 `run()` 的 CUDA Graph E2E
  latency。Production 调用方只接收最终 TopK indices。

## 内部 Indexer TopK Select 契约

`inference/msa_v1/indexer/_common/topk_select` 是 Q8KV4、Q8KV8 和不同本地 index head 数共用的
私有 stateless per-row stage，不是 production 公开入口：

```text
_topk_select:
    proxy_scores:    [num_rows, max_cols] float32
    num_valid_pages: [num_rows] int32
    out:             [num_rows, 16] int32
```

具体约束：

- `proxy_scores` 仅要求最后一维连续；`num_valid_pages` 是 CUDA contiguous
  `torch.int32`。
- `num_valid_pages[row]` 表示该行包含 forced tail 的有效候选列数。stage 排序
  `[0, num_valid_pages[row] - 1)`，并把 `num_valid_pages[row] - 1` 放在最后一个有效 slot。
- 当 `num_valid_pages[row] <= 16` 时，输出 `0..num_valid_pages[row]-1`，后缀填 `-1`；否则输出
  15 个历史候选和 forced tail。
- `num_valid_pages` 是 producer 生成的逐行候选长度；decode indexer 的 `seq_lens` 是请求级 token
  长度。二者语义不同，不得作为同一接口中的多个权威来源。
- stage 不接收 page table、page size、query/MTP 布局、KV dtype 或 head 数，不做
  logical-to-physical page transform，也不返回 TopK values。
- `out=None` 是易用路径；CUDA Graph capture 中必须传入预分配 `out`。
- TopK 使用 16-bit 行内量化近似排序，正确性 gate 允许最多 1.5 个量化 step；
  forced tail、有效项数量、索引范围、唯一性和 `-1` 后缀必须严格正确。

Q8KV4 producer 的组合关系为：

```text
proxy_scores = proxy_score.run(...).view(H * batch * Q, max_pages)
num_valid_pages[h * batch * Q + b * Q + q_idx] = (seq_lens[b] - Q + q_idx) // 128 + 1
topk_indices = _topk_select(proxy_scores, num_valid_pages, out=topk_out)
```

`num_valid_pages` 应在 request/plan 阶段通过 device-side prepare 生成并跨 layer 复用，不得读取
device `seq_lens` 到 host，也不得在每层重复生成。

## 接口重构约束

- Decode 和 indexer 的接口重构应以本文件为准，优先统一 wrapper 生命周期、paged metadata
  语义、output/workspace ownership 和 CUDA Graph 行为。
- 重构不得在同一提交中静默改变数学语义、TopK 契约或 page mapping。接口迁移和 kernel 算法
  修改应拆分验证。
- 对既有调用方的兼容层必须是显式 adapter，不得让新旧 metadata 在核心 kernel 接口中长期
  并存。
