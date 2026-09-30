# MiniMax Sparse Attention

[English companion](AGENTS.en.md). 本文件为权威版本；英文版必须随本文件同步更新。

## 路径与适用范围

- 本分支仅包含 MSA v1 的训练、推理及必要依赖，不导入其他模型实现或开发仓历史。
- 本文件中的仓库内路径均相对于仓库根目录；Markdown 链接相对于所在文档。
- 仓库外参考实现使用公开 Git 链接，固定到经过核验的 commit；不依赖个人目录布局。
- 始终使用简体中文回复；代码注释使用英文，设计文档使用简体中文。
  终端中的数学公式使用纯文本，不依赖 LaTeX 渲染。

## 协作前置条件

- 在实施前，必须明确需求、方案和验收目标。存在任何不清楚或歧义时，不得开始
  修改、测试或性能实验，必须先向用户确认。
- 开始 kernel 工作前，应明确目标架构、数据类型、shape、fixed/varlen 路径、
  correctness 阈值和性能指标；不得用未约定的替代 case 作为交付结果。
- 新增 op 或 kernel 重构开始前，必须由用户明确选择使用 CuTe DSL 或 CUTLASS C++；
  agent 不得自行切换实现技术路线。

## Git 与 worktree 工作流

- **Feature 开发**特指新增 op；新增 op 和性能优化必须基于用户指定的最新开发基线（本分支为 `nv_dev`）
  创建独立 worktree 和对应分支。新增 op 使用 `feature/<name>`，性能优化使用
  `perf/<name>`。
- 小改动不强制创建独立 worktree，例如规则、配置、文档和局部维护性修改；这类
  改动在用户指定的当前 worktree 中完成，并保留其中已有的用户改动。
- Bug fix 默认不强制创建独立 worktree，在用户指定的当前 worktree 中完成；只有
  用户明确要求隔离时才创建独立 worktree，并使用 `fix/<name>` 分支。
- 开发期间可以使用多个临时提交记录过程，但整个目标合入目标 worktree 前必须
  squash 为一个 commit。最终 commit 只能包含当前目标文件，不得夹带、覆盖或
  回退其他改动。
- 每次 push 前必须先将最终 diff 或 commit 交给用户 review，并取得用户明确的
  push 授权及目标远端分支；不得因代码已通过验证或此前已获 push 授权而自行推送
  后续修改。Review 说明和向用户的汇报使用简体中文。

## 工具链铁律

- 仓库内所有 Blackwell kernel（包括 inference 和 training）必须支持 SM100/B200
  与 SM103/B300。架构专用指令必须按实际设备选择，并保留另一架构的兼容实现；
  interface、编译目标、缓存、测试和公开文档的支持范围必须一致。仅有编译验证时，
  必须与对应 GPU 的运行验收明确区分。
- 本项目要求 `nvidia-cutlass-dsl[cu13]>=4.5.2`，代码必须兼容最低支持版本和最新稳定版。
- 默认安装和验证 cu13 backend；不同 CUDA backend 版本不得共用编译或 AOT 缓存。
- Kernel 兼容性修改必须分别在 4.5.2 和当前最新稳定版上验证；预发布版不作为
  默认验收版本。报告必须明确列出实际测试的版本，不能据此宣称未测试版本已通过。
- 开始 kernel 工作前必须核验当前进程实际加载的版本，并在测试或性能记录中
  留下版本信息。若低于 4.5.2，必须切换到受支持的隔离环境后再继续。
- 不得复用跨版本编译或 AOT 缓存；性能比较必须固定 DSL 版本，跨版本比较应单独标明。
- 不得擅自升级、降级或污染共享 Python 环境；优先使用项目已有的隔离环境。

## 开源与可复现约束

- 产品代码不得依赖 internal tuning knob、闭源 header、SASS patch 或不可公开复现的
  构建步骤。允许使用公开 CUDA 工具链正式支持的指令，例如 QMUL4。
- 可以参考公开实现的结构和算法，但必须适配本项目数据契约、重新验证，并保留必要
  的来源说明；不得复制无法确认许可或仅内部可访问的代码。

## 工程工具与参考

- CUDA、CuTe DSL 和 GPU 性能工作前，若当前环境提供相关 skill，先读取最相关的
  skill；工具仅作为工程参考，不覆盖本仓库规范，不要求安装私有工具或访问内部知识库。
- 使用标准 `CUDA_HOME`、`CUTLASS_ROOT`、`TORCH_EXTENSIONS_DIR` 和
  `TORCH_CUDA_ARCH_LIST` 配置，不硬编码开发者路径。
- 参考代码默认只读；先定位对应算子及实现路径，不通读无关仓库。
- 用户要求自行查看 NCU report 时，使用 `ncu --set full`，保存到可访问的稳定路径，
  提供压缩副本和对应实际主机及路径的 `scp` 命令。

## Harness 与算法约束

- **变长不可走定长 fast path**：varlen 场景禁止复用 fixed-length fast path；
  varlen correctness、benchmark 和 profile 必须使用真实变长代码路径，不得用
  定长输入近似，也不得在 interface 或 harness 中静默 bypass。
- **topK idxs 离散无序**：不得假设 topK indices 连续、单调或已排序，也不得为
  简化 kernel 在 harness 中排序或重排。
- **最后一个有效 block 为 local block**：每个 query 的有效 topK 列表中，
  最后一个 block 保证为 local block。可以利用该不变量只对 local block 做
  mask；其余选中 block 不做 mask。有效范围以对应长度 metadata 为准，padding
  不属于有效 topK。
- Harness 必须保持真实输入语义和 dispatch 路径。预处理、reference 或数据构造
  不得改变 topK 顺序、varlen 边界或 local-block-last 不变量。

## 编码规范

- interface、CuTe DSL kernel、warp specialization 和 AOT 写法遵循 FA4，并与整个
  repo 的既有风格保持一致。参考版本固定为 `145b1010051dbfd4bdc41a0ae55d495b08d7a458`；
  外部示例不替代本文件的明确约束。主要参考：
  - [interface.py](https://github.com/Dao-AILab/flash-attention/blob/145b1010051dbfd4bdc41a0ae55d495b08d7a458/flash_attn/cute/interface.py)
  - [flash_fwd_sm100.py](https://github.com/Dao-AILab/flash-attention/blob/145b1010051dbfd4bdc41a0ae55d495b08d7a458/flash_attn/cute/flash_fwd_sm100.py)
  - [flash_bwd_sm100.py](https://github.com/Dao-AILab/flash-attention/blob/145b1010051dbfd4bdc41a0ae55d495b08d7a458/flash_attn/cute/flash_bwd_sm100.py)
- **CuTe DSL 目录结构**：采用 FA4 风格的简洁算子目录。`interface.py` 位于算子
  根目录，负责公开接口、参数校验、架构 dispatch 和 compile/cache；device kernel
  与独立 kernel stage 也直接位于算子根目录，并使用描述性文件名。架构专用实现
  使用 `_sm90.py`、`_sm100.py`、`_sm120.py` 等后缀，独立阶段使用
  `_preprocess.py`、`_postprocess.py`、`_combine.py` 等后缀。不得为 CuTe DSL
  机械复制 CUTLASS C++ 的 `api/src/include/common/collective/device/kernel` 层次，
  也不创建额外的 `kernel/` 子目录。
- **CuTe DSL common**：可复用基础组件放入算子目录下的 `common/`，例如
  `copy_utils.py`、`mask.py`、`pipeline.py`、`softmax.py` 和
  `tile_scheduler.py`。只有至少被两个 kernel 文件复用的组件才能进入 `common/`；
  单个 kernel 私有 helper 必须保留在对应 kernel 文件中，不得为形式上的整齐提前
  抽象。
- **CUTLASS C++ 目录结构**：统一使用以下职责划分；不得为单个算子另建一套层次：

  ```text
  csrc/
  ├── api/          # PyBind 与公开 C++ host API
  ├── src/          # Plan、reduction 等编译单元
  ├── templates/    # JIT 实例化模板
  └── include/sm100/
      ├── common/       # Params、Traits 与基础 helper
      ├── collective/   # Load、dequant、MMA、softmax、correction 与 mainloop
      ├── device/       # Device adapter、plan 与 reduction
      └── kernel/       # Kernel entry、grid 与 launch policy
  ```

- **CUTLASS C++ 风格**：类型、类和 Traits 使用 PascalCase，函数和变量使用
  描述性 `snake_case`，编译期常量使用 `kPascalCase`。注释使用英文，代码使用
  2 空格缩进，并遵循 CUTLASS upstream 的既有格式。
- **CUTLASS C++ 自动格式化**：仓库根目录必须维护唯一的 `.clang-format`，所有
  修改过的手写 `.cpp`、`.cu`、`.hpp` 和 `.cuh` 文件在提交前必须通过统一的
  `clang-format` 检查。配置以 LLVM style 为基础，使用 2 空格缩进、4 空格续行
  缩进和 100 字符行宽。不得使用算子私有或开发者本地的另一套格式配置，也不得
  借格式化当前任务代码之机批量改写无关文件。生成代码应修改 source of truth，
  并检查生成后的 C++ 格式，不得直接格式化后手工覆盖生成产物。
- **张量前缀**：`m*`（GMEM）、`g*`（GMEM tile）、`s*`（SMEM）、
  `t*`（线程视图）、`acc_*`（累加器）。
- **CuTe DSL 命名规范**：变量、函数和 helper 使用描述性 `snake_case`，类使用
  PascalCase，架构专用实现使用清晰后缀（如 `Sm100`）。
- **Kernel 类结构**：`__init__`（host 配置）→ `@cute.jit __call__`（launch）
  → `@cute.kernel`（设备主体）。
- **CuTe DSL 格式**：注释使用英文，代码使用 4 空格缩进；类型标注使用 `cute.Tensor`、
  `Optional[cute.Tensor]`、`cutlass.Constexpr[...]`。
- 导入顺序：`cutlass` → `cutlass.cute as cute` → `cutlass.cute.nvgpu`
  → `cuda.bindings`。
- 优先使用 CuTe DSL/CUTLASS 已有抽象、tiled copy 和 `cute.copy`。TopK indices、
  LSE、`cu_seqlens` 等 metadata tensor 不强制使用 tiled copy。
- interface 必须显式校验 dtype、device、shape、stride、alignment 和 metadata
  约束；不得通过隐藏的输入转换掩盖 kernel 限制。

## Compile key 与 AOT 缓存

- 遵循 FA4 interface：compile key 只能包含确实影响 codegen 的稳定静态属性，
  例如 arch、dtype、head dim、GQA ratio、tile/stage/thread/cluster 配置、算法
  开关、可选 tensor 是否存在，以及 codegen 需要的 layout/broadcast pattern。
- 除下述 decode Q 静态特化例外外，**严禁包含运行时规模或取值**：`batch_size`/`bs`、`total_q`、`total_kv`、
  `seqlen_q`、`seqlen_kv`、`max_seqlen_q`、`max_seqlen_kv`、sequence 数量、
  具体 `cu_seqlens`、topK indices/lengths、tensor identity/pointer 或 stream。
  相同静态配置下改变 bs、KV 长度或 varlen 分布必须复用同一编译产物。
- **Decode Q 静态特化例外**：decode 的每请求 query length Q 允许为编译期量。
  当 Q 或本地 head 数 H 确实改变 MMA tile shape、layout、循环展开或资源配置时，
  可以将 Q/H 或其对应的有限 tile 配置加入 compile key。Q 应来自 host 已知参数或
  tensor shape metadata，不得为此读取设备 tensor 内容或引入 D2H 同步。
  该例外不允许把 batch、total_q、KV 长度、page capacity 或请求内容加入 key；
  不改变 codegen 的维度仍保持 runtime，不为 benchmark case 单独特化。
- varlen metadata 只允许以“是否存在”这类影响签名/codegen 的布尔属性进入 key，
  不得把 tensor 本身或 tensor 中的值加入 key。
- 不得让从 runtime tensor 派生的谓词泄漏进 compile key。尤其不能读取 tensor
  `max_seqlen` 构造 key，否则 tensor identity 或逐 step 数值变化会导致重复编译；
  也不得为构造 key 引入 device-to-host 同步。
- 若确实需要新增 specialization，必须证明它会生成不同代码，并将其表达为数量
  有限、稳定的 host-side 静态枚举；除上述 decode Q 外，不得直接使用 runtime size
  作为 specialization。
- compile-cache/AOT 测试在固定静态 Q/H 或 tile 配置时改变 bs、KV 长度、
  `cu_seqlens` 内容和 varlen 分布，验证编译复用。静态 Q/H 改变且生成不同代码时，
  允许生成对应产物；映射到相同静态配置的 runtime 尺寸应复用编译产物。

## 运行时状态与 Workspace 生命周期

- Decode indexer 的显式 `BatchDecodeIndexerPlan` 可在同一 step 的多个层之间共享只读 metadata；
  长度变化后必须更新。调用方通过 stream/event 排序所有消费者和后续更新，不得跨 step
  复用过时的 metadata。该例外不适用于依赖逐层 TopK 的 attention schedule。

- 进程级全局缓存只允许保存编译产物，以及不持有运行时 CUDA tensor、设备指针或
  workload 状态的不可变 host metadata。
- 严禁使用进程级全局变量、静态数组或按 runtime shape 建立的字典缓存 CUDA
  tensor，包括 workspace、中间输出、dequant buffer、counter、page metadata 和
  schedule。
- Runtime workspace 必须由对应的 wrapper、plan、request 或 CUDA Graph 实例独占，
  生命周期必须明确；对象销毁后不得因全局强引用继续占用显存。
- 同一 GPU 上的不同 plan、request、CUDA stream 或 CUDA Graph 不得共享可写
  workspace。若实现 buffer pool，必须具备容量上限、回收策略以及基于 stream/event
  的安全复用机制，并通过并发正确性测试。
- 依赖 indexer 输出、TopK、page mapping、sequence metadata 或其他运行时数据生成的
  schedule/plan，不得跨 layer、step 或 workload 缓存复用。仅允许同一次算子的
  forward 将其生成的 schedule 显式传递给对应 backward 使用。
- CUDA Graph 所需的稳定指针应通过 graph/wrapper-owned buffer 实现，不得通过
  全局共享 CUDA tensor 实现。
- 相关修改必须验证：连续运行不同 runtime shape 后显存不会随 shape 数量无界
  增长；wrapper/plan 销毁后其 workspace 可以被释放；同设备多 stream、多实例并发
  执行不存在 workspace alias 或结果污染。

## Code clean 验收规范

- Code clean 是提交门禁；未完成清理不得提交或交付。
- 删除 device `printf`、临时日志、debug 分支、注释掉的代码、未使用 import、
  变量/helper 和实验性常量；仅保留规定的 compile/run 耗时 logger。
- 不保留重复实现、不可达分支或无实际调用方的 fallback。必要的兼容分支必须
  注明适用条件，并有对应测试。
- 从参考实现移植代码时，只保留当前产品数据契约和实际调用路径需要的功能。
  没有生产调用方或正式测试覆盖的 dense、prefill、兼容 tuple、输出模式和历史
  参数必须在合入前删除；不得仅为保持与参考源码相似而保留无效兼容路径。
- 避免 magic number；影响算法或 codegen 的常量应具名，并放在最小合理作用域。
- helper 保持单一职责；优先复用 repo、CuTe DSL 或 CUTLASS 已有抽象，不复制
  相同逻辑。
- 控制流、warp role、pipeline/barrier ownership 和资源生命周期应清晰、对称，
  并符合 FA4 风格。注释解释“为什么”，不重复代码行为。
- 不得引入隐藏的 host-device 同步、临时 tensor allocation、数据复制、重复编译
  或输入重排。
- 不得通过 runtime shape 特化、测试专用生产分支或 harness bypass 掩盖实现问题。
- 修改保持最小作用域，不夹带无关重构或大规模格式化。
- 不得手工修改生成文件；应修改 source of truth 后重新生成。
- 提交前检查完整 diff，并运行 formatter/linter 和相关测试；涉及 compile cache
  或性能路径时，还必须运行缓存复用检查和 E2E benchmark。

## 正确性与计算精度

- 对语义上应确定的 kernel，相同输入、seed、配置和执行环境下多次运行必须得到
  bitwise-identical 结果。任何无法解释的非确定性都按未知 bug 处理，必须定位
  根因，不得直接放宽 tolerance。
- 对明确包含并行归约、atomic、online-softmax、NVLS/AllReduce 等非确定性
  累加顺序的路径，只要数学语义、数据类型和 FP32 累加精度不变，允许结果在
  约定阈值内波动，不要求 bitwise identical。必须能指出具体非确定性来源，
  不能在测试失败后笼统归因于浮点误差。
- **Tensor Core GEMM**：accumulator 必须为 FP32。
- **CUDA Core 计算**：乘加及 `sin`、`cos`、`tan`、`ex2` 等特殊函数必须在
  FP32 上执行；禁止在 CUDA Core 上用 FP16/BF16 做上述算术。
- **数据流精度**：GMEM/SMEM 可以存储 BF16/FP16 或更低精度量化格式；允许先
  转换为 FP32，再在 CUDA Core 上执行 FP32 计算。
- **Dequant 例外**：dequant 计算允许使用半精度。该例外仅限 dequant，不得扩展
  到 Tensor Core accumulator、softmax、归约或其他 CUDA Core 算术。
- **输出精度**：按接口约定在 FP32 计算完成后显式 downcast；不得在累加或归约
  阶段提前截断到低精度。
- correctness 必须同时检查输出和必要的中间/辅助结果（如 LSE）无 NaN/Inf，
  并与可信 reference 按约定阈值比较。
- 正式 correctness case 必须对整个输出 tensor 和必要辅助结果做全量比较，不得只
  抽样 query、head 或 token。Reference 必须独立实现并遵循输入格式的真实 dequant
  语义，不得调用被测 kernel 或复用其核心计算实现。
- 正式 case 必须使用非平凡随机输入和变化的 scale，不得只使用全零、全一或固定
  scale；同时保留零值、极值等独立边界 case。
- 共享 case manifest 只定义 shape、长度分布、seed、page/TopK 契约等通用 workload；
  dtype/op 特有的输入生成、reference 和数值阈值由各自测试文件定义。

## 验证流程

1. 新增 op 或 bug fix 必须先通过目标 op 的 smoke correctness；涉及性能路径时再
   运行 benchmark。
2. 新增 op、kernel 修改或 bug fix 在最终提交前必须运行目标 op 所属目录
   `AGENTS.md` 规定的 full correctness。开发验证过程中可以只运行 smoke suite。
3. 修改共享组件时必须运行所有受影响测试；无法确定影响范围时运行全量测试。
4. 新增功能和修复必须有测试覆盖；优先扩展已有测试函数，无必要不要新增测试函数。
5. sparse/varlen 测试至少覆盖：不规则 `cu_seqlens`、离散无序 topK、最后一个
   有效 block 为 local block、local block 边界/部分有效，以及最小值和上界 case。
6. 确定性路径必须增加重复执行检查；非确定性路径使用明确、固定的数值阈值。
7. 单个 test case 或 demo 的 kernel 执行超过 30 秒即认定为死锁，应终止进程并
   按 deadlock 流程定位；不得把 kernel hang 误判为编译慢。
8. 提交前按 Code clean 验收规范逐项检查，并确认无正确性或性能回退。
9. 开发分支可以按独立子任务创建临时 commit，message 应简洁描述该子任务；
   合入目标 worktree 前按 Git 与 worktree 工作流 squash 为一个 commit。
10. commit 前检查 README、接口文档和相关测试是否需要同步更新。

## 开发测试规范

- **编译与运行计时**：每次 `cute.compile(...)` 和测试/demo 的 kernel run 都必须
  通过 host logger 分别记录耗时。run 计时前后必须同步 GPU，避免只测到 launch
  时间。例如：

  ```python
  import logging
  import time

  import torch

  logger = logging.getLogger(__name__)

  t0 = time.time()
  compiled = cute.compile(demo, ...)
  logger.info("Compiled in %.3fs", time.time() - t0)

  torch.cuda.synchronize()
  t0 = time.time()
  compiled(...)  # If this times out, treat it as a deadlock.
  torch.cuda.synchronize()
  logger.info("Ran in %.3fms", (time.time() - t0) * 1e3)
  ```

- 使用非 PyTorch harness 时，应替换为对应 runtime 的 device synchronize。
- **超时 30s = 死锁**：30 秒阈值针对单个 kernel run，不包含编译；编译和运行
  必须分开记录。benchmark 的正式统计还必须排除编译和 warmup。

## 文件组织规范

- **统一 agent 工作目录**：agent 创建的临时测试、benchmark、profile、分析脚本、
  日志、JSON/CSV 汇总和 NCU/NSYS 报告均放在
  `agent/`，不得散落在仓库顶层或源码目录。
- **测试文件**：`agent/agent_tests`
- **Benchmark 文件**：`agent/agent_benchmark`
- **Profile 文件**：`agent/agent_space`
- **记忆总结目录**：`agent/memory`
- 上述临时产物和 agent 工作记录不得被 Git 跟踪或提交；`agent/` 由根目录 `.gitignore` 排除。

## 开发文档提交规范

- `design.md`、`optimization.md`、`results.md` 和性能调优过程文档仅保留本地，
  不得被 Git 跟踪或提交。
- 若上述文档已被跟踪，必须从 index 移除，同时保留本地文件并确保其被忽略。
- 面向使用者的 `README.md`、公开接口文档和正式产品文档必须随代码同步
  维护并提交。新增 op、修改公开接口、数据契约、依赖、支持范围、运行方式或用户可见
  行为时，同一 commit 必须更新最近的算子 README 和必要的上级 README。
- MSA 面向使用者的文档默认使用英文，`README.md` 为英文默认入口和权威版本。
  保留的其他语言版本必须与英文版互相链接，并在标题结构、表格、示例、命令、
  公开契约和限制上语义对齐；修改时同步维护。向用户提供的 review 说明使用简体中文。
- README 只面向算子使用者，只介绍功能、安装与依赖、公开 API、输入输出契约、支持范围、
  调用示例、用户可见错误与验收命令。不得记录 warp/CTA 分工、TMA/TMEM/SMEM 布局、
  pipeline/barrier、scheduler、私有 stage、内部 workspace 组织、SASS 或调优过程；这些内容
  应进入本地设计/优化文档、profile 报告或源码注释。
- README 不得暴露或建议用户直接调用私有 module、helper、stage 或未导出参数。公开示例必须
  与当前 package export 和 interface signature 一致，本地链接与命令必须在提交前检查可用。

## 性能优化记录规范

- 每一项优化都必须单独记录影响，不得只记录最终结果。
- 所有候选必须使用同一初始版本作为基线，同时记录相对初始基线的累计提升和
  相对上一有效版本的增量提升。
- 每项记录：优化原因、具体改动、benchmark case、测试环境、CuTe DSL 版本、
  优化前后正式性能指标，以及提升或下降百分比。正式指标由目标目录最近的
  `AGENTS.md` 定义；TFLOPS、cycles 和 main-kernel 数据只作为诊断信息。
- 未采用、性能回退或正确性失败的候选也必须记录结果和拒绝原因。
- 正式 benchmark 必须独占目标 GPU；检测到其他计算进程时不得开始或继续采集。
  不锁定 GPU 频率，也不要求记录 SM clock、温度或功耗。
- 多项优化不得捆绑后只报告整体提升；必须同时修改时，应补充逐项 ablation，
  确保收益可独立归因。
- benchmark 必须固定输入、warmup、重复次数和统计方法；优先报告稳定统计量，
  不以单次最好结果作为结论。

## 性能分析规范

- Nsight Compute 的 `gpu__cycles_elapsed.avg`、Tensor Core SOL / TC active 以及
  kernel TFLOPS 用于单 kernel 归因和诊断，不替代目标目录定义的正式性能结论。
- Sparse dense-equivalent causal case 的 FLOPs 必须按实际 causal attention
  计算量统计。
- 重点检查：
  - 主数据路径不应出现 scalar load/store，以及可避免的 `STS`、`LDS`、`STG`、
    `LDG.16`；优先使用 tiled copy 和 `cute.copy`。
  - 不应出现寄存器 spill 或 `LDL`/`STL` local-memory 读写。
  - 不应出现非合并访存或大量共享内存 bank conflict。
  - pipeline 应合理重叠 load、Tensor Core、softmax 和 `atomicAdd`，并量化等待与
    尾部效应。
- 参考实现只用于提取局部思路；任何改动都必须在本项目真实 varlen/topK 路径上
  重新验证正确性和性能，不得直接外推参考项目结论。
