# MSA v1 Tree/Func Indexer 使用说明

## 支持范围

Indexer 是 batch=1 的 SM100/SM103 训练 forward，固定约束如下：

```text
1 <= q_len <= kv_len <= 4,194,240
Q                     = BF16 contiguous [q_len, 4, 128]
K                     = BF16 contiguous [kv_len, 1, 128]
K block size          = 128
TopK                  = 16
score accumulator     = FP32
score workspace       = FP32（默认）或 FP16（use_fp16_score=True）
TopK score input      = 与 score workspace 相同
block sum / LSE math  = FP32
topk_indices          = int32 [4, q_len, 16]
selected_lse          = FP32 [4, q_len], natural log
execution             = K1 + K2 + optional K3; K2 supports deterministic mode
```

接口不接收 `cu_seqlens_q/cu_seqlens_k`，所有可见性均由 Func Tensor 描述。
当 `q_len < kv_len` 时，Q 对应 K 的连续 suffix：

```text
q_to_k_offset = kv_len - q_len
k_position(q) = q_to_k_offset + q
```

该映射只用于识别 TopK 最后一个有效位置必须保留的 local block；mask 语义仍完全
来自 Func。

## Func Tensor ABI

输入 `arbitrary_func` 必须满足：

```text
dtype       = torch.int32
device      = CUDA
layout      = contiguous
shape       = [1, 1, n_func, func_q_len]
func_q_len >= q_len + 256
n_func      = positive odd integer
```

令 `Fj(q) = arbitrary_func[0, 0, j, q]`，可见性为：

```text
visible(q) = [0, F0(q))
           U [F1(q), F2(q))
           U [F3(q), F4(q))
           U ...
```

所有 endpoint 必须位于 `[0, kv_len]`；每个区间满足 `begin <= end`；非空区间
按 begin 升序排列。未使用区间推荐编码为 `[kv_len, kv_len)`。

普通 bottom-right causal Func 的构造方式为：

```python
func = torch.full(
    (1, 1, 1, q_len + 256),
    kv_len,
    dtype=torch.int32,
    device=q.device,
)
func[0, 0, 0, :q_len] = (
    torch.arange(q_len, dtype=torch.int32, device=q.device)
    + kv_len - q_len + 1
)
```

## Tree/DFS 编码

Tree 节点应先按 DFS 顺序映射到互不重叠的连续 token 区间。对属于当前节点的 Q，
Func 依次编码：

1. root 到 parent 路径上每个祖先的完整 token 区间；
2. 当前节点的 `[node_begin, q + 1)` causal 区间；
3. 其余 endpoint 用空区间填充。

若最大路径深度包含 root 且为 `max_depth`，无需合并区间的直接配置是：

```text
n_func = 2 * max_depth - 1
```

生成器必须验证 parent chain 无环、存在唯一 root、节点区间不重叠，并确保 K suffix
中的每个 Q token 恰好属于一个节点。

## 编译与执行

```python
from msa_v1 import indexer_tree

plan = indexer_tree.compile_plan(arbitrary_func, q_len, kv_len)
topk_indices, selected_lse = indexer_tree.forward(q, k, plan)
deterministic_indices, deterministic_lse = indexer_tree.forward(
    q,
    k,
    plan,
    deterministic=True,
)
fp16_score_indices, fp16_score_lse = indexer_tree.forward(
    q,
    k,
    plan,
    use_fp16_score=True,
)
```

`compile_plan()` 是离线 exact-allocation API：它先在 GPU 上分类并生成大小 header，
随后读取 4 个标量到 host，再按精确大小分配最终 plan tensor。因此必须在训练 step 和
CUDA Graph capture 之前调用，并由调用方缓存返回值；不得在每个训练 step 中重复编译。
`forward()` 不包含该 D2H，同一长度和 Func 拓扑可以安全复用同一个 plan。

默认 `deterministic=False` 保留原有高性能路径，非 local TopK 的输出槽位顺序不作为
bitwise 契约。`deterministic=True` 时，非 local block 按 score 降序排列，score 相同则按
block id 升序排列；local block 仍位于最后一个有效槽位。相同输入、配置和设备环境下，
`topk_indices` 与 `selected_lse` 的重复执行结果必须 bitwise-identical。

当 `n_func == 1` 时，Func 的数学语义必然是单前缀 `[0, F0(q))`。Plan compiler 会使用
该等价关系直接根据一个 Q tile 内 endpoint 的最小值和最大值计算 FULL/PARTIAL block，
不构造 dense block scratch；其他正奇数 `n_func` 仍走通用 interval-union 路径，两条路径
生成完全相同的公开 plan ABI。

执行流为：

```text
K1: QK block score + block LSE
K2: 按 Func plan 选 TopK，并强制 local causal block 位于最后一个有效槽位
K3: 仅 FP32 gather 路径使用；FP16 gather 在 K2 内聚合 selected LSE
```

需要控制分配时可由调用方复用 workspace：

```python
score_workspace = torch.empty(
    (plan.num_plan_tiles, 2, 128), dtype=torch.float16, device=q.device
)
block_sum_workspace = torch.empty(
    (plan.num_plan_tiles, 2, 128), dtype=torch.float32, device=q.device
)
topk_indices = torch.empty((4, q_len, 16), dtype=torch.int32, device=q.device)
selected_lse = torch.empty((4, q_len), dtype=torch.float32, device=q.device)

topk_indices, selected_lse = indexer_tree.forward(
    q,
    k,
    plan,
    score_workspace=score_workspace,
    block_sum_workspace=block_sum_workspace,
    topk_indices=topk_indices,
    selected_lse=selected_lse,
    use_fp16_score=True,
)
```

`use_fp16_score=False` 为默认值，此时 score workspace 与 TopK 均保持 FP32。
设为 `True` 时，K1 在 FP32 累加和 block-max 计算完成后显式将 score 写为 FP16，
K2 直接根据 FP16 位模式构造有序 key 并执行 TopK；除 score 存储边界外不引入 FP16
CUDA Core 算术。FP16 non-gather 只为最终 Top15 加载 FP32 block sum；FP16 gather
在 K2 内完成 FP32 selected LSE，因此不再启动 K3。

`topk_indices` 可以直接传给 `msa_v1.attention.prepare()`。当 flat K 中的 block id
需要转换为 fragment-local id 时，向 `forward()` 传入 int32 `[q_len]` 的
`block_bases`；K2/K3 会在最终写回时原地完成重映射，有效 id 会减去对应 base，
`-1` 保持不变，不会为重映射额外分配 tensor。是否传入 `block_bases` 是静态编译
specialization，同一 specialization 下 base 的具体值不会进入 compile key。
Indexer 只负责 batch=1 的 score/TopK；packed-varlen attention metadata 仍由 attention 的
`cu_seqlens_q/cu_seqlens_k` 构建。
