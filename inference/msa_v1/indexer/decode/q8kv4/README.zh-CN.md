# 多 Head Q8KV4 Decode Indexer

[English](README.md)

## 功能

面向 SM100/SM103 的 paged decode indexer。Q 使用 E4M3，K 使用 packed E2M1 和
E4M3 scale，输出每个 query 选中的 logical page indices，并保证包含 local page。

## 公开接口

```python
from inference.msa_v1.indexer.decode.q8kv4 import (
    BatchDecodeIndexerWithPagedKVCacheWrapper,
)

wrapper = BatchDecodeIndexerWithPagedKVCacheWrapper()
wrapper.plan(page_table, seq_lens, num_index_heads=4, query_length=q.shape[1])
topk_indices = wrapper.run(q, packed_k_cache, k_scale=k_scale)
```

可通过 `workspace_size(batch_size)` 查询 workspace 大小并向 wrapper 传入外部 workspace。
查询时应将当前 CUDA device 设为目标 device；所需容量包含该设备的调度 metadata。
升级后请重新查询容量，不要复用旧版本的固定字节数。未使用显式共享 plan 时，长度或 page mapping
改变后重新调用 `plan()`；共享模式的更新方式见下文。
CUDA Graph 模式还需要在构造 wrapper 时传入地址稳定的 `page_table_buffer` 和
`seq_lens_buffer`，并向 `run()` 传入预分配输出。

### 共享 plan 与 Graph 更新

`query_length` 接受 1–16 的整数，默认 8，同一 batch 内 Q 相同。有效长度必须满足
`Q <= seq_lens[b] <= max_pages * 128`。

改变 Q 或 H 可能触发新的编译；请在 Graph capture 前预热需要的配置。

从 `inference.msa_v1.indexer.decode` 导入 `BatchDecodeIndexerPlan`，在 capture 外创建：

```python
from inference.msa_v1.indexer.decode import BatchDecodeIndexerPlan

shared_plan = BatchDecodeIndexerPlan(
    seq_lens, max_pages=page_table.shape[1], num_index_heads=4, query_length=q.shape[1]
)
wrapper.plan(
    page_table, seq_lens, num_index_heads=4, query_length=q.shape[1], shared_plan=shared_plan
)
shared_plan.update()
```

构造 plan、绑定 wrapper 和首次 `run()` 编译必须在 capture 外完成。
`shared_plan.update()` 可以捕获进 CUDA Graph，随后调用所有使用该 plan 的层的 `run()`。
更改同地址 `seq_lens` 后，replay 会刷新 plan；各层可以使用不同 page table，
但共享时必须具有相同的 B、Q、H、page capacity、长度源地址和 device。
共享 plan 的 wrapper 不应额外传入 `workspace_buffer`。
共享模式下，同地址 page table 可在消费者执行前原地更新，无需仅为映射内容变化重新绑定。
更换 metadata 地址或 shape 时，需要在 capture 外重新绑定；B、Q、H 或 page capacity 改变时，
创建匹配的新 plan，并重新 capture 使用旧绑定的 Graph。

首次消费前必须调用 `update()`。长度改变后，每一步更新一次；不能跨 step 复用旧结果。
跨 stream 使用时，由调用方用 event 保证 update 完成后再消费，且全部消费者完成后才能再次更新。
plan 与 wrapper 必须在相关 Graph 的生命周期内保持存活。Graph 内不允许重新绑定或分配。

## 数据契约

- `q`：`[B, Q, H, 128]`，E4M3。
- `packed_k_cache`：`[physical_pages, 128, 64]`，packed E2M1。
- `k_scale`：`[physical_pages, 128, 8]`，E4M3，线性非 swizzle 布局。
- `page_table`：`[B, max_pages]`，CUDA `torch.int32` logical-to-physical page mapping。
- `seq_lens`：`[B]`，CUDA `torch.int32`，包含当前 Q-token query chunk。
- 输出：`[H, B * Q, 16]`，`torch.int32` logical page indices。有效项位于前缀，最后
  一个有效项为 local page。

所有输入 tensor 必须在同一 CUDA 设备上且 contiguous。每个历史 page 的分数为该页 token 与对应 Q head 点积的最大值；各 head 独立选择历史候选，输出不足 16 项的位置填 `-1`。

`packed_k_cache` 的存储 dtype 为 `torch.uint8`。反量化先计算 E2M1 × scale，再饱和舍入到 E4M3 后参与点积，使用 FP32 accumulator。

## 运行约束

- 对 query `q_idx`，local page 为 `(seq_lens[b] - Q + q_idx) // 128`。
- 历史 page 的物理映射允许乱序和不连续。
- `plan()` 必须在 CUDA Graph capture 外调用。
- 支持 QMUL4 的工具链会自动使用该路径，否则自动使用 CUDA Toolkit 12.9 支持的
  精确 FP16 fallback；调用方不需要选择后端。

`num_index_heads` 只接受 1/2/4；默认 1。各 head 独立选择历史 page，共享单 head K。H=1 也必须传入四维 Q，并返回三维输出。输出 `out` 必须为相同设备上 contiguous int32，shape 为 `[H,B*Q,16]`。

## 验证与性能测试

```bash
MINIMAX_INFERENCE_TEST_SUITE=smoke python -m pytest tests/inference/msa_v1/indexer/decode/q8kv4
MINIMAX_INFERENCE_TEST_SUITE=full python -m pytest tests/inference/msa_v1/indexer/decode/q8kv4
python -m benchmarks.inference.msa_v1.indexer.decode.q8kv4.benchmark --suite full --num-index-heads 4 --verify --out result.json
```

### 低延迟 benchmark

`--suite low-latency` 独立覆盖 batch={1,2,4} × KV length={1000,4000,8000,32000}，
共 12 个 case，默认 Q=8，可通过 `--query-length` 指定 Q。batch=1 使用准确的标称长度；batch=2/4 使用真实变长
metadata，平均长度等于标称值。两者均走相同的公开接口。

该组独立于 28-case `full` 生产加权统计。`--baseline` 用于同 H 的结果对比，
要求设备、DSL 版本和计时协议一致。使用 disjoint-page cold-cache（复用距离≥2×L2）、
5 次 warmup、20 次 replay、每 Graph 120 calls；`--verify` 全量校验 scores 和 TopK。

```bash
python -m benchmarks.inference.msa_v1.indexer.decode.q8kv4.benchmark --suite low-latency --num-index-heads 1 --verify --out low_latency.json
python -m benchmarks.inference.msa_v1.indexer.decode.q8kv4.benchmark --suite low-latency --num-index-heads 1 --verify --baseline low_latency.json --out candidate.json
```

`--query-length` 可指定 Q=1–16；不同 Q 的结果分别报告。
共享 plan 的测试与分阶段计时：

```bash
python -m pytest tests/inference/msa_v1/indexer/decode/test_shared_plan.py
python -m benchmarks.inference.msa_v1.indexer.decode.plan --precision q8kv4 --num-index-heads 4 --query-length 8 --layers 1 --out plan.json
```

共享 plan benchmark 覆盖同一组 12 个 low-latency case，分别记录 host+device update、
Graph update、Graph run 和 Graph update+run。`--layers` 是显式指定的算子重复层数，
不代表某个模型配置；Graph 延迟对应这些层的一次整体调用，不除以层数。
独立 update 使用 metadata 热缓存计时；run 和 update+run 使用 disjoint K rotation，
K 的复用距离至少为 2×L2。两种缓存条件分别报告。
host+device 计时若三次尝试后仍超过 CV=3%，结果会标记为无效，命令返回失败；
独立有效的 Graph 计时仍保存在输出文件中。
