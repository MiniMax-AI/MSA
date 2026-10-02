# Training workloads

[English](README.md)

训练测试和 benchmark 的默认 workload 是 `real/` 中的 192K/CP16 manifest：每个
global pack 为 196608 tokens，CP size 为 16，每个 rank 分配 12 个 1024-token chunks。

`generate_cases.py` 从本地原始 CSV 生成去敏文件。1500 条调用记录归一化为
1253 个唯一 shape，selection 中的重复引用和权重保留真实频率。生成结果只包含
`case_id`、`cu_seqlens`、shape metrics 和测试/benchmark selection，不保留任何数据集或
source sample 标识。原始 CSV 由仓库 `.gitignore` 排除。

```bash
python3 datas/training/generate_cases.py
python3 datas/training/generate_cases.py --check
```

旧的 128K/CP16、256K/CP32 和 512K/CP64 人工 workload 位于 `synthetic/`，可用于额外
覆盖，不包含在默认测试和 benchmark selection 中。
