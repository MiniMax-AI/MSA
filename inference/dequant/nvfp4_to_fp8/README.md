# NVFP4 to E4M3 dequantization

[English](README.en.md)

## 功能

该模块提供通用 dense NVFP4 转 E4M3，以及仅转换 paged sparse decode 所选
`(physical_page, kv_head)` 的融合 K/V 路径。

## 公开接口

```python
from inference.dequant import (
    SparsePagedNvfp4ToFp8Wrapper,
    dequantize_nvfp4_to_fp8,
)
```

Dense 接口处理任意 `[..., 64]` packed E2M1 输入。Sparse wrapper 使用
`plan()` 接收 TopK 与 page-table metadata，使用 `run()` 转换每层 K/V 数据。

## 数据契约

- Head dim 固定为 128，scale group 固定为 16。
- NVFP4 数据与 E4M3 scale 均为线性、非 swizzle GMEM 布局。
- Sparse TopK 为有效前缀、`-1` 后缀，最后一个有效项必须为 local page。
- Sparse 接口只转换 TopK 选中的 `(physical_page, kv_head)`，并返回可供下游
  paged attention 使用的 compact K/V cache、block table 和 sequence lengths。

## 运行约束

仅支持 SM100/SM103。CUDA 13.4 及以上使用公开 QMUL4 指令；较早工具链使用
精确的 FP16 dequant fallback。`plan()` 必须在 CUDA Graph capture 外执行；capture
期间 `run()` 必须使用预分配输出。
