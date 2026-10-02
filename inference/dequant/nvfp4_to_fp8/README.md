# NVFP4 to E4M3 Dequantization

[简体中文](README.zh-CN.md)

## Purpose

This package provides general dense NVFP4-to-E4M3 conversion and a fused K/V path that
converts only the `(physical_page, kv_head)` pairs selected by paged sparse decode.

## Public API

```python
from inference.dequant import (
    SparsePagedNvfp4ToFp8Wrapper,
    dequantize_nvfp4_to_fp8,
)
```

The dense function accepts packed E2M1 inputs with shape `[..., 64]`. The sparse wrapper
accepts TopK and page-table metadata through `plan()` and converts per-layer K/V data through
`run()`.

## Data contract

- The head dimension is fixed at 128, and the scale-group size is fixed at 16.
- NVFP4 data and E4M3 scales use linear, non-swizzled GMEM layouts.
- Sparse TopK entries form a valid prefix followed by `-1`; the final valid entry must be the
  local page.
- The sparse API converts only selected `(physical_page, kv_head)` pairs and returns compact
  K/V caches, block tables, and sequence lengths suitable for downstream paged attention.

## Runtime requirements

Only SM100 and SM103 are supported. CUDA 13.4 or newer uses the public QMUL4 instruction;
older supported toolkits use the exact FP16 dequantization fallback. Call `plan()` outside
CUDA Graph capture and provide preallocated outputs to `run()` during capture.
