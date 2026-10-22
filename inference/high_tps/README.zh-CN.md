# High-TPS Decode 路径

[English](README.md)

本目录存放模型级 decode 路径，与可独立调用的通用算子包分开。
支持的硬件和 shape 见各模型文档。

| 包 | 模型与并行拓扑 | 入口 |
| --- | --- | --- |
| [`msa_v1`](msa_v1/README.zh-CN.md) | MiniMax-M3.1、TP4 | MegaAttention、MegaMoE、二者的两 kernel 串联与独立 draft backbone |

不保留原先平铺的 `inference.high_tps.<module>` 导入路径。
