# High-TPS decode paths

[简体中文](README.zh-CN.md)

This directory contains model-level decode paths, separate from reusable
operator packages. See each model's guide for supported hardware and shapes.

| Package | Model and topology | Entry |
| --- | --- | --- |
| [`msa_v1`](msa_v1/README.md) | MiniMax-M3.1, TP4 | MegaAttention, MegaMoE, their two-kernel pipeline, and the independent draft backbone |

The former flat `inference.high_tps.<module>` import paths are not retained.
