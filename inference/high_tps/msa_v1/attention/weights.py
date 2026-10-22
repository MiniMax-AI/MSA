"""Stable rank-local weight bindings for a MiniMax-M3.1 decode Attention layer."""

from dataclasses import dataclass, fields

import torch


@dataclass(frozen=True)
class AttentionWeights:
    """Prepared MXFP8 weights and Gemma-RMSNorm gamma tensors.

    The two GEMM scale tensors use the existing MegaAttention 128-by-4
    swizzled layout. Weight conversion belongs to model loading, not run().
    """

    projection: torch.Tensor
    projection_sf: torch.Tensor
    output_projection: torch.Tensor
    output_projection_sf: torch.Tensor
    input_norm: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    qi_norm: torch.Tensor
    ki_norm: torch.Tensor
    post_attention_norm: torch.Tensor

    def validate(self) -> torch.device:
        contracts = {
            "projection": ((2560, 6144), torch.float8_e4m3fn),
            "output_projection": ((6144, 2048), torch.float8_e4m3fn),
            "input_norm": ((6144,), torch.bfloat16),
            "q_norm": ((128,), torch.bfloat16),
            "k_norm": ((128,), torch.bfloat16),
            "qi_norm": ((128,), torch.bfloat16),
            "ki_norm": ((128,), torch.bfloat16),
            "post_attention_norm": ((6144,), torch.bfloat16),
        }
        scales = {
            "projection_sf": 2560 * (6144 // 32),
            "output_projection_sf": 6144 * (2048 // 32),
        }
        device = self.projection.device
        for field in fields(self):
            tensor = getattr(self, field.name)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{field.name} must be a tensor")
            if (
                not tensor.is_cuda
                or tensor.device != device
                or not tensor.is_contiguous()
                or tensor.requires_grad
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    f"{field.name} must be aligned, contiguous CUDA storage without gradients on {device}"
                )
            if field.name in contracts:
                shape, dtype = contracts[field.name]
                if tensor.shape != shape or tensor.dtype != dtype:
                    raise ValueError(
                        f"{field.name} must have shape {shape} and dtype {dtype}"
                    )
            elif (
                tensor.dtype != torch.uint8
                or tensor.ndim != 1
                or tensor.numel() != scales[field.name]
            ):
                raise ValueError(
                    f"{field.name} must contain {scales[field.name]} uint8 scale bytes"
                )
        return device
