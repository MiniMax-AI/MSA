"""Logical, rank-local weights and offline layout conversion."""

import math
from dataclasses import dataclass, fields

import torch


def _interleave_gate_up(tensor):
    shape = tensor.shape
    return (
        tensor.reshape(*shape[:-2], 2, 12, 64, shape[-1])
        .transpose(-4, -3)
        .contiguous()
        .reshape(shape)
    )


def _concatenate_gate_up(tensor):
    shape = tensor.shape
    return (
        tensor.reshape(*shape[:-2], 12, 2, 64, shape[-1])
        .transpose(-4, -3)
        .contiguous()
        .reshape(shape)
    )


def _pack_scales(tensor):
    """Pack logical row-major scales into contiguous 128-by-4 tiles."""
    shape = tensor.shape
    return (
        tensor.reshape(*shape[:-2], shape[-2] // 128, 4, 32, shape[-1] // 4, 4)
        .transpose(-4, -2)
        .contiguous()
        .reshape(*shape[:-2], -1)
    )


def _unpack_scales(tensor, rows, cols):
    return (
        tensor.reshape(*tensor.shape[:-1], rows // 128, cols // 4, 32, 4, 4)
        .transpose(-4, -2)
        .contiguous()
        .reshape(*tensor.shape[:-1], rows, cols)
    )


@dataclass(frozen=True)
class MoeWeights:
    """Contiguous CUDA weights; FC1 concatenates gate then up rows.

    Scales are logical row-major arrays, not hardware-swizzled storage.
    FC2 alpha excludes the routing multiplier in MoeConfig.
    Do not write them while a runner or graph is executing. To update weights,
    rewrite the tensors in place, then call ``MoeRunner.refresh_weights()``.
    """

    router_weight: torch.Tensor
    router_correction_bias: torch.Tensor
    shared_fc1_weight: torch.Tensor
    shared_fc1_weight_sf: torch.Tensor
    shared_fc2_weight: torch.Tensor
    shared_fc2_weight_sf: torch.Tensor
    routed_fc1_weight: torch.Tensor
    routed_fc1_weight_sf: torch.Tensor
    routed_fc1_alpha: torch.Tensor
    routed_fc2_weight: torch.Tensor
    routed_fc2_weight_sf: torch.Tensor
    routed_fc2_alpha: torch.Tensor

    def validate(self):
        contracts = (
            ((128, 6144), torch.float32),
            ((128,), torch.float32),
            ((1536, 6144), torch.float8_e4m3fn),
            ((1536, 192), torch.uint8),
            ((6144, 768), torch.float8_e4m3fn),
            ((6144, 24), torch.uint8),
            ((128, 1536, 3072), torch.uint8),
            ((128, 1536, 384), torch.float8_e4m3fn),
            ((128,), torch.float32),
            ((128, 6144, 384), torch.uint8),
            ((128, 6144, 48), torch.float8_e4m3fn),
            ((128,), torch.float32),
        )
        device = self.router_weight.device
        for field, (shape, dtype) in zip(fields(self), contracts):
            tensor = getattr(self, field.name)
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or not tensor.is_cuda
                or tensor.device != device
                or not tensor.is_contiguous()
                or tensor.requires_grad
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    f"{field.name} must be aligned contiguous CUDA {dtype} {shape} on {device}, without gradients"
                )
        for tensor in (self.routed_fc1_alpha, self.routed_fc2_alpha):
            if not bool((torch.isfinite(tensor) & (tensor > 0)).all()):
                raise ValueError("routed alpha must be finite and positive")
        return device

    @classmethod
    @torch.no_grad()
    def from_mega(cls, *, routed_scaling_factor=2.0, **weights):
        """Convert existing low-level weight arguments outside graph capture.

        The low-level FC2 alpha includes the routing multiplier; this method
        removes it. The factor must match the runner's MoeConfig.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("weight conversion is not allowed during capture")
        if not math.isfinite(routed_scaling_factor) or routed_scaling_factor <= 0:
            raise ValueError("routed_scaling_factor must be finite and positive")
        result = dict(weights)
        for prefix, rows, cols in (
            ("shared_fc1", 1536, 192),
            ("shared_fc2", 6144, 24),
            ("routed_fc1", 1536, 384),
            ("routed_fc2", 6144, 48),
        ):
            scales = result[prefix + "_weight_sf"]
            flat = (
                scales.reshape(128, -1)
                if prefix.startswith("routed")
                else scales.reshape(-1)
            )
            result[prefix + "_weight_sf"] = _unpack_scales(flat, rows, cols)
            if prefix.endswith("fc1"):
                result[prefix + "_weight"] = _concatenate_gate_up(
                    result[prefix + "_weight"]
                )
                result[prefix + "_weight_sf"] = _concatenate_gate_up(
                    result[prefix + "_weight_sf"]
                )
        result["routed_fc2_alpha"] = result["routed_fc2_alpha"] / routed_scaling_factor
        return cls(**result)

    @torch.no_grad()
    def _mega_arguments(self, factor):
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        for prefix in ("shared_fc1", "shared_fc2", "routed_fc1", "routed_fc2"):
            if prefix.endswith("fc1"):
                result[prefix + "_weight"] = _interleave_gate_up(
                    result[prefix + "_weight"]
                )
                result[prefix + "_weight_sf"] = _interleave_gate_up(
                    result[prefix + "_weight_sf"]
                )
            result[prefix + "_weight_sf"] = _pack_scales(result[prefix + "_weight_sf"])
        result["routed_fc2_alpha"] = self.routed_fc2_alpha * factor
        return result
