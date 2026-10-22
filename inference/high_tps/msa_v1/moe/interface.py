"""Unified fixed-query MoE interface for a single TP4 decode request."""

import math
from dataclasses import dataclass, fields

import torch

from .dispatch import select_backend
from .plan import MoePlan
from .weights import MoeWeights


@dataclass(frozen=True)
class MoeConfig:
    """Fixed M3 dimensions, with explicit routing scale and PDL selection."""

    routed_scaling_factor: float = 2.0
    enable_pdl: bool = True


class MoeRunner:
    """Own stable prepared-weight storage, shared by fixed-query plans."""

    @torch.no_grad()
    def __init__(self, *, weights, config, tp_context, backend):
        select_backend(backend, 8)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("runner creation is not allowed during capture")
        if not isinstance(weights, MoeWeights) or not isinstance(config, MoeConfig):
            raise TypeError("weights and config must be MoeWeights and MoeConfig")
        if type(config.enable_pdl) is not bool:
            raise TypeError("enable_pdl must be a bool")
        if (
            not math.isfinite(config.routed_scaling_factor)
            or config.routed_scaling_factor <= 0
        ):
            raise ValueError("routed_scaling_factor must be finite and positive")
        self.device = weights.validate()
        if tp_context.closed or self.device != tp_context.device:
            raise ValueError("weights require an open TP context on the same device")
        if self.device.index != torch.cuda.current_device():
            raise ValueError("select the weights' CUDA device before preparation")
        if torch.cuda.get_device_capability(self.device) not in ((10, 0), (10, 3)):
            raise ValueError("MoE backends require SM100 or SM103")
        self.weights = weights
        self._weight_bindings = self._weight_storage_bindings()
        self._weights_valid = True
        self.config = config
        self.tp_context = tp_context
        self.backend = backend
        self._backends = {}
        if tp_context.mega_workspace is None:
            raise ValueError("the TP context has no mega resources")
        from .backends.moe_mega import MegaBackend

        self._backends["mega"] = MegaBackend(weights, config)

    def prepare(self, *, q_len, hidden, residual):
        """Collectively bind BF16 [q_len, 6144] views without copying data."""
        self._require_valid_weights()
        backend = select_backend(self.backend, q_len)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("plan preparation is not allowed during capture")
        if self.tp_context.closed:
            raise RuntimeError("the TP context is closed")
        if self.device.index != torch.cuda.current_device():
            raise ValueError("select the weights' CUDA device before preparation")
        for tensor in (hidden, residual):
            if (
                tensor.shape != (q_len, 6144)
                or tensor.dtype != torch.bfloat16
                or tensor.device != self.device
                or not tensor.is_contiguous()
                or tensor.requires_grad
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    "inputs must be aligned contiguous BF16 [q_len, 6144] on the weights' CUDA device, without gradients"
                )
        if backend == "mega":
            output = self.tp_context.mega_workspace.next_hidden
            start = output.data_ptr()
            end = start + output.numel() * output.element_size()
            for tensor in (hidden, residual):
                if (
                    tensor.data_ptr() < end
                    and start
                    < tensor.data_ptr() + tensor.numel() * tensor.element_size()
                ):
                    raise ValueError(
                        "inputs must not overlap the mega output workspace"
                    )
        return MoePlan(self, backend, q_len, hidden, residual)

    def _weight_storage_bindings(self):
        return tuple(
            (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)
            for field in fields(self.weights)
            for tensor in (getattr(self.weights, field.name),)
        )

    def _require_valid_weights(self):
        if not self._weights_valid:
            raise RuntimeError("weight refresh failed; refresh successfully before using any plan or Graph")

    @torch.no_grad()
    def refresh_weights(self):
        """Re-derive backend weight storage in place after the caller rewrote the
        ``MoeWeights`` tensors in place; plans and captured Graphs stay valid."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("weight refresh is not allowed during capture")
        if self.tp_context.closed or torch.cuda.current_device() != self.device.index:
            raise RuntimeError("weight refresh requires the runner's CUDA device and an open TP context")
        self._weights_valid = False
        if self.weights.validate() != self.device:
            raise ValueError("refreshed weights must stay on the runner's device")
        if self._weight_storage_bindings() != self._weight_bindings:
            raise ValueError("weight storage cannot be replaced; create a new runner and recapture Graphs")
        for implementation in self._backends.values():
            implementation.refresh()
        self._weights_valid = True


def create_moe_runner(*, weights, tp_context, config=None, backend="mega"):
    """Prepare TP4 MegaMoE weights offline for q_len=8."""
    return MoeRunner(
        weights=weights,
        config=MoeConfig() if config is None else config,
        tp_context=tp_context,
        backend=backend,
    )
