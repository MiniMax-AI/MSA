"""Unified single-request MiniMax-M3.1 decode Attention interface."""

import math
from dataclasses import dataclass

import torch

from .communication import AttentionTpContext
from .plan import AttentionBuffers, AttentionCache, AttentionPlan
from .weights import AttentionWeights


@dataclass(frozen=True)
class AttentionConfig:
    topk: int = 16
    sm_scale: float = 1.0 / math.sqrt(128.0)
    input_norm_eps: float = 1.0e-6
    head_norm_eps: float = 1.0e-6
    post_attention_norm_eps: float = 1.0e-6
    k_dequant_scale: float = 1.0
    v_dequant_scale: float = 1.0
    ki_dequant_scale: float = 1.0
    enable_pdl: bool = False
    # MegaKernel backends: seconds after which a wait on another CTA or TP rank traps; 0 = never.
    # Nonzero selects a separate diagnostic binary; the duration remains runtime.
    wait_timeout_s: float = 0.0
    # MegaAttention only: adapt logical roles to the actual hardware cluster placement.
    enable_placement: bool = False


class AttentionRunner:
    """Own immutable weights and a selected implementation for bound plans."""

    def __init__(self, *, weights, config, tp_context, backend):
        if backend != "mega":
            raise ValueError(f"unknown Attention backend: {backend}")
        if not isinstance(weights, AttentionWeights):
            raise TypeError("weights must be AttentionWeights")
        if not isinstance(config, AttentionConfig):
            raise TypeError("config must be AttentionConfig")
        if not isinstance(tp_context, AttentionTpContext):
            raise TypeError("tp_context must be AttentionTpContext")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("runner construction is not allowed during capture")
        if config.topk != 16 or any(
            not math.isfinite(value) or value <= 0
            for value in (
                config.sm_scale,
                config.input_norm_eps,
                config.head_norm_eps,
                config.post_attention_norm_eps,
                config.k_dequant_scale,
                config.v_dequant_scale,
                config.ki_dequant_scale,
            )
        ):
            raise ValueError(
                "Attention configuration requires topK=16 and finite positive scales/eps"
            )
        if not math.isfinite(config.wait_timeout_s) or not 0 <= config.wait_timeout_s < (2**63 / 1e9):
            raise ValueError("wait_timeout_s must be finite, non-negative and fit int64 nanoseconds")
        if type(config.enable_placement) is not bool:
            raise TypeError("enable_placement must be a bool")
        self.device = weights.validate()
        tp_context.validate(self.device)
        if self.device.index != torch.cuda.current_device():
            raise ValueError("select the weights' CUDA device before construction")
        if torch.cuda.get_device_capability(self.device) not in ((10, 0), (10, 3)):
            raise ValueError("Attention backends require SM100 or SM103")
        tp_context.require_mega_workspace(self.device)
        from .backends.mega import MegaBackend

        implementation = MegaBackend()

        self.weights = weights
        self.config = config
        self.tp_context = tp_context
        self.backend = backend
        self._backend = implementation

    def prepare(
        self,
        *,
        q_len: int,
        layer_input: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cache: AttentionCache,
        buffers: AttentionBuffers,
        plan_info,
    ) -> AttentionPlan:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("plan preparation is not allowed during capture")
        if type(q_len) is not int or not 1 <= q_len <= 8:
            raise ValueError("q_len must be an integer in [1, 8]")
        if not isinstance(cache, AttentionCache) or not isinstance(
            buffers, AttentionBuffers
        ):
            raise TypeError(
                "cache and buffers must use the public Attention dataclasses"
            )
        if layer_input.shape != (8, 6144) or layer_input.dtype != torch.bfloat16:
            raise ValueError("layer_input must be physical BF16 [8, 6144]")
        if positions.shape != (8,) or slot_mapping.shape != (8,):
            raise ValueError("positions and slot_mapping must be physical [8]")
        if q_len < 8 and torch.any(slot_mapping[: 8 - q_len] != -1).item():
            raise ValueError("leading padded rows must use slot_mapping=-1")
        tensors = (
            layer_input,
            positions,
            slot_mapping,
            cos_sin_cache,
            *(getattr(cache, name) for name in cache.__dataclass_fields__),
            *(
                getattr(buffers, name)
                for name in buffers.__dataclass_fields__
            ),
        )
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor) or tensor.device != self.device:
                raise ValueError(
                    "every bound tensor must reside on the weights' CUDA device"
                )
        if buffers.attention_residual.shape != (
            8,
            6144,
        ) or buffers.post_attention_norm.shape != (8, 6144):
            raise ValueError("Attention outputs must be physical [8, 6144] buffers")
        plan = AttentionPlan(
            self,
            q_len=q_len,
            layer_input=layer_input,
            positions=positions,
            slot_mapping=slot_mapping,
            cos_sin_cache=cos_sin_cache,
            cache=cache,
            buffers=buffers,
            plan_info=plan_info,
        )
        self._backend.prepare(plan)
        return plan


def create_attention_runner(*, weights, config=None, tp_context, backend="mega"):
    """Create a runner outside capture; default to the one-kernel backend."""
    return AttentionRunner(
        weights=weights,
        config=AttentionConfig() if config is None else config,
        tp_context=tp_context,
        backend=backend,
    )
