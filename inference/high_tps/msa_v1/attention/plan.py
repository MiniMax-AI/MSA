"""Graph-stable bound inputs and outputs for one decode Attention call."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionCache:
    """Paged NVFP4 K/V/KI data and scales plus logical page metadata.

    ``k``, ``v``: uint8 ``[P, 1, 128, 64]`` packed E2M1 (low nibble first).
    ``ki``: uint8 ``[P, 1, 128, 64]`` (or ``[P, 128, 64]``).
    ``k_scale``, ``v_scale``: E4M3/uint8 ``[P, 1, 128, 8]``; ``ki_scale`` the same
    (or ``[P, 128, 8]``). One scale byte covers 16 values.

    Rows are dense inside a page (64 B data rows, 8 B scale rows). ``stride(0)``
    is the page stride in bytes: at least the dense page size (8192 B data,
    1024 B scales), below 2^40, and a multiple of 16 B (32 B, with a 32-byte-
    aligned base, for ``k``/``v`` data). Each component may use
    its own page stride, so the six tensors may be views into one or two larger
    per-page slabs. Dense tensors are the special case stride == dense size. The
    page-strided form is accepted by MegaAttention.

    ``kv_indices``: int32 ``[1, S]`` logical→physical page for attention;
    ``indexer_page_table``: int32 contiguous, ``numel >= S``, for KI.
    """

    k: torch.Tensor
    v: torch.Tensor
    ki: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    ki_scale: torch.Tensor
    kv_indices: torch.Tensor
    indexer_page_table: torch.Tensor


@dataclass(frozen=True)
class AttentionBuffers:
    """Preallocated intermediate and output storage owned by the caller."""

    q: torch.Tensor
    qi: torch.Tensor
    projection_data: torch.Tensor
    projection_scale: torch.Tensor
    projection_grid_epoch: torch.Tensor
    projection_special_partial: torch.Tensor
    projection_special_epochs: torch.Tensor
    kv_block_indexes: torch.Tensor
    output: torch.Tensor
    output_scale: torch.Tensor
    attention_residual: torch.Tensor
    post_attention_norm: torch.Tensor


class AttentionPlan:
    """Bind one slot's storage; run() performs no allocation or prepack."""

    def __init__(
        self,
        runner,
        *,
        q_len: int,
        layer_input: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cache: AttentionCache,
        buffers: AttentionBuffers,
        plan_info,
    ) -> None:
        self.runner = runner
        self.q_len = q_len
        self.layer_input = layer_input
        self.positions = positions
        self.slot_mapping = slot_mapping
        self.cos_sin_cache = cos_sin_cache
        self.cache = cache
        self.buffers = buffers
        self.plan_info = plan_info
        self.backend_state = None

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.runner._backend.run(self)
        return (
            self.buffers.attention_residual[-self.q_len :],
            self.buffers.post_attention_norm[-self.q_len :],
        )
