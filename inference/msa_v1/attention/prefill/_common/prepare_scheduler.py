"""Shared schedule data and sizing model for SM100 sparse attention."""

from dataclasses import dataclass
from typing import Optional

import torch

KL_VALID_ROWS_BITS = 8
KL_VALID_ROWS_MASK = (1 << KL_VALID_ROWS_BITS) - 1
KL_WRITER_RANK_SHIFT = KL_VALID_ROWS_BITS
KL_WRITER_RANK_BITS = 8
KL_WRITER_RANK_MASK = (1 << KL_WRITER_RANK_BITS) - 1


@dataclass
class SparseAttentionSchedule:
    """Prepared GPU worklist shared by sparse-attention forward and backward."""

    enabled: bool
    scheduler_metadata: Optional[torch.Tensor]
    work_count: Optional[torch.Tensor]
    qsplit_indices: Optional[torch.Tensor] = None
    split_counts: Optional[torch.Tensor] = None
    dkv_owner_counts: Optional[torch.Tensor] = None
    dkv_split_indices: Optional[torch.Tensor] = None
    dkv_split_count: Optional[torch.Tensor] = None
    physical_row_ptr: Optional[torch.Tensor] = None
    physical_q_indices: Optional[torch.Tensor] = None
    # Packs the valid-row count and deterministic dQI writer rank.
    physical_valid_rows: Optional[torch.Tensor] = None
    dki_owner_counts: Optional[torch.Tensor] = None
    dki_split_indices: Optional[torch.Tensor] = None
    dki_split_count: Optional[torch.Tensor] = None
    target_q_per_cta: int = 0

    @property
    def work_capacity(self) -> int:
        return (
            0
            if self.scheduler_metadata is None
            else int(self.scheduler_metadata.shape[0])
        )

    @property
    def dkv_split_capacity(self) -> int:
        return (
            0
            if self.dkv_split_indices is None
            else int(self.dkv_split_indices.shape[0])
        )

    @property
    def dki_split_capacity(self) -> int:
        return (
            0
            if self.dki_split_indices is None
            else int(self.dki_split_indices.shape[0])
        )


class SparseAttentionScheduleModel:
    """Host-side helpers for sparse attention schedule sizing."""

    @staticmethod
    def _round_up(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

    @staticmethod
    def _ceil_div(value: int, divisor: int) -> int:
        return (value + divisor - 1) // divisor

    def _target_q_per_cta(
        self,
        *,
        total_q: int,
        topk: int,
        head_kv: int,
        qhead_per_kv: int,
        device: torch.device,
        usable_sm_count: int = -1,
    ) -> int:
        num_sm = torch.cuda.get_device_properties(device).multi_processor_count
        if usable_sm_count > 0:
            num_sm = min(int(usable_sm_count), num_sm)
        # The forward/backward UMMA M tile has 128 packed query-head rows.
        q_tokens_per_group = 128 // qhead_per_kv
        total_refs_upper = total_q * topk * head_kv
        # At least one chunk per resident CTA is enough to expose full-device
        # parallelism without an empirical wave-count multiplier.
        desired_work_items = max(num_sm, 1)
        total_groups_upper = self._ceil_div(max(total_refs_upper, 1), q_tokens_per_group)
        target_groups_per_cta = max(
            1,
            self._ceil_div(total_groups_upper, desired_work_items),
        )
        return target_groups_per_cta * q_tokens_per_group

    def balanced_target_q_per_cta(
        self,
        *,
        total_q: int,
        topk: int,
        head_kv: int,
        qhead_per_kv: int,
        device: torch.device,
        usable_sm_count: int = -1,
    ) -> int:
        # Keep chunk boundaries aligned to one packed 128-row UMMA tile.
        q_tokens_per_group = 128 // qhead_per_kv
        occupancy_target = self._target_q_per_cta(
            total_q=total_q,
            topk=topk,
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            device=device,
            usable_sm_count=usable_sm_count,
        )
        target = max(occupancy_target, q_tokens_per_group)
        return self._round_up(target, q_tokens_per_group)

    def flat_schedule_capacity(
        self,
        *,
        total_rows: int,
        total_q: int,
        topk: int,
        head_kv: int,
        target_q_per_cta: int,
    ) -> int:
        row_upper = max(total_rows, 0) * max(head_kv, 1)
        refs_upper = max(total_q, 0) * max(topk, 1) * max(head_kv, 1)
        split_upper = self._ceil_div(max(refs_upper, 1), max(target_q_per_cta, 1))
        return max(1, row_upper + split_upper)

    def physical_kl_schedule_capacity(
        self,
        *,
        total_q: int,
        topk: int,
        head_kv: int,
        padded_kv_blocks: int,
        q_per_macro: int,
        macros_per_work: int,
    ) -> int:
        """Bound physical-KV work without depending on device-side counts."""

        refs_upper = max(total_q, 0) * max(topk, 1) * max(head_kv, 1)
        nonempty_head_rows = max(padded_kv_blocks, 0) * max(head_kv, 1)
        macro_upper = self._ceil_div(max(refs_upper, 1), max(q_per_macro, 1))
        macro_upper += nonempty_head_rows
        work_upper = self._ceil_div(macro_upper, max(macros_per_work, 1))
        return max(1, max(padded_kv_blocks, 0) + work_upper)


SPARSE_SCHEDULE_MODEL = SparseAttentionScheduleModel()


__all__ = [
    "SPARSE_SCHEDULE_MODEL",
    "SparseAttentionSchedule",
    "SparseAttentionScheduleModel",
]
