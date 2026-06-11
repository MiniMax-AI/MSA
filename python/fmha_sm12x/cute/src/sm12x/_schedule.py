# SPDX-License-Identifier: MIT

"""Host-side schedule metadata helpers for SM12x sparse attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SparseAttentionSchedule:
    enabled: bool
    scheduler_metadata: torch.Tensor | None
    work_count: torch.Tensor | None
    qsplit_indices: torch.Tensor | None = None
    split_counts: torch.Tensor | None = None
    target_q_per_cta: int = 0

    @property
    def work_capacity(self) -> int:
        return 0 if self.scheduler_metadata is None else int(self.scheduler_metadata.shape[0])


SparseSchedulePlan = SparseAttentionSchedule


class SparseAttentionScheduleModel:
    """Host-side helpers for sparse attention schedule sizing."""

    @staticmethod
    def _round_up(x: int, y: int) -> int:
        return ((x + y - 1) // y) * y

    @staticmethod
    def _ceil_div(x: int, y: int) -> int:
        return (x + y - 1) // y

    def _target_q_per_cta(
        self,
        *,
        total_q: int,
        topk: int,
        head_kv: int,
        qhead_per_kv: int,
        device: torch.device,
        usable_SM_count: int = -1,
    ) -> int:
        num_sm = torch.cuda.get_device_properties(device).multi_processor_count
        if usable_SM_count > 0:
            num_sm = min(int(usable_SM_count), num_sm)
        q_tokens_per_group = 128 // qhead_per_kv
        total_refs_upper = total_q * topk * head_kv
        desired_work_items = max(num_sm * 2, 1)
        total_groups_upper = self._ceil_div(max(total_refs_upper, 1), q_tokens_per_group)
        target_groups_per_cta = min(
            512,
            max(1, self._ceil_div(total_groups_upper, desired_work_items)),
        )
        return target_groups_per_cta * q_tokens_per_group

    def balanced_target_q_per_cta(
        self,
        *,
        total_q: int,
        topk: int,
        blk_kv: int,
        head_kv: int,
        qhead_per_kv: int,
        device: torch.device,
        usable_SM_count: int = -1,
    ) -> int:
        q_tokens_per_group = 128 // qhead_per_kv
        occupancy_target = self._target_q_per_cta(
            total_q=total_q,
            topk=topk,
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            device=device,
            usable_SM_count=usable_SM_count,
        )
        sink_balance_cap = max(q_tokens_per_group, int(topk) * int(blk_kv) * 2)
        target = min(max(occupancy_target, q_tokens_per_group), sink_balance_cap)
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


SPARSE_SCHEDULE_MODEL = SparseAttentionScheduleModel()
