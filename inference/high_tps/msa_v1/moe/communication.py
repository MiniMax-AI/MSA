"""Explicit ownership of single-node tensor-parallel communication state."""

import torch
import torch.distributed as dist

from ..common.communication import (
    create_tp_allreduce_workspace,
    destroy_tp_allreduce_workspace,
)
from .dispatch import select_backend


class MoeTpContext:
    """Collectively created resources shared by sequential plans.

    Call close on every rank after all uses and graphs have finished.
    Contexts must not be shared by concurrent execution streams.
    """

    def __init__(self, *, group, group_id, slot_id, backend):
        select_backend(backend, 8)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("context creation is not allowed during capture")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group_id must be a non-empty string")
        if type(slot_id) is not int or slot_id < 0:
            raise ValueError("slot_id must be a non-negative integer")
        if not dist.is_initialized() or dist.get_world_size(group) != 4:
            raise ValueError("an initialized single-node TP4 process group is required")
        self.group = group
        self.rank = dist.get_rank(group)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.closed = False
        self.mega_workspace = None
        self._token_parallel = None
        # Validate collective configuration before entering backend-specific setup.
        configurations = [None] * 4
        dist.all_gather_object(
            configurations, (group_id, slot_id, backend), group=group
        )
        if len(set(configurations)) != 1:
            raise ValueError("all TP ranks must select the same context configuration")
        self.mega_workspace = create_tp_allreduce_workspace(
            self.rank,
            4,
            group_id,
            slot_id,
            self.device,
        )

    def _token_parallel_workspace(self):
        # Created collectively by the first MegaMoE plan.
        if self._token_parallel is None:
            from .backends.moe_mega.token_parallel import Workspace

            self._token_parallel = Workspace(self)
        return self._token_parallel

    def close(self):
        if self.closed:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("context destruction is not allowed during capture")
        with torch.cuda.device(self.device):
            torch.cuda.synchronize()
            if self._token_parallel is not None:
                self._token_parallel.close()
                self._token_parallel = None
            if self.mega_workspace is not None:
                destroy_tp_allreduce_workspace(self.mega_workspace)
            self.closed = True


def create_moe_tp_context(*, group=None, group_id, slot_id=0, backend="mega"):
    """Create TP4 resources collectively on each rank's current CUDA device."""
    return MoeTpContext(
        group=group, group_id=group_id, slot_id=slot_id, backend=backend
    )
