"""Explicit ownership of rank-local TP4 Attention communication resources."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from ..common.communication import (
    Tp4AllReduceWorkspace,
    create_tp_allreduce_workspace,
    destroy_tp_allreduce_workspace,
)


@dataclass
class AttentionTpContext:
    """Collective TP4 storage kept alive through Graph replay."""

    workspace: Tp4AllReduceWorkspace | None = None
    owns_workspaces: bool = False
    closed: bool = False

    def validate(self, device) -> None:
        if self.closed:
            raise RuntimeError("the Attention TP context is closed")
        if self.workspace is not None:
            self.require_mega_workspace(device)

    def require_mega_workspace(self, device) -> None:
        if not isinstance(self.workspace, Tp4AllReduceWorkspace):
            raise TypeError("mega Attention requires a Tp4AllReduceWorkspace")
        if not self.workspace.ready or self.workspace.world_size != 4:
            raise ValueError("an open TP4 multicast workspace is required")
        if self.workspace.device != device.index:
            raise ValueError("the TP4 workspace must use the weights' CUDA device")

    def close(self) -> None:
        """Collectively release owned resources after Graphs and GPU work finish."""
        if self.closed:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("TP context destruction is not allowed during capture")
        if self.owns_workspaces:
            torch.cuda.synchronize()
            if self.workspace is not None:
                destroy_tp_allreduce_workspace(self.workspace)
        self.closed = True


def create_attention_tp_context(
    *, backend="mega", group=None, group_id, slot_id=0
):
    """Collectively allocate one backend's TP4 workspace outside capture."""
    if backend != "mega":
        raise ValueError(f"unknown Attention backend: {backend}")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("TP context creation is not allowed during capture")
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("group_id must be nonempty")
    if type(slot_id) is not int or slot_id < 0:
        raise ValueError("slot_id must be a nonnegative integer")
    if not dist.is_initialized() or dist.get_world_size(group) != 4:
        raise ValueError("an initialized single-node TP4 process group is required")
    rank = dist.get_rank(group)
    device = torch.cuda.current_device()
    configuration = (backend, group_id, slot_id)
    configurations = [None] * 4
    dist.all_gather_object(configurations, configuration, group=group)
    if any(value != configuration for value in configurations):
        raise ValueError("all TP ranks must select the same context configuration")
    workspace = create_tp_allreduce_workspace(rank, 4, group_id, slot_id, device)
    return AttentionTpContext(workspace=workspace, owns_workspaces=True)
