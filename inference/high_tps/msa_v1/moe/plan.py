"""Bound input views and per-query execution resources."""

import torch


class MoePlan:
    """Sequential execution plan with stable input and output addresses.

    Warm up collectively before graph capture. Retain this plan, its runner,
    and its TP context until all graphs using it are destroyed. Output is
    overwritten by the next invocation; mega plans share output.
    """

    def __init__(self, runner, backend, q_len, hidden, residual):
        self.runner = runner
        self.backend = backend
        self.q_len = q_len
        self.hidden = hidden
        self.residual = residual
        self.tp_context = runner.tp_context
        self._implementation = runner._backends[backend]
        self._implementation.prepare(self)

    @torch.no_grad()
    def run(self):
        """Enqueue the complete MoE boundary on the current stream."""
        self.runner._require_valid_weights()
        if self.tp_context.closed:
            raise RuntimeError("the TP context is closed")
        if torch.cuda.current_device() != self.hidden.device.index:
            raise RuntimeError("select the plan's CUDA device before execution")
        return self._implementation.run(self)
