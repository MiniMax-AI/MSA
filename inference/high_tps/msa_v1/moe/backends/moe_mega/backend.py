"""Single-kernel token-parallel execution for Q8."""

import torch

from .token_parallel import TokenParallelPlan


class MegaBackend:
    @torch.no_grad()
    def __init__(self, weights, config):
        self.weights = weights
        self.config = config
        # The kernel applies one effective FC2 scale per expert.
        self.fc2_alpha = weights.routed_fc2_alpha * config.routed_scaling_factor

    @torch.no_grad()
    def refresh(self):
        torch.mul(
            self.weights.routed_fc2_alpha,
            self.config.routed_scaling_factor,
            out=self.fc2_alpha,
        )

    def prepare(self, plan):
        workspace = plan.tp_context._token_parallel_workspace()
        plan.mega_workspace = plan.tp_context.mega_workspace
        plan.output = plan.mega_workspace.next_hidden
        plan._native = TokenParallelPlan(
            workspace, self.weights, self.fc2_alpha, plan.hidden, plan.residual, plan.output
        )

    def run(self, plan):
        plan._native.run()
        return plan.output
