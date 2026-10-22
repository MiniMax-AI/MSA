"""Graph-stable complete-layer execution plan."""

import torch


class DecodeLayerPlan:
    def __init__(self, *, run_attention, run_moe):
        self._run_attention = run_attention
        self._run_moe = run_moe

    @torch.no_grad()
    def run(self):
        self._run_attention()
        return self._run_moe()
