"""Prepared complete operations shared by correctness and performance tools."""

import torch

from inference.high_tps.msa_v1.common.communication import (
    create_tp_allreduce_workspace,
    destroy_tp_allreduce_workspace,
)
from inference.high_tps.msa_v1.moe import (
    MoeConfig, MoeWeights, create_moe_runner, create_moe_tp_context,
)
from .m3_decode_fixture import (
    configure_router,
    launch_mega_atten,
    make_model_weights,
    make_slot,
    synchronize_checked,
)

KINDS = (
    "attn_mega",
    "moe_mega",
    "attn_mega_moe_mega",
)


def normalized_input(slot):
    x = slot.layer_input.float()
    return (
        x
        * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        * (1.0 + slot.post_attention_rms_weight.float())
    ).to(torch.bfloat16)


def moe_weights(slot):
    arguments = dict(
        router_weight=slot.router_weight, router_correction_bias=slot.router_bias
    )
    for prefix in ("shared_fc1", "shared_fc2", "routed_fc1", "routed_fc2"):
        arguments[prefix + "_weight"] = getattr(slot.weights, prefix)
        arguments[prefix + "_weight_sf"] = getattr(slot.weights, prefix + "_sf")
        if prefix.startswith("routed"):
            arguments[prefix + "_alpha"] = getattr(slot.weights, prefix + "_alpha")
    return MoeWeights.from_mega(**arguments)


class ExecutionResources:
    """Own TP resources used sequentially by the prepared operations."""

    def __init__(self, kind, rank, device, group_id):
        if kind not in KINDS:
            raise ValueError(f"unsupported operation: {kind}")
        self.kind, self.rank, self.device = kind, rank, device
        self.attention_workspace = None
        self.moe_context = None
        if kind == "attn_mega" or kind.startswith("attn_mega_"):
            self.attention_workspace = create_tp_allreduce_workspace(
                rank, 4, group_id + "-attention", 0, device.index
            )
        if kind != "attn_mega":
            backend = "mega"
            self.moe_context = create_moe_tp_context(
                group_id=group_id + "-moe", backend=backend
            )

    def close(self):
        if self.moe_context is not None:
            self.moe_context.close()
        if self.attention_workspace is not None:
            destroy_tp_allreduce_workspace(self.attention_workspace)


class PreparedOperation:
    """Bind one independent input/weight slot to a complete operation."""

    def __init__(
        self,
        resources,
        *,
        q,
        kv_len,
        seed,
        zero_math,
        active_experts=20,
        enable_pdl=True,
    ):
        if not 1 <= q <= 8:
            raise ValueError("Q must be 1..8")
        self.resources, self.q = resources, q
        self.kind = resources.kind
        weights = make_model_weights(
            resources.device, seed + resources.rank * 100003, zero_math=zero_math
        )
        if zero_math:
            # Zero payloads remain an exact oracle with valid positive scales.
            weights.routed_fc1_alpha.fill_(1.0)
            weights.routed_fc2_alpha.fill_(1.0)
        self.valid_rows = slice(8 - q, 8)
        self.slot = make_slot(resources.device, resources.rank, kv_len, seed, weights)
        slot = self.slot
        if q < 8:
            slot.slot_mapping[:8 - q] = -1
        configure_router(slot, slot.layer_input, active_experts, q_len=q)
        self.plan = None
        self.runner = None
        if resources.moe_context is not None:
            if self.kind.startswith("attn_mega_"):
                launch_mega_atten(
                    slot, resources.attention_workspace, materialize_topk=True
                )
                synchronize_checked(resources.device, "prepare_attention_boundary")
            else:
                slot.split_attention_residual.copy_(slot.layer_input)
                slot.post_attention_norm.copy_(normalized_input(slot))
            configure_router(
                slot, slot.split_attention_residual, active_experts, q_len=q
            )
            backend = "mega"
            self.runner = create_moe_runner(
                weights=moe_weights(slot),
                config=MoeConfig(enable_pdl=enable_pdl),
                tp_context=resources.moe_context,
                backend=backend,
            )
            self.plan = self.runner.prepare(
                q_len=q,
                hidden=slot.post_attention_norm[self.valid_rows],
                residual=slot.split_attention_residual[self.valid_rows],
            )

        self.kernel_count = 2 if self.kind == "attn_mega_moe_mega" else 1

    def run(self):
        if self.kind == "attn_mega" or self.kind.startswith("attn_mega_"):
            launch_mega_atten(
                self.slot, self.resources.attention_workspace, materialize_topk=True
            )
        if self.plan is not None:
            return self.plan.run()
        return self.slot.post_attention_norm[self.valid_rows]

    def expected_zero_math(self):
        if self.kind == "attn_mega":
            return normalized_input(self.slot)[self.valid_rows]
        return self.slot.layer_input[self.valid_rows]
