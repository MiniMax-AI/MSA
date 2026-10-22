"""The plugin's routed experts against the ``MoeWeights`` weight contract.

``MoeWeights`` holds FC1 as all gate rows then all up rows, with logical
row-major scales; the plugin builds those tensors one checkpoint tensor at a
time, from full TP-unsharded checkpoint shapes, in any arrival order.
"""

from __future__ import annotations

import random

import pytest
import torch
from sglang_msa_mono.minimax_m3 import EXPERT_INTER, EXPERTS, HIDDEN, TP
from sglang_msa_mono.minimax_m3.weights import (
    COPIES_KIND,
    RoutedExperts,
    parse_expert_name,
    rank_rows,
)


def _checkpoint_expert(generator: torch.Generator) -> dict[str, torch.Tensor]:
    def randint(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)

    def scale():
        return torch.tensor(float(torch.randint(1, 1 << 15, (1,), generator=generator)))

    shared = scale()  # gate and up share one global scale in M3 checkpoints
    return {
        "w1.weight_packed": randint(EXPERT_INTER, HIDDEN // 2),
        "w3.weight_packed": randint(EXPERT_INTER, HIDDEN // 2),
        "w1.weight_scale": randint(EXPERT_INTER, HIDDEN // 16).view(torch.float8_e4m3fn),
        "w3.weight_scale": randint(EXPERT_INTER, HIDDEN // 16).view(torch.float8_e4m3fn),
        "w2.weight_packed": randint(HIDDEN, EXPERT_INTER // 2),
        "w2.weight_scale": randint(HIDDEN, EXPERT_INTER // 16).view(torch.float8_e4m3fn),
        "w1.weight_global_scale": shared,
        "w3.weight_global_scale": shared.clone(),
        "w2.weight_global_scale": scale(),
        "w1.input_global_scale": torch.tensor(1.0),
        "w3.input_global_scale": torch.tensor(1.0),
        "w2.input_global_scale": torch.tensor(1.0),
    }


def _reference(tensors: dict[str, torch.Tensor], rank: int) -> dict[str, torch.Tensor]:
    """One expert as the ``MoeWeights`` contract lays it out for this rank."""
    rows = rank_rows(EXPERT_INTER, rank)
    w2_cols = slice(rows.start // 2, rows.stop // 2)
    w2_sf_cols = slice(rows.start // 16, rows.stop // 16)
    fc1_sf = torch.cat((tensors["w1.weight_scale"][rows], tensors["w3.weight_scale"][rows]))
    return {
        "fc1_weight": torch.cat(
            (tensors["w1.weight_packed"][rows], tensors["w3.weight_packed"][rows])
        ),
        "fc1_weight_sf": fc1_sf.view(torch.uint8),
        "fc2_weight": tensors["w2.weight_packed"][:, w2_cols],
        "fc2_weight_sf": tensors["w2.weight_scale"][:, w2_sf_cols].view(torch.uint8),
        # MoeConfig.routed_scaling_factor scales FC2 at run time, not here
        "fc1_alpha": 1.0 / tensors["w1.weight_global_scale"],
        "fc2_alpha": 1.0 / tensors["w2.weight_global_scale"],
    }


def _stage(copies, layer, expert, tensors, rank, order):
    experts = RoutedExperts.in_copies(copies, layer, torch.device("cpu"))
    for key in order:
        proj, field = key.split(".")
        experts.stage(expert, proj, field, tensors[key], rank)
    return experts


@pytest.mark.parametrize("rank", range(TP))
def test_streamed_experts_match_the_moe_weight_contract(rank):
    generator = torch.Generator().manual_seed(1234 + rank)
    experts_under_test = (0, 77, EXPERTS - 1)
    checkpoint = {e: _checkpoint_expert(generator) for e in experts_under_test}
    copies: dict = {}
    for e in experts_under_test:
        order = list(checkpoint[e])
        random.Random(e).shuffle(order)  # loaders yield tensors in shard order
        experts = _stage(copies, 3, e, checkpoint[e], rank, order)
    experts.refresh_alphas()
    for e in experts_under_test:
        reference = _reference(checkpoint[e], rank)
        assert torch.equal(experts.fc1_weight[e], reference["fc1_weight"])
        assert torch.equal(experts.fc1_weight_sf[e].view(torch.uint8), reference["fc1_weight_sf"])
        assert torch.equal(experts.fc2_weight[e], reference["fc2_weight"])
        assert torch.equal(experts.fc2_weight_sf[e].view(torch.uint8), reference["fc2_weight_sf"])
        assert experts.fc1_alpha[e].item() == pytest.approx(reference["fc1_alpha"].item())
        assert experts.fc2_alpha[e].item() == pytest.approx(reference["fc2_alpha"].item())
    assert experts.scale_reasons() == []
    missing = experts.missing()
    assert len(missing) == (EXPERTS - len(experts_under_test)) * 9  # 6 tensors + 3 scales
    assert not any(f"expert {e} " in m for e in experts_under_test for m in missing)


def test_online_update_rewrites_the_same_storage():
    generator = torch.Generator().manual_seed(7)
    copies: dict = {}
    first = _stage(copies, 5, 2, _checkpoint_expert(generator), 1, list(_checkpoint_expert(generator)))
    pointers = {name: t.data_ptr() for name, t in first.__dict__.items()}
    update = _checkpoint_expert(generator)
    second = _stage(copies, 5, 2, update, 1, list(update))
    second.refresh_alphas()
    assert {name: t.data_ptr() for name, t in second.__dict__.items()} == pointers
    assert torch.equal(second.fc1_weight[2], _reference(update, 1)["fc1_weight"])
    assert {k[0] for k in copies} == {COPIES_KIND}


def test_mismatched_gate_up_scales_and_input_scales_are_refused():
    generator = torch.Generator().manual_seed(9)
    tensors = _checkpoint_expert(generator)
    tensors["w3.weight_global_scale"] = tensors["w1.weight_global_scale"] * 2
    tensors["w2.input_global_scale"] = torch.tensor(0.5)
    experts = _stage({}, 3, 4, tensors, 0, list(tensors))
    reasons = experts.scale_reasons()
    assert any("expert 4: w1 and w3 global scales differ" in r for r in reasons)
    assert any("expert 4 w2.input_global_scale 0.5" in r for r in reasons)


def test_unloaded_experts_get_unit_alphas():
    # a dummy load: the kernel must still see finite alphas until the first update
    experts = RoutedExperts.in_copies({}, 3, torch.device("cpu"))
    experts.refresh_alphas()
    assert torch.equal(experts.fc1_alpha, torch.ones(EXPERTS))
    assert torch.equal(experts.fc2_alpha, torch.ones(EXPERTS))


@pytest.mark.parametrize(
    ("name", "parsed"),
    [
        (
            "language_model.model.layers.3.block_sparse_moe.experts.17.w1.weight_packed",
            (3, 17, "w1", "weight_packed"),
        ),
        ("model.layers.59.mlp.experts.127.w2.weight_global_scale", (59, 127, "w2", "weight_global_scale")),
        ("model.layers.4.block_sparse_moe.experts.0.w3.input_global_scale", (4, 0, "w3", "input_global_scale")),
        # DSpark's draft layers, the shared expert, and the router are not routed experts
        ("language_model.model.dspark.layers.3.decoder_layer.mlp.experts.1.w1.weight_packed", None),
        ("model.layers.3.block_sparse_moe.shared_experts.gate_proj.weight", None),
        ("model.layers.3.block_sparse_moe.gate.weight", None),
        ("model.layers.3.block_sparse_moe.experts.1.w1.weight", None),
    ],
)
def test_expert_names(name, parsed):
    assert parse_expert_name(name) == parsed
