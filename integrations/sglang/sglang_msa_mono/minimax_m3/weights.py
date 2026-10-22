"""SGLang's MiniMax-M3.1 parameters as MSA-v1 Attention and MoE weights.

Attention, shared-expert and router operands are read from the SGLang module
after load: the MXFP8 ``weight`` / ``weight_scale_inv`` stay in checkpoint
layout under every MXFP8 dense GEMM backend except flashinfer_trtllm, which
shuffles the weight in place (refused by the spec). The routed experts are
not: their runner rewrites them into a TensorRT-LLM layout, so ``load_tensor``
streams each checkpoint expert tensor straight into this rank's slice of the
``MoeWeights`` logical layout on the GPU instead.

Operand addresses remain stable after loading. The serving plugin does not
support online weight updates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields

import torch

from inference.high_tps.msa_v1.attention import AttentionWeights
from inference.high_tps.msa_v1.moe import MoeWeights
from inference.high_tps.msa_v1.moe.weights import _pack_scales
from sglang_msa_mono.minimax_m3 import (
    EXPERT_INTER,
    EXPERTS,
    HEAD_DIM,
    HIDDEN,
    INTER,
    LOCAL_Q_HEADS,
    O_PROJ_K,
    PROJECTION_ROWS,
    SHARED_INTER,
    TP,
)

SCALE_BLOCK = 32  # MXFP8: one UE8M0 scale per 32 K-values
NVFP4_GROUP = 16  # NVFP4: one E4M3 scale per 16 values
# FC1 rows of one expert on one rank: gate 768 then up 768
FC1_ROWS = 2 * INTER

# A checkpoint routed-expert tensor of a decoder layer; names reach load_tensor
# raw, before SGLang's loader strips ``language_model.`` or renames
# ``block_sparse_moe``.
_EXPERT_NAME = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.(?:block_sparse_moe|mlp)\.experts\."
    r"(?P<expert>\d+)\.(?P<proj>w[123])\.(?P<field>weight_packed|weight_scale|"
    r"weight_global_scale|input_global_scale)$"
)
# DSpark's draft layers live under ``model.dspark.layers.*``; they are not ours.
_DRAFT = ".dspark."
PROJECTIONS = ("w1", "w3", "w2")
FIELDS = ("weight_packed", "weight_scale")


def parse_expert_name(name: str) -> tuple[int, int, str, str] | None:
    """(layer, expert, w1|w2|w3, field) of a target routed-expert tensor, else None."""
    if _DRAFT in name:
        return None
    match = _EXPERT_NAME.search(name)
    if match is None:
        return None
    return int(match["layer"]), int(match["expert"]), match["proj"], match["field"]


def rank_rows(total: int, rank: int, tp: int = TP) -> slice:
    """This rank's slice of a dimension TP-sharded into equal parts."""
    if total % tp:
        raise ValueError(f"{total} rows do not shard over {tp} ranks")
    part = total // tp
    return slice(rank * part, (rank + 1) * part)


def swizzled_scales(scale: torch.Tensor) -> torch.Tensor:
    """Row-major [R, C] UE8M0 scales in Attention's 128x4 tile layout."""
    rows, cols = scale.shape[-2:]
    if rows % 128 or cols % 4:
        raise ValueError(f"scales {tuple(scale.shape)} do not tile by 128x4")
    return _pack_scales(scale.contiguous())


@dataclass
class LayerOperands:
    """One sparse layer's attention, router and shared-expert operands."""

    input_norm: torch.Tensor  # bf16 [6144], raw Gemma w (the kernel applies 1+w)
    projection: torch.Tensor  # e4m3 [2560, 6144]: q | k | v | qi | ki
    projection_sf: torch.Tensor  # u8 [2560*192], 128x4 swizzled
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    qi_norm: torch.Tensor
    ki_norm: torch.Tensor
    output_projection: torch.Tensor  # e4m3 [6144, 2048]
    output_projection_sf: torch.Tensor  # u8 [6144*64], swizzled
    post_attention_norm: torch.Tensor  # bf16 [6144]
    router_weight: torch.Tensor  # f32 [128, 6144], raw
    router_correction_bias: torch.Tensor  # f32 [128]
    shared_fc1_weight: torch.Tensor  # e4m3 [1536, 6144], gate rows then up rows
    shared_fc1_weight_sf: torch.Tensor  # u8 [1536, 192], row-major
    shared_fc2_weight: torch.Tensor  # e4m3 [6144, 768]
    shared_fc2_weight_sf: torch.Tensor  # u8 [6144, 24], row-major

    def nbytes(self) -> int:
        return sum(getattr(self, f.name).nbytes for f in fields(self))

    def attention_weights(self) -> AttentionWeights:
        return AttentionWeights(
            **{f.name: getattr(self, f.name) for f in fields(AttentionWeights)}
        )


_LAYER_SHAPES = {
    "input_norm": ((HIDDEN,), torch.bfloat16),
    "projection": ((PROJECTION_ROWS, HIDDEN), torch.float8_e4m3fn),
    "projection_sf": ((PROJECTION_ROWS * HIDDEN // SCALE_BLOCK,), torch.uint8),
    "q_norm": ((HEAD_DIM,), torch.bfloat16),
    "k_norm": ((HEAD_DIM,), torch.bfloat16),
    "qi_norm": ((HEAD_DIM,), torch.bfloat16),
    "ki_norm": ((HEAD_DIM,), torch.bfloat16),
    "output_projection": ((HIDDEN, O_PROJ_K), torch.float8_e4m3fn),
    "output_projection_sf": ((HIDDEN * O_PROJ_K // SCALE_BLOCK,), torch.uint8),
    "post_attention_norm": ((HIDDEN,), torch.bfloat16),
    "router_weight": ((EXPERTS, HIDDEN), torch.float32),
    "router_correction_bias": ((EXPERTS,), torch.float32),
    "shared_fc1_weight": ((FC1_ROWS, HIDDEN), torch.float8_e4m3fn),
    "shared_fc1_weight_sf": ((FC1_ROWS, HIDDEN // SCALE_BLOCK), torch.uint8),
    "shared_fc2_weight": ((HIDDEN, INTER), torch.float8_e4m3fn),
    "shared_fc2_weight_sf": ((HIDDEN, INTER // SCALE_BLOCK), torch.uint8),
}


def allocate_layer(device: torch.device) -> LayerOperands:
    return LayerOperands(
        **{
            name: torch.zeros(shape, dtype=dtype, device=device)
            for name, (shape, dtype) in _LAYER_SHAPES.items()
        }
    )


def _mxfp8(module: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """A linear's checkpoint-layout MXFP8 weight and UE8M0 scales."""
    weight = module.weight
    scale = getattr(module, "weight_scale_inv", None)
    if weight.dtype != torch.float8_e4m3fn or scale is None or scale.dtype != torch.uint8:
        raise ValueError(
            f"{type(module).__name__} is not an MXFP8 linear "
            f"(weight {weight.dtype}, scale {None if scale is None else scale.dtype})"
        )
    if scale.shape != (weight.shape[0], weight.shape[1] // SCALE_BLOCK):
        raise ValueError(
            f"MXFP8 scales {tuple(scale.shape)} do not match weight {tuple(weight.shape)}"
        )
    return weight.data, scale.data


def _qkv_index_modules(attn: torch.nn.Module) -> list[torch.nn.Module]:
    """SGLang's fused [qkv | index] projection after load, else the two linears."""
    fused = getattr(attn, "fused_qkv_index_proj", None)
    return [fused] if fused is not None else [attn.qkv_proj, attn.index_qkv_proj]


def mxfp8_quant_method(module: torch.nn.Module):
    # the fused holder keeps its method as ``_qm`` so the loader skips it
    return getattr(module, "quant_method", None) or getattr(module, "_qm", None)


def mxfp8_modules(layer: torch.nn.Module) -> list[torch.nn.Module]:
    """The MXFP8 linears one sparse layer's operands are read from."""
    attn, shared = layer.self_attn, layer.mlp.shared_experts
    return [*_qkv_index_modules(attn), attn.o_proj, shared.gate_up_proj, shared.down_proj]


def _shared_slices(shared: torch.nn.Module, rank: int):
    """This rank's [gate 768 | up 768] FC1 rows and FC2 columns of the shared expert.

    SGLang holds the shared expert TP-sharded (768 a rank) or, under A2A MoE
    backends, replicated; the kernel always reads this rank's 768 slice.
    """
    gate_up, gate_up_sf = _mxfp8(shared.gate_up_proj)
    down, down_sf = _mxfp8(shared.down_proj)
    rows = gate_up.shape[0] // 2
    if rows == SHARED_INTER:
        part = rank_rows(SHARED_INTER, rank)
        scale_part = slice(part.start // SCALE_BLOCK, part.stop // SCALE_BLOCK)
        gate, up = gate_up[:rows][part], gate_up[rows:][part]
        gate_sf, up_sf = gate_up_sf[:rows][part], gate_up_sf[rows:][part]
        down, down_sf = down[:, part], down_sf[:, scale_part]
    elif rows == INTER:
        gate, up = gate_up[:rows], gate_up[rows:]
        gate_sf, up_sf = gate_up_sf[:rows], gate_up_sf[rows:]
    else:
        raise ValueError(
            f"shared expert intermediate {rows} is neither {SHARED_INTER} nor {INTER}"
        )
    return torch.cat((gate, up)), torch.cat((gate_sf, up_sf)), down, down_sf


def _projection(attn: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """[q 2048 | k | v | qi | ki] rows of this rank (K-only index: its V width is 0)."""
    parts = [_mxfp8(m) for m in _qkv_index_modules(attn)]
    weight = torch.cat([w for w, _ in parts])
    scale = torch.cat([s for _, s in parts])
    rows = LOCAL_Q_HEADS * HEAD_DIM + 2 * HEAD_DIM + 2 * HEAD_DIM
    if weight.shape[0] != rows:
        raise ValueError(
            f"qkv + index projection rows {weight.shape[0]} are not one TP4 rank's "
            "16 Q heads, 1 KV head and 1 index head"
        )
    return weight, scale


@torch.no_grad()
def fill_from_module(operands: LayerOperands, layer: torch.nn.Module, *, rank: int) -> None:
    """Copy one decoder layer's attention, router and shared expert into operands."""
    attn, mlp = layer.self_attn, layer.mlp
    weight, scale = _projection(attn)
    operands.projection.copy_(weight)
    operands.projection_sf.copy_(swizzled_scales(scale))
    o_weight, o_scale = _mxfp8(attn.o_proj)
    operands.output_projection.copy_(o_weight)
    operands.output_projection_sf.copy_(swizzled_scales(o_scale))
    for name, module in (
        ("input_norm", layer.input_layernorm),
        ("post_attention_norm", layer.post_attention_layernorm),
        ("q_norm", attn.q_norm),
        ("k_norm", attn.k_norm),
        ("qi_norm", attn.index_q_norm),
        ("ki_norm", attn.index_k_norm),
    ):
        getattr(operands, name).copy_(module.weight.data)
    operands.router_weight.copy_(mlp.gate.weight.data)
    bias = mlp.e_score_correction_bias
    if bias is None:
        operands.router_correction_bias.zero_()
    else:
        operands.router_correction_bias.copy_(bias.data)
    fc1, fc1_sf, fc2, fc2_sf = _shared_slices(mlp.shared_experts, rank)
    operands.shared_fc1_weight.copy_(fc1)
    operands.shared_fc1_weight_sf.copy_(fc1_sf)
    operands.shared_fc2_weight.copy_(fc2)
    operands.shared_fc2_weight_sf.copy_(fc2_sf)


COPIES_KIND = "routed experts"
# field -> (shape, dtype, on the GPU); the scale and seen fields are host bookkeeping
_ROUTED_SHAPES = {
    "fc1_weight": ((EXPERTS, FC1_ROWS, HIDDEN // 2), torch.uint8, True),
    "fc1_weight_sf": ((EXPERTS, FC1_ROWS, HIDDEN // NVFP4_GROUP), torch.float8_e4m3fn, True),
    "fc1_alpha": ((EXPERTS,), torch.float32, True),
    "fc2_weight": ((EXPERTS, HIDDEN, INTER // 2), torch.uint8, True),
    "fc2_weight_sf": ((EXPERTS, HIDDEN, INTER // NVFP4_GROUP), torch.float8_e4m3fn, True),
    "fc2_alpha": ((EXPERTS,), torch.float32, True),
    "global_scales": ((3, EXPERTS), torch.float32, False),
    "input_scales": ((3, EXPERTS), torch.float32, False),
    "seen": ((3, 2, EXPERTS), torch.bool, False),
}


@dataclass
class RoutedExperts:
    """One layer's routed experts on one rank, in ``MoeWeights``' logical layout.

    The kernel shards every expert's 3072 intermediate over TP4 (768 a rank):
    FC1 takes rows [r*768, (r+1)*768) of w1 (gate) and of w3 (up); FC2 takes the
    same intermediate columns of w2. The alphas are derived from the checkpoint
    global scales on every refresh: FC1 1/s(w1) (w1 and w3 must match: the kernel
    has one FC1 alpha), FC2 1/s(w2); ``MoeConfig`` applies the routing multiplier.
    """

    fc1_weight: torch.Tensor  # u8 [128, 1536, 3072], gate rows then up rows
    fc1_weight_sf: torch.Tensor  # e4m3 [128, 1536, 384], row-major
    fc1_alpha: torch.Tensor  # f32 [128]
    fc2_weight: torch.Tensor  # u8 [128, 6144, 384]
    fc2_weight_sf: torch.Tensor  # e4m3 [128, 6144, 48], row-major
    fc2_alpha: torch.Tensor  # f32 [128]
    # host bookkeeping: per (w1, w3, w2) the weight and input global scales, NaN
    # until loaded, and which (projection, field) tensors arrived per expert
    global_scales: torch.Tensor  # f32 [3, 128]
    input_scales: torch.Tensor  # f32 [3, 128]
    seen: torch.Tensor  # bool [3, 2, 128]

    @classmethod
    def in_copies(cls, copies: dict, layer: int, device: torch.device) -> RoutedExperts:
        """The layer's experts, kept as SGLang's load-hook copies (``(kind, layer,
        field) -> tensor``, which SGLang counts in its build log); allocated at the
        layer's first tensor, the same storage for every later load."""
        key = (COPIES_KIND, layer)
        if (*key, "fc1_weight") not in copies:
            for name, (shape, dtype, on_gpu) in _ROUTED_SHAPES.items():
                tensor = torch.zeros(shape, dtype=dtype, device=device if on_gpu else "cpu")
                if name.endswith("_scales"):
                    tensor.fill_(float("nan"))  # unseen until the checkpoint's arrives
                copies[(*key, name)] = tensor
        return cls(**{f.name: copies[(*key, f.name)] for f in fields(cls)})

    def nbytes(self) -> int:
        return sum(getattr(self, f.name).nbytes for f in fields(self))

    @torch.no_grad()
    def stage(self, expert: int, proj: str, field: str, tensor: torch.Tensor, rank: int):
        """Write one checkpoint tensor (full TP-unsharded shape) into this rank's slice."""
        if not 0 <= expert < EXPERTS:
            raise ValueError(f"expert {expert} is outside 0..{EXPERTS - 1}")
        p = PROJECTIONS.index(proj)
        if field in ("weight_global_scale", "input_global_scale"):
            target = self.global_scales if field == "weight_global_scale" else self.input_scales
            target[p, expert] = float(tensor.reshape(-1)[0])
            return
        rows, device = rank_rows(EXPERT_INTER, rank), self.fc1_weight.device
        if proj in ("w1", "w3"):
            cols = HIDDEN // 2 if field == "weight_packed" else HIDDEN // NVFP4_GROUP
            if tensor.shape != (EXPERT_INTER, cols):
                raise ValueError(f"{proj}.{field} has shape {tuple(tensor.shape)}")
            half = slice(0, INTER) if proj == "w1" else slice(INTER, FC1_ROWS)
            target = self.fc1_weight if field == "weight_packed" else self.fc1_weight_sf
            target[expert, half].view(torch.uint8).copy_(tensor[rows].view(torch.uint8))
        else:  # w2: [6144, 3072 / group], columns are the intermediate
            group = 2 if field == "weight_packed" else NVFP4_GROUP
            if tensor.shape != (HIDDEN, EXPERT_INTER // group):
                raise ValueError(f"w2.{field} has shape {tuple(tensor.shape)}")
            part = tensor[:, rows.start // group : rows.stop // group]
            target = self.fc2_weight if field == "weight_packed" else self.fc2_weight_sf
            target[expert].view(torch.uint8).copy_(part.to(device).view(torch.uint8))
        self.seen[p, FIELDS.index(field), expert] = True

    def missing(self) -> list[str]:
        """The checkpoint tensors this layer never received."""
        out = [
            f"expert {e} {PROJECTIONS[p]}.{FIELDS[f]}"
            for p, f, e in (~self.seen).nonzero().tolist()
        ]
        out += [
            f"expert {e} {PROJECTIONS[p]}.weight_global_scale"
            for p, e in self.global_scales.isnan().nonzero().tolist()
        ]
        return out

    def scale_reasons(self) -> list[str]:
        """Checkpoint global scales the kernel's fixed alphas cannot express."""
        reasons = []
        w1, w3 = self.global_scales[0], self.global_scales[1]
        both = ~(w1.isnan() | w3.isnan())
        for e in (both & (w1 != w3)).nonzero().flatten().tolist():
            reasons.append(
                f"expert {e}: w1 and w3 global scales differ ({w1[e]} vs {w3[e]}); "
                "the kernel has one FC1 alpha"
            )
        inputs = self.input_scales
        for p, e in (~inputs.isnan() & (inputs != 1.0)).nonzero().tolist():
            reasons.append(
                f"expert {e} {PROJECTIONS[p]}.input_global_scale {inputs[p, e]} is not 1 "
                "(the kernel quantizes activations with unit global scales)"
            )
        return reasons

    @torch.no_grad()
    def refresh_alphas(self) -> None:
        """Alphas of every expert whose global scales arrived; 1 for the others
        (a dummy load; real-checkpoint validation requires all scales)."""
        for alpha, scale in ((self.fc1_alpha, self.global_scales[0]), (self.fc2_alpha, self.global_scales[2])):
            alpha.copy_(torch.where(scale.isnan(), 1.0, 1.0 / scale))


def moe_weights(operands: LayerOperands, experts: RoutedExperts) -> MoeWeights:
    """The layer's ``MoeWeights``, viewing (not copying) operands and experts."""
    return MoeWeights(
        router_weight=operands.router_weight,
        router_correction_bias=operands.router_correction_bias,
        shared_fc1_weight=operands.shared_fc1_weight,
        shared_fc1_weight_sf=operands.shared_fc1_weight_sf,
        shared_fc2_weight=operands.shared_fc2_weight,
        shared_fc2_weight_sf=operands.shared_fc2_weight_sf,
        routed_fc1_weight=experts.fc1_weight,
        routed_fc1_weight_sf=experts.fc1_weight_sf,
        routed_fc1_alpha=experts.fc1_alpha,
        routed_fc2_weight=experts.fc2_weight,
        routed_fc2_weight_sf=experts.fc2_weight_sf,
        routed_fc2_alpha=experts.fc2_alpha,
    )
