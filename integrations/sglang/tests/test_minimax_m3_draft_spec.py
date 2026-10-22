"""The DSpark draft backbone spec: what SGLang routes to it, and what it refuses."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import sglang_msa_mono
import torch
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.monokernels import spec as registry
from sglang.srt.monokernels.runtime import _served_keys
from sglang_msa_mono.minimax_m3 import DRAFT_ROWS
from sglang_msa_mono.minimax_m3 import draft_runtime
from sglang_msa_mono.minimax_m3.draft_spec import SPEC

ARCH = "DSparkMiniMaxDraftModel"
# The draft config SGLang derives from MiniMax-M3.1-Flash/dspark, reduced to the fields read
CONFIG = {
    "hidden_size": 6144,
    "head_dim": 128,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "rotary_dim": 64,
    "num_hidden_layers": 5,
    "dense_intermediate_size": 12288,
    "hidden_act": "swigluoai",
    "swiglu_alpha": 1.702,
    "swiglu_limit": 7.0,
    "use_gemma_norm": True,
    "qk_norm_type": "per_head",
    "sliding_window": 4096,
    "rms_norm_eps": 1e-6,
    "dflash_config": {"layer_types": ["sliding_attention"] * 5, "sliding_window": 4096},
}


class _Backend:
    def __init__(self, name="flashinfer_cutlass"):
        self.name = name

    def is_flashinfer_trtllm(self):
        return self.name == "flashinfer_trtllm"


def _linear(rows, k, backend="flashinfer_cutlass"):
    return SimpleNamespace(
        weight=torch.empty((rows, k), dtype=torch.float8_e4m3fn),
        weight_scale_inv=torch.empty((rows, k // 32), dtype=torch.uint8),
        bias=None,
        quant_method=SimpleNamespace(use_mxfp8=True, mxfp8_dense_backend=_Backend(backend)),
    )


def _layer(rope, *, backend="flashinfer_cutlass", window=4095, kv_heads=1):
    attn = SimpleNamespace(
        qkv_proj=_linear(2048 + 2 * 128 * kv_heads, 6144, backend),
        o_proj=_linear(6144, 2048, backend),
        num_heads=16,
        num_kv_heads=kv_heads,
        attn=SimpleNamespace(sliding_window_size=window, layer_id=0),
        rotary_emb=rope,
    )
    mlp = SimpleNamespace(
        gate_up_proj=_linear(6144, 6144, backend), down_proj=_linear(6144, 3072, backend)
    )
    decoder = SimpleNamespace(self_attn=attn, mlp=mlp, is_layer_sparse=False)
    return SimpleNamespace(decoder_layer=decoder, self_attn=attn)


def _rope(rotary_dim=64, neox=True):
    return RotaryEmbedding(128, rotary_dim, 8192, 10_000_000, neox, torch.bfloat16)


def _model(layers=None, **config):
    rope = _rope()
    return SimpleNamespace(
        config=SimpleNamespace(**{**copy.deepcopy(CONFIG), **config}),
        layers=layers if layers is not None else [_layer(rope) for _ in range(5)],
    )


@pytest.fixture(autouse=True)
def _tp4(monkeypatch):
    sglang_msa_mono.register_specs()
    parallel = SimpleNamespace(tp_size=4, attn_tp_size=4, alignment=1)
    monkeypatch.setattr(draft_runtime, "_tp", lambda: parallel)
    monkeypatch.setattr(draft_runtime, "_row_alignment", lambda: parallel.alignment)
    return parallel


def _reasons(model, hf_config=None):
    return draft_runtime.model_disable_reasons(model, hf_config or SimpleNamespace(), {})


def test_the_draft_architecture_resolves_to_the_draft_spec():
    for isa in ("sm_100", "sm_103"):
        spec, why = registry.find_spec(
            backend=sglang_msa_mono.BACKEND, architectures=[ARCH], isa=isa
        )
        assert spec is SPEC, why


def test_the_draft_block_is_the_one_served_step():
    # the target's verify window is 8 rows; the draft's block is one row shorter
    assert SPEC.tokens_per_request(None) == DRAFT_ROWS
    widths = tuple(w for w in SPEC.widths("sm_103") if w % DRAFT_ROWS == 0)
    served = _served_keys(mode="target_verify", widths=widths, tokens=DRAFT_ROWS)
    assert served == frozenset(
        {registry.ShapeKey(forward_mode="target_verify", batch_size=1, tokens_per_request=7)}
    )


def test_the_flash_draft_is_accepted():
    assert _reasons(_model()) == []


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"num_hidden_layers": 4}, "config.num_hidden_layers 4"),
        ({"dense_intermediate_size": 6144}, "config.dense_intermediate_size 6144"),
        ({"sliding_window": 8192}, "config.sliding_window 8192"),
        ({"use_gemma_norm": False}, "config.use_gemma_norm False"),
        ({"rope_scaling": {"type": "yarn"}}, "config.rope_scaling"),
        (
            {"dflash_config": {"layer_types": ["full_attention"] * 5}},
            "the kernel runs sliding attention",
        ),
    ],
)
def test_draft_shapes_the_kernel_does_not_build_are_refused(config, expected):
    reasons = _reasons(_model(**config))
    assert any(expected in r for r in reasons), reasons


def test_an_eight_row_block_is_refused():
    reasons = _reasons(_model(), SimpleNamespace(sample_from_anchor=False))
    assert any("sample_from_anchor is off" in r for r in reasons), reasons


def test_other_tensor_parallel_layouts_are_refused(_tp4):
    _tp4.attn_tp_size = 2  # DP attention on TP4
    reasons = _reasons(_model())
    assert any("attention tp 2" in r for r in reasons), reasons


def test_graphs_padded_to_attention_tp_are_refused(_tp4):
    # an a2a MoE (megamoe) pads draft rows to attention TP: one block lands in a bs=4 graph
    _tp4.alignment = 4
    reasons = _reasons(_model())
    assert any("pass --disable-attn-tp-gather" in r for r in reasons), reasons


@pytest.mark.parametrize(
    ("layer", "expected"),
    [
        (dict(backend="flashinfer_trtllm"), "shuffles MXFP8 weights"),
        (dict(window=-1), "window_left -1"),
        (dict(kv_heads=2), "16 Q / 2 KV heads a rank"),
    ],
)
def test_layers_the_kernel_cannot_read_are_refused(layer, expected):
    rope = _rope()
    layers = [_layer(rope) for _ in range(4)] + [_layer(rope, **layer)]
    reasons = _reasons(_model(layers))
    assert any(expected in r for r in reasons), reasons


def test_a_bf16_linear_is_refused():
    model = _model()
    model.layers[2].decoder_layer.mlp.down_proj.quant_method = SimpleNamespace()
    reasons = _reasons(model)
    assert any("draft layer 2 down_proj is not an MXFP8 linear" in r for r in reasons), reasons


def test_rope_tables_the_kernel_does_not_read_are_refused():
    layers = [_layer(_rope()) for _ in range(5)]
    reasons = _reasons(_model(layers))
    assert any("do not share one RoPE table" in r for r in reasons), reasons
    shared = _rope(neox=False)
    reasons = _reasons(_model([_layer(shared) for _ in range(5)]))
    assert any("fp32 neox RoPE table" in r for r in reasons), reasons
