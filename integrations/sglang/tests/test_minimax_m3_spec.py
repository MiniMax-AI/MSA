"""SGLang's startup checks for the MiniMax-M3.1 MSA-v1 monokernel spec."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import sglang_msa_mono
from sglang.srt.monokernels import spec as registry
from sglang.srt.monokernels.guards import monokernel_disable_reasons
from sglang.srt.server_args import ServerArgs
from sglang_msa_mono.minimax_m3 import runtime
from sglang_msa_mono.minimax_m3.spec import SPEC

# The text config of MiniMaxAI/MiniMax-M3.1-Flash, reduced to the fields read
TEXT = {
    "hidden_size": 6144,
    "head_dim": 128,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "rotary_dim": 64,
    "num_local_experts": 128,
    "num_experts_per_tok": 4,
    "intermediate_size": 3072,
    "shared_intermediate_size": 3072,
    "n_shared_experts": 1,
    "scoring_func": "sigmoid",
    "hidden_act": "swigluoai",
    "use_gemma_norm": True,
    "qk_norm_type": "per_head",
    "swiglu_alpha": 1.702,
    "swiglu_limit": 7.0,
    "max_position_embeddings": 1048576,
    "moe_layer_freq": [0, 0, 0] + [1] * 57,
    "sparse_attention_config": {
        "use_sparse_attention": True,
        "sparse_index_dim": 128,
        "sparse_num_index_heads": 4,
        "sparse_topk_blocks": 16,
        "sparse_block_size": 128,
        "sparse_score_type": "max",
        "sparse_enable_kv_share": False,
        "sparse_enable_vo_rope": False,
        "sparse_kvshare_value_drop_rope_dim": False,
        "sparse_init_block": 0,
        "sparse_local_block": 1,
        "sparse_attention_freq": [1] * 60,
    },
}
ARCH = "MiniMaxM3SparseForConditionalGeneration"


def _hf_config(**text_overrides):
    text = copy.deepcopy(TEXT)
    for key, value in text_overrides.items():
        if key.startswith("sparse_"):
            text["sparse_attention_config"][key] = value
        else:
            text[key] = value
    return SimpleNamespace(architectures=[ARCH], text_config=SimpleNamespace(**text))


# One request's eight-token chain verify: the only step the kernels serve
VERIFY = {
    "speculative_algorithm": "EAGLE",
    "speculative_num_draft_tokens": 8,
    "speculative_eagle_topk": 1,
    "speculative_num_steps": 7,
}


def _server_args(**kw):
    defaults = {
        "model_path": "dummy",
        "tp_size": 4,
        "quantization": "minimax_m3_nvfp4",
        "monokernel_backend": sglang_msa_mono.BACKEND,
        **VERIFY,
    }
    return ServerArgs(**{**defaults, **kw})


def _reasons(cfg, hf_config=None, isa="sm_103"):
    model_config = SimpleNamespace(hf_config=hf_config or _hf_config(), dtype="bfloat16")
    return monokernel_disable_reasons(cfg=cfg, model_config=model_config, isa=isa)


@pytest.fixture(autouse=True)
def _registered():
    sglang_msa_mono.register_specs()


def test_the_m31_checkpoint_on_sm100_and_sm103_is_accepted():
    for isa in ("sm_100", "sm_103"):
        spec, reasons = _reasons(_server_args(), isa=isa)
        assert spec is SPEC
        assert reasons == []


def test_dense_layers_without_msa_are_accepted():
    # MiniMaxAI/MiniMax-M3 marks the dense layers 0-2 as full attention
    _, reasons = _reasons(_server_args(), _hf_config(sparse_attention_freq=[0, 0, 0] + [1] * 57))
    assert reasons == []


def test_every_registered_architecture_resolves_to_its_spec():
    from sglang_msa_mono.minimax_m3.draft_spec import SPEC as DRAFT_SPEC

    for architecture, path in sglang_msa_mono.SPECS.items():
        spec, why = registry.find_spec(
            backend=sglang_msa_mono.BACKEND, architectures=[architecture], isa="sm_100"
        )
        assert spec is (DRAFT_SPEC if path.endswith("draft_spec:SPEC") else SPEC), why


def test_other_gpus_are_refused():
    spec, why = registry.find_spec(
        backend=sglang_msa_mono.BACKEND, architectures=[ARCH], isa="sm_90"
    )
    assert spec is None and "sm_90" in why


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"hidden_size": 3072}, "config.hidden_size 3072"),
        ({"head_dim": 256}, "config.head_dim 256"),
        ({"swiglu_limit": 10.0}, "config.swiglu_limit 10.0"),
        ({"sparse_topk_blocks": 128}, "sparse_attention_config.sparse_topk_blocks 128"),
        ({"sparse_block_size": 16}, "sparse_attention_config.sparse_block_size 16"),
        ({"sparse_enable_kv_share": True}, "sparse_enable_kv_share is not supported"),
        ({"sparse_indexer_lookahead_num_layers": 1}, "sparse_indexer_lookahead_num_layers"),
        ({"sparse_index_compress_type": "softmax_pool"}, "index compression"),
        ({"sparse_attention_freq": [1] * 3 + [0] + [1] * 56}, "every MoE layer must be an MSA"),
        ({"rope_scaling": {"mrope_section": [16, 24, 24]}}, "config.rope_scaling"),
    ],
)
def test_model_shapes_the_kernel_does_not_build_are_refused(overrides, expected):
    _, reasons = _reasons(_server_args(), _hf_config(**overrides))
    assert any(expected in r for r in reasons), reasons


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({"quantization": "fp8"}, "--quantization 'fp8'"),
        ({"page_size": 64}, "--page-size 64"),
        ({"fp8_gemm_runner_backend": "flashinfer_trtllm"}, "flashinfer_trtllm"),
        ({"context_length": 2 * 1048576}, "exceeds the kernel's 1048576 tokens"),
        ({"tp_size": 8}, "tp_size 8"),
        ({"enable_dp_attention": True, "dp_size": 4}, "DP attention"),
        (
            {"speculative_algorithm": None, "speculative_num_draft_tokens": None,
             "speculative_eagle_topk": None, "speculative_num_steps": None},
            "serve only speculative verify steps",
        ),
        ({"speculative_num_draft_tokens": 4, "speculative_num_steps": 3}, "--speculative-num-draft-tokens 4 is not 8"),
        ({"speculative_num_draft_tokens": 9, "speculative_num_steps": 8}, "--speculative-num-draft-tokens 9 is not 8"),
        ({"speculative_eagle_topk": 2}, "tree verify"),
        ({"enable_deterministic_inference": True}, "needs a BITWISE spec"),
    ],
)
def test_server_setups_the_kernel_cannot_serve_are_refused(kw, expected):
    _, reasons = _reasons(_server_args(**kw))
    assert any(expected in r for r in reasons), reasons


@pytest.mark.parametrize(
    ("disable_overlap", "env", "required"),
    [
        # the scheduler stream only waits for an in-graph event: its kernels can hold SMs
        (False, {}, True),
        (False, {"SGLANG_FORCE_COARSE_WAR_BARRIER": "1"}, False),
        (True, {}, False),
        # the plan stream launches allocation kernels off both streams
        (False, {"SGLANG_FORCE_COARSE_WAR_BARRIER": "1", "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1"}, True),
        (True, {"SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1"}, True),
    ],
)
def test_canonical_placement_only_when_no_other_stream_runs_kernels(
    monkeypatch, disable_overlap, env, required
):
    for name in ("SGLANG_FORCE_COARSE_WAR_BARRIER", "SGLANG_ENABLE_OVERLAP_PLAN_STREAM"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    args = SimpleNamespace(disable_overlap_schedule=disable_overlap)
    monkeypatch.setattr(runtime, "get_server_args", lambda: args)
    assert runtime.placement_required() is required
