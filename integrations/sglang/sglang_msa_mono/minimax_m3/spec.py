"""MiniMax-M3.1 verify on MSA-v1 MegaAttention + token-parallel MoE: the monokernel spec.

SGLang routes one request's eight-token DSpark/EAGLE chain verify step to
``runtime.Built.forward``; every other step, including plain decode, takes the
standard path. The embedding and the dense-MLP layers 0-2 run SGLang's modules;
every MoE layer is two launches (MegaAttention, then the token-parallel MoE,
each with its TP4 all-reduce in an NVLS workspace). SGLang's final norm and
logits processor close the step.

MegaAttention writes K/V/index-K for its rows into SGLang's MSA-v1 NVFP4 pool
in place (page-strided views), encoding them as the standard path does, so
prefill-written and verify-written pages read alike.
"""

from __future__ import annotations

import logging

from sglang.srt.monokernels.spec import (
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from sglang_msa_mono import SUPPORTED_MONOKERNEL_API_VERSION
from sglang_msa_mono.minimax_m3 import (
    GRID_CTAS,
    HEAD_DIM,
    HIDDEN,
    MAX_CONTEXT,
    PAGE,
    ROWS,
    TP,
)

logger = logging.getLogger(__name__)

# The text config fields the kernel bakes in (checked against the HF config).
TEXT_CONFIG = {
    "hidden_size": HIDDEN,
    "head_dim": HEAD_DIM,
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
}
# sparse_attention_config fields, as SGLang's MSA-v1 backend reads them
SPARSE_CONFIG = {
    "sparse_index_dim": 128,
    "sparse_num_index_heads": 4,
    "sparse_topk_blocks": 16,
    "sparse_block_size": 128,
    "sparse_local_block": 1,
    "sparse_init_block": 0,
    "sparse_score_type": "max",
}
# sparse_attention_config switches the kernel does not implement
SPARSE_OFF = (
    "sparse_enable_kv_share",
    "sparse_enable_vo_rope",
    "sparse_kvshare_value_drop_rope_dim",
    "sparse_indexer_lookahead_num_layers",
)
# MXFP8 GEMM backends that keep each linear's checkpoint-layout weight
# (flashinfer_trtllm shuffles it in place); "auto" resolves to one of them on SM100.
MXFP8_GEMM_BACKENDS = frozenset({"auto", "deep_gemm", "flashinfer_cutlass", "flashinfer_cutedsl"})


def text_config(hf_config):
    return getattr(hf_config, "text_config", None) or hf_config


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    """Checks ``Requirements`` cannot state: nested HF fields and derived values."""
    reasons = []
    text = text_config(hf_config)
    for key, want in TEXT_CONFIG.items():
        value = getattr(text, key, None)
        if value != want:
            reasons.append(f"config.{key} {value!r} is not {want!r} (the kernel's fixed shape)")
    if getattr(text, "rope_scaling", None):
        reasons.append(f"config.rope_scaling {text.rope_scaling!r}: the kernel reads plain RoPE")
    sparse = getattr(text, "sparse_attention_config", None) or {}
    for key, want in SPARSE_CONFIG.items():
        if sparse.get(key) != want:
            reasons.append(f"sparse_attention_config.{key} {sparse.get(key)!r} is not {want!r}")
    for key in SPARSE_OFF:
        if sparse.get(key):
            reasons.append(f"sparse_attention_config.{key} is not supported")
    if sparse.get("sparse_index_compress_type") not in (None, "none"):
        reasons.append("MSA-v1 index compression is not supported")
    # the kernels run only the MoE layers; dense layers stay on SGLang's modules
    moe = getattr(text, "moe_layer_freq", None) or []
    freq = sparse.get("sparse_attention_freq") or [0] * len(moe)
    if not any(moe):
        reasons.append("config.moe_layer_freq has no MoE layer")
    elif len(freq) != len(moe) or not all(f for f, m in zip(freq, moe) if m):
        reasons.append("every MoE layer must be an MSA sparse attention layer")
    if cfg.quantization not in (None, "minimax_m3_nvfp4"):
        reasons.append(
            f"--quantization {cfg.quantization!r}: the kernel reads minimax_m3_nvfp4 "
            "checkpoints (MXFP8 attention, NVFP4 experts)"
        )
    if cfg.page_size not in (None, PAGE):
        reasons.append(f"--page-size {cfg.page_size} is not {PAGE}")
    if cfg.fp8_gemm_runner_backend not in MXFP8_GEMM_BACKENDS:
        reasons.append(
            f"--fp8-gemm-runner-backend {cfg.fp8_gemm_runner_backend!r} rewrites MXFP8 "
            f"weights in place; use one of {sorted(MXFP8_GEMM_BACKENDS)}"
        )
    context = cfg.context_length or getattr(text, "max_position_embeddings", 0)
    if context and context > MAX_CONTEXT:
        reasons.append(
            f"context length {context} exceeds the kernel's {MAX_CONTEXT} tokens; "
            f"pass --context-length <= {MAX_CONTEXT}"
        )
    # the token-parallel MoE kernel runs exactly eight rows: one request's chain verify
    if not cfg.speculative_algorithm:
        reasons.append(
            "the kernels serve only speculative verify steps; pass --speculative-algorithm "
            f"with --speculative-num-draft-tokens {ROWS}"
        )
    elif int(cfg.speculative_num_draft_tokens or 0) != ROWS:
        reasons.append(
            f"--speculative-num-draft-tokens {cfg.speculative_num_draft_tokens} is not "
            f"{ROWS} (the token-parallel MoE kernel's fixed rows)"
        )
    if (cfg.speculative_eagle_topk or 1) > 1:
        reasons.append("tree verify (--speculative-eagle-topk > 1) is not supported")
    return reasons


def _model_disable_reasons(model, hf_config, copies) -> list[str]:
    from sglang_msa_mono.minimax_m3.runtime import model_disable_reasons

    return model_disable_reasons(model, hf_config, copies)


def _build(model, hf_config, ctx):
    from sglang_msa_mono.minimax_m3.runtime import build

    return build(model, hf_config, ctx)


def _load_tensor(copies, model, name, tensor):
    from sglang_msa_mono.minimax_m3.runtime import load_tensor

    load_tensor(copies, model, name, tensor)


# one request's eight verify rows: a width of 8 at 8 tokens a request is batch size 1
_LIMITS = DeviceLimits(widths=(ROWS,), tp_sizes=(TP,), resident_blocks=GRID_CTAS)

SPEC = MonoSpec(
    name="minimax_m3_msa_v1",
    devices={"sm_100": _LIMITS, "sm_103": _LIMITS},
    # DSpark/EAGLE verify of one request, also on a PD decode server. M3.1 NVFP4
    # forces megamoe (EP = TP); the kernels read their own experts, captured from
    # the checkpoint before EP filtering, and either shared-expert layout
    features=frozenset(
        {Feature.SPECULATIVE_DECODING, Feature.PD_DISAGGREGATION, Feature.EXPERT_PARALLEL}
    ),
    requires=Requirements(dtypes=frozenset({"bfloat16"})),
    # FP8 attention activations, NVFP4 experts and a fused TP4 all-reduce: the
    # same quantization points as SGLang's path, in a different order
    numerics=Numerics(
        kind=NumericsKind.TOLERANCE,
        bounds={"layer.next_hidden.cosine": 0.999},
    ),
    model_disable_reasons=_model_disable_reasons,
    config_disable_reasons=_config_disable_reasons,
    build=_build,
    api_version=SUPPORTED_MONOKERNEL_API_VERSION,
    load_tensor=_load_tensor,
)
