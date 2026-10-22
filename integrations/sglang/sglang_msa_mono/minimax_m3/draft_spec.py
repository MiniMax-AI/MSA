"""MiniMax-M3.1 DSpark draft on the MSA-v1 draft backbone: the monokernel spec.

SGLang routes the draft runner's one-request block step (seven rows: the bonus
token and six mask tokens) to ``draft_runtime.Built.forward``: the vocab-parallel
embedding, the five dense decoder layers and the final norm run as one
cooperative launch per rank, with both TP4 sums of every layer inside it. The
draft head (lm_head, Markov head, sampling) stays on SGLang's modules, and the
context K/V the target's hidden states project into the draft cache are still
written by SGLang's injector; the kernel writes only the block's own K/V.

The draft runner has no launch-time check of its own (``handle_monokernel``
validates the target's spec), so every check runs on the loaded draft model.
"""

from __future__ import annotations

from sglang.srt.monokernels.spec import (
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from sglang_msa_mono import SUPPORTED_MONOKERNEL_API_VERSION
from sglang_msa_mono.minimax_m3 import DRAFT_ROWS, TP


def _model_disable_reasons(model, hf_config, copies) -> list[str]:
    from sglang_msa_mono.minimax_m3.draft_runtime import model_disable_reasons

    return model_disable_reasons(model, hf_config, copies)


def _build(model, hf_config, ctx):
    from sglang_msa_mono.minimax_m3.draft_runtime import build

    return build(model, hf_config, ctx)


# The launch grid is one CTA per SM of whatever device it runs on, and the
# workspace refuses a device that cannot co-schedule it, so no SM floor applies.
_LIMITS = DeviceLimits(widths=(DRAFT_ROWS,), tp_sizes=(TP,), resident_blocks=1)

SPEC = MonoSpec(
    name="minimax_m3_dspark_draft_backbone",
    devices={"sm_100": _LIMITS, "sm_103": _LIMITS},
    features=frozenset({Feature.SPECULATIVE_DECODING, Feature.PD_DISAGGREGATION}),
    requires=Requirements(dtypes=frozenset({"bfloat16"})),
    # MXFP8 activations requantized before every GEMM, FP32 accumulation and
    # rank-ordered TP4 sums: SGLang's quantization points, a different order
    numerics=Numerics(
        kind=NumericsKind.TOLERANCE,
        bounds={"backbone.hidden.cosine": 0.99},
    ),
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=SUPPORTED_MONOKERNEL_API_VERSION,
    # the draft block is one row short of the target's verify window
    tokens_per_request=lambda model: DRAFT_ROWS,
)
