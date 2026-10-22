"""SGLang monokernel plugin for MSA-v1 MegaAttention and the token-parallel MoE kernel.

SGLang holds the interface (``sglang.srt.monokernels``); this package holds the
MiniMax-M3.1 spec and its forward (``minimax_m3``). SGLang loads the package as
a general plugin (the ``sglang.srt.plugins`` entry point in ``pyproject.toml``),
which registers the spec for ``--monokernel-backend msa``.
"""

__version__ = "0.1.0"

# The --monokernel-backend name this package registers
BACKEND = "msa"
SUPPORTED_MONOKERNEL_API_VERSION = 1
# Each spec by the model's HF architecture; a spec module imports only when a
# server runs this backend
SPECS = {
    # the released MiniMax-M3 / M3.1 checkpoints (MiniMax-M3.1-Flash among them)
    "MiniMaxM3SparseForConditionalGeneration": "sglang_msa_mono.minimax_m3.spec:SPEC",
    "MiniMaxM3TinyForConditionalGeneration": "sglang_msa_mono.minimax_m3.spec:SPEC",
    "MiniMaxM3SparseForCausalLM": "sglang_msa_mono.minimax_m3.spec:SPEC",
    "MiniMaxM3TinyForCausalLM": "sglang_msa_mono.minimax_m3.spec:SPEC",
    # the DSpark draft that SGLang derives from the M3.1 checkpoints' dspark fields
    "DSparkMiniMaxDraftModel": "sglang_msa_mono.minimax_m3.draft_spec:SPEC",
}


def register_specs() -> None:
    """The plugin entry point, which SGLang runs in every process."""
    from sglang.srt.monokernels.spec import MONOKERNEL_API_VERSION, register

    if MONOKERNEL_API_VERSION != SUPPORTED_MONOKERNEL_API_VERSION:
        raise RuntimeError(
            f"msa requires monokernel API {SUPPORTED_MONOKERNEL_API_VERSION}; "
            f"installed SGLang provides {MONOKERNEL_API_VERSION}"
        )

    for architecture, path in SPECS.items():
        register(backend=BACKEND, architecture=architecture, path=path)
