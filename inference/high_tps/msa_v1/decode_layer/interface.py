"""Compose the TP4 MegaAttention and MegaMoE runners."""


def create_decode_layer_runner(*, structure="module-pipeline", attention, moe):
    """Bind prepared runners; each layer call launches two kernels at Q=8."""
    if structure != "module-pipeline":
        raise ValueError("structure must be 'module-pipeline'")
    from ..attention.interface import AttentionRunner
    from ..moe.interface import MoeRunner
    from .backends.module_pipeline import ModulePipeline

    if not isinstance(attention, AttentionRunner) or not isinstance(moe, MoeRunner):
        raise TypeError("module-pipeline requires AttentionRunner and MoeRunner")
    if attention.device != moe.device:
        raise ValueError("Attention and MoE runners must use the same device")
    return ModulePipeline(attention, moe)
