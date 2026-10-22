"""Compose separately selected Attention and MoE backends."""


class ModulePipeline:
    def __init__(self, attention, moe):
        self.attention = attention
        self.moe = moe

    def prepare(self, **attention_inputs):
        from ..plan import DecodeLayerPlan

        attention_plan = self.attention.prepare(**attention_inputs)
        q_len = attention_plan.q_len
        moe_plan = self.moe.prepare(
            q_len=q_len,
            hidden=attention_plan.buffers.post_attention_norm[-q_len:],
            residual=attention_plan.buffers.attention_residual[-q_len:],
        )
        return DecodeLayerPlan(
            run_attention=attention_plan.run,
            run_moe=moe_plan.run,
        )
