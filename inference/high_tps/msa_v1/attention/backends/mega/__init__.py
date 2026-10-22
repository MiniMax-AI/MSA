"""Existing one-kernel MegaAttention backend adapter."""

from ...api import m3_decode_mega_atten


class MegaBackend:
    def prepare(self, plan) -> None:
        pass

    def run(self, plan) -> None:
        weights = plan.runner.weights
        config = plan.runner.config
        cache = plan.cache
        buffers = plan.buffers
        m3_decode_mega_atten(
            buffers.q,
            cache.k,
            cache.v,
            plan.plan_info,
            kv_indices=cache.kv_indices,
            k_scale_tensor=cache.k_scale,
            v_scale_tensor=cache.v_scale,
            output_projection_weight=weights.output_projection,
            output_projection_weight_sf=weights.output_projection_sf,
            tp_workspace=plan.runner.tp_context.workspace,
            residual_in=plan.layer_input,
            rms_weight=weights.post_attention_norm,
            indexer_q=buffers.qi,
            indexer_k=cache.ki,
            indexer_k_scale=cache.ki_scale,
            indexer_page_table=cache.indexer_page_table,
            attention_projection_input_norm_weight=weights.input_norm,
            attention_projection_weight=weights.projection,
            attention_projection_weight_sf=weights.projection_sf,
            attention_projection_workspace_data=buffers.projection_data,
            attention_projection_workspace_scale=buffers.projection_scale,
            attention_projection_grid_epoch=buffers.projection_grid_epoch,
            attention_projection_special_partial=buffers.projection_special_partial,
            attention_projection_special_epochs=buffers.projection_special_epochs,
            attention_projection_positions=plan.positions,
            attention_projection_slot_mapping=plan.slot_mapping,
            attention_projection_q_norm_weight=weights.q_norm,
            attention_projection_k_norm_weight=weights.k_norm,
            attention_projection_qi_norm_weight=weights.qi_norm,
            attention_projection_ki_norm_weight=weights.ki_norm,
            attention_projection_cos_sin_cache=plan.cos_sin_cache,
            attention_residual=buffers.attention_residual,
            post_attention_norm=buffers.post_attention_norm,
            kv_block_indexes=buffers.kv_block_indexes,
            out=buffers.output,
            out_sf=buffers.output_scale,
            topk=config.topk,
            sm_scale=config.sm_scale,
            k_scale=config.k_dequant_scale,
            v_scale=config.v_dequant_scale,
            rms_eps=config.post_attention_norm_eps,
            attention_projection_input_norm_eps=config.input_norm_eps,
            attention_projection_post_norm_eps=config.head_norm_eps,
            attention_projection_ki_dequant_scale=config.ki_dequant_scale,
            wait_timeout_s=config.wait_timeout_s,
            enable_placement=config.enable_placement,
        )
