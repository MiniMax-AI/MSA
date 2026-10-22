"""The DSpark draft forward of MiniMax-M3.1 on the MSA-v1 draft backbone.

One routed step is one request's seven-row draft block. The kernel reads
SGLang's parameters in place (checkpoint-layout MXFP8 weights and row-major
UE8M0 scales, raw Gemma norm weights, the target's vocab shard of the
embedding, the fp32 RoPE table) and writes the block's K/V into SGLang's BF16
draft cache. Its plan binds every operand once, at the first routed forward,
which every rank runs eagerly together: the draft runner attaches the target's
embedding and sizes the KV cache after the build.

Every step, captured or not, copies its ids, positions, slots, page table and
KV length into the plan's persistent inputs with two launches; inside a graph
the sources are the runner's static buffers and SGLang's request table.
"""

from __future__ import annotations

import logging

import torch

from sglang_msa_mono.minimax_m3 import (
    DRAFT_FFN_INTER,
    DRAFT_LAYERS,
    DRAFT_QKV_ROWS,
    DRAFT_ROWS,
    DRAFT_WINDOW,
    HEAD_DIM,
    HIDDEN,
    O_PROJ_K,
    PAGE,
    ROTARY_DIM,
    TP,
)
from sglang_msa_mono.minimax_m3.prepare import prepare_step_inputs
from sglang_msa_mono.minimax_m3.weights import SCALE_BLOCK, mxfp8_quant_method

logger = logging.getLogger(__name__)

# The draft text config fields the kernel bakes in (checked against the loaded
# model's config, which SGLang derives from the checkpoint's dspark fields).
DRAFT_CONFIG = {
    "hidden_size": HIDDEN,
    "head_dim": HEAD_DIM,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "rotary_dim": ROTARY_DIM,
    "num_hidden_layers": DRAFT_LAYERS,
    "dense_intermediate_size": DRAFT_FFN_INTER * TP,
    "hidden_act": "swigluoai",
    "swiglu_alpha": 1.702,
    "swiglu_limit": 7.0,
    "use_gemma_norm": True,
    "qk_norm_type": "per_head",
    "sliding_window": DRAFT_WINDOW,
}
# (name, rows, K) of every MXFP8 linear of one layer, per TP4 rank
_LINEARS = (
    ("qkv_proj", DRAFT_QKV_ROWS, HIDDEN),
    ("o_proj", HIDDEN, O_PROJ_K),
    ("gate_up_proj", 2 * DRAFT_FFN_INTER, HIDDEN),
    ("down_proj", HIDDEN, DRAFT_FFN_INTER),
)


def _tp():
    from sglang.srt.runtime_context import get_parallel

    return get_parallel()


def _gather_tp(obj):
    import torch.distributed as dist

    group = _tp().tp_group
    out = [None] * group.world_size
    dist.all_gather_object(out, obj, group=group.cpu_group)
    return out


def _row_alignment() -> int:
    """The row multiple a captured draft step pads to: SGLang captures only the
    batch sizes whose rows divide by it, and pads every step up to one of them."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.runtime_context import get_exec
    from sglang.srt.utils.common import (
        get_cuda_graph_batch_size_alignment,
        require_attn_tp_gather,
    )

    graph = get_exec().graph.cuda_graph_config
    if graph is None or graph.decode.backend == Backend.DISABLED:
        return 1  # eager steps run at their own rows
    # the dense DSpark draft is DP-local, so only the attention-TP gather applies
    return get_cuda_graph_batch_size_alignment(gathered_buffer_required=require_attn_tp_gather())


def _decoder(layer: torch.nn.Module) -> torch.nn.Module:
    """``DSparkMiniMaxLayer`` wraps the ``MiniMaxM3DecoderLayer`` it runs."""
    return layer.decoder_layer


def _linear(module: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """A linear's checkpoint-layout MXFP8 weight and row-major UE8M0 scales."""
    return module.weight.data, module.weight_scale_inv.data


def _linear_reasons(module: torch.nn.Module, what: str, rows: int, k: int) -> list[str]:
    method = mxfp8_quant_method(module)
    backend = getattr(method, "mxfp8_dense_backend", None)
    if backend is None or not getattr(method, "use_mxfp8", False):
        return [f"{what} is not an MXFP8 linear"]
    if backend.is_flashinfer_trtllm():
        return [
            "--fp8-gemm-runner-backend flashinfer_trtllm shuffles MXFP8 weights in "
            "place; the kernel reads checkpoint layout"
        ]
    weight, scale = module.weight, getattr(module, "weight_scale_inv", None)
    if getattr(module, "bias", None) is not None:
        return [f"{what} has a bias"]
    if (
        weight.dtype != torch.float8_e4m3fn
        or scale is None
        or scale.dtype != torch.uint8
        or tuple(weight.shape) != (rows, k)
        or tuple(scale.shape) != (rows, k // SCALE_BLOCK)
        or not weight.is_contiguous()
        or not scale.is_contiguous()
    ):
        return [
            f"{what}: weight {weight.dtype} {tuple(weight.shape)}, scale "
            f"{None if scale is None else (scale.dtype, tuple(scale.shape))} is not this "
            f"rank's contiguous e4m3 [{rows}, {k}] with uint8 [{rows}, {k // SCALE_BLOCK}]"
        ]
    return []


def _rope_reasons(attn: torch.nn.Module, what: str) -> list[str]:
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    rope = attn.rotary_emb
    if type(rope) is not RotaryEmbedding:
        return [f"{what}: {type(rope).__name__} is not plain RoPE"]
    table = rope.cos_sin_cache
    if (
        not rope.is_neox_style
        or rope.rotary_dim != ROTARY_DIM
        or getattr(rope, "position_start", 0) != 0
        or table.dtype != torch.float32
        or table.dim() != 2
        or table.shape[1] != ROTARY_DIM
    ):
        return [f"{what}: the kernel reads one fp32 neox RoPE table of {ROTARY_DIM} dims"]
    return []


def model_disable_reasons(model, hf_config, copies) -> list[str]:
    from sglang.srt.environ import envs
    from sglang.srt.speculative.dspark_components.dspark_config import (
        get_dspark_sample_from_anchor,
    )

    config = model.config
    reasons = [
        f"config.{key} {getattr(config, key, None)!r} is not {want!r} (the kernel's fixed shape)"
        for key, want in DRAFT_CONFIG.items()
        if getattr(config, key, None) != want
    ]
    layer_types = (getattr(config, "dflash_config", None) or {}).get("layer_types") or []
    if layer_types != ["sliding_attention"] * DRAFT_LAYERS:
        reasons.append(f"draft layer types {layer_types}: the kernel runs sliding attention")
    if getattr(config, "rope_scaling", None):
        reasons.append(f"config.rope_scaling {config.rope_scaling!r}: the kernel reads plain RoPE")
    if not get_dspark_sample_from_anchor(hf_config):
        reasons.append(
            f"sample_from_anchor is off: the draft block has {DRAFT_ROWS + 1} rows, the "
            f"kernel {DRAFT_ROWS}"
        )
    if not envs.SGLANG_DSPARK_EMBED_IN_GRAPH.get():
        reasons.append("SGLANG_DSPARK_EMBED_IN_GRAPH=0 passes embeddings; the kernel embeds ids")
    # bfloat16 weights, cache and embedding are checked at binding and by the kernel
    parallel = _tp()
    if parallel.tp_size != TP or parallel.attn_tp_size != TP:
        reasons.append(
            f"tp {parallel.tp_size} / attention tp {parallel.attn_tp_size}: the kernel runs TP{TP}"
        )
    alignment = _row_alignment()
    if DRAFT_ROWS % alignment:
        # rows padded to attention TP put one request's block in a 4-request graph
        reasons.append(
            f"draft steps pad to a multiple of {alignment} rows, so a {DRAFT_ROWS}-row block "
            "never runs alone; with an a2a MoE backend (e.g. megamoe) pass "
            "--disable-attn-tp-gather"
        )
    layers = getattr(model, "layers", None)
    if layers is None or len(layers) != DRAFT_LAYERS:
        reasons.append(f"the draft model does not have {DRAFT_LAYERS} decoder layers")
        return reasons
    for i, layer in enumerate(layers):
        decoder = _decoder(layer)
        attn, mlp = decoder.self_attn, decoder.mlp
        what = f"draft layer {i}"
        if getattr(decoder, "is_layer_sparse", False) or not hasattr(mlp, "gate_up_proj"):
            reasons.append(f"{what} is not a dense MLP layer")
            continue
        for owner, (name, rows, k) in zip((attn, attn, mlp, mlp), _LINEARS):
            reasons += _linear_reasons(getattr(owner, name), f"{what} {name}", rows, k)
        if attn.num_heads * HEAD_DIM + 2 * attn.num_kv_heads * HEAD_DIM != DRAFT_QKV_ROWS:
            reasons.append(f"{what}: {attn.num_heads} Q / {attn.num_kv_heads} KV heads a rank")
        if attn.attn.sliding_window_size != DRAFT_WINDOW - 1:
            reasons.append(
                f"{what}: window_left {attn.attn.sliding_window_size} is not {DRAFT_WINDOW - 1}"
            )
        reasons += _rope_reasons(attn, what)
    tables = {layer.self_attn.rotary_emb.cos_sin_cache.data_ptr() for layer in layers}
    if len(tables) != 1:
        reasons.append("the draft layers do not share one RoPE table")
    if getattr(model, "is_nemotron_35_draft", False) or getattr(
        model, "uses_own_vocab_modules", False
    ):
        reasons.append("the draft does not embed with the target's vocab-parallel embedding")
    return reasons


class _Inputs:
    """The plan's persistent step inputs, rewritten in place before every run."""

    def __init__(self, device: torch.device, pages: int):
        self.input_ids = torch.zeros(DRAFT_ROWS, dtype=torch.int64, device=device)
        self.positions = torch.zeros(DRAFT_ROWS, dtype=torch.int64, device=device)
        self.slot_mapping = torch.full((DRAFT_ROWS,), -1, dtype=torch.int64, device=device)
        self.page_table = torch.zeros(pages, dtype=torch.int32, device=device)
        self.kv_len = torch.full((1,), DRAFT_ROWS, dtype=torch.int32, device=device)

    def prepare(self, input_ids, forward_batch, req_to_token: torch.Tensor) -> None:
        self.input_ids.copy_(input_ids)
        prepare_step_inputs(
            seq_lens=forward_batch.seq_lens,
            req_pool_indices=forward_batch.req_pool_indices,
            req_to_token=req_to_token,
            positions=forward_batch.positions,
            out_cache_loc=forward_batch.out_cache_loc,
            kv_len=self.kv_len,
            kv_indices=self.page_table,
            positions_out=self.positions,
            slot_mapping=self.slot_mapping,
            page=PAGE,
            rows=DRAFT_ROWS,
        )


class Built:
    """One rank's TP4 context, the plan bound at the first forward, and the forward."""

    def __init__(self, model, hf_config, ctx):
        from inference.high_tps.msa_v1.draft import create_draft_tp_context

        self.model = model
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.eps = float(model.config.rms_norm_eps)
        self.context = create_draft_tp_context(_tp().tp_group.cpu_group)
        self.plan = None
        self.inputs: _Inputs | None = None
        self.req_to_token = None
        logger.info(
            "msa-v1 draft backbone: %d dense layers, one cooperative launch a rank",
            DRAFT_LAYERS,
        )

    def _bind_reasons(self, pool, embed) -> list[str]:
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        reasons = []
        if type(pool) is not MHATokenToKVPool:
            reasons.append(
                f"the draft KV pool {type(pool).__name__} is not SGLang's plain MHA pool "
                "(fused or quantized draft caches are not supported)"
            )
        elif pool.dtype != torch.bfloat16 or pool.page_size != PAGE:
            reasons.append(
                f"the draft KV pool is {pool.dtype} with {pool.page_size}-token pages, "
                f"not bfloat16 with {PAGE}"
            )
        else:
            for layer in self.model.layers:
                layer_id = layer.self_attn.attn.layer_id
                for side in (pool.get_key_buffer(layer_id), pool.get_value_buffer(layer_id)):
                    if side.shape[1:] != (1, HEAD_DIM) or not side.is_contiguous():
                        reasons.append(
                            f"draft cache of layer {layer_id} is {tuple(side.shape)}, not a "
                            f"contiguous [slots, 1, {HEAD_DIM}]"
                        )
        shard = getattr(embed, "shard_indices", None)
        if embed is None or shard is None:
            reasons.append("the target's vocab-parallel embedding is not attached")
        elif shard.num_added_elements or embed.weight.dtype != torch.bfloat16:
            reasons.append("the embedding has added vocabulary or is not bfloat16")
        return reasons

    def _bind(self) -> None:
        """Check, agree on and bind the KV cache, embedding and weights."""
        from inference.high_tps.msa_v1.draft import (
            DraftBackboneInputs,
            DraftBackbonePlan,
            DraftBackboneWeights,
            DraftLayerWeights,
        )
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        pool = getattr(backend, "token_to_kv_pool", None)
        embed = self.model.embed_tokens
        reasons = self._bind_reasons(pool, embed)
        bad = [f"rank {r}: {why}" for r, rs in enumerate(_gather_tp(reasons)) for why in rs]
        if bad:
            raise ValueError("msa-v1 draft backbone: " + "; ".join(bad))
        self.req_to_token = backend.req_to_token_pool.req_to_token
        self.inputs = _Inputs(self.device, -(-int(self.req_to_token.shape[1]) // PAGE))
        layers, k_caches, v_caches = [], [], []
        for layer in self.model.layers:
            decoder = _decoder(layer)
            attn, mlp = decoder.self_attn, decoder.mlp
            qkv, qkv_sf = _linear(attn.qkv_proj)
            o, o_sf = _linear(attn.o_proj)
            gate_up, gate_up_sf = _linear(mlp.gate_up_proj)
            down, down_sf = _linear(mlp.down_proj)
            layers.append(
                DraftLayerWeights(
                    input_norm=decoder.input_layernorm.weight.data,
                    post_norm=decoder.post_attention_layernorm.weight.data,
                    q_norm=attn.q_norm.weight.data,
                    k_norm=attn.k_norm.weight.data,
                    qkv=qkv,
                    qkv_sf=qkv_sf,
                    o=o,
                    o_sf=o_sf,
                    gate_up=gate_up,
                    gate_up_sf=gate_up_sf,
                    down=down,
                    down_sf=down_sf,
                )
            )
            layer_id = attn.attn.layer_id
            k_caches.append(pool.get_key_buffer(layer_id).view(-1, HEAD_DIM))
            v_caches.append(pool.get_value_buffer(layer_id).view(-1, HEAD_DIM))
        shard = embed.shard_indices
        weights = DraftBackboneWeights(
            layers=layers,
            embed=embed.weight.data[: shard.num_org_elements],
            vocab_start=int(shard.org_vocab_start_index),
            final_norm=self.model.norm.weight.data,
            eps=self.eps,
        )
        inputs = DraftBackboneInputs(
            input_ids=self.inputs.input_ids,
            positions=self.inputs.positions,
            slot_mapping=self.inputs.slot_mapping,
            page_table=self.inputs.page_table,
            kv_len=self.inputs.kv_len,
            cos_sin=self.model.layers[0].self_attn.rotary_emb.cos_sin_cache,
            k_caches=k_caches,
            v_caches=v_caches,
        )
        out = torch.empty((DRAFT_ROWS, HIDDEN), dtype=torch.bfloat16, device=self.device)
        self.plan = DraftBackbonePlan(self.context, weights, inputs, out)
        logger.info(
            "msa-v1 draft backbone bound: vocab rows [%d, %d), %d pages a request",
            weights.vocab_start,
            weights.vocab_start + weights.embed.shape[0],
            self.inputs.page_table.numel(),
        )

    def forward(self, plan, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        """The draft model's forward for a routed step: the backbone's final-normed
        hidden states, which SGLang's draft head samples from."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        if input_embeds is not None:
            raise RuntimeError("msa-v1 draft backbone: the kernel embeds ids, got embeddings")
        if self.plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("msa-v1 draft backbone: the plan was not bound before capture")
            self._bind()
        self.inputs.prepare(input_ids, forward_batch, self.req_to_token)
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=self.plan.run())


def build(model, hf_config, ctx) -> Built:
    return Built(model, hf_config, ctx)
