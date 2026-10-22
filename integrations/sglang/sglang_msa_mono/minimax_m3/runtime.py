"""The routed verify forward of MiniMax-M3.1 on MSA-v1 MegaAttention + token-parallel MoE.

One routed step is one request's eight DSpark/EAGLE verify rows, the kernels'
eight physical rows:

1. Outside any graph, where the attention backend prepares its own metadata
   (``init_forward_metadata_out_graph``, before every capture and replay; eager
   steps prepare in the forward): the step's KV length, page table, positions
   and slots are copied into the binding's persistent inputs.
2. In the forward: the embedding and the dense-MLP layers run SGLang's modules;
   their residual stream ``hidden + residual`` is copied into the first MoE
   layer's input. Every MoE layer is two launches: MegaAttention (its TP4
   all-reduce in the attention NVLS workspace) writes the attention residual
   and post-attention norm, then the token-parallel MoE writes the next residual
   stream into the MoE context's workspace, which the next layer's MegaAttention
   reads in place. SGLang's final norm and logits processor close the step.

The two contexts own separate workspaces, so a layer's attention never
overwrites the MoE output it reads.

Plans reserve the page table's full capacity. MegaAttention reads the runtime
length from ``plan_info.kv_segment_lens`` every step, using the same compiled
kernel and monotonically advancing workspace generation across Graph replays.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

from sglang_msa_mono.minimax_m3 import (
    HEAD_DIM,
    HIDDEN,
    INDEXER_MAX_PAGES,
    LOCAL_Q_HEADS,
    PAGE,
    ROWS,
    TOPK_BLOCKS,
)
from sglang_msa_mono.minimax_m3.prepare import prepare_step_inputs
from sglang_msa_mono.minimax_m3.weights import (
    LayerOperands,
    RoutedExperts,
    allocate_layer,
    fill_from_module,
    moe_weights,
    mxfp8_modules,
    mxfp8_quant_method,
    parse_expert_name,
)

logger = logging.getLogger(__name__)

SCALE_BLOCK = 32
_BF16_EXPERT_SUFFIXES = (".w1.weight", ".w2.weight", ".w3.weight")


def _language_model(model: torch.nn.Module) -> torch.nn.Module:
    """``MiniMaxM3Model``, ``model.model`` of the causal-LM and VL wrappers alike."""
    return model.model


def _moe_layer_ids(lm: torch.nn.Module) -> list[int]:
    return [
        i
        for i in range(lm.start_layer, lm.end_layer)
        if getattr(lm.layers[i], "is_layer_sparse", False)
    ]


def _tp():
    from sglang.srt.runtime_context import get_parallel

    return get_parallel()


def get_server_args():
    from sglang.srt.runtime_context import get_server_args

    return get_server_args()


def _gather_tp(obj):
    group = _tp().tp_group
    out = [None] * group.world_size
    dist.all_gather_object(out, obj, group=group.cpu_group)
    return out


def model_disable_reasons(model, hf_config, copies) -> list[str]:
    lm = _language_model(model)
    layers = _moe_layer_ids(lm)
    if not layers:
        return ["the model has no MoE decoder layer"]
    reasons = []
    if layers != list(range(layers[0], lm.end_layer)):
        reasons.append(f"MoE layers {layers} are not one tail of the decoder")
    if getattr(model, "is_mrope_enabled", False):
        reasons.append("M-RoPE positions are not supported")
    for i in layers:
        layer = lm.layers[i]
        attn, mlp = layer.self_attn, layer.mlp
        if not attn.is_sparse_attention_layer:
            reasons.append(f"layer {i} is not an MSA sparse attention layer")
            continue
        if attn.index_k_norm is None or not attn.disable_index_value:
            reasons.append(f"layer {i}: the kernel reads a K-only index with its own norm")
            continue
        if attn.idx_head_dim != attn.head_dim or not attn.rotary_emb.is_neox_style:
            reasons.append(f"layer {i}: the kernel applies one neox RoPE table to K and index K")
        if mlp.num_fused_shared_experts or mlp.shared_experts is None:
            reasons.append(f"layer {i}: the shared expert is not a separate MLP")
            continue
        for module in mxfp8_modules(layer):
            backend = getattr(mxfp8_quant_method(module), "mxfp8_dense_backend", None)
            if backend is None:
                reasons.append(f"layer {i}: {type(module).__name__} is not an MXFP8 linear")
            elif backend.is_flashinfer_trtllm():
                reasons.append(
                    "--fp8-gemm-runner-backend flashinfer_trtllm shuffles MXFP8 weights "
                    "in place; the kernels read checkpoint layout"
                )
                return reasons
    return reasons


def load_tensor(copies: dict, model: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Stream routed experts into this rank's MoE slices; every other operand is
    read from the model after load.

    The experts' storage lives in ``copies`` (one entry set per layer). Loading
    finishes before runtime construction and Graph capture. The serving plugin
    rejects online weight updates.
    """
    parsed = parse_expert_name(name)
    if parsed is None:
        if ".experts." in name and ".dspark." not in name and name.endswith(_BF16_EXPERT_SUFFIXES):
            raise ValueError(
                f"{name}: a BF16 routed expert; the MSA-v1 MoE kernel reads NVFP4 "
                "experts (weight_packed / weight_scale / weight_global_scale)"
            )
        return
    layer, expert, proj, field = parsed
    device = torch.device("cuda", torch.cuda.current_device())
    experts = RoutedExperts.in_copies(copies, layer, device)
    experts.stage(expert, proj, field, tensor, _tp().tp_rank)


class _Binding:
    """The verify step's persistent inputs, scratch and per-layer plans."""

    def __init__(self, built: Built):
        from inference.high_tps.msa_v1.attention import (
            AttentionBuffers,
            AttentionCache,
            create_attention_plan,
        )

        device, pages = built.device, built.pages
        # Capacity reserves storage; each step publishes its actual device length.
        self.plan_info = create_attention_plan(
            torch.tensor([ROWS], dtype=torch.int32),
            torch.tensor([pages * PAGE], dtype=torch.int32),
            topk=TOPK_BLOCKS,
            device=device,
        )
        self.kv_len = self.plan_info.kv_segment_lens  # [1] int32, read by the kernel

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)

        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)

        e4m3 = torch.float8_e4m3fn
        self.positions = zeros(ROWS, torch.int64)
        self.slot_mapping = torch.full((ROWS,), -1, dtype=torch.int64, device=device)
        self.kv_indices = zeros((1, pages), torch.int32)
        self.layer_input = zeros((ROWS, HIDDEN), torch.bfloat16)
        # one launch's scratch, reused by every layer in stream order
        buffers = AttentionBuffers(
            q=empty((ROWS, LOCAL_Q_HEADS, HEAD_DIM), e4m3),
            qi=empty((ROWS, HEAD_DIM), e4m3),
            projection_data=empty(ROWS * HIDDEN, torch.uint8),
            projection_scale=empty(ROWS * HIDDEN // SCALE_BLOCK, torch.uint8),
            projection_grid_epoch=zeros(32, torch.int32),
            projection_special_partial=empty(4 * 4 * ROWS * HEAD_DIM, torch.float32),
            projection_special_epochs=zeros(4 * 32, torch.int32),
            kv_block_indexes=torch.full(
                (ROWS, 1, TOPK_BLOCKS), -1, dtype=torch.int32, device=device
            ),
            output=empty((ROWS, LOCAL_Q_HEADS, HEAD_DIM), e4m3),
            output_scale=empty(
                max(128 * ((LOCAL_Q_HEADS * HEAD_DIM // SCALE_BLOCK + 3) // 4) * 4, 16 * 512),
                torch.uint8,
            ),
            attention_residual=zeros((ROWS, HIDDEN), torch.bfloat16),
            post_attention_norm=zeros((ROWS, HIDDEN), torch.bfloat16),
        )
        self.output = built.moe_context.mega_workspace.next_hidden
        self.attention = []
        self.moe = []
        layer_input = self.layer_input
        for i in built.layer_ids:
            views = built.caches[i]
            cache = AttentionCache(
                k=views["k"],
                v=views["v"],
                ki=views["indexer_k"],
                k_scale=views["k_scale"],
                v_scale=views["v_scale"],
                ki_scale=views["indexer_k_scale"],
                kv_indices=self.kv_indices,
                indexer_page_table=self.kv_indices.view(-1),
            )
            self.attention.append(
                built.attention_runners[i].prepare(
                    q_len=ROWS,
                    layer_input=layer_input,
                    positions=self.positions,
                    slot_mapping=self.slot_mapping,
                    cos_sin_cache=built.cos_sin_cache,
                    cache=cache,
                    buffers=buffers,
                    plan_info=self.plan_info,
                )
            )
            self.moe.append(
                built.moe_runners[i].prepare(
                    q_len=ROWS,
                    hidden=buffers.post_attention_norm,
                    residual=buffers.attention_residual,
                )
            )
            # the next layer reads this layer's MoE output in place
            layer_input = self.output

    @torch.no_grad()
    def prepare(self, forward_batch, req_to_token: torch.Tensor) -> None:
        """Copy the step's length, page table, positions and slots in place."""
        prepare_step_inputs(
            seq_lens=forward_batch.seq_lens,
            req_pool_indices=forward_batch.req_pool_indices,
            req_to_token=req_to_token,
            positions=forward_batch.positions,
            out_cache_loc=forward_batch.out_cache_loc,
            kv_len=self.kv_len,
            kv_indices=self.kv_indices,
            positions_out=self.positions,
            slot_mapping=self.slot_mapping,
            page=PAGE,
            rows=ROWS,
        )


class Built:
    """One rank's operands, TP4 contexts, runners, KV binding and forward."""

    def __init__(self, model, hf_config, ctx):
        from inference.high_tps.msa_v1.attention import (
            AttentionConfig,
            create_attention_runner,
            create_attention_tp_context,
        )
        from inference.high_tps.msa_v1.moe import (
            MoeConfig,
            create_moe_runner,
            create_moe_tp_context,
        )

        self.model = model
        self.lm = _language_model(model)
        self.layer_ids = _moe_layer_ids(self.lm)
        self.first_moe = self.layer_ids[0]
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.rank = _tp().tp_rank
        text = getattr(hf_config, "text_config", None) or hf_config
        eps = float(text.rms_norm_eps)
        self.operands: dict[int, LayerOperands] = {
            i: allocate_layer(self.device) for i in self.layer_ids
        }
        self.experts = {
            i: RoutedExperts.in_copies(ctx.copies, i, self.device) for i in self.layer_ids
        }
        self._fill()
        if get_server_args().load_format != "dummy":
            missing = [
                f"layer {i} {what}" for i, e in self.experts.items() for what in e.missing()
            ]
            if missing:
                raise ValueError(
                    f"{len(missing)} routed-expert tensors never reached the load hook, "
                    f"e.g. {missing[:4]}"
                )
        attn = self.lm.layers[self.first_moe].self_attn
        # the kernel's table is bf16 [positions, cos 32 | sin 32]
        self.cos_sin_cache = attn.rotary_emb.cos_sin_cache.to(torch.bfloat16).contiguous()
        # the NVLS workspace names must be unique among this node's servers
        leader = _gather_tp(os.getpid())[0]
        group = _tp().tp_group.cpu_group
        self.attention_context = create_attention_tp_context(
            backend="mega", group=group, group_id=f"sglang-msa-v1-{leader}-attention"
        )
        self.moe_context = create_moe_tp_context(
            group=group, group_id=f"sglang-msa-v1-{leader}-moe", backend="mega"
        )
        # the MSA-v1 pool stores K/V/index K with unit outer scales
        attention_config = AttentionConfig(
            enable_placement=True,
            sm_scale=HEAD_DIM**-0.5,
            input_norm_eps=eps,
            head_norm_eps=eps,
            post_attention_norm_eps=eps,
        )
        moe_config = MoeConfig(routed_scaling_factor=float(text.routed_scaling_factor))
        self.attention_runners = {
            i: create_attention_runner(
                weights=self.operands[i].attention_weights(),
                config=attention_config,
                tp_context=self.attention_context,
                backend="mega",
            )
            for i in self.layer_ids
        }
        self.moe_runners = {
            i: create_moe_runner(
                weights=moe_weights(self.operands[i], self.experts[i]),
                config=moe_config,
                tp_context=self.moe_context,
                backend="mega",
            )
            for i in self.layer_ids
        }
        self.binding: _Binding | None = None  # bound at the first forward
        self.caches: dict[int, dict] | None = None
        self.req_to_token = None
        self.pages = 0
        self._hooked: set[int] = set()
        logger.info(
            "msa-v1 MegaAttention + token-parallel MoE: MoE layers %d-%d, weights "
            "%.1f GiB (routed experts %.1f GiB)",
            self.layer_ids[0],
            self.layer_ids[-1],
            sum(op.nbytes() for op in self.operands.values()) / 2**30,
            sum(e.nbytes() for e in self.experts.values()) / 2**30,
        )

    # ----- weights -----------------------------------------------------------

    @torch.no_grad()
    def _fill(self) -> None:
        """Rebuild every operand and alpha in place from the model and the experts."""
        reasons = []
        for i in self.layer_ids:
            fill_from_module(self.operands[i], self.lm.layers[i], rank=self.rank)
            experts = self.experts[i]
            reasons += [f"layer {i} {why}" for why in experts.scale_reasons()]
            experts.refresh_alphas()
        if reasons:
            raise ValueError(
                "routed-expert scales the MSA-v1 MoE kernel cannot express: "
                + "; ".join(reasons[:8])
            )

    # ----- binding -----------------------------------------------------------

    def _bind(self) -> None:
        """At the first routed forward, which every rank runs eagerly together:
        the KV caches SGLang sized after the build, checked, agreed, bound."""
        from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool
        from sglang.srt.model_executor.forward_context import get_attn_backend

        sparse = get_attn_backend()
        pool = getattr(sparse, "kv_pool", None)
        reasons = []
        pages = 0
        if not isinstance(pool, MiniMaxM3Nvfp4KVPool):
            reasons.append(f"the KV pool {type(pool).__name__} is not the M3.1 NVFP4 pool")
        elif pool.page_size != PAGE:
            reasons.append(f"the KV pool has {pool.page_size}-token pages, not {PAGE}")
        else:
            from inference.high_tps.msa_v1.attention.api import validate_paged_cache

            for i in self.layer_ids:
                try:
                    validate_paged_cache(*self._cache_views(pool, i))
                except ValueError as error:
                    reasons.append(f"layer {i} cache views: {error}")
                    break
            pages = -(-int(sparse.req_to_token.shape[1]) // PAGE)
            if pages > INDEXER_MAX_PAGES:
                pages = INDEXER_MAX_PAGES
                if int(sparse.max_context_len) > pages * PAGE:
                    reasons.append(
                        f"context length {sparse.max_context_len} exceeds the kernel's "
                        f"{pages * PAGE} tokens"
                    )
        bad = [f"rank {r}: {why}" for r, rs in enumerate(_gather_tp(reasons)) for why in rs]
        if bad:
            raise ValueError("msa-v1 monokernel: " + "; ".join(bad))
        names = ("k", "v", "k_scale", "v_scale", "indexer_k", "indexer_k_scale")
        self.caches = {
            i: dict(zip(names, self._cache_views(pool, i))) for i in self.layer_ids
        }
        self.req_to_token = sparse.req_to_token
        self.pages = max(TOPK_BLOCKS, pages)
        self.binding = _Binding(self)

    @staticmethod
    def _cache_views(pool, layer: int) -> tuple[torch.Tensor, ...]:
        return (
            pool.get_key_buffer(layer),
            pool.get_value_buffer(layer),
            pool.get_main_k_scale_buffer(layer),
            pool.get_main_v_scale_buffer(layer),
            pool.get_index_k_buffer(layer),
            pool.get_index_k_scale_buffer(layer),
        )

    def _hook_out_of_graph_prepare(self) -> None:
        """Prepare a step's inputs where its attention backend prepares its own
        metadata outside the graph, before every capture and replay."""
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        if id(backend) in self._hooked:
            return
        self._hooked.add(id(backend))
        original = backend.init_forward_metadata_out_graph

        def init_forward_metadata_out_graph(forward_batch, *args, **kwargs):
            result = original(forward_batch, *args, **kwargs)
            if (
                forward_batch.forward_mode.is_target_verify()
                and int(forward_batch.batch_size) == 1
                and int(forward_batch.input_ids.shape[0]) == ROWS
            ):
                self.binding.prepare(forward_batch, self.req_to_token)
            return result

        backend.init_forward_metadata_out_graph = init_forward_metadata_out_graph

    # ----- one step ----------------------------------------------------------

    def forward(self, plan, input_ids, positions, forward_batch, *args, **kwargs):
        """The model's forward for a routed step: the backbone, then SGLang's
        logits processor (or the hidden states, the VL wrapper's embedding mode)."""
        if self.binding is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("msa-v1: the verify binding was not made before capture")
            self._bind()
        if not torch.cuda.is_current_stream_capturing():
            # warmup, capture's eager runs and graph-less steps prepare here; a
            # replay's inputs were prepared by the hooked backend
            self._hook_out_of_graph_prepare()
            self.binding.prepare(forward_batch, self.req_to_token)
        hidden, aux = self._backbone(input_ids, positions, forward_batch)
        model = self.model
        if kwargs.get("get_embedding", False):
            return hidden
        return model.logits_processor(input_ids, hidden, model.lm_head, forward_batch, aux)

    def _backbone(self, input_ids, positions, forward_batch):
        """``MiniMaxM3Model.forward``: the embedding and dense layers on SGLang's
        modules, every MoE layer as MegaAttention + MoE, and SGLang's final norm."""
        from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        lm, binding = self.lm, self.binding
        hidden = lm.embed_tokens(input_ids)
        residual_batch.start(forward_batch)
        aux = AuxHiddenStateList()
        for i in range(lm.start_layer, self.first_moe):
            layer = lm.layers[i]
            hidden = layer(
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden,
                captured_last_layer_outputs=(
                    aux if getattr(layer, "_is_layer_to_capture", False) else None
                ),
            )
        # the residual stream entering the first MoE layer: complete the last dense
        # layer's owed all-reduce and apply its plain residual add
        stream = residual_batch.stream_of(forward_batch)
        if stream.pending is not None and not stream.pending.update.is_plain_add:
            raise RuntimeError("msa-v1: the last dense layer's residual update is not a plain add")
        output, residual = stream.export(hidden)
        residual_batch.clear(forward_batch)
        binding.layer_input.copy_(output if residual is None else output + residual)
        for n, i in enumerate(self.layer_ids):
            if getattr(lm.layers[i], "_is_layer_to_capture", False):
                # the residual stream entering layer i, as its attention boundary hands it
                aux.capture(binding.layer_input if n == 0 else binding.output)
            binding.attention[n].run()
            binding.moe[n].run()
        if lm.end_layer in lm.layers_to_capture:
            aux.capture(binding.output)
        hidden = lm.norm(binding.output)
        capture = getattr(self.model, "capture_aux_hidden_states", False)
        return hidden, (aux if capture and len(aux) else None)


def build(model, hf_config, ctx) -> Built:
    return Built(model, hf_config, ctx)
