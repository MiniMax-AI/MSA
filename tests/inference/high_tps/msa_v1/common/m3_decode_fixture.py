"""Self-contained operands for the public M3 decode MegaKernel tools."""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.high_tps.msa_v1.attention import (
    create_attention_plan,
    m3_decode_mega_atten,
)

TP_SIZE = 4
TOKENS = 8
HIDDEN_SIZE = 6144
NUM_Q_HEADS = 16
NUM_KV_HEADS = 1
HEAD_DIM = 128
PAGE_SIZE = 128
ATTENTION_TOPK = 16
SELECTED_KV = ATTENTION_TOPK * PAGE_SIZE
EXPERTS = 128
ROUTER_TOPK = 4
ROUTES = TOKENS * ROUTER_TOPK
INTERMEDIATE_SIZE = 768
ATTENTION_PROJECTION_OUTPUT = 2560
SCALE_VECTOR = 32
KERNEL_TIMEOUT_SECONDS = 10.0
DEFAULT_SEED = 20260731


@dataclass
class ModelWeights:
    attention_projection: torch.Tensor
    attention_projection_sf: torch.Tensor
    output_projection: torch.Tensor
    output_projection_sf: torch.Tensor
    shared_fc1: torch.Tensor
    shared_fc1_sf: torch.Tensor
    shared_fc2: torch.Tensor
    shared_fc2_sf: torch.Tensor
    routed_fc1: torch.Tensor
    routed_fc1_sf: torch.Tensor
    routed_fc1_alpha: torch.Tensor
    routed_fc2: torch.Tensor
    routed_fc2_sf: torch.Tensor
    routed_fc2_alpha: torch.Tensor

    def clone(self) -> ModelWeights:
        return ModelWeights(
            **{
                field.name: getattr(self, field.name).clone()
                for field in dataclasses.fields(self)
            }
        )


@dataclass
class ProjectionWorkspace:
    data: torch.Tensor
    scale: torch.Tensor
    grid_epoch: torch.Tensor
    special_partial: torch.Tensor
    special_epochs: torch.Tensor


@dataclass
class RoutedOutputs:
    active_expert_count: torch.Tensor
    route_count: torch.Tensor
    active_expert_ids: torch.Tensor
    expert_route_offsets: torch.Tensor
    route_token_ids: torch.Tensor
    route_topk_ids: torch.Tensor
    route_expert_ids: torch.Tensor
    route_active_slots: torch.Tensor
    route_scales: torch.Tensor
    activation: torch.Tensor
    activation_sf: torch.Tensor
    mid: torch.Tensor
    mid_sf: torch.Tensor


@dataclass
class M3DecodeSlot:
    kv_len: int
    plan: dict
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    kv_indices: torch.Tensor
    kv_block_indexes: torch.Tensor
    indexer_q: torch.Tensor
    indexer_k: torch.Tensor
    indexer_k_scale: torch.Tensor
    indexer_page_table: torch.Tensor
    layer_input: torch.Tensor
    input_norm_weight: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    qi_norm_weight: torch.Tensor
    ki_norm_weight: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    cos_sin_cache: torch.Tensor
    projection_workspace: ProjectionWorkspace
    weights: ModelWeights
    output: torch.Tensor
    output_sf: torch.Tensor
    post_attention_rms_weight: torch.Tensor
    shared_activation: torch.Tensor
    shared_activation_sf: torch.Tensor
    shared_mid: torch.Tensor
    router_weight: torch.Tensor
    router_bias: torch.Tensor
    router_indices: torch.Tensor
    router_weights: torch.Tensor
    routed: RoutedOutputs
    expected_router_indices: torch.Tensor | None = None
    expected_router_weights: torch.Tensor | None = None
    attention_residual: torch.Tensor | None = None
    split_attention_residual: torch.Tensor | None = None
    post_attention_norm: torch.Tensor | None = None


def _unit_scale_storage(elements: int, device: torch.device) -> torch.Tensor:
    return torch.full((elements,), 127, dtype=torch.uint8, device=device)


def _valid_fp8_weight(
    shape: tuple[int, ...],
    device: torch.device,
    generator: torch.Generator,
    *,
    zero: bool,
    scale: float,
) -> torch.Tensor:
    if zero:
        return torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
    return (
        torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
        * scale
    ).to(torch.float8_e4m3fn)


def make_model_weights(
    device: torch.device,
    seed: int,
    *,
    zero_math: bool,
) -> ModelWeights:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    attention_projection = torch.zeros(
        (ATTENTION_PROJECTION_OUTPUT, HIDDEN_SIZE),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    if not zero_math:
        rows = torch.arange(ATTENTION_PROJECTION_OUTPUT, device=device)
        columns = (rows * 1543 + 17) % HIDDEN_SIZE
        coefficients = torch.where(
            (rows & 1) == 0,
            torch.ones_like(rows, dtype=torch.float32),
            -torch.ones_like(rows, dtype=torch.float32),
        ).to(torch.float8_e4m3fn)
        attention_projection[rows, columns] = coefficients

    output_projection = _valid_fp8_weight(
        (HIDDEN_SIZE, NUM_Q_HEADS * HEAD_DIM),
        device,
        generator,
        zero=zero_math,
        scale=0.03125,
    )
    shared_fc1 = _valid_fp8_weight(
        (2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        device,
        generator,
        zero=zero_math,
        scale=0.03125,
    )
    shared_fc2 = _valid_fp8_weight(
        (HIDDEN_SIZE, INTERMEDIATE_SIZE),
        device,
        generator,
        zero=zero_math,
        scale=0.03125,
    )

    routed_fc1 = torch.zeros(
        (EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 2),
        dtype=torch.uint8,
        device=device,
    )
    routed_fc2 = torch.zeros(
        (EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE // 2),
        dtype=torch.uint8,
        device=device,
    )
    if not zero_math:
        routed_fc1.random_(0, 256, generator=generator)
        routed_fc2.random_(0, 256, generator=generator)

    routed_fc1_alpha = torch.zeros(EXPERTS, dtype=torch.float32, device=device)
    routed_fc2_alpha = torch.zeros(EXPERTS, dtype=torch.float32, device=device)
    if not zero_math:
        routed_fc1_alpha.copy_(
            torch.linspace(0.5, 1.5, EXPERTS, dtype=torch.float32, device=device)
        )
        routed_fc2_alpha.copy_(
            torch.linspace(5.0e-4, 1.5e-3, EXPERTS, dtype=torch.float32, device=device)
        )

    return ModelWeights(
        attention_projection=attention_projection,
        attention_projection_sf=_unit_scale_storage(
            ATTENTION_PROJECTION_OUTPUT * (HIDDEN_SIZE // SCALE_VECTOR), device
        ),
        output_projection=output_projection,
        output_projection_sf=_unit_scale_storage(
            HIDDEN_SIZE * (NUM_Q_HEADS * HEAD_DIM // SCALE_VECTOR), device
        ),
        shared_fc1=shared_fc1,
        shared_fc1_sf=_unit_scale_storage(
            2 * INTERMEDIATE_SIZE * (HIDDEN_SIZE // SCALE_VECTOR), device
        ),
        shared_fc2=shared_fc2,
        shared_fc2_sf=_unit_scale_storage(
            HIDDEN_SIZE * (INTERMEDIATE_SIZE // SCALE_VECTOR), device
        ),
        routed_fc1=routed_fc1,
        routed_fc1_sf=torch.ones(
            (EXPERTS, 2 * INTERMEDIATE_SIZE * (HIDDEN_SIZE // 16)),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        routed_fc1_alpha=routed_fc1_alpha,
        routed_fc2=routed_fc2,
        routed_fc2_sf=torch.ones(
            (EXPERTS, HIDDEN_SIZE * (INTERMEDIATE_SIZE // 16)),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        routed_fc2_alpha=routed_fc2_alpha,
    )


def _projection_workspace(device: torch.device) -> ProjectionWorkspace:
    return ProjectionWorkspace(
        data=torch.empty(TOKENS * HIDDEN_SIZE, dtype=torch.uint8, device=device),
        scale=torch.empty(
            TOKENS * (HIDDEN_SIZE // SCALE_VECTOR),
            dtype=torch.uint8,
            device=device,
        ),
        grid_epoch=torch.zeros(32, dtype=torch.int32, device=device),
        special_partial=torch.empty(
            4 * 4 * TOKENS * HEAD_DIM, dtype=torch.float32, device=device
        ),
        special_epochs=torch.zeros(4 * 32, dtype=torch.int32, device=device),
    )


def _routed_outputs(device: torch.device) -> RoutedOutputs:
    return RoutedOutputs(
        active_expert_count=torch.zeros(1, dtype=torch.int32, device=device),
        route_count=torch.zeros(1, dtype=torch.int32, device=device),
        active_expert_ids=torch.empty(ROUTES, dtype=torch.int32, device=device),
        expert_route_offsets=torch.empty(ROUTES + 1, dtype=torch.int32, device=device),
        route_token_ids=torch.empty(ROUTES, dtype=torch.int32, device=device),
        route_topk_ids=torch.empty(ROUTES, dtype=torch.int32, device=device),
        route_expert_ids=torch.empty(ROUTES, dtype=torch.int32, device=device),
        route_active_slots=torch.empty(ROUTES, dtype=torch.int32, device=device),
        route_scales=torch.empty(ROUTES, dtype=torch.float32, device=device),
        activation=torch.empty(
            (TOKENS, HIDDEN_SIZE // 2), dtype=torch.uint8, device=device
        ),
        activation_sf=torch.empty(
            (TOKENS, HIDDEN_SIZE // 16), dtype=torch.uint8, device=device
        ),
        mid=torch.empty(
            (ROUTES, TOKENS, INTERMEDIATE_SIZE // 2),
            dtype=torch.uint8,
            device=device,
        ),
        mid_sf=torch.empty(
            (ROUTES, 128 * (INTERMEDIATE_SIZE // 16)),
            dtype=torch.uint8,
            device=device,
        ),
    )


def _rope_cache(device: torch.device, rows: int = 16) -> torch.Tensor:
    result = torch.zeros((rows, 64), dtype=torch.bfloat16, device=device)
    result[:, :32] = 1.0
    return result


def make_slot(
    device: torch.device,
    rank: int,
    kv_len: int,
    seed: int,
    weights: ModelWeights,
) -> M3DecodeSlot:
    if kv_len < TOKENS or kv_len > 1024 * 1024:
        raise ValueError("kv_len must be in [8, 1048576]")
    logical_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    physical_pages = logical_pages + 1
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + rank * 100003)

    q = torch.empty(
        (TOKENS, NUM_Q_HEADS, HEAD_DIM),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    cache_shape = (physical_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM // 2)
    scale_shape = (physical_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM // 16)
    k = torch.zeros(cache_shape, dtype=torch.uint8, device=device)
    v = torch.zeros_like(k)
    k_scale = torch.ones(scale_shape, dtype=torch.float8_e4m3fn, device=device)
    v_scale = torch.ones_like(k_scale)
    indexer_k = torch.zeros(cache_shape, dtype=torch.uint8, device=device)
    indexer_k_scale = torch.ones(scale_shape, dtype=torch.float8_e4m3fn, device=device)
    poison = 0xA5
    for tensor in (k, v, indexer_k):
        tensor[-1].fill_(poison)
    k_scale[-1].view(torch.uint8).fill_(poison)
    v_scale[-1].view(torch.uint8).fill_(poison)
    indexer_k_scale[-1].view(torch.uint8).fill_(poison)

    layer_input = (
        torch.randn(
            (TOKENS, HIDDEN_SIZE),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.75
    ).to(torch.bfloat16)
    dimension = torch.arange(HIDDEN_SIZE, device=device, dtype=torch.float32)
    input_norm_weight = (((dimension % 17) - 8.0) / 256.0).to(torch.bfloat16)
    head_dimension = torch.arange(HEAD_DIM, device=device, dtype=torch.float32)

    qo_lens = torch.tensor([TOKENS], dtype=torch.int32)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32)
    plan = create_attention_plan(qo_lens, kv_lens, topk=ATTENTION_TOPK, device=device)
    output_sf_bytes = max(
        128 * ((NUM_Q_HEADS * HEAD_DIM // SCALE_VECTOR + 3) // 4) * 4,
        16 * 512,
    )
    return M3DecodeSlot(
        kv_len=kv_len,
        plan=plan,
        q=q,
        k=k,
        v=v,
        k_scale=k_scale,
        v_scale=v_scale,
        kv_indices=torch.arange(logical_pages, dtype=torch.int32, device=device).view(
            1, logical_pages
        ),
        kv_block_indexes=torch.full(
            (TOKENS, NUM_KV_HEADS, ATTENTION_TOPK),
            -1,
            dtype=torch.int32,
            device=device,
        ),
        indexer_q=torch.empty(
            (TOKENS, HEAD_DIM), dtype=torch.float8_e4m3fn, device=device
        ),
        indexer_k=indexer_k,
        indexer_k_scale=indexer_k_scale,
        indexer_page_table=torch.arange(
            logical_pages, dtype=torch.int32, device=device
        ),
        layer_input=layer_input,
        input_norm_weight=input_norm_weight,
        q_norm_weight=(((head_dimension % 8) - 3.0) / 128.0).to(torch.bfloat16),
        k_norm_weight=(((head_dimension % 9) - 4.0) / 128.0).to(torch.bfloat16),
        qi_norm_weight=(((head_dimension % 10) - 5.0) / 128.0).to(torch.bfloat16),
        ki_norm_weight=(((head_dimension % 11) - 5.0) / 128.0).to(torch.bfloat16),
        positions=torch.arange(TOKENS, dtype=torch.int64, device=device),
        slot_mapping=torch.arange(
            kv_len - TOKENS, kv_len, dtype=torch.int64, device=device
        ),
        cos_sin_cache=_rope_cache(device),
        projection_workspace=_projection_workspace(device),
        weights=weights,
        output=torch.empty(
            (TOKENS, NUM_Q_HEADS, HEAD_DIM),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        output_sf=torch.empty(output_sf_bytes, dtype=torch.uint8, device=device),
        post_attention_rms_weight=torch.zeros(
            HIDDEN_SIZE, dtype=torch.bfloat16, device=device
        ),
        shared_activation=torch.empty(
            (TOKENS, HIDDEN_SIZE), dtype=torch.float8_e4m3fn, device=device
        ),
        shared_activation_sf=torch.empty(
            128 * (HIDDEN_SIZE // SCALE_VECTOR),
            dtype=torch.uint8,
            device=device,
        ),
        shared_mid=torch.empty(
            TOKENS * INTERMEDIATE_SIZE + 128 * (INTERMEDIATE_SIZE // SCALE_VECTOR),
            dtype=torch.uint8,
            device=device,
        ),
        router_weight=torch.zeros(
            (EXPERTS, HIDDEN_SIZE), dtype=torch.float32, device=device
        ),
        router_bias=torch.zeros(EXPERTS, dtype=torch.float32, device=device),
        router_indices=torch.empty(
            (TOKENS, ROUTER_TOPK), dtype=torch.int32, device=device
        ),
        router_weights=torch.empty(
            (TOKENS, ROUTER_TOPK), dtype=torch.float32, device=device
        ),
        routed=_routed_outputs(device),
        split_attention_residual=torch.empty(
            (TOKENS, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
        ),
        post_attention_norm=torch.empty(
            (TOKENS, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
        ),
    )


def route_ids(
    active_experts: int, device: torch.device, route_pattern: str = "balanced",
    q_len: int = TOKENS,
) -> torch.Tensor:
    if active_experts < ROUTER_TOPK or active_experts > ROUTES:
        raise ValueError("active_experts must be in [4, 32]")
    if q_len < 1 or q_len > TOKENS or active_experts > ROUTER_TOPK * q_len:
        raise ValueError("active_experts must not exceed 4 * q_len for q_len in [1, 8]")
    routes = torch.arange(ROUTER_TOPK, dtype=torch.int64).repeat(TOKENS, 1)
    valid = torch.arange(q_len * ROUTER_TOPK, dtype=torch.int64)
    valid = valid.remainder(active_experts).view(q_len, ROUTER_TOPK)
    if route_pattern == "skewed":
        valid = torch.arange(ROUTER_TOPK, dtype=torch.int64).repeat(q_len, 1)
        for expert in range(ROUTER_TOPK, active_experts):
            index = expert - ROUTER_TOPK
            valid[index // ROUTER_TOPK, index % ROUTER_TOPK] = expert
    routes[-q_len:] = valid
    if route_pattern == "permuted":
        mapping = (
            torch.arange(active_experts, dtype=torch.int64) * 37 + 11
        ).remainder(EXPERTS)
        routes = mapping[routes]
    elif route_pattern == "random":
        generator = torch.Generator().manual_seed(20260731 + active_experts)
        mapping = torch.randperm(EXPERTS, generator=generator)[:active_experts]
        routes = mapping[routes]
        routes[-q_len:] = routes[-q_len:][torch.randperm(q_len, generator=generator)]
        routes = torch.stack(
            [row[torch.randperm(ROUTER_TOPK, generator=generator)] for row in routes]
        )
    elif route_pattern not in ("balanced", "skewed"):
        raise ValueError(f"unsupported route pattern: {route_pattern}")
    if any(len(set(row.tolist())) != ROUTER_TOPK for row in routes):
        raise AssertionError("generated duplicate experts within one token")
    if len(set(routes.flatten().tolist())) != active_experts:
        raise AssertionError("route generation did not cover active_experts")
    if len(set(routes[-q_len:].flatten().tolist())) != active_experts:
        raise AssertionError("valid query rows did not cover active_experts")
    return routes.to(device=device, dtype=torch.int32)


def configure_router(
    slot: M3DecodeSlot,
    attention_residual: torch.Tensor,
    active_experts: int,
    route_pattern: str = "balanced",
    q_len: int = TOKENS,
) -> None:
    targets = route_ids(active_experts, attention_residual.device, route_pattern, q_len)
    input_rows = attention_residual.shape[0]
    if input_rows not in (q_len, TOKENS):
        raise ValueError("attention_residual must contain Q or 8 rows")
    input_targets = targets[-input_rows:]
    logits = torch.full((input_rows, EXPERTS), -12.0, dtype=torch.float64)
    targets_cpu = input_targets.cpu()
    for token in range(input_rows):
        for topk in range(ROUTER_TOPK):
            logits[token, int(targets_cpu[token, topk])] = 12.0 - 2.0 * topk

    residual = attention_residual.float()
    norm = residual * torch.rsqrt(residual.square().mean(dim=-1, keepdim=True) + 1.0e-6)
    norm_f64 = norm.cpu().to(torch.float64)
    basis, triangular = torch.linalg.qr(norm_f64.t(), mode="reduced")
    coefficients = torch.linalg.solve_triangular(triangular.t(), logits, upper=False)
    effective_weight = (
        (basis @ coefficients)
        .t()
        .to(device=attention_residual.device, dtype=torch.float32)
    )
    slot.router_weight.copy_(effective_weight)
    slot.router_bias.zero_()

    raw_scores = torch.sigmoid(norm @ slot.router_weight.t())
    expected_weights = torch.gather(raw_scores, 1, input_targets.long())
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    slot.expected_router_indices = input_targets
    slot.expected_router_weights = expected_weights
    slot.attention_residual = attention_residual.clone()


def launch_mega_atten(
    slot: M3DecodeSlot,
    tp_workspace,
    *,
    materialize_topk: bool,
    wait_timeout_s: float = 0.0,
    enable_placement: bool = False,
) -> None:
    if slot.split_attention_residual is None or slot.post_attention_norm is None:
        raise AssertionError("split boundary buffers are missing")
    projection = slot.projection_workspace
    weights = slot.weights
    m3_decode_mega_atten(
        slot.q,
        slot.k,
        slot.v,
        slot.plan,
        topk=ATTENTION_TOPK,
        kv_indices=slot.kv_indices,
        kv_block_indexes=slot.kv_block_indexes if materialize_topk else None,
        k_scale_tensor=slot.k_scale,
        v_scale_tensor=slot.v_scale,
        output_projection_weight=weights.output_projection,
        output_projection_weight_sf=weights.output_projection_sf,
        tp_workspace=tp_workspace,
        residual_in=slot.layer_input,
        attention_residual=slot.split_attention_residual,
        post_attention_norm=slot.post_attention_norm,
        rms_weight=slot.post_attention_rms_weight,
        indexer_q=slot.indexer_q,
        indexer_k=slot.indexer_k,
        indexer_k_scale=slot.indexer_k_scale,
        indexer_page_table=slot.indexer_page_table,
        attention_projection_input_norm_weight=slot.input_norm_weight,
        attention_projection_weight=weights.attention_projection,
        attention_projection_weight_sf=weights.attention_projection_sf,
        attention_projection_workspace_data=projection.data,
        attention_projection_workspace_scale=projection.scale,
        attention_projection_grid_epoch=projection.grid_epoch,
        attention_projection_special_partial=projection.special_partial,
        attention_projection_special_epochs=projection.special_epochs,
        attention_projection_positions=slot.positions,
        attention_projection_slot_mapping=slot.slot_mapping,
        attention_projection_q_norm_weight=slot.q_norm_weight,
        attention_projection_k_norm_weight=slot.k_norm_weight,
        attention_projection_qi_norm_weight=slot.qi_norm_weight,
        attention_projection_ki_norm_weight=slot.ki_norm_weight,
        attention_projection_cos_sin_cache=slot.cos_sin_cache,
        attention_projection_input_norm_eps=1.0e-6,
        attention_projection_post_norm_eps=1.0e-6,
        attention_projection_ki_dequant_scale=1.0,
        out=slot.output,
        out_sf=slot.output_sf,
        sm_scale=1.0 / math.sqrt(HEAD_DIM),
        rms_eps=1.0e-6,
        wait_timeout_s=wait_timeout_s,
        enable_placement=enable_placement,
    )


def synchronize_checked(device: torch.device, label: str) -> None:
    def abort() -> None:
        print(
            f"KERNEL_DEADLOCK_TIMEOUT label={label} "
            f"timeout_s={KERNEL_TIMEOUT_SECONDS}",
            file=sys.stderr,
            flush=True,
        )
        os._exit(124)

    timer = threading.Timer(KERNEL_TIMEOUT_SECONDS, abort)
    timer.daemon = True
    timer.start()
    try:
        torch.cuda.synchronize(device)
    finally:
        timer.cancel()


def graph_summary(graph: torch.cuda.CUDAGraph) -> dict[str, object]:
    graph.instantiate()
    if hasattr(graph, "get_graph_data"):
        nodes = graph.get_graph_data()["nodes"]
    else:
        # Older PyTorch exposes the retained graph handle, but not node metadata.
        import ctypes as ct

        class KernelParams(ct.Structure):
            _fields_ = [("func", ct.c_void_p)] + [
                (name, ct.c_uint) for name in (
                    "grid_x", "grid_y", "grid_z", "block_x", "block_y",
                    "block_z", "shared_bytes",
                )
            ] + [("kernel_params", ct.c_void_p), ("extra", ct.c_void_p)]

        driver = ct.CDLL("libcuda.so.1")
        driver.cuGraphGetNodes.argtypes = [
            ct.c_void_p, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_size_t)
        ]
        driver.cuGraphNodeGetType.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
        driver.cuGraphKernelNodeGetParams.argtypes = [ct.c_void_p, ct.POINTER(KernelParams)]
        driver.cuFuncGetName.argtypes = [ct.POINTER(ct.c_char_p), ct.c_void_p]

        def checked(code):
            if code:
                raise RuntimeError(f"CUDA Graph inspection failed with driver status {code}")

        handle = ct.c_void_p(graph.raw_cuda_graph())
        count = ct.c_size_t()
        checked(driver.cuGraphGetNodes(handle, None, ct.byref(count)))
        handles = (ct.c_void_p * count.value)()
        checked(driver.cuGraphGetNodes(handle, handles, ct.byref(count)))
        nodes = []
        for node in handles:
            kind = ct.c_int()
            checked(driver.cuGraphNodeGetType(node, ct.byref(kind)))
            if kind.value == 0:  # CU_GRAPH_NODE_TYPE_KERNEL
                params = KernelParams()
                checked(driver.cuGraphKernelNodeGetParams(node, ct.byref(params)))
                name = ct.c_char_p()
                checked(driver.cuFuncGetName(ct.byref(name), params.func))
                nodes.append({"node_type": "kernel", "kernel_name": name.value.decode()})
            else:
                nodes.append({"node_type": f"cuda_node_type_{kind.value}"})
    node_types: dict[str, int] = {}
    kernel_names = []
    for node in nodes:
        node_type = node["node_type"]
        node_types[node_type] = node_types.get(node_type, 0) + 1
        if node_type == "kernel":
            kernel_names.append(node["kernel_name"])
    return {
        "node_count": len(nodes),
        "kernel_count": len(kernel_names),
        "node_types": node_types,
        "unique_kernel_names": list(dict.fromkeys(kernel_names)),
    }


def read_only_tensors(slot: M3DecodeSlot) -> list[torch.Tensor]:
    weights = slot.weights
    return [
        slot.layer_input,
        slot.input_norm_weight,
        weights.attention_projection,
        weights.attention_projection_sf,
        slot.q_norm_weight,
        slot.k_norm_weight,
        slot.qi_norm_weight,
        slot.ki_norm_weight,
        slot.cos_sin_cache,
        slot.indexer_k,
        slot.indexer_k_scale,
        slot.indexer_page_table,
        weights.output_projection,
        weights.output_projection_sf,
        slot.post_attention_rms_weight,
        weights.shared_fc1,
        weights.shared_fc1_sf,
        weights.shared_fc2,
        weights.shared_fc2_sf,
        slot.router_weight,
        slot.router_bias,
        weights.routed_fc1,
        weights.routed_fc1_sf,
        weights.routed_fc1_alpha,
        weights.routed_fc2,
        weights.routed_fc2_sf,
        weights.routed_fc2_alpha,
    ]
