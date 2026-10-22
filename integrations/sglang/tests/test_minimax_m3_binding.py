"""Synthetic plugin binding/metadata/Graph integration; no model-quality claim."""

import os
import traceback
from types import SimpleNamespace

import torch
import torch.distributed as dist
from sglang_msa_mono.minimax_m3.runtime import _Binding

from inference.high_tps.msa_v1.attention import (
    AttentionConfig,
    create_attention_runner,
    create_attention_tp_context,
)
from inference.high_tps.msa_v1.moe import create_moe_runner, create_moe_tp_context
from tests.inference.high_tps.msa_v1.attn_backends.test_backends import (
    make_attention_weights,
    make_inputs,
)
from tests.inference.high_tps.msa_v1.common.m3_decode_fixture import (
    graph_summary,
    synchronize_checked,
)
from tests.inference.high_tps.msa_v1.mega.test_strided_cache import (
    CACHE_FIELDS,
    LAYOUTS,
    _attach_strided_cache,
)
from tests.inference.high_tps.msa_v1.moe_mega.test_reference import make_operands


def main():
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("gloo")
    print(f"rank={rank} prepare synthetic binding", flush=True)
    key = "binding-" + os.environ["MASTER_PORT"]
    attn_context = create_attention_tp_context(
        backend="mega", group_id=key + "-attention"
    )
    moe_context = create_moe_tp_context(backend="mega", group_id=key + "-moe")
    raw = make_attention_weights(device, rank, 20261021)
    slot, aw, _, _ = make_inputs(device, rank, 8, 1048576, 20261021, raw, 1.0, True)
    _attach_strided_cache(slot, LAYOUTS[0])
    mw, _, _ = make_operands(device, rank, 8, 20261021, random_routes=False)
    angles = torch.arange(1048576, device=device, dtype=torch.float32)[
        :, None
    ] * torch.pow(
        1e7, -torch.arange(0, 64, 2, device=device, dtype=torch.float32)[None, :] / 64
    )
    rope = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1).to(torch.bfloat16)
    built = SimpleNamespace(
        device=device,
        pages=8192,
        layer_ids=[0],
        cos_sin_cache=rope,
        caches={0: {name: getattr(slot, name) for name in CACHE_FIELDS}},
        moe_context=moe_context,
        attention_runners={
            0: create_attention_runner(
                weights=aw, tp_context=attn_context, backend="mega",
                config=AttentionConfig(enable_placement=True),
            )
        },
        moe_runners={
            0: create_moe_runner(weights=mw, tp_context=moe_context, backend="mega")
        },
    )
    binding = _Binding(built)
    binding.layer_input.copy_(slot.layer_input)
    table = torch.arange(1048576, device=device, dtype=torch.int32).view(1, -1)
    batch = SimpleNamespace(
        seq_lens=torch.tensor([1016], device=device, dtype=torch.int32),
        req_pool_indices=torch.zeros(1, device=device, dtype=torch.int32),
        positions=torch.arange(1016, 1024, device=device, dtype=torch.int64),
        out_cache_loc=torch.arange(1016, 1024, device=device, dtype=torch.int64),
    )
    binding.prepare(batch, table)
    print(f"rank={rank} binding ready", flush=True)

    def run():
        binding.attention[0].run()
        binding.moe[0].run()

    dist.barrier()
    run()
    synchronize_checked(device, "binding warmup")
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        run()
    info = graph_summary(graph)
    assert info["kernel_count"] == 2 and set(info["node_types"]) == {"kernel"}, info
    for length in (1024, 131072, 1048576, 32768, 1024):
        batch.seq_lens.fill_(length - 8)
        batch.positions.copy_(torch.arange(length - 8, length, device=device))
        batch.out_cache_loc.copy_(batch.positions)
        binding.prepare(batch, table)
        synchronize_checked(device, "prepare outside graph")
        dist.barrier()
        graph.replay()
        synchronize_checked(device, "binding replay")
        assert binding.kv_len.item() == length
        assert torch.isfinite(binding.output).all()
        captured = binding.output.clone()
        run()
        synchronize_checked(device, "binding eager")
        torch.testing.assert_close(
            binding.output.float(), captured.float(), rtol=0.02, atol=0.0625
        )
        print(
            f"PASS plugin binding rank={rank} KV={length} graph_kernels=2", flush=True
        )
    graph.reset()
    moe_context.close()
    attn_context.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - abort communicating peers on every failure
        traceback.print_exc()
        os._exit(1)
