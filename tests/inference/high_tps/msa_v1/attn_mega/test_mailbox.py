"""Public MegaAttention regression for rank-ordered TP4 reduction and Graph reuse."""

import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from inference.high_tps.msa_v1.attention import create_attention_runner
from inference.high_tps.msa_v1.attention.communication import create_attention_tp_context
from ..attn_backends.test_backends import (
    _prewarm_mega_variant,
    check_topk_against_scores,
    make_attention_weights,
    make_inputs,
    reconstruct_mxfp8,
    reference_attention,
    reference_scores,
)
from ..common.m3_decode_fixture import graph_summary, synchronize_checked


def bf16_ulp(actual, expected):
    def ordered(value):
        bits = value.contiguous().view(torch.int16).int()
        return torch.where(bits < 0, -32768 - bits, bits)

    return int((ordered(actual) - ordered(expected)).abs().max())


def check_case(context, rank, device, q_len, kv_len, seed):
    model = make_attention_weights(device, rank, seed)
    # One product per output makes OProj exact, isolating the TP4 rounding contract.
    # Independent references below also check Indexer/FMHA using the produced Q/KV.
    model.output_projection.zero_()
    rows = torch.arange(6144, device=device)
    columns = (rows * 17 + rank) % 2048
    signs = torch.where(rows % 2 == 0, 1.0, -1.0)
    model.output_projection[rows, columns] = signs.to(torch.float8_e4m3fn)
    slot, weights, cache, buffers = make_inputs(
        device, rank, q_len, kv_len, seed, model, cache_scale=0.03125
    )
    generator = torch.Generator(device=device).manual_seed(seed + rank * 100003)
    pages = (kv_len + 127) // 128
    permutation = torch.randperm(pages, device=device, generator=generator).int()
    cache.kv_indices.copy_(permutation[None])
    cache.indexer_page_table.copy_(permutation)
    positions = torch.arange(kv_len - 8, kv_len, device=device)
    slot.positions.copy_(positions)
    slot.slot_mapping.copy_(permutation[positions // 128].long() * 128 + positions % 128)
    slot.slot_mapping[:8 - q_len] = -1
    angles = torch.arange(kv_len, device=device).float()[:, None] * (
        10000.0 ** (-torch.arange(32, device=device).float() / 32)
    )[None]
    slot.cos_sin_cache = torch.cat((angles.cos(), angles.sin()), -1).bfloat16()
    slot.post_attention_rms_weight.copy_(
        (torch.randn(6144, device=device, generator=generator) * 0.03).bfloat16()
    )
    slot.layer_input[:, ::31] = -0.0
    runner = create_attention_runner(weights=weights, tp_context=context, backend="mega")
    plan = runner.prepare(
        q_len=q_len, layer_input=slot.layer_input, positions=slot.positions,
        slot_mapping=slot.slot_mapping, cos_sin_cache=slot.cos_sin_cache,
        cache=cache, buffers=buffers, plan_info=slot.plan,
    )
    started = time.monotonic()
    _prewarm_mega_variant(kv_len)
    compile_seconds = time.monotonic() - started
    dist.barrier()
    # Snapshot all cache rows so skipped slots and unrelated pages are checked.
    cache_tensors = (cache.k, cache.v, cache.ki, cache.k_scale, cache.v_scale, cache.ki_scale)
    before = [value.view(torch.uint8).clone() for value in cache_tensors]
    started = time.monotonic()
    plan.run()
    synchronize_checked(device, "mailbox_smoke")
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        for _ in range(31):
            plan.run()
    info = graph_summary(graph)
    assert info["kernel_count"] == 31 and set(info["node_types"]) == {"kernel"}, info
    graph.replay()
    graph.replay()
    synchronize_checked(device, "mailbox_graph_wrap")
    run_seconds = time.monotonic() - started
    activation = reconstruct_mxfp8(buffers.output, buffers.output_scale).reshape(8, 2048)
    local = (activation[:, columns] * signs).bfloat16()
    # Transport on CPU allows Gloo-only tests; accumulation order is explicitly rank ordered.
    peers = [torch.empty_like(local.cpu()) for _ in range(4)]
    dist.all_gather(peers, local.cpu())
    total = torch.zeros_like(local, dtype=torch.float32)
    for peer in peers:
        total += peer.to(device).float()
    expected = (total + slot.layer_input.float()).bfloat16()
    norm = (
        expected.float()
        * torch.rsqrt(expected.float().square().mean(-1, keepdim=True) + 1e-6)
        * (1 + slot.post_attention_rms_weight.float())
    ).bfloat16()
    residual_error = bf16_ulp(buffers.attention_residual, expected)
    norm_error = bf16_ulp(buffers.post_attention_norm, norm)
    assert residual_error == 0, residual_error
    assert norm_error <= 2, norm_error
    active = torch.zeros(cache.k.shape[0] * 128, device=device, dtype=torch.bool)
    active[slot.slot_mapping[-q_len:].long()] = True
    for value, original in zip(cache_tensors, before):
        width = 8 if value.dtype == torch.float8_e4m3fn else 64
        assert torch.equal(
            value.view(torch.uint8).reshape(-1, width)[~active],
            original.reshape(-1, width)[~active],
        )
    scores = reference_scores(buffers.qi[-q_len:], cache.ki, cache.ki_scale,
                              cache.indexer_page_table)
    check_topk_against_scores(scores, buffers.kv_block_indexes[-q_len:, 0], kv_len)
    oracle = reference_attention(buffers.q[-q_len:], cache,
                                 buffers.kv_block_indexes[-q_len:], kv_len)
    actual = activation[-q_len:].reshape(q_len, 16, 128)
    assert torch.all((actual - oracle).abs() <= 0.05 + 0.125 * oracle.abs())
    assert torch.isfinite(buffers.attention_residual).all()
    assert torch.isfinite(buffers.post_attention_norm).all()
    graph.reset()
    return dict(q=q_len, kv_len=kv_len, rank=rank, prenorm_ulp=residual_error,
                norm_ulp=norm_error, compile_seconds=compile_seconds, run_seconds=run_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-len", type=int, nargs="+",
                        default=[128, 131072, 131073, 251252, 262144, 262145, 489324])
    parser.add_argument("--q", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 4
    context = create_attention_tp_context(
        backend="mega", group_id=f"mailbox-test-{os.environ['MASTER_PORT']}"
    )
    records = []
    try:
        for kv_len in args.kv_len:
            for q_len in args.q:
                record = check_case(context, rank, device, q_len, kv_len, 1701)
                records.append(record)
                print(json.dumps(record), flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"mailbox-rank{rank}.json").write_text(json.dumps(records, indent=2) + "\n")
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
    finally:
        context.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
