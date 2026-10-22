"""Check the TP4 M3.1 DSpark draft backbone against the FP64 reference.

Run with four single-node processes:
    torchrun --standalone --nproc-per-node=4 -m tests.inference.high_tps.msa_v1.draft.test_reference

Every checked call poisons the output, starts from known caches and requires: the output and every
layer's block K/V within 1.5x the distance of an independent FP32 implementation to the FP64
reference (and cosine >= 0.99), layer 0's K/V within one BF16 ulp of each row's largest reference
value, all ranks bitwise identical, and cache bytes outside the block's slots unchanged. Cases cover
kv lengths from the block alone to 64K (window partly filled, full, page edges), mask and token
blocks, CUDA graph capture equal to eager and replays after changed inputs, bitwise repeats, weights
changed in place after the plan was built, two plans alternating on separate caches, and large norm
weights.
"""

import argparse
from dataclasses import dataclass
from datetime import timedelta
import json
import logging
import os
import time

import torch
import torch.distributed as dist

from inference.high_tps.msa_v1.draft import (
    DraftBackboneInputs,
    DraftBackbonePlan,
    DraftBackboneWeights,
    DraftLayerWeights,
    create_draft_tp_context,
)

from .reference import BLOCK, HEAD_DIM, HIDDEN, INTER, LAYERS, PAGE, ROTARY_DIM, ROWS, TP, Reference, e8m0_scale

logger = logging.getLogger(__name__)

VOCAB = 200064
VOCAB_ROWS = VOCAB // TP
MASK_ID = VOCAB - 7
EPS = 1e-6
ROPE_THETA = 5e6
TOLERANCE_FACTOR = 1.5
MIN_COSINE = 0.99
LAYER_FIELDS = ("input_norm", "post_norm", "q_norm", "k_norm", "qkv", "qkv_sf", "o", "o_sf", "gate_up",
                "gate_up_sf", "down", "down_sf")
MATRICES = ("qkv", "qkv_sf", "o", "o_sf", "gate_up", "gate_up_sf", "down", "down_sf")


def mxfp8_weight(rows, k, generator, device, std):
    w = torch.randn(rows, k, generator=generator, device=device) * std
    groups = w.view(rows, k // BLOCK, BLOCK)
    e = e8m0_scale(groups.abs().amax(-1))
    codes = (groups * torch.exp2(-e.float())[..., None]).view(rows, k).to(torch.float8_e4m3fn)
    return codes.contiguous(), (e + 127).to(torch.uint8).contiguous()


def make_weights(rank, device, seed, norm_scale=0.1):
    """Random rank-local operands; norm weights are common to every rank."""
    shared = torch.Generator(device=device).manual_seed(seed)
    local = torch.Generator(device=device).manual_seed(seed * 7919 + rank + 1)

    def norm(n):
        return (torch.randn(n, generator=shared, device=device) * norm_scale).to(torch.bfloat16)
    layers = []
    for _ in range(LAYERS):
        input_norm, post_norm, q_norm, k_norm = norm(HIDDEN), norm(HIDDEN), norm(HEAD_DIM), norm(HEAD_DIM)
        qkv, qkv_sf = mxfp8_weight(2304, HIDDEN, local, device, HIDDEN**-0.5)
        o, o_sf = mxfp8_weight(HIDDEN, 2048, local, device, (TP * 2048) ** -0.5)
        gate_up, gate_up_sf = mxfp8_weight(2 * INTER, HIDDEN, local, device, HIDDEN**-0.5)
        down, down_sf = mxfp8_weight(HIDDEN, INTER, local, device, (TP * INTER) ** -0.5)
        layers.append(DraftLayerWeights(input_norm, post_norm, q_norm, k_norm, qkv, qkv_sf, o, o_sf,
                                        gate_up, gate_up_sf, down, down_sf))
    embed = (torch.randn(VOCAB_ROWS, HIDDEN, generator=local, device=device) * 0.05).to(torch.bfloat16)
    return DraftBackboneWeights(layers, embed, rank * VOCAB_ROWS, norm(HIDDEN), EPS)


def cos_sin_cache(positions, device):
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, ROTARY_DIM, 2, device=device, dtype=torch.float32) / ROTARY_DIM))
    freqs = torch.arange(positions, device=device, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def make_inputs(device, max_kv, seed):
    """Graph-stable step buffers for kv lengths up to max_kv over a shuffled page table."""
    pages = (max_kv + PAGE - 1) // PAGE + 1
    page_table = torch.randperm(pages + 8, generator=torch.Generator().manual_seed(seed))[:pages]
    slots = (pages + 8) * PAGE
    generator = torch.Generator(device=device).manual_seed(seed + 99)

    def cache():
        return torch.randn(slots, HEAD_DIM, generator=generator, device=device).to(torch.bfloat16)
    return DraftBackboneInputs(
        input_ids=torch.zeros(ROWS, dtype=torch.int64, device=device),
        positions=torch.zeros(ROWS, dtype=torch.int64, device=device),
        slot_mapping=torch.full((ROWS,), -1, dtype=torch.int64, device=device),
        page_table=page_table.to(torch.int32).to(device),
        kv_len=torch.zeros(1, dtype=torch.int32, device=device),
        cos_sin=cos_sin_cache(max_kv + 8, device),
        k_caches=[cache() for _ in range(LAYERS)],
        v_caches=[cache() for _ in range(LAYERS)])


def stage(inputs, kv_len, bonus, mask_rows):
    """Write one block (kv_len cached tokens including its seven rows) into the step buffers."""
    device = inputs.positions.device
    keys = torch.arange(kv_len - ROWS, kv_len, device=device)
    ids = [bonus] + [MASK_ID if mask_rows else (bonus + 13 * i) % VOCAB for i in range(1, ROWS)]
    inputs.input_ids.copy_(torch.tensor(ids, device=device))
    inputs.positions.copy_(keys)
    inputs.slot_mapping.copy_(inputs.page_table[keys // PAGE].long() * PAGE + keys % PAGE)
    inputs.kv_len.fill_(kv_len)


def gather(tensor, group):
    """Every rank's copy of tensor (bytes through host memory: the group is gloo)."""
    host = tensor.contiguous().cpu().view(torch.uint8)
    out = [torch.empty_like(host) for _ in range(TP)]
    dist.all_gather(out, host, group=group)
    return [o.view(tensor.dtype).view(tensor.shape).to(tensor.device) for o in out]


def reference_ranks(weights, group):
    gathered = [{k: gather(getattr(layer, k), group) for k in MATRICES} for layer in weights.layers]
    embeds = gather(weights.embed, group)
    out = []
    for r in range(TP):
        layers = []
        for layer, g in zip(weights.layers, gathered):
            d = {k: getattr(layer, k).double() for k in ("input_norm", "post_norm", "q_norm", "k_norm")}
            d.update({k: v[r] for k, v in g.items()})
            layers.append(d)
        out.append(dict(layers=layers, embed=embeds[r], vocab_start=r * VOCAB_ROWS,
                        final_norm=weights.final_norm.double(), eps=weights.eps))
    return out


def cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0))


def within_one_row_ulp(got, want):
    """|got - want| <= one BF16 ulp of the row's largest |want|.

    Elements far below their row's scale, or cancelled by RoPE, carry the absolute FP32 rounding
    of that scale, which no FP32 implementation keeps within the element's own ulp.
    """
    want = want.double()
    top = want.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    ulp = torch.exp2(torch.floor(torch.log2(top)) - 7)
    return bool(((got.double() - want).abs() <= ulp).all())


@dataclass
class Bench:
    group: object
    rank: int


def check(bench, label, run, inputs, out, weights, expect_bits=None):
    """One checked call of ``run`` from the current caches against both references."""
    group, rank = bench.group, bench.rank
    snap = {k: getattr(inputs, k) for k in ("input_ids", "positions", "slot_mapping", "page_table", "kv_len", "cos_sin")}
    rank_caches = [[(gather(k, group)[r].clone(), gather(v, group)[r].clone())
                    for k, v in zip(inputs.k_caches, inputs.v_caches)] for r in range(TP)]
    ranks = reference_ranks(weights, group)
    refs = {}
    for name, reference in (("exact", Reference(torch.float64)), ("bound", Reference(torch.float32, chunked=True))):
        caches = [[(k.clone(), v.clone()) for k, v in layers] for layers in rank_caches]
        refs[name] = (reference.forward(ranks, snap, caches), caches)
    before = [(k.clone(), v.clone()) for k, v in zip(inputs.k_caches, inputs.v_caches)]
    out.fill_(float("nan"))
    torch.cuda.synchronize()
    dist.barrier(group=group)
    start = time.time()
    run()  # A run exceeding 30 seconds is a deadlock.
    torch.cuda.synchronize()
    logger.info("%s ran in %.3f ms", label, (time.time() - start) * 1e3)
    valid = inputs.slot_mapping[inputs.slot_mapping >= 0]
    exact, exact_caches = refs["exact"]
    bound, bound_caches = refs["bound"]
    tensors = [("out", out, exact, bound)]
    for l in range(LAYERS):
        for i, name in ((0, "k"), (1, "v")):
            kv = inputs.k_caches[l] if i == 0 else inputs.v_caches[l]
            tensors.append((f"{name}{l}", kv[valid], exact_caches[rank][l][i][valid], bound_caches[rank][l][i][valid]))
    failures = [] if bool(torch.isfinite(out.float()).all()) else ["non-finite output"]
    for name, got, want, alt in tensors:
        c, c_bound = cosine(got, want), cosine(alt, want)
        limit = max(1.0 - TOLERANCE_FACTOR * max(1.0 - c_bound, 1e-6), MIN_COSINE)
        if c < limit:
            failures.append(f"{name}: cosine {c:.6f} < {limit:.6f}")
    for i, name in ((0, "k0"), (1, "v0")):
        cache = inputs.k_caches[0] if i == 0 else inputs.v_caches[0]
        if not within_one_row_ulp(cache[valid], exact_caches[rank][0][i][valid]):
            failures.append(f"{name}: not within one row ulp of the reference")
    for (k, v), (k0, v0) in zip(zip(inputs.k_caches, inputs.v_caches), before):
        if not (torch.equal(k.index_fill(0, valid, 0), k0.index_fill(0, valid, 0))
                and torch.equal(v.index_fill(0, valid, 0), v0.index_fill(0, valid, 0))):
            failures.append("cache changed outside the block's slots")
            break
    outs = [None] * TP
    dist.all_gather_object(outs, out.cpu(), group=group)
    if not all(torch.equal(outs[0], o) for o in outs):
        failures.append("ranks differ")
    if expect_bits is not None and not torch.equal(out, expect_bits):
        failures.append("not bitwise equal to the earlier call")
    verdicts = [None] * TP
    dist.all_gather_object(verdicts, failures, group=group)
    failures = sorted({f"rank {r}: {f}" for r, fs in enumerate(verdicts) for f in fs})
    if failures:
        raise AssertionError(f"{label}: " + "; ".join(failures))
    return {"case": label, "out_cosine": cosine(out, exact), "bound_cosine": cosine(bound, exact)}


def restore(inputs, snapshot):
    for (k, v), (k0, v0) in zip(zip(inputs.k_caches, inputs.v_caches), snapshot):
        k.copy_(k0)
        v.copy_(v0)


def run_suite(group, rank, args):
    device = torch.device("cuda", torch.cuda.current_device())
    bench = Bench(group, rank)
    start = time.time()
    context = create_draft_tp_context(group)  # builds or loads the extension
    logger.info("context created in %.3f s", time.time() - start)
    results = []
    try:
        weights = make_weights(rank, device, args.seed)
        inputs = make_inputs(device, max(args.kv_lens) + 64, args.seed + 1)
        out = torch.empty(ROWS, HIDDEN, dtype=torch.bfloat16, device=device)
        plan = DraftBackbonePlan(context, weights, inputs, out)
        for i, kv_len in enumerate(args.kv_lens):
            for mask_rows in (True, False):
                stage(inputs, kv_len, (1 + 7919 * i + 104729 * mask_rows) % VOCAB, mask_rows)
                results.append(check(bench, f"kv{kv_len}_{'mask' if mask_rows else 'tokens'}", plan.run, inputs, out,
                                     weights))
        # Bitwise repeat, then CUDA graph capture equal to eager and replays after changed inputs.
        stage(inputs, 8192, 123, True)
        snapshot = [(k.clone(), v.clone()) for k, v in zip(inputs.k_caches, inputs.v_caches)]
        results.append(check(bench, "repeat_first", plan.run, inputs, out, weights))
        first = out.clone()
        restore(inputs, snapshot)
        results.append(check(bench, "repeat_bitwise", plan.run, inputs, out, weights, expect_bits=first))
        restore(inputs, snapshot)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            plan.run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan.run()
        torch.cuda.synchronize()
        restore(inputs, snapshot)
        results.append(check(bench, "graph_equals_eager", graph.replay, inputs, out, weights, expect_bits=first))
        for i, (kv_len, mask_rows) in enumerate(((5000, True), (12, False), (8199, True))):
            stage(inputs, kv_len, 1000 + 17 * i, mask_rows)
            results.append(check(bench, f"graph_kv{kv_len}", graph.replay, inputs, out, weights))
        # Weights changed in place after the plan and the graph were built.
        generator = torch.Generator(device=device).manual_seed(args.seed + 7)
        for layer in weights.layers:
            for name in ("qkv", "o", "gate_up", "down"):
                flip = torch.randint(0, 2, getattr(layer, name).shape, generator=generator, device=device,
                                     dtype=torch.uint8)
                getattr(layer, name).view(torch.uint8).bitwise_xor_(flip * 0x80)  # E4M3 sign flips
            layer.input_norm.mul_(-1.0)
            layer.qkv_sf.bitwise_xor_(1)
        weights.embed.mul_(-0.5)
        stage(inputs, 8192, 2024, True)
        results.append(check(bench, "weights_changed_eager", plan.run, inputs, out, weights))
        stage(inputs, 3000, 2025, False)
        results.append(check(bench, "weights_changed_graph", graph.replay, inputs, out, weights))
        del graph
        # Two plans alternating on separate step buffers.
        inputs2 = make_inputs(device, 9000, args.seed + 3)
        out2 = torch.empty_like(out)
        plan2 = DraftBackbonePlan(context, weights, inputs2, out2)
        for i in range(2):
            stage(inputs, 2000 + ROWS * i, 300 + i, True)
            results.append(check(bench, f"alternate_a{i}", plan.run, inputs, out, weights))
            stage(inputs2, 9000 - ROWS * i, 400 + i, False)
            results.append(check(bench, f"alternate_b{i}", plan2.run, inputs2, out2, weights))
        # Larger norm weights (Gemma 1 + w far from 1) and larger embedding rows.
        big = make_weights(rank, device, args.seed + 11, norm_scale=1.5)
        big.embed.mul_(4.0)
        out3 = torch.empty_like(out)
        plan3 = DraftBackbonePlan(context, big, inputs, out3)
        stage(inputs, 6000, 160001, True)
        results.append(check(bench, "large_norm_weights", plan3.run, inputs, out3, big))
        torch.cuda.synchronize()
    finally:
        context.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-lens", type=int, nargs="+", default=[7, 8, 1000, 4096, 4103, 4224, 8192, 65536])
    parser.add_argument("--seed", type=int, default=4879)
    args = parser.parse_args()
    if min(args.kv_lens) < ROWS:
        parser.error(f"kv lengths include the block's {ROWS} rows")
    logging.basicConfig(level=logging.INFO if os.environ.get("RANK") == "0" else logging.WARNING)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("gloo", timeout=timedelta(minutes=30))
    try:
        if dist.get_world_size() != TP:
            raise ValueError("run four single-node processes")
        rank = dist.get_rank()
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
        group = dist.new_group(ranks=list(range(TP)), backend="gloo")
        results = run_suite(group, rank, args)
        if rank == 0:
            logger.info(json.dumps({"passed": True, "checks": len(results),
                                    "worst_out_cosine": min(r["out_cosine"] for r in results)}))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
