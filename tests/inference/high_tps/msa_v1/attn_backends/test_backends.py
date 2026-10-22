"""Check TP4 MegaAttention against independent numerical references."""

import argparse
import os
import random
import traceback
from types import SimpleNamespace

import torch
import torch.distributed as dist

from inference.high_tps.msa_v1.attention import (
    AttentionConfig,
    create_attention_runner,
)
from inference.high_tps.msa_v1.attention.communication import AttentionTpContext
from inference.high_tps.msa_v1.attention.plan import (
    AttentionBuffers,
    AttentionCache,
)
from inference.high_tps.msa_v1.attention.weights import AttentionWeights
from inference.high_tps.msa_v1.common.communication import (
    create_tp_allreduce_workspace,
    destroy_tp_allreduce_workspace,
)

from ..common.m3_decode_fixture import (
    make_slot,
    synchronize_checked,
)


def make_attention_weights(device, rank, seed):
    """Allocate Attention weights without unrelated MoE tensors."""
    generator = torch.Generator(device=device).manual_seed(seed + rank * 100003)
    projection = torch.zeros((2560, 6144), device=device, dtype=torch.float8_e4m3fn)
    rows = torch.arange(2560, device=device)
    columns = (rows * 1543 + 17) % 6144
    projection[rows, columns] = torch.where(
        (rows & 1) == 0, 1.0, -1.0
    ).to(torch.float8_e4m3fn)
    output_projection = (
        torch.randn((6144, 2048), device=device, generator=generator) * 0.03125
    ).to(torch.float8_e4m3fn)
    return SimpleNamespace(
        attention_projection=projection,
        attention_projection_sf=torch.full(
            (2560 * 192,), 127, device=device, dtype=torch.uint8
        ),
        output_projection=output_projection,
        output_projection_sf=torch.full(
            (6144 * 64,), 127, device=device, dtype=torch.uint8
        ),
    )


def reference_scores(qi, packed_ki, ki_scale, page_table):
    """FP8-world page-max reference matching the MSA indexer test contract."""
    physical = page_table.reshape(-1).long()
    decoded = dequantize_cache(
        packed_ki.index_select(0, physical)[:, 0],
        ki_scale.index_select(0, physical)[:, 0],
    )
    return torch.einsum("qd,ptd->qpt", qi.float(), decoded).amax(dim=-1)


def dequantize_cache(payload, scales):
    lookup = torch.tensor(
        (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6),
        device=payload.device, dtype=torch.float32,
    )
    nibbles = torch.stack((payload & 15, payload >> 4), dim=-1)
    nibbles = nibbles.reshape(*payload.shape[:-1], 128).long()
    decoded = lookup[nibbles] * scales.float().repeat_interleave(16, -1)
    return decoded.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()


def reference_nvfp4_storage(source, global_scale):
    """Reference the independent 16-value E2M1 cache encoding."""
    values = source.float().reshape(-1, 8, 16)
    maximum = values.abs().amax(-1)
    scale = (maximum / (6.0 * global_scale)).to(torch.float8_e4m3fn)
    denominator = scale.float() * global_scale
    normalized = torch.where(
        denominator[..., None] != 0,
        values / denominator.clamp_min(torch.finfo(torch.float32).tiny)[..., None],
        0.0,
    )
    levels = torch.tensor(
        (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6),
        device=source.device, dtype=torch.float32,
    )
    distances = (normalized[..., None] - levels).abs()
    nearest = distances.argmin(-1)
    indexes = torch.arange(16, device=source.device)
    # E2M1 round-to-nearest-even also preserves the sign of a rounded zero.
    even_ties = (
        (distances == distances.amin(-1, keepdim=True)) &
        ((indexes & 1) == 0)
    )
    nearest_even = torch.where(even_ties, indexes, 16).amin(-1)
    codes = torch.where(nearest_even < 16, nearest_even, nearest).to(torch.uint8)
    codes = torch.where((normalized < 0) & (codes == 0), 8, codes)
    packed = codes[..., ::2] | (codes[..., 1::2] << 4)
    return packed.reshape(-1, 64), scale.reshape(-1, 8)


def assert_nvfp4_storage(source, actual_data, actual_scale, global_scale):
    """Accept either E2M1 encoding only when the source is an exact tie."""
    expected_data, expected_scale = reference_nvfp4_storage(source, global_scale)
    assert torch.equal(actual_scale, expected_scale)
    if torch.equal(actual_data, expected_data):
        return
    actual_codes = torch.stack((actual_data & 15, actual_data >> 4), dim=-1)
    expected_codes = torch.stack((expected_data & 15, expected_data >> 4), dim=-1)
    actual_codes = actual_codes.reshape(source.shape).long()
    expected_codes = expected_codes.reshape(source.shape).long()
    mismatch = actual_codes != expected_codes
    levels = torch.tensor(
        (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6),
        device=source.device, dtype=torch.float32,
    )
    scale = expected_scale.float().repeat_interleave(16, dim=-1)
    actual_error = (source.float() - levels[actual_codes] * scale).abs()
    expected_error = (source.float() - levels[expected_codes] * scale).abs()
    assert torch.equal(actual_error[mismatch], expected_error[mismatch])
    assert torch.all((actual_codes[mismatch] & 7) != 0)


def reference_attention(q, cache, indexes, kv_len, *, splits=1):
    """Bounded-memory FP32 sparse reference with FP8 probability storage."""
    rows, heads, _ = q.shape
    result = torch.empty((rows, heads, 128), dtype=torch.float32, device=q.device)
    tokens = torch.arange(128, device=q.device)
    for row in range(rows):
        logical = indexes[row, 0].long()
        physical = cache.kv_indices[0, logical.clamp_min(0)].long()
        k = dequantize_cache(cache.k[physical, 0], cache.k_scale[physical, 0])
        v = dequantize_cache(cache.v[physical, 0], cache.v_scale[physical, 0])
        scores = torch.einsum("hd,ptd->hpt", q[row].float(), k) / (128.0 ** 0.5)
        position = kv_len - rows + row
        valid = (logical[:, None] >= 0) & (
            logical[:, None] * 128 + tokens[None, :] <= position
        )
        scores.masked_fill_(~valid[None, :, :], -torch.inf)
        # Probability quantization occurs before combining independent splits.
        # A serial softmax is not a numerical reference for the split storage path.
        selected_pages = int((logical >= 0).sum().item())
        pages_per_split = (selected_pages + splits - 1) // splits
        partials = []
        for split in range(splits):
            maxima = torch.full((heads,), -torch.inf, device=q.device)
            denominator = torch.zeros((heads,), device=q.device)
            accumulator = torch.zeros((heads, 128), device=q.device)
            for page in range(split * pages_per_split,
                              min((split + 1) * pages_per_split, selected_pages)):
                new_maxima = torch.maximum(maxima, scores[:, page].amax(-1))
                safe_maxima = torch.where(torch.isfinite(new_maxima), new_maxima, 0.0)
                correction = torch.exp(maxima - safe_maxima)
                probability = torch.exp(scores[:, page] - safe_maxima[:, None]) * 448.0
                probability_fp8 = probability.to(torch.float8_e4m3fn).float()
                accumulator = accumulator * correction[:, None] + probability_fp8 @ v[page]
                denominator = denominator * correction + probability.sum(-1)
                maxima = new_maxima
            if splits > 1:
                accumulator = accumulator.to(torch.bfloat16).float()
            partials.append((maxima, denominator, accumulator))
        global_max = torch.stack([part[0] for part in partials]).amax(0)
        numerator = torch.zeros((heads, 128), device=q.device)
        denominator = torch.zeros((heads,), device=q.device)
        for maxima, local_sum, accumulator in partials:
            correction = torch.exp(maxima - global_max)
            numerator += accumulator * correction[:, None]
            denominator += local_sum * correction
        result[row] = numerator / denominator.clamp_min(1.0)[:, None]
    return result


def reconstruct_mxfp8(data, scales, row_offset=0):
    rows, heads, dim = data.shape
    offsets = [
        ((row + row_offset) // 128) * heads * 512 + head * 512 +
        ((row + row_offset) % 32) * 16 +
        (((row + row_offset) % 128) // 32) * 4 + col
        for row in range(rows) for head in range(heads) for col in range(4)
    ]
    indices = torch.tensor(offsets, dtype=torch.int64, device=data.device)
    sf = scales[indices].reshape(rows, heads, dim // 32)
    decoded_sf = torch.exp2(sf.float() - 127).repeat_interleave(32, -1)
    return data.float() * decoded_sf


def _prewarm_mega_variant(kv_len):
    from inference.high_tps.msa_v1.common.jit import (
        get_m3_decode_mega_atten_variant,
    )

    get_m3_decode_mega_atten_variant()


def reference_output_tail(activation, weight, layer_input, gamma, epsilon):
    """Reference the OProj and TP4 tail from its already-quantized input."""
    local = (activation.reshape(-1, 2048) @ weight.float().T).to(torch.bfloat16)
    tp_sum = local.float().cpu()
    dist.all_reduce(tp_sum)
    tp_sum = tp_sum.to(layer_input.device)
    residual = (layer_input.float() + tp_sum).to(torch.bfloat16)
    residual_fp32 = residual.float()
    reciprocal = torch.rsqrt(
        residual_fp32.square().mean(-1, keepdim=True) + epsilon
    )
    norm = (residual_fp32 * reciprocal * (1.0 + gamma.float())).to(torch.bfloat16)
    return residual, norm


def check_topk_against_scores(scores, selected, kv_len):
    for token, row in enumerate(selected):
        pages = (kv_len - len(selected) + token) // 128 + 1
        if pages <= 16:
            assert set(row[:pages].tolist()) == set(range(pages))
            assert row[pages - 1].item() == pages - 1
            assert torch.all(row[pages:] == -1)
            continue
        ids = row.tolist()
        assert ids[-1] == pages - 1
        assert len(set(ids)) == 16
        assert all(0 <= page < pages for page in ids)
        previous = scores[token, :pages - 1]
        threshold = torch.topk(previous, 15).values.min().item()
        tolerance = 1.5 * (previous.max().item() - previous.min().item()) / 65534.0
        for page in ids[:-1]:
            assert previous[page].item() >= threshold - tolerance


def make_inputs(device, rank, q_len, kv_len, seed, model_weights,
                cache_scale=0.125, random_cache=True):
    slot = make_slot(device, rank, kv_len, seed, model_weights)
    if q_len != 8:
        slot.slot_mapping[:8 - q_len] = -1
    generator = torch.Generator(device=device).manual_seed(seed + rank * 100003)
    if random_cache:
        for payload in (slot.k, slot.v, slot.indexer_k):
            payload[:-1].copy_(
                torch.randint(256, payload[:-1].shape, device=device,
                              dtype=torch.uint8, generator=generator)
            )
        for scale in (slot.k_scale, slot.v_scale, slot.indexer_k_scale):
            scale[:-1].fill_(cache_scale)
    weights = AttentionWeights(
        projection=model_weights.attention_projection,
        projection_sf=model_weights.attention_projection_sf,
        output_projection=model_weights.output_projection,
        output_projection_sf=model_weights.output_projection_sf,
        input_norm=slot.input_norm_weight,
        q_norm=slot.q_norm_weight,
        k_norm=slot.k_norm_weight,
        qi_norm=slot.qi_norm_weight,
        ki_norm=slot.ki_norm_weight,
        post_attention_norm=slot.post_attention_rms_weight,
    )
    buffers = AttentionBuffers(
        q=slot.q,
        qi=slot.indexer_q,
        projection_data=slot.projection_workspace.data,
        projection_scale=slot.projection_workspace.scale,
        projection_grid_epoch=slot.projection_workspace.grid_epoch,
        projection_special_partial=slot.projection_workspace.special_partial,
        projection_special_epochs=slot.projection_workspace.special_epochs,
        kv_block_indexes=slot.kv_block_indexes,
        output=slot.output,
        output_scale=slot.output_sf,
        attention_residual=slot.split_attention_residual,
        post_attention_norm=slot.post_attention_norm,
    )
    cache = AttentionCache(
        k=slot.k,
        v=slot.v,
        ki=slot.indexer_k,
        k_scale=slot.k_scale,
        v_scale=slot.v_scale,
        ki_scale=slot.indexer_k_scale,
        kv_indices=slot.kv_indices,
        indexer_page_table=slot.indexer_page_table,
    )
    return slot, weights, cache, buffers


def run_case(device, rank, q_len, kv_len, seed, contexts, cache_scale,
             skip_cache_write=False, enable_pdl=False, verbose=True,
             random_cache=True, prewarm_mega_variant=False,
             backends=("mega",)):
    def emit(*values):
        if verbose and rank == 0:
            print(*values, flush=True)

    model_weights = make_attention_weights(device, rank, seed)
    emit("weights_ready")
    results = {}
    for backend in backends:
        slot, weights, cache, buffers = make_inputs(
            device, rank, q_len, kv_len, seed, model_weights, cache_scale,
            random_cache,
        )
        emit(backend, "inputs_ready")
        if skip_cache_write:
            slot.slot_mapping[-1] = -1
        runner = create_attention_runner(
            weights=weights, tp_context=contexts[backend], backend=backend,
            config=AttentionConfig(enable_pdl=enable_pdl),
        )
        emit(backend, "runner_ready")
        plan = runner.prepare(
            q_len=q_len, layer_input=slot.layer_input,
            positions=slot.positions, slot_mapping=slot.slot_mapping,
            cos_sin_cache=slot.cos_sin_cache, cache=cache, buffers=buffers,
            plan_info=slot.plan,
        )
        emit(backend, "plan_ready")
        if backend == "mega" and prewarm_mega_variant:
            _prewarm_mega_variant(kv_len)
            emit(backend, "variant_prewarmed")
        dist.barrier()
        residual, norm = plan.run()
        emit(backend, "run_enqueued")
        synchronize_checked(device, f"{backend}_q{q_len}_kv{kv_len}")
        for tensor in (cache.k, cache.v, cache.ki, cache.k_scale,
                       cache.v_scale, cache.ki_scale):
            assert torch.all(tensor[-1].view(torch.uint8) == 0xA5)
        active_slots = slot.slot_mapping[-q_len:]
        active_slots = active_slots[active_slots >= 0].long()
        scores = reference_scores(
            buffers.qi[-q_len:], cache.ki, cache.ki_scale,
            cache.indexer_page_table,
        )
        check_topk_against_scores(
            scores, buffers.kv_block_indexes[-q_len:, 0], kv_len
        )
        oracle = reference_attention(
            buffers.q[-q_len:], cache, buffers.kv_block_indexes[-q_len:], kv_len
        )
        projected_activation = reconstruct_mxfp8(
            buffers.output[-q_len:], buffers.output_scale,
            row_offset=8 - q_len,
        )
        attention_value = projected_activation
        attention_delta = (attention_value.float() - oracle).abs()
        emit(backend, "attention_oracle_max_abs", attention_delta.max().item())
        residual_ref, norm_ref = reference_output_tail(
            projected_activation, weights.output_projection,
            slot.layer_input[-q_len:], weights.post_attention_norm,
            1.0e-6,
        )
        emit(backend, "tail_reference_residual_max_abs",
             (residual.float() - residual_ref.float()).abs().max().item())
        emit(backend, "tail_reference_norm_max_abs",
             (norm.float() - norm_ref.float()).abs().max().item())
        torch.testing.assert_close(
            residual.float(), residual_ref.float(), atol=0.05, rtol=0.05
        )
        torch.testing.assert_close(
            norm.float(), norm_ref.float(), atol=0.05, rtol=0.05
        )
        # E4M3 has at most one-eighth relative spacing within a normal binade.
        quantization_bound = 0.05 + 0.125 * oracle.abs()
        assert torch.all(attention_delta <= quantization_bound)
        del plan, runner
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--cache-scale", type=float, default=0.125)
    parser.add_argument("--skip-cache-write", action="store_true")
    parser.add_argument("--pdl", action="store_true")
    parser.add_argument("--cases", type=int, default=1)
    parser.add_argument("--random-kv", action="store_true")
    parser.add_argument("--boundary-cases", action="store_true")
    parser.add_argument("--zero-cache", action="store_true")
    parser.add_argument("--prewarm-mega-variant", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--backends", nargs="+",
        choices=("mega",),
        default=("mega",),
    )
    args = parser.parse_args()
    if args.boundary_cases and (args.random_kv or args.cases != 1):
        parser.error("--boundary-cases cannot be combined with --random-kv or --cases")
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 4
    mega_workspace = None
    contexts = {}
    if "mega" in args.backends:
        mega_workspace = create_tp_allreduce_workspace(
            rank, 4, f"attn-backend-check-{os.environ['MASTER_PORT']}", 0,
            device.index,
        )
        contexts["mega"] = AttentionTpContext(workspace=mega_workspace)
    try:
        generator = random.Random(args.seed)
        if args.boundary_cases:
            lengths = (
                8, 127, 128, 129, 2047, 2048, 2049,
                32767, 32768, 32769, 50000, 65535, 65536, 65537,
                131071, 131072, 131073, 262143, 262144, 262145,
                524287, 524288, 524289, 1048575, 1048576,
            )
            cases = [(q, kv) for kv in lengths for q in range(1, 9)]
        else:
            cases = [(args.q, args.kv_len)] * args.cases
        for case, (q_len, kv_len) in enumerate(cases):
            if args.random_kv:
                exponent = generator.uniform(3.0, 20.0)
                kv_len = min(1 << 20, max(8, int(2.0 ** exponent)))
                q_len = generator.randint(1, min(8, kv_len))
            if args.progress and rank == 0:
                print("case_start", case, q_len, kv_len, flush=True)
            run_case(
                device, rank, q_len, kv_len,
                args.seed + case * 7919, contexts, args.cache_scale,
                args.skip_cache_write or (case % 17 == 0 and len(cases) > 1),
                args.pdl, verbose=len(cases) == 1,
                random_cache=not args.zero_cache,
                prewarm_mega_variant=args.prewarm_mega_variant,
                backends=args.backends,
            )
            if rank == 0 and (case + 1) % 128 == 0:
                print("validated_cases", case + 1, flush=True)
            elif args.progress and rank == 0:
                print("case_complete", case, flush=True)
    except BaseException:  # noqa: BLE001 - abort all communicating peers
        traceback.print_exc()
        os._exit(1)
    finally:
        dist.barrier()
        if mega_workspace is not None:
            destroy_tp_allreduce_workspace(mega_workspace)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
