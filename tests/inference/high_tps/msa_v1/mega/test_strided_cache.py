"""Page-strided K/V/KI caches must preserve cache bytes and numerical outputs.

The strided caches are views into per-page slabs laid out like SGLang's
``MiniMaxMSAKVPool``: one MAIN row per page holds k, k_scale, v, and v_scale and
one INDEX row per page holds ki and ki_scale. A second, padded layout puts
16-byte gaps between components and checks that the kernels never touch them.

    torchrun --standalone --nproc-per-node=4 \\
        -m tests.inference.high_tps.msa_v1.mega.test_strided_cache --kv-len 8192
"""

import argparse
import json
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from inference.high_tps.msa_v1.common.communication import (
    create_tp_allreduce_workspace,
    destroy_tp_allreduce_workspace,
)
from inference.high_tps.msa_v1.common.jit import (
    get_m3_decode_mega_atten_variant,
)
from tests.inference.high_tps.msa_v1.common.m3_decode_fixture import (
    configure_router,
    launch_mega_atten,
    make_model_weights,
    make_slot,
    synchronize_checked,
)

DATA_PAGE_BYTES = 128 * 64
SCALE_PAGE_BYTES = 128 * 8
GAP_SENTINEL = 0x5A


@dataclass(frozen=True)
class SlabLayout:
    name: str
    main_row: int
    k: int
    k_scale: int
    v: int
    v_scale: int
    index_row: int
    ki: int
    ki_scale: int


LAYOUTS = (
    # Exactly SGLang's MiniMaxMSAKVPool rows.
    SlabLayout("sglang", 18432, 0, 8192, 9216, 17408, 9216, 0, 8192),
    # Gaps before/between/after components (K/V data stays 32-byte aligned).
    SlabLayout("padded", 18528, 32, 8240, 9280, 17488, 9264, 16, 8224),
)

CACHE_FIELDS = ("k", "v", "k_scale", "v_scale", "indexer_k", "indexer_k_scale")


def _bytes(tensor):
    return tensor.view(torch.uint8)


def _attach_strided_cache(slot, layout):
    """Replace the slot's dense caches with slab views holding identical bytes."""
    pages = slot.k.shape[0]
    device = slot.k.device
    main = torch.full(
        (pages, layout.main_row), GAP_SENTINEL, dtype=torch.uint8, device=device
    )
    index = torch.full(
        (pages, layout.index_row), GAP_SENTINEL, dtype=torch.uint8, device=device
    )

    def view(slab, offset, size, shape, dtype):
        return slab[:, offset:offset + size].view(shape).view(dtype)

    strided = {
        "k": view(main, layout.k, DATA_PAGE_BYTES, (pages, 1, 128, 64), torch.uint8),
        "k_scale": view(
            main, layout.k_scale, SCALE_PAGE_BYTES, (pages, 1, 128, 8),
            torch.float8_e4m3fn,
        ),
        "v": view(main, layout.v, DATA_PAGE_BYTES, (pages, 1, 128, 64), torch.uint8),
        "v_scale": view(
            main, layout.v_scale, SCALE_PAGE_BYTES, (pages, 1, 128, 8),
            torch.float8_e4m3fn,
        ),
        # SGLang's index getters return 3-D [P, 128, row] views.
        "indexer_k": view(
            index, layout.ki, DATA_PAGE_BYTES, (pages, 128, 64), torch.uint8
        ),
        "indexer_k_scale": view(
            index, layout.ki_scale, SCALE_PAGE_BYTES, (pages, 128, 8),
            torch.float8_e4m3fn,
        ),
    }
    for field, tensor in strided.items():
        dense = getattr(slot, field)
        _bytes(tensor).view(dense.shape).copy_(_bytes(dense))
        setattr(slot, field, tensor)
    assert slot.k.stride(0) == layout.main_row
    assert slot.indexer_k_scale.stride(0) == layout.index_row
    assert not slot.k.is_contiguous() and not slot.indexer_k.is_contiguous()

    main_mask = torch.ones(layout.main_row, dtype=torch.bool, device=device)
    for offset, size in (
        (layout.k, DATA_PAGE_BYTES), (layout.k_scale, SCALE_PAGE_BYTES),
        (layout.v, DATA_PAGE_BYTES), (layout.v_scale, SCALE_PAGE_BYTES),
    ):
        main_mask[offset:offset + size] = False
    index_mask = torch.ones(layout.index_row, dtype=torch.bool, device=device)
    for offset, size in ((layout.ki, DATA_PAGE_BYTES), (layout.ki_scale, SCALE_PAGE_BYTES)):
        index_mask[offset:offset + size] = False
    return (main, main_mask), (index, index_mask)


def _check_gaps(slabs, layout):
    for slab, mask in slabs:
        gaps = slab[:, mask]
        if gaps.numel() and not torch.all(gaps == GAP_SENTINEL):
            raise AssertionError(f"{layout.name}: kernel wrote bytes outside cache components")


def _assert_equal(label, actual, expected):
    if actual.dtype in (torch.bfloat16, torch.float16, torch.float32):
        # TP reductions may change FP32 addition order across placements.
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0.05, atol=0.05)
        return
    if not torch.equal(_bytes(actual).flatten(), _bytes(expected).flatten()):
        raise AssertionError(f"{label}: strided result differs from dense result")


def _compare_caches(label, strided, dense):
    for field in CACHE_FIELDS:
        a = _bytes(getattr(strided, field)).reshape(-1, 128, getattr(dense, field).shape[-1])
        _assert_equal(f"{label} {field}", a, getattr(dense, field))


def _check_written_rows(label, before, slot, q):
    """Only the cache rows selected by valid slot_mapping entries may change."""
    written = slot.slot_mapping[8 - q:].tolist()
    for field in CACHE_FIELDS:
        row_bytes = getattr(slot, field).shape[-1]
        old = _bytes(before[field]).reshape(-1, row_bytes)
        new = _bytes(getattr(slot, field)).reshape(-1, row_bytes)
        changed = torch.nonzero((old != new).any(dim=1)).flatten().tolist()
        if not set(changed) <= set(written):
            raise AssertionError(
                f"{label} {field}: rows {sorted(set(changed) - set(written))} "
                "changed outside slot_mapping"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--q", type=int, nargs="+", default=[8, 3])
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("gloo")
    if dist.get_world_size() != 4:
        raise RuntimeError("strided cache test requires single-node TP4")

    workspaces = []
    try:
        get_m3_decode_mega_atten_variant()
        dist.barrier()
        group_id = f"strided-cache-{os.environ['MASTER_PORT']}"
        dense_workspace = create_tp_allreduce_workspace(
            rank, 4, group_id + "-dense", 0, device.index
        )
        workspaces.append(dense_workspace)
        strided_workspace = create_tp_allreduce_workspace(
            rank, 4, group_id + "-strided", 0, device.index
        )
        workspaces.append(strided_workspace)

        for layout in LAYOUTS:
            for q in args.q:
                label = f"{layout.name} q{q}"
                seed = 20261019 + q
                weights = make_model_weights(device, seed + rank * 100003, zero_math=False)
                dense = make_slot(device, rank, args.kv_len, seed, weights)
                strided = make_slot(device, rank, args.kv_len, seed, weights)
                for slot in (dense, strided):
                    slot.post_attention_rms_weight.fill_(0.125)
                    slot.slot_mapping[:8 - q] = -1
                    configure_router(slot, slot.layer_input, min(20, 4 * q), q_len=q)
                    # Start every compared output from identical bytes.
                    for tensor in (slot.q, slot.indexer_q, slot.output, slot.output_sf,
                                   slot.router_indices, slot.router_weights,
                                   slot.split_attention_residual, slot.post_attention_norm):
                        _bytes(tensor).zero_()
                slabs = _attach_strided_cache(strided, layout)
                _compare_caches(f"{label} initial", strided, dense)
                before = {field: getattr(dense, field).clone() for field in CACHE_FIELDS}

                # Attention-only MegaKernel shares the same cache code.
                dist.barrier()
                launch_mega_atten(dense, dense_workspace, materialize_topk=True)
                launch_mega_atten(strided, strided_workspace, materialize_topk=True)
                synchronize_checked(device, f"{label} mega_atten")
                dist.barrier()
                _check_written_rows(f"{label} mega_atten", before, dense, q)
                _compare_caches(f"{label} mega_atten", strided, dense)
                _check_gaps(slabs, layout)
                for name in ("kv_block_indexes", "output", "split_attention_residual",
                             "post_attention_norm"):
                    _assert_equal(f"{label} mega_atten {name}", getattr(strided, name),
                                  getattr(dense, name))

                record = {"rank": rank, "case": label, "kv_len": args.kv_len,
                          "cache_bit_identical": True, "outputs_within_tolerance": True}
                print(json.dumps(record), flush=True)
                dist.barrier()
        if rank == 0:
            print("PASS page-strided cache matches dense cache", flush=True)
    finally:
        for workspace in reversed(workspaces):
            destroy_tp_allreduce_workspace(workspace)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
