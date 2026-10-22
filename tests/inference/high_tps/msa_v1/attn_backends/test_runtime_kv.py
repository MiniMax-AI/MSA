"""TP4 runtime-KV Graph replay with independent FP32 page and output references.

Run with torchrun --standalone --nproc-per-node=4 -m
 tests.inference.high_tps.msa_v1.attn_backends.test_runtime_kv
"""

import argparse
import ctypes
import dataclasses
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from inference.high_tps.msa_v1.common import jit
from tests.inference.high_tps.msa_v1.attn_backends.test_backends import (
    reconstruct_mxfp8,
    reference_attention,
    reference_output_tail,
)
from tests.inference.high_tps.msa_v1.common.execution import ExecutionResources
from tests.inference.high_tps.msa_v1.common.m3_decode_fixture import (
    ModelWeights,
    graph_summary,
    launch_mega_atten,
    make_slot,
    synchronize_checked,
)

rank = int(os.environ.get("RANK", "0"))
device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))


def phase(message):
    print(f"{time.strftime('%FT%T')} rank={rank} {message}", flush=True)


def weights(seed):
    generator = torch.Generator(device=device).manual_seed(seed + rank * 100003)
    # The attention-only fixture never consumes MoE matrices. Keep empty
    # placeholders instead of preparing gigabytes of unrelated expert weights.
    values = {field.name: torch.empty(0, dtype=torch.uint8, device=device)
              for field in dataclasses.fields(ModelWeights)}
    for name, shape in (("attention_projection", (2560, 6144)),
                        ("output_projection", (6144, 2048))):
        values[name] = (torch.randn(shape, generator=generator, device=device) * 0.03125).to(torch.float8_e4m3fn)
        values[name + "_sf"] = torch.full((shape[0] * shape[1] // 32,), 127,
                                          dtype=torch.uint8, device=device)
    return ModelWeights(**values)


class Case:
    kind = "attn_mega"
    kernel_count = 1
    runner = None

    def __init__(self, resources, length, seed, capacity=None, enable_placement=False):
        self.resources = resources
        self.enable_placement = enable_placement
        self.slot = make_slot(device, rank, capacity or length, seed, weights(seed))
        self.length = length
        generator = torch.Generator(device=device).manual_seed(seed + rank * 100003 + 17)
        for name in ("k", "v", "indexer_k"):
            getattr(self.slot, name)[:-1].random_(0, 256, generator=generator)
        self.set_length(length)

    def set_length(self, length):
        self.length = self.slot.kv_len = length
        self.slot.plan.kv_segment_lens.fill_(length)
        self.slot.slot_mapping.copy_(torch.arange(length - 8, length, device=device))

    def run(self):
        launch_mega_atten(self.slot, self.resources.attention_workspace, materialize_topk=True,
                         enable_placement=self.enable_placement)


def fp32_pages(case):
    slot = case.slot
    pages = (case.length + 127) // 128
    physical = slot.indexer_page_table[:pages].long()
    packed = slot.indexer_k[physical].view(pages, 128, 64)
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).view(pages, 128, 128).long()
    table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=device)
    scales = slot.indexer_k_scale[physical].float().view(pages, 128, 8).repeat_interleave(16, -1)
    key = (table[nibbles] * scales).to(torch.float8_e4m3fn).float().reshape(-1, 128)
    scores = slot.indexer_q.float() @ key.T
    tokens = torch.arange(pages * 128, device=device)
    queries = case.length - 8 + torch.arange(8, device=device)
    scores.masked_fill_(tokens[None, :] > queries[:, None], -torch.inf)
    scores = scores.view(8, pages, 128).amax(-1).cpu().numpy()
    expected = np.full((8, 16), -1, dtype=np.int32)
    for row in range(8):
        visible = (case.length - 8 + row) // 128 + 1
        if visible <= 16:
            expected[row, :visible] = np.arange(visible)
        else:
            ids = np.arange(visible - 1)
            historical = np.lexsort((ids, -scores[row, :visible - 1]))[:15]
            expected[row] = np.sort(np.append(historical, visible - 1))
    return expected



def check(case, label):
    slot = case.slot
    synchronize_checked(device, label)
    np.testing.assert_array_equal(slot.kv_block_indexes.cpu().numpy().reshape(8, 16), fp32_pages(case))
    cache = SimpleNamespace(k=slot.k, v=slot.v, k_scale=slot.k_scale,
                                   v_scale=slot.v_scale, kv_indices=slot.kv_indices)
    oracle = reference_attention(slot.q, cache, slot.kv_block_indexes, case.length, splits=8)
    activation = reconstruct_mxfp8(slot.output, slot.output_sf)
    # Same FP8 storage bound as test_backends; compare the complete tensor.
    error = (activation - oracle).abs()
    assert torch.all(error <= 0.05 + 0.125 * oracle.abs()), (label, error.max().item())
    residual, norm = reference_output_tail(activation, slot.weights.output_projection,
        slot.layer_input, slot.post_attention_rms_weight, 1e-6)
    for name, expected in (("split_attention_residual", residual), ("post_attention_norm", norm)):
        value = getattr(slot, name)
        assert torch.isfinite(value).all(), (label, name)
        torch.testing.assert_close(value.float(), expected.float(), rtol=0.05, atol=0.05)
    phase("PASS " + label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-placement", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 4
    resources = None
    try:
        jit.get_m3_decode_mega_atten_variant(enable_placement=args.enable_placement)
        dist.barrier()
        resources = ExecutionResources("attn_mega", rank, device,
                                       f"runtime-kv-{os.environ['MASTER_PORT']}")
        case = Case(resources, 1048576, 20261021, capacity=1048576,
                    enable_placement=args.enable_placement)
        initial = {name: getattr(case.slot, name).clone() for name in
                   ("k", "v", "indexer_k", "k_scale", "v_scale", "indexer_k_scale")}
        dist.barrier()
        case.run()
        synchronize_checked(device, "warmup")
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            case.run()
        info = graph_summary(graph)
        assert info["kernel_count"] == 1 and set(info["node_types"]) == {"kernel"}, info
        compiled = dict(jit._m3_decode_variant_manager._loaded)
        runtime = ctypes.CDLL(str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "lib64/libcudart.so"))
        runtime.cudaMemset.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t)
        lengths = [1024, 131072, 1048576, 32768, 1024, 8, 127, 128, 129,
                   2048, 2049, 2056, 32767, 32769, 32896, 32897, 65536,
                   131071, 131073, 262144, 262145, 524288, 524289]
        for index, length in enumerate(lengths):
            for name, value in initial.items():
                getattr(case.slot, name).copy_(value)
            case.set_length(length)
            synchronize_checked(device, "reset inputs")
            # Poison score scratch only, preserving the workspace invocation state.
            assert runtime.cudaMemset(resources.attention_workspace.unicast_ptr + 299008,
                                      0x7f, 8 * 8192 * 4) == 0
            dist.barrier()
            graph.replay()
            check(case, f"kv={length} replay={index}")
            assert jit._m3_decode_variant_manager._loaded == compiled
        small = Case(resources, 1024, 20261021, enable_placement=args.enable_placement)
        dist.barrier()
        small.run()
        check(small, "changed capacity")
        assert jit._m3_decode_variant_manager._loaded == compiled
        phase("PASS runtime KV, one binary, repeated single-node Graph")
        # A different static mode may add exactly one binary, then both captured
        # graphs must safely advance the same workspace's invocation generation.
        case.enable_placement = not args.enable_placement
        jit.get_m3_decode_mega_atten_variant(enable_placement=case.enable_placement)
        # Finish rank-local compilation/module loading before starting TP waits.
        dist.barrier()
        case.set_length(1024)
        case.run()
        check(case, "alternate placement warmup")
        alternate = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(alternate, stream=stream):
            case.run()
        assert graph_summary(alternate)["kernel_count"] == 1
        for index in range(4):
            dist.barrier()
            (graph if index % 2 == 0 else alternate).replay()
            check(case, f"alternating placement Graph {index}")
        assert len(jit._m3_decode_variant_manager._loaded) == len(compiled) + 1
        alternate.reset()
        graph.reset()
        resources.close()
        resources = None
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - terminate communicating peers on every failure
        traceback.print_exc()
        # A rank failure must abort the group instead of waiting in collective teardown.
        os._exit(1)


if __name__ == "__main__":
    main()
