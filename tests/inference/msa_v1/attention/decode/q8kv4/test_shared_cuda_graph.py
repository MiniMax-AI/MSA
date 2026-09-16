"""CUDA Graph replay checks on canonical decode benchmark workloads."""

from __future__ import annotations

import pytest
import torch

from benchmarks.inference.msa_v1.decode.cases import (
    FULL_CASES,
    DecodeBenchmarkCase,
    make_seq_lens,
)
from inference.msa_v1.attention.decode.q8kv4 import (
    BatchDecodeWithPagedKVCacheWrapper,
)
from tests.inference.msa_v1.attention.decode.q8kv4.real_cases import (
    make_decode_attention_inputs,
    shared_workload_seed,
)
from tests.inference.msa_v1.attention.decode.q8kv4.reference import (
    assert_decode_topk_contract,
    decode_attention_reference,
)
from tests.inference.msa_v1.attention.decode.q8kv4.runtime import (
    compile_modules,
)
from tests.inference.msa_v1.attention.decode.runtime import run_cuda

pytestmark = pytest.mark.gpu
_GRAPH_CASE_KEYS = ((32, 1_000), (64, 10_000), (128, 100_000), (8, 200_000))
_GRAPH_CASES = tuple(
    next(case for case in FULL_CASES if (case.batch_size, case.nominal_seq_len) == key)
    for key in _GRAPH_CASE_KEYS
)


def _require_supported_gpu() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3), (10, 7)}:
        pytest.skip("Q8KV4 decode attention requires SM100, SM103, or SM107")
    return device


@pytest.mark.parametrize("case", _GRAPH_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("gqa_ratio", (8, 16), ids=("gqa8", "gqa16"))
def test_shared_workload_cuda_graph_replay(
    case: DecodeBenchmarkCase, gqa_ratio: int
) -> None:
    """Prove that a captured production-shape launch performs real work."""

    device = _require_supported_gpu()
    inputs = make_decode_attention_inputs(
        make_seq_lens(case),
        seed=shared_workload_seed(case),
        device=device,
        q_len_per_req=case.q_len_per_req,
        page_layout="permuted",
        num_q_heads=gqa_ratio * 4,
    )
    assert_decode_topk_contract(inputs)
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    wrapper.plan(
        inputs.topk_indices,
        inputs.page_table,
        inputs.seq_lens,
        q_len_per_req=inputs.q_len_per_req,
        num_q_heads=gqa_ratio * 4,
    )
    out = torch.empty_like(inputs.q, dtype=torch.bfloat16)
    compile_modules(gqa_ratio, device)
    run_cuda(
        "shared graph warmup",
        lambda: wrapper.run(
            inputs.q,
            (inputs.packed_k, inputs.packed_v),
            kv_cache_sf=(inputs.k_scale, inputs.v_scale),
            out=out,
        ),
    )
    baseline = out.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(
            inputs.q,
            (inputs.packed_k, inputs.packed_v),
            kv_cache_sf=(inputs.k_scale, inputs.v_scale),
            out=out,
        )
    out.fill_(-777)
    run_cuda("shared graph replay", graph.replay)
    assert torch.equal(out, baseline)

    expected = decode_attention_reference(inputs)
    torch.testing.assert_close(
        out.float(),
        expected.float(),
        atol=5.0e-2,
        rtol=5.0e-2,
    )
    torch.cuda.empty_cache()
