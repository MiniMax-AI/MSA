"""CUDA Graph replay checks on fixed real prefill shapes."""

from __future__ import annotations

import pytest
import torch

from datas.inference.cases import load_prefill_test_cases
from inference.msa_v1.attention.prefill.q8kv4 import (
    BatchPrefillWithPagedKVCacheWrapper,
)
from tests.inference.msa_v1.attention.prefill.q8kv4.real_cases import (
    make_real_prefill_attention_inputs,
)
from tests.inference.msa_v1.attention.prefill.q8kv4.reference import (
    paged_sparse_attention_reference,
)


pytestmark = pytest.mark.gpu
_GRAPH_CASES = load_prefill_test_cases("cuda_graph")


@pytest.mark.parametrize("case", _GRAPH_CASES, ids=lambda case: case.case_id)
@pytest.mark.parametrize("num_kv_heads", (4, 1), ids=("tp1", "tp4"))
def test_real_case_cuda_graph_replay(case, num_kv_heads: int) -> None:
    """Poison every output stage before replay to prove non-empty execution."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
        pytest.skip("Q8KV4 prefill attention requires SM100 or SM103")

    inputs = make_real_prefill_attention_inputs(
        case, device=device, num_kv_heads=num_kv_heads
    )
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    wrapper.plan(
        inputs.topk_indices,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        inputs.page_table,
        total_k=sum(case.final_kv_lens),
        total_rows=sum((length + 127) // 128 for length in case.final_kv_lens),
        max_seqlen_q=case.max_query_len,
        max_seqlen_k=case.max_final_kv,
    )
    out = torch.empty(
        (case.total_q, num_kv_heads * 16, 128),
        dtype=torch.bfloat16,
        device=device,
    )
    lse = torch.empty(
        (case.total_q, num_kv_heads * 16),
        dtype=torch.float32,
        device=device,
    )
    wrapper.run(
        inputs.q,
        (inputs.packed_k, inputs.packed_v),
        kv_cache_sf=(inputs.k_scale, inputs.v_scale),
        out=out,
        lse=lse,
        return_lse=True,
    )
    torch.cuda.synchronize()
    baseline_out = out.clone()
    baseline_lse = lse.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(
            inputs.q,
            (inputs.packed_k, inputs.packed_v),
            kv_cache_sf=(inputs.k_scale, inputs.v_scale),
            out=out,
            lse=lse,
            return_lse=True,
        )

    state = wrapper._plan_state
    assert state is not None
    state.o_partial.fill_(-777)
    state.lse_partial.fill_(float("nan"))
    out.fill_(-777)
    lse.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(out, baseline_out, atol=0, rtol=0)
    torch.testing.assert_close(lse, baseline_lse, atol=0, rtol=0)
    expected_out, expected_lse = paged_sparse_attention_reference(
        inputs.q,
        inputs.packed_k,
        inputs.packed_v,
        inputs.k_scale,
        inputs.v_scale,
        inputs.page_table,
        inputs.topk_indices,
        case.query_lens,
        case.final_kv_lens,
    )
    torch.testing.assert_close(
        out.float(),
        expected_out,
        atol=3.0e-2,
        rtol=3.0e-2,
    )
    torch.testing.assert_close(
        lse,
        expected_lse,
        atol=5.0e-4,
        rtol=5.0e-4,
    )
