"""CUDA Graph replay checks on recorded Q8KV8 prefill shapes."""

from __future__ import annotations

import pytest
import torch

from datas.inference.cases import load_prefill_test_cases
from inference.msa_v1.attention.prefill.q8kv8 import (
    BatchPrefillWithPagedKVCacheWrapper,
)
from tests.inference.msa_v1.attention.prefill.q8kv8.real_cases import (
    make_real_prefill_attention_inputs,
)
from tests.inference.msa_v1.attention.prefill.q8kv8.reference import (
    sampled_real_case_reference,
)

pytestmark = pytest.mark.gpu
_CASES = load_prefill_test_cases("cuda_graph")


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.case_id)
def test_real_case_cuda_graph_replay(case) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3), (10, 7)}:
        pytest.skip("Q8KV8 prefill attention requires SM100, SM103, or SM107")
    inputs = make_real_prefill_attention_inputs(case, device=device)
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
    out = torch.empty((case.total_q, 64, 128), dtype=torch.bfloat16, device=device)
    lse = torch.empty((case.total_q, 64), dtype=torch.float32, device=device)
    wrapper.run(
        inputs.q,
        (inputs.k_cache, inputs.v_cache),
        out=out,
        lse=lse,
        return_lse=True,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(
            inputs.q,
            (inputs.k_cache, inputs.v_cache),
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
    rows, q_heads, expected_out, expected_lse = sampled_real_case_reference(
        case, inputs
    )
    torch.testing.assert_close(
        out[rows[:, None], q_heads].float(),
        expected_out,
        atol=3.0e-2,
        rtol=1.0e-1,
    )
    torch.testing.assert_close(
        lse[rows[:, None], q_heads],
        expected_lse,
        atol=3.0e-3,
        rtol=3.0e-3,
    )
