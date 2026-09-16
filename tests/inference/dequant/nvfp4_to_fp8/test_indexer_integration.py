"""End-to-end NVFP4 dequant plus Q8K8 prefill indexer test."""

from __future__ import annotations

import pytest
import torch

from inference.dequant import dequantize_nvfp4_to_fp8
from inference.msa_v1.indexer.prefill.tp4_q8kv8 import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)
from tests.inference.cases import selected_msa_v1_prefill_test_cases
from tests.inference.msa_v1.indexer.prefill.tp4_q8kv8.cases import (
    RealPrefillInputs,
    make_real_prefill_inputs,
)
from tests.inference.msa_v1.indexer.prefill.tp4_q8kv8.reference import (
    assert_sampled_scores,
    assert_sampled_topk_quality,
    assert_score_structure,
    assert_topk_structure,
)

pytestmark = pytest.mark.gpu


def test_dequantized_cache_runs_through_q8k8_indexer() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
        pytest.skip("TP4 Q8K8 prefill requires SM100 or SM103")

    case = selected_msa_v1_prefill_test_cases()[1]
    baseline_inputs = make_real_prefill_inputs(case, device=device)
    physical_pages = baseline_inputs.k_cache.shape[0]
    generator = torch.Generator(device=device).manual_seed(case.seed + 1)
    packed_k = torch.randint(
        0,
        256,
        (physical_pages, 1, 128, 64),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    k_scale = torch.full(
        (physical_pages, 1, 128, 8),
        0.25,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    k_cache = dequantize_nvfp4_to_fp8(packed_k, k_scale)
    inputs = RealPrefillInputs(
        q=baseline_inputs.q,
        k_cache=k_cache,
        page_table=baseline_inputs.page_table,
        cu_seqlens_q=baseline_inputs.cu_seqlens_q,
        cu_seqlens_k=baseline_inputs.cu_seqlens_k,
    )

    wrapper = BatchPrefillIndexerWithPagedKVCacheWrapper()
    wrapper.plan(
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        inputs.page_table,
        total_q=case.total_q,
        max_seqlen_q=case.max_query_len,
        max_seqlen_k=case.max_final_kv,
    )
    state = wrapper._proxy_score.plan_state
    state.proxy_scores.fill_(float("nan"))
    state.topk_indices.fill_(-777)
    result = wrapper.run(inputs.q, inputs.k_cache)
    torch.cuda.synchronize()

    assert_score_structure(case, state.proxy_scores, state.num_valid_pages)
    assert_sampled_scores(case, inputs, state.proxy_scores)
    assert_topk_structure(result, state.num_valid_pages)
    assert_sampled_topk_quality(
        case,
        state.proxy_scores,
        state.num_valid_pages,
        result,
    )
