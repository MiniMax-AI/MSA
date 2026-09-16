"""Q8KV4 prefill attention checks on the selected real-inference tier."""

from __future__ import annotations

import logging
import time

import pytest
import torch

from inference.msa_v1.attention.prefill.q8kv4 import (
    BatchPrefillWithPagedKVCacheWrapper,
)
from tests.inference.msa_v1.attention.prefill.q8kv4.real_cases import (
    make_real_prefill_attention_inputs,
)
from tests.inference.msa_v1.attention.prefill.q8kv4.reference import (
    assert_attention_topk_contract,
    paged_sparse_attention_reference,
)
from tests.inference.cases import (
    active_inference_suite,
    selected_msa_v1_prefill_test_cases,
)


pytestmark = pytest.mark.gpu
_CASES = selected_msa_v1_prefill_test_cases()
_SUITE = active_inference_suite()
logger = logging.getLogger(__name__)


def _require_sm100_or_sm103() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
        pytest.skip("Q8KV4 prefill attention requires SM100 or SM103")
    return device


def _run_case(case, device: torch.device, num_kv_heads: int) -> None:
    inputs = make_real_prefill_attention_inputs(
        case, device=device, num_kv_heads=num_kv_heads
    )
    assert_attention_topk_contract(case, inputs.topk_indices)
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
    torch.cuda.synchronize()
    compile_start = time.perf_counter()
    wrapper.run(
        inputs.q,
        (inputs.packed_k, inputs.packed_v),
        kv_cache_sf=(inputs.k_scale, inputs.v_scale),
        return_lse=True,
    )
    torch.cuda.synchronize()
    logger.info("Compiled and warmed in %.3fs", time.perf_counter() - compile_start)
    start = time.perf_counter()
    actual_out, actual_lse = wrapper.run(
        inputs.q,
        (inputs.packed_k, inputs.packed_v),
        kv_cache_sf=(inputs.k_scale, inputs.v_scale),
        return_lse=True,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    logger.info("%s ran in %.3fms", case.case_id, elapsed * 1e3)
    assert elapsed < 30.0, f"{case.case_id} exceeded the 30-second deadlock threshold"

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

    assert bool(torch.isfinite(actual_out).all()), case.case_id
    assert bool(torch.isfinite(actual_lse).all()), case.case_id
    torch.testing.assert_close(
        actual_out.float(),
        expected_out,
        atol=3.0e-2,
        rtol=3.0e-2,
        msg=lambda message: f"{case.case_id}: output mismatch\n{message}",
    )
    torch.testing.assert_close(
        actual_lse,
        expected_lse,
        atol=5.0e-4,
        rtol=5.0e-4,
        msg=lambda message: f"{case.case_id}: LSE mismatch\n{message}",
    )
    if _SUITE == "full":
        actual_out, actual_lse = actual_out.clone(), actual_lse.clone()
        for _ in range(2):
            repeated_out, repeated_lse = wrapper.run(
                inputs.q,
                (inputs.packed_k, inputs.packed_v),
                kv_cache_sf=(inputs.k_scale, inputs.v_scale),
                return_lse=True,
            )
            torch.cuda.synchronize()
            assert torch.equal(actual_out, repeated_out), case.case_id
            assert torch.equal(actual_lse, repeated_lse), case.case_id


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.case_id)
@pytest.mark.parametrize("num_kv_heads", (4, 1), ids=("tp1", "tp4"))
def test_real_varlen_attention_cases(case, num_kv_heads: int) -> None:
    """Check every selected real shape as an independently timed test case."""

    device = _require_sm100_or_sm103()
    _run_case(case, device, num_kv_heads)
    torch.cuda.empty_cache()
