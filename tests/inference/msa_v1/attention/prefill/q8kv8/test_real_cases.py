"""Q8KV8 prefill attention checks on the selected real-inference tier."""

from __future__ import annotations

import logging
import time

import pytest
import torch

from inference.msa_v1.attention.prefill.q8kv8 import (
    BatchPrefillWithPagedKVCacheWrapper,
)
from tests.inference.msa_v1.attention.prefill.q8kv8.real_cases import (
    make_real_prefill_attention_inputs,
)
from tests.inference.msa_v1.attention.prefill.q8kv8.reference import (
    paged_sparse_attention_reference,
)
from tests.inference.cases import (
    active_inference_suite,
    selected_msa_v1_prefill_test_cases,
)

pytestmark = pytest.mark.gpu
_CASES = selected_msa_v1_prefill_test_cases()
_SHARDS = min(128, len(_CASES))
_SUITE = active_inference_suite()
logger = logging.getLogger(__name__)


@pytest.mark.parametrize("shard_index", range(_SHARDS))
def test_real_varlen_attention_cases(shard_index: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
        pytest.skip("Q8KV8 prefill attention requires SM100 or SM103")

    for case in _CASES[shard_index::_SHARDS]:
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
        torch.cuda.synchronize()
        start = time.perf_counter()
        actual_out, actual_lse = wrapper.run(
            inputs.q,
            (inputs.k_cache, inputs.v_cache),
            return_lse=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        logger.info("%s ran in %.3fms", case.case_id, elapsed * 1e3)
        assert elapsed < 30.0, (
            f"{case.case_id} exceeded the 30-second deadlock threshold"
        )

        expected_out, expected_lse = paged_sparse_attention_reference(
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
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
            rtol=1.0e-1,
        )
        torch.testing.assert_close(
            actual_lse,
            expected_lse,
            atol=3.0e-3,
            rtol=3.0e-3,
        )
        if _SUITE == "full":
            for _ in range(2):
                repeated_out, repeated_lse = wrapper.run(
                    inputs.q,
                    (inputs.k_cache, inputs.v_cache),
                    return_lse=True,
                )
                torch.cuda.synchronize()
                assert torch.equal(actual_out, repeated_out), case.case_id
                assert torch.equal(actual_lse, repeated_lse), case.case_id
        del wrapper, inputs, actual_out, actual_lse
        torch.cuda.empty_cache()
