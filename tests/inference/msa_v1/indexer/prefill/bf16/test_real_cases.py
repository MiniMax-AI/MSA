"""BF16 indexer correctness on the shared prefill suite."""

import logging
import time

import pytest
import torch

from inference.msa_v1.indexer.prefill.bf16 import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)
from tests.inference.cases import (
    active_inference_suite,
    selected_msa_v1_prefill_test_cases,
)
from tests.inference.msa_v1.indexer.prefill.bf16.real_cases import (
    make_real_prefill_inputs,
)
from tests.inference.msa_v1.indexer.prefill.tp4_q8kv8.cases import (
    RealPrefillInputs as ReferenceInputs,
)
from tests.inference.msa_v1.indexer.prefill.tp4_q8kv8.reference import (
    assert_full_topk_quality,
    assert_topk_structure,
    expected_lengths,
    full_score_reference,
)

pytestmark = pytest.mark.gpu
_CASES = selected_msa_v1_prefill_test_cases()
_SHARD_COUNT = min(64, len(_CASES))
_SUITE = active_inference_suite()
logger = logging.getLogger(__name__)


@pytest.mark.parametrize("shard_index", range(_SHARD_COUNT))
def test_real_varlen_gemm_and_topk_cases(shard_index: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        pytest.skip("BF16 prefill indexer requires SM100 or SM103")

    for case_index, case in enumerate(_CASES[shard_index::_SHARD_COUNT]):
        num_index_heads = 1 if (shard_index + case_index) % 2 == 0 else 4
        inputs = make_real_prefill_inputs(
            case,
            num_index_heads=num_index_heads,
            device=device,
        )
        wrapper = BatchPrefillIndexerWithPagedKVCacheWrapper()
        wrapper.plan(
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            inputs.page_table,
            total_q=case.total_q,
            max_seqlen_q=case.max_query_len,
            max_seqlen_k=case.max_final_kv,
            num_index_heads=num_index_heads,
        )
        state = wrapper._plan_state
        assert state is not None
        state.proxy_scores.fill_(float("nan"))
        state.topk_indices.fill_(-777)
        torch.cuda.synchronize()
        started_at = time.perf_counter()
        result = wrapper.run(inputs.q, inputs.k_cache)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started_at
        logger.info("%s ran in %.3fms", case.case_id, elapsed * 1e3)
        assert elapsed < 30.0
        assert result.data_ptr() == state.topk_indices.data_ptr()

        lengths = expected_lengths(case).to(device=device)
        for head_index in range(num_index_heads):
            reference_inputs = ReferenceInputs(
                q=inputs.q[:, head_index : head_index + 1],
                k_cache=inputs.k_cache,
                page_table=inputs.page_table,
                cu_seqlens_q=inputs.cu_seqlens_q,
                cu_seqlens_k=inputs.cu_seqlens_k,
            )
            expected_scores = full_score_reference(case, reference_inputs)
            history_mask = torch.arange(
                case.max_cols,
                device=device,
            ).reshape(1, -1) < (lengths.reshape(-1, 1) - 1)
            torch.testing.assert_close(
                state.proxy_scores[head_index][history_mask],
                expected_scores[history_mask] / (128**0.5),
                atol=2.0e-4,
                rtol=2.0e-4,
            )
            assert_topk_structure(result[head_index], lengths)
            assert_full_topk_quality(
                expected_scores,
                lengths,
                result[head_index],
            )
        if _SUITE == "full":
            baseline = result.clone()
            for _ in range(2):
                repeated = wrapper.run(inputs.q, inputs.k_cache)
                torch.cuda.synchronize()
                assert torch.equal(baseline, repeated)
        del wrapper, inputs, result
        torch.cuda.empty_cache()
