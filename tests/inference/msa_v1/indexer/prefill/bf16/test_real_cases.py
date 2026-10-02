"""BF16 indexer correctness on the shared prefill suite."""

from contextlib import contextmanager
from functools import partial
from unittest.mock import patch

import pytest
import torch

from inference.msa_v1.attention.prefill.bf16 import (
    BatchPrefillWithPagedKVCacheWrapper,
)
from inference.msa_v1.indexer._common.topk_select.build import load_extension
from inference.msa_v1.indexer.prefill.bf16 import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)
from inference.msa_v1.indexer.prefill.bf16.interface import _compile_gemm
from tests.inference.msa_v1.attention.decode.runtime import cuda_execution, run_cuda
from tests.inference.cases import (
    active_inference_suite,
    selected_msa_v1_prefill_test_cases,
)
from tests.inference.msa_v1.attention.prefill.q8kv8.reference import (
    paged_sparse_attention_reference,
)
from tests.inference.msa_v1.indexer.prefill.bf16.real_cases import (
    make_real_prefill_inputs,
)
from tests.inference.msa_v1.indexer.prefill.q8kv8.cases import (
    RealPrefillInputs as ReferenceInputs,
)
from tests.inference.msa_v1.indexer.prefill.q8kv8.reference import (
    assert_full_topk_quality,
    assert_topk_structure,
    expected_lengths,
    full_score_reference,
)

pytestmark = pytest.mark.gpu
_CASES = selected_msa_v1_prefill_test_cases()
_SHARD_COUNT = min(64, len(_CASES))
_SUITE = active_inference_suite()


@pytest.mark.parametrize("shard_index", range(_SHARD_COUNT))
@pytest.mark.parametrize("num_index_heads", (1, 2, 4))
def test_real_varlen_gemm_and_topk_cases(
    shard_index: int, num_index_heads: int
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        pytest.skip("BF16 prefill indexer requires SM100 or SM103")

    for case in _CASES[shard_index::_SHARD_COUNT]:
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
        # Prepare compilation before starting the execution watchdog.
        _compile_gemm(
            inputs.q.view(-1, 128), inputs.k_cache[:, 0].permute(1, 2, 0), state
        )
        load_extension()
        state.proxy_scores.fill_(float("nan"))
        state.topk_indices.fill_(-777)
        result = run_cuda(case.case_id, partial(wrapper.run, inputs.q, inputs.k_cache))
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
                repeated = run_cuda(
                    case.case_id, partial(wrapper.run, inputs.q, inputs.k_cache)
                )
                assert torch.equal(baseline, repeated)
        _check_attention_chain(case, inputs, result, num_index_heads)
        del wrapper, inputs, result
        torch.cuda.empty_cache()


def _check_attention_chain(case, inputs, topk_indices, num_index_heads: int) -> None:
    """Consume the actual indexer output without sorting or replacing its pages."""
    device = inputs.q.device
    generator = torch.Generator(device=device).manual_seed(case.seed + 1)
    gqa_ratio = 8
    q_shape = (case.total_q, num_index_heads * gqa_ratio, 128)
    kv_shape = (inputs.k_cache.shape[0], num_index_heads, 128, 128)
    q = (torch.randn(q_shape, generator=generator, device=device) * 0.25).bfloat16()
    k = (torch.randn(kv_shape, generator=generator, device=device) * 0.25).bfloat16()
    v = (torch.randn(kv_shape, generator=generator, device=device) * 0.25).bfloat16()
    attention = BatchPrefillWithPagedKVCacheWrapper()
    attention.plan(
        topk_indices,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        inputs.page_table,
        num_q_heads=num_index_heads * gqa_ratio,
        num_kv_heads=num_index_heads,
        total_k=sum(case.final_kv_lens),
        total_rows=sum((length + 127) // 128 for length in case.final_kv_lens),
        max_seqlen_q=case.max_query_len,
        max_seqlen_k=case.max_final_kv,
    )
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=device)
    # Existing launch ranges start after compilation, so a cold compile cannot
    # consume the execution timeout. Preserve the production launch sequence.
    launch_ranges = {"Fwd_PageKV_SparseAttn", "K2_Combine"}
    observed_ranges = set()
    original_range = torch.cuda.nvtx.range

    @contextmanager
    def timed_launch_range(message, *args, **kwargs):
        with original_range(message, *args, **kwargs):
            if message in launch_ranges:
                observed_ranges.add(message)
                with cuda_execution(f"{case.case_id} {message}"):
                    yield
            else:
                yield

    with patch.object(torch.cuda.nvtx, "range", timed_launch_range):
        attention.run(q, (k, v), out=out, lse=lse)
    assert observed_ranges == launch_ranges
    run_cuda(
        f"{case.case_id} attention",
        partial(attention.run, q, (k, v), out=out, lse=lse),
    )
    expected_out, expected_lse = paged_sparse_attention_reference(
        q, k, v, inputs.page_table, topk_indices, case.query_lens, case.final_kv_lens
    )
    assert bool(torch.isfinite(out).all())
    assert bool(torch.isfinite(lse).all())
    torch.testing.assert_close(out.float(), expected_out, atol=3.0e-2, rtol=3.0e-2)
    torch.testing.assert_close(lse, expected_lse, atol=1.0e-3, rtol=1.0e-3)
