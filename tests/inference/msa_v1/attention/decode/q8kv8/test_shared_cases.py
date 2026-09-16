"""Full-output Q8K8 checks over the shared decode correctness suite."""

from __future__ import annotations

import pytest
import torch

from inference.msa_v1.attention.decode.q8kv8 import (
    BatchDecodeWithPagedKVCacheWrapper,
)
from tests.inference.cases import active_inference_suite
from tests.inference.msa_v1.attention.decode.q8kv8.real_cases import (
    DecodeAttentionInputs,
    make_decode_attention_inputs,
    shared_workload_seed,
)
from tests.inference.msa_v1.attention.decode.q8kv8.reference import (
    assert_decode_topk_contract,
    decode_attention_reference,
)
from tests.inference.msa_v1.attention.decode.q8kv8.runtime import (
    compile_module,
    run_decode,
)
from tests.inference.msa_v1.decode.cases import (
    DecodeCorrectnessCase,
    correctness_cases,
    make_seq_lens,
)

pytestmark = pytest.mark.gpu
_SUITE = active_inference_suite()
_CASES = correctness_cases(_SUITE)
_SHARD_COUNT = min(128 if _SUITE == "full" else 16, len(_CASES))


def _require_backend() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
        pytest.skip("FlashInfer Q8K8 block-sparse decode requires SM100 or SM103")
    compile_module()
    return device


def _run(
    wrapper: BatchDecodeWithPagedKVCacheWrapper,
    inputs: DecodeAttentionInputs,
) -> torch.Tensor:
    return run_decode(wrapper, inputs)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    assert bool(torch.isfinite(actual_fp32).all())
    assert bool(torch.isfinite(expected_fp32).all())
    cosine = torch.nn.functional.cosine_similarity(
        actual_fp32.flatten(), expected_fp32.flatten(), dim=0
    )
    assert float(cosine) >= 0.999
    torch.testing.assert_close(actual_fp32, expected_fp32, atol=0.05, rtol=0.05)


def _run_and_check(
    case: DecodeCorrectnessCase, device: torch.device, num_q_heads: int
) -> None:
    inputs = make_decode_attention_inputs(
        make_seq_lens(case),
        seed=shared_workload_seed(case),
        device=device,
        q_len_per_req=case.q_len_per_req,
        page_layout=case.page_layout,
        num_q_heads=num_q_heads,
    )
    assert_decode_topk_contract(inputs)
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    wrapper.plan(
        inputs.topk_indices,
        inputs.page_table,
        inputs.seq_lens,
        q_len_per_req=inputs.q_len_per_req,
        num_q_heads=num_q_heads,
    )
    _run(wrapper, inputs)
    actual = _run(wrapper, inputs).clone()
    expected = decode_attention_reference(inputs)
    _assert_close(actual, expected)
    if _SUITE == "full" and case.deterministic:
        second = _run(wrapper, inputs).clone()
        third = _run(wrapper, inputs)
        assert torch.equal(actual, second)
        assert torch.equal(actual, third)


@pytest.mark.parametrize("num_q_heads", (32, 64), ids=("gqa8", "gqa16"))
@pytest.mark.parametrize("shard_index", range(_SHARD_COUNT))
def test_canonical_decode_attention_cases(shard_index: int, num_q_heads: int) -> None:
    device = _require_backend()
    for case in _CASES[shard_index::_SHARD_COUNT]:
        _run_and_check(case, device, num_q_heads)
        torch.cuda.empty_cache()
