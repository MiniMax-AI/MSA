"""Full-output checks over the canonical MSA v1 decode correctness suite."""

from __future__ import annotations

import importlib.metadata
import logging

import pytest
from packaging.version import Version
import torch

from inference.msa_v1.attention.decode.q8kv4 import (
    BatchDecodeWithPagedKVCacheWrapper,
)
from tests.inference.cases import active_inference_suite
from tests.inference.msa_v1.attention.decode.q8kv4.plan_inspection import (
    assert_split_count,
)
from tests.inference.msa_v1.attention.decode.q8kv4.real_cases import (
    DecodeAttentionInputs,
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
from tests.inference.msa_v1.decode.cases import (
    DecodeCorrectnessCase,
    correctness_cases,
    make_seq_lens,
)

pytestmark = pytest.mark.gpu
logger = logging.getLogger(__name__)
_SUITE = active_inference_suite()
_CASES = correctness_cases(_SUITE)
_SHARD_COUNT = min(128 if _SUITE == "full" else 16, len(_CASES))


def _require_supported_gpu() -> torch.device:
    version = importlib.metadata.version("nvidia-cutlass-dsl")
    assert Version(version) >= Version("4.5.2")
    logger.info(
        "CuTe DSL %s; Torch %s; CUDA %s", version, torch.__version__, torch.version.cuda
    )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3), (10, 7)}:
        pytest.skip("Q8KV4 decode attention requires SM100, SM103, or SM107")
    return device


def _run(
    wrapper: BatchDecodeWithPagedKVCacheWrapper,
    inputs: DecodeAttentionInputs,
) -> torch.Tensor:
    return run_cuda(
        inputs.page_layout,
        lambda: wrapper.run(
            inputs.q,
            (inputs.packed_k, inputs.packed_v),
            kv_cache_sf=(inputs.k_scale, inputs.v_scale),
        ),
    )


def _compile_and_warm(
    wrapper: BatchDecodeWithPagedKVCacheWrapper,
    inputs: DecodeAttentionInputs,
) -> None:
    ratio = inputs.q.shape[1] // inputs.packed_k.shape[1]
    compile_modules(ratio, inputs.q.device)
    _run(wrapper, inputs)


def _run_and_check(
    case: DecodeCorrectnessCase, device: torch.device, gqa_ratio: int
) -> None:
    inputs = make_decode_attention_inputs(
        make_seq_lens(case),
        seed=shared_workload_seed(case),
        device=device,
        q_len_per_req=case.q_len_per_req,
        page_layout=case.page_layout,
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
    _compile_and_warm(wrapper, inputs)
    actual = _run(wrapper, inputs).clone()
    expected = decode_attention_reference(inputs)
    assert bool(torch.isfinite(actual).all())
    assert bool(torch.isfinite(expected).all())
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=5.0e-2,
        rtol=5.0e-2,
    )
    if _SUITE == "full" and case.deterministic:
        second = _run(wrapper, inputs).clone()
        third = _run(wrapper, inputs).clone()
        assert torch.equal(actual, second)
        assert torch.equal(actual, third)


@pytest.mark.parametrize("shard_index", range(_SHARD_COUNT))
@pytest.mark.parametrize("gqa_ratio", (8, 16), ids=("gqa8", "gqa16"))
def test_canonical_decode_attention_cases(shard_index: int, gqa_ratio: int) -> None:
    device = _require_supported_gpu()
    for case in _CASES[shard_index::_SHARD_COUNT]:
        _run_and_check(case, device, gqa_ratio)
        torch.cuda.empty_cache()


@pytest.mark.parametrize("num_kv_splits", (1, 2, 4, 8))
@pytest.mark.parametrize("gqa_ratio", (8, 16), ids=("gqa8", "gqa16"))
@pytest.mark.parametrize("q_len_per_req", (8, 129, 257))
def test_split_variants_match_independent_reference(
    num_kv_splits: int,
    gqa_ratio: int,
    q_len_per_req: int,
) -> None:
    device = _require_supported_gpu()
    case = DecodeCorrectnessCase(
        case_id="decode_split_00000001",
        batch_size=3,
        q_len_per_req=q_len_per_req,
        nominal_seq_len=2_048,
        distribution="boundary",
        page_layout="permuted",
        seed=1701,
        production=False,
    )
    inputs = make_decode_attention_inputs(
        make_seq_lens(case),
        seed=shared_workload_seed(case),
        device=device,
        q_len_per_req=case.q_len_per_req,
        page_layout=case.page_layout,
        num_q_heads=gqa_ratio * 4,
    )
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    wrapper.plan(
        inputs.topk_indices,
        inputs.page_table,
        inputs.seq_lens,
        q_len_per_req=inputs.q_len_per_req,
        num_kv_splits=num_kv_splits,
        num_q_heads=gqa_ratio * 4,
    )
    _compile_and_warm(wrapper, inputs)
    actual = _run(wrapper, inputs)
    assert_split_count(wrapper, num_kv_splits)
    expected = decode_attention_reference(inputs)
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=5.0e-2,
        rtol=5.0e-2,
    )
