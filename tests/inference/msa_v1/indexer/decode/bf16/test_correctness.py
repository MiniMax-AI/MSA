"""BF16 decode scores and TopK on the shared workload and all query lengths."""

import pytest
import torch

from inference.msa_v1.indexer.decode.bf16 import (
    BatchDecodeIndexerWithPagedKVCacheWrapper,
)
from tests.inference.cases import active_inference_suite
from tests.inference.msa_v1.decode.cases import correctness_cases, make_seq_lens
from tests.inference.msa_v1.indexer._common.topk_select.reference import (
    assert_quantized_topk_contract,
)
from tests.inference.msa_v1.indexer.decode.bf16.reference import (
    indexer_gemm_reference,
    make_inputs,
)
from tests.inference.msa_v1.indexer.decode.q8kv8.test_correctness import _timed_run
from tests.inference.msa_v1.indexer.decode.test_multihead_runtime import _synchronize

pytestmark = pytest.mark.gpu
_SUITE = active_inference_suite()
_CASES = correctness_cases(_SUITE)
_SHARDS = tuple(_CASES[start : start + 8] for start in range(0, len(_CASES), 8))


def _check_inputs(values, heads, query_length):
    q, k, table, lengths = values
    wrapper = BatchDecodeIndexerWithPagedKVCacheWrapper()
    wrapper.plan(table, lengths, num_index_heads=heads, query_length=query_length)
    output = torch.empty(
        (heads, q.shape[0] * query_length, 16), dtype=torch.int32, device=q.device
    )
    # Separate score and TopK compilation from execution timing.
    wrapper._proxy_score._compile_and_bind(q, k, wrapper._proxy_scores)
    from inference.msa_v1.indexer._common.topk_select.build import load_extension

    load_extension()
    wrapper._proxy_scores.fill_(float("nan"))
    _timed_run("bf16 decode", lambda: wrapper.run(q, k, out=output))
    expected = indexer_gemm_reference(*values)
    local = (
        lengths[:, None] - query_length + torch.arange(query_length, device=q.device)
    ) // 128
    mask = torch.arange(table.shape[1], device=q.device).view(1, 1, -1) < local.reshape(
        1, -1, 1
    )
    actual = wrapper._proxy_scores.masked_select(mask)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, expected.masked_select(mask), atol=1e-4, rtol=1e-4
    )
    assert_quantized_topk_contract(
        expected.reshape(-1, table.shape[1]).cpu().numpy(),
        (local + 1).reshape(-1).repeat(heads).cpu().numpy(),
        output.reshape(-1, 16).cpu().numpy(),
    )
    if _SUITE == "full":
        saved = output.clone()
        for _ in range(2):
            wrapper.run(q, k, out=output)
            _synchronize()
            assert torch.equal(saved, output)


@pytest.mark.parametrize("heads", (1, 2, 4))
@pytest.mark.parametrize("shard", _SHARDS, ids=lambda cases: cases[0].case_id)
def test_shared_cases(heads, shard):
    for case in shard:
        lengths = make_seq_lens(case)
        pages = (int(lengths.max()) + 127) // 128
        values = make_inputs(
            case.batch_size,
            pages,
            lengths,
            seed=case.seed ^ int(case.case_id[-8:], 16),
            device=torch.device("cuda"),
            num_index_heads=heads,
            query_length=case.q_len_per_req,
        )
        _check_inputs(values, heads, case.q_len_per_req)


@pytest.mark.parametrize("heads", (1, 2, 4))
@pytest.mark.parametrize("query_length", range(1, 17))
def test_all_query_lengths(heads, query_length):
    lengths = torch.tensor([query_length, 129, 2053, 4096], dtype=torch.int32)
    values = make_inputs(
        4,
        32,
        lengths,
        seed=1701,
        device=torch.device("cuda"),
        num_index_heads=heads,
        query_length=query_length,
    )
    _check_inputs(values, heads, query_length)
