"""Complete-group block scores against an independent FP32 reference."""

import pytest
import torch

from inference.msa_v1.indexer.decode import BatchDecodeIndexerPlan
from inference.msa_v1.indexer.decode.interface import BatchDecodeBlockScoreWrapper
from tests.inference.cases import active_inference_suite
from tests.inference.msa_v1.attention.decode.runtime import run_cuda

# Smoke keeps one single-head and one multi-head query shape.
_HEADS_AND_QUERIES = (
    ((1, 1), (4, 3))
    if active_inference_suite() == "smoke"
    else tuple((heads, query) for heads in (1, 2, 4) for query in (1, 3))
)


def _reference(q, k, table, lengths, factor, include_partial_block=True):
    # Independently expand physical pages and then group 128 original tokens.
    keys = k[table.long()].float().repeat_interleave(factor, dim=-2).flatten(1, 2)
    scores = torch.einsum("bqhd,btd->bhqt", q.float(), keys)
    positions = torch.arange(keys.shape[1], device=q.device)
    query_positions = (
        lengths[:, None] - q.shape[1] + torch.arange(q.shape[1], device=q.device)
    )
    visible = positions[None, None] <= query_positions[:, :, None] - (factor - 1)
    scores = scores.masked_fill(~visible[:, None], -torch.inf)
    if keys.shape[1] % 128:
        scores = torch.nn.functional.pad(
            scores, (0, 128 - keys.shape[1] % 128), value=-torch.inf
        )
    scores = scores.unflatten(-1, (-1, 128)).amax(-1)
    if not include_partial_block:
        blocks = torch.arange(scores.shape[-1], device=q.device)
        scores.masked_fill_(
            blocks[None, None, None] >= (query_positions // 128)[:, None, :, None],
            -torch.inf,
        )
    return scores.permute(1, 0, 2, 3).flatten(1, 2)


def _run(scorer, q, k, out=None):
    # Compile first so the 30-second watchdog times only kernel execution.
    shape = (q.shape[2], q.shape[0] * q.shape[1], scorer._max_pages)
    scorer._compile_and_bind(
        q, k, torch.empty(shape, device=q.device) if out is None else out
    )
    return run_cuda("block scores", lambda: scorer.run(q, k, out=out))


@pytest.mark.gpu
@pytest.mark.parametrize("include_partial_block", (False, True))
@pytest.mark.parametrize("page_size", (64, 128))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float8_e4m3fn))
@pytest.mark.parametrize("factor", (4, 2, 1))
@pytest.mark.parametrize("heads,query_length", _HEADS_AND_QUERIES)
def test_block_scores_complete_groups_and_graph(
    dtype, factor, query_length, heads, page_size, include_partial_block
):
    torch.manual_seed(217)
    batch, blocks = 8, 24
    q = torch.randn(batch, query_length, heads, 128, device="cuda").to(dtype)
    physical_pages = blocks * (128 // page_size)
    entries_per_page = page_size // factor
    # A footer pitch stresses short-page TMA without allowing cross-page reads.
    storage = torch.full(
        (batch * physical_pages, entries_per_page * 128 + 64), -384.0, device="cuda"
    ).to(dtype)
    k = storage[:, : entries_per_page * 128].view(-1, entries_per_page, 128)
    k.copy_(torch.randn(k.shape, device="cuda").to(dtype))
    table = (
        torch.randperm(batch * physical_pages, device="cuda")
        .view(batch, physical_pages)
        .int()
    )
    lengths = torch.tensor(
        [0, 3, 4, 127, 128, 131, 132, 3072], dtype=torch.int32, device="cuda"
    )
    plan = BatchDecodeIndexerPlan(
        lengths,
        max_pages=physical_pages,
        num_index_heads=heads,
        query_length=query_length,
        pooling_factor=factor,
        include_partial_block=include_partial_block,
        page_size=page_size,
    )
    plan.update()
    scorer = BatchDecodeBlockScoreWrapper(
        input_dtype=dtype,
        pooling_factor=factor,
        include_partial_block=include_partial_block,
        page_size=page_size,
        sm_scale=1.0,
    )
    scorer.plan(
        table,
        lengths,
        num_index_heads=heads,
        query_length=query_length,
        shared_plan=plan,
    )
    actual = torch.empty(heads, batch * query_length, blocks, device="cuda")
    _run(scorer, q, k, out=actual)
    expected = _reference(q, k, table, lengths, factor, include_partial_block)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-5)
    first = actual.clone()
    _run(scorer, q, k, out=actual)
    assert torch.equal(first, actual)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.update()
        scorer.run(q, k, out=actual)
    for state in (
        [255, 128, 132, 3, 4, 3071, 0, 131],
        [132, 3, 127, 128, 3072, 0, 4, 129],
    ):
        lengths.copy_(torch.tensor(state, dtype=torch.int32, device="cuda"))
        table.copy_(table.roll(1, dims=1))
        run_cuda("block scores graph", graph.replay)
        expected = _reference(q, k, table, lengths, factor, include_partial_block)
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-5)


@pytest.mark.parametrize(
    "kwargs,message",
    (
        ({"sm_scale": 0.0}, "finite and positive"),
        ({"sm_scale": -1.0}, "finite and positive"),
        ({"sm_scale": float("inf")}, "finite and positive"),
        ({"sm_scale": float("nan")}, "finite and positive"),
        ({"pooling_factor": 3}, "pooling_factor"),
        ({"include_partial_block": 1}, "include_partial_block"),
        ({"page_size": 32}, "page_size"),
    ),
)
def test_block_scores_reject_invalid_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BatchDecodeBlockScoreWrapper(**kwargs)


@pytest.mark.gpu
def test_shared_pooled_plan_rejects_default_consumer():
    lengths = torch.tensor([132], dtype=torch.int32, device="cuda")
    table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    plan = BatchDecodeIndexerPlan(
        lengths,
        max_pages=2,
        num_index_heads=4,
        query_length=1,
        pooling_factor=4,
        include_partial_block=True,
    )
    scorer = BatchDecodeBlockScoreWrapper(input_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="binding"):
        scorer.plan(table, lengths, num_index_heads=4, query_length=1, shared_plan=plan)


@pytest.mark.gpu
@pytest.mark.parametrize("include_partial_block", (False, True))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float8_e4m3fn))
@pytest.mark.parametrize("minimum_footer", (False, True))
def test_block_scores_page64_odd_capacity_negative_scores_and_allocated_output(
    include_partial_block, dtype, minimum_footer
):
    # Three physical pages represent one full block and one half block.
    table = torch.tensor([[3, 0, 2]], device="cuda", dtype=torch.int32)
    footer = (8 if dtype == torch.bfloat16 else 16) if minimum_footer else 64
    storage = torch.full((4, 16 * 128 + footer), 384.0, device="cuda").to(dtype)
    k = storage[:, : 16 * 128].view(4, 16, 128)
    k.fill_(-1)
    q = torch.ones(1, 1, 1, 128, device="cuda").to(dtype)
    lengths = torch.tensor([192], device="cuda", dtype=torch.int32)
    scorer = BatchDecodeBlockScoreWrapper(
        input_dtype=dtype,
        pooling_factor=4,
        page_size=64,
        include_partial_block=include_partial_block,
        sm_scale=1.0,
    )
    for length in (3, 4, 63, 64, 65, 127, 128, 129, 131, 132, 191, 192):
        lengths.fill_(length)
        table.copy_(
            torch.tensor(
                [[3, 0 if length >= 68 else -1, 2 if length >= 132 else -1]],
                device="cuda",
                dtype=torch.int32,
            )
        )
        scorer.plan(table, lengths, query_length=1)
        actual = _run(scorer, q, k)
        expected = _reference(q, k, table, lengths, 4, include_partial_block)
        assert actual.shape == (1, 1, 2)
        assert torch.equal(actual, expected)
        # Reusing a previously populated output cannot expose stale suffixes.
        actual.fill_(float("nan"))
        _run(scorer, q, k, out=actual)
        assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("page_size", (64, 128))
def test_block_scores_maximum_logical_capacity(page_size):
    blocks = 16384
    table = torch.zeros(
        (1, blocks * (128 // page_size)), device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor([blocks * 128], device="cuda", dtype=torch.int32)
    q = torch.ones((1, 1, 1, 128), device="cuda", dtype=torch.bfloat16)
    k = -torch.ones((1, page_size // 4, 128), device="cuda", dtype=torch.bfloat16)
    scorer = BatchDecodeBlockScoreWrapper(
        input_dtype=torch.bfloat16,
        pooling_factor=4,
        page_size=page_size,
        include_partial_block=True,
        sm_scale=1.0,
    )
    scorer.plan(table, lengths, query_length=1)
    out = _run(scorer, q, k)
    assert out.shape == (1, 1, blocks)
    assert torch.equal(out, torch.full_like(out, -128.0))
    oversized = torch.zeros((1, table.shape[1] + 1), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="max_pages"):
        scorer.plan(oversized, lengths, query_length=1)


@pytest.mark.gpu
@pytest.mark.parametrize("include_partial_block", (False, True))
@pytest.mark.parametrize("page_size,blocks", ((64, 24), (64, 4099), (128, 4099)))
def test_block_scores_maximum_query_and_head_count(
    include_partial_block, page_size, blocks
):
    from inference.msa_v1.indexer.decode._interface import _COMPILE_CACHE

    torch.manual_seed(821)
    batch, query_length, heads = 4, 16, 4
    # Odd long capacity crosses persistent segments and leaves a partial tile;
    # page64 additionally ends in half a logical block.
    pages = blocks * (128 // page_size) - (page_size == 64 and blocks > 24)
    capacity = pages * page_size
    q = torch.randn(
        batch, query_length, heads, 128, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(batch * pages, page_size // 4, 128, device="cuda", dtype=q.dtype)
    table = torch.randperm(k.shape[0], device="cuda").view(batch, pages).int()
    values = (
        [65, 127, 129, 3072]
        if blocks == 24
        else [capacity - 511, capacity - 129, capacity - 3, capacity]
    )
    lengths = torch.tensor(values, dtype=torch.int32, device="cuda")
    settings = dict(
        pooling_factor=4,
        page_size=page_size,
        include_partial_block=include_partial_block,
    )
    plan = BatchDecodeIndexerPlan(
        lengths,
        max_pages=pages,
        num_index_heads=heads,
        query_length=query_length,
        **settings,
    )
    scorer = BatchDecodeBlockScoreWrapper(input_dtype=q.dtype, sm_scale=1.0, **settings)
    scorer.plan(
        table,
        lengths,
        num_index_heads=heads,
        query_length=query_length,
        shared_plan=plan,
    )
    plan.update()
    actual = _run(scorer, q, k)
    expected = _reference(q, k, table, lengths, 4, include_partial_block)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-5)
    assert torch.equal(actual, _run(scorer, q, k))
    keys = set(_COMPILE_CACHE)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.update()
        scorer.run(q, k, out=actual)
    lengths.copy_(
        torch.tensor(
            [0, query_length, capacity // 2 + 1, capacity - 131],
            dtype=torch.int32,
            device="cuda",
        )
    )
    table.copy_(table.roll(1, dims=1))
    table.masked_fill_(
        torch.arange(pages, device="cuda")[None] * page_size >= lengths[:, None], -1
    )
    actual.fill_(float("nan"))
    run_cuda("block scores graph", graph.replay)
    expected = _reference(q, k, table, lengths, 4, include_partial_block)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-5)
    previous = actual.clone()
    run_cuda("block scores graph", graph.replay)
    assert torch.equal(actual, previous)
    assert set(_COMPILE_CACHE) == keys
