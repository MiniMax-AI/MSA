"""Runtime cache strides must survive decode indexer compilation and reuse."""

import pytest
import torch

from inference.msa_v1.indexer._common.topk_select.build import load_extension
from inference.msa_v1.indexer.decode._interface import (
    _COMPILE_CACHE,
    _BatchDecodeProxyScoreWrapper,
)
from inference.msa_v1.indexer.decode.bf16 import (
    BatchDecodeIndexerWithPagedKVCacheWrapper as Bf16Indexer,
)
from inference.msa_v1.indexer.decode.q8kv8 import (
    BatchDecodeIndexerWithPagedKVCacheWrapper as Fp8Indexer,
)
from tests.inference.msa_v1.cache_layout import CACHE_LAYOUTS, cache_view
from tests.inference.msa_v1.indexer._common.topk_select.reference import (
    assert_quantized_topk_contract,
)
from tests.inference.msa_v1.indexer.decode.q8kv8.reference import (
    indexer_gemm_reference,
    make_inputs,
)
from tests.inference.msa_v1.indexer.decode.q8kv8.test_correctness import (
    _assert_scores,
    _timed_run,
)

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float8_e4m3fn))
@pytest.mark.parametrize(
    ("first_layout", "singleton"),
    (
        ("contiguous", False),
        ("contiguous", True),
        ("singleton_zero", True),
        ("singleton_unaligned", True),
    ),
)
def test_cache_layout_reuses_compilation(dtype, first_layout, singleton):
    lengths = torch.tensor([259] if singleton else [2059, 4099], dtype=torch.int32)
    batch, max_pages = (1, 3) if singleton else (2, 33)
    q, k, table, lengths = make_inputs(
        batch,
        max_pages,
        lengths,
        seed=1701,
        device=torch.device("cuda"),
        num_index_heads=4,
        query_length=4,
    )
    q, k = q.to(dtype), k.to(dtype)
    if singleton:
        k = k[:1]
        table.zero_()
    initial_cache = cache_view(k, first_layout)
    wrapper = _BatchDecodeProxyScoreWrapper(input_dtype=dtype)
    wrapper.plan(table, lengths, num_index_heads=4, query_length=4)
    out = torch.empty((4, batch * 4, max_pages), device="cuda", dtype=torch.float32)
    expected = indexer_gemm_reference(q, k, table, lengths)
    if dtype == torch.bfloat16:
        expected *= 128**-0.5
    _COMPILE_CACHE.clear()
    wrapper._compile_and_bind(q, initial_cache, out)
    public_wrapper = Bf16Indexer() if dtype == torch.bfloat16 else Fp8Indexer()
    public_wrapper.plan(table, lengths, num_index_heads=4, query_length=4)
    public_wrapper._proxy_score._compile_and_bind(
        q, initial_cache, public_wrapper._proxy_scores
    )
    load_extension()
    topk = torch.empty((4, batch * 4, 16), device="cuda", dtype=torch.int32)
    positions = lengths[:, None] - 4 + torch.arange(4, device="cuda")
    valid_pages = (positions // 128 + 1).reshape(-1).repeat(4).cpu().numpy()
    expected_scores = expected.reshape(-1, max_pages).cpu().numpy()
    compiled_count = len(_COMPILE_CACHE)
    layouts = (
        (*CACHE_LAYOUTS, "singleton_zero", "singleton_unaligned")
        if singleton
        else CACHE_LAYOUTS
    )
    for layout in layouts:
        cache = cache_view(k, layout)
        out.fill_(123.0)
        actual = _timed_run(layout, lambda cache=cache: wrapper.run(q, cache, out=out))
        _assert_scores(actual, expected, lengths)
        assert len(_COMPILE_CACHE) == compiled_count
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper.run(q, cache, out=out)
        out.fill_(123.0)
        _timed_run(f"{layout} graph", graph.replay)
        _assert_scores(out, expected, lengths)
        _timed_run(
            f"{layout} public TopK",
            lambda cache=cache: public_wrapper.run(q, cache, out=topk),
        )
        assert_quantized_topk_contract(
            expected_scores, valid_pages, topk.reshape(-1, 16).cpu().numpy()
        )
        topk_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(topk_graph):
            public_wrapper.run(q, cache, out=topk)
        topk.fill_(-123)
        public_wrapper._proxy_scores.fill_(float("nan"))
        _timed_run(f"{layout} public TopK graph", topk_graph.replay)
        assert_quantized_topk_contract(
            expected_scores, valid_pages, topk.reshape(-1, 16).cpu().numpy()
        )
        assert len(_COMPILE_CACHE) == compiled_count


@pytest.mark.parametrize(
    "invalid_layout", ("channel", "alignment", "broadcast", "token")
)
def test_invalid_cache_layout_is_rejected_before_launch(invalid_layout):
    from inference.msa_v1._cache_layout import check_cache_layout

    storage = torch.empty(2 * 128 * 256 + 16, device="cuda", dtype=torch.bfloat16)
    shape = (2, 128, 128)
    if invalid_layout == "channel":
        cache = storage.as_strided(shape, (32768, 256, 2))
        message = "last dimension"
    elif invalid_layout == "alignment":
        cache = storage.as_strided(shape, (16384, 128, 1), 1)
        message = "data pointer"
    elif invalid_layout == "broadcast":
        cache = storage.as_strided(shape, (0, 128, 1))
        message = "positive stride"
    else:
        cache = storage.as_strided(shape, (32768, 130, 1))
        message = "axis 1 byte stride"
    with pytest.raises(ValueError, match=message):
        check_cache_layout(cache, name="k_cache")
