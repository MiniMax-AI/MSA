"""Cache views must preserve attention and indexer numerical contracts.

Cover independently strided data and packed scale pages, outer data padding/permutations, and
unused singleton strides. Page-row cases also model serving caches that pack
every component of one (layer, page) into a shared byte row.
"""

from __future__ import annotations

import pytest
import torch

from benchmarks.inference.msa_v1.attention.prefill.cases import warmup_case
from benchmarks.inference.msa_v1.decode.cases import DecodeBenchmarkCase, make_seq_lens
from inference.msa_v1.attention.decode.q8kv4 import BatchDecodeWithPagedKVCacheWrapper
from inference.msa_v1.attention.prefill.q8kv4 import BatchPrefillWithPagedKVCacheWrapper
from inference.msa_v1.indexer.decode.q8kv4 import (
    BatchDecodeIndexerWithPagedKVCacheWrapper,
)
from tests.inference.msa_v1.attention.decode.q8kv4.real_cases import (
    make_decode_attention_inputs,
)
from tests.inference.msa_v1.attention.decode.q8kv4.reference import (
    decode_attention_reference,
)
from tests.inference.msa_v1.attention.decode.runtime import run_cuda
from tests.inference.msa_v1.attention.prefill.q8kv4.real_cases import (
    make_real_prefill_attention_inputs,
)
from tests.inference.msa_v1.attention.prefill.q8kv4.reference import (
    paged_sparse_attention_reference,
)
from tests.inference.msa_v1.cache_layout import (
    CACHE_LAYOUTS,
    cache_view,
    scale_cache_view,
)
from tests.inference.msa_v1.indexer._common.topk_select.reference import (
    assert_quantized_topk_contract,
)
from tests.inference.msa_v1.indexer.decode.q8kv4.reference import (
    indexer_gemm_reference,
    make_inputs,
)
from tests.inference.msa_v1.indexer.decode.q8kv4.test_correctness import _assert_scores

pytestmark = pytest.mark.gpu

_PAGE = 128
_KV_HEADS = 4
# MSA-v1 serving row for 4 KV heads: K | K scale | V | V scale | index K | index
# K scale, each component starting on a 128-byte boundary.
_COMPONENTS = {
    "k": (0, (_KV_HEADS, _PAGE, 64)),
    "k_scale": (32768, (_KV_HEADS, _PAGE, 8)),
    "v": (36864, (_KV_HEADS, _PAGE, 64)),
    "v_scale": (69632, (_KV_HEADS, _PAGE, 8)),
    "index_k": (73728, (_PAGE, 64)),
    "index_k_scale": (81920, (_PAGE, 8)),
}
_ROW_BYTES = 82944
_Q4_LAYOUTS = (*CACHE_LAYOUTS, "singleton_zero", "singleton_unaligned")


def _require_sm100() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
        pytest.skip("Q8KV4 kernels require SM100 or SM103")
    return device


def _page_rows(packed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy packed caches into random page rows and return the strided views."""

    pages = next(iter(packed.values())).shape[0]
    device = next(iter(packed.values())).device
    generator = torch.Generator(device=device).manual_seed(4049)
    # Random filler between components catches reads that ignore the stride.
    rows = torch.randint(
        0,
        256,
        (pages, _ROW_BYTES),
        dtype=torch.uint8,
        generator=generator,
        device=device,
    )
    views = {}
    for name, tensor in packed.items():
        offset, shape = _COMPONENTS[name]
        size = tensor[0].numel() * tensor.element_size()
        view = rows[:, offset : offset + size].view(pages, *shape)
        view.copy_(tensor.view(torch.uint8))
        views[name] = view.view(tensor.dtype)
        assert views[name].stride(0) * views[name].element_size() == _ROW_BYTES
    return views


def _invalid_scale_views(tensor):
    for layout in ("token", "permuted", "head"):
        if layout == "head" and (tensor.ndim != 4 or tensor.shape[1] == 1):
            continue
        yield cache_view(tensor, layout), "packed below the page dimension"
    yield cache_view(tensor, "offset", alignment_bytes=8), "aligned|alignment"
    yield cache_view(tensor, "page", alignment_bytes=8), "aligned|alignment"
    strides = list(tensor.stride())
    strides[0] = tensor[0].numel() - 16
    yield tensor.as_strided(tensor.shape, strides), "page stride"


@pytest.mark.parametrize("q_len", (1, 4))
@pytest.mark.parametrize("layout", _Q4_LAYOUTS)
def test_decode_attention_page_stride(q_len: int, layout: str) -> None:
    device = _require_sm100()
    singleton = layout.startswith("singleton_")
    kv_heads = 1 if singleton else _KV_HEADS
    case = DecodeBenchmarkCase(
        batch_size=1 if singleton else 5,
        nominal_seq_len=17 if singleton else 3_000,
        weight=0,
        q_len_per_req=q_len,
    )
    inputs = make_decode_attention_inputs(
        make_seq_lens(case),
        seed=17 + q_len,
        device=device,
        q_len_per_req=q_len,
        page_layout="shared_prefix" if singleton else "permuted",
        num_q_heads=kv_heads * 16,
        num_kv_heads=kv_heads,
    )
    packed = {
        "k": inputs.packed_k,
        "v": inputs.packed_v,
        "k_scale": inputs.k_scale,
        "v_scale": inputs.v_scale,
    }
    strided = (
        _page_rows(packed)
        if layout == "page"
        else {
            name: scale_cache_view(tensor, layout)
            if "scale" in name
            else cache_view(tensor, layout, alignment_bytes=32)
            for name, tensor in packed.items()
        }
    )
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    wrapper.plan(
        inputs.topk_indices,
        inputs.page_table,
        inputs.seq_lens,
        q_len_per_req=q_len,
        num_q_heads=kv_heads * 16,
        num_kv_heads=kv_heads,
    )
    if layout == "contiguous":
        for invalid_scale, message in _invalid_scale_views(inputs.k_scale):
            with pytest.raises((ValueError, RuntimeError), match=message):
                wrapper.run(
                    inputs.q,
                    (inputs.packed_k, inputs.packed_v),
                    kv_cache_sf=(invalid_scale, inputs.v_scale),
                )
    outputs = []
    for caches in (packed, strided):
        out = wrapper.run(
            inputs.q,
            (caches["k"], caches["v"]),
            kv_cache_sf=(caches["k_scale"], caches["v_scale"]),
        )
        outputs.append(out.clone())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper.run(
                inputs.q,
                (caches["k"], caches["v"]),
                kv_cache_sf=(caches["k_scale"], caches["v_scale"]),
                out=out,
            )
        out.fill_(float("nan"))
        run_cuda("strided Q8KV4 decode attention replay", graph.replay)
        outputs.append(out.clone())
    expected = decode_attention_reference(inputs)
    for output in outputs:
        assert torch.isfinite(output).all()
        torch.testing.assert_close(
            output.float(), expected.float(), atol=0.05, rtol=0.05
        )


@pytest.mark.parametrize("q_len", (1, 4))
@pytest.mark.parametrize("layout", _Q4_LAYOUTS)
def test_decode_indexer_page_stride(q_len: int, layout: str) -> None:
    device = _require_sm100()
    singleton = layout.startswith("singleton_")
    case = DecodeBenchmarkCase(
        batch_size=1 if singleton else 5,
        nominal_seq_len=259 if singleton else 3_000,
        weight=0,
        q_len_per_req=q_len,
    )
    lengths = make_seq_lens(case)
    max_pages = (int(lengths.max()) + _PAGE - 1) // _PAGE
    q, packed_k, k_scale, page_table, seq_lens = make_inputs(
        case.batch_size,
        max_pages,
        lengths,
        seed=29 + q_len,
        device=device,
        num_index_heads=4,
        query_length=q_len,
    )
    if singleton:
        packed_k, k_scale = packed_k[:1], k_scale[:1]
        page_table.zero_()
    packed = {"index_k": packed_k, "index_k_scale": k_scale}
    strided = (
        _page_rows(packed)
        if layout == "page"
        else {
            name: scale_cache_view(tensor, layout)
            if "scale" in name
            else cache_view(tensor, layout, alignment_bytes=16)
            for name, tensor in packed.items()
        }
    )
    results = []
    wrapper = BatchDecodeIndexerWithPagedKVCacheWrapper()
    wrapper.plan(page_table, seq_lens, num_index_heads=4, query_length=q_len)
    if layout == "contiguous":
        for invalid_scale, message in _invalid_scale_views(k_scale):
            with pytest.raises((ValueError, RuntimeError), match=message):
                wrapper.run(q, packed_k, k_scale=invalid_scale)
    for caches in (packed, strided):
        # Unwritten score slots must stay untouched in both layouts.
        wrapper._proxy_scores.fill_(123.0)
        topk = wrapper.run(
            q, caches["index_k"], k_scale=caches["index_k_scale"]
        ).clone()
        results.append((topk.clone(), wrapper._proxy_scores.clone()))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper.run(q, caches["index_k"], k_scale=caches["index_k_scale"], out=topk)
        topk.fill_(-1)
        wrapper._proxy_scores.fill_(123.0)
        run_cuda("strided Q8KV4 decode indexer replay", graph.replay)
        results.append((topk.clone(), wrapper._proxy_scores.clone()))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    expected = indexer_gemm_reference(q, packed_k, k_scale, page_table, seq_lens)
    for topk, scores in results:
        valid_pages = _assert_scores(scores, expected, seq_lens)
        assert_quantized_topk_contract(
            expected.reshape(-1, max_pages).cpu().numpy(),
            valid_pages.repeat(4).cpu().numpy(),
            topk.reshape(-1, 16).cpu().numpy(),
        )


@pytest.mark.parametrize("layout", _Q4_LAYOUTS)
def test_prefill_attention_page_stride(layout: str) -> None:
    device = _require_sm100()
    case = warmup_case().shape
    inputs = make_real_prefill_attention_inputs(
        case,
        device=device,
        num_kv_heads=1 if layout.startswith("singleton_") else _KV_HEADS,
    )
    packed = {
        "k": inputs.packed_k,
        "v": inputs.packed_v,
        "k_scale": inputs.k_scale,
        "v_scale": inputs.v_scale,
    }
    strided = (
        _page_rows(packed)
        if layout == "page"
        else {
            name: scale_cache_view(tensor, layout)
            if "scale" in name
            else cache_view(tensor, layout, alignment_bytes=16)
            for name, tensor in packed.items()
        }
    )
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
    if layout == "contiguous":
        for invalid_scale, message in _invalid_scale_views(inputs.k_scale):
            with pytest.raises((ValueError, RuntimeError), match=message):
                wrapper.run(
                    inputs.q,
                    (inputs.packed_k, inputs.packed_v),
                    kv_cache_sf=(invalid_scale, inputs.v_scale),
                )
    results = []
    for caches in (packed, strided):
        out, lse = wrapper.run(
            inputs.q,
            (caches["k"], caches["v"]),
            kv_cache_sf=(caches["k_scale"], caches["v_scale"]),
            return_lse=True,
        )
        results.append((out.clone(), lse.clone()))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper.run(
                inputs.q,
                (caches["k"], caches["v"]),
                kv_cache_sf=(caches["k_scale"], caches["v_scale"]),
                out=out,
                lse=lse,
                return_lse=True,
            )
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        run_cuda("strided Q8KV4 prefill attention replay", graph.replay)
        results.append((out.clone(), lse.clone()))
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
    for out, lse in results:
        assert torch.isfinite(out).all() and torch.isfinite(lse).all()
        torch.testing.assert_close(out.float(), expected_out, atol=0.03, rtol=0.03)
        torch.testing.assert_close(lse, expected_lse, atol=5e-4, rtol=5e-4)
