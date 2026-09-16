"""Public API naming and lifecycle contract tests."""

from __future__ import annotations

import inspect
from importlib.util import find_spec

from inference.msa_v1 import indexer as indexer_package
from inference.msa_v1.indexer.prefill import tp4_q8kv8 as indexer


def test_wrapper_is_the_only_public_entrypoint() -> None:
    assert indexer.__all__ == ["BatchPrefillIndexerWithPagedKVCacheWrapper"]
    assert not hasattr(indexer, "BatchPrefillIndexerGemmWrapper")
    assert not hasattr(indexer, "BatchPrefillIndexerWrapper")
    wrapper_type = indexer.BatchPrefillIndexerWithPagedKVCacheWrapper
    assert tuple(inspect.signature(wrapper_type.plan).parameters) == (
        "self",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "page_table",
        "total_q",
        "max_seqlen_q",
        "max_seqlen_k",
    )
    assert tuple(inspect.signature(wrapper_type.run).parameters) == (
        "self",
        "q",
        "paged_k_cache",
        "out",
    )


def test_topk_select_is_internal_only() -> None:
    assert indexer_package.__all__ == ["decode", "prefill"]
    assert find_spec("inference.msa_v1.indexer.topk") is None
