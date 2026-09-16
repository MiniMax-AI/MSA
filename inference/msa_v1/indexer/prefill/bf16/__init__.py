"""BF16 paged prefill indexer."""

from inference.msa_v1.indexer.prefill.bf16.interface import (
    BatchPrefillIndexerWithPagedKVCacheWrapper,
)

__all__ = ["BatchPrefillIndexerWithPagedKVCacheWrapper"]
