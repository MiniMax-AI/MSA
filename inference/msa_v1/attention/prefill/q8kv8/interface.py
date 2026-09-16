"""FlashInfer-style wrapper for native-E4M3 paged sparse prefill."""

from inference.msa_v1.attention.prefill._common.paged_interface import (
    _BatchPrefillWithPagedKVCacheWrapperBase,
)


class BatchPrefillWithPagedKVCacheWrapper(_BatchPrefillWithPagedKVCacheWrapperBase):
    """Stateful Q8KV8 PageKV sparse-prefill wrapper."""


__all__ = ["BatchPrefillWithPagedKVCacheWrapper"]
