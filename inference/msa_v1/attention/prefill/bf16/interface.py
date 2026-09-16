"""FlashInfer-style wrapper for BF16 paged sparse prefill."""

import torch

from inference.msa_v1.attention.prefill._common.paged_interface import (
    _BatchPrefillWithPagedKVCacheWrapperBase,
)


class BatchPrefillWithPagedKVCacheWrapper(_BatchPrefillWithPagedKVCacheWrapperBase):
    """Stateful BF16 PageKV sparse-prefill wrapper."""

    storage_dtype = torch.bfloat16
    op_name = "BF16"
    allow_strided_kv = True
    supported_gqa_group_sizes = (8, 16)


__all__ = ["BatchPrefillWithPagedKVCacheWrapper"]
